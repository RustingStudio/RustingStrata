//! Vulkan backend on a dynamically loaded libvulkan (`ash`). Every buffer has its own allocation and a device
//! address, so `Buf` addresses are real pointers that kernels take as push constants. Ops are recorded into one
//! command buffer, which a host copy or `sync` submits and waits for.

use crate::backend::{Backend, Buf, GpuTensor};
use crate::quant::GgmlType;
use anyhow::{Context, anyhow, ensure};
use ash::vk;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// Host-visible buffer that host copies go through, in chunks of this size.
const STAGING: usize = 64 << 20;

pub struct Vk(Arc<Inner>);

struct Inner {
    _entry: ash::Entry,
    instance: ash::Instance,
    pd: vk::PhysicalDevice,
    dev: ash::Device,
    queue: vk::Queue,
    mem: vk::PhysicalDeviceMemoryProperties,
    budget: bool,
    st: Mutex<State>,
}

struct State {
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    /// `cb` is recording.
    open: bool,
    fence: vk::Fence,
    staging: (vk::Buffer, vk::DeviceMemory),
    /// Mapped `staging` (a pointer kept as an integer so the state is `Send`).
    map: usize,
    /// Live buffers by start address: (buffer, bytes).
    bufs: BTreeMap<u64, (vk::Buffer, usize)>,
}

/// Owner of one `Buf`: the buffer and its memory, freed on drop.
struct Mem { vk: Arc<Inner>, buf: vk::Buffer, mem: vk::DeviceMemory, addr: u64 }

impl Drop for Mem {
    fn drop(&mut self) {
        let mut st = self.vk.st();
        let _ = self.vk.flush(&mut st); // queued work may still use the buffer
        st.bufs.remove(&self.addr);
        unsafe { self.vk.dev.destroy_buffer(self.buf, None); self.vk.dev.free_memory(self.mem, None) }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let st = self.st.get_mut().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let _ = self.dev.device_wait_idle();
            self.dev.destroy_buffer(st.staging.0, None);
            self.dev.free_memory(st.staging.1, None);
            self.dev.destroy_fence(st.fence, None);
            self.dev.destroy_command_pool(st.pool, None);
            self.dev.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Why `pd` cannot run the kernels, if it cannot.
fn unusable(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<String> {
    let mut sg = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
    unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
    let p = p2.properties;
    if p.device_type == vk::PhysicalDeviceType::CPU { return Some("a CPU device".into()); }
    if p.api_version < vk::API_VERSION_1_3 { return Some("Vulkan 1.3 is needed".into()); }
    if !(sg.min_subgroup_size..=sg.max_subgroup_size).contains(&32) || !sg.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE) {
        return Some(format!("subgroup size 32 is needed (it has {}-{})", sg.min_subgroup_size, sg.max_subgroup_size));
    }
    let (mut f11, mut f12, mut f13) = Default::default();
    let mut f = features(&mut f11, &mut f12, &mut f13);
    unsafe { instance.get_physical_device_features2(pd, &mut f) };
    let int64 = f.features.shader_int64;
    let missing: Vec<_> = [
        ("shaderInt64", int64), ("storageBuffer16BitAccess", f11.storage_buffer16_bit_access),
        ("bufferDeviceAddress", f12.buffer_device_address), ("storageBuffer8BitAccess", f12.storage_buffer8_bit_access),
        ("shaderInt8", f12.shader_int8), ("shaderFloat16", f12.shader_float16),
        ("subgroupSizeControl", f13.subgroup_size_control), ("computeFullSubgroups", f13.compute_full_subgroups),
    ].into_iter().filter(|f| f.1 == vk::FALSE).map(|f| f.0).collect();
    (!missing.is_empty()).then(|| format!("missing {}", missing.join(", ")))
}

/// Feature chain for querying (all false) or enabling (filled by the caller).
fn features<'a>(f11: &'a mut vk::PhysicalDeviceVulkan11Features<'static>, f12: &'a mut vk::PhysicalDeviceVulkan12Features<'static>,
                f13: &'a mut vk::PhysicalDeviceVulkan13Features<'static>) -> vk::PhysicalDeviceFeatures2<'a> {
    vk::PhysicalDeviceFeatures2::default().push_next(f11).push_next(f12).push_next(f13)
}

impl Vk {
    /// The `device`-th usable GPU. Fails with "Vulkan init" (and the reasons) when there is none.
    pub fn new(device: usize) -> anyhow::Result<Vk> {
        let entry = unsafe { ash::Entry::load() }.context("Vulkan init")?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }
            .context("Vulkan init")?;
        let mut usable = vec![];
        let mut why = vec![];
        for pd in unsafe { instance.enumerate_physical_devices() }.unwrap_or_default() {
            let p = unsafe { instance.get_physical_device_properties(pd) };
            let name = p.device_name_as_c_str().map_or("?".into(), |s| s.to_string_lossy().into_owned());
            match unusable(&instance, pd) { None => usable.push((pd, name)), Some(w) => why.push(format!("{name}: {w}")) }
        }
        let Some((pd, name)) = usable.get(device).cloned() else {
            unsafe { instance.destroy_instance(None) };
            return Err(anyhow!("no usable Vulkan device {device} ({} usable; {})", usable.len(), why.join("; "))).context("Vulkan init");
        };
        // ponytail: the instance leaks if device setup below fails; it is a one-time startup error.
        let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) }.iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)).context("no compute queue")? as u32;
        let budget = unsafe { instance.enumerate_device_extension_properties(pd) }?.iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
        let exts = if budget { vec![ash::ext::memory_budget::NAME.as_ptr()] } else { vec![] };
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true)
            .storage_buffer8_bit_access(true).shader_int8(true).shader_float16(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().subgroup_size_control(true).compute_full_subgroups(true);
        let mut f = features(&mut f11, &mut f12, &mut f13);
        f.features.shader_int64 = vk::TRUE;
        let prio = [1.0];
        let qi = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        let dev = unsafe { instance.create_device(pd, &vk::DeviceCreateInfo::default().queue_create_infos(&qi)
            .enabled_extension_names(&exts).push_next(&mut f), None) }.with_context(|| format!("create device {name}"))?;
        let queue = unsafe { dev.get_device_queue(qf, 0) };
        let mem = unsafe { instance.get_physical_device_memory_properties(pd) };
        let pool = unsafe { dev.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(qf)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None) }?;
        let cb = unsafe { dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool)
            .command_buffer_count(1)) }?[0];
        let fence = unsafe { dev.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let st = State { pool, cb, open: false, fence, staging: Default::default(), map: 0, bufs: BTreeMap::new() };
        let inner = Inner { _entry: entry, instance, pd, dev, queue, mem, budget, st: Mutex::new(st) };
        let staging = inner.buffer(STAGING, true)?;
        let map = unsafe { inner.dev.map_memory(staging.1, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) };
        let mut st = inner.st();
        st.staging = staging;
        st.map = map? as usize;
        drop(st);
        Ok(Vk(Arc::new(inner)))
    }
}

impl Inner {
    fn st(&self) -> MutexGuard<'_, State> { self.st.lock().unwrap_or_else(|e| e.into_inner()) }

    /// A buffer usable by kernels (device address) and copies, of at least `bytes`, in device-local memory or, with
    /// `host`, in host-visible coherent memory.
    fn buffer(&self, bytes: usize, host: bool) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let size = bytes.max(1).next_multiple_of(4) as u64;
        let buf = unsafe { self.dev.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None) }?;
        let req = unsafe { self.dev.get_buffer_memory_requirements(buf) };
        let want = if host { vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT } else { vk::MemoryPropertyFlags::DEVICE_LOCAL };
        let ty = (0..self.mem.memory_type_count).find(|&i| req.memory_type_bits & (1 << i) != 0
            && self.mem.memory_types[i as usize].property_flags.contains(want));
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let r = ty.context("no memory type").and_then(|ty| Ok(unsafe { self.dev.allocate_memory(&vk::MemoryAllocateInfo::default()
            .allocation_size(req.size).memory_type_index(ty).push_next(&mut flags), None) }
            .with_context(|| format!("allocate {bytes} bytes"))?));
        let mem = match r { Ok(m) => m, Err(e) => { unsafe { self.dev.destroy_buffer(buf, None) }; return Err(e) } };
        unsafe { self.dev.bind_buffer_memory(buf, mem, 0) }?;
        Ok((buf, mem))
    }

    /// The open command buffer, begun if needed.
    fn cb(&self, st: &mut State) -> anyhow::Result<vk::CommandBuffer> {
        if !st.open {
            unsafe { self.dev.begin_command_buffer(st.cb, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)) }?;
            st.open = true;
        }
        Ok(st.cb)
    }

    /// Orders everything recorded so far before what follows (`HOST`: also before host reads).
    fn barrier(&self, cb: vk::CommandBuffer, host: bool) {
        let all = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let (dst, dst_access) = if host { (vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ) } else {
            (all, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
        };
        let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(dst_access);
        unsafe { self.dev.cmd_pipeline_barrier(cb, all, dst, vk::DependencyFlags::empty(), &[m], &[], &[]) }
    }

    /// Submit the open command buffer and wait for it.
    fn flush(&self, st: &mut State) -> anyhow::Result<()> {
        if !st.open { return Ok(()); }
        st.open = false;
        let cbs = [st.cb];
        unsafe {
            self.dev.end_command_buffer(st.cb)?;
            self.dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], st.fence)?;
            self.dev.wait_for_fences(&[st.fence], true, u64::MAX)?;
            self.dev.reset_fences(&[st.fence])?;
            self.dev.reset_command_buffer(st.cb, vk::CommandBufferResetFlags::empty())?;
        }
        Ok(())
    }
}

impl State {
    /// The buffer holding `bytes` from device address `addr`, and the offset of `addr` in it.
    fn find(&self, addr: u64, bytes: usize) -> anyhow::Result<(vk::Buffer, u64)> {
        let (&a, &(b, len)) = self.bufs.range(..=addr).next_back().with_context(|| format!("{addr:#x} is not a device buffer"))?;
        ensure!(addr - a + bytes as u64 <= len as u64, "{bytes} bytes at {addr:#x} pass the end of its buffer");
        Ok((b, addr - a))
    }
}

/// An op the Vulkan backend does not have yet.
fn todo(op: &str) -> anyhow::Result<()> { anyhow::bail!("{op} is not implemented on Vulkan yet") }

#[allow(unused_variables)]
impl Backend for Vk {
    fn name(&self) -> &'static str { "vulkan" }

    fn alloc(&self, bytes: usize) -> anyhow::Result<Buf> {
        let (buf, mem) = self.0.buffer(bytes, false)?;
        let addr = unsafe { self.0.dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf)) };
        let owner = Mem { vk: self.0.clone(), buf, mem, addr };
        let mut st = self.0.st();
        st.bufs.insert(addr, (buf, bytes));
        let cb = self.0.cb(&mut st)?;
        unsafe { self.0.dev.cmd_fill_buffer(cb, buf, 0, vk::WHOLE_SIZE, 0) };
        self.0.barrier(cb, false);
        Ok(Buf::new(addr, bytes, owner))
    }

    fn htod(&self, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()> {
        ensure!(off + src.len() <= dst.len(), "htod: {} bytes at {off} into {}", src.len(), dst.len());
        let mut st = self.0.st();
        for (i, c) in src.chunks(STAGING).enumerate() {
            let (b, o) = st.find(dst.addr() + (off + i * STAGING) as u64, c.len())?;
            unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), st.map as *mut u8, c.len()) };
            let cb = self.0.cb(&mut st)?;
            let r = vk::BufferCopy::default().dst_offset(o).size(c.len() as u64);
            unsafe { self.0.dev.cmd_copy_buffer(cb, st.staging.0, b, &[r]) };
            self.0.barrier(cb, false);
            self.0.flush(&mut st)?;
        }
        Ok(())
    }

    fn dtoh(&self, src: &Buf, off: usize, dst: &mut [u8]) -> anyhow::Result<()> {
        ensure!(off + dst.len() <= src.len(), "dtoh: {} bytes at {off} from {}", dst.len(), src.len());
        let mut st = self.0.st();
        for (i, c) in dst.chunks_mut(STAGING).enumerate() {
            let (b, o) = st.find(src.addr() + (off + i * STAGING) as u64, c.len())?;
            let cb = self.0.cb(&mut st)?;
            let r = vk::BufferCopy::default().src_offset(o).size(c.len() as u64);
            unsafe { self.0.dev.cmd_copy_buffer(cb, b, st.staging.0, &[r]) };
            self.0.barrier(cb, true);
            self.0.flush(&mut st)?;
            unsafe { std::ptr::copy_nonoverlapping(st.map as *const u8, c.as_mut_ptr(), c.len()) };
        }
        Ok(())
    }

    fn dtod(&self, src: &Buf, src_off: usize, dst: &Buf, dst_off: usize, bytes: usize) -> anyhow::Result<()> {
        ensure!(src_off + bytes <= src.len() && dst_off + bytes <= dst.len(), "dtod: {bytes} bytes out of range");
        if bytes == 0 { return Ok(()); }
        let mut st = self.0.st();
        let (s, so) = st.find(src.addr() + src_off as u64, bytes)?;
        let (d, d_o) = st.find(dst.addr() + dst_off as u64, bytes)?;
        let cb = self.0.cb(&mut st)?;
        unsafe { self.0.dev.cmd_copy_buffer(cb, s, d, &[vk::BufferCopy::default().src_offset(so).dst_offset(d_o).size(bytes as u64)]) };
        self.0.barrier(cb, false);
        Ok(())
    }

    fn memset(&self, dst: &Buf) -> anyhow::Result<()> {
        let mut st = self.0.st();
        let (b, _) = st.find(dst.addr(), dst.len())?;
        let cb = self.0.cb(&mut st)?;
        unsafe { self.0.dev.cmd_fill_buffer(cb, b, 0, vk::WHOLE_SIZE, 0) };
        self.0.barrier(cb, false);
        Ok(())
    }

    /// Free from `VK_EXT_memory_budget` when the driver has it, else the heap size less this backend's buffers.
    fn mem_info(&self) -> anyhow::Result<(usize, usize)> {
        let m = &self.0.mem;
        let h = (0..m.memory_heap_count as usize).find(|&h| m.memory_heaps[h].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .context("no device-local heap")?;
        let total = m.memory_heaps[h].size as usize;
        if !self.0.budget { return Ok((total.saturating_sub(self.0.st().bufs.values().map(|b| b.1).sum()), total)); }
        let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut b);
        unsafe { self.0.instance.get_physical_device_memory_properties2(self.0.pd, &mut p) };
        let _ = p;
        Ok((b.heap_budget[h].saturating_sub(b.heap_usage[h]) as usize, total))
    }

    fn sync(&self) -> anyhow::Result<()> { self.0.flush(&mut self.0.st()) }

    fn graph(&self, key: u64, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> { todo("graph") }
    fn host_mailbox(&self, words: usize) -> anyhow::Result<(Buf, *mut u32)> { todo("host_mailbox").map(|_| unreachable!()) }
    fn kv_words(&self, n: usize) -> usize { n }

    fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> { todo("gemm") }
    fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> { todo("gemm_dense") }
    fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()> { todo("gemv") }
    fn gemvn(&self, w: &GpuTensor, x: u64, x_stride: usize, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> { todo("gemvn") }
    fn gemvq(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> { todo("gemvq") }
    fn moe_gemv(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize, xg: usize, y: u64, pair: Option<(i64, i64)>, xq: u64) -> anyhow::Result<()> { todo("moe_gemv") }
    fn dequant(&self, w: &GpuTensor, rows: usize, out: u64) -> anyhow::Result<()> { todo("dequant") }
    fn quant_xq(&self, x: u64, x_stride: usize, n: usize, nt: usize, out: u64) -> anyhow::Result<()> { todo("quant_xq") }
    fn map_hits(&self, ids: u64, table: u64, slots: u64, wts: u64, wh: u64, k: usize) -> anyhow::Result<()> { todo("map_hits") }
    fn rms_norm(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32) -> anyhow::Result<()> { todo("rms_norm") }
    fn rms_norm_q(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32, xq: u64) -> anyhow::Result<()> { todo("rms_norm_q") }
    fn argmax_rows(&self, x: u64, n: usize, rows: usize, out: u64) -> anyhow::Result<()> { todo("argmax_rows") }
    fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()> { todo("l2_norm") }
    fn rope_neox(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> { todo("rope_neox") }
    fn add(&self, x: u64, y: u64, n: usize) -> anyhow::Result<()> { todo("add") }
    fn swiglu(&self, g: u64, u: u64, n: usize) -> anyhow::Result<()> { todo("swiglu") }
    fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()> { todo("ssm_gates") }
    fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize) -> anyhow::Result<()> { todo("conv_silu") }
    fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> { todo("deltanet") }
    fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> { todo("conv_seq") }
    fn attn_prefill(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32) -> anyhow::Result<()> { todo("attn_prefill") }
    fn moe_combine_seq(&self, x: u64, ye: u64, wts: u64, inv: u64, k: usize, sh: u64, sg: u64, n: usize, t: usize) -> anyhow::Result<()> { todo("moe_combine_seq") }
    fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()> { todo("norm_gate") }
    fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64, hd: usize) -> anyhow::Result<()> { todo("kv_store") }
    fn attn_decode_gq(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize) -> anyhow::Result<()> { todo("attn_decode") }
    fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize) -> anyhow::Result<()> { todo("router_topk") }
    fn moe_publish(&self, ids: u64, wts: u64, slots: u64, h: u64, mb: u64, k: usize, n: usize, h_off: usize, step: u64, t: usize) -> anyhow::Result<()> { todo("moe_publish") }
    fn moe_combine(&self, x: u64, ye: u64, w: u64, k: usize, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, slots: u64, n: usize, t: usize) -> anyhow::Result<()> { todo("moe_combine") }
}
