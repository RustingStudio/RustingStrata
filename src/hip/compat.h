// Device-side compatibility layer for the HIP backend. `src/hip/mod.rs` prepends this (and the quant tables) to
// `../cuda/kernels.cu`, which the CUDA backend hands to NVRTC unchanged: `kernels.cu` is one source with an explicit
// `STRATA_HIP` branch where the two architectures cannot share code, and this header supplies everything CUDA's
// headers give it for free (warp intrinsics, packed-byte helpers, the f16 SIMD ops) from HIP.
//
// The portable intrinsic implementations are adapted from Strata's include/strata/hip_compat/intrinsics.hpp
// (MIT, (c) 2025 Niko1221); see THANKS.md. They assume the wave32 model the kernels are written for (lane =
// threadIdx.x & 31), which is the default wavefront size on RDNA (gfx10+) — the same assumption the CUDA source
// makes about a warp.
// Prepended to a concatenated source string, so a plain include guard: `#pragma once` would warn in a main file.
#ifndef STRATA_HIP_COMPAT_H
#define STRATA_HIP_COMPAT_H
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

// kernels.cu selects its code paths by the CUDA architecture level: ldmatrix from sm_75, cp.async/m16n8k16/int8-mma
// from sm_80, m8n8k4 (mma884) from sm_70. AMD has none of those instructions, so the build takes the portable
// pre-Volta level: per-lane shared loads instead of ldmatrix, plain typed copies instead of cp.async, and the
// CUDA-core (f16 FMA) multiplicands instead of mma.sync. On gfx11 and gfx12 the prefill GEMM and attention use the
// WMMA matrix cores instead (GEMM_WMMA, below); other targets and STRATA_HIP_WMMA=0 keep the f16 FMA path. Decode is
// unaffected (it has no tensor-core path).
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 700
#endif
#define STRATA_HIP 1
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif
// The wave is 32 lanes wide here; a kernel that ever runs with a 64-wide wave would corrupt its reductions.
#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "the HIP backend needs wave32 (RDNA); wave64 targets are not supported"
#endif

namespace strata_hip {

__device__ __forceinline__ int signed_byte(unsigned word, int lane) {
    unsigned v = (word >> (lane * 8)) & 0xffu;
    return v < 0x80u ? (int)v : (int)v - 0x100;
}

// CUDA's signed __dp4a: four signed byte products accumulated modulo 2^32.
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
     defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)) && __has_builtin(__builtin_amdgcn_sudot4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)) && __has_builtin(__builtin_amdgcn_sdot4)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    unsigned ua = (unsigned)a, ub = (unsigned)b, sum = (unsigned)c;
#pragma unroll
    for (int i = 0; i < 4; ++i) sum += (unsigned)(signed_byte(ua, i) * signed_byte(ub, i));
    return (int)sum;
#endif
}

// CUDA's __byte_perm: result byte i is byte s.nibble[i] & 7 of {y:x}. One v_perm_b32 instead of HIP's byte array.
__device__ __forceinline__ unsigned byte_perm(unsigned x, unsigned y, unsigned s) {
    unsigned sel = (s & 0x7u) | ((s & 0x70u) << 4) | ((s & 0x700u) << 8) | ((s & 0x7000u) << 12);
    return __builtin_amdgcn_perm(y, x, sel);
}

// Packed byte arithmetic, four lanes at once (SWAR, Hacker's Delight 2-18): gfx11/gfx12 have no packed 8-bit
// add/sub, and a per-lane loop costs the Q3_K/Q6_K dots several times the instructions.
constexpr unsigned kHigh = 0x80808080u;

__device__ __forceinline__ int vsub4(int a, int b) {
    unsigned ua = (unsigned)a, ub = (unsigned)b;
    return (int)(((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}
__device__ __forceinline__ int vadd4(int a, int b) {
    unsigned ua = (unsigned)a, ub = (unsigned)b;
    return (int)(((ua & ~kHigh) + (ub & ~kHigh)) ^ ((ua ^ ub) & kHigh));
}

// The mask is informational here: an AMD wave either takes part in the corresponding `ds_bpermute` or does not, so a
// lane that is not in the mask still receives whatever the exchange computes (CUDA leaves it undefined). That matches
// CUDA whenever the lanes a participating lane exchanges with are participating too, which is how kernels.cu uses it
// (`quant8` shuffles inside groups of four active lanes; the rest pass 0xffffffff).
template <typename T> __device__ __forceinline__ T shfl_xor_sync(unsigned, T v, int lane_mask, int width = 32) { return __shfl_xor(v, lane_mask, width); }
template <typename T> __device__ __forceinline__ T shfl_down_sync(unsigned, T v, unsigned d, int width = 32) { return __shfl_down(v, d, width); }
template <typename T> __device__ __forceinline__ T shfl_up_sync(unsigned, T v, unsigned d, int width = 32) { return __shfl_up(v, d, width); }
template <typename T> __device__ __forceinline__ T shfl_sync(unsigned, T v, int src, int width = 32) { return __shfl(v, src, width); }
__device__ __forceinline__ unsigned ballot_sync(unsigned, int predicate) { return (unsigned)__ballot(predicate); }
__device__ __forceinline__ int any_sync(unsigned, int predicate) { return (int)(__ballot(predicate) != 0); }
// CUDA's __ldcg loads through L2 only, so a block reads what other blocks wrote (the split-K segments). hiprtc
// declares it only for halves; agent-scope atomic loads skip the CU's L1 the same way.
__device__ __forceinline__ float4 ldcg(const float4* p) {
    const float* f = (const float*)p;
    return make_float4(__hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), __hip_atomic_load(f + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT),
                       __hip_atomic_load(f + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), __hip_atomic_load(f + 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
}

// ---- matrix cores (gfx11/gfx12 WMMA) ----
// D[16x16] += A[16x16] * B^T[16x16]: both operands are 16 rows of 16 halves, which is exactly how the prefill GEMM
// stages X (positions) and W (output rows) in shared memory, so no transpose is needed. Each lane supplies the
// k-values of its own row of both operands and holds 8 results of column lane & 15, in rows STRATA_WMMA_ROW(lane, i).
// gfx12 splits the k dimension across the two half-waves (each lane gives 8 halves, lanes 16-31 the upper half) and
// gives lanes 16-31 rows 8..15; gfx11 takes all 16 halves from every lane (lanes 16-31 repeat lanes 0-15) and
// interleaves the rows, even ones in lanes 0-15 and odd ones in lanes 16-31 (as llama.cpp's mma.cuh has it for RDNA3).
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__) || defined(__gfx1152__)
#define STRATA_WMMA 1
#define STRATA_WMMA_K 16
#define STRATA_WMMA_ROW(lane, i) (2 * (i) + ((lane) >> 4))
#elif defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_WMMA 1
#define STRATA_WMMA_K 8
#define STRATA_WMMA_ROW(lane, i) (8 * ((lane) >> 4) + (i))
#else
#define STRATA_WMMA 0
#define STRATA_WMMA_K 0
#endif
// First k half the lane supplies: gfx12 lanes 16-31 give k 8..15, every lane on gfx11 gives k 0..15.
#define STRATA_WMMA_K0(lane) (((lane) >> 4) * (16 - STRATA_WMMA_K))

#if STRATA_WMMA
typedef _Float16 strata_hk __attribute__((ext_vector_type(STRATA_WMMA_K)));
typedef float strata_f8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ strata_f8 strata_wmma(strata_hk a, strata_hk b, strata_f8 c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}

/// STRATA_WMMA_K halves at `p` (16-byte aligned); lanes 16-31 read their own upper half of the same row.
__device__ __forceinline__ strata_hk strata_ld_wmma(const unsigned short* p) {
    union { uint4 u[STRATA_WMMA_K / 8]; strata_hk h; } c;
#pragma unroll
    for (int i = 0; i < STRATA_WMMA_K / 8; i++) c.u[i] = *(const uint4*)(p + 8 * i);
    return c.h;
}

/// The same STRATA_WMMA_K halves one tile row apart from each other: the B operand when the operand's rows are the
/// transposed side (the prefill attention's V, whose rows are head dimensions and whose k are keys). One load a have
/// instead of a 16-byte vector load.
__device__ __forceinline__ strata_hk strata_ld_wmma_t(const unsigned short* p, int stride) {
    strata_hk r;
#pragma unroll
    for (int i = 0; i < STRATA_WMMA_K; i++) r[i] = *(const _Float16*)(p + i * stride);
    return r;
}
#endif

}  // namespace strata_hip

#define __dp4a(a, b, c) (strata_hip::dp4a((a), (b), (c)))
#define __byte_perm(x, y, s) (strata_hip::byte_perm((x), (y), (s)))
#define __vsub4(a, b) (strata_hip::vsub4((a), (b)))
#define __vadd4(a, b) (strata_hip::vadd4((a), (b)))
#define __shfl_xor_sync(...) (strata_hip::shfl_xor_sync(__VA_ARGS__))
#define __shfl_down_sync(...) (strata_hip::shfl_down_sync(__VA_ARGS__))
#define __shfl_up_sync(...) (strata_hip::shfl_up_sync(__VA_ARGS__))
#define __shfl_sync(...) (strata_hip::shfl_sync(__VA_ARGS__))
#define __ballot_sync(m, p) (strata_hip::ballot_sync((m), (p)))
#define __any_sync(m, p) (strata_hip::any_sync((m), (p)))
#define __ldcg(p) (strata_hip::ldcg(p))
// The compiled-out sm_80+ paths still mention these; no CUDA-only helper may survive into a HIP build.
#define __trap() __builtin_trap()
#define __nanosleep(cycles) __builtin_amdgcn_s_sleep(1)

#endif  // STRATA_HIP_COMPAT_H
