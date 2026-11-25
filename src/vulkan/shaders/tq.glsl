// TurboQuant KV helpers, mirroring src/cuda/kernels.cu (TheTom's llama-cpp-turboquant design).
// Included from common.glsl after the K/V mode constants. K/V stay WHT-rotated in the cache; Q is rotated in
// attention and only the attention output is inverse-rotated.
const uint K_MODE = KV_MODE;
const uint V_MODE = VKV_MODE;
const bool TQ_K = KV_MODE >= 4u;
const bool TQ_V = VKV_MODE >= 4u;
const uint TQ_KBITS = KV_MODE == 5u ? 3u : (KV_MODE == 6u ? 2u : 4u);
const uint TQ_VBITS = VKV_MODE == 5u ? 3u : (VKV_MODE == 6u ? 2u : 4u);

float tq_flip(float x, uint b) { return uintBitsToFloat(floatBitsToUint(x) ^ (b << 31)); }

uint tq_sw(uint lane, bool inv) {
    uint i = lane >> 3u;
    if (inv) return i == 0u ? 0x16ACEE90u : i == 1u ? 0x3F628FDCu : i == 2u ? 0xB7A5357Au : 0xBEA56562u;
    return i == 0u ? 0xE46A0359u : i == 1u ? 0x42F85949u : i == 2u ? 0xA4C63BBBu : 0x49F250A9u;
}
uint tq_sw64(uint lane, bool inv) {
    uint w = (2u * lane) & 31u;
    uint i = lane >> 4u;
    return inv ? (i == 0u ? 0x16ACEE90u : 0x3F628FDCu) : (i == 0u ? 0xE46A0359u : 0x42F85949u);
}

// v.xyzw are dims 4 * lane .. 4 * lane + 3. All five butterfly stages are inside one logical 32-lane warp.
vec4 tq_wht4(vec4 vin, uint lane, bool inv) {
    precise vec4 v = vin;
    uint a = (tq_sw(lane, inv) >> (4u * (lane & 7u))) & 15u;
    v.x = tq_flip(v.x, a & 1u); v.y = tq_flip(v.y, (a >> 1u) & 1u);
    v.z = tq_flip(v.z, (a >> 2u) & 1u); v.w = tq_flip(v.w, (a >> 3u) & 1u);
    { precise float x = v.x, y = v.y; v.x = x + y; v.y = x - y;
      precise float z = v.z, w = v.w; v.z = z + w; v.w = z - w; }
    { precise float x = v.x, z = v.z; v.x = x + z; v.z = x - z;
      precise float y = v.y, w = v.w; v.y = y + w; v.w = y - w; }
    [[unroll]] for (uint m = 1u; m <= 16u; m <<= 1u) {
        precise vec4 p = vec4(subgroupShuffleXor(v.x, m), subgroupShuffleXor(v.y, m),
                              subgroupShuffleXor(v.z, m), subgroupShuffleXor(v.w, m));
        bool hi = (lane & m) != 0u;
        v = hi ? p - v : v + p;
    }
    uint b = (tq_sw(lane, !inv) >> (4u * (lane & 7u))) & 15u;
    v *= 0.08838834764831845;  // 1 / sqrt(128)
    v.x = tq_flip(v.x, b & 1u); v.y = tq_flip(v.y, (b >> 1u) & 1u);
    v.z = tq_flip(v.z, (b >> 2u) & 1u); v.w = tq_flip(v.w, (b >> 3u) & 1u);
    return v;
}

// hd == 64 (Inkling only): v.xy are dims 2 * lane and 2 * lane + 1.
vec2 tq_wht2(vec2 vin, uint lane, bool inv) {
    precise vec2 v = vin;
    uint a = (tq_sw64(lane, inv) >> ((2u * lane) & 31u)) & 3u;
    v.x = tq_flip(v.x, a & 1u); v.y = tq_flip(v.y, (a >> 1u) & 1u);
    { precise float x = v.x, y = v.y; v.x = x + y; v.y = x - y; }
    [[unroll]] for (uint m = 1u; m <= 16u; m <<= 1u) {
        precise vec2 p = vec2(subgroupShuffleXor(v.x, m), subgroupShuffleXor(v.y, m));
        bool hi = (lane & m) != 0u;
        v = hi ? p - v : v + p;
    }
    uint b = (tq_sw64(lane, !inv) >> ((2u * lane) & 31u)) & 3u;
    v *= 0.125;  // 1 / sqrt(64)
    v.x = tq_flip(v.x, b & 1u); v.y = tq_flip(v.y, (b >> 1u) & 1u);
    return v;
}

// Lloyd-Max midpoints and centroids for N(0, 1 / 128); the comparisons are the same as CUDA tq_nearest.
int tq_q4(float v) {
    int q = 0;
    q += int(v >= -0.212203); q += int(v >= -0.162947); q += int(v >= -0.127026);
    q += int(v >= -0.097164); q += int(v >= -0.070671); q += int(v >= -0.046174);
    q += int(v >= -0.022824); q += int(v >= 0.0);
    q += int(v >= 0.022824); q += int(v >= 0.046174); q += int(v >= 0.070671);
    q += int(v >= 0.097164); q += int(v >= 0.127026); q += int(v >= 0.162947);
    q += int(v >= 0.212203);
    return q;
}
int tq_q3(float v) {
    int q = 0;
    q += int(v >= -0.154496); q += int(v >= -0.092804); q += int(v >= -0.044243);
    q += int(v >= 0.0); q += int(v >= 0.044243); q += int(v >= 0.092804); q += int(v >= 0.154496);
    return q;
}
int tq_q2(float v) {
    int q = int(v >= -0.086728) + int(v >= 0.0) + int(v >= 0.086728);
    return q;
}
float tq_mag4(uint k) {
    return k == 0u ? 0.011349 : k == 1u ? 0.034299 : k == 2u ? 0.058050 : k == 3u ? 0.083292 :
           k == 4u ? 0.111036 : k == 5u ? 0.143016 : k == 6u ? 0.182877 : 0.241529;
}
float tq_mag3(uint k) {
    return k == 0u ? 0.021663 : k == 1u ? 0.066822 : k == 2u ? 0.118786 : 0.190207;
}
float tq_c4(int q) { bool p = q >= 8; float m = tq_mag4(p ? uint(q - 8) : uint(7 - q)); return p ? m : -m; }
float tq_c3(int q) { bool p = q >= 4; float m = tq_mag3(p ? uint(q - 4) : uint(3 - q)); return p ? m : -m; }
float tq_c2(int q) { bool p = q >= 2; float m = p ? 0.133462 : 0.039994; return p ? m : -m; }
int tq_nearest(uint bits, float v) { return bits == 3u ? tq_q3(v) : bits == 2u ? tq_q2(v) : tq_q4(v); }
float tq_centroid(uint bits, int q) { return bits == 3u ? tq_c3(q) : bits == 2u ? tq_c2(q) : tq_c4(q); }

shared float tq_red[256];
// Per-128-group sum, in the same butterfly order as CUDA tq_group_sum. `slot` must be unique per (row, group)
// within the workgroup; every invocation of that group calls this with the same slot.
float tq_group_reduce(float cs, uint slot) {
    float v = warp_sum(cs);
    uint lane = LANE, w = (gl_LocalInvocationID.x & 127u) >> 5u;
    if (lane < 4u) tq_red[4u * slot + lane] = 0.0;
    barrier();
    if (lane == 0u) tq_red[4u * slot + w] = v;
    barrier();
    float t = lane < 4u ? tq_red[4u * slot + lane] : 0.0;
    barrier();
    return warp_sum(t);
}
// Store one rotated, quantized value. Active lanes join the reduction; inactive ones contribute no energy and write nothing.
void tq_put(U16 c, uint e, uint hd, float x, float norm, bool valid, uint bits, uint slot) {
    int q = tq_nearest(bits, fdiv(x, norm));
    uint nib = uint(q), pair = nib | (subgroupShuffleXor(nib, 1u) << 4);
    uint we = e % hd, g = we >> 7u, w = we & 127u;
    uint r = e / hd * (hd / 2u + hd / 16u);
    if (valid && (we & 1u) == 0u) U8(uint64_t(c)).v[r + we / 2u] = uint8_t(pair);
    float cen = tq_centroid(bits, q);
    float recon = sqrt_(tq_group_reduce(valid ? cen * cen : 0.0, slot));
    if (valid && w == 0u)
        c.v[(r + hd / 2u) / 2u + g] = uint16_t(f2h(fdiv(norm, max(recon, 1e-6))));
}
float tq_ldkv(U16 c, uint off, uint hd, uint bits) {
    uint we = off % hd, r = off / hd * (hd / 2u + hd / 16u);
    float norm = float(uint16BitsToHalf(c.v[(r + hd / 2u) / 2u + (we >> 7u)]));
    uint b = uint(U8(uint64_t(c)).v[r + we / 2u]);
    int q = int((we & 1u) == 0u ? b & 15u : b >> 4);
    return rh(tq_centroid(bits, q) * norm);
}
uint tq_ldkv2(U16 c, uint off, uint hd, uint bits) {
    uint a0 = off % hd, r = off / hd * (hd / 2u + hd / 16u);
    float n0 = float(uint16BitsToHalf(c.v[(r + hd / 2u) / 2u + (a0 >> 7u)]));
    uint b = uint(U8(uint64_t(c)).v[r + a0 / 2u]);
    uint lo = f2h(tq_centroid(bits, int(b & 15u)) * n0);
    uint a1 = (off + 1u) % hd;
    float n1 = float(uint16BitsToHalf(c.v[(r + hd / 2u) / 2u + (a1 >> 7u)]));
    return lo | (f2h(tq_centroid(bits, int(b >> 4)) * n1) << 16);
}
uvec2 tq_ldkv4(U16 c, uint off, uint hd, uint bits) {
    return uvec2(tq_ldkv2(c, off, hd, bits), tq_ldkv2(c, off + 2u, hd, bits));
}
