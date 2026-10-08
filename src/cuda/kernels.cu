// Decode-path kernels. Dequant ports of src/quant/mod.rs (from llama.cpp ggml-quants.c, MIT License, ggml authors).
// Tables (iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kvalues_iq4nl) are prepended by src/cuda/mod.rs.

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ float h2f(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ u16 f2h(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ u32 ld16(const u8* p) { return p[0] | (p[1] << 8); }
__device__ __forceinline__ u32 ld32(const u8* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((u32)p[3] << 24); }
__device__ __forceinline__ float ldh(const u8* p) { return h2f((u16)ld16(p)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// Sum over the whole block; every thread gets the result. blockDim.x must be a multiple of 32.
__device__ float block_sum(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) s[w] = v;
    __syncthreads();
    v = lane < nw ? s[lane] : 0.0f;
    return warp_sum(v);
}

// ---- dequant: 8 values [8*sub, 8*sub+8) of a 256-element super-block ----

template <int T> __device__ void deq8(const u8* b, int sub, float* o);

template <> __device__ __forceinline__ void deq8<0>(const u8* b, int sub, float* o) {  // F32
    const float* f = (const float*)b + 8 * sub;
    for (int i = 0; i < 8; i++) o[i] = f[i];
}
template <> __device__ __forceinline__ void deq8<30>(const u8* b, int sub, float* o) {  // BF16
    for (int i = 0; i < 8; i++) o[i] = __uint_as_float(ld16(b + 16 * sub + 2 * i) << 16);
}
template <> __device__ __forceinline__ void deq8<8>(const u8* b, int sub, float* o) {  // Q8_0, 8 blocks of 34
    const u8* p = b + 34 * (sub >> 2);
    float d = ldh(p);
    const signed char* q = (const signed char*)p + 2 + (sub & 3) * 8;
    for (int i = 0; i < 8; i++) o[i] = (float)q[i] * d;
}
template <> __device__ __forceinline__ void deq8<10>(const u8* b, int sub, float* o) {  // Q2_K
    int e = sub * 8, n = e >> 7, r = e & 127, j = r >> 5, r2 = r & 31;
    u8 s = b[n * 8 + j * 2 + (r2 >> 4)];
    float dl = ldh(b + 80) * (float)(s & 0xF), ml = ldh(b + 82) * (float)(s >> 4);
    const u8* q = b + 16 + 32 * n + r2;
    for (int i = 0; i < 8; i++) o[i] = dl * (float)((q[i] >> (2 * j)) & 3) - ml;
}
template <> __device__ __forceinline__ void deq8<11>(const u8* b, int sub, float* o) {  // Q3_K
    int e = sub * 8, n = e >> 7, r = e & 127, j = r >> 5, i0 = r & 31, is = n * 8 + j * 2 + (i0 >> 4);
    const u8* sc = b + 96;
    int low = is < 8 ? (sc[is] & 0xF) : (sc[is - 8] >> 4);
    int hi = (sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
    float dl = ldh(b + 108) * (float)((low | (hi << 4)) - 32);
    u8 m = 1 << (n * 4 + j);
    for (int i = 0; i < 8; i++) {
        int qi = i0 + i;
        int v = ((b[32 + 32 * n + qi] >> (2 * j)) & 3) - ((b[qi] & m) ? 0 : 4);
        o[i] = dl * (float)v;
    }
}
__device__ __forceinline__ void scale_min_k4(int j, const u8* q, int& s, int& m) {
    if (j < 4) { s = q[j] & 63; m = q[j + 4] & 63; }
    else { s = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
template <> __device__ __forceinline__ void deq8<12>(const u8* b, int sub, float* o) {  // Q4_K
    int e = sub * 8, j = e >> 6, hi = (e >> 5) & 1, l = e & 31, s, m;
    scale_min_k4(2 * j + hi, b + 4, s, m);
    float d = ldh(b) * (float)s, mn = ldh(b + 2) * (float)m;
    const u8* q = b + 16 + 32 * j + l;
    for (int i = 0; i < 8; i++) o[i] = d * (float)(hi ? q[i] >> 4 : q[i] & 0xF) - mn;
}
template <> __device__ __forceinline__ void deq8<13>(const u8* b, int sub, float* o) {  // Q5_K
    int e = sub * 8, j = e >> 6, hi = (e >> 5) & 1, l = e & 31, s, m;
    scale_min_k4(2 * j + hi, b + 4, s, m);
    float d = ldh(b) * (float)s, mn = ldh(b + 2) * (float)m;
    const u8* qh = b + 16 + l;
    const u8* q = b + 48 + 32 * j + l;
    u8 bit = 1 << (2 * j + hi);
    for (int i = 0; i < 8; i++)
        o[i] = d * ((float)(hi ? q[i] >> 4 : q[i] & 0xF) + ((qh[i] & bit) ? 16.0f : 0.0f)) - mn;
}
template <> __device__ __forceinline__ void deq8<14>(const u8* b, int sub, float* o) {  // Q6_K
    int e = sub * 8, n = e >> 7, r = e & 127, qt = r >> 5, l = r & 31;
    const u8* ql = b + 64 * n;
    const u8* qh = b + 128 + 32 * n;
    float d = ldh(b + 208) * (float)((const signed char*)b)[192 + 8 * n + (l >> 4) + 2 * qt];
    for (int i = 0; i < 8; i++) {
        int li = l + i, q;
        switch (qt) {
            case 0: q = (ql[li] & 0xF) | ((qh[li] & 3) << 4); break;
            case 1: q = (ql[li + 32] & 0xF) | (((qh[li] >> 2) & 3) << 4); break;
            case 2: q = (ql[li] >> 4) | (((qh[li] >> 4) & 3) << 4); break;
            default: q = (ql[li + 32] >> 4) | (((qh[li] >> 6) & 3) << 4); break;
        }
        o[i] = d * (float)(q - 32);
    }
}
template <> __device__ __forceinline__ void deq8<17>(const u8* b, int sub, float* o) {  // IQ2_XS
    int ib = sub >> 2, l = sub & 3;
    u32 q = ld16(b + 2 + 2 * (4 * ib + l));
    u64 g = iq2xs_grid[q & 511];
    u8 sg = ksigns_iq2xs[q >> 9], sc = b[66 + ib];
    float db = ldh(b) * (0.5f + (float)(l < 2 ? sc & 0xF : sc >> 4)) * 0.25f;
    for (int j = 0; j < 8; j++) o[j] = db * (float)((g >> (8 * j)) & 0xFF) * ((sg >> j) & 1 ? -1.0f : 1.0f);
}
template <> __device__ __forceinline__ void deq8<18>(const u8* b, int sub, float* o) {  // IQ3_XXS
    int ib = sub >> 2, l = sub & 3;
    u32 aux = ld32(b + 66 + 4 * ib);
    float db = ldh(b) * (0.5f + (float)(aux >> 28)) * 0.5f;
    u8 sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
    u32 g1 = iq3xxs_grid[b[2 + 8 * ib + 2 * l]], g2 = iq3xxs_grid[b[3 + 8 * ib + 2 * l]];
    for (int j = 0; j < 4; j++) {
        o[j] = db * (float)((g1 >> (8 * j)) & 0xFF) * ((sg >> j) & 1 ? -1.0f : 1.0f);
        o[j + 4] = db * (float)((g2 >> (8 * j)) & 0xFF) * ((sg >> (j + 4)) & 1 ? -1.0f : 1.0f);
    }
}
template <> __device__ __forceinline__ void deq8<23>(const u8* b, int sub, float* o) {  // IQ4_XS
    int ib = sub >> 2, r = (sub & 3) * 8;
    u32 sh = ld16(b + 2);
    int ls = ((b[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    float dl = ldh(b) * (float)(ls - 32);
    const u8* q = b + 8 + 16 * ib + (r & 15);
    for (int i = 0; i < 8; i++) o[i] = dl * (float)kvalues_iq4nl[r < 16 ? q[i] & 0xF : q[i] >> 4];
}

// One warp per output row; lane `l` covers elements [8l, 8l+8) of every super-block.
template <int T, int SBB>
__device__ __forceinline__ void gemv_row(const u8* __restrict__ w, const float* __restrict__ x, float* y, int row, int nsb) {
    int lane = threadIdx.x & 31;
    float acc = 0.0f, v[8];
#pragma unroll 4
    for (int s = 0; s < nsb; s++) {
        deq8<T>(w + (size_t)s * SBB, lane, v);
        const float4* xs = (const float4*)(x + s * 256 + lane * 8);
        float4 a = xs[0], c = xs[1];
        acc += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * c.x + v[5] * c.y + v[6] * c.z + v[7] * c.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ---- integer path: 8 weights of a super-block as int8 (two packed words), with dequant value = d * q - m ----
// Activations are quantized per 32 elements to int8 in shared memory, and dot products use dp4a.

__device__ __forceinline__ int pack4(int a, int b, int c, int d) { return (a & 255) | ((b & 255) << 8) | ((c & 255) << 16) | (d << 24); }
__device__ __forceinline__ u32 ldu32(const u8* p) { return *(const u32*)p; }  // 4-byte aligned only
// Byte j of the result is 0xFF when bit j of s is set (j < 4).
__device__ __forceinline__ u32 smask4(u32 s) { return ((s * 0x00204081u) & 0x01010101u) * 0xFFu; }
__device__ __forceinline__ int apply_signs(u32 g, u32 s) { u32 m = smask4(s); return (int)__vsub4(g ^ m, m); }

template <int T> struct QI;
template <> struct QI<8> {  // Q8_0, 8 blocks of 34
    static constexpr bool MIN = false;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        const u8* p = b + 34 * (sub >> 2);
        d = ldh(p);
        const u8* q = p + 2 + (sub & 3) * 8;  // 2-byte aligned
        lo = (int)(ld16(q) | (ld16(q + 2) << 16)); hi = (int)(ld16(q + 4) | (ld16(q + 6) << 16));
    }
};
template <> struct QI<10> {  // Q2_K (84 bytes: rows 4-aligned)
    static constexpr bool MIN = true;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        int e = sub * 8, n = e >> 7, r = e & 127, j = r >> 5, r2 = r & 31;
        u8 s = b[n * 8 + j * 2 + (r2 >> 4)];
        d = ldh(b + 80) * (float)(s & 0xF); m = ldh(b + 82) * (float)(s >> 4);
        const u8* q = b + 16 + 32 * n + r2;
        lo = (int)((ldu32(q) >> (2 * j)) & 0x03030303u); hi = (int)((ldu32(q + 4) >> (2 * j)) & 0x03030303u);
    }
};
template <> struct QI<11> {  // Q3_K
    static constexpr bool MIN = false;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        float v[8];
        deq8<11>(b, sub, v);  // ponytail: via floats; Q3_K is rare in the target quants
        int e = sub * 8, n = e >> 7, r = e & 127, j = r >> 5, i0 = r & 31, is = n * 8 + j * 2 + (i0 >> 4);
        const u8* sc = b + 96;
        int low = is < 8 ? (sc[is] & 0xF) : (sc[is - 8] >> 4);
        int hs = (sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
        d = ldh(b + 108) * (float)((low | (hs << 4)) - 32);
        int q[8];
        for (int i = 0; i < 8; i++) q[i] = d != 0.0f ? __float2int_rn(v[i] / d) : 0;
        lo = pack4(q[0], q[1], q[2], q[3]); hi = pack4(q[4], q[5], q[6], q[7]);
    }
};
template <> struct QI<12> {  // Q4_K (144 bytes)
    static constexpr bool MIN = true;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        int e = sub * 8, j = e >> 6, h = (e >> 5) & 1, l = e & 31, s, mi;
        scale_min_k4(2 * j + h, b + 4, s, mi);
        d = ldh(b) * (float)s; m = ldh(b + 2) * (float)mi;
        const u8* q = b + 16 + 32 * j + l;
        lo = (int)((ldu32(q) >> (4 * h)) & 0x0F0F0F0Fu); hi = (int)((ldu32(q + 4) >> (4 * h)) & 0x0F0F0F0Fu);
    }
};
template <> struct QI<13> {  // Q5_K (176 bytes)
    static constexpr bool MIN = true;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        int e = sub * 8, j = e >> 6, h = (e >> 5) & 1, l = e & 31, s, mi, bit = 2 * j + h;
        scale_min_k4(bit, b + 4, s, mi);
        d = ldh(b) * (float)s; m = ldh(b + 2) * (float)mi;
        const u8* qh = b + 16 + l;
        const u8* q = b + 48 + 32 * j + l;
        lo = (int)(((ldu32(q) >> (4 * h)) & 0x0F0F0F0Fu) | (((ldu32(qh) >> bit) & 0x01010101u) << 4));
        hi = (int)(((ldu32(q + 4) >> (4 * h)) & 0x0F0F0F0Fu) | (((ldu32(qh + 4) >> bit) & 0x01010101u) << 4));
    }
};
template <> struct QI<14> {  // Q6_K (210 bytes: rows 2-aligned)
    static constexpr bool MIN = false;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        int e = sub * 8, n = e >> 7, r = e & 127, qt = r >> 5, l = r & 31;
        const u8* ql = b + 64 * n + (qt & 1) * 32 + l;
        const u8* qh = b + 128 + 32 * n + l;
        d = ldh(b + 208) * (float)((const signed char*)b)[192 + 8 * n + (l >> 4) + 2 * qt];
        u32 l0 = ld16(ql) | (ld16(ql + 2) << 16), l1 = ld16(ql + 4) | (ld16(ql + 6) << 16);
        u32 h0 = ld16(qh) | (ld16(qh + 2) << 16), h1 = ld16(qh + 4) | (ld16(qh + 6) << 16);
        int ns = 4 * (qt >> 1), hs = 2 * qt;
        u32 a = ((l0 >> ns) & 0x0F0F0F0Fu) | (((h0 >> hs) & 0x03030303u) << 4);
        u32 c = ((l1 >> ns) & 0x0F0F0F0Fu) | (((h1 >> hs) & 0x03030303u) << 4);
        lo = (int)__vsub4(a, 0x20202020u); hi = (int)__vsub4(c, 0x20202020u);
    }
};
template <> struct QI<17> {  // IQ2_XS (74 bytes: rows 2-aligned)
    static constexpr bool MIN = false;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        int ib = sub >> 2, l = sub & 3;
        u32 q = ld16(b + 2 + 2 * (4 * ib + l));
        u64 g = iq2xs_grid[q & 511];
        u32 sg = ksigns_iq2xs[q >> 9], sc = b[66 + ib];
        d = ldh(b) * (0.5f + (float)(l < 2 ? sc & 0xF : sc >> 4)) * 0.25f;
        lo = apply_signs((u32)g, sg & 15); hi = apply_signs((u32)(g >> 32), sg >> 4);
    }
};
template <> struct QI<18> {  // IQ3_XXS (98 bytes: rows 2-aligned)
    static constexpr bool MIN = false;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        int ib = sub >> 2, l = sub & 3;
        u32 aux = ld16(b + 66 + 4 * ib) | (ld16(b + 68 + 4 * ib) << 16);
        d = ldh(b) * (0.5f + (float)(aux >> 28)) * 0.5f;
        u32 sg = ksigns_iq2xs[(aux >> (7 * l)) & 127], qq = ld16(b + 2 + 8 * ib + 2 * l);
        lo = apply_signs(iq3xxs_grid[qq & 255], sg & 15); hi = apply_signs(iq3xxs_grid[qq >> 8], sg >> 4);
    }
};
template <> struct QI<23> {  // IQ4_XS (136 bytes)
    static constexpr bool MIN = false;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        int ib = sub >> 2, r = (sub & 3) * 8;
        u32 sh = ld16(b + 2);
        int ls = ((b[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        d = ldh(b) * (float)(ls - 32);
        const u8* q = b + 8 + 16 * ib + (r & 15);
        u32 w0 = ldu32(q), w1 = ldu32(q + 4), sft = r < 16 ? 0 : 4;
        int v[8];
        for (int i = 0; i < 4; i++) { v[i] = kvalues_iq4nl[(w0 >> (8 * i + sft)) & 0xF]; v[i + 4] = kvalues_iq4nl[(w1 >> (8 * i + sft)) & 0xF]; }
        lo = pack4(v[0], v[1], v[2], v[3]); hi = pack4(v[4], v[5], v[6], v[7]);
    }
};

#ifndef LLAMA_Q8
#define LLAMA_Q8 0
#endif
// q8_K scale over the lanes that share one 256-block (all 32 for 8 values per lane, the 16 of a half-warp for 16):
// lane holds candidate value v at index i (largest |v| of its values, lowest index); the block keeps the largest
// |v|, lowest index on ties, as llama.cpp's quantize_row_q8_K does. id = -127 / v, d = 1 / id.
// |id v| <= 127 for every value of the block, so llama.cpp's clamp to 127 never fires and is left out.
template <int LANES = 32>
__device__ __forceinline__ void q8k_scale(float v, int i, float& d, float& id) {
    unsigned mask = __activemask();  // quant_q8 can leave a half-warp idle
    #pragma unroll
    for (int o = 1; o < LANES; o <<= 1) {
        float ov = __shfl_xor_sync(mask, v, o); int oi = __shfl_xor_sync(mask, i, o);
        if (fabsf(ov) > fabsf(v) || (fabsf(ov) == fabsf(v) && oi < i)) { v = ov; i = oi; }
    }
    id = v != 0.0f ? -127.0f / v : 0.0f;
    d = v != 0.0f ? 1.0f / id : 0.0f;
}

// Quantize x[0, n) into shared int8 qs[n] and per-32 scales xd[n / 32]; the whole block participates.
__device__ __forceinline__ void quant_x(const float* __restrict__ x, int n, signed char* qs, float* xd) {
    for (int i = threadIdx.x * 8; i < n; i += blockDim.x * 8) {
        float4 a = *(const float4*)(x + i), c = *(const float4*)(x + i + 4);
        float mx = fmaxf(fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))),
                         fmaxf(fmaxf(fabsf(c.x), fabsf(c.y)), fmaxf(fabsf(c.z), fabsf(c.w))));
#if LLAMA_Q8
        float d, id;
        {  // llama.cpp q8_K: one scale per 256 (the warp), -127 / the signed value of largest magnitude
            float sv = 0.f; int si = 1 << 30;
            float vv[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
            for (int j = 0; j < 8; j++) if (fabsf(vv[j]) == mx && si > i + j) { sv = vv[j]; si = i + j; }
            q8k_scale(sv, si, d, id);
        }
#else
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        float d = mx / 127.0f, id = mx > 0.0f ? 127.0f / mx : 0.0f;
#endif
        int2 q = make_int2(pack4(__float2int_rn(a.x * id), __float2int_rn(a.y * id), __float2int_rn(a.z * id), __float2int_rn(a.w * id)),
                           pack4(__float2int_rn(c.x * id), __float2int_rn(c.y * id), __float2int_rn(c.z * id), __float2int_rn(c.w * id)));
        *(int2*)(qs + i) = q;
        if ((threadIdx.x & 3) == 0) xd[i >> 5] = d;
    }
    __syncthreads();
}

template <int T, int SBB>
__device__ __forceinline__ void gemv_row_q(const u8* __restrict__ w, const signed char* qs, const float* xd, float* y, int row, int nsb) {
    int lane = threadIdx.x & 31;
    float acc = 0.0f;
#pragma unroll 4
    for (int s = 0; s < nsb; s++) {
        int lo, hi; float d, m = 0.0f;
        QI<T>::get(w + (size_t)s * SBB, lane, lo, hi, d, m);
        int2 xq = *(const int2*)(qs + s * 256 + lane * 8);
        float dx = xd[s * 8 + (lane >> 2)];
        acc += d * dx * (float)__dp4a(lo, xq.x, __dp4a(hi, xq.y, 0));
        if (QI<T>::MIN) acc -= m * dx * (float)__dp4a(0x01010101, xq.x, __dp4a(0x01010101, xq.y, 0));
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// gemv_row_q for nt <= 4 inputs: x t is qs + t * 256 * nsb with scales xd + t * 8 * nsb; y t is y + t * ys.
template <int T, int SBB, int NT>
__device__ __forceinline__ void gemv_row_qn(const u8* __restrict__ w, const signed char* qs, const float* xd, float* y, int ys,
                                            int row, int nsb) {
    int lane = threadIdx.x & 31;
    float acc[NT] = {};
#pragma unroll 4
    for (int s = 0; s < nsb; s++) {
        int lo, hi; float d, m = 0.0f;
        QI<T>::get(w + (size_t)s * SBB, lane, lo, hi, d, m);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            int2 xq = *(const int2*)(qs + (t * nsb + s) * 256 + lane * 8);
            float dx = xd[(t * nsb + s) * 8 + (lane >> 2)];
            acc[t] += d * dx * (float)__dp4a(lo, xq.x, __dp4a(hi, xq.y, 0));
            if (QI<T>::MIN) acc[t] -= m * dx * (float)__dp4a(0x01010101, xq.x, __dp4a(0x01010101, xq.y, 0));
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        float a = warp_sum(acc[t]);
        if (lane == 0) y[(size_t)t * ys + row] = a;
    }
}

#define FTYPES(X) X(0, f32, 1024) X(30, bf16, 512)
#define ITYPES(X) X(8, q8_0, 272) X(10, q2_k, 84) X(11, q3_k, 110) X(12, q4_k, 144) X(13, q5_k, 176) X(14, q6_k, 210) \
    X(17, iq2_xs, 74) X(18, iq3_xxs, 98) X(23, iq4_xs, 136)
#define QTYPES(X) FTYPES(X) ITYPES(X)

#define DEF_DEQUANT(T, NAME, SBB)                                                                                 \
    extern "C" __global__ void dequant_##NAME(const u8* w, float* out, int nsb) {                                 \
        const u8* r = w + (size_t)blockIdx.x * nsb * SBB;                                                       \
        for (int s = threadIdx.x; s < nsb * 32; s += blockDim.x)                                                \
            deq8<T>(r + (size_t)(s >> 5) * SBB, s & 31, out + ((size_t)blockIdx.x * nsb + (s >> 5)) * 256 + (s & 31) * 8); \
    }
QTYPES(DEF_DEQUANT)

// Float types read x directly.
#define DEF_FGEMV(T, NAME, SBB)                                                                                   \
    extern "C" __global__ void gemv_##NAME(const u8* w, const float* x, float* y, int rows, int nsb, int) {   \
        int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                                 \
        if (row < rows) gemv_row<T, SBB>(w + (size_t)row * nsb * SBB, x, y, row, nsb);                         \
    }                                                                                                           \
    extern "C" __global__ void moe_gemv_##NAME(const u8* w, const int* ids, long long stride, const float* x,  \
                                               int x_stride, int xg, float* y, int rows, int nsb, int) {      \
        int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, k = blockIdx.y;                                 \
        if (row < rows && ids[k] >= 0)                                                                          \
            gemv_row<T, SBB>(w + ids[k] * stride + (size_t)row * nsb * SBB, x + (size_t)(k / xg) * x_stride,    \
                             y + (size_t)k * rows, row, nsb);                                                   \
    }
FTYPES(DEF_FGEMV)

// Quantized types: dynamic shared memory holds x as int8 (256 * nsb bytes) then its scales (8 * nsb floats).
// Each block covers 8 * rpw rows (warp w: rows w, w + 8, ...) so small matrices amortize the x quantization.
// moe_gemv: grid.y = k; weights at w + ids[k] * stride (skipped when ids[k] < 0), input x + (k / xg) * x_stride,
// output y + k * rows. grid.z = 2 also runs a second matrix at byte offset w2 into output at byte offset y2.
#define DEF_IGEMV(T, NAME, SBB)                                                                                   \
    extern "C" __global__ void gemv_##NAME(const u8* w, const float* x, float* y, int rows, int nsb, int rpw) { \
        extern __shared__ __align__(16) signed char xs[];                                                       \
        float* xd = (float*)(xs + 256 * nsb);                                                                   \
        quant_x(x, 256 * nsb, xs, xd);                                                                          \
        int r0 = blockIdx.x * 8 * rpw + (threadIdx.x >> 5), r1 = min(rows, r0 - (threadIdx.x >> 5) + 8 * rpw);  \
        for (int row = r0; row < r1; row += 8) gemv_row_q<T, SBB>(w + (size_t)row * nsb * SBB, xs, xd, y, row, nsb); \
    }                                                                                                           \
    extern "C" __global__ void moe_gemv_##NAME(const u8* w, const int* ids, long long stride, const float* x,  \
                                               int x_stride, int xg, float* y, int rows, int nsb, int rpw,      \
                                               long long w2, long long y2) {                                   \
        extern __shared__ __align__(16) signed char xs[];                                                       \
        int k = blockIdx.y, e = ids[k];                                                                         \
        if (e < 0) return;                                                                                      \
        float* xd = (float*)(xs + 256 * nsb);                                                                   \
        quant_x(x + (size_t)(k / xg) * x_stride, 256 * nsb, xs, xd);                                            \
        const u8* we = w + e * stride + blockIdx.z * w2;                                                        \
        y = (float*)((char*)y + blockIdx.z * y2);                                                               \
        int r0 = blockIdx.x * 8 * rpw + (threadIdx.x >> 5), r1 = min(rows, r0 - (threadIdx.x >> 5) + 8 * rpw);  \
        for (int row = r0; row < r1; row += 8)                                                                  \
            gemv_row_q<T, SBB>(we + (size_t)row * nsb * SBB, xs, xd, y + (size_t)k * rows, row, nsb);           \
    }
ITYPES(DEF_IGEMV)

// gemvn: y[t] = W x[t] for t < nt <= 4 (x t at x + t * xs, y t at y + t * ys); each weight block is read once.
#define DEF_IGEMVN(T, NAME, SBB)                                                                                  \
    extern "C" __global__ void gemvn_##NAME(const u8* w, const float* x, int xs_, float* y, int ys, int rows, int nsb, \
                                            int rpw, int nt) {                                                    \
        extern __shared__ __align__(16) signed char xs[];                                                       \
        float* xd = (float*)(xs + nt * 256 * nsb);                                                              \
        for (int t = 0; t < nt; t++) quant_x(x + (size_t)t * xs_, 256 * nsb, xs + t * 256 * nsb, xd + t * 8 * nsb); \
        int r0 = blockIdx.x * 8 * rpw + (threadIdx.x >> 5), r1 = min(rows, r0 - (threadIdx.x >> 5) + 8 * rpw);  \
        for (int row = r0; row < r1; row += 8) {                                                               \
            const u8* wr = w + (size_t)row * nsb * SBB;                                                         \
            if (nt == 2) gemv_row_qn<T, SBB, 2>(wr, xs, xd, y, ys, row, nsb);                                    \
            else if (nt == 3) gemv_row_qn<T, SBB, 3>(wr, xs, xd, y, ys, row, nsb);                               \
            else if (nt == 4) gemv_row_qn<T, SBB, 4>(wr, xs, xd, y, ys, row, nsb);                               \
            else gemv_row_qn<T, SBB, 1>(wr, xs, xd, y, ys, row, nsb);                                            \
        }                                                                                                       \
    }
ITYPES(DEF_IGEMVN)

// ---- prefill GEMM: Y[m, n] = sum_k X[m, k] W[n, k] on tensor cores (fp16 inputs, fp32 accumulate) ----
// Work is a list of tiles of up to GEMM_BM consecutive positions m0 .. m0 + cnt, each with its own weight matrix (so one
// launch covers every expert of a MoE layer). X row of position m is xrow[m] (or m when xrow is null).
// grid = (ceil(rows / 64), tiles), block = 2 * GEMM_BM.
struct GemmTile { const u8* w; int m0, cnt; };

__device__ __forceinline__ u32 ld_s32(const u16* p) { return *(const u32*)p; }
// Four 8x8 f16 matrices from shared memory; lane l gives the row address for matrix l / 8 (.trans transposes each).
__device__ __forceinline__ void ldsm4(u32* r, const u16* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(u32* r, const u16* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 h2pack(float a, float b) { return (u32)f2h(a) | ((u32)f2h(b) << 16); }

// act: 0 = store, 1 = y = silu(y) * result (SwiGLU over a gate already in y), 2 = y += result.
// tiles = null: one dense matrix wd over positions 0 .. nd, tile blockIdx.y = positions BM * blockIdx.y ...
// BM positions x 64 rows per block, K in steps of BK; warps form a (BM / 32) x 2 grid, each owning 32 x 32 outputs.
template <int T, int SBB, int BM, int BK>
__device__ __forceinline__ void gemm_q(const GemmTile* tiles, const u8* wd, int nd, const float* __restrict__ x, int x_stride,
                                       const int* xrow, float* y, int y_stride, int rows, int nsb, int act) {
    constexpr int P = BK + 8, NT = BM * 2, SUBS = BK / 8;  // smem pitch in halves (padding: conflict-free fragments)
    __shared__ __align__(16) u16 xs[BM * P], ws[64 * P];
    GemmTile tl = tiles ? tiles[blockIdx.y] : GemmTile{wd, (int)blockIdx.y * BM, min(BM, nd - (int)blockIdx.y * BM)};
    int n0 = blockIdx.x * 64, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const float* xr[BM * BK / 4 / NT];  // row pointers of the X elements this thread loads (null past cnt)
    #pragma unroll
    for (int q = 0; q < BM * BK / 4 / NT; q++) {
        int r = (tid + q * NT) / (BK / 4);
        xr[q] = r < tl.cnt ? x + (size_t)(xrow ? xrow[tl.m0 + r] : tl.m0 + r) * x_stride : nullptr;
    }
    float acc[2][4][4] = {};
    for (int kb = 0; kb < nsb * (256 / BK); kb++) {
        int sb = kb / (256 / BK), part = kb % (256 / BK);
        #pragma unroll
        for (int i = tid; i < 64 * SUBS; i += NT) {
            int r = i / SUBS, j = i % SUBS;
            float v[8] = {};
            if (n0 + r < rows) deq8<T>(tl.w + ((size_t)(n0 + r) * nsb + sb) * SBB, part * SUBS + j, v);
            *(uint4*)(ws + r * P + j * 8) = make_uint4(h2pack(v[0], v[1]), h2pack(v[2], v[3]), h2pack(v[4], v[5]), h2pack(v[6], v[7]));
        }
        #pragma unroll
        for (int q = 0; q < BM * BK / 4 / NT; q++) {
            int i = tid + q * NT, r = i / (BK / 4), c = (i % (BK / 4)) * 4;
            float4 v = xr[q] ? *(const float4*)(xr[q] + kb * BK + c) : make_float4(0.f, 0.f, 0.f, 0.f);
            *(uint2*)(xs + r * P + c) = make_uint2(h2pack(v.x, v.y), h2pack(v.z, v.w));
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            u32 a[2][4], b[4][2];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                const u16* p = xs + (wm + mt * 16 + g) * P + kk + 2 * t;
                a[mt][0] = ld_s32(p); a[mt][1] = ld_s32(p + 8 * P); a[mt][2] = ld_s32(p + 8); a[mt][3] = ld_s32(p + 8 * P + 8);
            }
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const u16* p = ws + (wn + nt * 8 + g) * P + kk + 2 * t;
                b[nt][0] = ld_s32(p); b[nt][1] = ld_s32(p + 8);
            }
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) mma16816(acc[mt][nt], a[mt][0], a[mt][1], a[mt][2], a[mt][3], b[nt][0], b[nt][1]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int r = wm + mt * 16 + g + 8 * h, n = n0 + wn + nt * 8 + 2 * t;
                if (r >= tl.cnt) continue;
                float* yr = y + (size_t)(tl.m0 + r) * y_stride;
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    if (n + e >= rows) break;
                    float v = acc[mt][nt][2 * h + e];
                    yr[n + e] = act == 0 ? v : act == 1 ? silu(yr[n + e]) * v : yr[n + e] + v;
                }
            }
}

#ifndef GEMM_BM
#define GEMM_BM 128
#define GEMM_BK 64
#endif
#define DEF_GEMM(T, NAME, SBB)                                                                                    \
    extern "C" __global__ void __launch_bounds__(GEMM_BM * 2) gemm_##NAME(const GemmTile* tiles, const u8* wd, int nd, \
            const float* x, int x_stride, const int* xrow, float* y, int y_stride, int rows, int nsb, int act) {  \
        gemm_q<T, SBB, GEMM_BM, GEMM_BK>(tiles, wd, nd, x, x_stride, xrow, y, y_stride, rows, nsb, act);       \
    }
QTYPES(DEF_GEMM)

// ---- prefill GEMM on int8 tensor cores (integer weight types) ----
// X is first quantized by quant_q8 into q8 rows: cols int8, then cols / 32 f32 scales, then cols / 16 f32 sums of the
// dequantized values (row bytes Q8_ROW(cols)). Weights come from QI<T> as int8 with a scale (and min) per 16, and each
// 16-deep int32 mma result is scaled into f32 accumulators: y += dx dw sum(qx qw) - m_w sum(dx qx).
// Same tiles, grid and block as gemm_q; K steps of 64 with X double-buffered by cp.async.
#define Q8_ROW(cols) ((cols) + (cols) / 8 + (cols) / 4)

// grid = rows of x, block = 256: q8 row r of out = x row r.
extern "C" __global__ void quant_q8(const float* x, int x_stride, u8* out, int cols) {
    const float* xr = x + (size_t)blockIdx.x * x_stride;
    u8* o = out + (size_t)blockIdx.x * Q8_ROW(cols);
    for (int i = threadIdx.x * 16; i < cols; i += blockDim.x * 16) {
        float v[16];
        #pragma unroll
        for (int j = 0; j < 4; j++) *(float4*)(v + 4 * j) = *(const float4*)(xr + i + 4 * j);
        float mx = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++) mx = fmaxf(mx, fabsf(v[j]));
#if LLAMA_Q8
        float d, id;
        {
            float sv = 0.f; int si = 1 << 30;
            for (int j = 0; j < 16; j++) if (fabsf(v[j]) == mx && si > i + j) { sv = v[j]; si = i + j; }
            q8k_scale<16>(sv, si, d, id);
        }
#else
        mx = fmaxf(mx, __shfl_xor_sync(__activemask(), mx, 1));  // lanes 2k, 2k + 1 share one 32-block
        float d = mx / 127.0f, id = mx > 0.0f ? 127.0f / mx : 0.0f;
#endif
        int q[16], sum = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) { q[j] = __float2int_rn(v[j] * id); sum += q[j]; }
        *(int4*)(o + i) = make_int4(pack4(q[0], q[1], q[2], q[3]), pack4(q[4], q[5], q[6], q[7]),
                                    pack4(q[8], q[9], q[10], q[11]), pack4(q[12], q[13], q[14], q[15]));
        if (!(threadIdx.x & 1)) ((float*)(o + cols))[i / 32] = d;
        ((float*)(o + cols + cols / 8))[i / 16] = d * (float)sum;
    }
}

__device__ __forceinline__ void mma_s8_16816(int* c, u32 a0, u32 a1, u32 b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma_s8_16832(int* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// int -> float for |c| < 2^22 without the quarter-rate I2F: 2^23 + 2^22 + c has c in its low mantissa bits.
__device__ __forceinline__ float i2f_small(int c) { return __int_as_float(0x4B400000 + c) - 12582912.0f; }
// Weight types whose scale (and min) is constant over 32 weights: one 32-deep mma per scale.
template <int T> __host__ __device__ constexpr bool scale32() { return T == 8 || T == 12 || T == 13 || T == 18 || T == 23; }
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(d), "l"(src), "r"(ok ? 8 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

template <int T, int SBB, int BM>
__device__ __forceinline__ void gemmi_q(const GemmTile* tiles, const u8* wd, int nd, const u8* __restrict__ x,
                                        const int* xrow, float* y, int y_stride, int rows, int nsb, int act) {
    constexpr int P = 80;  // smem pitch in bytes: 64 int8 + padding (conflict-free fragments, 16-byte aligned rows)
    static_assert(BM == 128, "one X row half per thread");
    __shared__ __align__(16) signed char xq[2][BM * P], wq[64 * P];
    __shared__ __align__(16) float xd[2][BM][2], xsum[2][BM][4];
    __shared__ float wsc[64][4], wmn[64][4];
    GemmTile tl = tiles ? tiles[blockIdx.y] : GemmTile{wd, (int)blockIdx.y * BM, min(BM, nd - (int)blockIdx.y * BM)};
    int n0 = blockIdx.x * 64, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int wm = (warp >> 1) * 32, wn = (warp & 1) * 32, cols = nsb * 256;
    // X: thread copies 32 int8 (half of the row's 64) of row xr_, plus the row's scales (xh = 0) or sums (xh = 1).
    int xr_ = tid >> 1, xh = tid & 1;
    bool xok = xr_ < tl.cnt;
    const u8* xp = x + (size_t)(xok ? (xrow ? xrow[tl.m0 + xr_] : tl.m0 + xr_) : 0) * Q8_ROW(cols);
    auto ldx = [&](int kb) {
        int b = kb & 1;
        cp_async16(&xq[b][xr_ * P + xh * 32], xp + kb * 64 + xh * 32, xok);
        cp_async16(&xq[b][xr_ * P + xh * 32 + 16], xp + kb * 64 + xh * 32 + 16, xok);
        if (xh) cp_async16(&xsum[b][xr_][0], xp + cols + cols / 8 + kb * 16, xok);
        else cp_async8(&xd[b][xr_][0], xp + cols + kb * 8, xok);
        cp_async_commit();
    };
    // W: thread decodes the 16 weights (subs 2 wi, 2 wi + 1 of the step) of row wr_.
    int wr_ = tid >> 2, wi = tid & 3;
    const u8* wp = n0 + wr_ < rows ? tl.w + (size_t)(n0 + wr_) * nsb * SBB : nullptr;
    float acc[2][4][4] = {};
    const int nk = nsb * 4;
    // W step kb is decoded into registers during step kb - 1's mma and stored after it.
    int lo0, hi0, lo1, hi1; float wd0, wm0;
    auto ldw = [&](int kb) {
        lo0 = hi0 = lo1 = hi1 = 0; wd0 = wm0 = 0.f;
        float d1, m1;
        if (wp) {
            const u8* sbp = wp + (size_t)(kb >> 2) * SBB;
            int sub = (kb & 3) * 8 + 2 * wi;
            QI<T>::get(sbp, sub, lo0, hi0, wd0, wm0);
            QI<T>::get(sbp, sub + 1, lo1, hi1, d1, m1);
        }
    };
    auto stw = [&]() {
        *(int4*)(wq + wr_ * P + wi * 16) = make_int4(lo0, hi0, lo1, hi1);
        wsc[wr_][wi] = wd0; wmn[wr_][wi] = wm0;
    };
    ldx(0);
    ldw(0);
    stw();
    for (int kb = 0; kb < nk; kb++) {
        if (kb + 1 < nk) ldx(kb + 1);
        if (kb + 1 < nk) cp_async_wait<1>(); else cp_async_wait<0>();
        __syncthreads();
        if (kb + 1 < nk) ldw(kb + 1);
        const int b = kb & 1;
        constexpr int KS = scale32<T>() ? 32 : 16;  // depth of one mma (one weight scale)
        #pragma unroll
        for (int h = 0; h < 64 / KS; h++) {
            u32 a[2][4], bw[4][2];
            float da[2][2], sa[2][2], dbv[4][2], mb[4][2];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                int r = wm + mt * 16 + g;
                const signed char* p = &xq[b][r * P + h * KS + 4 * t];
                a[mt][0] = *(const u32*)p; a[mt][1] = *(const u32*)(p + 8 * P);
                if (KS == 32) { a[mt][2] = *(const u32*)(p + 16); a[mt][3] = *(const u32*)(p + 8 * P + 16); }
                da[mt][0] = xd[b][r][h * KS / 32]; da[mt][1] = xd[b][r + 8][h * KS / 32];
                if (QI<T>::MIN) {
                    sa[mt][0] = KS == 32 ? xsum[b][r][2 * h] + xsum[b][r][2 * h + 1] : xsum[b][r][h];
                    sa[mt][1] = KS == 32 ? xsum[b][r + 8][2 * h] + xsum[b][r + 8][2 * h + 1] : xsum[b][r + 8][h];
                }
            }
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                int n = wn + nt * 8;
                const signed char* p = wq + (n + g) * P + h * KS + 4 * t;
                bw[nt][0] = *(const u32*)p;
                if (KS == 32) bw[nt][1] = *(const u32*)(p + 16);
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    dbv[nt][e] = wsc[n + 2 * t + e][h * KS / 16];
                    if (QI<T>::MIN) mb[nt][e] = wmn[n + 2 * t + e][h * KS / 16];
                }
            }
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    int c[4] = {0, 0, 0, 0};
                    if (KS == 32) mma_s8_16832(c, a[mt][0], a[mt][1], a[mt][2], a[mt][3], bw[nt][0], bw[nt][1]);
                    else mma_s8_16816(c, a[mt][0], a[mt][1], bw[nt][0]);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float f = i2f_small(c[i]) * (da[mt][i >> 1] * dbv[nt][i & 1]);
                        if (QI<T>::MIN) f -= sa[mt][i >> 1] * mb[nt][i & 1];
                        acc[mt][nt][i] += f;
                    }
                }
        }
        __syncthreads();
        if (kb + 1 < nk) stw();
    }
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int r = wm + mt * 16 + g + 8 * h, n = n0 + wn + nt * 8 + 2 * t;
                if (r >= tl.cnt) continue;
                float* yr = y + (size_t)(tl.m0 + r) * y_stride;
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    if (n + e >= rows) break;
                    float v = acc[mt][nt][2 * h + e];
                    yr[n + e] = act == 0 ? v : act == 1 ? silu(yr[n + e]) * v : yr[n + e] + v;
                }
            }
}

#define DEF_GEMMI(T, NAME, SBB)                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) gemmi_##NAME(const GemmTile* tiles, const u8* wd, int nd,    \
            const u8* x, const int* xrow, float* y, int y_stride, int rows, int nsb, int act) {                    \
        gemmi_q<T, SBB, 128>(tiles, wd, nd, x, xrow, y, y_stride, rows, nsb, act);                              \
    }
ITYPES(DEF_GEMMI)

// ---- elementwise / norm ----

// grid = rows. out[r] = rmsnorm(x[r]) * w (same w for every row).
extern "C" __global__ void rms_norm(const float* x, int x_stride, const float* w, float* out, int out_stride, int n, float eps) {
    const float* xr = x + (size_t)blockIdx.x * x_stride;
    float* o = out + (size_t)blockIdx.x * out_stride;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    float sc = 1.0f / sqrtf(block_sum(s) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * sc * w[i];
}

// grid = rows, block 1024. out[r] = index of the largest x in row r (first one on ties, like ops::argmax).
extern "C" __global__ void argmax_rows(const float* x, int n, int* out) {
    const float* xr = x + (size_t)blockIdx.x * n;
    float b = __int_as_float(0xff800000); int bi = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (xr[i] > b || bi == n) { b = xr[i]; bi = i; }
    for (int o = 16; o > 0; o >>= 1) {
        float v = __shfl_xor_sync(0xffffffff, b, o); int vi = __shfl_xor_sync(0xffffffff, bi, o);
        if (v > b || (v == b && vi < bi)) { b = v; bi = vi; }
    }
    __shared__ float sv[32]; __shared__ int si[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = b; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        b = lane < (blockDim.x >> 5) ? sv[lane] : __int_as_float(0xff800000); bi = lane < (blockDim.x >> 5) ? si[lane] : n;
        for (int o = 16; o > 0; o >>= 1) {
            float v = __shfl_xor_sync(0xffffffff, b, o); int vi = __shfl_xor_sync(0xffffffff, bi, o);
            if (v > b || (v == b && vi < bi)) { b = v; bi = vi; }
        }
        if (lane == 0) out[blockIdx.x] = bi;
    }
}

// grid = rows, in place. Row r is vector r % per of token r / per (tokens `stride` floats apart).
extern "C" __global__ void l2_norm(float* x, int n, int per, int stride, float eps) {
    float* xr = x + (size_t)(blockIdx.x / per) * stride + (size_t)(blockIdx.x % per) * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    float sc = 1.0f / fmaxf(sqrtf(block_sum(s)), eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) xr[i] *= sc;
}

// Per-token values live in device memory so a captured graph can replay: step[0] = position, step[1] = CPU worker seq.

// grid = tokens * per heads, block = n_rot / 2. NEOX pairs (i, i + n_rot/2). Token t sits at position step[0] + t.
extern "C" __global__ void rope_neox(float* x, int stride, int per, int n_rot, const int* step, float base) {
    int pos = step[0] + blockIdx.x / per;
    float* h = x + (size_t)blockIdx.x * stride;
    int i = threadIdx.x, half = n_rot / 2;
    double theta = (double)pos * pow((double)base, -2.0 * i / n_rot);
    float s = (float)sin(theta), c = (float)cos(theta), a = h[i], b = h[i + half];
    h[i] = a * c - b * s;
    h[i + half] = a * s + b * c;
}

extern "C" __global__ void add(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

extern "C" __global__ void swiglu(float* g, const float* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = silu(g[i]) * u[i];
}

// ---- Gated DeltaNet ----

// n = tokens * nv values; per-head parameters repeat every nv.
extern "C" __global__ void ssm_gates(float* beta, float* g, const float* dt_bias, const float* a, int n, int nv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, h = i % nv;
    if (i < n) { beta[i] = sigmoid(beta[i]); g[i] = softplus(g[i] + dt_bias[h]) * a[h]; }
}

// Causal depthwise conv over [state..., current] then SiLU; state keeps the last kw-1 inputs, oldest first.
extern "C" __global__ void conv_silu(const float* qkv, float* state, const float* w, float* y, int ch, int kw) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float acc = 0.0f;
    for (int k = 0; k < kw - 1; k++) acc += state[k * ch + c] * w[c * kw + k];
    acc += qkv[c] * w[c * kw + kw - 1];
    y[c] = silu(acc);
    for (int k = 0; k < kw - 2; k++) state[k * ch + c] = state[(k + 1) * ch + c];
    state[(kw - 2) * ch + c] = qkv[c];
}

// Parallel over tokens for a chunk: y[t] = silu(conv over inputs t-kw+1 .. t), inputs before the chunk from state.
extern "C" __global__ void conv_seq(const float* qkv, const float* state, const float* w, float* y, int ch, int kw, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * ch) return;
    int t = i / ch, c = i % ch;
    float acc = 0.0f;
    for (int k = 0; k < kw; k++) {
        int s = t - (kw - 1) + k;  // input index; < 0 reads state slot kw - 1 + s
        acc += (s < 0 ? state[(kw - 1 + s) * ch + c] : qkv[(size_t)s * ch + c]) * w[c * kw + k];
    }
    y[i] = silu(acc);
}

// After conv_seq: state = last kw-1 inputs. One thread per channel, ascending slots read only higher ones.
// Also writes the state after each of the first min(nsnap, T - 1) tokens to snap (one state each), for rollback.
extern "C" __global__ void conv_state(const float* qkv, float* state, int ch, int kw, int T, float* snap, int nsnap) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    for (int n = 1; n <= min(nsnap, T - 1); n++)
        for (int k = 0; k < kw - 1; k++) {
            int s = n - (kw - 1) + k;
            snap[((size_t)(n - 1) * (kw - 1) + k) * ch + c] = s < 0 ? state[(kw - 1 + s) * ch + c] : qkv[(size_t)s * ch + c];
        }
    for (int k = 0; k < kw - 1; k++) {
        int s = T - (kw - 1) + k;
        state[k * ch + c] = s < 0 ? state[(kw - 1 + s) * ch + c] : qkv[(size_t)s * ch + c];
    }
}

// grid = (value heads, dv / (8 * R)), block = 256. Each state row (dk floats, dk in 32/64/128/256) lives in the
// registers of L = dk / 16 lanes, 16 floats each (float4 chunks s, s + L, ...), so a warp holds R = 32 / L rows and
// row dot products reduce over L lanes. The warp scans the T tokens in order.
// Token t: y + t * ych = [q | k | v] after conv and l2 norm; beta, g at t * nv; output at o + t * nv * dv.
// The state after each of the first min(nsnap, T - 1) tokens also goes to snap (S's layout, one state each).
// ponytail: sequential scan (latency-bound per token); a chunked WY form would use tensor cores for long prompts.
extern "C" __global__ void deltanet_seq(const float* y, float* S, const float* beta, const float* g, float* o,
                                        int nk, int nv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap) {
    int L = dk >> 4, R = 32 / L, lane = threadIdx.x & 31, s = lane % L, vh = blockIdx.x, kh = vh % nk;
    int j0 = (blockIdx.y * 8 + (threadIdx.x >> 5)) * R, j = j0 + lane / L;
    if (j0 >= dv) return;  // whole warp past the end (R divides dv)
    float4* row = (float4*)(S + ((size_t)vh * dv + j) * dk);
    float r[16];
#pragma unroll
    for (int i = 0; i < 4; i++) { float4 v = row[s + L * i]; r[4 * i] = v.x; r[4 * i + 1] = v.y; r[4 * i + 2] = v.z; r[4 * i + 3] = v.w; }
    // Token inputs do not depend on the state: token tok + 1's are loaded while token tok computes.
    float nk_[16], nq[16], ng, nb, nvv;
    auto ld = [&](int tok) {
        const float* yt = y + (size_t)tok * ych;
        const float4* q4 = (const float4*)(yt + kh * dk);
        const float4* k4 = (const float4*)(yt + nk * dk + kh * dk);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float4 v = k4[s + L * i]; nk_[4 * i] = v.x; nk_[4 * i + 1] = v.y; nk_[4 * i + 2] = v.z; nk_[4 * i + 3] = v.w;
            v = q4[s + L * i]; nq[4 * i] = v.x; nq[4 * i + 1] = v.y; nq[4 * i + 2] = v.z; nq[4 * i + 3] = v.w;
        }
        ng = g[tok * nv + vh]; nb = beta[tok * nv + vh]; nvv = yt[2 * nk * dk + vh * dv + j];
    };
    ld(0);
    for (int tok = 0; tok < T; tok++) {
        float kk[16], qv[16], decay = expf(ng), bt = nb, vt = nvv, sk = 0.0f, oq = 0.0f;
#pragma unroll
        for (int e = 0; e < 16; e++) { kk[e] = nk_[e]; qv[e] = nq[e]; }
        if (tok + 1 < T) ld(tok + 1);
#pragma unroll
        for (int e = 0; e < 16; e++) { r[e] *= decay; sk += r[e] * kk[e]; }
        for (int off = L >> 1; off; off >>= 1) sk += __shfl_xor_sync(0xffffffff, sk, off);
        float d = bt * (vt - sk);
#pragma unroll
        for (int e = 0; e < 16; e++) { r[e] += d * kk[e]; oq += r[e] * qv[e]; }
        for (int off = L >> 1; off; off >>= 1) oq += __shfl_xor_sync(0xffffffff, oq, off);
        if (s == 0) o[(size_t)tok * nv * dv + vh * dv + j] = oq * scale;
        if (tok < min(nsnap, T - 1)) {
            float4* sr = (float4*)(snap + (size_t)tok * nv * dv * dk + ((size_t)vh * dv + j) * dk);
#pragma unroll
            for (int i = 0; i < 4; i++) sr[s + L * i] = make_float4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) row[s + L * i] = make_float4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
}

// grid = heads: out = rmsnorm(o_h) * w * silu(z_h), in place on o.
extern "C" __global__ void norm_gate(float* o, const float* w, const float* z, int n, float eps) {
    float* h = o + (size_t)blockIdx.x * n;
    const float* zh = z + (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += h[i] * h[i];
    float sc = 1.0f / sqrtf(block_sum(s) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[i] = h[i] * sc * w[i] * silu(zh[i]);
}

// ---- attention ----

// Store `total` values (consecutive tokens of n each) starting at position step[0].
extern "C" __global__ void kv_store(const float* k, const float* v, u16* kc, u16* vc, int n, int total, const int* step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, pos = step[0];
    if (i < total) { kc[(size_t)pos * n + i] = f2h(k[i]); vc[(size_t)pos * n + i] = f2h(v[i]); }
}

#define INFINITY __int_as_float(0x7f800000)
// PER consecutive f16 values to f32 with one vector load (p aligned to 2 * PER bytes).
template <int PER> __device__ __forceinline__ void ldh_vec(const u16* p, float* o) {
    if constexpr (PER >= 8) {
        for (int c = 0; c < PER / 8; c++) {
            uint4 v = ((const uint4*)p)[c];
            const u16* h = (const u16*)&v;
            for (int i = 0; i < 8; i++) o[c * 8 + i] = h2f(h[i]);
        }
    } else if constexpr (PER == 4) {
        uint2 v = *(const uint2*)p;
        const u16* h = (const u16*)&v;
        for (int i = 0; i < 4; i++) o[i] = h2f(h[i]);
    } else {
        unsigned v = *(const unsigned*)p;
        o[0] = h2f((u16)v); o[1] = h2f((u16)(v >> 16));
    }
}

// Split-position decode attention. grid = (nh / G, ns splits): block (b, s) serves the G query heads b*G .. b*G+G-1
// (one KV head) over an ns-th of the positions, so each K/V row is read once for all G heads. Each warp walks its
// share with online softmax (lane owns PER = hd / 32 consecutive dims), warps merge head by head, and the block
// writes per head its unnormalized partial [acc (hd) | max | sum] to part; attn_merge combines the splits.
template <int PER, int G>
__device__ void attn_decode_t(const float* q, const u16* kc, const u16* vc, float* part, const int* step,
                              int nh, int nkv, float scale, int bx) {
    constexpr int hd = PER * 32;
    // At least 16 positions per split: short contexts use few blocks instead of many near-empty ones.
    int n_pos = step[0] + 1, ns = gridDim.y, per_s = max((n_pos + ns - 1) / ns, 16);
    int p0 = blockIdx.y * per_s, p1 = min(n_pos, p0 + per_s);
    if (p0 >= n_pos) {  // empty split
        for (int i = 0; i < G; i++)
            for (int d = threadIdx.x; d < hd + 2; d += blockDim.x)
                part[((size_t)(bx * G + i) * ns + blockIdx.y) * (hd + 2) + d] = d == hd ? -INFINITY : 0.0f;
        return;
    }
    extern __shared__ float sacc[];  // nw * hd
    __shared__ float sm[32], sl[32];
    int h0 = bx * G, g = h0 / (nh / nkv), w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float qv[G][PER], acc[G][PER], kv[PER], vv[PER], kn[PER], vn[PER], m[G], l[G];
#pragma unroll
    for (int i = 0; i < G; i++) {
        m[i] = -INFINITY; l[i] = 0.0f;
        for (int d = 0; d < PER; d++) { qv[i][d] = q[(h0 + i) * hd + lane * PER + d]; acc[i][d] = 0.0f; }
    }
    auto row = [&](int t) { return ((size_t)t * nkv + g) * hd + lane * PER; };
    if (p0 + w < p1) { ldh_vec<PER>(kc + row(p0 + w), kn); ldh_vec<PER>(vc + row(p0 + w), vn); }
    for (int t = p0 + w; t < p1; t += nw) {
        for (int d = 0; d < PER; d++) { kv[d] = kn[d]; vv[d] = vn[d]; }
        if (t + nw < p1) { ldh_vec<PER>(kc + row(t + nw), kn); ldh_vec<PER>(vc + row(t + nw), vn); }  // prefetch
        float c[G], p[G];
#pragma unroll
        for (int i = 0; i < G; i++) {
            float s = 0.0f;
            for (int d = 0; d < PER; d++) s += qv[i][d] * kv[d];
            p[i] = s;
        }
#pragma unroll
        for (int i = 0; i < G; i++) {
            float s = warp_sum(p[i]) * scale, mn = fmaxf(m[i], s);
            c[i] = expf(m[i] - mn); p[i] = expf(s - mn);
            l[i] = l[i] * c[i] + p[i]; m[i] = mn;
        }
#pragma unroll
        for (int i = 0; i < G; i++)
            for (int d = 0; d < PER; d++) acc[i][d] = acc[i][d] * c[i] + p[i] * vv[d];
    }
#pragma unroll
    for (int i = 0; i < G; i++) {
        if (lane == 0) { sm[w] = m[i]; sl[w] = l[i]; }
        for (int d = 0; d < PER; d++) sacc[w * hd + lane * PER + d] = acc[i][d];
        __syncthreads();
        float* pp = part + ((size_t)(h0 + i) * ns + blockIdx.y) * (hd + 2);
        float M = -INFINITY;
        for (int k = 0; k < nw; k++) M = fmaxf(M, sm[k]);
        float L = 0.0f;
        for (int k = 0; k < nw; k++) L += sl[k] * expf(sm[k] - M);
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            float o = 0.0f;
            for (int k = 0; k < nw; k++) o += sacc[k * hd + d] * expf(sm[k] - M);
            pp[d] = o;
        }
        if (threadIdx.x == 0) { pp[hd] = M; pp[hd + 1] = L; }
        __syncthreads();
    }
}

// Two kernels, both in the decode graph; exactly one does work for a given context length. Below gq_min positions,
// nt query tokens: block x serves token x % nt and head block x / nt (tokens side by side, so their blocks run at
// the same time and share K/V rows in L2), with q, part and step at that token's offsets (pz part floats per token).
// attn_decode (grid (nh, ns1 splits)) serves one head per block. From gq_min on, attn_decode_gqa (grid (nh / G, splits))
// serves the G query heads of a KV head per block, reading each K/V row once; it needs ~250 registers per thread,
// so it is a separate entry point to keep attn_decode's occupancy.
#define ATT_PER(F) switch (hd) { case 64: F(2) break; case 128: F(4) break; case 256: F(8) break; case 512: F(16) break; }
#define ATT_1(PER) attn_decode_t<PER, 1>(q, kc, vc, part, step, nh, nkv, scale, blockIdx.x / nt);
extern "C" __global__ void attn_decode(const float* q, const u16* kc, const u16* vc, float* part, const int* step,
                                       int nh, int nkv, int hd, float scale, int gq_min, int pz, int nt) {
    int t = blockIdx.x % nt;
    q += (size_t)t * nh * hd; part += (size_t)t * pz; step += 2 * t;
    if (step[0] + 1 >= gq_min) return;
    ATT_PER(ATT_1)
}
#define ATT_G(PER)                                                                             \
    switch (nh * nt / gridDim.x) {                                                             \
        case 1: attn_decode_t<PER, 1>(q, kc, vc, part, step, nh, nkv, scale, blockIdx.x / nt); break;           \
        case 2: attn_decode_t<PER, 2>(q, kc, vc, part, step, nh, nkv, scale, blockIdx.x / nt); break;           \
        case 4: if constexpr (PER <= 16) attn_decode_t<PER, 4>(q, kc, vc, part, step, nh, nkv, scale, blockIdx.x / nt); break; \
        case 8: if constexpr (PER <= 8) attn_decode_t<PER, 8>(q, kc, vc, part, step, nh, nkv, scale, blockIdx.x / nt); break;  \
    }
extern "C" __global__ void attn_decode_gqa(const float* q, const u16* kc, const u16* vc, float* part, const int* step,
                                           int nh, int nkv, int hd, float scale, int gq_min, int pz, int nt) {
    int t = blockIdx.x % nt;
    q += (size_t)t * nh * hd; part += (size_t)t * pz; step += 2 * t;
    if (step[0] + 1 < gq_min) return;
    ATT_PER(ATT_G)
}

// grid = query heads: out = (sum of split partials) / (sum of split sums) * sigmoid(gate). Split 0 is never empty.
extern "C" __global__ void attn_merge(const float* part, const float* qg, float* out, int hd, int ns, const int* step,
                                       int ns1, int gq_min) {
    int h = blockIdx.x;
    part += (size_t)blockIdx.z * gridDim.x * ns * (hd + 2); step += 2 * blockIdx.z;
    qg += (size_t)blockIdx.z * gridDim.x * 2 * hd; out += (size_t)blockIdx.z * gridDim.x * hd;
    if (step[0] + 1 < gq_min) ns = ns1;  // attn_decode ran, not attn_decode_gqa
    const float* pp = part + (size_t)h * ns * (hd + 2);
    __shared__ float ws[256];  // split max, then split weight (ns <= blockDim.x <= 256)
    if (threadIdx.x < ns) ws[threadIdx.x] = pp[threadIdx.x * (hd + 2) + hd];
    __syncthreads();
    float M = -INFINITY, L = 0.0f;
    for (int s = 0; s < ns; s++) M = fmaxf(M, ws[s]);
    __syncthreads();
    if (threadIdx.x < ns) ws[threadIdx.x] = expf(ws[threadIdx.x] - M);
    __syncthreads();
    for (int s = 0; s < ns; s++) L += pp[s * (hd + 2) + hd + 1] * ws[s];
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < ns; s++) o += pp[s * (hd + 2) + d] * ws[s];
        out[h * hd + d] = o / L * sigmoid(qg[h * 2 * hd + hd + d]);
    }
}

// ---- MoE ----

// One warp per token (grid = tokens): softmax over n logits (in place), top-k (ties to the lower index), renormalized
// weights.
extern "C" __global__ void router_topk(float* p, int n, int k, float extra_scale, int* ids, float* wts) {
    int lane = threadIdx.x;
    p += (size_t)blockIdx.x * n; ids += blockIdx.x * k; wts += blockIdx.x * k;
    float m = -INFINITY;
    for (int i = lane; i < n; i += 32) m = fmaxf(m, p[i]);
    m = warp_max(m);
    float s = 0.0f;
    for (int i = lane; i < n; i += 32) { float e = expf(p[i] - m); p[i] = e; s += e; }
    s = warp_sum(s);
    for (int i = lane; i < n; i += 32) p[i] /= s;
    __syncwarp();
    float sum = 0.0f;
    for (int j = 0; j < k; j++) {
        float bv = -1.0f; int bi = 0x7fffffff;
        for (int i = lane; i < n; i += 32) {
            bool taken = false;
            for (int t = 0; t < j; t++) taken |= ids[t] == i;
            if (!taken && (p[i] > bv || (p[i] == bv && i < bi))) { bv = p[i]; bi = i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { ids[j] = bi; wts[j] = bv; }
        sum += bv;
        __syncwarp();
    }
    float scale = 1.0f / fmaxf(sum, 6.103515625e-5f);
    if (extra_scale != 0.0f) scale *= extra_scale;
    if (lane < k) wts[lane] *= scale;
}

// x += sum_k w[k] * ye[k] + sigmoid(sg[0]) * sh + cpu (sh and cpu may be null).
// Expert cache lookup: slots[j] = table[ids[j]] (-1 = not resident); wh[j] = routing weight of hits, else 0.
extern "C" __global__ void map_hits(const int* ids, const int* table, int* slots, const float* wts, float* wh, int k) {
    int j = threadIdx.x;
    if (j >= k) return;
    int s = table[ids[j]];
    slots[j] = s;
    wh[j] = s >= 0 ? wts[j] : 0.0f;
}

// Copy one layer's routing for T tokens (k entries each) to the host-mapped mailbox `mb` (activations only if an
// expert missed), store T in word 1, then raise the request flag. One block.
extern "C" __global__ void moe_publish(const int* ids, const float* wts, const int* slots, const float* h,
                                       unsigned int* mb, int k, int n, int h_off, const int* step, int T) {
    int t = threadIdx.x, kt = k * T;
    unsigned int seq = step[1];
    int miss = __syncthreads_or(t < kt && slots[t] < 0);
    if (t < kt) { mb[32 + t] = ids[t]; mb[32 + kt + t] = __float_as_uint(wts[t]); mb[32 + 2 * kt + t] = slots[t]; }
    if (miss) for (int i = t; i < n * T; i += blockDim.x) mb[h_off + i] = __float_as_uint(h[i]);
    if (t == 0) mb[1] = T;
    __threadfence_system();
    __syncthreads();
    if (t == 0) *(volatile unsigned int*)mb = seq;
}

// x += sum_j w[j] ye[j] + sigmoid(sg) sh + cpu. If a slot missed, wait for the CPU worker's done flag first.
// grid.y = token: x, ye, w, slots, sh, sg and the CPU result advance by one token's worth.
extern "C" __global__ void moe_combine(float* x, const float* ye, const float* w, int k, const float* sh,
                                       const float* sg, const unsigned int* mb, int y_off, const int* step,
                                       const int* slots, int n) {
    unsigned int seq = step[1];
    int tk = blockIdx.y;
    x += (size_t)tk * n; ye += (size_t)tk * k * n; w += tk * k; slots += tk * k; y_off += tk * n;
    if (sh) { sh += (size_t)tk * n; sg += tk; }
    __shared__ int miss;
    if (threadIdx.x == 0) {
        int m = 0;
        if (mb) for (int j = 0; j < k; j++) m |= slots[j] < 0;
        if (m) {
            unsigned int spins = 0;
            while (((volatile const unsigned int*)mb)[16] != seq) {
                __nanosleep(100);
                if (++spins == (1u << 27)) __trap();  // ~15 s: CPU worker gone
            }
            __threadfence_system();
        }
        miss = m;
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float y = 0.0f;
    for (int j = 0; j < k; j++) if (w[j] != 0.0f) y += w[j] * ye[(size_t)j * n + i];
    if (miss) y += __uint_as_float(((volatile const unsigned int*)mb)[y_off + i]);
    if (sh) y += sigmoid(sg[0]) * sh[i];
    x[i] += y;
}

// Prefill combine over T tokens: x[t] += sum_j wts[t, j] ye[inv[t, j]] + sigmoid(sg[t]) sh[t] (sh may be null).
extern "C" __global__ void moe_combine_seq(float* x, const float* ye, const float* wts, const int* inv, int k,
                                           const float* sh, const float* sg, int n, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    int t = i / n, d = i % n;
    float y = 0.0f;
    for (int j = 0; j < k; j++) y += wts[t * k + j] * ye[(size_t)inv[t * k + j] * n + d];
    if (sh) y += sigmoid(sg[t]) * sh[i];
    x[i] += y;
}

// ---- attention prefill (flash, tensor cores) ----

// grid = (ceil(T / 64), heads), block = 128: warp w owns queries 16w .. 16w + 15 of the tile. Query t (position
// step[0] + t) attends to cached keys 0 .. step[0] + t. q is [T, nh * HD] after rope, qg [T, nh * 2 * HD] holds the
// gate at +HD per head; out [T, nh * HD] = softmax(q K^T * scale) V * sigmoid(gate). Shared: Q, K and V tiles, all
// row-major; mma fragments come from ldmatrix (.trans for V).
#define FA_BQ 64
#define FA_BK 32
template <int HD>
__device__ __forceinline__ void attn_prefill_t(const float* q, const u16* kc, const u16* vc, const float* qg, float* out,
                                               const int* step, int T, int nh, int nkv, float scale) {
    constexpr int PQ = HD + 8;
    extern __shared__ __align__(16) u16 fa_sm[];
    u16 *qs = fa_sm, *ks = qs + FA_BQ * PQ, *vs = ks + FA_BK * PQ;
    int pos0 = step[0], q0 = blockIdx.x * FA_BQ, h = blockIdx.y, kvh = h / (nh / nkv), kvd = nkv * HD;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    int lr = (lane & 7) + ((lane >> 3) & 1) * 8, lc = (lane >> 4) * 8;  // ldmatrix row / column offsets of this lane
    for (int i = tid; i < FA_BQ * HD / 4; i += 128) {
        int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
        float4 v = q0 + r < T ? *(const float4*)(q + (size_t)(q0 + r) * nh * HD + h * HD + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        *(uint2*)(qs + r * PQ + c) = make_uint2(h2pack(v.x * scale, v.y * scale), h2pack(v.z * scale, v.w * scale));
    }
    int last = pos0 + min(q0 + FA_BQ, T) - 1;  // last key any query of the tile sees
    int qp[2] = {pos0 + q0 + warp * 16 + g, pos0 + q0 + warp * 16 + g + 8};
    float o[HD / 8][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int k0 = 0; k0 <= last; k0 += FA_BK) {
        __syncthreads();
        for (int i = tid; i < FA_BK * HD / 8; i += 128) {
            int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (k0 + r <= last) {
                size_t off = (size_t)(k0 + r) * kvd + kvh * HD + c;
                kv = *(const uint4*)(kc + off); vv = *(const uint4*)(vc + off);
            }
            *(uint4*)(ks + r * PQ + c) = kv;
            *(uint4*)(vs + r * PQ + c) = vv;
        }
        __syncthreads();
        float s[FA_BK / 8][4] = {};
#pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            u32 a[4], b[4];
            ldsm4(a, qs + (warp * 16 + lr) * PQ + kk + lc);
            for (int nt = 0; nt < FA_BK / 8; nt += 2) {
                // matrices: keys nt*8 / dims kk, keys nt*8 / dims kk+8, keys nt*8+8 / dims kk, keys nt*8+8 / dims kk+8
                ldsm4(b, ks + (nt * 8 + (lane & 7) + (lane >> 4) * 8) * PQ + kk + ((lane >> 3) & 1) * 8);
                mma16816(s[nt], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma16816(s[nt + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }
        float mx[2] = {m[0], m[1]};
        for (int nt = 0; nt < FA_BK / 8; nt++)
            for (int e = 0; e < 4; e++) {
                int key = k0 + nt * 8 + 2 * t4 + (e & 1), r = e >> 1;
                if (key > qp[r]) s[nt][e] = -INFINITY;
                mx[r] = fmaxf(mx[r], s[nt][e]);
            }
        float c[2];
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
            c[r] = expf(m[r] - mx[r]);  // key 0 is in the first block, so mx is finite from then on
            m[r] = mx[r];
            l[r] *= c[r];
        }
        for (int nt = 0; nt < FA_BK / 8; nt++)
            for (int e = 0; e < 4; e++) { float p = expf(s[nt][e] - m[e >> 1]); s[nt][e] = p; l[e >> 1] += p; }
        for (int d = 0; d < HD / 8; d++) { o[d][0] *= c[0]; o[d][1] *= c[0]; o[d][2] *= c[1]; o[d][3] *= c[1]; }
#pragma unroll
        for (int kb = 0; kb < FA_BK / 16; kb++) {
            u32 a0 = h2pack(s[2 * kb][0], s[2 * kb][1]), a1 = h2pack(s[2 * kb][2], s[2 * kb][3]);
            u32 a2 = h2pack(s[2 * kb + 1][0], s[2 * kb + 1][1]), a3 = h2pack(s[2 * kb + 1][2], s[2 * kb + 1][3]);
#pragma unroll
            for (int d = 0; d < HD / 8; d += 2) {
                u32 b[4];  // transposed: keys kb*16 / dims d*8, keys +8 / dims d*8, keys / dims d*8+8, keys +8 / dims d*8+8
                ldsm4t(b, vs + (kb * 16 + lr) * PQ + d * 8 + lc);
                mma16816(o[d], a0, a1, a2, a3, b[0], b[1]);
                mma16816(o[d + 1], a0, a1, a2, a3, b[2], b[3]);
            }
        }
    }
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
    }
    for (int r = 0; r < 2; r++) {
        int t = q0 + warp * 16 + g + 8 * r;
        if (t >= T) continue;
        float* orow = out + (size_t)t * nh * HD + h * HD;
        const float* gate = qg + (size_t)t * nh * 2 * HD + h * 2 * HD + HD;
        for (int d = 0; d < HD / 8; d++)
            for (int e = 0; e < 2; e++) {
                int col = d * 8 + 2 * t4 + e;
                orow[col] = o[d][2 * r + e] / l[r] * sigmoid(gate[col]);
            }
    }
}
#define DEF_FA(HD) \
    extern "C" __global__ void __launch_bounds__(128) attn_prefill_##HD(const float* q, const u16* kc, const u16* vc, \
            const float* qg, float* out, const int* step, int T, int nh, int nkv, float scale) {                      \
        attn_prefill_t<HD>(q, kc, vc, qg, out, step, T, nh, nkv, scale);                                            \
    }
DEF_FA(64) DEF_FA(128) DEF_FA(256)
