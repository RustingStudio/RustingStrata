// Dequantization of weight type T (the ggml type id, specialization constant 1) with SBB bytes per 256-element
// super-block (constant 2): ports of QI<T> and deq8<T> in kernels.cu, which port src/quant/mod.rs (from llama.cpp
// ggml-quants.c, MIT License, ggml authors).
#include "tables.glsl"

layout(constant_id = 1) const uint T = 0;
layout(constant_id = 2) const uint SBB = 1024;

uint u8_(U8 w, uint o) { return uint(w.v[o]); }
uint ld16(U8 w, uint o) { return uint(U16(uint64_t(w)).v[o >> 1]); }  // 2-aligned only (every quant field is)
uint ld32(U8 w, uint o) { return ld16(w, o) | ld16(w, o + 2) << 16; }
float h2f(uint h) { return float(uint16BitsToFloat16(uint16_t(h))); }
float ldh(U8 w, uint o) { return h2f(ld16(w, o)); }
int sb(uint x) { return int(x << 24) >> 24; }  // the low byte, signed

// IQ1_M (T 29) or IQ1_S (19) sub-block l of block ib: grid index, scale and the sign of the 1/8 offset.
void iq1_sub(U8 w, uint b, uint ib, uint l, out uint idx, out float dl, out bool neg) {
    if (T == 29) {  // d is split over the top nibbles of the four scale words
        uint s0 = ld16(w, b + 48), s1 = ld16(w, b + 50), s2 = ld16(w, b + 52), s3 = ld16(w, b + 54);
        float d = h2f((s0 >> 12) | ((s1 >> 8) & 0xF0) | ((s2 >> 4) & 0xF00) | (s3 & 0xF000));
        uint sc = ld16(w, b + 48 + 2 * (ib / 2)), qh = u8_(w, b + 32 + 2 * ib + l / 2) >> (4 * (l & 1));
        dl = d * float(2 * ((sc >> (6 * (ib & 1) + 3 * (l >> 1))) & 7) + 1);
        idx = u8_(w, b + 4 * ib + l) | (qh & 7) << 8;
        neg = (qh & 8) != 0;
    } else {
        uint qh = ld16(w, b + 34 + 2 * ib);
        dl = ldh(w, b) * float(2 * ((qh >> 12) & 7) + 1);
        idx = u8_(w, b + 2 + 4 * ib + l) | ((qh >> (3 * l)) & 7) << 8;
        neg = (qh & 0x8000) != 0;
    }
}

// Sign j of the 8 sign bits sg applied to grid byte j of g.
int sgn(uint g, uint sg, uint j) { int v = int((g >> (8 * (j & 3))) & 255); return ((sg >> j) & 1) != 0 ? -v : v; }

layout(buffer_reference, std430, buffer_reference_align = 8) buffer U32x2 { uvec2 v[]; };
// Q4_K, Q5_K and IQ4_XS blocks (144, 176, 136 bytes) and every field ld8 reads in them are 8-aligned; Q2_K's (84) 4-aligned.
const bool A8 = T == 12 || T == 13 || T == 23;
const bool A4 = T == 10;
// 8 bytes at 2-aligned w + o: one load for the A8 types, two word loads for A4, else the three words around them
// shifted (ld8a2 in kernels.cu) without a branch: in Q8_0 the lanes of a subgroup differ in alignment.
uvec2 ld8(U8 w, uint o) {
    uint64_t a = uint64_t(w) + o;
    if (A8) return U32x2(a).v[0];
    if (A4) { U32 u = U32(a); return uvec2(u.v[0], u.v[1]); }
    uint sh = uint(a & 2) * 8, w0 = U32(a & ~3ul).v[0], w1 = U32((a & ~3ul) + 4).v[0], w2 = U32((a + 7) & ~3ul).v[0];
    return uvec2(uint(packUint2x32(uvec2(w0, w1)) >> sh), uint(packUint2x32(uvec2(w1, w2)) >> sh));
}
// Bytewise a - c (each byte of a below 128; CUDA __vsub4).
uint vsub4(uint a, uint c) { return ((a | 0x80808080u) - c) ^ 0x80808080u; }
// The types QI<T> in kernels.cu reads with word loads: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
const bool WORDS = T == 8 || (T >= 10 && T <= 14);

// qi8 for the WORDS types, the 8 int8 values packed in lo, hi (QI<T>::get).
void qw_k(U8 w, uint b, uint sub, out uint lo, out uint hi, out float d, out float m) {
    m = 0.0;
    uvec2 v;
    if (T == 8) {  // Q8_0, 8 blocks of 34
        uint p = b + 34 * (sub >> 2);
        d = ldh(w, p);
        v = ld8(w, p + 2 + (sub & 3) * 8); lo = v.x; hi = v.y;
    } else if (T == 10) {  // Q2_K
        uint e = sub * 8, n = e >> 7, r = e & 127, j = r >> 5, r2 = r & 31;
        uint s = u8_(w, b + n * 8 + j * 2 + (r2 >> 4));
        d = ldh(w, b + 80) * float(s & 15); m = ldh(w, b + 82) * float(s >> 4);
        v = ld8(w, b + 16 + 32 * n + r2);
        lo = (v.x >> (2 * j)) & 0x03030303u; hi = (v.y >> (2 * j)) & 0x03030303u;
    } else if (T == 11) {  // Q3_K: value = 2 low bits + 4 * high bit - 4
        uint e = sub * 8, n = e >> 7, r = e & 127, j = r >> 5, i0 = r & 31, is = n * 8 + j * 2 + (i0 >> 4), sc = b + 96;
        uint low = is < 8 ? u8_(w, sc + is) & 15 : u8_(w, sc + is - 8) >> 4;
        uint hs = (u8_(w, sc + 8 + (is & 3)) >> (2 * (is >> 2))) & 3, hb = 4 * n + j;
        d = ldh(w, b + 108) * float(int(low | hs << 4) - 32);
        uvec2 h = ld8(w, b + i0);
        v = ld8(w, b + 32 + 32 * n + i0);
        lo = vsub4(((v.x >> (2 * j)) & 0x03030303u) | ((h.x >> hb) & 0x01010101u) << 2, 0x04040404u);
        hi = vsub4(((v.y >> (2 * j)) & 0x03030303u) | ((h.y >> hb) & 0x01010101u) << 2, 0x04040404u);
    } else if (T == 12 || T == 13) {  // Q4_K, Q5_K
        uint e = sub * 8, j = e >> 6, h = (e >> 5) & 1, r = e & 31, s, mi, bit = 2 * j + h, dm;
        // k4_dm<true>: d, dmin and the 12 scale bytes from word loads (both blocks are 4-aligned, 144 and 176 bytes)
        U32 u = U32(uint64_t(w) + b);
        uint sh = 8 * (bit & 3), sa = (u.v[1] >> sh) & 255, sc = (u.v[2] >> sh) & 255, se = (u.v[3] >> sh) & 255;
        dm = u.v[0];
        if (bit < 4) { s = sa & 63; mi = sc & 63; }
        else { s = (se & 15) | (sa >> 6) << 4; mi = (se >> 4) | (sc >> 6) << 4; }
        d = h2f(dm & 0xFFFF) * float(s); m = h2f(dm >> 16) * float(mi);
        v = ld8(w, b + (T == 12 ? 16 : 48) + 32 * j + r);
        lo = (v.x >> (4 * h)) & 0x0F0F0F0Fu; hi = (v.y >> (4 * h)) & 0x0F0F0F0Fu;
        if (T == 13) {
            uvec2 qh = ld8(w, b + 16 + r);
            lo |= ((qh.x >> bit) & 0x01010101u) << 4; hi |= ((qh.y >> bit) & 0x01010101u) << 4;
        }
    } else {  // Q6_K
        uint e = sub * 8, n = e >> 7, r = e & 127, qt = r >> 5, k = r & 31, ns = 4 * (qt >> 1), hs = 2 * qt;
        d = ldh(w, b + 208) * float(sb(u8_(w, b + 192 + 8 * n + (k >> 4) + 2 * qt)));
        uvec2 l = ld8(w, b + 64 * n + (qt & 1) * 32 + k), h = ld8(w, b + 128 + 32 * n + k);
        lo = vsub4(((l.x >> ns) & 0x0F0F0F0Fu) | ((h.x >> hs) & 0x03030303u) << 4, 0x20202020u);
        hi = vsub4(((l.y >> ns) & 0x0F0F0F0Fu) | ((h.y >> hs) & 0x03030303u) << 4, 0x20202020u);
    }
}

// IQ2_XXS (16), IQ2_XS (17), IQ2_S (22), IQ3_XXS (18), IQ3_S (21): grid bytes of sub-block l of block ib (4 in g0,
// 4 in g1), their 8 sign bits and the scale.
const bool IQG = T == 16 || T == 17 || T == 22 || T == 18 || T == 21;
// ksigns_iq2xs[i]: the 7 sign bits i and their parity as bit 7, computed instead of a table load.
uint ksign(uint i) { return i | (bitCount(i) & 1) << 7; }
void iq_grid(U8 w, uint b, uint ib, uint l, out uint g0, out uint g1, out uint sg, out float d) {
    if (T == 16 || T == 17 || T == 22) {
        uint64_t g;
        if (T == 16) {
            uint aux = ld32(w, b + 6 + 8 * ib);
            g = iq2xxs_grid[u8_(w, b + 2 + 8 * ib + l)];
            sg = ksign((aux >> (7 * l)) & 127);
            d = ldh(w, b) * (0.5 + float(aux >> 28)) * 0.25;
        } else {
            uint sc = u8_(w, b + (T == 22 ? 74 : 66) + ib);
            if (T == 17) {
                uint k = ld16(w, b + 2 + 2 * (4 * ib + l));
                g = iq2xs_grid[k & 511];
                sg = ksign(k >> 9);
            } else {
                g = iq2s_grid[u8_(w, b + 2 + 4 * ib + l) | ((u8_(w, b + 66 + ib) << (8 - 2 * l)) & 0x300)];
                sg = u8_(w, b + 34 + 4 * ib + l);
            }
            d = ldh(w, b) * (0.5 + float(l < 2 ? sc & 15 : sc >> 4)) * 0.25;
        }
        g0 = uint(g); g1 = uint(g >> 32);
    } else {
        uint qq = ld16(w, b + 2 + 8 * ib + 2 * l);
        if (T == 18) {
            uint aux = ld32(w, b + 66 + 4 * ib);
            d = ldh(w, b) * (0.5 + float(aux >> 28)) * 0.5;
            sg = ksign((aux >> (7 * l)) & 127);
            g0 = iq3xxs_grid[qq & 255]; g1 = iq3xxs_grid[qq >> 8];
        } else {
            uint hb = u8_(w, b + 66 + ib);
            sg = u8_(w, b + 74 + 4 * ib + l);
            d = ldh(w, b) * float(1 + 2 * ((u8_(w, b + 106 + ib / 2) >> (4 * (ib & 1))) & 15));
            g0 = iq3s_grid[(qq & 255) | ((hb << (8 - 2 * l)) & 256)];
            g1 = iq3s_grid[(qq >> 8) | ((hb << (7 - 2 * l)) & 256)];
        }
    }
}
// The 4 bytes of g, negated where bit j of s is set (j < 4), as apply_signs in kernels.cu: ~g + 1 per byte, one
// 32-bit add since no grid byte is 0 (no carries).
uint apply_signs(uint g, uint s) { uint n = (s * 0x00204081u) & 0x01010101u; return (g ^ (n * 0xFFu)) + n; }

// Values [8 sub, 8 sub + 8) of the quantized super-block at byte b as value i = d * q[i] - m with q[i] an int8
// (QI<T> in kernels.cu).
void qi8(U8 w, uint b, uint sub, out int q[8], out float d, out float m) {
    m = 0.0;
    uint ib = sub >> 2, l = sub & 3;
    if (WORDS) {
        uint lo, hi;
        qw_k(w, b, sub, lo, hi, d, m);
        [[unroll]] for (uint i = 0; i < 4; i++) { q[i] = sb(lo >> (8 * i)); q[i + 4] = sb(hi >> (8 * i)); }
    } else if (T == 19 || T == 29) {  // IQ1_S, IQ1_M: dl * (g +- 1/8) = (dl / 8) * (8 g +- 1)
        uint idx; float dl; bool neg;
        iq1_sub(w, b, ib, l, idx, dl, neg);
        uint64_t g = iq1s_grid[idx];
        d = dl * 0.125;
        [[unroll]] for (uint j = 0; j < 8; j++) q[j] = 8 * sb(uint(g >> (8 * j))) + (neg ? -1 : 1);
    } else if (IQG) {
        uint g0, g1, sg;
        iq_grid(w, b, ib, l, g0, g1, sg, d);
        [[unroll]] for (uint j = 0; j < 8; j++) q[j] = sgn(j < 4 ? g0 : g1, sg, j);
    } else {  // IQ4_XS (23), IQ4_NL (20, 8 blocks of 18)
        uint r = l * 8, o;
        if (T == 23) {
            uint sh = ld16(w, b + 2);
            uint ls = ((u8_(w, b + 4 + ib / 2) >> (4 * (ib & 1))) & 15) | ((sh >> (2 * ib)) & 3) << 4;
            d = ldh(w, b) * float(int(ls) - 32);
            o = b + 8 + 16 * ib + (r & 15);
        } else {
            uint p = b + 18 * ib;
            d = ldh(w, p);
            o = p + 2 + (r & 15);
        }
        uvec2 v = ld8(w, o);
        [[unroll]] for (uint i = 0; i < 8; i++) q[i] = kvalues_iq4nl[((i < 4 ? v.x : v.y) >> (8 * (i & 3) + (r < 16 ? 0 : 4))) & 15];
    }
}

// qi8 with the values packed in lo, hi (calling qw_k directly for the WORDS types trips glslc's optimizer).
void qw(U8 w, uint b, uint sub, out uint lo, out uint hi, out float d, out float m) {
    if (IQG) {
        uint g0, g1, sg;
        m = 0.0;
        iq_grid(w, b, sub >> 2, sub & 3, g0, g1, sg, d);
        lo = apply_signs(g0, sg & 15); hi = apply_signs(g1, sg >> 4);
        return;
    }
    int q[8];
    qi8(w, b, sub, q, d, m);
    lo = pack4(q[0], q[1], q[2], q[3]); hi = pack4(q[4], q[5], q[6], q[7]);
}

// Values [8 sub, 8 sub + 8) of the super-block at byte b (deq8<T> in kernels.cu; d * q - m is the same float).
void deq8(U8 w, uint b, uint sub, out float o[8]) {
    if (T == 0) {  // F32
        [[unroll]] for (uint i = 0; i < 8; i++) o[i] = F32(uint64_t(w)).v[b / 4 + 8 * sub + i];
    } else if (T == 30) {  // BF16
        [[unroll]] for (uint i = 0; i < 8; i++) o[i] = uintBitsToFloat(ld16(w, b + 16 * sub + 2 * i) << 16);
    } else {
        int q[8]; float d, m;
        qi8(w, b, sub, q, d, m);
        [[unroll]] for (uint i = 0; i < 8; i++) o[i] = d * float(q[i]) - m;
    }
}
