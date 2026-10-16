// Dequantization of weight type T (the ggml type id, specialization constant 1) with SBB bytes per 256-element
// super-block (constant 2): ports of deq8<T> in kernels.cu, which port src/quant/mod.rs (from llama.cpp
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

void scale_min_k4(uint j, U8 w, uint q, out uint s, out uint m) {
    if (j < 4) { s = u8_(w, q + j) & 63; m = u8_(w, q + j + 4) & 63; }
    else { s = (u8_(w, q + j + 4) & 15) | (u8_(w, q + j - 4) >> 6) << 4; m = (u8_(w, q + j + 4) >> 4) | (u8_(w, q + j) >> 6) << 4; }
}

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

// Grid bytes g (8 of a u64, or 4 + 4 of two u32) times db, negated where the sign bits sg say.
float sgn(uint sg, uint j) { return ((sg >> j) & 1) != 0 ? -1.0 : 1.0; }

// Values [8 sub, 8 sub + 8) of the super-block at byte b.
void deq8(U8 w, uint b, uint sub, out float o[8]) {
    if (T == 0) {  // F32
        for (uint i = 0; i < 8; i++) o[i] = F32(uint64_t(w)).v[b / 4 + 8 * sub + i];
    } else if (T == 30) {  // BF16
        for (uint i = 0; i < 8; i++) o[i] = uintBitsToFloat(ld16(w, b + 16 * sub + 2 * i) << 16);
    } else if (T == 8) {  // Q8_0, 8 blocks of 34
        uint p = b + 34 * (sub >> 2), q = p + 2 + (sub & 3) * 8;
        float d = ldh(w, p);
        for (uint i = 0; i < 8; i++) o[i] = float(sb(u8_(w, q + i))) * d;
    } else if (T == 10) {  // Q2_K
        uint e = sub * 8, n = e >> 7, r = e & 127, j = r >> 5, r2 = r & 31;
        uint s = u8_(w, b + n * 8 + j * 2 + (r2 >> 4));
        float dl = ldh(w, b + 80) * float(s & 15), ml = ldh(w, b + 82) * float(s >> 4);
        uint q = b + 16 + 32 * n + r2;
        for (uint i = 0; i < 8; i++) o[i] = dl * float((u8_(w, q + i) >> (2 * j)) & 3) - ml;
    } else if (T == 11) {  // Q3_K
        uint e = sub * 8, n = e >> 7, r = e & 127, j = r >> 5, i0 = r & 31, is = n * 8 + j * 2 + (i0 >> 4), sc = b + 96;
        uint low = is < 8 ? u8_(w, sc + is) & 15 : u8_(w, sc + is - 8) >> 4;
        uint hi = (u8_(w, sc + 8 + (is & 3)) >> (2 * (is >> 2))) & 3;
        float dl = ldh(w, b + 108) * float(int(low | hi << 4) - 32);
        uint m = 1u << (n * 4 + j);
        for (uint i = 0; i < 8; i++) {
            uint qi = i0 + i;
            o[i] = dl * float(int((u8_(w, b + 32 + 32 * n + qi) >> (2 * j)) & 3) - ((u8_(w, b + qi) & m) != 0 ? 0 : 4));
        }
    } else if (T == 12 || T == 13) {  // Q4_K, Q5_K
        uint e = sub * 8, j = e >> 6, hi = (e >> 5) & 1, l = e & 31, s, m;
        scale_min_k4(2 * j + hi, w, b + 4, s, m);
        float d = ldh(w, b) * float(s), mn = ldh(w, b + 2) * float(m);
        if (T == 12) {
            uint q = b + 16 + 32 * j + l;
            for (uint i = 0; i < 8; i++) o[i] = d * float(hi != 0 ? u8_(w, q + i) >> 4 : u8_(w, q + i) & 15) - mn;
        } else {
            uint qh = b + 16 + l, q = b + 48 + 32 * j + l, bit = 1u << (2 * j + hi);
            for (uint i = 0; i < 8; i++)
                o[i] = d * (float(hi != 0 ? u8_(w, q + i) >> 4 : u8_(w, q + i) & 15) + ((u8_(w, qh + i) & bit) != 0 ? 16.0 : 0.0)) - mn;
        }
    } else if (T == 14) {  // Q6_K
        uint e = sub * 8, n = e >> 7, r = e & 127, qt = r >> 5, l = r & 31, ql = b + 64 * n, qh = b + 128 + 32 * n;
        float d = ldh(w, b + 208) * float(sb(u8_(w, b + 192 + 8 * n + (l >> 4) + 2 * qt)));
        for (uint i = 0; i < 8; i++) {
            uint li = l + i, h = u8_(w, qh + li), q;
            if (qt == 0) q = (u8_(w, ql + li) & 15) | (h & 3) << 4;
            else if (qt == 1) q = (u8_(w, ql + li + 32) & 15) | ((h >> 2) & 3) << 4;
            else if (qt == 2) q = (u8_(w, ql + li) >> 4) | ((h >> 4) & 3) << 4;
            else q = (u8_(w, ql + li + 32) >> 4) | ((h >> 6) & 3) << 4;
            o[i] = d * float(int(q) - 32);
        }
    } else if (T == 19 || T == 29) {  // IQ1_S, IQ1_M: grid bytes -1, 0, 1 and a sub-block offset of +-1/8
        uint idx; float dl; bool neg;
        iq1_sub(w, b, sub >> 2, sub & 3, idx, dl, neg);
        uint64_t g = iq1s_grid[idx];
        for (uint j = 0; j < 8; j++) o[j] = dl * (float(sb(uint(g >> (8 * j)))) + (neg ? -0.125 : 0.125));
    } else if (T == 16 || T == 17 || T == 22) {  // IQ2_XXS, IQ2_XS, IQ2_S
        uint ib = sub >> 2, l = sub & 3, sg;
        uint64_t g;
        float db;
        if (T == 16) {
            uint aux = ld16(w, b + 6 + 8 * ib) | ld16(w, b + 8 + 8 * ib) << 16;
            g = iq2xxs_grid[u8_(w, b + 2 + 8 * ib + l)];
            sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
            db = ldh(w, b) * (0.5 + float(aux >> 28)) * 0.25;
        } else {
            uint sc = u8_(w, b + 66 + (T == 22 ? 8 : 0) + ib);
            if (T == 17) {
                uint q = ld16(w, b + 2 + 2 * (4 * ib + l));
                g = iq2xs_grid[q & 511];
                sg = ksigns_iq2xs[q >> 9];
            } else {
                g = iq2s_grid[u8_(w, b + 2 + 4 * ib + l) | ((u8_(w, b + 66 + ib) << (8 - 2 * l)) & 0x300)];
                sg = u8_(w, b + 34 + 4 * ib + l);
            }
            db = ldh(w, b) * (0.5 + float(l < 2 ? sc & 15 : sc >> 4)) * 0.25;
        }
        for (uint j = 0; j < 8; j++) o[j] = db * float(uint(g >> (8 * j)) & 0xFF) * sgn(sg, j);
    } else if (T == 18 || T == 21) {  // IQ3_XXS, IQ3_S
        uint ib = sub >> 2, l = sub & 3, sg, g1, g2;
        float db;
        if (T == 18) {
            uint aux = ld32(w, b + 66 + 4 * ib);
            db = ldh(w, b) * (0.5 + float(aux >> 28)) * 0.5;
            sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
            g1 = iq3xxs_grid[u8_(w, b + 2 + 8 * ib + 2 * l)]; g2 = iq3xxs_grid[u8_(w, b + 3 + 8 * ib + 2 * l)];
        } else {
            uint hb = u8_(w, b + 66 + ib);
            sg = u8_(w, b + 74 + 4 * ib + l);
            db = ldh(w, b) * float(1 + 2 * ((u8_(w, b + 106 + ib / 2) >> (4 * (ib & 1))) & 15));
            g1 = iq3s_grid[u8_(w, b + 2 + 8 * ib + 2 * l) | ((hb << (8 - 2 * l)) & 256)];
            g2 = iq3s_grid[u8_(w, b + 3 + 8 * ib + 2 * l) | ((hb << (7 - 2 * l)) & 256)];
        }
        for (uint j = 0; j < 4; j++) {
            o[j] = db * float((g1 >> (8 * j)) & 0xFF) * sgn(sg, j);
            o[j + 4] = db * float((g2 >> (8 * j)) & 0xFF) * sgn(sg, j + 4);
        }
    } else {  // IQ4_XS (23), IQ4_NL (20, 8 blocks of 18)
        uint r = (sub & 3) * 8, q;
        float dl;
        if (T == 23) {
            uint ib = sub >> 2, sh = ld16(w, b + 2);
            uint ls = ((u8_(w, b + 4 + ib / 2) >> (4 * (ib & 1))) & 15) | ((sh >> (2 * ib)) & 3) << 4;
            dl = ldh(w, b) * float(int(ls) - 32);
            q = b + 8 + 16 * ib + (r & 15);
        } else {
            uint p = b + 18 * (sub >> 2);
            dl = ldh(w, p);
            q = p + 2 + (r & 15);
        }
        for (uint i = 0; i < 8; i++) o[i] = dl * float(kvalues_iq4nl[r < 16 ? u8_(w, q + i) & 15 : u8_(w, q + i) >> 4]);
    }
}
