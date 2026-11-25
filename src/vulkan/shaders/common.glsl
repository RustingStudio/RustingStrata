// Shared by every kernel; include it after the local size. Pointers arrive as buffer references in the push
// constants; the CUDA counterparts are in src/cuda/kernels.cu.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require

// 1 on devices whose physical subgroup is 64 lanes, split into two logical 32-lane warps. Set on every pipeline.
layout(constant_id = 14) const bool W64 = false;
// 1 on AMD devices (including wave32 RADV) so quantizer reciprocals use exact double division.
layout(constant_id = 13) const bool RCP_D = false;
#define LANE (W64 ? (gl_SubgroupInvocationID & 31u) : gl_SubgroupInvocationID)
#define WARP (W64 ? (gl_SubgroupID * (gl_SubgroupSize / 32u) + gl_SubgroupInvocationID / 32u) : gl_SubgroupID)
#define NWARPS ((gl_WorkGroupSize.x + 31u) / 32u)

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer I32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8 { uint8_t v[]; };
// Host-mapped memory that the host reads or writes while kernels run (the CPU expert mailbox).
layout(buffer_reference, std430, buffer_reference_align = 4) coherent volatile buffer MB { uint v[]; };

// a / b and sqrt(x) refined to the IEEE (CUDA) result: GLSL's are a few ulp off. `precise` keeps fma fused. 1 / b
// gets its own Newton step first: Vulkan allows it 2.5 ulp, and AMD's is rough enough that one step on the quotient
// still ends 1 ulp low.
float fdiv(float a, float b) {
    float y0 = 1.0 / b;
    precise float y = fma(fma(-b, y0, 1.0), y0, y0), q = a * y, r = fma(fma(-q, b, a), y, q);
    // An infinite operand or b = 0 makes the steps inf * 0 = NaN (sigmoid(-100): 1 / inf); a / b is exact there.
    return isinf(a) || isinf(b) || b == 0.0 ? a / b : r;
}
// Quantizer reciprocals on wave64 and AMD wave32: a correctly rounded double division keeps values on a
// roundEven() tie from moving by one int8 step. Non-AMD wave32 keeps the fdiv path that its kernels were tuned with.
float rcp_d(float a, float b) { return (W64 || RCP_D) ? float(double(a) / double(b)) : fdiv(a, b); }
float sqrt_(float x) { float s = sqrt(x); precise float r = fma(fma(-s, s, x), fdiv(0.5, s), s); return x > 0.0 && !isinf(x) ? r : s; }

// expf as CUDA compiles it (its PTX): x = j ln2 + r with j from a rounded-down fma, then ex2.approx(r) * 2^j.
float expf_(float x) {
    precise float t = clamp(fma(x, uintBitsToFloat(0x3bbb989du), 0.5), 0.0, 1.0), y = floor(t * 252.0);
    if (fma(t, 252.0, -y) < 0.0) y -= 1.0;  // t * 252 rounded up to an integer: fma.rm takes the floor
    precise float r = fma(x, uintBitsToFloat(0x32a57060u), fma(x, uintBitsToFloat(0x3fb8aa3bu), 126.0 - y));
    return exp2(r) * uintBitsToFloat(uint(y + 1.0) << 23);
}
// x rounded to f16 (nearest even) in integer ops: the NVIDIA driver drops a float -> half -> float round trip.
// No overflow to inf (callers' values are far below 65504).
float rh(float x) {
    uint u = floatBitsToUint(x), a = u & 0x7fffffffu;
    if (a < 0x38800000u) return roundEven(x * 16777216.0) * (1.0 / 16777216.0);  // f16 subnormal: steps of 2^-24
    return uintBitsToFloat((u & 0x80000000u) | ((a + 0xfffu + ((a >> 13) & 1u)) & 0xffffe000u));
}
float silu(float x) { return fdiv(x, 1.0 + expf_(-x)); }
float sigmoid(float x) { return fdiv(1.0, 1.0 + expf_(-x)); }
// log1pf as CUDA compiles it (its PTX): 1 + a rounded toward zero picks the scale 2^e, then a polynomial in the
// scaled argument.
float log1p_(float a) {
    precise float m = a + 1.0, bb = m - a, err = (a - (m - bb)) + (1.0 - bb);
    if (err != 0.0 && (err < 0.0) != (m < 0.0)) m = uintBitsToFloat(floatBitsToUint(m) - 1u);  // round toward zero
    int e = (floatBitsToInt(m) - 0x3f400000) & int(0xff800000u);
    precise float s = fma(0.25, intBitsToFloat(0x40800000 - e), -1.0) + intBitsToFloat(floatBitsToInt(a) - e);
    precise float q = fma(uintBitsToFloat(0xbd39bf78u), s, uintBitsToFloat(0x3dd80012u));
    q = fma(q, s, uintBitsToFloat(0xbe0778e0u));
    q = fma(q, s, uintBitsToFloat(0x3e146475u));
    q = fma(q, s, uintBitsToFloat(0xbe2a68ddu));
    q = fma(q, s, uintBitsToFloat(0x3e4caf9eu));
    q = fma(q, s, uintBitsToFloat(0xbe800042u));
    q = fma(q, s, uintBitsToFloat(0x3eaaaae6u));
    q = fma(q, s, -0.5);
    precise float r = fma(float(e) * uintBitsToFloat(0x34000000u), uintBitsToFloat(0x3f317218u), fma(s * q, s, s));
    if (floatBitsToUint(a) >= 0x7f800000u) {  // negative, inf or NaN
        if (floatBitsToInt(a) >= int(0xbf800001u)) r = fma(a, uintBitsToFloat(0x7f800000u), uintBitsToFloat(0x7f800000u));
        if (a == 0.0) r = -0.0;
    }
    return r;
}
float softplus(float x) { return x > 20.0 ? x : log1p_(expf_(x)); }

// Reductions over each logical 32-lane warp, including either half of a 64-lane subgroup.
uint warp_max_uint(uint v) { for (uint o = 16; o > 0; o >>= 1) v = max(v, subgroupShuffleXor(v, o)); return v; }
uint warp_min_uint(uint v) { for (uint o = 16; o > 0; o >>= 1) v = min(v, subgroupShuffleXor(v, o)); return v; }
uint warp_scan(uint v) {
    uint lane = gl_SubgroupInvocationID & 31u, base = gl_SubgroupInvocationID & ~31u;
    for (uint o = 1; o < 32; o <<= 1) { uint x = subgroupShuffle(v, base + (lane >= o ? lane - o : lane)); if (lane >= o) v += x; }
    return v;
}

// Sums, maxima and minima over a subgroup of 32 in the butterfly order of CUDA warp_sum / warp_max.
float warp_sum(float v) { for (uint o = 16; o > 0; o >>= 1) v += subgroupShuffleXor(v, o); return v; }
float warp_max(float v) { for (uint o = 16; o > 0; o >>= 1) v = max(v, subgroupShuffleXor(v, o)); return v; }
float warp_min(float v) { for (uint o = 16; o > 0; o >>= 1) v = min(v, subgroupShuffleXor(v, o)); return v; }

// Sum over the workgroup; every invocation gets the result. Call in uniform control flow.
shared float block_sum_s[32];
float block_sum(float v) {
    v = warp_sum(v);
    barrier();
    if (LANE == 0) block_sum_s[WARP] = v;
    barrier();
    return warp_sum(LANE < NWARPS ? block_sum_s[LANE] : 0.0);
}

// QSA mask (see qsa_select): whether query row t (mw words a row) sees key position `key`.
bool qsa_bit(U32 qm, uint mw, uint t, uint key) { return ((qm.v[t * mw + (key >> 7)] >> ((key >> 2) & 31)) & 1) != 0; }

// f32 to f16 bits, rounded to nearest even (cvt.rn.f16.f32).
uint f2h(float x) {
    uint b = floatBitsToUint(x), s = b >> 16 & 0x8000u, a = b & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7e00u;
    if (a >= 0x477ff000u) return s | 0x7c00u;  // 65520 and up round to infinity
    if (a < 0x38800000u) return s | uint(roundEven(uintBitsToFloat(a) * 16777216.0));  // subnormal: units of 2^-24
    uint r = a - 0x38000000u;
    return s | (r + 0xfffu + (r >> 13 & 1u)) >> 13;
}

uint pack4(int a, int b, int c, int d) { return uint(a & 255) | uint(b & 255) << 8 | uint(c & 255) << 16 | uint(d & 255) << 24; }

// llama.cpp q8_K numerics (--llama-numerics) for kernels that quantize x.
layout(constant_id = 0) const bool LLAMA_Q8 = false;

// KV cache rows (one per position and KV head) hold hd f16 values (KV_MODE 0), hd int8 values followed by hd / 32 f16 scales
// (KV_MODE 1, `--kv q8`), hd/2 packed uint4 nibbles + hd/32 f16 scales (KV_MODE 2, `--kv q4`), or affine 2-bit codes
// (hd/4 B) + hd/32 f16 scales + hd/32 f16 mins (KV_MODE 3, `--kv q2`). Offsets below count values.
layout(constant_id = 15) const uint KV_MODE = 0u;   // K cache: 0 f16, 1 q8, 2 q4, 3 q2, 4 turbo4, 5 turbo3, 6 turbo2
layout(constant_id = 16) const uint VKV_MODE = 0u;  // V cache: same encoding
#include "tq.glsl"
// Store x at value offset e; with quantized KV the 32 lanes of the subgroup hold one 32-value block.
void kv_put(U16 c, uint e, uint hd, float x, uint mode) {
    if (mode == 0u) { c.v[e] = uint16_t(f2h(x)); return; }
    if (mode == 3u) {
        uint16_t mnh = uint16_t(f2h(warp_min(x)));
        float mn = float(uint16BitsToHalf(mnh)), mx = warp_max(x);
        float d_raw = rcp_d(max(0.0, mx - mn), 3.0);
        uint16_t dh = uint16_t(f2h(d_raw));
        float d = float(uint16BitsToHalf(dh)), y = d == 0.0 ? 0.0 : rcp_d(x - mn, d), qf = trunc(y);
        if (abs(y - qf) >= 0.5) qf += sign(y);
        uint uq = uint(clamp(int(qf), 0, 3));
        uint packed = uq
            | (subgroupShuffleXor(uq, 1u) << 2)
            | (subgroupShuffleXor(uq, 2u) << 4)
            | (subgroupShuffleXor(uq, 3u) << 6);
        uint w = e % hd, r = e / hd * (3u * hd / 8u);
        if ((w & 3u) == 0u) U8(uint64_t(c)).v[r + w / 4u] = uint8_t(packed);
        if ((w & 31u) == 0u) {
            c.v[(r + hd / 4u) / 2u + w / 32u] = dh;
            c.v[(r + hd / 4u + hd / 16u) / 2u + w / 32u] = mnh;
        }
        return;
    }
    if (mode == 2u) {
        float d = rcp_d(warp_max(abs(x)), 7.0), y = d == 0.0 ? 0.0 : rcp_d(x, d), qf = trunc(y);
        if (abs(y - qf) >= 0.5) qf += sign(y);  // roundf: halves away from zero
        uint nib = uint(clamp(int(qf), -7, 7) + 8);
        uint pair = nib | (subgroupShuffleXor(nib, 1u) << 4);
        uint w = e % hd, r = e / hd * (hd / 2 + hd / 16);
        if ((w & 1u) == 0u) U8(uint64_t(c)).v[r + w / 2] = uint8_t(pair);
        if ((w & 31u) == 0u) c.v[(r + hd / 2) / 2 + w / 32] = uint16_t(f2h(d));
        return;
    }
    float d = rcp_d(warp_max(abs(x)), 127.0), y = d == 0.0 ? 0.0 : rcp_d(x, d), q = trunc(y);
    if (abs(y - q) >= 0.5) q += sign(y);  // roundf: halves away from zero
    uint w = e % hd, r = e / hd * (hd + hd / 16);
    U8(uint64_t(c)).v[r + w] = uint8_t(int(q) & 255);
    if (w % 32 == 0) c.v[(r + hd) / 2 + w / 32] = uint16_t(f2h(d));
}
// The cache value at offset off. q8/q4/q2: dequantized and rounded once to f16, as ldkv in kernels.cu.
float ldkv(U16 c, uint off, uint hd, uint mode) {
    if (mode == 0u) return float(uint16BitsToHalf(c.v[off]));
    if (mode == 3u) {
        uint w = off % hd, r = off / hd * (3u * hd / 8u), b = uint(U8(uint64_t(c)).v[r + w / 4u]);
        float d = unpackHalf2x16(uint(c.v[(r + hd / 4u) / 2u + w / 32u])).x;
        float mn = unpackHalf2x16(uint(c.v[(r + hd / 4u + hd / 16u) / 2u + w / 32u])).x;
        uint q = (b >> (2u * (w & 3u))) & 3u;
        return rh(fma(float(q), d, mn));
    }
    if (mode == 2u) {
        uint w = off % hd, r = off / hd * (hd / 2 + hd / 16), b = uint(U8(uint64_t(c)).v[r + w / 2]);
        float d = unpackHalf2x16(uint(c.v[(r + hd / 2) / 2 + w / 32])).x;
        int q = int((w & 1u) == 0u ? (b & 15u) : (b >> 4)) - 8;
        return rh(float(q) * d);
    }
    uint w = off % hd, r = off / hd * (hd + hd / 16);
    float d = unpackHalf2x16(uint(c.v[(r + hd) / 2 + w / 32])).x;
    return rh(float(int(uint(U8(uint64_t(c)).v[r + w]) << 24) >> 24) * d);
}
// q8/q4/q2 values off and off + 1 (off even) as one f16 pair word, from one load and one scale(+min).
uint ldkv2(U16 c, uint off, uint hd, uint mode) {
    if (mode == 3u) {
        uint w = off % hd, r = off / hd * (3u * hd / 8u), b = uint(U8(uint64_t(c)).v[r + w / 4u]);
        uint s = (b >> (2u * (w & 2u))) & 15u;
        float d = unpackHalf2x16(uint(c.v[(r + hd / 4u) / 2u + w / 32u])).x;
        float mn = unpackHalf2x16(uint(c.v[(r + hd / 4u + hd / 16u) / 2u + w / 32u])).x;
        return packHalf2x16(vec2(rh(fma(float(s & 3u), d, mn)), rh(fma(float(s >> 2), d, mn))));
    }
    if (mode == 2u) {
        uint w = off % hd, r = off / hd * (hd / 2 + hd / 16), b = uint(U8(uint64_t(c)).v[r + w / 2]);
        float d = unpackHalf2x16(uint(c.v[(r + hd / 2) / 2 + w / 32])).x;
        return packHalf2x16(vec2(rh(float(int(b & 15u) - 8) * d), rh(float(int(b >> 4) - 8) * d)));
    }
    uint w = off % hd, r = off / hd * (hd + hd / 16), b = uint(c.v[(r + w) / 2]);
    float d = unpackHalf2x16(uint(c.v[(r + hd) / 2 + w / 32])).x;
    return packHalf2x16(vec2(rh(float(int(b << 24) >> 24) * d), rh(float(int(b << 16) >> 24) * d)));
}
// q8/q4/q2 values off .. off + 3 (off % 4 == 0) as two f16 pair words using packed FP16 arithmetic.
uvec2 ldkv4(U16 c, uint off, uint hd, uint mode) {
    if (mode == 3u) {
        uint w = off % hd, r = off / hd * (3u * hd / 8u), b = uint(U8(uint64_t(c)).v[r + w / 4u]);
        precise f16vec2 d2 = f16vec2(uint16BitsToHalf(c.v[(r + hd / 4u) / 2u + w / 32u]));
        precise f16vec2 mn2 = f16vec2(uint16BitsToHalf(c.v[(r + hd / 4u + hd / 16u) / 2u + w / 32u]));
        precise f16vec2 m = f16vec2(1024.0hf);
        precise f16vec2 a = fma(unpackFloat2x16(((b & 3u) | ((b & 12u) << 14)) | 0x64006400u) - m, d2, mn2);
        precise f16vec2 z = fma(unpackFloat2x16((((b >> 4) & 3u) | ((b & 0xc0u) << 10)) | 0x64006400u) - m, d2, mn2);
        return uvec2(packFloat2x16(a), packFloat2x16(z));
    }
    if (mode == 2u) {
        uint w = off % hd, r = off / hd * (hd / 2 + hd / 16), b = uint(c.v[(r + w / 2) / 2]);
        float16_t d = uint16BitsToHalf(c.v[(r + hd / 2) / 2 + w / 32]);
        precise f16vec2 m = f16vec2(1032.0hf);
        precise f16vec2 a = (unpackFloat2x16(((b & 0x000Fu) | ((b & 0x00F0u) << 12)) | 0x64006400u) - m) * d;
        precise f16vec2 z = (unpackFloat2x16((((b >> 8) & 0x000Fu) | ((b & 0xF000u) << 4)) | 0x64006400u) - m) * d;
        return uvec2(packFloat2x16(a), packFloat2x16(z));
    }
    if (hd % 64 != 0) return uvec2(ldkv2(c, off, hd, mode), ldkv2(c, off + 2, hd, mode));
    uint w = off % hd, r = off / hd * (hd + hd / 16), b = U32(uint64_t(c)).v[(r + w) / 4];
    // q + 128 as the low mantissa bits of 1024 gives the f16 1152 + q, so q comes out exact without int -> float
    // converts. The f16 product q * d is rounded once, as rh(float(q) * d).
    uint x = b ^ 0x80808080u, lo = (x & 0xffu) | (x & 0xff00u) << 8, hi = (x >> 16 & 0xffu) | (x >> 8 & 0xff0000u);
    float16_t d = uint16BitsToHalf(c.v[(r + hd) / 2 + w / 32]);
    precise f16vec2 m = f16vec2(1152.0hf), a = (unpackFloat2x16(lo | 0x64006400u) - m) * d, z = (unpackFloat2x16(hi | 0x64006400u) - m) * d;
    return uvec2(packFloat2x16(a), packFloat2x16(z));
}

// sin and cos of theta = pos * ts^i (ts = base^(-2 / n_rot) from the host) in f64, as the CPU and CUDA take them
// (GLSL has no f64 pow, sin or cos).
void rope_cs(double pos, double ts, uint i, out float sn, out float cs) {
    double f = 1.0LF, b = ts;
    for (uint e = i; e > 0; e >>= 1, b *= b) if ((e & 1) != 0) f *= b;
    double theta = pos * f, k = roundEven(theta * 0.63661977236758134LF);  // 2 / pi
    double y = fma(-k, 1.5707963267948966LF, theta);
    y = fma(-k, 6.123233995736766e-17LF, y);
    double y2 = y * y;
    double s = 1.0LF, c = 1.0LF;  // Taylor to y^17 and y^16 on |y| <= pi / 4
    for (int j = 8; j >= 1; j--) { s = 1.0LF - y2 / double(2 * j * (2 * j + 1)) * s; c = 1.0LF - y2 / double((2 * j - 1) * 2 * j) * c; }
    s *= y;
    int q = int(k) & 3;
    sn = float(q == 0 ? s : q == 1 ? c : q == 2 ? -s : -c); cs = float(q == 0 ? c : q == 1 ? -s : q == 2 ? -c : s);
}

// Quantize x[0, n) (times sc * w[i] when w is not null, or silu(x) * u when u is not null) to int8 at q (as words, from word qo) with per-32 scales at
// d[do_]; the whole workgroup takes part. As quant_x in kernels.cu; reading x, not an output just written, needs no
// barrier. With o not null the values are also stored at o[oo, oo + n).
void quant_x(F32 x, uint xo, uint n, F32 w, float sc, F32 u, U32 q, uint qo, F32 d, uint do_, F32 o, uint oo) {
    for (uint i = gl_LocalInvocationID.x * 8; i < n; i += gl_WorkGroupSize.x * 8) {
        float v[8], mx = 0.0;
        [[unroll]] for (uint j = 0; j < 8; j++) {
            v[j] = uint64_t(w) != 0 ? x.v[xo + i + j] * sc * w.v[i + j] : x.v[xo + i + j];
            if (uint64_t(u) != 0) v[j] = silu(v[j]) * u.v[xo + i + j];
            mx = max(mx, abs(v[j]));
        }
        if (uint64_t(o) != 0) [[unroll]] for (uint j = 0; j < 8; j++) o.v[oo + i + j] = v[j];
        float dd, id;
        if (LLAMA_Q8) {  // one scale per 256 (the subgroup): -127 / the signed value of largest magnitude, lowest index
            float sv = 0.0; uint si = 1u << 30;
            [[unroll]] for (uint j = 0; j < 8; j++) if (abs(v[j]) == mx && si > i + j) { sv = v[j]; si = i + j; }
            [[unroll]] for (uint o = 1; o < 32; o <<= 1) {
                float ov = subgroupShuffleXor(sv, o); uint oi = subgroupShuffleXor(si, o);
                if (abs(ov) > abs(sv) || (abs(ov) == abs(sv) && oi < si)) { sv = ov; si = oi; }
            }
            id = sv != 0.0 ? rcp_d(-127.0, sv) : 0.0;
            dd = sv != 0.0 ? rcp_d(1.0, id) : 0.0;
        } else {
            mx = max(mx, subgroupShuffleXor(mx, 1));
            mx = max(mx, subgroupShuffleXor(mx, 2));
            dd = rcp_d(mx, 127.0); id = mx > 0.0 ? rcp_d(127.0, mx) : 0.0;
        }
        int r[8];
        [[unroll]] for (uint j = 0; j < 8; j++) r[j] = int(roundEven(v[j] * id));
        q.v[qo + i / 4] = pack4(r[0], r[1], r[2], r[3]);
        q.v[qo + i / 4 + 1] = pack4(r[4], r[5], r[6], r[7]);
        if ((gl_LocalInvocationID.x & 3) == 0) d.v[do_ + i / 32] = dd;
    }
}

// Mode-dispatched entry points. The *_k / *_v wrappers keep the non-turbo half bit-identical in mixed layouts.
void kv_put_ex(U16 c, uint e, uint hd, float x, float norm, bool valid, uint mode, uint bits) {
    if (mode >= 4u) { tq_put(c, e, hd, x, norm, valid, bits, ((e / hd) & 7u) * 8u + ((e % hd) >> 7u)); return; }
    if (!valid) return;
    kv_put(c, e, hd, x, mode);
}
void kv_put_k(U16 c, uint e, uint hd, float x) { kv_put_ex(c, e, hd, x, 1.0f, true, KV_MODE, TQ_KBITS); }
void kv_put_v(U16 c, uint e, uint hd, float x) { kv_put_ex(c, e, hd, x, 1.0f, true, VKV_MODE, TQ_VBITS); }
void kv_put_tq_which(U16 c, uint e, uint hd, float x, float norm, bool valid, uint which) {
    uint m = which == 0u ? KV_MODE : VKV_MODE, b = which == 0u ? TQ_KBITS : TQ_VBITS;
    kv_put_ex(c, e, hd, x, norm, valid, m, b);
}
float ldkv_k(U16 c, uint off, uint hd) { return KV_MODE >= 4u ? tq_ldkv(c, off, hd, TQ_KBITS) : ldkv(c, off, hd, KV_MODE); }
float ldkv_v(U16 c, uint off, uint hd) { return VKV_MODE >= 4u ? tq_ldkv(c, off, hd, TQ_VBITS) : ldkv(c, off, hd, VKV_MODE); }
uint ldkv2_k(U16 c, uint off, uint hd) { return KV_MODE >= 4u ? tq_ldkv2(c, off, hd, TQ_KBITS) : ldkv2(c, off, hd, KV_MODE); }
uint ldkv2_v(U16 c, uint off, uint hd) { return VKV_MODE >= 4u ? tq_ldkv2(c, off, hd, TQ_VBITS) : ldkv2(c, off, hd, VKV_MODE); }
uvec2 ldkv4_k(U16 c, uint off, uint hd) { return KV_MODE >= 4u ? tq_ldkv4(c, off, hd, TQ_KBITS) : ldkv4(c, off, hd, KV_MODE); }
uvec2 ldkv4_v(U16 c, uint off, uint hd) { return VKV_MODE >= 4u ? tq_ldkv4(c, off, hd, TQ_VBITS) : ldkv4(c, off, hd, VKV_MODE); }
