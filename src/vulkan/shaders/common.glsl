// Shared by every kernel; include it after the local size. Pointers arrive as buffer references in the push
// constants; the CUDA counterparts are in src/cuda/kernels.cu.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer I32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8 { uint8_t v[]; };
// Host-mapped memory that the host reads or writes while kernels run (the CPU expert mailbox).
layout(buffer_reference, std430, buffer_reference_align = 4) coherent volatile buffer MB { uint v[]; };

// a / b and sqrt(x) refined to the IEEE (CUDA) result: GLSL's are a few ulp off. `precise` keeps fma fused.
float fdiv(float a, float b) { float y = 1.0 / b, q = a * y; precise float r = fma(fma(-q, b, a), y, q); return r; }
float sqrt_(float x) { float s = sqrt(x); precise float r = fma(fma(-s, s, x), fdiv(0.5, s), s); return x > 0.0 ? r : s; }

// expf as CUDA compiles it (its PTX): x = j ln2 + r with j from a rounded-down fma, then ex2.approx(r) * 2^j.
float expf_(float x) {
    precise float t = clamp(fma(x, uintBitsToFloat(0x3bbb989du), 0.5), 0.0, 1.0), y = floor(t * 252.0);
    if (fma(t, 252.0, -y) < 0.0) y -= 1.0;  // t * 252 rounded up to an integer: fma.rm takes the floor
    precise float r = fma(x, uintBitsToFloat(0x32a57060u), fma(x, uintBitsToFloat(0x3fb8aa3bu), 126.0 - y));
    return exp2(r) * uintBitsToFloat(uint(y + 1.0) << 23);
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

// Sums and maxima over a subgroup of 32 in the butterfly order of the CUDA warp_sum / warp_max.
float warp_sum(float v) { for (uint o = 16; o > 0; o >>= 1) v += subgroupShuffleXor(v, o); return v; }
float warp_max(float v) { for (uint o = 16; o > 0; o >>= 1) v = max(v, subgroupShuffleXor(v, o)); return v; }

// Sum over the workgroup; every invocation gets the result. Call in uniform control flow.
shared float block_sum_s[32];
float block_sum(float v) {
    v = warp_sum(v);
    barrier();
    if (gl_SubgroupInvocationID == 0) block_sum_s[gl_SubgroupID] = v;
    barrier();
    return warp_sum(gl_SubgroupInvocationID < gl_NumSubgroups ? block_sum_s[gl_SubgroupInvocationID] : 0.0);
}

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

// Quantize x[0, n) (times sc * w[i] when w is not null) to int8 at q (as words, from word qo) with per-32 scales at
// d[do_]; the whole workgroup takes part. As quant_x in kernels.cu; reading x, not an output just written, needs no
// barrier.
void quant_x(F32 x, uint xo, uint n, F32 w, float sc, U32 q, uint qo, F32 d, uint do_) {
    for (uint i = gl_LocalInvocationID.x * 8; i < n; i += gl_WorkGroupSize.x * 8) {
        float v[8], mx = 0.0;
        for (uint j = 0; j < 8; j++) {
            v[j] = uint64_t(w) != 0 ? x.v[xo + i + j] * sc * w.v[i + j] : x.v[xo + i + j];
            mx = max(mx, abs(v[j]));
        }
        float dd, id;
        if (LLAMA_Q8) {  // one scale per 256 (the subgroup): -127 / the signed value of largest magnitude, lowest index
            float sv = 0.0; uint si = 1u << 30;
            for (uint j = 0; j < 8; j++) if (abs(v[j]) == mx && si > i + j) { sv = v[j]; si = i + j; }
            for (uint o = 1; o < 32; o <<= 1) {
                float ov = subgroupShuffleXor(sv, o); uint oi = subgroupShuffleXor(si, o);
                if (abs(ov) > abs(sv) || (abs(ov) == abs(sv) && oi < si)) { sv = ov; si = oi; }
            }
            id = sv != 0.0 ? fdiv(-127.0, sv) : 0.0;
            dd = sv != 0.0 ? fdiv(1.0, id) : 0.0;
        } else {
            mx = max(mx, subgroupShuffleXor(mx, 1));
            mx = max(mx, subgroupShuffleXor(mx, 2));
            dd = fdiv(mx, 127.0); id = mx > 0.0 ? fdiv(127.0, mx) : 0.0;
        }
        int r[8];
        for (uint j = 0; j < 8; j++) r[j] = int(roundEven(v[j] * id));
        q.v[qo + i / 4] = pack4(r[0], r[1], r[2], r[3]);
        q.v[qo + i / 4 + 1] = pack4(r[4], r[5], r[6], r[7]);
        if ((gl_LocalInvocationID.x & 3) == 0) d.v[do_ + i / 32] = dd;
    }
}
