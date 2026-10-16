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

// a / b and sqrt(x) refined to the IEEE (CUDA) result: GLSL's are a few ulp off. `precise` keeps fma fused.
float fdiv(float a, float b) { float y = 1.0 / b, q = a * y; precise float r = fma(fma(-q, b, a), y, q); return r; }
float sqrt_(float x) { float s = sqrt(x); precise float r = fma(fma(-s, s, x), fdiv(0.5, s), s); return x > 0.0 ? r : s; }

float silu(float x) { return x / (1.0 + exp(-x)); }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
// log1p without the loss of log(1 + x) for small x (GLSL has no log1p).
float log1p_(float x) { float u = 1.0 + x; return u == 1.0 ? x : log(u) * x / (u - 1.0); }
float softplus(float x) { return x > 20.0 ? x : log1p_(exp(x)); }

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
