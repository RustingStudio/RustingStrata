// Flash attention prefill, as attn_prefill_t in kernels.cu, with cooperative matrices in place of mma.sync (the same
// 16 x 16 x 16 f16 products summed in f32, in the same order) and the softmax done per lane as CUDA's fragment layout
// assigns it, so results match bit for bit. Include after defining HD. Workgroup (x, h): queries 64x .. 64x + 63 of
// head h; subgroup w owns queries 16w .. 16w + 15. Query t (position step[0] + t) attends to cached keys
// 0 .. step[0] + t. out = softmax(q K^T * scale) V * sigmoid(gate), the gate at +HD per head of qg (qg may be null).
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 128) in;
#include "common.glsl"
layout(push_constant) uniform P { F32 q; U16 kc; U16 vc; F32 qg; F32 o; I32 step; uint t; uint nh; uint nkv; float scale; } p;
const uint BQ = 64, BK = 32, PQ = HD + 8, PP = BK + 8;
#define ACC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
// The Q tile at the start, then the K and V tiles of each key block (BQ = 2 * BK rows either way).
shared float16_t kvs[BQ * PQ];
shared float ss[4 * 16 * BK];      // scores of each subgroup; at the start the row and column index matrices
shared float16_t ps[4 * 16 * PP];  // softmax weights of each subgroup
shared float rs[4 * 16];           // per row: the rescale factor, at the end the sum l

void main() {
    uint pos0 = uint(p.step.v[0]), q0 = gl_WorkGroupID.x * BQ, h = gl_WorkGroupID.y, kvh = h / (p.nh / p.nkv), kvd = p.nkv * HD;
    uint tid = gl_LocalInvocationID.x, w = gl_SubgroupID, lane = gl_SubgroupInvocationID, g = lane >> 2, t4 = lane & 3;
    float ninf = uintBitsToFloat(0xff800000u);
    for (uint i = tid; i < BQ * HD; i += 128) {
        uint r = i / HD, c = i % HD;
        float v = q0 + r < p.t ? p.q.v[(q0 + r) * p.nh * HD + h * HD + c] : 0.0;
        kvs[r * PQ + c] = uint16BitsToHalf(uint16_t(f2h(v * p.scale)));
    }
    for (uint i = tid; i < 256; i += 128) { ss[i] = float(i / 16); ss[256 + i] = float(i % 16); }
    barrier();
    // Accumulator element i of this invocation is at row ri[i], column ci[i] (the layout is the implementation's).
    ACC ri, ci;
    coopMatLoad(ri, ss, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(ci, ss, 256, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> qa[HD / 16];
    for (uint kk = 0; kk < HD / 16; kk++) coopMatLoad(qa[kk], kvs, w * 16 * PQ + kk * 16, PQ, gl_CooperativeMatrixLayoutRowMajor);
    ACC acc[HD / 16];
    for (uint d = 0; d < HD / 16; d++) acc[d] = ACC(0.0);
    uint last = pos0 + min(q0 + BQ, p.t) - 1;  // last key any query of the tile sees
    uint qp[2] = {pos0 + q0 + w * 16 + g, pos0 + q0 + w * 16 + g + 8};
    float m[2] = {ninf, ninf};
    precise float l[2] = {0.0, 0.0};
    uint sb = w * 16 * BK, pb = w * 16 * PP, rb = w * 16;
    for (uint k0 = 0; k0 <= last; k0 += BK) {
        barrier();
        for (uint i = tid; i < BK * HD; i += 128) {
            uint r = i / HD, c = i % HD, off = (k0 + r) * kvd + kvh * HD + c;
            bool in_ = k0 + r <= last;
            kvs[r * PQ + c] = uint16BitsToHalf(in_ ? p.kc.v[off] : uint16_t(0));
            kvs[(BK + r) * PQ + c] = uint16BitsToHalf(in_ ? p.vc.v[off] : uint16_t(0));
        }
        barrier();
        ACC s0 = ACC(0.0), s1 = ACC(0.0);
        for (uint kk = 0; kk < HD / 16; kk++) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b0, b1;  // K^T: dims kk.. by keys 0.. / 16..
            coopMatLoad(b0, kvs, kk * 16, PQ, gl_CooperativeMatrixLayoutColumnMajor);
            coopMatLoad(b1, kvs, 16 * PQ + kk * 16, PQ, gl_CooperativeMatrixLayoutColumnMajor);
            s0 = coopMatMulAdd(qa[kk], b0, s0);
            s1 = coopMatMulAdd(qa[kk], b1, s1);
        }
        coopMatStore(s0, ss, sb, BK, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(s1, ss, sb + 16, BK, gl_CooperativeMatrixLayoutRowMajor);
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        // CUDA's lane holds s[nt][e]: row g + 8 (e >> 1), key nt * 8 + 2 t4 + (e & 1).
        float s[4][4], mx[2] = {m[0], m[1]}, c[2];
        for (uint nt = 0; nt < 4; nt++)
            for (uint e = 0; e < 4; e++) {
                uint col = nt * 8 + 2 * t4 + (e & 1), r = e >> 1;
                float x = ss[sb + (g + 8 * r) * BK + col];
                if (k0 + col > qp[r]) x = ninf;
                s[nt][e] = x;
                mx[r] = max(mx[r], x);
            }
        for (uint r = 0; r < 2; r++) {
            mx[r] = max(mx[r], subgroupShuffleXor(mx[r], 1));
            mx[r] = max(mx[r], subgroupShuffleXor(mx[r], 2));
            c[r] = expf_(m[r] - mx[r]);  // key 0 is in the first block, so mx is finite from then on
            m[r] = mx[r];
        }
        for (uint nt = 0; nt < 4; nt++)
            for (uint e = 0; e < 4; e++) {
                uint r = e >> 1;
                float pe = expf_(s[nt][e] - m[r]);
                // l *= c fuses into the first add, as nvcc compiles it.
                l[r] = nt == 0 && (e & 1) == 0 ? fma(l[r], c[r], pe) : l[r] + pe;
                ps[pb + (g + 8 * r) * PP + nt * 8 + 2 * t4 + (e & 1)] = uint16BitsToHalf(uint16_t(f2h(pe)));
            }
        if (t4 == 0) { rs[rb + g] = c[0]; rs[rb + g + 8] = c[1]; }
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        for (uint d = 0; d < HD / 16; d++)
            for (uint i = 0; i < acc[d].length(); i++) acc[d][i] *= rs[rb + uint(ri[i])];
        for (uint kb = 0; kb < BK / 16; kb++) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> pa;
            coopMatLoad(pa, ps, pb + kb * 16, PP, gl_CooperativeMatrixLayoutRowMajor);
            for (uint d = 0; d < HD / 16; d++) {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> vb;
                coopMatLoad(vb, kvs, (BK + kb * 16) * PQ + d * 16, PQ, gl_CooperativeMatrixLayoutRowMajor);
                acc[d] = coopMatMulAdd(pa, vb, acc[d]);
            }
        }
    }
    for (uint r = 0; r < 2; r++) {
        l[r] += subgroupShuffleXor(l[r], 1);
        l[r] += subgroupShuffleXor(l[r], 2);
    }
    subgroupBarrier();
    if (t4 == 0) { rs[rb + g] = l[0]; rs[rb + g + 8] = l[1]; }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    for (uint d = 0; d < HD / 16; d++)
        for (uint i = 0; i < acc[d].length(); i++) {
            uint row = uint(ri[i]), col = d * 16 + uint(ci[i]), t = q0 + w * 16 + row;
            if (t >= p.t) continue;
            float gate = uint64_t(p.qg) != 0 ? sigmoid(p.qg.v[t * p.nh * 2 * HD + h * 2 * HD + HD + col]) : 1.0;
            p.o.v[t * p.nh * HD + h * HD + col] = fdiv(acc[d][i], rs[rb + row]) * gate;
        }
}
