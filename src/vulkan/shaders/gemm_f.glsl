// Prefill GEMM on f16 cooperative matrices, as gemm_q in kernels.cu (the same 16 x 16 x 16 f16 products summed in f32,
// in the same order, so results match bit for bit). Workgroup (x, y): output rows 64x .. 64x + 63 of tile y, whose
// positions m0 .. m0 + cnt read x row xrow[m] (m when xrow is null) and write y row m. tiles holds [w, m0 | cnt << 32]
// per tile; tiles = null is one dense matrix wd over positions 0 .. nd. act: 0 stores, 1 y = silu(y) * result, 2 adds.
// COOP 0 (gemm_f_dp, devices without cooperative matrices): the same f16 inputs, each output summed by one invocation
// with fma in k order, so results differ from CUDA in the last bits.
#if COOP
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#endif
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x = 256) in;
#include "common.glsl"
#include "deq.glsl"
layout(buffer_reference, std430, buffer_reference_align = 8) buffer U64 { uint64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer V4 { vec4 v[]; };
layout(push_constant) uniform P { U64 tiles; uint64_t wd; V4 x; I32 xrow; F32 y; uint nd; uint xs; uint ys; uint rows; uint nsb; uint act; } p;
const uint BM = 128, BK = 64, PW = (BK + 8) / 2;  // smem pitch in words (2 halves each)
#if COOP
#define ACC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define NE ri.length()
#define RI(i) uint(ri[i])
#define CI(i) uint(ci[i])
#else  // element i of an invocation's 16 x 16 tile as in mma.sync
#define NE 8
#define RI(i) (g + 8 * ((i) >> 2))
#define CI(i) (2 * t + ((i) & 1) + 8 * (((i) >> 1) & 1))
#endif
shared uint xs[BM * PW], ws[64 * PW];
uint h2(float a, float b) { return packFloat2x16(f16vec2(a, b)); }
shared float ix[512];  // the row and column index matrices

void main() {
    uint tid = gl_LocalInvocationID.x, w = gl_SubgroupID, n0 = gl_WorkGroupID.x * 64;
    uint wm = (w >> 1) * 32, wn = (w & 1) * 32, m0, cnt;
    U8 wt;
    if (uint64_t(p.tiles) != 0) {
        uint64_t t = p.tiles.v[2 * gl_WorkGroupID.y + 1];
        wt = U8(p.tiles.v[2 * gl_WorkGroupID.y]); m0 = uint(t); cnt = uint(t >> 32);
    } else {
        wt = U8(p.wd); m0 = gl_WorkGroupID.y * BM; cnt = min(BM, p.nd - m0);
    }
    for (uint i = tid; i < 256; i += 256) { ix[i] = float(i / 16); ix[256 + i] = float(i % 16); }
    uint g = gl_SubgroupInvocationID >> 2, t = gl_SubgroupInvocationID & 3;
#if COOP
    ACC acc[2][2];
    for (uint a = 0; a < 2; a++) for (uint b = 0; b < 2; b++) acc[a][b] = ACC(0.0);
#else
    float acc[2][2][8];
    [[unroll]] for (uint a = 0; a < 2; a++) for (uint b = 0; b < 2; b++) for (uint i = 0; i < 8; i++) acc[a][b][i] = 0.0;
#endif
    // X: invocation converts 4 floats at column xc of rows tid / 16 + 16 q per step (16-byte loads: strides are
    // multiples of 4). xb: their offsets in vec4, ~0 past cnt.
    uint xc = (tid & 15) * 4, xb[8];
    [[unroll]] for (uint q = 0; q < 8; q++) {
        uint r = (tid >> 4) + 16 * q;
        xb[q] = r < cnt ? ((uint64_t(p.xrow) != 0 ? uint(p.xrow.v[m0 + r]) : m0 + r) * p.xs + xc) / 4 : ~0u;
    }
    for (uint kb = 0; kb < p.nsb * 4; kb++) {
        uint sb = kb / 4, part = kb % 4;
        [[unroll]] for (uint q = 0; q < 2; q++) {
            uint r = tid / 8 + 32 * q, j = tid % 8;
            float v[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            if (n0 + r < p.rows) deq8(wt, ((n0 + r) * p.nsb + sb) * SBB, part * 8 + j, v);
            [[unroll]] for (uint e = 0; e < 4; e++) ws[r * PW + j * 4 + e] = h2(v[2 * e], v[2 * e + 1]);
        }
        [[unroll]] for (uint q = 0; q < 8; q++) {
            uint r = (tid >> 4) + 16 * q;
            vec4 v = xb[q] != ~0u ? p.x.v[xb[q] + kb * BK / 4] : vec4(0.0);
            xs[r * PW + xc / 2] = h2(v.x, v.y); xs[r * PW + xc / 2 + 1] = h2(v.z, v.w);
        }
        barrier();
#if COOP
        // Expert tiles are often part full; a subgroup whose 32 positions are all past cnt has nothing to multiply.
        if (wm < cnt)
        [[unroll]] for (uint kk = 0; kk < BK; kk += 16) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a[2];
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b[2];
            [[unroll]] for (uint i = 0; i < 2; i++) {
                coopMatLoad(a[i], xs, (wm + 16 * i) * PW + kk / 2, PW, gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(b[i], ws, (wn + 16 * i) * PW + kk / 2, PW, gl_CooperativeMatrixLayoutColumnMajor);
            }
            [[unroll]] for (uint mt = 0; mt < 2; mt++)
                [[unroll]] for (uint nt = 0; nt < 2; nt++) acc[mt][nt] = coopMatMulAdd(a[mt], b[nt], acc[mt][nt]);
        }
#else
        for (uint k = 0; k < BK / 2; k++) {
            vec2 xv[2][2], wv[2][4];
            [[unroll]] for (uint m = 0; m < 2; m++) [[unroll]] for (uint e = 0; e < 2; e++) xv[m][e] = unpackHalf2x16(xs[(wm + 16 * m + RI(4 * e)) * PW + k]);
            [[unroll]] for (uint m = 0; m < 2; m++) [[unroll]] for (uint e = 0; e < 4; e++) wv[m][e] = unpackHalf2x16(ws[(wn + 16 * m + CI(e)) * PW + k]);
            [[unroll]] for (uint mt = 0; mt < 2; mt++) [[unroll]] for (uint nt = 0; nt < 2; nt++) [[unroll]] for (uint i = 0; i < 8; i++) {
                vec2 a = xv[mt][i >> 2], b = wv[nt][i & 3];
                acc[mt][nt][i] = fma(a.y, b.y, fma(a.x, b.x, acc[mt][nt][i]));
            }
        }
#endif
        barrier();
    }
#if COOP
    // Accumulator element i of this invocation is at row ri[i], column ci[i] (the layout is the implementation's).
    ACC ri, ci;
    coopMatLoad(ri, ix, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(ci, ix, 256, 16, gl_CooperativeMatrixLayoutRowMajor);
#endif
    for (uint mt = 0; mt < 2; mt++)
        for (uint nt = 0; nt < 2; nt++)
            for (uint i = 0; i < NE; i++) {
                uint r = wm + 16 * mt + RI(i), n = n0 + wn + 16 * nt + CI(i);
                if (r >= cnt || n >= p.rows) continue;
                uint o = (m0 + r) * p.ys + n;
                float v = acc[mt][nt][i];
                p.y.v[o] = p.act == 0 ? v : p.act == 1 ? silu(p.y.v[o]) * v : p.y.v[o] + v;
            }
}
