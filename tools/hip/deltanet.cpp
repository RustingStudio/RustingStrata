// deltanet_seq (the gated delta-rule scan that Qwen3.6/3.8's linear-attention layers run in prefill) in isolation.
// Answering "can this kernel be made faster without changing the algorithm", with numbers instead of guesses: a
// model-level prefill A/B cannot resolve a 5% change (two runs of the same build differ by more), so the variants are
// timed here. Fixed shape of the 35B model: dk 128, nk 16, nv 32, dv 128, T 4096, i.e. 128 blocks of 256 threads.
//
//   hipcc --offload-arch=gfx1201 -O3 deltanet.cpp -o deltanet && ./deltanet
//
// Measured on RX 9070 XT (3.11 ms for one layer's scan over 4096 tokens, ~1750 cycles a token; the ratios move by
// +-4% between runs, so treat anything under 5% as noise):
//
//   base                    3.11 ms   1.00x   deltanet_seq as committed
//   dn16 (running pointers) 3.26 ms   0.93x   the loop's `tok * stride` products hoisted into pointer increments
//   dn16s (+ 2 accumulators) 3.22 ms   0.96x   one chain of 16 dependent adds split in two
//   dn32 (32 floats a lane) 3.84 ms   0.81x   half the blocks (one wave on 60 CUs), twice the work a lane
//   dn32s                   3.73 ms   0.83x
//   no expf(g) per token    2.76 ms   1.12x   the per-token decay, which only half the calls need as an exp
//   no warp reductions      2.33 ms   1.33x   both dependent shuffle rounds removed (wrong results, timing only)
//   same q/k/v every step   3.13 ms   0.99x   per-token reads held in L1: the strided, 128-block-redundant reads
//                                            cost nothing, so this kernel is latency, not bandwidth
//
// So: no rearrangement of the scan wins (several lose), the reads are free, and the 33% inside reach is the two
// dependent shuffle rounds plus the per-token exp -- both of which only disappear if the per-token recurrence
// becomes a chunked one (the WY/UT-transform form: a triangular inverse and matmuls over a chunk, and with f16
// tensor cores it would lose state precision). 12% of a 4096-token prefill is in this kernel, so the whole
// theoretical prize is ~4% prefill for that rewrite. Left alone deliberately.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
typedef unsigned short u16;
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
// kernels.cu's HIP shim routes the mask-less form here; the HIP builtin itself wants a 64-bit mask.
template <typename T> __device__ __forceinline__ T shflx(T v, int off) { return __shfl_xor(v, off, 32); }
__global__ void dn_base(const float* y, float* S, const float* beta, const float* g, float* o,
                                        int nk, int nv, int ilv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap,
                                        const float* dtb, const float* ga) {
    int L = dk >> 4, R = 32 / L, lane = threadIdx.x & 31, s = lane % L, vh = blockIdx.x, kh = ilv ? vh / (nv / nk) : vh % nk;
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
        if (dtb) { ng = softplus(ng + dtb[vh]) * ga[vh]; nb = sigmoid(nb); }
    };
    ld(0);
    for (int tok = 0; tok < T; tok++) {
        float kk[16], qv[16], decay = expf(ng), bt = nb, vt = nvv, sk = 0.0f, oq = 0.0f;
#pragma unroll
        for (int e = 0; e < 16; e++) { kk[e] = nk_[e]; qv[e] = nq[e]; }
        if (tok + 1 < T) ld(tok + 1);
#pragma unroll
        for (int e = 0; e < 16; e++) { r[e] *= decay; sk += r[e] * kk[e]; }
        for (int off = L >> 1; off; off >>= 1) sk += shflx(sk, off);
        float d = bt * (vt - sk);
#pragma unroll
        for (int e = 0; e < 16; e++) { r[e] += d * kk[e]; oq += r[e] * qv[e]; }
        for (int off = L >> 1; off; off >>= 1) oq += shflx(oq, off);
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

template <int DNP, bool SPLIT, int PROBE = 0>
__device__ __forceinline__ void dn_opt(const float* y, float* S, const float* beta, const float* g, float* o,
                                        int nk, int nv, int ilv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap,
                                        const float* dtb, const float* ga) {
    int L = dk / DNP, R = 32 / L, lane = threadIdx.x & 31, s = lane % L, vh = blockIdx.x, kh = ilv ? vh / (nv / nk) : vh % nk;
    int j0 = (blockIdx.y * (int)(blockDim.x >> 5) + (threadIdx.x >> 5)) * R, j = j0 + lane / L;
    if (j0 >= dv) return;  // whole warp past the end (R divides dv)
    float4* row = (float4*)(S + ((size_t)vh * dv + j) * dk);
    float r[DNP];
#pragma unroll
    for (int i = 0; i < DNP / 4; i++) { float4 v = row[s + L * i]; r[4 * i] = v.x; r[4 * i + 1] = v.y; r[4 * i + 2] = v.z; r[4 * i + 3] = v.w; }
    // Running pointers, one step per token: the loop body used to recompute `tok * stride` (and the 64-bit
    // multiplies and divides that drag in) for every operand, which was most of its instruction count.
    const float4* kp = (const float4*)(y + (size_t)kh * dk);        // chunk s + L*i of this lane's row
    const float4* qp = (const float4*)(y + (size_t)(nk + kh) * dk);
    const float* vp = y + (size_t)2 * nk * dk + (size_t)vh * dv + j;  // token's v[j]
    const float* gp = g + vh, *bp = beta + vh;                        // this head's decay gate and beta
    float* op = o + (size_t)vh * dv + j;                              // token's output element
    float* sp = snap + (size_t)((size_t)vh * dv + j) * dk;            // snapshot of this lane's chunks
    const int kstep = ych / 4, snapstep = nv * dv * dk;
    const int ns = min(nsnap, T - 1);
    // Token inputs do not depend on the state: token tok + 1's are loaded while token tok computes.
    float nk_[DNP], nq[DNP], ng, nb, nvv;
    auto ld = [&]() {
#pragma unroll
        for (int i = 0; i < DNP / 4; i++) {
            float4 v = kp[s + L * i]; nk_[4 * i] = v.x; nk_[4 * i + 1] = v.y; nk_[4 * i + 2] = v.z; nk_[4 * i + 3] = v.w;
            v = qp[s + L * i]; nq[4 * i] = v.x; nq[4 * i + 1] = v.y; nq[4 * i + 2] = v.z; nq[4 * i + 3] = v.w;
        }
        ng = *gp; nb = *bp; nvv = *vp;
        if constexpr (false) { ng = softplus(ng + dtb[vh]) * ga[vh]; nb = sigmoid(nb); }
        kp += kstep; qp += kstep; gp += nv; bp += nv; vp += ych;
    };
    ld();
    for (int tok = 0; tok < T; tok++) {
        float kk[DNP], qv[DNP], decay = PROBE == 1 ? 0.99f : expf(ng), bt = nb, vt = nvv, sk = 0.0f, oq = 0.0f;
#pragma unroll
        for (int e = 0; e < DNP; e++) { kk[e] = nk_[e]; qv[e] = nq[e]; }
        if (tok + 1 < T) ld();
        float sk1 = 0.0f, oq1 = 0.0f;
        if constexpr (SPLIT) {
#pragma unroll
            for (int e = 0; e < DNP; e += 2) { r[e] *= decay; r[e + 1] *= decay; sk += r[e] * kk[e]; sk1 += r[e + 1] * kk[e + 1]; }
            sk += sk1;
        } else {
#pragma unroll
            for (int e = 0; e < DNP; e++) { r[e] *= decay; sk += r[e] * kk[e]; }
        }
        if (PROBE != 2) for (int off = L >> 1; off; off >>= 1) sk += shflx(sk, off);
        float d = bt * (vt - sk);
        if constexpr (SPLIT) {
#pragma unroll
            for (int e = 0; e < DNP; e += 2) { r[e] += d * kk[e]; r[e + 1] += d * kk[e + 1]; oq += r[e] * qv[e]; oq1 += r[e + 1] * qv[e + 1]; }
            oq += oq1;
        } else {
#pragma unroll
            for (int e = 0; e < DNP; e++) { r[e] += d * kk[e]; oq += r[e] * qv[e]; }
        }
        if (PROBE != 2) for (int off = L >> 1; off; off >>= 1) oq += shflx(oq, off);
        if (s == 0) *op = oq * scale;
        if (tok < ns) {
            float4* sr = (float4*)sp;
#pragma unroll
            for (int i = 0; i < DNP / 4; i++) sr[s + L * i] = make_float4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
        }
        op += nv * dv; sp += snapstep;
    }
#pragma unroll
    for (int i = 0; i < DNP / 4; i++) row[s + L * i] = make_float4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
}

__global__ void dn16(const float* y, float* S, const float* beta, const float* g, float* o,
                        int nk, int nv, int ilv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap,
                        const float* dtb, const float* ga) {
    dn_opt<16, false>(y, S, beta, g, o, nk, nv, ilv, dk, dv, scale, T, ych, snap, nsnap, dtb, ga);
}
__global__ void dn32(const float* y, float* S, const float* beta, const float* g, float* o,
                        int nk, int nv, int ilv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap,
                        const float* dtb, const float* ga) {
    dn_opt<32, false>(y, S, beta, g, o, nk, nv, ilv, dk, dv, scale, T, ych, snap, nsnap, dtb, ga);
}
__global__ void dn16s(const float* y, float* S, const float* beta, const float* g, float* o,
                        int nk, int nv, int ilv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap,
                        const float* dtb, const float* ga) {
    dn_opt<16, true>(y, S, beta, g, o, nk, nv, ilv, dk, dv, scale, T, ych, snap, nsnap, dtb, ga);
}
__global__ void dn32s(const float* y, float* S, const float* beta, const float* g, float* o,
                        int nk, int nv, int ilv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap,
                        const float* dtb, const float* ga) {
    dn_opt<32, true>(y, S, beta, g, o, nk, nv, ilv, dk, dv, scale, T, ych, snap, nsnap, dtb, ga);
}

__global__ void dn16x(const float* y, float* S, const float* beta, const float* g, float* o,
                        int nk, int nv, int ilv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap,
                        const float* dtb, const float* ga) {
    dn_opt<16, false, 1>(y, S, beta, g, o, nk, nv, ilv, dk, dv, scale, T, ych, snap, nsnap, dtb, ga);
}
__global__ void dn16r(const float* y, float* S, const float* beta, const float* g, float* o,
                        int nk, int nv, int ilv, int dk, int dv, float scale, int T, int ych, float* snap, int nsnap,
                        const float* dtb, const float* ga) {
    dn_opt<16, false, 2>(y, S, beta, g, o, nk, nv, ilv, dk, dv, scale, T, ych, snap, nsnap, dtb, ga);
}
// ---------------------------------------------------------------------------------------------------------------
constexpr int DKN = 128, NKN = 16, NVN = 32, DVN = 128, TN = 4096, YCH = NVN * DVN + 2 * NKN * DKN;
struct Args {
    std::vector<float> y, S, beta, g, o;
    float *dy, *dS, *dbeta, *dg, *dout, *dsnap;
};
static void init(Args& a) {
    size_t ny = (size_t)TN * YCH;
    a.y.assign(ny, 0.f); a.S.assign((size_t)NVN * DVN * DKN, 0.f);
    a.beta.assign(TN * NVN, 0.f); a.g.assign(TN * NVN, 0.f); a.o.assign((size_t)TN * NVN * DVN, 0.f);
    srand(1234);
    auto rnd = [] { return (float)rand() / 2147483647.f - 0.5f; };
    for (auto& v : a.y) v = rnd();
    for (auto& v : a.S) v = rnd() * 0.1f;
    for (auto& v : a.beta) v = rnd();
    for (auto& v : a.g) v = -0.01f - fabsf(rnd()) * 0.02f;  // decay < 1: a real gated delta rule
    hipMalloc(&a.dy, ny * 4); hipMalloc(&a.dS, a.S.size() * 4); hipMalloc(&a.dbeta, a.beta.size() * 4);
    hipMalloc(&a.dg, a.g.size() * 4); hipMalloc(&a.dout, a.o.size() * 4); hipMalloc(&a.dsnap, 4);
    hipMemcpy(a.dy, a.y.data(), ny * 4, hipMemcpyHostToDevice);
    hipMemcpy(a.dbeta, a.beta.data(), a.beta.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(a.dg, a.g.data(), a.g.size() * 4, hipMemcpyHostToDevice);
}
struct Res { float ms; std::vector<float> out; };
template <typename K> static Res run(K k, Args& a, int dnp, int ych = YCH) {
    int rows = 32 * dnp / DKN, gy = DVN / (8 * rows);
    dim3 grid(NVN, gy), block(256);
    auto launch = [&] { k<<<grid, block>>>(a.dy, a.dS, a.dbeta, a.dg, a.dout, NKN, NVN, 0, DKN, DVN, 1.0f / sqrtf(DKN), TN, ych, a.dsnap, 0, nullptr, nullptr); };
    hipMemcpy(a.dS, a.S.data(), a.S.size() * 4, hipMemcpyHostToDevice);
    hipMemset(a.dout, 0, a.o.size() * 4);
    launch(); hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    const int reps = 10;
    hipEventRecord(e0);
    for (int i = 0; i < reps; i++) launch();
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1);
    hipMemcpy(a.o.data(), a.dout, a.o.size() * 4, hipMemcpyDeviceToHost);
    return {ms / reps, a.o};
}
int main() {
    Args a; init(a);
    Res ref = run(dn_base, a, 16);
    printf("%-10s %8.3f ms  %5.2fx  max |reldiff| vs base\n", "base", ref.ms, 1.0);
    struct V { const char* name; void (*k)(const float*, float*, const float*, const float*, float*, int, int, int, int, int, float, int, int, float*, int, const float*, const float*); int dnp; };
    V vs[] = {{"dn16", dn16, 16}, {"dn16s", dn16s, 16}, {"dn32", dn32, 32}, {"dn32s", dn32s, 32},
              {"no expf(g)", dn16x, 16}, {"no reductions", dn16r, 16}};
    for (auto& v : vs) {
        Res r = run(v.k, a, v.dnp);
        double worst = 0;
        for (size_t i = 0; i < ref.out.size(); i++) {
            double d = ref.out[i] != 0 ? fabs((ref.out[i] - r.out[i]) / ref.out[i]) : fabs(ref.out[i] - r.out[i]);
            if (d > worst) worst = d;
        }
        printf("%-13s %8.3f ms  %5.2fx  %.2e\n", v.name, r.ms, ref.ms / r.ms, worst);
    }
    // The same scan with every token reading token 0's q/k/v (ych = 0): the per-token reads then stay in L1, which
    // separates "cost of the data" from "cost of the strided, 128-blocks-redundant reads".
    Res l1 = run(dn16, a, 16, 0);
    printf("%-10s %8.3f ms  %5.2fx  (same token's q/k/v every step)\n", "dn16-l1", l1.ms, ref.ms / l1.ms);
    return 0;
}
