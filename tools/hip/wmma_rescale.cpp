// Inner-loop shapes of the prefill GEMM: f16 WMMA vs int8 WMMA with the per-16 rescale fold that int8 needs.
// Both read their fragments from the same shared-memory tiles and run 16 WMMA per k-step (2x2 tiles of 16x16, 4
// k-groups), which is one GEMM_TC step of the 128-thread warp tile.
#include <hip/hip_runtime.h>
#include <cstdio>
using h8 = _Float16 __attribute__((ext_vector_type(8)));
using f8 = float __attribute__((ext_vector_type(8)));
using b8 = int __attribute__((ext_vector_type(2)));
using i8v = int __attribute__((ext_vector_type(8)));

#define P 72
__global__ void f16_loop(float* out, int iters) {
    __shared__ __align__(16) _Float16 xs[128 * P], ws[64 * P];
    int tid = threadIdx.x;
    for (int i = tid; i < 128 * P; i += blockDim.x) xs[i] = (_Float16)(i & 7) * 0.01f;
    for (int i = tid; i < 64 * P; i += blockDim.x) ws[i] = (_Float16)(i & 7) * 0.01f;
    __syncthreads();
    int warp = tid >> 5, lane = tid & 31, l16 = lane & 15, w0 = (lane >> 4) * 8;
    int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    f8 acc[4]; for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
    for (int t = 0; t < iters; t++) {
        #pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
            h8 a[2], b[2];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) *(uint4*)&a[mt] = *(const uint4*)(xs + (wm + mt * 16 + l16) * P + kk + w0);
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) *(uint4*)&b[nt] = *(const uint4*)(ws + (wn + nt * 16 + l16) * P + kk + w0);
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) for (int nt = 0; nt < 2; nt++)
                acc[mt * 2 + nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[mt], b[nt], acc[mt * 2 + nt]);
        }
    }
    float s = 0; for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) s += acc[i][j];
    if (tid == 0) out[blockIdx.x] = s;
}
__global__ void i8_loop(float* out, int iters) {
    __shared__ __align__(16) signed char xq[128 * P], wq[64 * P];
    __shared__ float xd[128][2], xs2[128][4];   // X scale per 32 keys, X sums per 16 keys
    __shared__ float2 wdm[4][64];               // W scale and min per 16 weights
    int tid = threadIdx.x;
    for (int i = tid; i < 128 * P; i += blockDim.x) xq[i] = (signed char)(i & 31);
    for (int i = tid; i < 64 * P; i += blockDim.x) wq[i] = (signed char)(i & 31);
    for (int i = tid; i < 128; i += blockDim.x) { xd[i][0] = 0.01f; xd[i][1] = 0.02f; for (int j = 0; j < 4; j++) xs2[i][j] = 0.5f; }
    for (int i = tid; i < 64; i += blockDim.x) for (int g = 0; g < 4; g++) wdm[g][i] = make_float2(0.01f, 0.3f);
    __syncthreads();
    int warp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4, w0 = hi * 8;
    int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    f8 acc[4]; for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
    for (int t = 0; t < iters; t++) {
        #pragma unroll
        for (int g = 0; g < 4; g++) {   // 16-key groups = one weight scale
            b8 a[2], b[2];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) *(uint2*)&a[mt] = *(const uint2*)(xq + (wm + mt * 16 + l16) * P + g * 16 + w0);
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) *(uint2*)&b[nt] = *(const uint2*)(wq + (wn + nt * 16 + l16) * P + g * 16 + w0);
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    i8v c = {0,0,0,0,0,0,0,0};
                    i8v r = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a[mt], true, b[nt], c, false);
                    float2 wq2 = wdm[g][wn + nt * 16 + l16];
                    int base = wm + mt * 16 + 8 * hi;
                    f8 a4 = acc[mt * 2 + nt];
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float dx = xd[base + i][g >> 1];
                        float sx = xs2[base + i][g];
                        a4[i] += (float)r[i] * dx * wq2.x - sx * wq2.y;
                    acc[mt * 2 + nt] = a4;
                    }
                }
        }
    }
    float s = 0; for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) s += acc[i][j];
    if (tid == 0) out[blockIdx.x] = s;
}

__global__ void i8_nofold(float* out, int iters) {
    __shared__ __align__(16) signed char xq[128 * P], wq[64 * P];
    int tid = threadIdx.x;
    for (int i = tid; i < 128 * P; i += blockDim.x) xq[i] = (signed char)(i & 31);
    for (int i = tid; i < 64 * P; i += blockDim.x) wq[i] = (signed char)(i & 31);
    __syncthreads();
    int warp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4, w0 = hi * 8;
    int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    i8v acc[4]; for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0;
    for (int t = 0; t < iters; t++) {
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            b8 a[2], b[2];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) *(uint2*)&a[mt] = *(const uint2*)(xq + (wm + mt * 16 + l16) * P + g * 16 + w0);
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) *(uint2*)&b[nt] = *(const uint2*)(wq + (wn + nt * 16 + l16) * P + g * 16 + w0);
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) for (int nt = 0; nt < 2; nt++)
                acc[mt * 2 + nt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a[mt], true, b[nt], acc[mt * 2 + nt], false);
        }
    }
    int s = 0; for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) s += acc[i][j];
    if (tid == 0) out[blockIdx.x] = (float)s;
}
// SoA fold: 8 rows of the lane are consecutive, so a scale is one 8-wide vector of separate arrays.
__global__ void i8_soa(float* out, int iters) {
    __shared__ __align__(16) signed char xq[128 * P], wq[64 * P];
    __shared__ float dx0[128], sx0[128];   // X scale (one per 32 keys) and X sum (one per 16 keys)
    __shared__ float2 wdm[4][64];
    int tid = threadIdx.x;
    for (int i = tid; i < 128 * P; i += blockDim.x) xq[i] = (signed char)(i & 31);
    for (int i = tid; i < 64 * P; i += blockDim.x) wq[i] = (signed char)(i & 31);
    for (int i = tid; i < 128; i += blockDim.x) { dx0[i] = 0.01f; sx0[i] = 0.5f; }
    for (int i = tid; i < 64; i += blockDim.x) for (int g = 0; g < 4; g++) wdm[g][i] = make_float2(0.01f, 0.3f);
    __syncthreads();
    int warp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4, w0 = hi * 8;
    int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    f8 acc[4]; for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
    for (int t = 0; t < iters; t++) {
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            b8 a[2], b[2];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) *(uint2*)&a[mt] = *(const uint2*)(xq + (wm + mt * 16 + l16) * P + g * 16 + w0);
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) *(uint2*)&b[nt] = *(const uint2*)(wq + (wn + nt * 16 + l16) * P + g * 16 + w0);
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    i8v c = {0,0,0,0,0,0,0,0};
                    i8v r = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a[mt], true, b[nt], c, false);
                    float2 wq2 = wdm[g][wn + nt * 16 + l16];
                    int base = wm + mt * 16 + 8 * hi;
                    const float* dx = dx0 + base;
                    const float* sx = sx0 + base;
                    f8 a4 = acc[mt * 2 + nt];
                    #pragma unroll
                    for (int i = 0; i < 8; i++) a4[i] += (float)r[i] * dx[i] * wq2.x - sx[i] * wq2.y;
                    acc[mt * 2 + nt] = a4;
                }
        }
    }
    float s = 0; for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) s += acc[i][j];
    if (tid == 0) out[blockIdx.x] = s;
}

template <typename K> static float run(K k, int blocks, int iters) {
    float* out; hipMalloc(&out, blocks * 4);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    k<<<blocks, 64>>>(out, 16); hipDeviceSynchronize();
    hipEventRecord(a); k<<<blocks, 64>>>(out, iters); hipEventRecord(b); hipEventSynchronize(b);
    float ms = 0; hipEventElapsedTime(&ms, a, b); hipFree(out);
    return (float)ms;
}
int main() {
    int iters = 200000;
    for (int blocks : {256, 1024}) {
        float f = run(f16_loop, blocks, iters), i = run(i8_loop, blocks, iters);
        float n = run(i8_nofold, blocks, iters), so = run(i8_soa, blocks, iters);
        printf("blocks %4d: f16 %.2f | i8 nofold %.2f (%.2fx) | i8 SoA fold %.2f (%.2fx) | i8 scalar fold %.2f (%.2fx)\n",
               blocks, f, n, n / f, so, so / f, i, i / f);
    }
    return 0;
}
