// Isolated matrix-core rate: f16 WMMA vs int8 WMMA, same 16x16x16 shape, operands in registers so the number is the
// instruction rate, not memory bandwidth.
#include <hip/hip_runtime.h>
#include <cstdio>
using h8 = _Float16 __attribute__((ext_vector_type(8)));
using f8 = float __attribute__((ext_vector_type(8)));
using i8v = int __attribute__((ext_vector_type(8)));
using b8 = int __attribute__((ext_vector_type(2)));  // 8 int8 values

__global__ void f16_loop(float* out, int iters) {
    h8 a, b; f8 c;
    for (int i = 0; i < 8; i++) { a[i] = (_Float16)i * 0.001f; b[i] = (_Float16)i * 0.002f; c[i] = 0.f; }
    for (int t = 0; t < iters; t++) { c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c); a[0] += 1e-7f; }
    float s = 0; for (int i = 0; i < 8; i++) s += c[i];
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}
__global__ void i8_loop(float* out, int iters) {
    b8 a = {0x01020304, 0x05060708};
    b8 b = {0x11121314, 0x15161718};
    i8v c; for (int i = 0; i < 8; i++) c[i] = 0;
    for (int t = 0; t < iters; t++) { c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false); a[0] ^= 1; }
    int s = 0; for (int i = 0; i < 8; i++) s += c[i];
    if (threadIdx.x == 0) out[blockIdx.x] = (float)s;
}
template <typename K> static float run(K k, int blocks, int iters) {
    float* out; hipMalloc(&out, blocks * 4);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    k<<<blocks, 64>>>(out, 16);                      // warm up
    hipDeviceSynchronize();
    hipEventRecord(a); k<<<blocks, 64>>>(out, iters); hipEventRecord(b); hipEventSynchronize(b);
    float ms = 0; hipEventElapsedTime(&ms, a, b);
    hipFree(out);
    double macs = (double)blocks * 2.0 * 4096.0 * iters;  // 2 warps a block, 16x16x16
    return (float)(macs / (ms * 1e-3) / 1e12);
}
int main() {
    int iters = 20000;
    for (int blocks : {256, 1024}) {
        printf("blocks %4d: f16 %.3f TMAC/s | i8 %.3f TMAC/s\n", blocks, run(f16_loop, blocks, iters), run(i8_loop, blocks, iters));
    }
    return 0;
}
