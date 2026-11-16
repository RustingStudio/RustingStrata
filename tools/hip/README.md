# HIP micro-benchmarks

Device-side probes behind the decisions in `docs/HIP.md`. Build and run with the ROCm compiler; they need no model and
no RustingStrata binary:

```sh
hipcc --offload-arch=gfx1201 -O3 wmma_rate.cpp -o wmma_rate && ./wmma_rate
hipcc --offload-arch=gfx1201 -O3 wmma_rescale.cpp -o wmma_rescale && ./wmma_rescale
```

- `wmma_rate.cpp`: the matrix-core instruction rate of the two operand types, operands in registers, so the numbers
  are the instruction rate and not memory bandwidth. On RX 9070 XT (gfx1201): f16 57 TMAC/s, int8 137 TMAC/s.
- `wmma_rescale.cpp`: the inner-loop shape of the prefill GEMM (`gemm_q`: 2x2 tiles of 16x16, four 16-deep k-steps per
  BK = 64, fragments from shared memory), as f16 and as int8 with the epilogue the integer weight formats need. The
  int8 instruction is twice as fast on its own (0.35x the time) but the per-16-weight rescale takes it to 6.6x the f16
  time even written SoA with 8 rows contiguous per lane, which is why the HIP backend keeps the f16 matrix-core path.
