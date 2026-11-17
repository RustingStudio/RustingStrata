# HIP micro-benchmarks

Device-side probes behind the decisions in `docs/HIP.md`. Build and run with the ROCm compiler; they need no model and
no RustingStrata binary:

```sh
hipcc --offload-arch=gfx1201 -O3 wmma_rate.cpp -o wmma_rate && ./wmma_rate
hipcc --offload-arch=gfx1201 -O3 wmma_rescale.cpp -o wmma_rescale && ./wmma_rescale
```

- `deltanet.cpp`: the gated delta-rule scan (`deltanet_seq`) alone, at the 35B model's shape, with the variants that
  were tried against it: fewer, fatter lanes, two accumulators a dot sum, the loop's address math hoisted into
  pointer increments, and two ablations (the per-token `expf`, the two warp reductions). All of them are neutral or
  slower; the reads are free (a variant that re-reads token 0 every step is as fast), and the reductions plus the
  exp are the only parts with headroom — which is the chunked WY form's job, not a rearrangement of the scan. The
  file's header carries the numbers.
- `wmma_rate.cpp`: the matrix-core instruction rate of the two operand types, operands in registers, so the numbers
  are the instruction rate and not memory bandwidth. On RX 9070 XT (gfx1201): f16 57 TMAC/s, int8 137 TMAC/s.
- `wmma_rescale.cpp`: the inner-loop shape of the prefill GEMM (`gemm_q`: 2x2 tiles of 16x16, four 16-deep k-steps per
  BK = 64, fragments from shared memory), as f16 and as int8 with the epilogue the integer weight formats need. The
  int8 instruction is twice as fast on its own (0.35x the time) but the per-16-weight rescale takes it to 6.6x the f16
  time even written SoA with 8 rows contiguous per lane, which is why the HIP backend keeps the f16 matrix-core path.
