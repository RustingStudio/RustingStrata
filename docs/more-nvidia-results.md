# RTX 2080 Ti, Quadro RTX 6000, V100, A40, RTX A6000, RTX 4500 Ada, A100 40 GB and GTX 1080 Ti results

A second Slurm cluster (not the one in [datacenter-nvidia-results.md](datacenter-nvidia-results.md)), one GPU per job
with 8 CPU cores and 48 GB RAM. Rocky Linux 8.10, Rust 1.99.1, `target-cpu=native`, commit `f47e370`. NVRTC 13.0 from
the CUDA 13.0.1 toolkit, except 12.6 from CUDA 12.6.1 on the V100s and in the GTX 1080 Ti CUDA attempt. Model:
`Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (sha256 `ed7cda7e...f37fb0c`). Numbers are comparable within one job only: the cards
sit in hosts with different CPUs. The re-run and Q8_0 sections are at `ea4c8a8` and say so.

| GPU | VRAM | Arch | Driver | Host CPU (bench job) |
| --- | --- | --- | --- | --- |
| A100-PCIE-40GB | 40960 MiB | sm_80 | 615.71.09 | Xeon Gold 6330 |
| RTX A6000 | 49140 MiB | sm_86 | 615.71.09 | EPYC 7413 |
| A40 | 46068 MiB | sm_86 | 615.71.09 | Xeon Gold 6330 |
| RTX 4500 Ada Generation | 24570 MiB | sm_89 | 615.71.09 | Xeon Gold 6246 |
| Quadro RTX 6000 | 23040 MiB | sm_75 | 615.71.09 | Xeon Gold 6230R |
| GeForce RTX 2080 Ti | 11264 MiB | sm_75 | 615.71.09 | EPYC 7402P |
| Tesla V100-SXM2-16GB | 16384 MiB | sm_70 | 580.95.05 | Xeon Silver 4114 |
| Tesla V100-SXM2-32GB | 32768 MiB | sm_70 | 580.95.05 | Xeon Gold 6148 |
| GeForce GTX 1080 Ti | 11264 MiB | sm_61 | 580.95.05 | Xeon Gold 5120 |

## 2026-11-11 — re-run at `ea4c8a8` without `STRATA_VK_F16_TRUNC`

`ea4c8a8` rounds the f16 GEMM inputs in the shader (`rh` in `gemm_f.glsl`) and removes the variable. The four cards
that needed it were run again, test suite and bench, with the variable unset.

| GPU | Driver | `cargo test` at `ea4c8a8` |
| --- | --- | --- |
| RTX 2080 Ti | 615.71.09 | 185 passed, 0 failed, 5 ignored |
| Quadro RTX 6000 | 615.71.09 | 185 passed, 0 failed, 5 ignored |
| V100 16 GB | 580.95.05 | 185 passed, 0 failed, 5 ignored |
| V100 32 GB | 580.95.05 | 185 passed, 0 failed, 5 ignored |

The bench below is the Q2_K_XL bench of this file, repeated on the node of the `f47e370` run (the nodes hold several
cards of the type, so it may be another card of that node). Calibrated, `--mtp 0`, mean (min-max) tok/s, `f47e370` ->
`ea4c8a8`. The V100 32 GB bench had not started when this was written.

| GPU | Backend | pp 512 prefill | pp 4096 prefill | decode after 4096 |
| --- | --- | --- | --- | --- |
| RTX 2080 Ti | Vulkan | 1934.8 (1764.2-2051.2) -> 1888.1 (1863.0-1909.8) | 2629.7 (2608.8-2642.7) -> 2498.7 (2477.6-2510.2) | 126.3 -> 126.0 |
| Quadro RTX 6000 | Vulkan | 1872.6 (1688.9-1977.9) -> 1784.5 (1715.7-1832.6) | 2719.4 (2715.7-2724.0) -> 2669.9 (2659.6-2675.8) | 109.0 -> 108.3 |
| V100 16 GB | Vulkan | 804.6 (796.3-809.8) -> 756.9 (736.6-767.0) | 1676.3 (1675.2-1678.0) -> 1630.2 (1627.1-1632.7) | 119.7 -> 119.7 |
| RTX 2080 Ti | CUDA | 2023.3 (2018.7-2030.1) -> 2020.3 (2009.9-2028.2) | 2713.5 (2712.0-2715.0) -> 2713.2 (2709.6-2718.4) | 139.7 -> 139.7 |
| Quadro RTX 6000 | CUDA | 1978.4 (1889.7-2029.1) -> 1978.1 (1881.1-2032.1) | 2794.6 (2791.2-2798.2) -> 2827.2 (2823.7-2829.7) | 122.3 -> 122.4 |
| V100 16 GB | CUDA | 2253.3 (2233.9-2275.7) -> 2251.5 (2238.8-2276.5) | 3144.9 (3138.7-3154.1) -> 3147.3 (3140.4-3152.5) | 128.1 -> 128.1 |

- **Vulkan prefill at pp 4096 is 5.0%, 1.8% and 2.8% lower** on the RTX 2080 Ti, the Quadro RTX 6000 and the V100
  16 GB; the ranges of the two commits do not overlap. CUDA on the same nodes moved by 0.0%, +1.2% and +0.1%. So on
  the RTX 2080 Ti and the V100 the difference is in the Vulkan path. The Quadro RTX 6000 re-run got another card of
  its node (bus `D8:00.0`, before `3B:00.0`), and its 1.8% is the size of the CUDA shift there. Decode changed by at
  most 0.6%.
- The other numbers in this file for these cards are still the `f47e370` ones.

## 2026-11-11 — Q8_0

`Qwen3.6-35B-A3B-Q8_0.gguf` (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, 37.8 GB, sha256 `c1283d8b...9a76131`), the file of
[datacenter-nvidia-results.md](datacenter-nvidia-results.md). Commit `ea4c8a8`, 16 cores and 88 GB RAM per job, the
same four steps per backend as [below](#2026-11-11--benchmarks-before-and-after-calibrate). The model is on a shared
parallel file system (BeeGFS). The RTX A6000 and V100 32 GB jobs had not started when this was written.

| GPU | Host CPU | `calibrate`, CUDA (chunk / MTP drafts / short threshold) | `calibrate`, Vulkan |
| --- | --- | --- | --- |
| A100 40 GB | EPYC 7502 | 4096 / 1 / 8 | 4096 / 2 / 8 |
| A40 | Xeon Gold 6348 | 4096 / 1 / 8 | 4096 / 1 / 8 |
| RTX 4500 Ada | Xeon Gold 6246 | 4096 / 0 / 32 | 4096 / 0 / 64 |
| Quadro RTX 6000 | Xeon Gold 6230R | 4096 / 1 / 128 | 4096 / 1 / 64 |
| RTX 2080 Ti | Xeon Silver 4214 | 4096 / 0 / 128 | 4096 / 0 / 256 |
| V100 16 GB | Xeon Silver 4114 | 4096 / 1 / 128 | 4096 / 0 / 128 |

- **The A100 40 GB and the A40 hold every expert** (10240, 10496 with MTP). With CUDA the A100 prefills pp 4096 at
  6480 tok/s and decodes at 187 tok/s with `--mtp 0`, 279 calibrated (MTP 1); the A40 5397, 126 and 180.
- **The 24 GB cards hold 57-64% of the experts and hit 98-99%.** CUDA with `--mtp 0`: RTX 4500 Ada 1680 tok/s prefill
  and 96 decode, Quadro RTX 6000 1307 and 88.
- **The V100 16 GB holds 37-39% (hit rate 89-91%), the RTX 2080 Ti 23% (74-76%).** CUDA with `--mtp 0`: 803 tok/s
  prefill and 65 decode on the V100, 697 and 57 on the RTX 2080 Ti.
- **Where experts miss, runs with the same settings differ.** `calibrate` chose MTP 0 on the RTX 4500 Ada (both
  backends), on the RTX 2080 Ti (both) and on the V100 with Vulkan, so there the "calibrated" and "`--mtp 0`" rows
  are the same configuration; they still differ by up to 31% at pp 4096 prefill (RTX 4500 Ada Vulkan 1028.0 against
  1346.5, V100 Vulkan 717.9 against 568.6). The first run on the RTX 4500 Ada (CUDA, uncalibrated) decoded at 4.4
  tok/s after pp 512 and 11.7 after pp 4096, against 98 and 96 in the CUDA `--mtp 0` run: likely a cold start (2898
  cache swaps, against 1036 and 1035 in the next two runs).

Mean (min-max) of 3 repetitions, tok/s.

| GPU | Backend | Run | MTP | Resident experts | Hit rate % | pp 512 prefill | decode after 512 | pp 4096 prefill | decode after 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A100 40 GB | CUDA | uncalibrated | 0 | 10240/10240 | 100.0 | 3341.2 (2338.5-3843.8) | 192.1 (187.3-194.5) | 6506.3 (6498.9-6512.8) | 187.4 (186.9-187.7) |
| A100 40 GB | CUDA | calibrated | 1 | 10496/10496 | 100.0 | 3317.3 (2581.6-3739.4) | 293.9 (292.0-294.9) | 6027.3 (6019.9-6036.4) | 279.2 (277.3-280.3) |
| A100 40 GB | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 100.0 | 3362.6 (2457.6-3846.0) | 193.8 (192.5-194.4) | 6479.6 (6467.4-6500.8) | 187.4 (186.9-187.7) |
| A100 40 GB | Vulkan | uncalibrated | 0 | 10240/10240 | 100.0 | 2212.0 (207.2-3262.2) | 124.4 (90.2-142.1) | 5345.9 (4181.7-5928.6) | 128.1 (110.1-137.3) |
| A100 40 GB | Vulkan | calibrated | 2 | 10496/10496 | 100.0 | 2942.7 (2597.5-3122.4) | 256.2 (253.3-257.8) | 5480.7 (5400.3-5536.4) | 221.6 (167.6-248.7) |
| A100 40 GB | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 100.0 | 2999.0 (2610.5-3203.8) | 143.7 (142.6-144.3) | 5882.3 (5798.3-5925.0) | 139.1 (138.8-139.3) |
| A40 | CUDA | uncalibrated | 0 | 10240/10240 | 100.0 | 3156.3 (2894.4-3331.7) | 128.7 (127.8-129.2) | 5470.2 (5466.0-5472.4) | 126.0 (125.8-126.1) |
| A40 | CUDA | calibrated | 1 | 10496/10496 | 100.0 | 3101.0 (2969.2-3218.1) | 183.6 (182.5-184.1) | 5088.4 (5080.3-5093.2) | 179.6 (179.1-179.9) |
| A40 | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 100.0 | 3236.9 (3106.3-3306.0) | 128.8 (128.1-129.1) | 5396.8 (5393.7-5401.5) | 126.0 (125.8-126.1) |
| A40 | Vulkan | uncalibrated | 0 | 10240/10240 | 100.0 | 1884.7 (225.0-2750.6) | 99.7 (82.4-108.4) | 4407.4 (3599.7-4811.8) | 100.6 (89.2-106.3) |
| A40 | Vulkan | calibrated | 1 | 10496/10496 | 100.0 | 2241.3 (1823.3-2459.7) | 145.1 (143.9-145.8) | 4417.7 (4370.9-4476.1) | 135.3 (111.5-147.4) |
| A40 | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 100.0 | 2373.9 (1881.0-2641.5) | 107.2 (105.7-108.2) | 4753.7 (4747.4-4763.2) | 105.4 (105.1-105.7) |
| RTX 4500 Ada | CUDA | uncalibrated | 0 | 6482/10240 | 98.6 | 428.5 (2.8-750.1) | 4.4 (1.6-8.9) | 1204.6 (254.2-1764.2) | 11.7 (10.0-13.5) |
| RTX 4500 Ada | CUDA | calibrated | 0 | 6482/10240 | 98.9 | 388.3 (297.2-458.7) | 76.3 (32.5-98.5) | 1697.2 (1508.8-1845.2) | 72.3 (26.1-96.2) |
| RTX 4500 Ada | CUDA | calibrated, `--mtp 0` | 0 | 6482/10240 | 98.9 | 390.2 (303.7-457.9) | 98.0 (97.5-98.4) | 1680.1 (1492.4-1825.5) | 95.8 (94.4-96.8) |
| RTX 4500 Ada | Vulkan | uncalibrated | 0 | 6530/10240 | 98.9 | 270.7 (140.0-383.2) | 83.9 (71.8-90.2) | 999.9 (834.1-1110.9) | 83.4 (75.8-87.4) |
| RTX 4500 Ada | Vulkan | calibrated | 0 | 6530/10240 | 98.9 | 298.9 (274.0-332.5) | 89.6 (88.3-90.3) | 1028.0 (970.3-1091.0) | 85.3 (83.6-86.2) |
| RTX 4500 Ada | Vulkan | calibrated, `--mtp 0` | 0 | 6530/10240 | 98.9 | 345.3 (273.3-424.7) | 90.5 (89.7-91.3) | 1346.5 (1231.5-1422.7) | 87.1 (86.7-87.8) |
| Quadro RTX 6000 | CUDA | uncalibrated | 0 | 5975/10240 | 97.6 | 243.4 (44.7-362.7) | 87.1 (82.2-90.0) | 1203.9 (931.2-1382.4) | 89.4 (87.1-91.0) |
| Quadro RTX 6000 | CUDA | calibrated | 1 | 5935/10496 | 97.7 | 283.8 (222.8-328.1) | 130.4 (126.4-135.1) | 1227.6 (1130.8-1306.2) | 127.1 (125.8-127.9) |
| Quadro RTX 6000 | CUDA | calibrated, `--mtp 0` | 0 | 5975/10240 | 98.0 | 308.1 (243.8-355.9) | 93.8 (93.2-94.4) | 1306.6 (1198.7-1392.2) | 88.3 (86.2-90.8) |
| Quadro RTX 6000 | Vulkan | uncalibrated | 0 | 6017/10240 | 98.2 | 257.3 (95.7-354.1) | 74.9 (47.7-89.1) | 988.1 (931.9-1028.5) | 79.6 (70.0-84.5) |
| Quadro RTX 6000 | Vulkan | calibrated | 1 | 5981/10496 | 97.9 | 277.6 (219.7-324.8) | 120.0 (118.0-121.3) | 919.3 (807.2-983.2) | 107.9 (91.7-116.3) |
| Quadro RTX 6000 | Vulkan | calibrated, `--mtp 0` | 0 | 6017/10240 | 98.2 | 296.8 (241.6-351.0) | 88.2 (87.0-89.4) | 949.3 (910.7-1011.8) | 84.6 (83.6-85.5) |
| RTX 2080 Ti | CUDA | uncalibrated | 0 | 2313/10240 | 74.2 | 86.3 (20.7-121.3) | 46.9 (44.5-48.5) | 636.7 (499.6-709.9) | 46.2 (45.3-46.6) |
| RTX 2080 Ti | CUDA | calibrated | 0 | 2313/10240 | 75.5 | 112.5 (95.0-122.1) | 59.1 (56.7-60.5) | 695.8 (669.3-711.8) | 55.7 (55.2-56.0) |
| RTX 2080 Ti | CUDA | calibrated, `--mtp 0` | 0 | 2313/10240 | 75.4 | 114.1 (98.9-122.2) | 59.9 (58.1-62.2) | 697.4 (673.1-713.2) | 56.7 (55.9-57.5) |
| RTX 2080 Ti | Vulkan | uncalibrated | 0 | 2354/10240 | 76.1 | 97.9 (57.6-119.4) | 44.7 (33.4-50.5) | 588.8 (563.8-612.4) | 44.7 (41.0-46.6) |
| RTX 2080 Ti | Vulkan | calibrated | 0 | 2354/10240 | 76.0 | 87.4 (84.4-89.8) | 53.0 (51.8-54.4) | 572.7 (554.2-583.8) | 47.6 (47.3-48.0) |
| RTX 2080 Ti | Vulkan | calibrated, `--mtp 0` | 0 | 2354/10240 | 76.0 | 110.7 (95.2-118.6) | 56.2 (54.7-59.2) | 688.4 (657.0-705.5) | 54.1 (53.5-54.8) |
| V100 16 GB | CUDA | uncalibrated | 0 | 3930/10240 | 89.2 | 87.9 (25.9-156.8) | 59.7 (42.6-69.1) | 685.8 (514.0-836.0) | 61.6 (55.7-67.1) |
| V100 16 GB | CUDA | calibrated | 1 | 3890/10496 | 89.5 | 178.0 (134.7-203.2) | 94.4 (85.4-99.8) | 987.8 (923.5-1035.8) | 75.3 (72.1-77.0) |
| V100 16 GB | CUDA | calibrated, `--mtp 0` | 0 | 3930/10240 | 90.1 | 140.6 (110.6-157.7) | 75.4 (71.1-81.7) | 803.4 (753.4-837.6) | 65.3 (64.0-66.4) |
| V100 16 GB | Vulkan | uncalibrated | 0 | 4016/10240 | 90.6 | 121.9 (83.1-143.2) | 67.4 (51.1-75.7) | 677.9 (628.9-708.0) | 61.0 (52.0-66.2) |
| V100 16 GB | Vulkan | calibrated | 0 | 4016/10240 | 90.8 | 136.3 (107.5-152.4) | 70.9 (66.9-76.3) | 717.9 (669.4-748.1) | 67.9 (64.3-72.3) |
| V100 16 GB | Vulkan | calibrated, `--mtp 0` | 0 | 4016/10240 | 90.9 | 124.2 (106.9-151.9) | 72.8 (67.0-76.8) | 568.6 (535.3-591.8) | 69.5 (68.9-70.1) |

## 2026-11-11 — what does not pass

- **GTX 1080 Ti (Pascal) runs on neither backend.** `--backend vulkan`: `no usable Vulkan device 0 (0 usable; NVIDIA
  GeForce GTX 1080 Ti: missing shaderFloat16)` for `bench`, `calibrate` and `check`. `--backend cuda` with NVRTC 12.6:
  the startup line is `cuda device 0: NVIDIA GeForce GTX 1080 Ti, sm_61 (Volta, m8n8k4)`, then `NVRTC compile of
  kernels.cu failed` with `default_program(2321): error: identifier "__nanosleep" is undefined` (the only error; the
  rest are warnings). `cargo test` reports 185 passed there, but that job had no CUDA toolkit loaded (`SKIP: cuda:
  CUDA init: the NVRTC library (libnvrtc.so) is not installed`) and no usable Vulkan device, so the GPU tests
  skipped; it is not a pass for the card. The suite was not run on this card with NVRTC present.
- **Four Vulkan kernel tests failed on Turing and Volta at `f47e370` unless `STRATA_VK_F16_TRUNC=1` was set.** On the
  RTX 2080 Ti, the Quadro RTX 6000 and both V100s, `cargo test` was 181 passed, 4 failed, 5 ignored; with
  `STRATA_VK_F16_TRUNC=1` it was 185 passed, 0 failed on all four. The variable changed nothing in the engine: it told
  the kernel tests to expect an f16 conversion that truncates (`tests/kernels.rs`). This was known for the V100; the
  two Turing cards have cooperative matrices (f16 and int8) and driver 615.71.09 and needed it too. The Ampere and
  Ada cards on the same driver passed without it. The variable was passed to those four jobs with
  `sbatch --export=ALL`; the logs do not echo it. The runs with and without it were on different nodes of the same
  card type, except for the Quadro RTX 6000 (same node). The failures, with the same values on both Turing cards and
  (in parentheses) on both V100s:

  | Test | Message |
  | --- | --- |
  | `f16_weight_uploads_as_f32` | `vulkan gemm_dense rel err 0.0000665` (0.0000654) |
  | `gemm_matches_reference_for_every_type` | `IQ2_XS gemm (blk.0.ffn_gate_exps.weight): rel err 0.000512` (0.000511) |
  | `gemv_partial_superblocks_and_q2_0` | `vulkan Q8_0 cols 320: gemm rel err 0.000534` (0.000533) |
  | `vulkan_type_matrix` | `type matrix failures`, 4 |

  The f16 GEMM now rounds its inputs in software (`rh` in `gemm_f.glsl`), so the variable is gone and all four
  cards pass 185 tests without it. The rest of this section records the runs at `f47e370`.

## 2026-11-11 — test suite

`STRATA_MODEL=$M cargo test --release --no-fail-fast -- --test-threads=1 --nocapture`, one job per card (separate from
the bench job, sometimes on another node of the same card type). The second column is the run at `f47e370`.

| GPU | Result | With `STRATA_VK_F16_TRUNC=1` (at `f47e370`) |
| --- | --- | --- |
| A100 40 GB | 185 passed, 0 failed, 5 ignored | not run |
| RTX A6000 | 185 passed, 0 failed, 5 ignored | not run |
| A40 | 185 passed, 0 failed, 5 ignored | not run |
| RTX 4500 Ada | 185 passed, 0 failed, 5 ignored | not run |
| Quadro RTX 6000 | 181 passed, 4 failed, 5 ignored | 185 passed, 0 failed, 5 ignored |
| RTX 2080 Ti | 181 passed, 4 failed, 5 ignored | 185 passed, 0 failed, 5 ignored |
| V100 16 GB | 181 passed, 4 failed, 5 ignored | 185 passed, 0 failed, 5 ignored |
| V100 32 GB | 181 passed, 4 failed, 5 ignored | 185 passed, 0 failed, 5 ignored |

Both backends were open in every run in this table: each log has 48 `cuda device 0:` and 47 `vulkan:` device lines.
The skips are `no glslc` (`spirv_is_fresh`), `STRATA_TEST_BACKENDS does not select cuda`, the two DeepSeek tests
(`STRATA_DS_MODEL` unset) and, with NVRTC 13.0, `nvrtc_compiles_full_source_for_sm70`.
No run hung; all were serial (`--test-threads=1`), so this says nothing new about the parallel `vkDestroyDevice` hang.

## 2026-11-11 — benchmarks before and after `calibrate`

Each job builds, then runs on one card, with a new `XDG_CACHE_HOME` per job:

```sh
rustingstrata bench --model $M --gpu --backend cuda --pp 512,4096 --tg 128 --reps 3      # uncalibrated
rustingstrata calibrate --model $M --backend cuda
rustingstrata bench --model $M --gpu --backend cuda --pp 512,4096 --tg 128 --reps 3      # calibrated
rustingstrata bench --model $M --gpu --backend cuda --pp 512,4096 --tg 128 --reps 3 --mtp 0
# the same four steps with --backend vulkan, then
rustingstrata check --model $M --backend vulkan
```

What `calibrate` chose (chunk / MTP drafts / short threshold):

| Backend | A100 40 GB | RTX A6000 | A40 | RTX 4500 Ada | Quadro RTX 6000 | RTX 2080 Ti | V100 16 GB | V100 32 GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA | 4096 / 1 / 8 | 4096 / 2 / 8 | 4096 / 2 / 8 | 4096 / 2 / 8 | 4096 / 1 / 8 | 4096 / 1 / 8 | 4096 / 1 / 8 | 4096 / 1 / 8 |
| Vulkan | 4096 / 2 / 8 | 4096 / 2 / 8 | 4096 / 2 / 8 | 4096 / 2 / 8 | 4096 / 1 / 8 | 2048 / 1 / 8 | 4096 / 2 / 32 | 4096 / 2 / 32 |

Mean (min-max) of 3 repetitions, tok/s:

| GPU | Backend | Run | MTP | Resident experts | pp 512 prefill | decode after 512 | pp 4096 prefill | decode after 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A100 40 GB | CUDA | uncalibrated | 0 | 10240/10240 | 3306.1 (2120.6-3921.3) | 173.7 (171.2-175.1) | 5999.1 (5987.1-6010.4) | 168.9 (168.4-169.2) |
| A100 40 GB | CUDA | calibrated | 1 | 10496/10496 | 3507.8 (3066.0-3741.0) | 288.5 (285.8-289.9) | 5509.5 (5503.8-5517.0) | 274.6 (273.4-275.4) |
| A100 40 GB | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 3640.3 (3148.9-3907.5) | 174.4 (173.4-175.0) | 5877.3 (5870.5-5887.3) | 168.9 (168.6-169.1) |
| A100 40 GB | Vulkan | uncalibrated | 0 | 10240/10240 | 2237.2 (118.6-3310.1) | 117.5 (72.6-139.9) | 5263.8 (4518.1-5640.4) | 127.1 (112.0-134.8) |
| A100 40 GB | Vulkan | calibrated | 2 | 10496/10496 | 2739.2 (1984.9-3132.6) | 249.3 (242.5-254.1) | 5266.9 (5260.1-5278.8) | 222.8 (176.4-246.3) |
| A100 40 GB | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 2978.0 (2421.2-3270.4) | 140.8 (140.0-141.6) | 5639.1 (5601.2-5660.8) | 135.9 (135.9-136.0) |
| RTX A6000 | CUDA | uncalibrated | 0 | 10240/10240 | 3459.3 (3071.2-3662.8) | 194.4 (193.5-195.0) | 5170.1 (5167.0-5172.5) | 188.5 (188.3-188.8) |
| RTX A6000 | CUDA | calibrated | 2 | 10496/10496 | 3490.0 (3457.3-3522.8) | 346.2 (344.6-347.2) | 4813.7 (4800.0-4832.9) | 328.9 (325.6-330.7) |
| RTX A6000 | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 3626.6 (3595.0-3647.2) | 193.6 (192.5-194.2) | 5093.6 (5082.3-5102.7) | 187.9 (187.2-188.4) |
| RTX A6000 | Vulkan | uncalibrated | 0 | 10240/10240 | 2258.2 (133.1-3340.5) | 143.0 (87.5-170.8) | 4775.7 (4170.4-5086.4) | 152.8 (134.2-164.8) |
| RTX A6000 | Vulkan | calibrated | 2 | 10496/10496 | 2983.3 (2782.1-3088.4) | 282.5 (266.9-291.2) | 4782.3 (4749.4-4815.8) | 251.8 (196.5-281.7) |
| RTX A6000 | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 3140.0 (2973.9-3296.0) | 163.7 (153.2-169.9) | 5051.9 (5027.7-5066.1) | 164.6 (163.2-166.2) |
| A40 | CUDA | uncalibrated | 0 | 10240/10240 | 3399.6 (3263.6-3473.5) | 163.9 (162.8-164.5) | 4790.9 (4776.8-4799.6) | 159.3 (158.9-159.5) |
| A40 | CUDA | calibrated | 2 | 10496/10496 | 3316.4 (3299.4-3341.2) | 303.6 (302.2-304.5) | 4579.1 (4560.0-4593.7) | 289.3 (286.3-291.1) |
| A40 | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 3467.6 (3458.3-3486.0) | 164.1 (163.3-164.6) | 4805.2 (4802.8-4809.6) | 158.9 (157.7-159.6) |
| A40 | Vulkan | uncalibrated | 0 | 10240/10240 | 2127.1 (117.4-3137.2) | 120.4 (69.9-145.7) | 4513.4 (3953.1-4795.7) | 132.6 (115.1-141.4) |
| A40 | Vulkan | calibrated | 2 | 10496/10496 | 2924.5 (2758.5-3029.8) | 252.5 (245.0-256.3) | 4539.1 (4517.9-4551.1) | 222.1 (174.4-246.5) |
| A40 | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 3060.7 (2936.0-3146.0) | 145.2 (144.3-145.8) | 4795.5 (4779.3-4807.4) | 141.2 (141.1-141.4) |
| RTX 4500 Ada | CUDA | uncalibrated | 0 | 10240/10240 | 3627.5 (3331.3-3785.7) | 159.6 (158.8-160.0) | 5056.2 (5052.3-5063.3) | 152.8 (152.5-153.0) |
| RTX 4500 Ada | CUDA | calibrated | 2 | 10496/10496 | 3552.4 (3525.9-3598.8) | 284.6 (283.3-285.2) | 4631.0 (4625.1-4634.1) | 272.5 (270.0-273.9) |
| RTX 4500 Ada | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 3704.7 (3684.5-3730.9) | 159.3 (158.6-159.7) | 4933.8 (4923.7-4940.2) | 152.6 (152.2-152.8) |
| RTX 4500 Ada | Vulkan | uncalibrated | 0 | 10240/10240 | 2237.7 (121.6-3316.5) | 124.9 (74.8-150.3) | 4743.8 (4150.8-5045.6) | 134.7 (118.4-143.4) |
| RTX 4500 Ada | Vulkan | calibrated | 2 | 10496/10496 | 2906.3 (2736.9-2998.3) | 243.5 (238.8-246.0) | 4644.8 (4599.1-4670.9) | 221.1 (173.5-245.0) |
| RTX 4500 Ada | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 3137.4 (3031.9-3201.8) | 147.1 (146.3-147.6) | 4979.4 (4916.3-5011.6) | 141.8 (140.6-142.8) |
| Quadro RTX 6000 | CUDA | uncalibrated | 0 | 10240/10240 | 1924.8 (1727.6-2024.5) | 126.4 (125.9-126.7) | 2822.2 (2818.7-2824.3) | 122.2 (121.9-122.4) |
| Quadro RTX 6000 | CUDA | calibrated | 1 | 10496/10496 | 1900.1 (1791.8-1962.7) | 186.6 (185.9-187.0) | 2660.2 (2654.7-2664.9) | 178.6 (177.8-178.9) |
| Quadro RTX 6000 | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 1978.4 (1889.7-2029.1) | 126.4 (125.7-126.7) | 2794.6 (2791.2-2798.2) | 122.3 (122.0-122.5) |
| Quadro RTX 6000 | Vulkan | uncalibrated | 0 | 10240/10240 | 1320.7 (113.2-1934.6) | 95.5 (64.8-111.3) | 2636.5 (2431.1-2743.2) | 102.7 (92.0-108.1) |
| Quadro RTX 6000 | Vulkan | calibrated | 1 | 10496/10496 | 1572.0 (1065.5-1875.4) | 156.8 (154.4-158.9) | 2557.8 (2552.5-2562.6) | 142.5 (122.0-152.8) |
| Quadro RTX 6000 | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 1872.6 (1688.9-1977.9) | 112.3 (112.1-112.6) | 2719.4 (2715.7-2724.0) | 109.0 (108.7-109.2) |
| RTX 2080 Ti | CUDA | uncalibrated | 0 | 8334/10240 | 1448.6 (267.6-2095.7) | 141.8 (137.7-143.9) | 2748.6 (2600.3-2823.4) | 140.8 (140.2-141.5) |
| RTX 2080 Ti | CUDA | calibrated | 1 | 8213/10496 | 1898.6 (1852.0-1974.4) | 207.4 (206.8-207.7) | 2594.8 (2593.8-2595.6) | 197.2 (196.0-198.1) |
| RTX 2080 Ti | CUDA | calibrated, `--mtp 0` | 0 | 8334/10240 | 2023.3 (2018.7-2030.1) | 145.0 (144.3-145.5) | 2713.5 (2712.0-2715.0) | 139.7 (139.4-140.0) |
| RTX 2080 Ti | Vulkan | uncalibrated | 0 | 8482/10240 | 1317.5 (108.0-1947.8) | 107.7 (60.5-131.4) | 2620.6 (2381.8-2743.3) | 118.7 (103.1-126.7) |
| RTX 2080 Ti | Vulkan | calibrated | 1 | 8583/10496 | 1777.4 (1685.9-1861.5) | 178.0 (175.9-179.0) | 1864.3 (1471.9-2442.3) | 158.0 (130.9-171.6) |
| RTX 2080 Ti | Vulkan | calibrated, `--mtp 0` | 0 | 8712/10240 | 1934.8 (1764.2-2051.2) | 130.8 (130.3-131.0) | 2629.7 (2608.8-2642.7) | 126.3 (126.0-126.7) |
| V100 16 GB | CUDA | uncalibrated | 0 | 10240/10240 | 2194.5 (2053.1-2266.2) | 133.2 (132.4-133.6) | 3199.5 (3197.4-3203.2) | 128.1 (127.7-128.3) |
| V100 16 GB | CUDA | calibrated | 1 | 10496/10496 | 2103.6 (1953.7-2200.6) | 202.6 (201.0-203.4) | 2964.6 (2956.9-2973.9) | 192.5 (191.5-193.0) |
| V100 16 GB | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 2253.3 (2233.9-2275.7) | 133.2 (132.3-133.8) | 3144.9 (3138.7-3154.1) | 128.1 (127.7-128.4) |
| V100 16 GB | Vulkan | uncalibrated | 0 | 10240/10240 | 571.0 (97.2-809.5) | 98.4 (43.1-126.1) | 1682.0 (1679.5-1685.3) | 107.8 (84.0-119.8) |
| V100 16 GB | Vulkan | calibrated | 2 | 10496/10496 | 766.8 (711.0-794.9) | 209.2 (204.3-211.8) | 1597.7 (1596.9-1599.2) | 170.4 (120.0-195.8) |
| V100 16 GB | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 804.6 (796.3-809.8) | 124.9 (124.1-125.4) | 1676.3 (1675.2-1678.0) | 119.7 (119.6-119.9) |
| V100 32 GB | CUDA | uncalibrated | 0 | 10240/10240 | 1978.8 (1527.4-2205.0) | 125.4 (124.4-126.0) | 3164.0 (3159.2-3167.9) | 120.9 (120.6-121.1) |
| V100 32 GB | CUDA | calibrated | 1 | 10496/10496 | 2124.0 (2109.9-2149.0) | 192.3 (191.3-193.0) | 2951.1 (2948.3-2956.0) | 182.6 (181.5-183.2) |
| V100 32 GB | CUDA | calibrated, `--mtp 0` | 0 | 10240/10240 | 2203.4 (2190.7-2217.5) | 125.5 (124.6-126.0) | 3117.6 (3113.5-3123.9) | 120.9 (120.6-121.1) |
| V100 32 GB | Vulkan | uncalibrated | 0 | 10240/10240 | 586.7 (150.8-806.1) | 102.4 (64.5-121.3) | 1666.7 (1664.5-1668.6) | 107.5 (91.1-115.8) |
| V100 32 GB | Vulkan | calibrated | 2 | 10496/10496 | 784.8 (774.4-790.6) | 203.0 (198.6-205.3) | 1584.6 (1579.7-1587.2) | 169.4 (130.5-189.0) |
| V100 32 GB | Vulkan | calibrated, `--mtp 0` | 0 | 10240/10240 | 798.3 (791.1-802.4) | 120.9 (120.0-121.3) | 1664.7 (1662.1-1667.0) | 115.6 (115.6-115.7) |

- The model fits in VRAM everywhere except the RTX 2080 Ti (11 GB), where 8213-8712 experts are resident and the
  expert hit rate is 99.7-99.9% (uncalibrated CUDA: 637 misses, 1878 cache swaps; calibrated `--mtp 0`: 231 and 9).
- As on the other cluster, calibrated with `--mtp 0` decodes at the uncalibrated CUDA speed (within 0.5%; 2.3% on the
  RTX 2080 Ti, where the uncalibrated run starts cold: 637 misses against 231), and the gain is MTP:
  decode after 512 with MTP on is 1.43-1.85x the `--mtp 0` run with CUDA (RTX 2080 Ti 1.43x, A40 1.85x).
- The V100 Vulkan prefill is about half its CUDA prefill (pp 4096: 1676 against 3145 tok/s on the 16 GB card): no
  cooperative matrices there, so the `_dp` kernels run. Decode is close (120 against 128 tok/s with `--mtp 0`).
- The first Vulkan repetition of the uncalibrated `pp 512` includes pipeline compilation (minima of 97-151 tok/s).

Startup lines:

```
cuda device 0: NVIDIA A100-PCIE-40GB, sm_80
cuda shared memory: opt-in 166912 B per block, attn_decode_gqa*: 2112 B static, dynamic max 73728 B, carveout default
cuda device 0: NVIDIA RTX A6000, sm_86          (the A40 and RTX 4500 Ada, sm_89, print the same shared-memory line)
cuda shared memory: opt-in 101376 B per block, attn_decode_gqa*: 2112 B static, dynamic max 73728 B, carveout default
cuda device 0: Quadro RTX 6000, sm_75 (Turing, m16n8k8)          (same suffix and next line for the GeForce RTX 2080 Ti)
cuda shared memory: opt-in 65536 B per block, attn_decode_gqa*: 2112 B static, dynamic max 63424 B, carveout default
cuda device 0: Tesla V100-SXM2-16GB, sm_70 (Volta, m8n8k4)          (same for the 32 GB card)
cuda shared memory: opt-in 98304 B per block, attn_decode_gqa*: 2112 B static, dynamic max 73728 B, carveout default

vulkan: NVIDIA A100-PCIE-40GB (NVIDIA 615.71.09), cooperative matrix f16 yes int8 yes, conditional rendering yes, maxComputeSharedMemorySize 49152
vulkan: NVIDIA GeForce RTX 2080 Ti (NVIDIA 615.71.09), cooperative matrix f16 yes int8 yes, conditional rendering yes, maxComputeSharedMemorySize 49152
vulkan: Quadro RTX 6000 (NVIDIA 615.71.09), cooperative matrix f16 yes int8 yes, conditional rendering yes, maxComputeSharedMemorySize 49152
vulkan: Tesla V100-SXM2-16GB (NVIDIA 580.95.05), cooperative matrix f16 no int8 no, conditional rendering yes, maxComputeSharedMemorySize 49152
```

`rustingstrata check --backend vulkan`: 96 (kernel, specialization) pairs, largest 49152 bytes (`gemmi`) of 49152
allowed, on the six cards with cooperative matrices; 84 pairs, largest 49152 bytes (`gemmi_dp`), on both V100s.
