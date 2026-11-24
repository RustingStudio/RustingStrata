# Thanks

RustingStrata runs on many machines it was never developed on. That is only possible because people tried it on their
own hardware, told us what broke, came back to test the fix, and often wrote the fix themselves. Thank you all.

If you helped and your name is missing here, that is our mistake. Open an issue and we will add you.

## Testing and reports

- [runningman84](https://github.com/runningman84): tested multi-GPU on two AMD cards (issue #1) and found that AMD
  cards clock down in the gaps of a layer split.
- [charmoky](https://github.com/charmoky): RX 9070 XT on Vulkan/RADV, where decode first ran at 0.8 tok/s
  (issue #3). Came back with a rerun for build after build, all the way to v1.2.4.
- [ksergk](https://github.com/ksergk): pushed prefill to prompts of 128K to 256K tokens and wrote three very careful
  reports: garbage routing ids above 128K (issue #25), the tiled attention path failing around 50K (issue #42), and a
  server that kept taking requests after the GPU was lost (issue #43).
- [fujcam](https://github.com/fujcam): `CUDA_ERROR_ILLEGAL_ADDRESS` on the RTX 5090 (issue #34). Sent three runs and
  a `compute-sanitizer` pass, and confirmed the fix the same day.
- [gitmylo](https://github.com/gitmylo): hit the same error on Windows with another model (issue #34), which told us
  it was not one odd machine, and confirmed the fix on an RTX 5070 Ti.
- [Saul5662](https://github.com/Saul5662): slow decode on the Orca IQ3_XXS file (issue #44). Stayed with it through
  many test rounds, then found the cause and fixed it (see Code).
- [BrightCandle](https://github.com/BrightCandle): a 9070 XT on Windows that reported the wrong shared memory size
  and could not load a model (issue #20), tested on a 9070 XT plus 3080 pair.
- [dezzezer](https://github.com/dezzezer): CUDA never started on Windows because we looked for NVRTC under its Linux
  name (issue #35). Confirmed the fix with RTX 3060 numbers.
- [xDragon249](https://github.com/xDragon249): the Windows build that failed to link (issue #22), reported with
  everything we needed to fix it in one go.
- [pberenyi](https://github.com/pberenyi): the calibration failure with Swift-1.5-Qwen3.8-27B (issue #11).
- [hermeticvm](https://github.com/hermeticvm): RX 6700 XT / RADV report on slow 30K Qwen3.6 prefill (issue #10).
- [SugiKi1996](https://github.com/SugiKi1996): found that Turing (sm_75) was rejected on NVRTC 13.x, with the fix and
  RTX 2080 Ti numbers (issue #17).
- [frederikhors](https://github.com/frederikhors): RTX 3090 numbers next to llama.cpp on Windows (issue #33) and
  questions about 3090 and 5090 support (issue #21).
- [PericoPa](https://github.com/PericoPa): requests with a chunked body were answered with a 400 (issue #48).
- [ShaggyDragon](https://github.com/ShaggyDragon): noticed that a container silently loses its calibration on exit
  and runs slower the next time (issue #52).
- [amaranus](https://github.com/amaranus): with `--mmproj` no expert is loaded on the GPU (issue #39).
- [Fort-fog](https://github.com/Fort-fog): a full write-up of Flint Q8_0 with a DFlash2 draft model on an RX 7800 XT
  at a context of 122880 (issue #55).
- [luckydevil13](https://github.com/luckydevil13): AMD benchmarks and a lot of talk about which models to support.
- [ike287](https://github.com/ike287): feedback and feature requests.

## Code

- [Niko1221](https://github.com/Niko1221/Strata): the HIP backend's device-side compatibility layer (`src/hip/compat.h`)
  follows the structure and several implementations of Strata's `include/strata/hip_compat/intrinsics.hpp` (MIT).

## Ideas

- [FredWalter](https://github.com/FredWalter): asked for multi-GPU and Qwen3.8-Flash-Next (issue #1), and later
  whether the split should follow the speed of each card and its PCIe link (issues #40, #41). `calibrate` now
  measures the split because of that.
- [Darksecond](https://github.com/Darksecond): four requests from a two-GPU machine, all of them in now: `calibrate`
  with several devices (issue #23), timing data that llama-swap can read (issue #31), a `--vram-margin` per device
  (issue #32) and MTP on a split model (issue #38).
- [luckydevil13](https://github.com/luckydevil13): DeepSeek support (issue #12).
- [beyondzvincent](https://github.com/beyondzvincent): Gemma 4 26B A4B (issue #19), and
  [Anc813](https://github.com/Anc813) for the tip about its QAT file.
- [david-boles](https://github.com/david-boles): image input (issue #29).
- [JackieJK](https://github.com/JackieJK): TQ1_0 and TQ2_0 files (issue #30).
- [Gornist](https://github.com/Gornist): a thinking budget and sampling penalties, from running coding agents on
  `serve` (issue #45).
- [OneOfOne](https://github.com/OneOfOne): reading the model into RAM at start (issue #49), with the numbers that
  showed how much a cold start cost us.
- [ShaggyDragon](https://github.com/ShaggyDragon): keeping the most used experts on a single GPU, with measured hit
  rates (issue #53).
- [rocktboy1982](https://github.com/rocktboy1982): shared a collection of research notes on single-card speed
  (issue #51).
- [Ocean572](https://github.com/Ocean572): configurable allowed-host setting for LAN access (issue #13).
- [Bush-cat](https://github.com/Bush-cat): a Docker image (issue #58).
- [micahlt](https://github.com/micahlt): unloading the model when the server is idle (issue #59).
- [KotPrezesOfficial](https://github.com/KotPrezesOfficial): MCP support and GUI options (issue #18).
- [outbackdingo](https://github.com/outbackdingo): cuda-oxide kernels (issue #4).

## Code

- [TheBrutalX](https://github.com/TheBrutalX): the whole Tesla V100 port, more than ninety commits: CUDA on Volta
  (sm_70), the Vulkan shared-memory check, and the first `SUPPORTED_GPUS.md`.
- [ybenyedder](https://github.com/ybenyedder): EAGLE3 speculative decoding and the N-gram cascade (PR #36), a
  rounding fix for AMD wave32 cards (PR #37), and DFlash, FlashCascade, SenseNova-Loop, external draft models and
  the q4/q2 KV cache (PR #50, merged as #56).
- [yugt](https://github.com/yugt): ran RustingStrata on a cluster of data-center cards and wrote down every result:
  A100, A30, L40S and H100 (PR #24, #26), eight more cards (PR #27), and a clear error for cards older than Volta
  (PR #28).
- [bettercallcaleb](https://github.com/bettercallcaleb): GGUF loader rejects malformed metadata instead of panicking
  (PR #2), hiding incomplete GGUF shard sets from the model list (PR #8) and prequantizing the DeltaNet gate before
  batched GEMM (PR #9).
- [jazir555](https://github.com/jazir555): runtime LoRA abliteration without modifying base weights (PR #15) and the
  host RAM cap (PR #16).
- [Saul5662](https://github.com/Saul5662): integer IQ4_NL dot product for the CPU experts (PR #47), which fixed the
  slow decode they had reported.
- [dinukasal](https://github.com/dinukasal): presence and frequency penalties and DRY, so replies stop looping
  (PR #54).
- [x-Pepsy](https://github.com/x-Pepsy): the Turing (sm_75) CUDA path (PR #6).
- [masanork](https://github.com/masanork): Vulkan wave64 for the Radeon VII (PR #5).
- [pberenyi](https://github.com/pberenyi): server improvements for `strata serve` (PR #14).
- [FFAX-max](https://github.com/FFAX-max): documented how to keep the calibration in a container (PR #57).

## Packaging

- [Crandel](https://github.com/Crandel): made and maintains the AUR package
  [rustingstrata-bin](https://aur.archlinux.org/packages/rustingstrata-bin) (issue #46).

## Reddit

- [gcavalcante8808](https://github.com/gcavalcante8808): the first helper ever. Ran many benchmarks and wrote reports
  on AMD GPUs, a very big contribution!
