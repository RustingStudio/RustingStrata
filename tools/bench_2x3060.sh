#!/usr/bin/env bash
# Inkling 2 on two RTX 3060s: prompt lengths at a 32k and a 200k context, greedy decode, one repetition each.
#   tools/bench_2x3060.sh [--model PATH] [--out STEM] [--pp LIST] [--ctx LIST] [--tg N]
# Prints one table row per (ctx, pp); with --out writes STEM-<ctx>-<pp>.json (`strata.bench.v1`) per row.
#
# The load line is printed under each row: how many expert layers are in VRAM and where decides most of the speed,
# and it depends on the free VRAM at that moment (a desktop on GPU 0 costs layers). A run whose line differs from
# the one in docs/BENCHMARKS.md is not comparable to it.
set -euo pipefail
cd "$(dirname "$0")/.."
model=${STRATA_INKLING2_MODEL:-/home/andre/strata/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ4_XS.gguf}
out=""
pp_list="512,2048,4096"
ctx_list="32768 200000"
tg=32
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --pp) pp_list=$2; shift 2 ;;
    --ctx) ctx_list=$2; shift 2 ;;
    --tg) tg=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
bin=${BIN:-target/release/rustingstrata}
# `bench` turns the plain-weights path on itself (`gpu_defaults` sets STRATA_DS_PLAIN_FIRST=1 when it is unset), so
# the environment is passed through: set STRATA_DS_PLAIN_FIRST=0 here to measure the other path (mat.rs reads "0" as off).
# The runtime CUDA libraries the driver ships (cu13 here) come first: an older /usr/local/cuda libnvrtc compiles
# the kernels and the forward falls back to the host.
export LD_LIBRARY_PATH=$HOME/nvrtc13/lib/python3.11/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ -f "$bin" ] || { echo "no $bin (cargo build --release)" >&2; exit 1; }
printf '%-8s %-10s %10s %10s\n' ctx pp prefill decode
fail=0
for ctx in $ctx_list; do
  for pp in ${pp_list//,/ }; do
    log=$(mktemp)
    # A fresh process per row: the prefill scratch and the KV are sized by the context, and a second prompt in the
    # same process reuses them (a different measurement).
    if [ -n "$out" ]; then
      if ! env STRATA_DS_KV_TOKENS="$ctx" STRATA_DS_PF_TOKENS=512 \
        "$bin" bench --model "$model" --pp "$pp" --tg "$tg" --reps 1 --ctx "$ctx" --device 0,1 --gpu \
        --json-out "$out-$ctx-$pp.json" >"$log" 2>&1; then
        echo "ctx $ctx pp $pp: no result"; sed -n '1,20p' "$log"; rm -f "$log"; fail=$((fail+1)); continue
      fi
    else
      if ! env STRATA_DS_KV_TOKENS="$ctx" STRATA_DS_PF_TOKENS=512 \
        "$bin" bench --model "$model" --pp "$pp" --tg "$tg" --reps 1 --ctx "$ctx" --device 0,1 --gpu \
        >"$log" 2>&1; then
        echo "ctx $ctx pp $pp: no result"; sed -n '1,20p' "$log"; rm -f "$log"; fail=$((fail+1)); continue
      fi
    fi
    row=$(grep -E '^pp ' "$log" | tail -1 || true)
    if [ -z "$row" ]; then
      echo "ctx $ctx pp $pp: no result"; sed -n '1,20p' "$log"; rm -f "$log"; fail=$((fail+1)); continue
    fi
    # `pp  512: prefill V (...) tok/s | tg 32: decode D (...) tok/s`.
    printf '%-8s %-10s %10s %10s\n' "$ctx" "$pp" "$(awk '{print $4}' <<<"$row")" "$(awk '{print $11}' <<<"$row")"
    grep -E '^inkling2: preloaded' "$log" | sed 's/^inkling2: /  /' || true
    rm -f "$log"
  done
done
[ "$fail" -eq 0 ] || exit 1
