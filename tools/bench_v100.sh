#!/bin/sh
# Benchmark on the Tesla V100 (32 GB). Usage: tools/bench_v100.sh <model.gguf> [backend] [out.md]
# Appends one timestamped block to docs/V100-results.md (or $3) so the numbers of every run stay together.
set -e
M=${1:?usage: tools/bench_v100.sh <model.gguf> [cuda|vulkan] [out.md]}
B=${2:-vulkan}
O=${3:-docs/V100-results.md}
BIN=${STRATA_BIN:-target/release/strata}
CTX=${STRATA_CTX:-32896}

test -x "$BIN" || { echo "build it first: cargo build --release" >&2; exit 1; }
test -f "$M" || { echo "no such model: $M" >&2; exit 1; }

{
    echo "## $(date -u '+%Y-%m-%d %H:%M UTC') — $(basename "$M")"
    echo
    echo '```'
    echo "\$ strata bench --backend $B --pp 512,4096,32768 --tg 128 --reps 3 --ctx $CTX"
    "$BIN" bench --model "$M" --gpu --backend "$B" --pp 512,4096,32768 --tg 128 --reps 3 --ctx "$CTX" 2>&1
    echo '```'
    echo
} | tee -a "$O"
