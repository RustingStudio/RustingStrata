#!/usr/bin/env bash
# Comparatif rigoureux : Base vs N-gram vs Standalone MTP (Route C WP11)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_DIR}/target/release/rustingstrata"
BLOB="/var/lib/ollama/blobs/sha256-0dae47840f8115d894e74c5cd1a6aca27e01feaa4f07890a50bdcf7db4c56227"
MTP_MODEL="/home/cvsbd/models-local/mtp-Q8_0.gguf"
WORK_DIR="/tmp/opencode/wp11_bench"
mkdir -p "${WORK_DIR}"

if systemctl --user is-active --quiet qwen-ngram 2>/dev/null; then systemctl --user stop qwen-ngram; fi
if systemctl --user is-active --quiet strata-route-c 2>/dev/null; then systemctl --user stop strata-route-c; fi

python3 - "${WORK_DIR}" <<'PY'
import sys, pathlib
w = pathlib.Path(sys.argv[1])
novel = (
    "<|im_start|>user\n"
    "Write a detailed technical essay on the physics of extreme ultraviolet lithography (EUV), "
    "including laser-produced tin plasma, Mo/Si Bragg reflectors, stochastic photon shot noise, "
    "and high-NA anamorphic optics.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
(w / "novel.txt").write_text(novel)

snippet = (
    "def process_batch(items, workers=4):\n"
    "    results = []\n"
    "    for chunk in chunks(items, workers):\n"
    "        for item in chunk:\n"
    "            if item is None: continue\n"
    "            try: results.append(transform(item))\n"
    "            except ValueError as e: log.warning('skip %s', e)\n"
)
raw_copy = snippet * 30
chat_copy = (
    "<|im_start|>user\n"
    "Reproduce the following Python block verbatim without any modification:\n\n"
    + raw_copy +
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
(w / "copy.txt").write_text(chat_copy)
PY

bench_run() {
    local label="$1"
    local prompt_file="$2"
    local ntoks="$3"
    shift 3
    local out_err="${WORK_DIR}/${label}.err"
    local out_txt="${WORK_DIR}/${label}.out"

    "${BIN}" generate --model "${BLOB}" --gpu --backend vulkan --ctx 4096 --kv f16 \
        -n "${ntoks}" --temp 0 --seed 42 "$@" \
        --prompt - < "${prompt_file}" \
        > "${out_txt}" 2> "${out_err}"

    local spd=$(grep -E "gen [0-9]+ tok [0-9.]+ tok/s" "${out_err}" | sed -E 's/.*gen [0-9]+ tok ([0-9.]+) tok\/s.*/\1/' | head -n1)
    local draft_info=$(grep -E "(speculative|MTP drafts accepted)" "${out_err}" | head -n1 || echo "none")
    printf "%-25s => %6s tok/s | %s\n" "${label}" "${spd}" "${draft_info}"
}

echo "=== 1. TEXTE NEUF TECHNIQUE (128 TOKENS) ==="
bench_run "novel_base_none" "${WORK_DIR}/novel.txt" 128 --spec-type none
bench_run "novel_ngram_simple" "${WORK_DIR}/novel.txt" 128 --spec-type ngram-simple --spec-draft-n-max 16
bench_run "novel_mtp_k1" "${WORK_DIR}/novel.txt" 128 --mtp-model "${MTP_MODEL}" --mtp 1
bench_run "novel_mtp_k2" "${WORK_DIR}/novel.txt" 128 --mtp-model "${MTP_MODEL}" --mtp 2

echo ""
echo "=== 2. COPIE / RÉÉCRITURE DE CODE (400 TOKENS) ==="
bench_run "copy_base_none" "${WORK_DIR}/copy.txt" 400 --spec-type none
bench_run "copy_ngram_simple" "${WORK_DIR}/copy.txt" 400 --spec-type ngram-simple --spec-draft-n-max 16
bench_run "copy_mtp_k1" "${WORK_DIR}/copy.txt" 400 --mtp-model "${MTP_MODEL}" --mtp 1
bench_run "copy_mtp_k2" "${WORK_DIR}/copy.txt" 400 --mtp-model "${MTP_MODEL}" --mtp 2
