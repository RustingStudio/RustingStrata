#!/usr/bin/env bash
# Grid Search systématique des hyperparamètres spéculatifs N-gram pour Radeon 890M (Route C WP10)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_DIR}/target/release/rustingstrata"
BLOB="${BLOB:-/var/lib/ollama/blobs/sha256-0dae47840f8115d894e74c5cd1a6aca27e01feaa4f07890a50bdcf7db4c56227}"
WORK_DIR="/tmp/opencode/wp10_grid"
mkdir -p "${WORK_DIR}"

if systemctl --user is-active --quiet qwen-ngram 2>/dev/null; then
    systemctl --user stop qwen-ngram
fi
if systemctl --user is-active --quiet strata-route-c 2>/dev/null; then
    systemctl --user stop strata-route-c
fi

echo "================================================================================"
echo " WP10 : GRID SEARCH N-GRAM SPECULATIVE (Radeon 890M / gfx1150)"
echo " Modèle : ${BLOB}"
echo "================================================================================"

RESULTS_FILE="${WORK_DIR}/results.csv"
echo "test_type,spec_type,draft_max,n,m,min_hits,gen_tok_s,acc_pct,acc_toks" > "${RESULTS_FILE}"

run_bench() {
    local task="$1"
    local stype="$2"
    local dmax="$3"
    local nval="$4"
    local mval="$5"
    local mhits="$6"
    local ntoks="$7"
    local prompt_file="$8"

    local out_err="${WORK_DIR}/run_${task}_${stype}_d${dmax}_n${nval}_m${mval}_h${mhits}.err"

    "${BIN}" generate --model "${BLOB}" --gpu --backend vulkan --ctx 4096 --kv f16 \
        -n "${ntoks}" --temp 0 --seed 42 \
        --spec-type "${stype}" \
        --spec-draft-n-max "${dmax}" \
        --spec-ngram-size-n "${nval}" \
        --spec-ngram-size-m "${mval}" \
        --spec-ngram-min-hits "${mhits}" \
        --prompt - < "${prompt_file}" \
        > /dev/null 2> "${out_err}"

    local spd=$(grep -E "gen [0-9]+ tok [0-9.]+ tok/s" "${out_err}" | sed -E 's/.*gen [0-9]+ tok ([0-9.]+) tok\/s.*/\1/' | head -n1)
    if [[ -z "${spd}" ]]; then spd="0.0"; fi

    local acc_pct=$(grep "accepted " "${out_err}" | sed -E 's/.*accepted ([0-9.]+)%.*/\1/' | head -n1)
    if [[ -z "${acc_pct}" ]]; then acc_pct="0.0"; fi

    local acc_toks=$(grep "speculative: " "${out_err}" | sed -E 's/.*speculative: ([0-9]+\/[0-9]+).*/\1/' | head -n1)
    if [[ -z "${acc_toks}" ]]; then acc_toks="0/0"; fi

    echo "${task},${stype},${dmax},${nval},${mval},${mhits},${spd},${acc_pct},${acc_toks}" >> "${RESULTS_FILE}"
    printf "%-8s | %-13s | dmax=%2d | N=%2d | M=%2d | hits=%d => %5s tok/s (acc: %5s%%, drafts: %s)\n" \
        "${task}" "${stype}" "${dmax}" "${nval}" "${mval}" "${mhits}" "${spd}" "${acc_pct}" "${acc_toks}"
}

echo "=== TEST 1 : BASELINE (spec-type none) ==="
run_bench "novel" "none" 16 12 48 1 128 "${WORK_DIR}/novel_prompt.txt"
run_bench "copy"  "none" 16 12 48 1 400 "${WORK_DIR}/repro_chat.txt"

echo ""
echo "=== TEST 2 : IMPACT DRAFT_MAX (8, 12, 16) sur ngram-simple ==="
for d in 8 12 16; do
    run_bench "copy" "ngram-simple" "${d}" 12 48 1 400 "${WORK_DIR}/repro_chat.txt"
done

echo ""
echo "=== TEST 3 : IMPACT TYPE D'ALGORITHME N-GRAM (simple vs map-k vs map-k4v vs mod) ==="
for t in ngram-simple ngram-map-k ngram-map-k4v ngram-mod; do
    run_bench "copy" "${t}" 16 12 48 1 400 "${WORK_DIR}/repro_chat.txt"
done

echo ""
echo "=== TEST 4 : IMPACT DE LA TAILLE DE CLÉ N (6, 8, 12, 16) ==="
for n in 6 8 12 16; do
    run_bench "copy" "ngram-simple" 16 "${n}" 48 1 400 "${WORK_DIR}/repro_chat.txt"
done

echo ""
echo "=== TEST 5 : IMPACT DE LA FENÊTRE M (24, 48, 64) ==="
for m in 24 48 64; do
    run_bench "copy" "ngram-simple" 16 12 "${m}" 1 400 "${WORK_DIR}/repro_chat.txt"
done

echo ""
echo "=== TEST 6 : IMPACT SUR TEXTE NEUF (Vérification zéro-régression) ==="
for t in ngram-simple ngram-map-k; do
    run_bench "novel" "${t}" 16 12 48 1 128 "${WORK_DIR}/novel_prompt.txt"
done

echo ""
echo "Grid search terminé. Résultats dans ${RESULTS_FILE}"
