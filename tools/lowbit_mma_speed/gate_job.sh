#!/bin/bash
# tools/lowbit_mma_speed/gate_job.sh -- lane/lowbit-mma-speed, one of the
# lane's two gates on one box, with its arms that must fail. Each phase has
# its own exit code and EXPECTED verdict in status.tsv; a later phase runs
# even when an earlier one fails (tools/lowbit_units/chunk_job.sh's form).
#
#   bash tools/lowbit_mma_speed/gate_job.sh quant
#     quant                 pixi run check-quantize-int8-par
#                           EXPECTED exit 0: the parallel quantizer's codes
#                           and exponents are the host's and the reference
#                           device quantizer's, on both blocks.
#     quant-sabotage        ...-sabotage: the tree's first level skipped.
#                           EXPECTED non-zero, naming both gates as FAILED.
#     quant-value-sabotage  ...-value-sabotage: every code's lowest bit
#                           flipped. EXPECTED non-zero, naming both gates.
#   Runs on every column.
#
#   bash tools/lowbit_mma_speed/gate_job.sh unit
#     unit                  pixi run check-gemm-int8-mma-tuned
#                           EXPECTED exit 0: every tuned plan == the
#                           reference unit plan == the flat plan == the host
#                           oracle, quantized fixtures and planted cases.
#     unit-sabotage         ...-sabotage: the staging's padding rule broken.
#                           EXPECTED non-zero, naming the two gates that
#                           hold ragged k as FAILED.
#     unit-value-sabotage   ...-value-sabotage: every stored value flipped.
#                           EXPECTED non-zero, naming the three gates.
#   Runs on a column that has the integer matrix unit (NVIDIA, AMD). On any
#   other it is NOT RUN, which is not a pass.
#
# Writes <results>/<box>/gate_<which>/ and prints the logs' gate lines, so a
# steward's stdout carries them home.
set -u
cd "$(dirname "$0")/../.." || exit 9
WHICH=${1:-}
case "$WHICH" in
    quant|unit) ;;
    *) echo "gate_job.sh quant|unit" >&2; exit 2 ;;
esac
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/${MOJOLEARN_LOWBIT_RESULTS:-bench/results/lowbit_mma_speed}/$BOX/gate_$WHICH"
rm -rf "$OUT"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
. tools/lowbit_mma_speed/box_env.sh
if [ "$WHICH" = unit ] && [ "$(uname -s)" = Darwin ]; then
    echo "gate_job: box=$BOX is an Apple box; it has no integer matrix unit. NOT RUN, which is not a pass." | tee "$OUT/gate.txt"
    exit 3
fi
{
    echo "box=$BOX gate=$WHICH"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # On a patch-synced tree (the shared NVIDIA pods) HEAD is the merge base
    # and the lane's commit lies over it as a patch; on a steward's worktree
    # HEAD is the submitted commit.
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
    box_describe
} > "$OUT/gate.txt"
red=0
run() {
    _name=$1
    _want=$2
    shift 2
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    _held=no
    if [ "$_want" = pass ] && [ "$_code" -eq 0 ]; then _held=yes; fi
    if [ "$_want" = fail ] && [ "$_code" -ne 0 ]; then _held=yes; fi
    printf '%s\t%s\t%s\texpected=%s\theld=%s\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))s" "$_want" "$_held" >> "$OUT/status.tsv"
    [ "$_held" = yes ] || red=1
    return "$_code"
}
must_name() {
    if ! grep -q "GATE FAILED: $2" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not fail $2" >> "$OUT/gate.txt"
        red=1
    fi
}
must_pass() {
    if ! grep -q "^ok $2" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not pass $2" >> "$OUT/gate.txt"
        red=1
    fi
}
if [ "$WHICH" = quant ]; then
    PHASES="quant quant-sabotage quant-value-sabotage"
    run quant pass pixi run check-quantize-int8-par
    must_pass quant check_par_quantizer_matches
    must_pass quant check_par_quantizer_feeds_the_product
    run quant-sabotage fail pixi run check-quantize-int8-par-sabotage
    must_name quant-sabotage check_par_quantizer_matches
    must_name quant-sabotage check_par_quantizer_feeds_the_product
    run quant-value-sabotage fail pixi run check-quantize-int8-par-value-sabotage
    must_name quant-value-sabotage check_par_quantizer_matches
    must_name quant-value-sabotage check_par_quantizer_feeds_the_product
else
    PHASES="unit unit-sabotage unit-value-sabotage"
    run unit pass pixi run check-gemm-int8-mma-tuned
    must_pass unit check_tuned_plans_match_reference_flat_oracle
    must_pass unit check_tuned_planted_worst_cases
    must_pass unit check_tuned_dispatch_is_batch_invariant
    run unit-sabotage fail pixi run check-gemm-int8-mma-tuned-sabotage
    must_name unit-sabotage check_tuned_plans_match_reference_flat_oracle
    must_name unit-sabotage check_tuned_planted_worst_cases
    run unit-value-sabotage fail pixi run check-gemm-int8-mma-tuned-value-sabotage
    must_name unit-value-sabotage check_tuned_plans_match_reference_flat_oracle
    must_name unit-value-sabotage check_tuned_planted_worst_cases
    must_name unit-value-sabotage check_tuned_dispatch_is_batch_invariant
fi
{
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN every expected outcome held (see status.tsv)"
    else
        echo "verdict=RED an expected outcome did not hold (see status.tsv and the reach lines above)"
    fi
} >> "$OUT/gate.txt"
cat "$OUT/status.tsv" "$OUT/gate.txt"
for f in $PHASES; do
    echo "== $f.log"
    # every line but the clean cases' ok lines
    grep -v -E '^   ok ' "$OUT/$f.log" | tail -120
done
echo "gate_job: box=$BOX gate=$WHICH red=$red"
exit "$red"
