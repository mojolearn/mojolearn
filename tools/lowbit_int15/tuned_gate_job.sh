#!/bin/bash
# tools/lowbit_int15/tuned_gate_job.sh -- lane/lowbit-int15: the gate of the
# TUNED plan of mojolearn.identical.gemm.int15i64.v1 (contract W-13) on ONE
# box that has the integer matrix unit, with its four arms that must fail.
# Before it, lane/lowbit-mma-speed's own gate of the sums kernel, because
# this lane is the first to build that kernel from this branch.
#
#   pieces                     pixi run check-gemm-int8-pieces-tuned            EXPECTED exit 0
#   tuned                      pixi run check-gemm-int15-tuned                  EXPECTED exit 0
#   tuned-sabotage             ... -tuned-sabotage           EXPECTED non-zero (every stored cell flipped)
#   tuned-pieces-sabotage      ... -tuned-pieces-sabotage    EXPECTED non-zero (the sums kernel's middle sum takes HL twice)
#   tuned-epilogue-sabotage    ... -tuned-epilogue-sabotage  EXPECTED non-zero (the epilogue reads the wrong sum)
#   tuned-host-sabotage        ... -tuned-host-sabotage      EXPECTED non-zero (every oracle cell flipped)
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_int15/$BOX/tuned_gate"
rm -rf "$OUT"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
if [ "$(uname -s)" = Darwin ]; then
    echo "tuned_gate_job: box=$BOX is an Apple box; the tuned plan runs on the integer matrix units only. NOT RUN, which is not a pass." | tee "$OUT/gate.txt"
    exit 3
fi
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi
{
    echo "profile=mojolearn.identical.gemm.int15i64.v1 plan=tuned"
    echo "box=$BOX"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
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
    if [ "$_want" = fail ] && [ "$_code" -ne 0 ] && grep -q -E '^== [0-9]+ gates, [1-9][0-9]* failed ==$' "$OUT/$_name.log"; then _held=yes; fi
    printf '%s\t%s\t%s\texpected=%s\theld=%s\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))s" "$_want" "$_held" >> "$OUT/status.tsv"
    [ "$_held" = yes ] || red=1
    return "$_code"
}
must_name() {
    if ! grep -q "GATE FAILED: $2" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not fail $2" >> "$OUT/gate.txt"
        red=1
    else
        echo "reach: $1 failed $2, as it must" >> "$OUT/gate.txt"
    fi
}
must_pass() {
    if ! grep -q "^ok $2\$" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not leave $2 passing" >> "$OUT/gate.txt"
        red=1
    fi
}
run pieces pass pixi run check-gemm-int8-pieces-tuned
run tuned pass pixi run check-gemm-int15-tuned
for arm in sabotage pieces-sabotage epilogue-sabotage host-sabotage; do
    run "tuned-$arm" fail pixi run "check-gemm-int15-tuned-$arm"
    must_name "tuned-$arm" check_int15_tuned_matches_oracle
    must_name "tuned-$arm" check_int15_tuned_planted_worst_cases
    must_pass "tuned-$arm" check_int15_tuned_refuses
done
# The epilogue fold's own arm: the column exponent read at the row index.
# The planted cases give every row one exponent, so only the shapes gate can
# see it; the planted gate's outcome is recorded, not required.
run tuned-exponent-sabotage fail pixi run check-gemm-int15-tuned-exponent-sabotage
must_name tuned-exponent-sabotage check_int15_tuned_matches_oracle
must_pass tuned-exponent-sabotage check_int15_tuned_refuses
grep -h "^   fused path: " "$OUT/tuned.log" 2>/dev/null | head -1 >> "$OUT/gate.txt"
echo "fused kernel's store: $(grep -E '^from gemm\.checks\.gemm_(int15_epilogue|int8_pieces_epilogue_stub) import int15_store_cell' gemm/checks/gemm_int8_mma_tuned.mojo || echo 'NO int15_store_cell import found')" >> "$OUT/gate.txt"
grep -h "^   DIGEST " "$OUT/tuned.log" 2>/dev/null | sed 's/^   //' > "$OUT/digests.tsv"
{
    echo "digests=$(grep -c . "$OUT/digests.tsv")"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN every expected outcome held (see status.tsv)"
    else
        echo "verdict=RED an expected outcome did not hold (see status.tsv and the reach lines above)"
    fi
} >> "$OUT/gate.txt"
cat "$OUT/status.tsv" "$OUT/gate.txt"
for f in pieces tuned tuned-sabotage tuned-pieces-sabotage tuned-epilogue-sabotage tuned-host-sabotage tuned-exponent-sabotage; do
    echo "== $f (every line that is not a digest or a per-shape ok; last 40)"
    grep -v -E '^   (DIGEST|ok) |mbind' "$OUT/$f.log" 2>/dev/null | cut -c1-500 | tail -40
done
echo "== tuned gate digests (the clean run)"
cat "$OUT/digests.tsv"
echo "tuned_gate_job: box=$BOX red=$red"
exit "$red"
