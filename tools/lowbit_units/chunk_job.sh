#!/bin/bash
# tools/lowbit_units/chunk_job.sh -- lane/lowbit-units, the Apple
# exact-chunk probe's gate on an Apple box, with its two arms that must fail.
# Each phase has its own exit code and EXPECTED verdict in status.tsv; a
# later phase runs even when an earlier one fails (tools/lowbit_mma_leg.sh's
# form).
#
#   chunk                 pixi run check-gemm-int8-apple-chunk
#                         EXPECTED exit 0: bits equal to the flat int8
#                         kernel and the host oracle on the quantized
#                         fixtures, both geometries and every planted case.
#   chunk-sabotage        pixi run check-gemm-int8-apple-chunk-sabotage
#                         The chunk boundary removed. EXPECTED non-zero, and
#                         the log must name
#                         check_apple_chunk_planted_worst_cases as FAILED.
#   chunk-value-sabotage  pixi run check-gemm-int8-apple-chunk-value-sabotage
#                         Every stored value flipped. EXPECTED non-zero, and
#                         the log must name both oracle gates as FAILED.
#
# Writes bench/results/lowbit_units/<box>/chunk/ and prints the logs' gate
# lines, so a steward's stdout carries them home.
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_units/$BOX/chunk"
rm -rf "$OUT"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
if [ "$(uname -s)" != Darwin ]; then
    echo "chunk_job: box=$BOX is not an Apple box; the probe runs on Metal only. NOT RUN, which is not a pass." | tee "$OUT/gate.txt"
    exit 3
fi
{
    echo "box=$BOX"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # On a patch-synced tree (the shared NVIDIA pods) HEAD is the merge base
    # and the lane's commit lies over it as a patch; on a steward's worktree
    # HEAD is the submitted commit.
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
    echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) macOS $(sw_vers -productVersion 2>/dev/null)"
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
run chunk pass pixi run check-gemm-int8-apple-chunk
run chunk-sabotage fail pixi run check-gemm-int8-apple-chunk-sabotage
must_name chunk-sabotage check_apple_chunk_planted_worst_cases
# The bound is host arithmetic and the sabotage does not touch it.
must_pass chunk-sabotage check_chunk_bound_is_exact_and_tight
run chunk-value-sabotage fail pixi run check-gemm-int8-apple-chunk-value-sabotage
must_name chunk-value-sabotage check_apple_chunk_matches_flat_and_oracle
must_name chunk-value-sabotage check_apple_chunk_planted_worst_cases
{
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN every expected outcome held (see status.tsv)"
    else
        echo "verdict=RED an expected outcome did not hold (see status.tsv and the reach lines above)"
    fi
} >> "$OUT/gate.txt"
cat "$OUT/status.tsv" "$OUT/gate.txt"
for f in chunk chunk-sabotage chunk-value-sabotage; do
    echo "== $f.log"
    # every line but the per-case ok lines of the clean run's long lists
    grep -v -E '^   ok int8 chunk planted' "$OUT/$f.log" | tail -150
done
echo "chunk_job: box=$BOX red=$red"
exit "$red"
