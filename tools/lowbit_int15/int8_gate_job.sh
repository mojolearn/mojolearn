#!/bin/bash
# tools/lowbit_int15/int8_gate_job.sh -- lane/lowbit-int15: the EXISTING
# low-bit gate (gemm/checks/gemm_lowbit_check.mojo: bf16f32.v1 and
# int8i32.v1) on ONE box, because this lane stated the alignment of the
# matrix-unit fragment loads in gemm/checks/gemm_int8_mma.mojo (DEVIATION
# 2975) and the int8 profile runs through them. tools/lowbit_mma_leg.sh's
# four phases and expectations, in the tree the job starts in, and one more:
#
#   lowbit                  pixi run check-gemm-lowbit                 EXPECTED exit 0
#   lowbit-force-flat       pixi run check-gemm-lowbit-force-flat      EXPECTED exit 0
#   lowbit-unstated-loads   pixi run check-gemm-lowbit-unstated-loads  EXPECTED exit 0, the clean run's cells
#   lowbit-sabotage         pixi run check-gemm-lowbit-sabotage        EXPECTED non-zero
#   lowbit-host-sabotage    pixi run check-gemm-lowbit-host-sabotage   EXPECTED non-zero
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_int15/$BOX/int8_gate"
rm -rf "$OUT"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
UNIT=yes
if [ "$(uname -s)" = Darwin ]; then
    UNIT=no
elif command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi
{
    echo "profiles=mojolearn.identical.gemm.bf16f32.v1, mojolearn.identical.gemm.int8i32.v1"
    echo "box=$BOX integer_matrix_unit=$UNIT"
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
    if [ "$UNIT" = no ] && [ "$2" = check_int8_mma_matches_flat ]; then
        echo "reach: $1 cannot name $2 on this box (the gate does not run there)" >> "$OUT/gate.txt"
        return 0
    fi
    if ! grep -q "GATE FAILED: $2" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not fail $2" >> "$OUT/gate.txt"
        red=1
    else
        echo "reach: $1 failed $2, as it must" >> "$OUT/gate.txt"
    fi
}
cells() { grep -E '^   ok .*c\[0\]=' "$OUT/$1.log" 2>/dev/null; }
run lowbit pass pixi run check-gemm-lowbit
run lowbit-force-flat pass pixi run check-gemm-lowbit-force-flat
run lowbit-unstated-loads pass pixi run check-gemm-lowbit-unstated-loads
cells lowbit > "$OUT/cells.txt"
cells lowbit-unstated-loads > "$OUT/cells_unstated_loads.txt"
cells lowbit-force-flat > "$OUT/cells_force_flat.txt"
if [ -s "$OUT/cells.txt" ] && cmp -s "$OUT/cells.txt" "$OUT/cells_unstated_loads.txt" && cmp -s "$OUT/cells.txt" "$OUT/cells_force_flat.txt"; then
    echo "loads: stated alignment, unstated alignment and the flat plan printed the same $(grep -c . "$OUT/cells.txt") cell lines" >> "$OUT/gate.txt"
else
    echo "loads: the three runs' cell lines DIFFER, or a run printed none" >> "$OUT/gate.txt"
    red=1
fi
run lowbit-sabotage fail pixi run check-gemm-lowbit-sabotage
must_name lowbit-sabotage check_int8_device_matches_oracle
must_name lowbit-sabotage check_int8_mma_matches_flat
must_name lowbit-sabotage check_bf16_device_matches_oracle
run lowbit-host-sabotage fail pixi run check-gemm-lowbit-host-sabotage
must_name lowbit-host-sabotage check_int8_device_matches_oracle
must_name lowbit-host-sabotage check_int8_mma_matches_flat
must_name lowbit-host-sabotage check_bf16_device_matches_oracle
grep -h "int8 dispatch:" "$OUT/lowbit.log" "$OUT/lowbit-force-flat.log" 2>/dev/null >> "$OUT/gate.txt"
{
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN every expected outcome held (see status.tsv)"
    else
        echo "verdict=RED an expected outcome did not hold (see status.tsv and the reach lines above)"
    fi
} >> "$OUT/gate.txt"
cat "$OUT/status.tsv" "$OUT/gate.txt"
for f in lowbit lowbit-force-flat lowbit-unstated-loads lowbit-sabotage lowbit-host-sabotage; do
    echo "== $f (the gate lines; last 40)"
    grep -E '^(ok|!!|==) |error' "$OUT/$f.log" 2>/dev/null | cut -c1-400 | tail -40
done
echo "int8_gate_job: box=$BOX red=$red"
exit "$red"
