#!/bin/bash
# tools/lowbit_int15/sim_job.sh -- lane/lowbit-int15: the cross-check of the
# host oracle and the device against the quality lane's PyTorch simulation
# (contract section 6.6), on ONE box, with its arms that must fail.
#
#   export              bench/lowbit_quality/int15_export.py, ONLY on a box
#                       whose python3 has PyTorch. It writes the vectors to
#                       a scratch file and the job requires that file to be
#                       byte for byte the committed one, so the committed
#                       vectors are shown to be what arith.py computes. On a
#                       box with no PyTorch the committed file is read and
#                       the log says the export DID NOT RUN there.
#   sim                 pixi run check-gemm-int15-sim                  EXPECTED exit 0
#   sim-host-sabotage   pixi run check-gemm-int15-sim-host-sabotage    EXPECTED non-zero, naming check_sim_host_product
#   sim-convert-sabotage pixi run check-gemm-int15-sim-convert-sabotage EXPECTED non-zero, naming check_sim_host_codes
#   sim-device-sabotage pixi run check-gemm-int15-sim-device-sabotage  EXPECTED non-zero, naming check_sim_device_product
#
# MOJOLEARN_INT15_SIM_WRITE=1 makes the export write the committed path
# itself (the first run, before any vectors are committed).
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_int15/$BOX/sim"
VEC="$PWD/gemm/checks/vectors/int15_sim_vectors.q15"
rm -rf "$OUT"
mkdir -p "$OUT" "$(dirname "$VEC")"
export PATH="$HOME/.pixi/bin:$PATH"
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi
red=0
{
    echo "profile=mojolearn.identical.gemm.int15i64.v1"
    echo "box=$BOX"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
    echo "arith_pin=$(cat bench/lowbit_quality/ARITH_PIN)"
} > "$OUT/sim.txt"
sha() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
PYTORCH=${MOJOLEARN_INT15_SIM_PYTHON:-python3}
if "$PYTORCH" -c 'import torch' > /dev/null 2>&1; then
    if [ "${MOJOLEARN_INT15_SIM_WRITE:-0}" = 1 ]; then
        "$PYTORCH" bench/lowbit_quality/int15_export.py --out "$VEC" > "$OUT/export.log" 2>&1
        rc=$?
        echo "export exit=$rc (wrote the committed path)" >> "$OUT/sim.txt"
        [ "$rc" -eq 0 ] || red=1
    else
        "$PYTORCH" bench/lowbit_quality/int15_export.py --out "$OUT/int15_vectors.regenerated.bin" > "$OUT/export.log" 2>&1
        rc=$?
        echo "export exit=$rc" >> "$OUT/sim.txt"
        [ "$rc" -eq 0 ] || red=1
        if [ -f "$VEC" ] && [ -f "$OUT/int15_vectors.regenerated.bin" ] && cmp -s "$VEC" "$OUT/int15_vectors.regenerated.bin"; then
            echo "export: the regenerated vectors are byte for byte the committed ones" >> "$OUT/sim.txt"
        else
            echo "export: the regenerated vectors DIFFER from the committed ones, or one is missing" >> "$OUT/sim.txt"
            red=1
        fi
    fi
    cat "$OUT/export.log"
else
    echo "export DID NOT RUN: python3 on this box has no PyTorch; the committed vectors are read" >> "$OUT/sim.txt"
fi
[ -f "$VEC" ] && echo "vectors_sha256=$(sha "$VEC")" >> "$OUT/sim.txt"

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
        echo "reach: $1 did not fail $2" >> "$OUT/sim.txt"
        red=1
    else
        echo "reach: $1 failed $2, as it must" >> "$OUT/sim.txt"
    fi
}
must_pass() {
    if ! grep -q "^ok $2\$" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not leave $2 passing" >> "$OUT/sim.txt"
        red=1
    fi
}
run sim pass pixi run check-gemm-int15-sim
run sim-host-sabotage fail pixi run check-gemm-int15-sim-host-sabotage
must_name sim-host-sabotage check_sim_host_product
must_pass sim-host-sabotage check_sim_host_codes
run sim-convert-sabotage fail pixi run check-gemm-int15-sim-convert-sabotage
must_name sim-convert-sabotage check_sim_host_codes
must_pass sim-convert-sabotage check_sim_host_product
run sim-device-sabotage fail pixi run check-gemm-int15-sim-device-sabotage
must_name sim-device-sabotage check_sim_device_product
must_pass sim-device-sabotage check_sim_host_product
grep -h "^   DIGEST " "$OUT/sim.log" 2>/dev/null | sed 's/^   //' > "$OUT/digests.tsv"
{
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN every expected outcome held (see status.tsv)"
    else
        echo "verdict=RED an expected outcome did not hold (see status.tsv and the reach lines above)"
    fi
} >> "$OUT/sim.txt"
cat "$OUT/status.tsv" "$OUT/sim.txt"
for f in sim sim-host-sabotage sim-convert-sabotage sim-device-sabotage; do
    echo "== $f (every line that is not a digest; last 40)"
    grep -v -E '^   DIGEST |mbind' "$OUT/$f.log" 2>/dev/null | cut -c1-600 | tail -40
done
echo "== sim digests (the clean run)"
cat "$OUT/digests.tsv"
echo "sim_job: box=$BOX red=$red"
exit "$red"
