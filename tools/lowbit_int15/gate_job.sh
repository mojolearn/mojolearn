#!/bin/bash
# tools/lowbit_int15/gate_job.sh -- lane/lowbit-int15: the gate of
# mojolearn.identical.gemm.int15i64.v1 on ONE box, with its arms that must
# fail. Each phase has its own exit code and EXPECTED verdict in status.tsv;
# a later phase runs even when an earlier one fails, because a red phase is
# a finding (tools/lowbit_mma_leg.sh's form).
#
#   int15                 pixi run check-gemm-int15
#                         EXPECTED exit 0. On a column that has an integer
#                         matrix unit (NVIDIA, AMD) the dispatchers take the
#                         MMA plan and check_int15_plans_agree runs flat,
#                         pieces and mma; on Apple flat and pieces.
#   int15-force-flat      pixi run check-gemm-int15-force-flat
#                         The dispatchers pinned off the unit. EXPECTED exit 0.
#   int15-unstated-loads  pixi run check-gemm-int15-unstated-loads
#                         The matrix-unit fragment loads with no alignment
#                         stated (DEVIATION 2975). EXPECTED exit 0 and the
#                         clean run's digests, line for line.
#   int15-sabotage        pixi run check-gemm-int15-sabotage
#                         THE DEVICE ARM: every stored cell flipped. EXPECTED
#                         non-zero, naming the three device oracle gates.
#   int15-host-sabotage   pixi run check-gemm-int15-host-sabotage
#                         THE HOST ARM: every oracle cell flipped. EXPECTED
#                         non-zero, naming the host pieces gate and the three
#                         device oracle gates.
#   int15-piece-sabotage  pixi run check-gemm-int15-piece-sabotage
#                         THE DEFECT ARM: the split writes -127 for a high
#                         piece of -128. EXPECTED non-zero, naming the
#                         conversions gate and the planted worst cases.
#   int15-quant-sabotage  pixi run check-gemm-int15-quant-sabotage
#                         THE DEFECT ARM of the parallel quantizer: the last
#                         chunk of a row never enters its absmax. EXPECTED
#                         non-zero, naming the conversions gate.
#
# Runs from the tree the queue or the steward starts it in; writes
# bench/results/lowbit_int15/<box>/gate/ and prints the gate lines and the
# digests, so a steward's stdout carries them home.
#
#   MOJOLEARN_LOWBIT_BOX=h100 bash tools/lowbit_int15/gate_job.sh
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_int15/$BOX/gate"
rm -rf "$OUT"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
if [ "$(uname -s)" = Darwin ]; then
    VENDOR=apple
elif command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    VENDOR=nvidia
    # tools/lowbit_mma_leg.sh's lines: an older driver uses its own assembler.
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
elif [ -e /dev/kfd ] || command -v rocm-smi > /dev/null 2>&1 || command -v amd-smi > /dev/null 2>&1; then
    VENDOR=amd
else
    VENDOR=unknown
fi
{
    echo "profile=mojolearn.identical.gemm.int15i64.v1"
    echo "box=$BOX vendor=$VENDOR gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset}"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # On a patch-synced tree (the shared NVIDIA pods) HEAD is the merge base
    # and the lane's commit lies over it as a patch; on a steward's worktree
    # HEAD is the submitted commit.
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
    if [ "$VENDOR" = apple ]; then
        echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) macOS $(sw_vers -productVersion 2>/dev/null)"
    elif [ "$VENDOR" = nvidia ]; then
        echo "machine=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    fi
} > "$OUT/gate.txt"
if [ "$VENDOR" = unknown ]; then
    echo "vendor=unknown: not a Mac, no working nvidia-smi, no /dev/kfd; NOTHING RUN, which is not a pass" | tee -a "$OUT/gate.txt"
    exit 9
fi
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
    # An arm HELD only when the program ran to its verdict line and failed
    # there: a build that does not compile also exits non-zero, and that is
    # not a sabotage seen failing (job nvc3-0012).
    if [ "$_want" = fail ] && [ "$_code" -ne 0 ] && grep -q -E '^== [0-9]+ gates, [1-9][0-9]* failed ==$' "$OUT/$_name.log"; then _held=yes; fi
    printf '%s\t%s\t%s\texpected=%s\theld=%s\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))s" "$_want" "$_held" >> "$OUT/status.tsv"
    [ "$_held" = yes ] || red=1
    return "$_code"
}
must_name() {
    # must_name <log name> <gate>: the sabotage log must list the gate as
    # FAILED; a sabotage that does not reach a gate is a finding.
    if ! grep -q "GATE FAILED: $2" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not fail $2" >> "$OUT/gate.txt"
        red=1
    else
        echo "reach: $1 failed $2, as it must" >> "$OUT/gate.txt"
    fi
}
must_pass() {
    # must_pass <log name> <gate>: an arm that does not reach a gate must
    # leave it passing; a gate that fails under every arm says nothing.
    if ! grep -q "^ok $2\$" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not leave $2 passing" >> "$OUT/gate.txt"
        red=1
    fi
}

run int15 pass pixi run check-gemm-int15
run int15-force-flat pass pixi run check-gemm-int15-force-flat
# DEVIATION 2975: the unit plan with the fragment loads as they were before
# their alignment was stated. Same gates, and the SAME DIGESTS as the clean
# run, which is what says the stated alignment moved no bit.
run int15-unstated-loads pass pixi run check-gemm-int15-unstated-loads
grep -h "^   DIGEST " "$OUT/int15-unstated-loads.log" 2>/dev/null | sed 's/^   //' > "$OUT/digests_unstated_loads.tsv"
grep -h "^   DIGEST " "$OUT/int15.log" 2>/dev/null | sed 's/^   //' > "$OUT/digests.tsv"
if [ -s "$OUT/digests.tsv" ] && cmp -s "$OUT/digests.tsv" "$OUT/digests_unstated_loads.tsv"; then
    echo "loads: stated and unstated alignment printed the same $(grep -c . "$OUT/digests.tsv") digests" >> "$OUT/gate.txt"
else
    echo "loads: stated and unstated alignment printed DIFFERENT digests, or a run printed none" >> "$OUT/gate.txt"
    red=1
fi
run int15-sabotage fail pixi run check-gemm-int15-sabotage
must_name int15-sabotage check_int15_device_matches_oracle
must_name int15-sabotage check_int15_plans_agree
must_name int15-sabotage check_int15_planted_worst_cases
must_pass int15-sabotage check_int15_pieces_oracle_matches_oracle
must_pass int15-sabotage check_int15_device_conversions_match_host
run int15-host-sabotage fail pixi run check-gemm-int15-host-sabotage
must_name int15-host-sabotage check_int15_pieces_oracle_matches_oracle
must_name int15-host-sabotage check_int15_device_matches_oracle
must_name int15-host-sabotage check_int15_plans_agree
must_name int15-host-sabotage check_int15_planted_worst_cases
must_pass int15-host-sabotage check_int15_device_conversions_match_host
run int15-piece-sabotage fail pixi run check-gemm-int15-piece-sabotage
must_name int15-piece-sabotage check_int15_device_conversions_match_host
must_name int15-piece-sabotage check_int15_planted_worst_cases
must_pass int15-piece-sabotage check_int15_pieces_oracle_matches_oracle
run int15-quant-sabotage fail pixi run check-gemm-int15-quant-sabotage
must_name int15-quant-sabotage check_int15_device_conversions_match_host
must_pass int15-quant-sabotage check_int15_plans_agree
must_pass int15-quant-sabotage check_int15_planted_worst_cases

grep -h "int15 dispatch:" "$OUT/int15.log" "$OUT/int15-force-flat.log" 2>/dev/null >> "$OUT/gate.txt"
grep -h "^   DIGEST " "$OUT/int15.log" 2>/dev/null | sed 's/^   //' > "$OUT/digests.tsv"
grep -h "^   DIGEST " "$OUT/int15-force-flat.log" 2>/dev/null | sed 's/^   //' > "$OUT/digests_force_flat.tsv"
{
    echo "digests=$(grep -c . "$OUT/digests.tsv") digests_force_flat=$(grep -c . "$OUT/digests_force_flat.tsv")"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN every expected outcome held (see status.tsv)"
    else
        echo "verdict=RED an expected outcome did not hold (see status.tsv and the reach lines above)"
    fi
} >> "$OUT/gate.txt"

cat "$OUT/status.tsv" "$OUT/gate.txt"
for f in int15 int15-force-flat int15-unstated-loads int15-sabotage int15-host-sabotage int15-piece-sabotage int15-quant-sabotage; do
    echo "== $f (every line that is not a digest or a per-shape ok; last 60)"
    grep -v -E '^   (DIGEST|ok) ' "$OUT/$f.log" 2>/dev/null | cut -c1-600 | tail -60
done
echo "== gate digests (the clean run)"
cat "$OUT/digests.tsv"
echo "gate_job: box=$BOX red=$red"
exit "$red"
