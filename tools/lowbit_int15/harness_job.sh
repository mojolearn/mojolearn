#!/bin/bash
# tools/lowbit_int15/harness_job.sh -- lane/lowbit-int15, step 2: the fifteen
# bit GEMM in THE REPO'S OWN VERIFICATION HARNESS, on ONE box.
#
#   lane-check   tools/algos_lane_check.sh gemm-int15 --pass 2
#                  --sabotage gemm/checks/sabotage/int15_device_value_flip.patch
#                The one lane check: it builds the linalg binding and the
#                linalg host binding where they are stale, runs the seam
#                drivers of tools/identity_lanes/int15.checks (each must
#                PASS, FAIL under its patch and PASS after reversal), fits
#                the lane on the device and on the CPU over every fixture,
#                and requires AGREE, then DISAGREE under the patch, then
#                AGREE again. Its last line is RESULT: PASS or RESULT: FAIL.
#   neighbors    tools/algos_lane_check.sh gemm-int8,gemm-bf16
#                The two lanes whose kernels share the fragment loads this
#                lane edited (DEVIATION 2975) and the binding this lane
#                extended: clean AGREE, no patch.
#   verify       python -m mojolearn verify --lanes gemm-int15 --repeats 2
#                What the shipped verifier says of the lane on this box. The
#                lane is in PUBLIC_PENDING_LANES ("no reference"), so its
#                parts read OWED, not IDENTICAL, until a record carries it.
#                RECORDED, never the verdict of this job.
#
# The two columns of the clean check (device and CPU) are printed as
# base64 of gzip between BEGIN-COLUMN and END-COLUMN lines, so a steward's
# stdout carries them home and the four boxes' columns can be diffed cell by
# cell with `tools/identity_break.py --diff`.
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_int15/$BOX/harness"
rm -rf "$OUT"
mkdir -p "$OUT"
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
    echo "profile=mojolearn.identical.gemm.int15i64.v1 lane=gemm-int15"
    echo "box=$BOX"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
} > "$OUT/harness.txt"
# The harness records the commit of every column. On a patch-synced tree HEAD
# is the merge base; the lane's own commit is what the sync wrote beside it.
if [ -z "${MOJOLEARN_COMMIT:-}" ] && [ -f .lowbit_int15_commit ]; then
    export MOJOLEARN_COMMIT=$(cat .lowbit_int15_commit)
fi

t0=$(date +%s)
sh tools/algos_lane_check.sh gemm-int15 --pass 2 \
    --sabotage gemm/checks/sabotage/int15_device_value_flip.patch \
    --out "$OUT/lane_check" > "$OUT/lane_check.stdout" 2>&1
rc=$?
echo "lane-check exit=$rc $(( $(date +%s) - t0 ))s: $(grep -E '^RESULT: ' "$OUT/lane_check.stdout" | tail -1)" >> "$OUT/harness.txt"
[ "$rc" -eq 0 ] || red=1

t0=$(date +%s)
sh tools/algos_lane_check.sh gemm-int8,gemm-bf16 --out "$OUT/neighbors" > "$OUT/neighbors.stdout" 2>&1
rc=$?
echo "neighbors exit=$rc $(( $(date +%s) - t0 ))s: $(grep -E '^RESULT: ' "$OUT/neighbors.stdout" | tail -1)" >> "$OUT/harness.txt"
[ "$rc" -eq 0 ] || red=1

# The verifier's comparator self-test fits `ols` before any lane, so it needs
# the estimators binding, which the lane check does not build (it builds only
# what gemm-int15 and its neighbors import). Without it `verify` stops at the
# self-test (H100 nvc3-0029 and M3 Ultra 1790657536591, 2026-09-29). Built
# here when absent; a build failure is recorded and verify reads what it reads.
if [ ! -f python/mojolearn/identical/_mojolearn_estimators.so ]; then
    t0=$(date +%s)
    MOJOLEARN_NUMERIC_MODE=identical sh bindings/build_estimators.sh > "$OUT/build_estimators.log" 2>&1
    echo "build_estimators (for verify's self-test) exit=$? $(( $(date +%s) - t0 ))s" >> "$OUT/harness.txt"
fi

t0=$(date +%s)
PYTHONPATH="$PWD/python" MOJOLEARN_NUMERIC_MODE=identical pixi run -e default python -m mojolearn verify \
    --lanes gemm-int15 --repeats 2 --json-out "$OUT/verify.json" > "$OUT/verify.stdout" 2>&1
echo "verify exit=$? $(( $(date +%s) - t0 ))s (recorded, not this job's verdict)" >> "$OUT/harness.txt"

{
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN the lane check passed with its sabotage seen, and the neighbor lanes agree"
    else
        echo "verdict=RED (see the RESULT lines above)"
    fi
} >> "$OUT/harness.txt"

cat "$OUT/harness.txt"
echo "== lane-check (every line the check printed)"
cut -c1-500 "$OUT/lane_check.stdout" | grep -v mbind | tail -80
echo "== neighbors"
cut -c1-500 "$OUT/neighbors.stdout" | grep -v mbind | tail -30
echo "== verify (last 40)"
cut -c1-300 "$OUT/verify.stdout" | grep -v mbind | tail -40
if [ "$red" -ne 0 ]; then
    echo "== lane_check.log (last 80)"
    tail -80 "$OUT/lane_check/lane_check.log" 2>/dev/null | cut -c1-400
    echo "== neighbors lane_check.log (last 40)"
    tail -40 "$OUT/neighbors/lane_check.log" 2>/dev/null | cut -c1-400
fi
for stage in clean sabotaged restored; do
    for arm in gpu cpu; do
        f="$OUT/lane_check/$stage.gemm-int15.$arm.json"
        [ -f "$f" ] || continue
        echo "BEGIN-COLUMN $BOX $stage gemm-int15 $arm"
        gzip -9 -c "$f" | base64 | tr -d '\n' | fold -w 200
        echo
        echo "END-COLUMN"
    done
done
echo "harness_job: box=$BOX red=$red"
exit "$red"
