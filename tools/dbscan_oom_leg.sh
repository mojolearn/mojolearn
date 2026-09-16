#!/bin/sh
# tools/dbscan_oom_leg.sh: the on-box body of lane/amd-dbscan-oom, 2026-09-16.
#
#   AMD, DigitalOcean MI325X:
#     MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
#     MOJOLEARN_GPU_ARCHS=gfx942 \
#     MOJOLEARN_GEMM_LEG_EXTRA=tools/dbscan_oom_leg.sh \
#     MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<stamp>-amd-mi325x-dbscan-oom \
#     bash tools/do_extra_leg.sh amd
#
#   NVIDIA, RunPod, the CONTROL and the rehearsal:
#     MOJOLEARN_GEMM_LEG_EXTRA=tools/dbscan_oom_leg.sh \
#     sh tools/gemm_remote_leg.sh nvidia --rent ...
#
# WHAT IS BEING ASKED. lane/classical-host-recordings took the AMD column for
# the saved-model predict lanes on a RunPod MI300X and read
# `cells=54 stable=31 moved=0 refused=23`, every refusal a hipErrorOutOfMemory
# raised in `dbscan_fit_core` fitting 6000 rows of four columns on a 192 GB
# card. Nothing diverged. That lane wrote down, correctly, that it did not know
# whether this was a leak or fragmentation, which allocation it was, or whether
# it reproduced at all: one card, one process, no instrumentation.
#
# WHAT THIS BODY ADDS THAT THAT RUN COULD NOT HAVE. It reads the device's used
# memory from OUTSIDE our library before and after every fit and keeps fitting
# past every raise, so the answer is a TREND and not a stopping point. And it
# fits the largest neighbourhood FIRST, in a process that has done no other GPU
# work, which is the one measurement that separates "this shape cannot be
# served" from "this shape cannot be served AFTER other fits".
#
# ARM ORDER IS DELIBERATE: the two arms that decide the question run before the
# eight that describe it, because a box that dies early still comes home with
# an answer.
#
# POSIX sh. `set -u`, deliberately NOT `set -e`: a phase that fails is the
# finding and its log has to come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/dbscan-oom
mkdir -p "$OUT/logs" "$OUT/json"
G="$OUT/gate.txt"
JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
PROBE=tools/dbscan_memory_probe.py
LANES=dbscan,dbscan-brute-l1,dbscan-weighted

say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    say "--- $_n exit=$_e ---"
    return "$_e"
}
# every arm's last lines land in the gate file, so gate.txt alone tells the
# story if the JSON does not come home
arm() {
    _n=$1; shift
    run "$_n" "$@"
    _e=$?
    sed -n '/--- trend/,$p' "$OUT/logs/$_n.log" >> "$G" 2>/dev/null
    grep -c RAISE "$OUT/logs/$_n.log" 2>/dev/null | sed "s/^/$_n raises=/" >> "$G"
    return "$_e"
}

say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"

# ------------------------------------------------------------- the card itself
VENDOR=nvidia
if command -v rocm-smi > /dev/null 2>&1; then VENDOR=amd; fi
say "vendor_tool=$VENDOR"
if [ "$VENDOR" = amd ]; then
    rocm-smi --showproductname > "$OUT/logs/device.txt" 2>&1
    rocm-smi --showmeminfo vram --csv >> "$OUT/logs/device.txt" 2>&1
    for f in /sys/class/drm/card*/device/mem_info_vram_total; do
        [ -r "$f" ] && echo "$f $(cat "$f")" >> "$OUT/logs/device.txt"
    done
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        _gfx=$(rocminfo 2>/dev/null | sed -n 's/.*Name: *\(gfx[0-9a-f]*\).*/\1/p' | head -1)
        case "$_gfx" in
            gfx[0-9]*) MOJOLEARN_GPU_ARCHS="$_gfx"; export MOJOLEARN_GPU_ARCHS ;;
            *) say "NO GPU ARCH: rocminfo gave '$_gfx' and none was passed in. REFUSING to guess."
               say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit 2 ;;
        esac
    fi
    say "gpu_archs=$MOJOLEARN_GPU_ARCHS"
else
    nvidia-smi > "$OUT/logs/device.txt" 2>&1
fi
say "device=$(head -20 "$OUT/logs/device.txt" | tr '\n' ' ' | cut -c1-200)"
(grep -m1 'model name' /proc/cpuinfo; uname -m; free -g | head -2) > "$OUT/logs/cpu.txt" 2>&1
say "host=$(tr '\n' ' ' < "$OUT/logs/cpu.txt" | cut -c1-160)"

# -------------------------------------------------------------- the two builds
# DBSCAN's fit is `dbscan_fit_core` in _mojolearn_estimators and its predict is
# `labeled_reference_predict` in the same family; the base family is what
# `import mojolearn` needs. Nothing else is built, because nothing else is
# fitted here.
BUILD="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS"
run build-base       $BUILD sh bindings/build.sh            || say "BUILD FAILED: bindings/build.sh"
run build-estimators $BUILD sh bindings/build_estimators.sh || say "BUILD FAILED: bindings/build_estimators.sh"

IB="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python"
say "vendor=$($IB pixi run python -c 'import mojolearn; print(mojolearn.vendor())' 2>&1 | tail -1)"
$IB pixi run python "$PROBE" --out /dev/null --fixture-digest > "$OUT/logs/fixture-digest.log" 2>&1
say "fixture_digests_written=$(wc -l < "$OUT/logs/fixture-digest.log")"

P="$IB pixi run python $PROBE"

# =========================================================== 1. THE ORDER TEST
# `wide` is 36,000,000 edges at eps=0.9, the largest neighbourhood any fixture
# asks for, and it is the fixture the MI300X column first refused. Here it is
# the FIRST GPU work the process does. If it raises now, the refusal is shaped
# and the word "ordered" has to come out of the write-up. If it does not raise
# now but raised there, the order is real and something is accumulating.
arm cold $P --out "$OUT/json/cold.json" --arm cold --self-check-fixture wide

# ====================================================== 2. THE REPRODUCTION ARM
# identity_break's own order, its own three lanes, its own nine fixtures, two
# repeats: 54 fits, the same 54 the AMD column ran, with memory read around
# each one and no stop at the first raise.
arm seq $P --out "$OUT/json/seq.json" --arm seq --repeats 2

# ================================================= 3. THE PER FIT LEAK, ALONE
# One shape, fitted 24 times. The shape cannot change, so a climb is a per fit
# leak and a flat line with failures is fragmentation. This is the arm the fix,
# if there is one, has to flatten.
arm repeat-base $P --out "$OUT/json/repeat-base.json" --arm repeat --n 24 --repeat-fixture base
arm repeat-negative $P --out "$OUT/json/repeat-negative.json" --arm repeat --n 12 --repeat-fixture negative

# ============================================ 4. SHAPE, ASCENDING, ONE PASS EACH
arm shapes $P --out "$OUT/json/shapes.json" --arm shapes

# ======================================== 5. WHICH PATH INSIDE THE FIT CLIMBS
# prediction_data on and off is the `dbscan_fit_core` core-mask readback
# against plain `dbscan_fit`; brute against rbc is the ball cover against the
# n^2 arm; weighted adds the per batch `ja1` scratch that only the weighted
# path allocates (dbscan/impl/runner.mojo:555).
arm arms-base $P --out "$OUT/json/arms-base.json" --arm arms --n 8 --repeat-fixture base
arm arms-negative $P --out "$OUT/json/arms-negative.json" --arm arms --n 6 --repeat-fixture negative

# ========================================================== 6. THE BUDGET PATH
# max_mbytes_per_batch=None makes dbscan/impl/dbscan.mojo:248 derive the budget
# from 80% of TOTAL device memory. That is the one input that differs between
# the 192 GB card that refused and the 80 GB card that did not, so it is asked
# directly rather than reasoned about.
arm budget-wide $P --out "$OUT/json/budget-wide.json" --arm budget --n 6 --repeat-fixture wide
arm budget-base $P --out "$OUT/json/budget-base.json" --arm budget --n 6 --repeat-fixture base

# =================================== 7. THE HARNESS ITSELF, FOR THE SAME WORDS
# The probe drives the estimator directly. This runs tools/identity_break.py,
# which is what produced the original reading, so the two can be compared in
# the same vocabulary (`cells= stable= moved= refused=`).
LABEL="$VENDOR-leg"
run identity $IB pixi run python tools/identity_break.py --lanes "$LANES" \
    --repeats 2 --vendor "$LABEL" --json "$OUT/json/identity.$LABEL.json"
grep -E '^cells=|^summary|REFUSED|hipError|CUDA_ERROR|out of memory' \
    "$OUT/logs/identity.log" 2>/dev/null | head -40 >> "$G"

say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
exit 0
