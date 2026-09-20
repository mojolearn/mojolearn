#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA for lane/gpu-confirm-never-launched, leg 6:
# `verify --par` AGAINST A REAL CUDA DRIVER, FOR THE FIRST TIME.
#
# TWO QUESTIONS, AND THE SECOND ONE IS ABOUT THE FIRST ONE'S HONESTY.
#
# (1) par-queries-nn's batch part, 18 of 18 failures on AMD, cause unknown.
#     The peer-copy story is dead: the other lane read the path instead of
#     assuming it, and this family performs NO cross-device transfer of any
#     kind. DevicePool._start gives each worker its own process with
#     HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES pinned to ONE index, so a
#     worker cannot address a second device even in principle; shards are cut
#     in Python, merged with a host memcopy, and returned by pickle over a
#     pipe. Nor is it a two-device property: the same lane PASSED on two AMD
#     devices the day before, at bench/results/identity_break/
#     2026-09-19_hardware-gaps/amd-par-queries-nn-two.json (amd-gfx942,
#     par_devices "0,1", commit c3b5783cf), batch STABLE on all nine fixtures
#     with exactly the nine hashes the failing one-device column carries.
#     So it is a regression or an environment difference.
#
#     THE IN-WORKER FRAME IS THE WHOLE PRIZE, and it was unrecoverable from
#     the original capture because the 300-character error cap kept the
#     OUTERMOST frames -- the part that says a worker died, not the part that
#     says why. ERROR_TEXT_LIMIT is now 8000 and keeps the INNERMOST frames
#     (tools/identity_break.py:435, :515). This is the first run on a real
#     driver since that changed.
#
#     PASSING IS AN EQUALLY GOOD RESULT and must be reported as one: it means
#     the regression is already gone and the item closes.
#
# (2) `verify --par --par-self-test` MUST EXIT 0. The self test runs an arm
#     that is SUPPOSED to fail, so a zero exit means the comparison can
#     actually catch a moved bit. Its witness guard has only ever been watched
#     on a CPU-only install, where it correctly printed "WITNESS REFUSED: the
#     two-device column started NO device pool at all" and CANNOT RUN instead
#     of a clean AGREE. It has never run against a real CUDA or HIP driver, so
#     a two-device box is the only place the guard's happy path is stateable.
#     Run FIRST, because a `verify --par` result from a comparison nobody has
#     watched catch anything is not evidence.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/verify-par
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
JOBS="${MOJOLEARN_COMPILE_JOBS:-16}"

T0=$(date +%s)
BUDGET="${MOJOLEARN_VP_BUDGET:-2700}"
cap() {
    _want=$1; _left=$(( T0 + BUDGET - $(date +%s) ))
    [ "$_left" -lt 60 ] && _left=60
    if [ "$_want" -gt "$_left" ]; then echo "$_left"; else echo "$_want"; fi
}
say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s); "$@" > "$OUT/logs/$_n.log" 2>&1; _e=$?
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1 | awk '{print $1}')
    case "$_c" in [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;; esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || {
    say "NO NVIDIA DEVICE"; exit 8; }
VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
say "visible_gpus=$VISIBLE"
nvidia-smi -L >> "$G" 2>&1
# TWO DEVICES OR NEITHER QUESTION EXISTS. `verify --par` flips
# MOJOLEARN_PAR_DEVICES between two run_cell calls IN ONE PROCESS; with one
# device both calls are the same call and AGREE is a tautology.
if [ "$VISIBLE" -lt 2 ]; then
    say "REFUSED: $VISIBLE visible GPU(s). Both arms of `verify --par` need a second device; on one device AGREE is a tautology and the self test cannot fail. Nothing was run."
    exit 8
fi
_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case "$_cc" in
    9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;;
    8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;;
    12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;;
    *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;;
esac
export MOJOLEARN_GPU_ARCHS
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"

# The host math library, or `import mojolearn` cannot complete. .libs/ is
# gitignored and nothing under bindings/ builds it.
run portable_math timeout "$(cap 300)" env PYTHONPATH=/root/mojolearn/packaging/portable_math \
    pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))"
say "portable_math_exit=$(awk -F'	' '$1=="portable_math"{print $2}' "$OUT/status.tsv")"

build() {
    run "$1" timeout "$(cap 420)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$1.sh"
}
build_host() {
    run "$1" timeout "$(cap 420)" env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu \
        MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" sh "bindings/$1.sh"
}
failed=""
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh)
    [ "$n" = build_host_family ] && continue
    case "$n" in
        build_*_host) build_host "$n" || failed="$failed $n" ;;
        *)            build "$n"      || failed="$failed $n" ;;
    esac
done
say "build_failed=${failed:-none}"
say "elapsed_after_build=$(( $(date +%s) - T0 ))"

run import_check timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python -c "import mojolearn; print('IMPORT_OK', mojolearn.vendor(), mojolearn.__version__)"
say "import_check=$(tail -1 "$OUT/logs/import_check.log" 2>/dev/null)"

# ---- (2) FIRST: can the comparison catch anything at all?
run par_self_test timeout "$(cap 900)" env MOJOLEARN_NUMERIC_MODE=identical \
    PYTHONPATH=/root/mojolearn/python \
    pixi run python -m mojolearn verify --par --par-self-test
say "par_self_test_exit=$(awk -F'	' '$1=="par_self_test"{print $2}' "$OUT/status.tsv")  (MUST BE 0: the self test runs an arm that is supposed to fail)"
tail -40 "$OUT/logs/par_self_test.log" >> "$G" 2>&1

# ---- (1) THEN: the lane itself, with the innermost frames now retained
run verify_par_nn timeout "$(cap 900)" env MOJOLEARN_NUMERIC_MODE=identical \
    PYTHONPATH=/root/mojolearn/python \
    pixi run python -m mojolearn verify --par --lanes par-queries-nn
say "verify_par_nn_exit=$(awk -F'	' '$1=="verify_par_nn"{print $2}' "$OUT/status.tsv")  (1 with batch ONE-COLUMN reproduces it; 0 means the regression is GONE and the item closes)"
tail -120 "$OUT/logs/verify_par_nn.log" >> "$G" 2>&1

# The raw identity_break pair too, so the cell's own error text comes home at
# the new 8000-character limit rather than only verify's rendering of it.
run ib_one timeout "$(cap 600)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PAR_DEVICES=0 \
    PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
    --lanes par-queries-nn --repeats 2 --vendor "nvidia-$MOJOLEARN_GPU_ARCHS" --json "$OUT/nn.one-device.json"
run ib_two timeout "$(cap 600)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PAR_DEVICES=0,1 \
    PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
    --lanes par-queries-nn --repeats 2 --vendor "nvidia-$MOJOLEARN_GPU_ARCHS" --json "$OUT/nn.two-device.json"
run ib_diff timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --diff "$OUT/nn.one-device.json" "$OUT/nn.two-device.json"
say "ib_one_exit=$(awk -F'	' '$1=="ib_one"{print $2}' "$OUT/status.tsv") ib_two_exit=$(awk -F'	' '$1=="ib_two"{print $2}' "$OUT/status.tsv") ib_diff_exit=$(awk -F'	' '$1=="ib_diff"{print $2}' "$OUT/status.tsv")"
grep -E 'summary|DIVERGENT|MOVED|REFUSED|ONE-COLUMN|NOT-COMPARED' "$OUT/logs/ib_diff.log" | head -40 >> "$G"
# The errors sidecar is where the innermost worker frame lands.
for f in "$OUT"/nn.*.json.errors.txt; do
    [ -f "$f" ] || continue
    say "--- $(basename "$f") ---"
    head -200 "$f" >> "$G"
done

say "elapsed_total=$(( $(date +%s) - T0 ))"
cp "$OUT/logs/"*.log "$OUT/" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
