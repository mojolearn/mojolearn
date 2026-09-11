#!/bin/sh
# tools/knn_selection_gate.sh -- DEVIATION 2496, the kNN selection lane's
# on-box work. Runs ON THE POD as tools/gemm_remote_leg.sh's
# MOJOLEARN_GEMM_LEG_EXTRA hook (after the leg's own IDENTICAL device check
# and card), from /root/mojolearn with pixi on PATH; everything it writes
# under /root/gemm_leg_out/knn-selection/ comes home with the leg's fetch.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/knn_selection_gate.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-selection \
#   sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" \
#       --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
#
# The H100 is named because the cached cuML row (bench/OPPONENT_REFERENCE.md,
# 400k x 32 / 4k / k10 = 10.225 ms, k15 = 10.817 ms, driver 580.126.09,
# dyadic-v1) is an H100 tuple; on any other GPU the gate still runs and the
# JSON still carries the ratio, labeled cached-reference, but it is not
# admissible against that row. cuML is NOT rerun for dyadic: the tuple
# exists. THAT CACHED ROW IS DYADIC ONLY. The second kind (DEVIATION 2524,
# ENGINEERING_RULES.md section 9), the real HIGGS prefix the gate's `higgs`
# fixture times, has its own opponent tuple, measured ONCE on its first leg
# by the optional `opponent` phase below and then cached in
# bench/OPPONENT_REFERENCE.md ("kNN second kind (HIGGS rows)"); after that
# the phase stays off and the gate quotes the cached HIGGS row through
# MOJOLEARN_KNN_SELECTION_CACHED_OPPONENT_HIGGS (k10=ms,k15=ms; default
# empty until the row exists).
#
# THREE PHASES plus an optional fourth, each with its own exit code in
# status.tsv; a later phase runs even when an earlier one fails, because a
# red phase is a finding:
#
#   profile   bench/knn_reference_price_main.mojo under IDENTICAL with
#             -D MOJOLEARN_KNN_PHASE_TIMERS=1, 400k/4k/d32, at k = 1, 2, 5,
#             10, 15 with -D MOJOLEARN_KNN_IDENTICAL_GENERIC_K=1 (one
#             selector bucket across every k, so the k-slope of select_ms
#             is one kernel's slope) and at k = 10, 15 with the shipped
#             specialization. No source change; this is the measurement
#             behind the brief's cost model (select_ms ~ intercept + slope*k).
#   build     python/mojolearn/identical/_mojolearn.so through
#             bindings/build.sh with the trial hook appended by its
#             MOJOLEARN_BUILD_EXTRA_DEFINES hook (-D MOJOLEARN_KNN_SELECT_TRIAL=1),
#             so the flags live in one place and the Mac/Linux differences
#             (target cpu, linker floor) are build.sh's.
#   gate      tools/knn_selection_gate.py against that binding: fixtures,
#             arm equality, order/tie/oracle checks, reach by sabotage,
#             then ordinary-request timing, under the 300 s deadline.
#             Preceded by `prefetch-higgs` (tools/knn_datasets.py
#             --prefetch higgs): the 2.6 GB HIGGS.csv.gz download when the
#             box lacks it and the 404,000-line decode into the
#             GBM_BENCH_DATA cache (default ~/datasets/gbm-bench, the trees
#             lane's store), as its own status.tsv row and never inside the
#             gate's deadline. The fresh gemm-leg pod has no volume, so a
#             new pod pays the download once per leg.
#   opponent  OPTIONAL, MOJOLEARN_KNN_SELECTION_OPPONENT=1 (default off):
#             cuML brute-force NearestNeighbors on the SAME HIGGS prefix
#             (tools/knn_cuml_reference.py --dataset higgs, index 400,000,
#             queries 4,000, k 10 and 15, 7 rounds, request and device
#             regions), in a venv built the way tools/knn_reference_leg.sh
#             builds it (numpy==2.4.6 cupy-cuda12x==14.2.0 cuml-cu12==26.8.0
#             from pypi.nvidia.com), AFTER the gate. Runs ONCE, on the first
#             leg that needs the HIGGS tuple; its JSON lands in
#             $OUT/opponent-higgs/ and its numbers go into
#             bench/OPPONENT_REFERENCE.md, after which the switch stays off.
#             MOJOLEARN_KNN_REF_PY names an existing cuML python to skip the
#             venv, as in tools/knn_reference_leg.sh.
#
# FIXTURE SELECTION: MOJOLEARN_KNN_SELECTION_TIME_FIXTURES (default the
# harness default, `dyadic,large,higgs` since DEVIATION 2524) is passed as
# --time-fixtures; MOJOLEARN_KNN_SELECTION_FIXTURES (default the harness
# default, `large,dyadic,ties,divergent_tail,higgs`) as --fixtures. Both
# are passed only when set, so the harness's own defaults govern otherwise.
# The prefetch is skipped when FIXTURES is set and does not name higgs.
#
# THE ARMS (MOJOLEARN_KNN_SELECTION_ARMS, default baseline,headbound):
# `baseline` is the 2026-09-09 kernel, `uniform` is C4 alone (DEVIATION
# 2497), `headbound` is C4 + C1 (DEVIATION 2498, NEGATIVE on the H100
# 2026-09-11), `warpbound` is C4 + C2 (DEVIATION 2515), `deferred`
# (DEVIATION 2517), `capk` and `capk_selp` (DEVIATION 2521), `voteguard`
# (DEVIATION 2522: the K-chain behind a warp-uniform vote guard). Names
# pass through to the harness unchecked; the native side raises on an
# unknown one. C4 was gated alone (ARMS=baseline,uniform); a candidate arm
# is gated as ARMS=uniform,<arm>. Since DEVIATION 2522 the harness times
# EVERY later arm against the first, so ARMS=uniform,a,b yields two timed
# pairs (a vs uniform, b vs uniform). On a build without the hook the gate
# phase FAILS at its reach check and says so.
# MOJOLEARN_KNN_SELECTION_SKIP_GATE=1 runs the profile alone;
# MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1 skips the profile (already measured
# 2026-09-11 on the H100; the brief's "Run 1 results").
#
# TIMING-ONLY ARMS (DEVIATION 2516): MOJOLEARN_KNN_SELECTION_TIMING_ONLY_ARMS
# (default empty; e.g. skiprank,skipscan,scanonly1,noshift) names arms whose
# OUTPUT IS INVALID by construction; the harness keeps them out of
# correctness, oracle and reach, times each against the first ARMS arm, and
# reports them under `timing_only` ("output invalid; phase cost only").
# `votecount` (DEVIATION 2522) goes in the same list: its output is valid
# (the harness asserts equality for it) but its launcher synchronizes per
# launch, so its time is not a price; what it yields is the
# `KNN_ADMIT_RATE` line per launch, printed under the phase-timer build
# only, which the harness folds into `admit_rate` in the phase medians. So
# votecount needs MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1.
# MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1 adds -D MOJOLEARN_KNN_PHASE_TIMERS=1
# to the BINDING build (the profile phase's own switch; a build define, not
# an environment variable) and makes the harness REQUIRE the per-request
# `KNN_PHASE_TIMERS` line, so every timed sample carries distance_ms /
# select_ms / merge_ms read from the binding. Such a build serializes the
# launch queue: its request medians are not comparable to untimed ones and
# never feed a promotion; the phase split is the measurement.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_KNN_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_KNN_SELECTION_OUT:-/root/gemm_leg_out/knn-selection}
ARMS=${MOJOLEARN_KNN_SELECTION_ARMS:-baseline,headbound}
TIMING_ONLY=${MOJOLEARN_KNN_SELECTION_TIMING_ONLY_ARMS:-}
PHASE_TIMERS=${MOJOLEARN_KNN_SELECTION_PHASE_TIMERS:-0}
PAIRS=${MOJOLEARN_KNN_SELECTION_PAIRS:-3}
CACHED=${MOJOLEARN_KNN_SELECTION_CACHED_OPPONENT:-k10=10.225,k15=10.817}
# The HIGGS opponent tuple (DEVIATION 2524): empty until measured once and
# cached in bench/OPPONENT_REFERENCE.md; then k10=ms,k15=ms.
CACHED_HIGGS=${MOJOLEARN_KNN_SELECTION_CACHED_OPPONENT_HIGGS:-}
TIME_FIXTURES=${MOJOLEARN_KNN_SELECTION_TIME_FIXTURES:-}
FIXTURES=${MOJOLEARN_KNN_SELECTION_FIXTURES:-}
OPPONENT=${MOJOLEARN_KNN_SELECTION_OPPONENT:-0}
JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
mkdir -p "$OUT/bin"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
# One dataset store for the prefetch, the gate and the opponent: the trees
# lane's (tools/speed_gbdt_arm.py::data_root), so a box that has HIGGS
# already fetches nothing.
GBM_BENCH_DATA=${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}
export GBM_BENCH_DATA

# MAX's bundled CUDA 13 assembler needs driver 580. Older-driver pods use
# their installed assembler at BOTH build and runtime (tools/knn_selector_pod_run.sh).
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
case "$driver_major" in
    ''|*[!0-9]*) ;;
    *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
           export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
       fi ;;
esac

rc=0
run() {
    # run <name> <command...>: log to $OUT/<name>.log, record the exit code.
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    printf '%s\t%s\t%ss\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    [ "$_code" -eq 0 ] || rc=1
    return "$_code"
}

{
    echo "deviation=2496"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "arms=$ARMS"
    echo "timing_only_arms=$TIMING_ONLY"
    echo "phase_timers=$PHASE_TIMERS"
    echo "pairs=$PAIRS"
    echo "cached_opponent=$CACHED"
    echo "cached_opponent_higgs=$CACHED_HIGGS"
    echo "time_fixtures=${TIME_FIXTURES:-(harness default)}"
    echo "fixtures=${FIXTURES:-(harness default)}"
    echo "opponent=$OPPONENT"
    echo "data_root=$GBM_BENCH_DATA"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor)=' /root/gemm_leg_out/leg.txt
} > "$OUT/gate.txt"
nvidia-smi --query-gpu=name,driver_version,uuid,clocks.sm,temperature.gpu --format=csv > "$OUT/gpu_before.csv" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"

# ---- profile: the k-slope of the selection class, no source change -------
if [ "${MOJOLEARN_KNN_SELECTION_SKIP_PROFILE:-0}" != "1" ]; then
# shellcheck disable=SC2086
run build-profile-generic pixi run mojo build -j "$JOBS" -I . $IDENT \
    -D MOJOLEARN_KNN_PHASE_TIMERS=1 -D MOJOLEARN_KNN_IDENTICAL_GENERIC_K=1 \
    bench/knn_reference_price_main.mojo -o "$OUT/bin/profile-generic"
# shellcheck disable=SC2086
run build-profile-default pixi run mojo build -j "$JOBS" -I . $IDENT \
    -D MOJOLEARN_KNN_PHASE_TIMERS=1 \
    bench/knn_reference_price_main.mojo -o "$OUT/bin/profile-default"
fi
for k in 1 2 5 10 15; do
    if [ -x "$OUT/bin/profile-generic" ]; then
        MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_FEATURES=32 \
        MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=3 \
        run "profile-generic-k$k" "$OUT/bin/profile-generic"
    fi
done
for k in 10 15; do
    if [ -x "$OUT/bin/profile-default" ]; then
        MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_FEATURES=32 \
        MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=3 \
        run "profile-default-k$k" "$OUT/bin/profile-default"
    fi
done
# One line per run: the last KNN_PHASE_TIMERS distance/select/merge split.
for f in "$OUT"/profile-*-k*.log; do
    [ -f "$f" ] || continue
    printf '%s\t' "$(basename "$f" .log)"
    grep 'KNN_PHASE_TIMERS distance_ms' "$f" | tail -1
    echo
done > "$OUT/profile_summary.tsv"

if [ "${MOJOLEARN_KNN_SELECTION_SKIP_GATE:-0}" = "1" ]; then
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) (gate skipped)" >> "$OUT/gate.txt"
    exit "$rc"
fi

# ---- build: the public binding with the trial hook ----------------------
# bindings/build.sh owns the command (target cpu, linker floor, include
# paths, the identical output directory python/mojolearn/identical/); the
# trial define rides its MOJOLEARN_BUILD_EXTRA_DEFINES hook. One build is
# one GPU architecture: this box's.
# `env`, not a prefix assignment: dash does not reliably pass prefix
# assignments through a shell function.
PHASE_DEFINE=""
PHASE_FLAG="--phase-timers auto"
if [ "$PHASE_TIMERS" = "1" ]; then
    PHASE_DEFINE="-D MOJOLEARN_KNN_PHASE_TIMERS=1"
    PHASE_FLAG="--phase-timers require"
fi
run build-binding-trial env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS="$JOBS" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_KNN_SELECT_TRIAL=1 $PHASE_DEFINE ${MOJOLEARN_KNN_SELECTION_EXTRA_DEFINES:-}" \
    sh bindings/build.sh
# The binary is witnessed by hash only: binaries are not evidence and the
# repository's blob fences refuse them (no-oversized-blobs rule).
if [ -f python/mojolearn/identical/_mojolearn.so ]; then
    sha256sum python/mojolearn/identical/_mojolearn.so > "$OUT/binding.sha256"
fi

# ---- gate ----------------------------------------------------------------
# The interpreter must be the one the extension was built against (the
# pixi python); numpy is needed for the fixtures. The default environment
# is tried first, then the gbmbench environment tools/e1_bootstrap.sh uses.
PY=""
if pixi run python3 -c 'import numpy' > /dev/null 2>&1; then
    PY="pixi run python3"
elif pixi run -e gbmbench python3 -c 'import numpy' > "$OUT/gbmbench-install.log" 2>&1; then
    PY="pixi run -e gbmbench python3"
fi
echo "python=$PY" >> "$OUT/gate.txt"
if [ -z "$PY" ]; then
    echo "no pixi python with numpy; gate not run" >> "$OUT/gate.txt"
    printf 'gate\t9\t0s\n' >> "$OUT/status.tsv"
    rc=1
else
    # The real fixture's fetch and decode, OUTSIDE the gate's deadline and
    # with its own status row (DEVIATION 2524). Skipped only when FIXTURES
    # is set and leaves higgs out. Generous fence: this is a network fetch.
    NEED_HIGGS=1
    case ",$FIXTURES," in
        ,,) ;;
        *,higgs,*) ;;
        *) NEED_HIGGS=0 ;;
    esac
    if [ "$NEED_HIGGS" = "1" ]; then
        # shellcheck disable=SC2086
        run prefetch-higgs timeout -k 30 1800 $PY tools/knn_datasets.py --prefetch higgs
    fi
    FIXTURE_FLAGS=""
    [ -n "$TIME_FIXTURES" ] && FIXTURE_FLAGS="$FIXTURE_FLAGS --time-fixtures $TIME_FIXTURES"
    [ -n "$FIXTURES" ] && FIXTURE_FLAGS="$FIXTURE_FLAGS --fixtures $FIXTURES"
    [ -n "$CACHED_HIGGS" ] && FIXTURE_FLAGS="$FIXTURE_FLAGS --cached-opponent-higgs $CACHED_HIGGS"
    # `timeout` is a second fence outside the harness's own 300 s deadline.
    # shellcheck disable=SC2086
    PYTHONPATH="$ROOT/python" MOJOLEARN_NUMERIC_MODE=identical \
    run gate timeout 420 $PY tools/knn_selection_gate.py \
        --out "$OUT" --arms "$ARMS" --timing-only-arms "$TIMING_ONLY" \
        --pairs "$PAIRS" --deadline 300 $PHASE_FLAG \
        --cached-opponent "$CACHED" $FIXTURE_FLAGS
fi

# ---- opponent (optional): the HIGGS cuML row, measured ONCE --------------
# ENGINEERING_RULES.md section 9: an opponent row is measured once per (GPU,
# driver, opponent version, dataset) and cached; the second kind is a new
# tuple. Same venv recipe as tools/knn_reference_leg.sh (system python3,
# --system-site-packages, the pinned NVIDIA wheels); MOJOLEARN_KNN_REF_PY
# names an existing cuML python instead. Runs after the gate so the gate's
# GPU window is not shared with a 1 GB wheel install. The venv lives OUTSIDE
# /root/gemm_leg_out: the leg tars that whole directory home over the Mac's
# uplink, and a gigabyte of wheels is not evidence.
if [ "$OPPONENT" = "1" ]; then
    OPP_OUT="$OUT/opponent-higgs"
    mkdir -p "$OPP_OUT"
    OPY=${MOJOLEARN_KNN_REF_PY:-}
    if [ -z "$OPY" ]; then
        VENV=${MOJOLEARN_KNN_CUML_VENV:-/root/knn-cuml-venv}
        run opponent-venv python3 -m venv --system-site-packages "$VENV"
        OPY="$VENV/bin/python"
        run opponent-wheels "$OPY" -m pip install --disable-pip-version-check --no-input --only-binary=:all: \
            numpy==2.4.6 cupy-cuda12x==14.2.0 cuml-cu12==26.8.0 --extra-index-url https://pypi.nvidia.com
    fi
    run opponent-freeze "$OPY" -m pip freeze
    run opponent timeout -k 30 1800 "$OPY" tools/knn_cuml_reference.py --dataset higgs \
        --index 400000 --queries 4000 --k 10 15 --rounds 7 --out "$OPP_OUT"
    [ -f "$OPP_OUT/cuml-reference-higgs.json" ] && grep -h 'CUML_REF ' "$OUT/opponent.log" > "$OPP_OUT/summary.txt" 2>/dev/null
fi

nvidia-smi --query-gpu=name,driver_version,uuid,clocks.sm,temperature.gpu --format=csv > "$OUT/gpu_after.csv" 2>&1
# Profile binaries stay on the box for the same reason.
rm -rf "$OUT/bin"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
[ -f "$OUT/summary.txt" ] && cat "$OUT/summary.txt"
cat "$OUT/status.tsv"
exit "$rc"
