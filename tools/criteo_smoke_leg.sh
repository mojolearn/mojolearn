#!/bin/sh
# RUNS ON THE POD. lane/criteo-smoke: the FIRST fit of any arm on criteo, the
# categorical dataset, on one rented NVIDIA H100.
#
#   nohup sh tools/criteo_smoke_leg.sh > /root/criteo_out/body.log 2>&1 &
#
# WHAT THIS PROVES, AND WHAT IT DOES NOT
# ---------------------------------------
# `cat_features` had never appeared in any benchmark before 2026-09-12, so
# DEVIATION 2634 ("skip the CTR target prep when no column is categorical")
# had only ever run on its SKIP side: its 0.9603 flip verdict was measured on
# taxi and Istella-S, and neither has a categorical column. On criteo the CTR
# prep actually runs, so every number here is a NEW measurement and none of it
# confirms the earlier verdict.
#
# REACH BEFORE SPEED. The probe stages run FIRST and separately (one process
# per library, so two CUDA runtimes never share one): a timing on an arm that
# silently fell back to numeric splits would be a measurement of the wrong
# problem, which is worse than a crash. The probe's with-cats/without-cats
# A/B is what distinguishes "the categorical path ran" from "the indices were
# accepted and ignored".
#
# TIMED CELLS NEVER OVERLAP. One lane per process, run one after another, and
# no build or probe shares the device with a timed round -- the calls below are
# sequential for exactly that reason.
#
# POSIX sh (dash), `set -u` and not `set -e`: a red phase must not stop the
# phases after it, because a partial table is still owed results.
set -u
ROOT=/root/mojolearn
OUT=/root/criteo_out
mkdir -p "$OUT/logs" "$OUT/speed" "$OUT/probe"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"
export PATH
GBM_BENCH_DATA=/root/datasets/gbm-bench
export GBM_BENCH_DATA
DEBIAN_FRONTEND=noninteractive
export DEBIAN_FRONTEND
PY=python3
JOBS=$(nproc 2>/dev/null || echo 8)
say() { printf '[%s] %s\n' "$(date -u +%T)" "$*"; }
status() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date -u +%T)" >> "$OUT/status.tsv"; }
status body_start 0
say "body start jobs=$JOBS"

# ------------------------------------------------------------------ box
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  uname -r; nproc; free -g | head -2; } > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"

# ------------------------------------------------------------------ deps
# The opponents plus the parquet reader. pyarrow and pandas are DOWNLOAD-STEP
# and frame-conversion dependencies; neither is imported inside a timed fit by
# our arm.
timeout -k 10 1200 $PY -m pip install --break-system-packages \
    --disable-pip-version-check --no-input \
    catboost xgboost scikit-learn pandas pyarrow > "$OUT/logs/pip.log" 2>&1
_rc=$?
if [ "$_rc" != 0 ]; then
    timeout -k 10 1200 $PY -m pip install --disable-pip-version-check --no-input \
        catboost xgboost scikit-learn pandas pyarrow >> "$OUT/logs/pip.log" 2>&1
    _rc=$?
fi
status pip "$_rc"
$PY - <<'PY' > "$OUT/versions.txt" 2>&1
for m in ("numpy", "catboost", "xgboost", "sklearn", "pandas", "pyarrow"):
    try:
        print(m, __import__(m).__version__)
    except Exception as exc:
        print(m, "IMPORT FAILED", exc)
PY
status versions $?
cat "$OUT/versions.txt"

# ------------------------------------------------------------------ data (background)
# ~291 MB of parquet over three parts, then ONE decode that freezes the
# category codes into an npz. Explicitly a separate step from any timed run.
(
    timeout -k 10 2400 $PY tools/speed_gbdt_arm.py --download criteo \
        > "$OUT/logs/download_criteo.log" 2>&1
    echo "download_criteo=$? $(date -u +%T)" >> "$OUT/data.txt"
    : > "$OUT/data.done"
) &

# ------------------------------------------------------------------ pixi + builds
if ! command -v pixi > /dev/null 2>&1; then
    timeout -k 10 600 sh -c 'curl -fsSL --max-time 540 https://pixi.sh/install.sh | sh' \
        > "$OUT/logs/pixi_install.log" 2>&1
    status pixi_install $?
    PATH="$HOME/.pixi/bin:$PATH"
    export PATH
fi
timeout -k 30 2400 pixi install > "$OUT/logs/pixi_env.log" 2>&1
status pixi_env $?

# ONE architecture per build (the compiler takes exactly one name); sm_90a is
# this box's H100.
MOJOLEARN_GPU_ARCHS=sm_90a
export MOJOLEARN_GPU_ARCHS
build() {  # <script> <so>
    _t0=$(date +%s)
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" \
        timeout -k 30 2700 bash "bindings/$1" > "$OUT/logs/$1.log" 2>&1
    _rc=$?
    status "build.$1" "$_rc s=$(( $(date +%s) - _t0 ))"
    if [ "$_rc" != 0 ]; then
        grep -m 8 -i -A3 error "$OUT/logs/$1.log" >> "$OUT/build_errors.txt"
    fi
    return $_rc
}
build build.sh _mojolearn.so
build build_gbdt.sh _mojolearn_gbdt.so
ls -la python/mojolearn/identical/ > "$OUT/builds.txt" 2>&1

# The mode/vendor readback the harness itself refuses without, checked here so
# a failure is one line rather than a dead cell.
imp() {
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" timeout -k 10 300 $PY -c \
"import mojolearn
m = mojolearn.GradientBoosting()
b = m._bind('_mojolearn_gbdt')
print('import OK', m.numeric_mode_used(), m.vendor_used(), b.gbdt_per_round_paths())
print('path', b.__file__)" > "$OUT/import.txt" 2>&1
}
imp
_ri=$?
if [ "$_ri" != 0 ]; then
    # A fresh box builds one binding at a time, so the package import can fail
    # on a sibling .so that does not exist yet.
    for b in build_rf.sh build_trees.sh; do build "$b" sibling; done
    imp
    _ri=$?
fi
status import "$_ri"
cat "$OUT/import.txt"
if [ "$_ri" != 0 ]; then
    status no_binding 3
    say "our binding does not import; the opponent cells still run below"
fi

# ------------------------------------------------------------------ wait for data
while [ ! -f "$OUT/data.done" ]; do sleep 15; done
cat "$OUT/data.txt"
ls -la "$GBM_BENCH_DATA/criteo" >> "$OUT/builds.txt" 2>&1
if [ ! -f "$GBM_BENCH_DATA/criteo/criteo_speed.npz" ]; then
    status no_criteo 4
    say "criteo did not decode; REFUSING to run, because load_with_fallback would"
    say "silently measure the synthetic fixture instead"
    exit 4
fi

# ------------------------------------------------------------------ reach probes
# One process per library: the point is reach, and two CUDA runtimes in one
# process is a known way to lose a box. Before every timed cell.
probe() {  # <stage>
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_STAGE_TIMES=1 \
        PYTHONPATH="$ROOT/python" timeout -k 30 1800 $PY -u tools/criteo_reach_probe.py \
        --stage "$1" > "$OUT/probe/$1.log" 2>&1
    status "probe.$1" $?
    grep -E '^PROBE' "$OUT/probe/$1.log" | head -40
}
probe ours
probe catboost
probe xgboost

# ------------------------------------------------------------------ timed cells
# gbdt-symmetric is CatBoost ONLY (DEVIATION 1831: XGBoost has no symmetric
# grower). devices defaults to `auto`, which is GPU-only on an NVIDIA box.
#
# There is NO --rounds flag on this harness; rounds are MOJOLEARN_SPEED_ROUNDS.
cell() {  # <lane>
    _lane="$1"
    say "cell $_lane"
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda \
        MOJOLEARN_SPEED_SIZE=shipped MOJOLEARN_SPEED_ROUNDS=3 \
        MOJOLEARN_SPEED_BUDGET_S=1500 MOJOLEARN_SPEED_DEADLINE_S=3300 \
        PYTHONPATH="$ROOT/python" \
        timeout -k 30 3600 $PY -u bench/speed/forest_speed_arm.py \
        --lane "$_lane" --dataset criteo --rows 1000000 \
        > "$OUT/speed/$_lane.log" 2>&1
    status "cell.$_lane" $?
    grep -E '^FSPEED' "$OUT/speed/$_lane.log" | grep -v WARMUP
}
cell gbdt-symmetric
cell gbdt-depthwise
cell gbdt-lossguide

# ------------------------------------------------------------------ our CTR cost at the benchmark's scale
# What the CTR path costs OUR arm at 1M rows, with the categorical indices
# declared and withheld, alternating in one process. This is the 2634 question
# on the only dataset that can ask it.
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda \
    PYTHONPATH="$ROOT/python" timeout -k 30 3000 $PY -u tools/criteo_ours_cat_ab.py \
    > "$OUT/ours_cat_ab.log" 2>&1
status ours_cat_ab $?
grep -E '^CATAB' "$OUT/ours_cat_ab.log"

grep -hE '^FSPEED|^BENCH_BINDING|^BENCH_PATHS' "$OUT"/speed/*.log > "$OUT/summary_fspeed.txt" 2>/dev/null
status body_done 0
say "body done"
exit 0
