#!/bin/sh
# RUNS ON THE POD. The baseline sweep: one commit, one pod, one driver, one set
# of library versions, every classical and tree lane, arms interleaved inside
# each cell. Launched detached, one phase per sentinel, so a failure in a late
# phase cannot cost the phases already paid for.
#
#   nohup sh bench/results/baseline_sweep_2026-09-12/pod_body.sh > /root/sweep.console 2>&1 &
#
# Datasets are STAGED FROM THE MAC before this runs (tools/dataset_store.sh
# stage), so nothing here downloads taxi or Istella-S. Phases:
#   setup   pip opponents, pixi, every binding built twice (the gate is circular)
#   sweep   tools/bench_all_ours.sh --rows full --rounds 5 --opponents
#   owed    the three cheap owed items, AFTER the sweep is safely recorded
set -u
ROOT=/root/mojolearn
OUT=/root/trees_out
LOG="$OUT/bench_all/sweep.console"
mkdir -p "$OUT/bench_all" "$OUT/logs" /root/ctd-data /root/ctd-work
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export MOJOLEARN_GPU_ARCHS=sm_90a          # H100; exactly one arch per build
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS

say() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/logs/body.txt"; }
step() { # step <name> <timeout> <cmd...>
    _n="$1"; _t="$2"; shift 2
    _t0=$(date +%s)
    timeout -k 30 "$_t" "$@" > "$OUT/logs/$_n.log" 2>&1
    _rc=$?
    say "$_n rc=$_rc $(( $(date +%s) - _t0 ))s"
    return $_rc
}

# ---------------------------------------------------------------- phase setup
if [ ! -f "$OUT/setup.done" ]; then
    say "PHASE setup"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
        | tee "$OUT/logs/gpu.txt"
    # The opponents all live on the image's system python3 on NVIDIA -- that is
    # where cuML installs -- and pixi is used only to build our bindings.
    step pip_base 900 python3 -m pip install --no-input -q --disable-pip-version-check \
        catboost xgboost lightgbm scikit-learn pandas pyarrow threadpoolctl
    step pip_cuml 1500 python3 -m pip install --no-input -q --disable-pip-version-check \
        --extra-index-url=https://pypi.nvidia.com cuml-cu12==26.8.0
    # lightgbm-cuda needs a source build; it is a compiler workload that must
    # never overlap a timed cell, so it happens here or not at all.
    if [ "${SWEEP_LIGHTGBM_CUDA:-0}" = 1 ]; then
        CMAKE_ARGS="-DUSE_CUDA=ON" CUDACXX=/usr/local/cuda/bin/nvcc PATH="/usr/local/cuda/bin:$PATH" \
        step lightgbm_cuda 2400 python3 -m pip install --no-input --no-cache-dir \
            --force-reinstall --no-deps --no-binary lightgbm \
            --config-settings=cmake.define.USE_CUDA=ON lightgbm
    fi
    step pixi_install 900 pixi install
    # EVERY binding, twice. A lane whose extension is missing REFUSES by name
    # rather than running, and the build gate is circular (it imports the tier
    # it is building), so pass 1 runs with the gate off and pass 2 with it live.
    _b="bindings/build.sh $(ls bindings/build_*.sh 2>/dev/null | tr '\n' ' ')"
    for _pass in 1 2; do
        [ "$_pass" = 1 ] && _skip=1 || _skip=0
        for _s in $_b; do
            MOJOLEARN_SKIP_BUILD_GATE="$_skip" \
                step "build.p$_pass.$(basename "$_s" .sh)" 1500 bash "$_s"
        done
    done
    # The witness that the tier and vendor we are about to time are the ones
    # that got built. A silent import failure here would be a whole wasted pod.
    step import_witness 300 python3 -c "
import mojolearn
m = mojolearn.RandomForestClassifier(device='gpu')
print('numeric_mode_used', m.numeric_mode_used())
print('vendor_used', m.vendor_used())
print('version', getattr(mojolearn, '__version__', '?'))"
    step versions 120 python3 -c "
import cuml, cupy, sklearn, numpy, scipy, catboost, xgboost
print('cuml', cuml.__version__); print('cupy', cupy.__version__)
print('sklearn', sklearn.__version__); print('numpy', numpy.__version__)
print('scipy', scipy.__version__); print('catboost', catboost.__version__)
print('xgboost', xgboost.__version__)
try:
    import lightgbm; print('lightgbm', lightgbm.__version__)
except Exception as e: print('lightgbm UNAVAILABLE', e)
import torch; print('torch', torch.__version__)"
    touch "$OUT/setup.done"
    say "PHASE setup done"
fi

# ---------------------------------------------------------------- phase sweep
# THE BOARD. Everything serializes: two timed arms on one GPU corrupt both.
if [ ! -f "$OUT/sweep.done" ]; then
    say "PHASE sweep (this is the owed run; do not reap while it is in flight)"
    _t0=$(date +%s)
    sh tools/bench_all_ours.sh --rows full --rounds 5 --opponents \
        >> "$LOG" 2>&1
    say "sweep rc=$? $(( $(date +%s) - _t0 ))s"
    touch "$OUT/sweep.done"
    say "PHASE sweep done"
fi

# ----------------------------------------------------------------- phase owed
# AFTER the sweep is recorded, so a failure here cannot cost the board.
if [ ! -f "$OUT/owed.done" ]; then
    say "PHASE owed"
    # (1) cuML's forest JSON schema: settles why every rf/et fit-equivalence
    # cell reads UNKNOWN. One minute, no dataset.
    step owed_cuml_json 300 python3 tools/cuml_forest_json_probe.py
    # (2) the _buffer.py ctypes clash against a REAL cuML, owed on NVIDIA since
    # 0.8.1 shipped the fix.
    step owed_buffer_argtypes 300 python3 tools/check_buffer_foreign_argtypes.py --real-cuml
    # (3) DEVIATION 2634 on criteo, now attributable: the fit prints the
    # [ctr-2634] marker under MOJOLEARN_CTR_TRACE=1, which
    # bench/speed/forest_speed_arm.py sets on every run. criteo is NOT in the
    # R2 store, so this pays a ~291 MB fetch and a decode; it is last on
    # purpose and skipped unless asked for.
    if [ "${SWEEP_CRITEO:-0}" = 1 ]; then
        step owed_criteo_dl 2400 python3 tools/speed_gbdt_arm.py --download criteo
        MOJOLEARN_CTR_TRACE=1 MOJOLEARN_SPEED_ARMS=catboost-gpu MOJOLEARN_SPEED_TAG=ctr2634 \
            step owed_criteo_ab 3600 sh tools/trees_identical_ab.sh \
                speed all gbdt-symmetric criteo 1000000 3 full
        grep -h 'ctr-2634' "$OUT"/speed/*criteo* "$OUT/logs/owed_criteo_ab.log" 2>/dev/null \
            | head -5 | tee "$OUT/logs/ctr2634_marker.txt"
    fi
    touch "$OUT/owed.done"
    say "PHASE owed done"
fi

say "ALL PHASES DONE"
echo "=== board ==="
cat "$OUT/bench_all/board.tsv" 2>/dev/null || echo "(no board.tsv)"
