#!/bin/sh
# tools/istella_rank_leg.sh -- lane istella-ranking-bench: the learning-to-rank
# GBDT on Istella-S against CatBoost, XGBoost and LightGBM. RUNS ON THE POD
# from /root/mojolearn, one phase per call, under nohup:
#
#   TREES_LEG_NAME=mojolearn-rank TREES_LEG_CUDA_VERSIONS=13.0 \
#   TREES_LEG_STATE=$HOME/mojolearn-evidence/istella-rank/pod \
#   MOJOLEARN_STAGE_KEYS="gbm-bench/istella/istella_speed.npz gbm-bench/istella/istella_rank.npz" \
#     sh tools/trees_leg.sh rent --gpu "NVIDIA H100 80GB HBM3" --minutes 60
#   sh tools/trees_leg.sh ssh 'cd /root/mojolearn && nohup env RANK_PHASE=setup sh tools/istella_rank_leg.sh > /root/rank_out/setup.console 2>&1 &'
#   ... RANK_PHASE=smoke, then RANK_PHASE=cells, then RANK_PHASE=lgbm_cuda
#   sh tools/trees_leg.sh pull /root/rank_out/ <local dir>; sh tools/trees_leg.sh reap
#
# PHASES
#   setup      box record; pip pins catboost 1.2.10, xgboost 3.2.0,
#              lightgbm 4.7.0 (the versions of the H100 rows in
#              bench/OPPONENT_REFERENCE.md); pixi install; IDENTICAL bindings
#              build.sh and build_gbdt.sh; a LightGBM USE_CUDA source build
#              into /root/lgbm_cuda in the background (8 jobs, 1500 s cap),
#              never replacing the CPU wheel
#   smoke      every cell at 20,000 rows, trees 1,2, one repeat
#   cells      every cell at full size, CELLS overrides the list
#   lgbm_cuda  LightGBM lambdarank on device cuda, only if its probe passed
# Each cell appends name, exit, seconds to status.tsv. POSIX sh, set -u.
set -u
R=/root/mojolearn
OUT=/root/rank_out
L="$OUT/logs"
mkdir -p "$L" "$OUT/cells" "$OUT/smoke"
cd "$R" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export PYTHONPATH="$R/python:$R/tools"
export GBM_BENCH_DATA=/root/datasets/gbm-bench
REPEATS="${RANK_REPEATS:-5}"
note() { echo "$* $(date -u +%H:%M:%S)" | tee -a "$OUT/progress.txt"; }

step() {
    _n=$1; _cap=$2; shift 2
    _t=$(date +%s)
    timeout -k 30 "$_cap" "$@" > "$L/$_n.log" 2>&1
    _rc=$?
    printf '%s\t%s\t%s\n' "$_n" "$_rc" "$(( $(date +%s) - _t ))" >> "$OUT/status.tsv"
    note "$_n=$_rc"
    return $_rc
}

ALL_CELLS="ours:QueryRMSE ours:PairLogit ours:YetiRank catboost:QueryRMSE catboost:PairLogit catboost:YetiRank xgboost:rank:pairwise xgboost:rank:ndcg lightgbm:lambdarank:cpu"

run_cells() {
    _dir=$1; shift
    for c in ${CELLS:-$ALL_CELLS}; do
        lib=${c%%:*}; rest=${c#*:}
        dev=gpu
        case "$rest" in *:cpu) dev=cpu; rest=${rest%:cpu} ;; esac
        tag=$(printf '%s' "$lib-$rest-$dev" | tr ':' '_')
        step "$_dir-$tag" "${RANK_CELL_CAP:-900}" python3 tools/speed_gbdt_rank.py \
            --library "$lib" --loss "$rest" --device "$dev" \
            --json "$OUT/$_dir/$tag.json" "$@"
        cp "$L/$_dir-$tag.log" "$OUT/$_dir/$tag.log" 2>/dev/null
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> "$OUT/$_dir/$tag.smi_after" 2>&1
    done
}

case "${RANK_PHASE:-}" in
setup)
    note start commit="$(cat "$R/SHIPPED_COMMIT.txt" 2>/dev/null)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > "$OUT/gpu.txt" 2>&1
    { nproc; grep -m1 'model name' /proc/cpuinfo; uname -a; python3 --version; free -g | head -2; } > "$OUT/box.txt" 2>&1
    ls -la "$GBM_BENCH_DATA/istella" > "$OUT/data_listing.txt" 2>&1
    (
      step pip_opponents 900 python3 -m pip install --no-input --disable-pip-version-check \
          catboost==1.2.10 xgboost==3.2.0 lightgbm==4.7.0
      python3 -c "import catboost, xgboost, lightgbm, numpy; print('catboost', catboost.__version__, 'xgboost', xgboost.__version__, 'lightgbm', lightgbm.__version__, 'numpy', numpy.__version__); print('catboost gpus', catboost.utils.get_gpu_device_count())" > "$OUT/versions.txt" 2>&1
      : > "$OUT/pip.done"
      command -v cmake > /dev/null 2>&1 || python3 -m pip install -q cmake > "$L/cmake.log" 2>&1
      CMAKE_BUILD_PARALLEL_LEVEL=8 CUDACXX=/usr/local/cuda/bin/nvcc PATH="/usr/local/cuda/bin:$PATH" \
          step lgbm_cuda_build 1500 python3 -m pip install --no-input --no-cache-dir --no-deps \
          --target /root/lgbm_cuda --no-binary lightgbm \
          --config-settings=cmake.define.USE_CUDA=ON lightgbm==4.7.0
      PYTHONPATH=/root/lgbm_cuda timeout 300 python3 -c "
import numpy as np, lightgbm as lgb
x = np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32)
y = (x[:, 0] > 0).astype(np.float32)
lgb.LGBMRegressor(device_type='cuda', n_estimators=1, num_leaves=2, min_child_samples=1, verbose=-1).fit(x, y)
print('LIGHTGBM_CUDA_PROBE=ok', lgb.__file__)" > "$L/lgbm_cuda_probe.log" 2>&1
      : > "$OUT/lgbm.done"
    ) &
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$L/pixi_get.log" 2>&1
    step pixi_install 1500 pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    export MOJOLEARN_COMPILE_JOBS=16
    step build_base 1500 bash bindings/build.sh
    step build_gbdt 1500 bash bindings/build_gbdt.sh
    sha256sum python/mojolearn/identical/*.so > "$OUT/so_sha256.txt" 2>&1
    step import_identical 300 python3 -c "import mojolearn; m = mojolearn.GradientBoosting(); print('import OK', m.numeric_mode_used(), m.vendor_used())"
    step selftest 60 python3 tools/speed_gbdt_rank.py --selftest
    note setup_done
    : > "$OUT/setup.done"
    ;;
smoke)
    run_cells smoke --rows 20000 --trees 1,2 --repeats 1
    : > "$OUT/smoke.done"
    ;;
cells)
    run_cells cells --repeats "$REPEATS"
    : > "$OUT/cells.done"
    ;;
lgbm_cuda)
    if grep -q LIGHTGBM_CUDA_PROBE=ok "$L/lgbm_cuda_probe.log" 2>/dev/null; then
        PYTHONPATH="/root/lgbm_cuda:$PYTHONPATH" CELLS="lightgbm:lambdarank" run_cells cells --repeats "$REPEATS"
    else
        note "lgbm_cuda SKIPPED: probe did not pass ($(tail -1 "$L/lgbm_cuda_probe.log" 2>/dev/null))"
    fi
    : > "$OUT/lgbm_cuda.done"
    ;;
*)
    echo "RANK_PHASE must be setup, smoke, cells or lgbm_cuda" >&2
    exit 2
    ;;
esac
