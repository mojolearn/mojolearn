#!/bin/sh
# RUNS ON THE POD (tools/trees_leg.sh). Sets a fresh NVIDIA box up for the
# trees lane: pixi, the IDENTICAL-tier bindings this lane measures (base,
# gbdt, rf, trees), the opponents (CatBoost GPU, cuML RF, LightGBM built
# with USE_CUDA=ON), and the HIGGS fixture. Each step is bounded and logged
# under /root/trees_out/logs; exit codes land in /root/trees_out/setup.txt.
#
#   nohup sh tools/trees_identical_remote.sh > /root/trees_out/setup_console.log 2>&1 &
#
# The three independent tracks (pixi+bindings, pip+higgs, LightGBM CUDA)
# run concurrently; the sentinel /root/trees_out/setup.done is written last.
set -u
ROOT=/root/mojolearn
OUT=/root/trees_out
LOGS="$OUT/logs"
mkdir -p "$LOGS"
cd "$ROOT" || exit 9
rm -f "$OUT/setup.done"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat SHIPPED_COMMIT.txt 2>/dev/null)" > "$OUT/setup.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt" 2>&1
uname -a > "$OUT/uname.txt"
nvcc --version > "$OUT/nvcc.txt" 2>&1

step() {  # <name> <seconds> <cmd...>
    _n="$1"; _s="$2"; shift 2
    timeout -k 30 "$_s" "$@" > "$LOGS/$_n.log" 2>&1
    _rc=$?
    echo "$_n=$_rc $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    return $_rc
}

track_mojo() {
    if [ ! -x "$HOME/.pixi/bin/pixi" ]; then
        curl -fsSL https://pixi.sh/install.sh | sh > "$LOGS/pixi_install.log" 2>&1
    fi
    PATH="$HOME/.pixi/bin:$PATH"; export PATH
    step pixi_install 1500 pixi install
    # IDENTICAL tier, gates skipped (the gate imports the whole package, which
    # needs every sibling binding; the harness below is the real exercise).
    export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
    export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-4}"
    step build_base 1200 bash bindings/build.sh
    step build_gbdt 1500 bash bindings/build_gbdt.sh
    step build_rf 1200 bash bindings/build_rf.sh
    step build_trees 1200 bash bindings/build_trees.sh
    ls -la python/mojolearn/identical/ > "$OUT/bindings_listing.txt" 2>&1
    ( cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -c "import mojolearn, numpy; print('import OK', mojolearn.__file__)" ) \
        > "$LOGS/import_identical.log" 2>&1
    echo "import_identical=$? $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    : > "$OUT/track_mojo.done"
}

track_pip() {
    step pip_base 900 python3 -m pip install --no-input --disable-pip-version-check \
        catboost lightgbm scikit-learn pandas
    step pip_cuml 1200 python3 -m pip install --no-input --disable-pip-version-check \
        --extra-index-url=https://pypi.nvidia.com cuml-cu12
    python3 -c "import catboost, lightgbm, sklearn, numpy; print('catboost', catboost.__version__, 'lightgbm', lightgbm.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)" > "$OUT/versions.txt" 2>&1
    python3 -c "import cuml; print('cuml', cuml.__version__)" >> "$OUT/versions.txt" 2>&1
    python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" >> "$OUT/versions.txt" 2>&1
    : > "$OUT/track_pip_base.done"
    # HIGGS from UCI's static zip: measured 6-8 MB/s per connection on
    # 2026-09-09 where the /ml/machine-learning-databases path gave 0.2 MB/s.
    # Same bytes (the zip holds the same HIGGS.csv.gz); decoded by the
    # harness's own --download step into higgs_speed.npz.
    _hd=/root/datasets/gbm-bench/higgs
    mkdir -p "$_hd"
    if [ ! -s "$_hd/HIGGS.csv.gz" ]; then
        step higgs_zip 1800 curl -sSL --retry 3 -o "$_hd/higgs.zip" https://archive.ics.uci.edu/static/public/280/higgs.zip
        step higgs_unzip 600 python3 -c "import zipfile; zipfile.ZipFile('$_hd/higgs.zip').extract('HIGGS.csv.gz', '$_hd')"
        rm -f "$_hd/higgs.zip"
    fi
    step download_higgs 2400 python3 tools/speed_gbdt_arm.py --download higgs
    : > "$OUT/track_pip.done"
}

track_lgbm() {
    # wait for the base pip install (lightgbm's build wants numpy/scikit-build)
    while [ ! -f "$OUT/track_pip_base.done" ]; do sleep 15; done
    rm -rf /root/.cache/pip/wheels 2>/dev/null || true
    CMAKE_ARGS="-DUSE_CUDA=ON" CUDACXX=/usr/local/cuda/bin/nvcc PATH="/usr/local/cuda/bin:$PATH" \
        step lightgbm_cuda_build 2400 python3 -m pip install --no-input --no-cache-dir \
            --force-reinstall --no-deps --no-binary lightgbm \
            --config-settings=cmake.define.USE_CUDA=ON lightgbm
    python3 - > "$LOGS/lightgbm_cuda_probe.log" 2>&1 <<'PY'
import numpy as np, lightgbm as lgb
x = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
y = (x[:, 0] > 0).astype(np.float32)
m = lgb.LGBMRegressor(device="cuda", n_estimators=1, num_leaves=2, min_child_samples=1, verbose=-1)
m.fit(x, y)
print("LIGHTGBM_CUDA_PROBE=ok", lgb.__version__)
PY
    if grep -q LIGHTGBM_CUDA_PROBE=ok "$LOGS/lightgbm_cuda_probe.log"; then
        echo "lightgbm_cuda_works=yes" >> "$OUT/setup.txt"
    else
        echo "lightgbm_cuda_works=NO $(tail -1 "$LOGS/lightgbm_cuda_probe.log")" >> "$OUT/setup.txt"
    fi
    : > "$OUT/track_lgbm.done"
}

track_mojo &
track_pip &
track_lgbm &
wait
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/setup.txt"
: > "$OUT/setup.done"
