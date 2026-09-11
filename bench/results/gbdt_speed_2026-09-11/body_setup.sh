#!/bin/sh
# gbdt-speed lane (2026-09-11 night), H100 on RunPod. RUNS ON THE POD from
# /root/mojolearn. Setup only: pixi, IDENTICAL base + gbdt bindings (set
# `baseline`), the GPU opponents (CatBoost, XGBoost; the LightGBM pip wheel has
# no CUDA learner and is not built), taxi and Istella-S caches.
set -u
R=/root/mojolearn; OUT=/root/trees_out; L=$OUT/logs
mkdir -p "$L" /root/bins
cd "$R" || exit 9
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
note() { echo "$* $(date -u +%H:%M:%S)" | tee -a "$OUT/setup.txt"; }
note start commit="$(cat "$R/SHIPPED_COMMIT.txt" 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
{ nproc; grep -m1 'model name' /proc/cpuinfo; uname -a; python3 --version; } > "$OUT/box.txt" 2>&1
(
  python3 -m pip install --no-input -q --disable-pip-version-check catboost xgboost scikit-learn pandas pyarrow > "$L/pip.log" 2>&1; note pip=$?
  python3 -c "import catboost, xgboost, sklearn, numpy; print('catboost', catboost.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)" > "$OUT/versions.txt" 2>&1
  timeout -k 30 1800 python3 tools/speed_gbdt_arm.py --download taxi > "$L/download_taxi.log" 2>&1; note download_taxi=$?
  timeout -k 30 1800 python3 tools/speed_gbdt_arm.py --download istella > "$L/download_istella.log" 2>&1; note download_istella=$?
  : > "$OUT/data.done"
) &
[ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$L/pixi_get.log" 2>&1
timeout -k 30 1500 pixi install > "$L/pixi_install.log" 2>&1; note pixi_install=$?
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
timeout -k 30 1500 bash bindings/build.sh > "$L/build_base.log" 2>&1; note build_base=$?
timeout -k 30 1500 bash bindings/build_gbdt.sh > "$L/build_gbdt.log" 2>&1; note build_gbdt=$?
mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
sha256sum /root/bins/baseline/*.so > "$OUT/bins_baseline.sha256"
: > "$OUT/build.done"
wait
note setup_done
: > "$OUT/setup.done"
