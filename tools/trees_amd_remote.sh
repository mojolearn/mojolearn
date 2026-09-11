#!/bin/sh
# RUNS ON THE DIGITALOCEAN AMD DROPLET (tools/trees_amd_leg.sh). Sets a fresh
# MI325X up for the trees lane under ENGINEERING_RULES.md section 10: pixi,
# the IDENTICAL-tier bindings (base, gbdt, rf, trees), the opponents that run
# on this box (CatBoost, scikit-learn, LightGBM and XGBoost on the CPU; AMD's
# ROCm XGBoost build `amd_xgboost` and LightGBM's OpenCL learner on the GPU
# when they install and a probe proves the GPU ran), and the two section 9
# datasets onto the persistent volume. Every step is bounded and logged under
# /root/trees_out/logs; exit codes land in /root/trees_out/setup.txt.
#
#   nohup sh tools/trees_amd_remote.sh > /root/trees_out/setup_console.log 2>&1 &
#
# Tracks run concurrently; /root/trees_out/setup.done is written last.
#   MOJOLEARN_TREES_SKIP_XGB_ROCM=1       skip the amd_xgboost attempt
#   MOJOLEARN_TREES_SKIP_LGBM_OPENCL=1    skip the LightGBM USE_GPU build
# POSIX sh (dash on Ubuntu).
set -u
ROOT=/root/mojolearn
OUT=/root/trees_out
LOGS="$OUT/logs"
GBM_BENCH_DATA="${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"
export GBM_BENCH_DATA
VENV=/root/venv-gpu
mkdir -p "$LOGS" "$GBM_BENCH_DATA"
cd "$ROOT" || exit 9
rm -f "$OUT/setup.done" "$OUT"/track_*.done
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat SHIPPED_COMMIT.txt 2>/dev/null) data=$GBM_BENCH_DATA" > "$OUT/setup.txt"
export DEBIAN_FRONTEND=noninteractive

step() {  # <name> <seconds> <cmd...>
    _n="$1"; _s="$2"; shift 2
    timeout -k 30 "$_s" "$@" > "$LOGS/$_n.log" 2>&1
    _rc=$?
    echo "$_n=$_rc $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    return $_rc
}

box_facts() {
    { rocm-smi --showproductname --showdriverversion; rocm-smi --showmeminfo vram; } > "$OUT/gpu.txt" 2>&1
    { cat /opt/rocm/.info/version; ls -d /opt/rocm-*; } > "$OUT/rocm_version.txt" 2>&1
    { echo "sysfs $(cat /sys/module/amdgpu/version 2>/dev/null)"; modinfo amdgpu 2>/dev/null | grep -E '^(version|filename|vermagic):'; } > "$OUT/amdgpu.txt" 2>&1
    lscpu > "$OUT/lscpu.txt" 2>&1
    echo "nproc $(nproc)" > "$OUT/cpu.txt"
    grep -m1 'model name' /proc/cpuinfo >> "$OUT/cpu.txt"
    free -g > "$OUT/free.txt" 2>&1
    uname -a > "$OUT/uname.txt"
    cat /etc/os-release > "$OUT/os.txt" 2>&1
    python3 --version > "$OUT/python.txt" 2>&1
    df -h / /mnt/mojolearn-data > "$OUT/df.txt" 2>&1
}

track_mojo() {
    if [ ! -x "$HOME/.pixi/bin/pixi" ]; then
        curl -fsSL https://pixi.sh/install.sh | sh > "$LOGS/pixi_get.log" 2>&1
    fi
    PATH="$HOME/.pixi/bin:$PATH"; export PATH
    step pixi_install 1500 pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
    export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
    step build_base 1500 bash bindings/build.sh
    step build_gbdt 1500 bash bindings/build_gbdt.sh
    step build_rf 1500 bash bindings/build_rf.sh
    step build_trees 1500 bash bindings/build_trees.sh
    ls -la python/mojolearn/identical/ > "$OUT/bindings_listing.txt" 2>&1
    mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
    sha256sum /root/bins/baseline/*.so > "$OUT/setup_so_sha256.txt" 2>&1
    : > "$OUT/track_mojo.done"
}

track_pip() {
    if ! python3 -c 'import pip, venv, ensurepip' > /dev/null 2>&1; then
        step apt_pip 600 sh -c 'apt-get -o DPkg::Lock::Timeout=180 update -qq; apt-get -o DPkg::Lock::Timeout=180 install -y --no-install-recommends python3-pip python3-venv'
    fi
    step pip_base 900 python3 -m pip install --break-system-packages --no-input --disable-pip-version-check \
        numpy pandas pyarrow scikit-learn catboost lightgbm xgboost
    python3 -c "import catboost, lightgbm, sklearn, numpy, xgboost; print('catboost', catboost.__version__, 'lightgbm', lightgbm.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'xgboost(pypi)', xgboost.__version__)" > "$OUT/versions.txt" 2>&1
    : > "$OUT/track_pip_base.done"
    # The two section 9 datasets, onto the volume; a decoded cache there is
    # reused (the loaders read the .npz and skip the fetch).
    step download_taxi 1800 python3 tools/speed_gbdt_arm.py --download taxi
    step download_istella 2400 python3 tools/speed_gbdt_arm.py --download istella
    ls -la "$GBM_BENCH_DATA"/taxi "$GBM_BENCH_DATA"/istella > "$OUT/datasets_listing.txt" 2>&1
    : > "$OUT/track_pip.done"
}

track_import() {
    while [ ! -f "$OUT/track_mojo.done" ] || [ ! -f "$OUT/track_pip_base.done" ]; do sleep 10; done
    ( cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -c "import mojolearn, numpy; print('import OK', mojolearn.__file__)" ) \
        > "$LOGS/import_identical.log" 2>&1
    echo "import_identical=$? $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    : > "$OUT/track_import.done"
}

gpu_probe() {  # <python> <label>: XGBoost on device cuda beside a rocm-smi sampler
    ( while :; do echo "t $(date -u +%T)"; rocm-smi --showuse --showmemuse 2>/dev/null | grep -i 'GPU use\|VRAM'; sleep 1; done ) > "$LOGS/$2_smi.log" 2>&1 &
    _smi=$!
    timeout -k 10 300 "$1" - > "$LOGS/$2.log" 2>&1 <<'PY'
import json, time, warnings
import numpy as np
import xgboost as xgb
print("xgboost", xgb.__version__, xgb.__file__)
print("build_info", json.dumps({k: v for k, v in xgb.build_info().items()
                                if k.startswith("USE") or "ROCM" in k.upper() or "HIP" in k.upper()}))
rng = np.random.default_rng(0)
x = rng.normal(size=(1_000_000, 50)).astype(np.float32)
y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.float32)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for dev in ("cuda", "cpu", "cuda"):
        t = time.perf_counter()
        m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist",
                              device=dev, verbosity=1)
        m.fit(x, y)
        ms = (time.perf_counter() - t) * 1e3
        try:
            cfg_dev = json.loads(m.get_booster().save_config())["learner"]["generic_param"]["device"]
        except Exception as exc:                   # noqa: BLE001
            cfg_dev = "unreadable(%s)" % exc.__class__.__name__
        print("fit device=%s %.1f ms config_device=%s" % (dev, ms, cfg_dev))
    for item in w:
        print("WARNING", item.message)
print("XGB_GPU_PROBE_DONE")
PY
    _rc=$?
    kill "$_smi" 2>/dev/null
    echo "$2=$_rc $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
}

track_gpu_opponents() {
    while [ ! -f "$OUT/track_pip_base.done" ]; do sleep 10; done
    python3 -m venv --system-site-packages "$VENV" > "$LOGS/venv.log" 2>&1
    echo "venv=$? $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    _rv="$(cut -d. -f1 /opt/rocm/.info/version 2>/dev/null)"
    if [ "$_rv" = 7 ]; then _idx=https://pypi.amd.com/rocm-7.0.2/simple; else _idx=https://pypi.amd.com/rocm-6.4.4/simple; fi
    echo "xgb_rocm_index=$_idx (rocm major '$_rv')" >> "$OUT/setup.txt"
    if [ "${MOJOLEARN_TREES_SKIP_XGB_ROCM:-0}" != 1 ]; then
        # The brief's bound: twenty minutes of box time for this attempt.
        step pip_amd_xgboost 1200 "$VENV/bin/python" -m pip install --no-input --disable-pip-version-check \
            --extra-index-url "$_idx" amd_xgboost
        gpu_probe "$VENV/bin/python" xgb_rocm_probe
        # Works = the probe finished, no "not compiled"/fallback warning,
        # and rocm-smi saw GPU use while it ran (nothing else touches the
        # GPU during setup). The fits' config device is printed beside it.
        if grep -q XGB_GPU_PROBE_DONE "$LOGS/xgb_rocm_probe.log" \
           && ! grep -qi 'not compiled\|falling back\|fall back\|changed from GPU\|no visible GPU' "$LOGS/xgb_rocm_probe.log" \
           && awk -F: '/GPU use/ { v = $NF; gsub(/[^0-9]/, "", v); if (v + 0 > 0) f = 1 } END { exit !f }' "$LOGS/xgb_rocm_probe_smi.log"; then
            echo "xgb_rocm_works=yes" >> "$OUT/setup.txt"
        else
            echo "xgb_rocm_works=NO" >> "$OUT/setup.txt"
        fi
    else
        echo "xgb_rocm_works=SKIPPED" >> "$OUT/setup.txt"
    fi
    "$VENV/bin/python" -c "import xgboost, lightgbm, catboost, sklearn; print('venv xgboost', xgboost.__version__, xgboost.__file__); print('venv lightgbm', lightgbm.__version__, lightgbm.__file__)" >> "$OUT/versions.txt" 2>&1
    if [ "${MOJOLEARN_TREES_SKIP_LGBM_OPENCL:-0}" != 1 ]; then
        # Lowest priority, fifteen minutes including the build dependencies.
        _t0=$(date +%s)
        step apt_lgbm_opencl 420 sh -c 'apt-get -o DPkg::Lock::Timeout=180 update -qq; apt-get -o DPkg::Lock::Timeout=180 install -y --no-install-recommends cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev ocl-icd-opencl-dev opencl-headers clinfo'
        clinfo -l > "$LOGS/clinfo.log" 2>&1
        _left=$((900 - $(date +%s) + _t0))
        [ "$_left" -lt 60 ] && _left=60
        step lightgbm_opencl_build "$_left" "$VENV/bin/python" -m pip install --no-input --disable-pip-version-check \
            --no-cache-dir --force-reinstall --no-deps --no-binary lightgbm \
            --config-settings=cmake.define.USE_GPU=ON lightgbm
        timeout -k 10 180 "$VENV/bin/python" - > "$LOGS/lightgbm_opencl_probe.log" 2>&1 <<'PY'
import numpy as np, lightgbm as lgb
x = np.random.default_rng(0).normal(size=(20000, 8)).astype(np.float32)
y = (x[:, 0] > 0).astype(np.float32)
m = lgb.LGBMClassifier(device_type="gpu", n_estimators=5, num_leaves=8, verbose=1)
m.fit(x, y)
print("LIGHTGBM_OPENCL_PROBE=ok", lgb.__version__)
PY
        if grep -q LIGHTGBM_OPENCL_PROBE=ok "$LOGS/lightgbm_opencl_probe.log"; then
            echo "lightgbm_opencl_works=yes" >> "$OUT/setup.txt"
        else
            echo "lightgbm_opencl_works=NO" >> "$OUT/setup.txt"
        fi
    else
        echo "lightgbm_opencl_works=SKIPPED" >> "$OUT/setup.txt"
    fi
    : > "$OUT/track_gpu_opponents.done"
}

box_facts
track_mojo &
track_pip &
track_import &
track_gpu_opponents &
wait
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/setup.txt"
: > "$OUT/setup.done"
