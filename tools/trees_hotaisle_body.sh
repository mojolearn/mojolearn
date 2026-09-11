#!/bin/sh
# RUNS ON THE HOT AISLE MI300X VM, as the leg body of tools/hotaisle_leg.sh
# (lane trees-hotaisle, 2026-09-11). Mirrors tools/trees_amd_remote.sh and the
# MI325X batches (bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/
# logs/). A VM lives 60 minutes and has no shared volume.
#
#   MOJOLEARN_HOTAISLE_SPEC=13core MOJOLEARN_HOTAISLE_LANE=trees-hotaisle \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/trees_hotaisle_body.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/trees-hotaisle/<leg> \
#   MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_TREES_HA_LEG=taxi' \
#   bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
#
#   MOJOLEARN_TREES_HA_LEG=taxi|istella|taxi,istella   datasets, in order
#   MOJOLEARN_TREES_HA_CELLS=all|xgb   all (default): every cell below.
#       xgb: only depthwise and lossguide against XGBoost on the GPU, ours
#       interleaved, rocm-smi sampled per cell; refuses before any timing
#       unless amd_xgboost installed and its probe ran on the GPU. AMD ships
#       amd_xgboost only as manylinux_2_39 wheels, so the runtime needs glibc
#       2.39 (MOJOLEARN_HOTAISLE_IMAGE=rocm/dev-ubuntu-24.04:6.4.1-complete);
#       the default ubuntu-22.04 image has 2.35 and pip finds no wheel.
#   MOJOLEARN_TREES_HA_LGBM_OPENCL=1|0      try the LightGBM OpenCL build (default 1)
#   MOJOLEARN_TREES_HA_OUT=<dir>            default /root/gemm_leg_out (the runner fetches it)
#
# Setup (untimed, before any cell): box facts; the IDENTICAL bindings base,
# gbdt, rf, trees and (cells all) the FAST bindings gbdt, rf, trees; a Python
# 3.12 venv with CatBoost, scikit-learn, LightGBM and AMD's ROCm XGBoost build
# `amd_xgboost` (GPU probe beside rocm-smi); (cells all) the ONE LightGBM
# retry (min_child_weight 1e-3, an OpenCL build tried inside a 15 minute
# cap); the datasets, sha256 checked against the Mac's copies. Then per
# dataset, 1 warm-up plus 5 rounds, arms interleaved round by round,
# opponents imported first: RF 1M, ET 1M, RF 2M (scikit-learn CPU),
# symmetric (CatBoost CPU), depthwise (CatBoost CPU, XGBoost), lossguide
# (CatBoost CPU, XGBoost, LightGBM), then FAST beside IDENTICAL for the five
# lanes, ours only. Every FSPEED line carries its round's hash. POSIX sh.
set -u
LEG="${MOJOLEARN_TREES_HA_LEG:-}"
LEGS="$(printf '%s' "$LEG" | tr ',' ' ')"
[ -n "$LEGS" ] || { echo "MOJOLEARN_TREES_HA_LEG must name taxi, istella or taxi,istella"; exit 2; }
for _d in $LEGS; do
    case "$_d" in
        taxi|istella) ;;
        *) echo "MOJOLEARN_TREES_HA_LEG: unknown dataset '$_d' (taxi, istella)"; exit 2 ;;
    esac
done
CELLS="${MOJOLEARN_TREES_HA_CELLS:-all}"
case "$CELLS" in
    all|xgb) ;;
    *) echo "MOJOLEARN_TREES_HA_CELLS must be all or xgb (got '$CELLS')"; exit 2 ;;
esac
[ "$(id -u)" = 0 ] || { echo "the body needs root (/root paths); id -u is $(id -u)"; exit 5; }
ROOT=/root/mojolearn
LEGOUT="${MOJOLEARN_TREES_HA_OUT:-/root/gemm_leg_out}"
mkdir -p "$LEGOUT/trees_out"
# The helper writes /root/trees_out; point it inside what the runner fetches.
if [ ! -L /root/trees_out ]; then
    [ -e /root/trees_out ] && mv /root/trees_out "/root/trees_out.prev.$$"
    ln -s "$LEGOUT/trees_out" /root/trees_out
fi
OUT=/root/trees_out
LOGS="$OUT/logs"
DATA=/root/datasets/gbm-bench
VENV=/root/venv-gpu
PY="$VENV/bin/python"
AB="sh tools/trees_identical_ab.sh"
export GBM_BENCH_DATA="$DATA"
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$LOGS" "$DATA" "$OUT/speed"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) datasets=$LEGS cells=$CELLS commit=$(cat SHIPPED_COMMIT.txt 2>/dev/null) gpu_archs=${MOJOLEARN_GPU_ARCHS:-} target_column=${MOJOLEARN_TARGET_COLUMN:-}" > "$OUT/setup.txt"

# The Mac's copies (the MI325X leg uploaded the same taxi bytes).
TAXI_SHA_01=c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510
TAXI_SHA_02=c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20
ISTELLA_SHA=41b21116a3650cc043dbe16f02ee39f4467f9405b37fdbcc9a6a05e230a38981
ISTELLA_BYTES=472129615
UA="Mozilla/5.0 (X11; Linux x86_64) mojolearn-bench"

step() {  # <name> <seconds> <cmd...>
    _n="$1"; _s="$2"; shift 2
    timeout -k 30 "$_s" "$@" > "$LOGS/$_n.log" 2>&1
    _rc=$?
    echo "$_n=$_rc $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    return $_rc
}
mark() { echo "$1 $(date -u +%T)" | tee -a "$OUT/ab.txt"; : > "$OUT/phase.$1"; }
wait_for() { for _f in "$@"; do while [ ! -f "$_f" ]; do sleep 10; done; done; }
sha_of() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }

box_facts() {
    { rocm-smi --showproductname --showdriverversion; rocm-smi --showmeminfo vram; } > "$OUT/gpu.txt" 2>&1
    rocminfo > "$LOGS/rocminfo.log" 2>&1
    { echo "userland (the body's runtime) $(cat /opt/rocm/.info/version 2>/dev/null)"; ls -d /opt/rocm-* 2>/dev/null; } > "$OUT/rocm_version.txt" 2>&1
    { echo "sysfs $(cat /sys/module/amdgpu/version 2>/dev/null)"; modinfo amdgpu 2>/dev/null | grep -E '^(version|filename|vermagic):'; } > "$OUT/amdgpu.txt" 2>&1
    lscpu > "$OUT/lscpu.txt" 2>&1
    { echo "nproc $(nproc)"; grep -m1 'model name' /proc/cpuinfo; echo "cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; } > "$OUT/cpu.txt"
    free -g > "$OUT/free.txt" 2>&1
    uname -a > "$OUT/uname.txt"
    cat /etc/os-release > "$OUT/os.txt" 2>&1
    { ldd --version 2>&1 | head -1; } > "$OUT/glibc.txt"
    df -h / /root > "$OUT/df.txt" 2>&1
}

track_mojo() {
    if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi > /dev/null 2>&1; then
        curl -fsSL https://pixi.sh/install.sh | sh > "$LOGS/pixi_get.log" 2>&1
    fi
    step pixi_install 1200 pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8
    step build_base 1500 bash bindings/build.sh
    step build_gbdt 1500 bash bindings/build_gbdt.sh
    step build_rf 1500 bash bindings/build_rf.sh
    step build_trees 1500 bash bindings/build_trees.sh
    ls -la python/mojolearn/identical/ > "$OUT/bindings_listing.txt" 2>&1
    mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
    sha256sum /root/bins/baseline/*.so > "$OUT/setup_so_sha256.txt" 2>&1
    if [ "$CELLS" = all ]; then
        # FAST (base ships IDENTICAL only, DEVIATION 2490).
        for _b in gbdt rf trees; do
            step "build_fast_$_b" 1500 env MOJOLEARN_NUMERIC_MODE=fast bash "bindings/build_$_b.sh"
        done
        sha256sum python/mojolearn/*.so > "$OUT/fast_so_sha256.txt" 2>&1
    fi
    : > "$OUT/track_mojo.done"
}

gpu_probe() {  # XGBoost on device cuda beside a rocm-smi sampler
    ( while :; do echo "t $(date -u +%T)"; rocm-smi --showuse --showmemuse 2>/dev/null | grep -i 'GPU use\|VRAM'; sleep 1; done ) > "$LOGS/xgb_rocm_probe_smi.log" 2>&1 &
    _smi=$!
    timeout -k 10 300 "$PY" - > "$LOGS/xgb_rocm_probe.log" 2>&1 <<'PYEOF'
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
PYEOF
    echo "xgb_rocm_probe=$? $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    kill "$_smi" 2>/dev/null
    if grep -q XGB_GPU_PROBE_DONE "$LOGS/xgb_rocm_probe.log" \
       && ! grep -qi 'not compiled\|falling back\|fall back\|changed from GPU\|no visible GPU' "$LOGS/xgb_rocm_probe.log" \
       && awk -F: '/GPU use/ { v = $NF; gsub(/[^0-9]/, "", v); if (v + 0 > 0) f = 1 } END { exit !f }' "$LOGS/xgb_rocm_probe_smi.log"; then
        echo "xgb_rocm_works=yes" >> "$OUT/setup.txt"
    else
        echo "xgb_rocm_works=NO" >> "$OUT/setup.txt"
    fi
}

track_py() {
    # The MI325X route first: this runtime's python3 when it is 3.12 (the
    # MI325X ran 3.12.3) and can make a venv. Otherwise uv supplies CPython
    # 3.12 and a seeded venv without apt. Every package comes from pip.
    _sys="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
    if [ "$_sys" = 3.12 ] && step venv 300 python3 -m venv "$VENV" && [ -x "$PY" ] && "$PY" -m pip --version > /dev/null 2>&1; then
        echo "venv_via=system python3 $_sys" >> "$OUT/setup.txt"
    else
        rm -rf "$VENV"
        if [ ! -x /root/.uvbin/uv ]; then
            curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.uvbin UV_NO_MODIFY_PATH=1 sh > "$LOGS/uv_get.log" 2>&1
        fi
        step venv_uv 600 /root/.uvbin/uv venv --seed --python 3.12 "$VENV"
        echo "venv_via=uv CPython 3.12 (this runtime's python3 is '$_sys')" >> "$OUT/setup.txt"
    fi
    # Mojo loads a libpython; make it the venv interpreter's, not python3 on PATH.
    "$PY" -c "import os, sysconfig; d = sysconfig.get_config_var('LIBDIR') or ''; n = sysconfig.get_config_var('INSTSONAME') or sysconfig.get_config_var('LDLIBRARY') or ''; print(os.path.join(d, n))" > "$OUT/libpython.txt" 2>&1
    step pip_base 900 "$PY" -m pip install --no-input --disable-pip-version-check \
        numpy pandas pyarrow scikit-learn catboost lightgbm joblib
    : > "$OUT/track_pip_base.done"
    _rv="$(cut -d. -f1 /opt/rocm/.info/version 2>/dev/null)"
    if [ "$_rv" = 7 ]; then _idx=https://pypi.amd.com/rocm-7.0.2/simple; else _idx=https://pypi.amd.com/rocm-6.4.4/simple; fi
    echo "xgb_rocm_index=$_idx (rocm major '$_rv')" >> "$OUT/setup.txt"
    if step pip_amd_xgboost 900 "$PY" -m pip install --no-input --disable-pip-version-check \
            --extra-index-url "$_idx" amd_xgboost; then
        gpu_probe
    else
        echo "xgb_rocm_works=NO (install failed)" >> "$OUT/setup.txt"
    fi
    if ! grep -q '^xgb_rocm_works=yes' "$OUT/setup.txt" && [ "$CELLS" = all ]; then
        # No AMD GPU path installed: the PyPI build on the CPU, labeled CPU.
        step pip_xgboost_cpu 600 "$PY" -m pip install --no-input --disable-pip-version-check --force-reinstall xgboost
    fi
    "$PY" -c "import sys, numpy, sklearn, catboost, lightgbm, joblib; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__); print('catboost', catboost.__version__); print('lightgbm', lightgbm.__version__, lightgbm.__file__); print('joblib.cpu_count', joblib.cpu_count())" > "$OUT/versions.txt" 2>&1
    "$PY" -c "import xgboost; print('xgboost', xgboost.__version__, xgboost.__file__)" >> "$OUT/versions.txt" 2>&1
    : > "$OUT/track_py.done"
}

track_lgbm() {
    # THE ONE LightGBM RETRY, 15 minutes from here, no second attempt.
    wait_for "$OUT/track_py.done"
    if [ "$CELLS" != all ] || [ "${MOJOLEARN_TREES_HA_LGBM_OPENCL:-1}" != 1 ]; then
        echo "lightgbm_opencl_works=SKIPPED (cells=$CELLS, MOJOLEARN_TREES_HA_LGBM_OPENCL=${MOJOLEARN_TREES_HA_LGBM_OPENCL:-1}; the one OpenCL attempt ran on another leg)" >> "$OUT/setup.txt"
        : > "$OUT/track_lgbm.done"
        return 0
    fi
    _t0=$(date +%s)
    _wheel="$("$PY" -c 'import lightgbm; print(lightgbm.__version__)' 2>/dev/null)"
    step apt_lgbm_opencl 420 sh -c 'apt-get -o DPkg::Lock::Timeout=180 update -qq; apt-get -o DPkg::Lock::Timeout=180 install -y --no-install-recommends cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev ocl-icd-opencl-dev opencl-headers clinfo'
    clinfo -l > "$LOGS/clinfo.log" 2>&1
    _left=$((900 - $(date +%s) + _t0))
    if grep -q 'gfx' "$LOGS/clinfo.log" && [ "$_left" -gt 120 ]; then
        step lightgbm_opencl_build "$((_left - 90))" "$PY" -m pip install --no-input --disable-pip-version-check \
            --no-cache-dir --force-reinstall --no-deps --no-binary lightgbm \
            --config-settings=cmake.define.USE_GPU=ON "lightgbm==$_wheel"
        _left=$((900 - $(date +%s) + _t0)); [ "$_left" -lt 30 ] && _left=30
        timeout -k 10 "$_left" "$PY" - > "$LOGS/lightgbm_opencl_probe.log" 2>&1 <<'PYEOF'
import numpy as np, lightgbm as lgb
rng = np.random.default_rng(0)
x = rng.normal(size=(200000, 16)).astype(np.float32)
y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(np.float32)
m = lgb.LGBMClassifier(device_type="gpu", n_estimators=20, num_leaves=64, max_depth=6,
                       min_child_samples=1, min_child_weight=1e-3, min_split_gain=0.0,
                       max_bin=255, gpu_use_dp=True, verbose=1)
m.fit(x, y)
print("LIGHTGBM_OPENCL_PROBE=ok", lgb.__version__)
PYEOF
        if grep -q LIGHTGBM_OPENCL_PROBE=ok "$LOGS/lightgbm_opencl_probe.log"; then
            echo "lightgbm_opencl_works=yes" >> "$OUT/setup.txt"
        else
            echo "lightgbm_opencl_works=NO" >> "$OUT/setup.txt"
            step lightgbm_wheel_restore 300 "$PY" -m pip install --no-input --disable-pip-version-check \
                --force-reinstall --no-deps "lightgbm==$_wheel"
        fi
    else
        echo "lightgbm_opencl_works=NO (no OpenCL gfx device listed, or the cap was spent on apt)" >> "$OUT/setup.txt"
    fi
    echo "lgbm_retry_seconds=$(($(date +%s) - _t0))" >> "$OUT/setup.txt"
    "$PY" -c "import lightgbm; print('lightgbm after retry setup', lightgbm.__version__, lightgbm.__file__)" >> "$OUT/versions.txt" 2>&1
    : > "$OUT/track_lgbm.done"
}

fetch_taxi() {
    _d="$DATA/taxi"; mkdir -p "$_d"
    for _m in 2024-01 2024-02; do
        curl -fL --retry 3 --max-time 600 -A "$UA" -o "$_d/yellow_tripdata_$_m.parquet.part" \
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$_m.parquet" \
            && mv "$_d/yellow_tripdata_$_m.parquet.part" "$_d/yellow_tripdata_$_m.parquet"
    done > "$LOGS/fetch_taxi.log" 2>&1
    rm -f "$_d"/*.part
    sha256sum "$_d"/*.parquet > "$OUT/taxi_parquet_sha256.txt" 2>&1
    if [ "$(sha_of "$_d/yellow_tripdata_2024-01.parquet")" = "$TAXI_SHA_01" ] \
       && [ "$(sha_of "$_d/yellow_tripdata_2024-02.parquet")" = "$TAXI_SHA_02" ]; then
        echo "taxi_fetch=cdn, sha256 of both months matches the Mac $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
        return 0
    fi
    echo "taxi_fetch=FAILED or bytes differ from the Mac $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    return 3
}

fetch_istella() {
    _d="$DATA/istella"; mkdir -p "$_d"; _t="$_d/istella-s-letor.tar.gz"
    _i=0
    while [ "$_i" -lt 4 ]; do
        _i=$((_i + 1))
        [ "$(size_of "$_t")" = "$ISTELLA_BYTES" ] && break
        # A resume the server refuses (the MI325X got HTTP 504) restarts clean.
        [ "$_i" -ge 3 ] && rm -f "$_t"
        echo "istella attempt $_i $(date -u +%T) have=$(size_of "$_t")"
        curl -fL -C - --retry 3 --retry-delay 5 --max-time 1500 -A "$UA" -o "$_t" \
            http://library.istella.it/dataset/istella-s-letor.tar.gz
        echo "curl_exit=$? size=$(size_of "$_t") $(date -u +%T)"
    done > "$LOGS/fetch_istella.log" 2>&1
    if [ "$(sha_of "$_t")" = "$ISTELLA_SHA" ]; then
        echo "istella_fetch=ok, sha256 matches the Mac $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
        return 0
    fi
    echo "istella_fetch=FAILED size=$(size_of "$_t") $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    return 3
}

track_data() {
    # Downloads need no Python and start at once; each decode waits for pip.
    for _d in $LEGS; do
        if "fetch_$_d"; then
            wait_for "$OUT/track_pip_base.done"
            step "decode_$_d" 1500 "$PY" tools/speed_gbdt_arm.py --download "$_d"
            if [ -s "$DATA/$_d/${_d}_speed.npz" ]; then : > "$OUT/data.ok.$_d"; fi
        fi
        ls -la "$DATA/$_d" >> "$OUT/datasets_listing.txt" 2>&1
    done
    : > "$OUT/track_data.done"
}

# ---------------------------------------------------------------- setup
box_facts
track_data &
track_mojo &
track_py &
track_lgbm &
wait
LIBPY="$(tail -1 "$OUT/libpython.txt" 2>/dev/null)"
if [ -n "$LIBPY" ] && [ -f "$LIBPY" ]; then
    export MOJO_PYTHON_LIBRARY="$LIBPY"
    echo "MOJO_PYTHON_LIBRARY=$LIBPY" >> "$OUT/setup.txt"
else
    echo "MOJO_PYTHON_LIBRARY unset (no libpython at '$LIBPY')" >> "$OUT/setup.txt"
fi
( cd python && MOJOLEARN_NUMERIC_MODE=identical "$PY" -c "import mojolearn, numpy; print('import OK', mojolearn.__file__, mojolearn.vendor())" ) \
    > "$LOGS/import_identical.log" 2>&1
echo "import_identical=$? $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
if [ "$CELLS" = all ]; then
    ( cd python && "$PY" -c "
import mojolearn
for cls in (mojolearn.GradientBoosting, mojolearn.RandomForestClassifier, mojolearn.ExtraTreesClassifier):
    m = cls(numeric_mode='fast')
    print(cls.__name__, 'fast ->', m.numeric_mode_used(), m.vendor_used())
" ) > "$LOGS/import_fast.log" 2>&1
    echo "import_fast=$? $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
fi
echo "setup_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/setup.txt"
: > "$OUT/setup.done"
cat "$OUT/setup.txt"
grep -q '^import_identical=0' "$OUT/setup.txt" || { echo "import_identical failed"; tail -20 "$LOGS/import_identical.log"; exit 3; }
XGB=xgboost-cpu; LGB=lightgbm-cpu
grep -q '^xgb_rocm_works=yes' "$OUT/setup.txt" && XGB=xgboost-gpu
if [ "$CELLS" = xgb ] && [ "$XGB" != xgboost-gpu ]; then
    echo "cells=xgb REFUSED: amd_xgboost is not on the GPU here ($(grep '^xgb_rocm_works' "$OUT/setup.txt" | tail -1)); nothing timed"
    exit 6
fi

# ---------------------------------------------------------------- cells
LGB_PARAMS="min_child_weight=1e-3"
if grep -q '^lightgbm_opencl_works=yes' "$OUT/setup.txt"; then
    LGB=lightgbm-opencl; LGB_PARAMS="min_child_weight=1e-3,gpu_use_dp=True"
fi
export MOJOLEARN_SPEED_PY="$PY" MOJOLEARN_SPEED_DEVICES=cpu,gpu,opencl
echo "arms: cells=$CELLS xgboost=$XGB lightgbm=$LGB lightgbm_params=$LGB_PARAMS python=$PY" | tee -a "$OUT/ab.txt"
"$PY" -c "import catboost, sklearn, xgboost, lightgbm, numpy, sys; print('python', sys.version.split()[0]); print('catboost', catboost.__version__); print('sklearn', sklearn.__version__); print('xgboost', xgboost.__version__, xgboost.__file__); print('lightgbm', lightgbm.__version__, lightgbm.__file__); print('numpy', numpy.__version__)" > "$OUT/versions_used.txt" 2>&1
full() {  # <dataset> <lane> <rows> <arms>
    echo "cell $2 $1 $3 $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/ab.txt"
    MOJOLEARN_SPEED_OPPONENTS_FIRST=1 MOJOLEARN_SPEED_ARMS="$4" $AB speed baseline "$2" "$1" "$3" 5 full
}
for DS in $LEGS; do
    if [ ! -f "$OUT/data.ok.$DS" ]; then
        echo "no $DS data; its cells are skipped (the loader would fall back to a synthetic fixture)" | tee -a "$OUT/ab.txt"
        continue
    fi
    if [ "$CELLS" = xgb ]; then
        # XGBoost on the GPU against ours, rocm-smi sampled beside each cell.
        MOJOLEARN_SPEED_SMI_SAMPLE=1 MOJOLEARN_SPEED_TAG=xgbgpu full "$DS" gbdt-depthwise 1000000 "$XGB"
        MOJOLEARN_SPEED_SMI_SAMPLE=1 MOJOLEARN_SPEED_TAG=xgbgpu full "$DS" gbdt-lossguide 1000000 "$XGB"
        mark "PHASE_XGB_${DS}_DONE"
        continue
    fi
    full "$DS" rf 1000000 sklearn-rf-cpu
    full "$DS" et 1000000 sklearn-et-cpu
    mark "PHASE_F1_${DS}_DONE"
    full "$DS" rf 2000000 sklearn-rf-cpu
    mark "PHASE_F2_${DS}_DONE"
    full "$DS" gbdt-symmetric 1000000 catboost-cpu
    full "$DS" gbdt-depthwise 1000000 "catboost-cpu,$XGB"
    MOJOLEARN_SPEED_LGBM_PARAMS="$LGB_PARAMS" full "$DS" gbdt-lossguide 1000000 "catboost-cpu,$XGB,$LGB"
    mark "PHASE_G_${DS}_DONE"
    # FAST beside IDENTICAL, ours only, interleaved in one process (ET, the
    # slowest, last).
    for L in rf gbdt-symmetric gbdt-depthwise gbdt-lossguide et; do
        echo "cell fast.$L $DS $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/ab.txt"
        MOJOLEARN_SPEED_OURS_AB="numeric_mode='fast'" MOJOLEARN_SPEED_TAG=fastab \
            $AB speed baseline "$L" "$DS" 1000000 5 ours
    done
    mark "PHASE_FAST_${DS}_DONE"
done
echo "body end $(date -u +%T)" | tee -a "$OUT/ab.txt"
