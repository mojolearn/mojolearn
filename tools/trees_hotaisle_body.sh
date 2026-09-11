#!/bin/sh
# RUNS ON THE HOT AISLE MI300X VM, as the leg body of tools/hotaisle_leg.sh
# (lane trees-hotaisle, 2026-09-11). Mirrors tools/trees_amd_remote.sh and the
# MI325X batches (bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/
# logs/) for ONE dataset per VM, because a VM lives 60 minutes and has no
# shared volume:
#
#   MOJOLEARN_TREES_HA_LEG=taxi|istella   which dataset this VM measures
#   MOJOLEARN_TREES_HA_OUT=<dir>          where the runner fetches from
#                                         (default /root/gemm_leg_out)
#
# Setup (untimed, before any cell): box facts; the IDENTICAL bindings base,
# gbdt, rf, trees and the FAST bindings gbdt, rf, trees; a Python 3.12 venv
# with CatBoost, scikit-learn, LightGBM and AMD's ROCm XGBoost build
# `amd_xgboost` (GPU probe beside rocm-smi); the ONE LightGBM retry
# (min_child_weight 1e-3, an OpenCL build tried inside a 15 minute cap); the
# dataset, sha256 checked against the Mac's copy. Then the cells, 1 warm-up
# plus 5 rounds, arms interleaved round by round, opponents imported first:
# RF 1M, ET 1M, RF 2M (scikit-learn CPU), symmetric (CatBoost CPU), depthwise
# (CatBoost CPU, XGBoost GPU), lossguide (CatBoost CPU, XGBoost GPU,
# LightGBM), then FAST beside IDENTICAL for the five lanes, ours only.
# Our GBDT IDENTICAL arms are not deterministic on AMD today; the per-round
# hashes in every FSPEED line are the record. POSIX sh.
set -u
LEG="${MOJOLEARN_TREES_HA_LEG:-}"
case "$LEG" in
    taxi|istella) ;;
    *) echo "MOJOLEARN_TREES_HA_LEG must be taxi or istella (got '$LEG')"; exit 2 ;;
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
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) leg=$LEG commit=$(cat SHIPPED_COMMIT.txt 2>/dev/null) gpu_archs_env=${MOJOLEARN_GPU_ARCHS:-}" > "$OUT/setup.txt"

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

box_facts() {
    { rocm-smi --showproductname --showdriverversion; rocm-smi --showmeminfo vram; } > "$OUT/gpu.txt" 2>&1
    rocminfo > "$LOGS/rocminfo.log" 2>&1
    { cat /opt/rocm/.info/version; ls -d /opt/rocm-*; } > "$OUT/rocm_version.txt" 2>&1
    { echo "sysfs $(cat /sys/module/amdgpu/version 2>/dev/null)"; modinfo amdgpu 2>/dev/null | grep -E '^(version|filename|vermagic):'; } > "$OUT/amdgpu.txt" 2>&1
    lscpu > "$OUT/lscpu.txt" 2>&1
    { echo "nproc $(nproc)"; grep -m1 'model name' /proc/cpuinfo; echo "cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; } > "$OUT/cpu.txt"
    free -g > "$OUT/free.txt" 2>&1
    uname -a > "$OUT/uname.txt"
    cat /etc/os-release > "$OUT/os.txt" 2>&1
    df -h / > "$OUT/df.txt" 2>&1
}

pick_arch() {
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        MOJOLEARN_GPU_ARCHS="$(grep -o -m1 'gfx[0-9a-f]*' "$LOGS/rocminfo.log" 2>/dev/null | head -1)"
        [ -n "$MOJOLEARN_GPU_ARCHS" ] || MOJOLEARN_GPU_ARCHS="$(sed -n 's/.*GFX Version:[[:space:]]*//p' "$OUT/gpu.txt" | head -1 | tr -d ' ')"
    fi
    echo "gpu_archs=${MOJOLEARN_GPU_ARCHS:-NONE}" >> "$OUT/setup.txt"
    [ -n "${MOJOLEARN_GPU_ARCHS:-}" ] && export MOJOLEARN_GPU_ARCHS
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
    # FAST (base ships IDENTICAL only, DEVIATION 2490).
    for _b in gbdt rf trees; do
        step "build_fast_$_b" 1500 env MOJOLEARN_NUMERIC_MODE=fast bash "bindings/build_$_b.sh"
    done
    sha256sum python/mojolearn/*.so > "$OUT/fast_so_sha256.txt" 2>&1
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
    # uv only supplies CPython 3.12 (the MI325X leg's 3.12.3) and a seeded
    # venv without apt; every package then comes from pip, as on the MI325X.
    if [ ! -x /root/.uvbin/uv ]; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.uvbin UV_NO_MODIFY_PATH=1 sh > "$LOGS/uv_get.log" 2>&1
    fi
    step venv 600 /root/.uvbin/uv venv --seed --python 3.12 "$VENV"
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
    if ! grep -q '^xgb_rocm_works=yes' "$OUT/setup.txt"; then
        # No AMD GPU path installed: the PyPI build on the CPU, labeled CPU.
        step pip_xgboost_cpu 600 "$PY" -m pip install --no-input --disable-pip-version-check --force-reinstall xgboost
    fi
    "$PY" -c "import sys, numpy, sklearn, catboost, lightgbm, xgboost, joblib; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__); print('catboost', catboost.__version__); print('lightgbm', lightgbm.__version__, lightgbm.__file__); print('xgboost', xgboost.__version__, xgboost.__file__); print('joblib.cpu_count', joblib.cpu_count())" > "$OUT/versions.txt" 2>&1
    : > "$OUT/track_py.done"
}

track_lgbm() {
    # THE ONE LightGBM RETRY, 15 minutes from here, no second attempt.
    wait_for "$OUT/track_py.done"
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

data_taxi() {
    _d="$DATA/taxi"; mkdir -p "$_d"
    for _m in 2024-01 2024-02; do
        curl -fL --retry 3 --max-time 600 -A "$UA" -o "$_d/yellow_tripdata_$_m.parquet.part" \
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$_m.parquet" \
            && mv "$_d/yellow_tripdata_$_m.parquet.part" "$_d/yellow_tripdata_$_m.parquet"
    done > "$LOGS/fetch_taxi.log" 2>&1
    _ok() { [ "$(sha_of "$_d/yellow_tripdata_2024-01.parquet")" = "$TAXI_SHA_01" ] \
            && [ "$(sha_of "$_d/yellow_tripdata_2024-02.parquet")" = "$TAXI_SHA_02" ]; }
    if _ok; then
        echo "taxi_fetch=cdn sha256 matches the Mac $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    else
        # The TLC CDN refused or the bytes differ: wait for the Mac's upload.
        echo "taxi_fetch=FAILED, waiting for an upload into $_d $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
        rm -f "$_d"/*.part
        : > "$OUT/taxi_upload_wanted"
        _w=0
        while ! _ok && [ "$_w" -lt 1500 ]; do sleep 10; _w=$((_w + 10)); done
        _ok || { echo "taxi_data=MISSING $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"; return 3; }
        echo "taxi_fetch=uploaded sha256 matches the Mac $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    fi
    sha256sum "$_d"/*.parquet > "$OUT/taxi_parquet_sha256.txt"
    step decode_taxi 1200 "$PY" tools/speed_gbdt_arm.py --download taxi
    [ -s "$_d/taxi_speed.npz" ]
}

data_istella() {
    _d="$DATA/istella"; mkdir -p "$_d"; _t="$_d/istella-s-letor.tar.gz"
    _i=0
    while [ "$_i" -lt 4 ]; do
        _i=$((_i + 1))
        [ "$(stat -c %s "$_t" 2>/dev/null || echo 0)" = "$ISTELLA_BYTES" ] && break
        # A resume the server refuses (the MI325X got HTTP 504) restarts clean.
        if [ "$_i" -ge 3 ]; then rm -f "$_t"; fi
        echo "istella attempt $_i $(date -u +%T) have=$(stat -c %s "$_t" 2>/dev/null || echo 0)"
        curl -fL -C - --retry 3 --retry-delay 5 --max-time 1500 -A "$UA" -o "$_t" \
            http://library.istella.it/dataset/istella-s-letor.tar.gz
        echo "curl_exit=$? size=$(stat -c %s "$_t" 2>/dev/null || echo 0) $(date -u +%T)"
    done > "$LOGS/fetch_istella.log" 2>&1
    if [ "$(sha_of "$_t")" != "$ISTELLA_SHA" ]; then
        echo "istella_fetch=FAILED size=$(stat -c %s "$_t" 2>/dev/null || echo 0) $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
        return 3
    fi
    echo "istella_fetch=ok sha256 matches the Mac $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
    step decode_istella 1500 "$PY" tools/speed_gbdt_arm.py --download istella
    [ -s "$_d/istella_speed.npz" ]
}

track_data() {
    wait_for "$OUT/track_pip_base.done"
    if "data_$LEG"; then : > "$OUT/data.ok"; fi
    : > "$OUT/track_data.done"
}

# ---------------------------------------------------------------- setup
box_facts
pick_arch
track_mojo &
track_py &
track_lgbm &
track_data &
wait
( cd python && MOJOLEARN_NUMERIC_MODE=identical "$PY" -c "import mojolearn, numpy; print('import OK', mojolearn.__file__)" ) \
    > "$LOGS/import_identical.log" 2>&1
echo "import_identical=$? $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
( cd python && "$PY" -c "
import mojolearn
for cls in (mojolearn.GradientBoosting, mojolearn.RandomForestClassifier, mojolearn.ExtraTreesClassifier):
    m = cls(numeric_mode='fast')
    print(cls.__name__, 'fast ->', m.numeric_mode_used(), m.vendor_used())
" ) > "$LOGS/import_fast.log" 2>&1
echo "import_fast=$? $(date -u +%H:%M:%S)" >> "$OUT/setup.txt"
echo "setup_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/setup.txt"
: > "$OUT/setup.done"
cat "$OUT/setup.txt"
grep -q '^import_identical=0' "$OUT/setup.txt" || { echo "import_identical failed"; exit 3; }
[ -f "$OUT/data.ok" ] || { echo "no $LEG data; refusing (the loader would fall back to a synthetic fixture)"; exit 4; }

# ---------------------------------------------------------------- cells
XGB=xgboost-cpu; LGB=lightgbm-cpu
grep -q '^xgb_rocm_works=yes' "$OUT/setup.txt" && XGB=xgboost-gpu
LGB_PARAMS="min_child_weight=1e-3"
if grep -q '^lightgbm_opencl_works=yes' "$OUT/setup.txt"; then
    LGB=lightgbm-opencl; LGB_PARAMS="min_child_weight=1e-3,gpu_use_dp=True"
fi
export MOJOLEARN_SPEED_PY="$PY" MOJOLEARN_SPEED_DEVICES=cpu,gpu,opencl MOJOLEARN_SPEED_OPPONENTS_FIRST=1
echo "arms: xgboost=$XGB lightgbm=$LGB lightgbm_params=$LGB_PARAMS python=$PY" | tee -a "$OUT/ab.txt"
"$PY" -c "import catboost, sklearn, xgboost, lightgbm, numpy, sys; print('python', sys.version.split()[0]); print('catboost', catboost.__version__); print('sklearn', sklearn.__version__); print('xgboost', xgboost.__version__, xgboost.__file__); print('lightgbm', lightgbm.__version__, lightgbm.__file__); print('numpy', numpy.__version__)" > "$OUT/versions_used.txt" 2>&1
full() {  # <lane> <rows> <arms>
    echo "cell $1 $LEG $2 $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/ab.txt"
    MOJOLEARN_SPEED_ARMS="$3" $AB speed baseline "$1" "$LEG" "$2" 5 full
}
full rf 1000000 sklearn-rf-cpu
full et 1000000 sklearn-et-cpu
mark PHASE_F1_DONE
full rf 2000000 sklearn-rf-cpu
mark PHASE_F2_DONE
full gbdt-symmetric 1000000 catboost-cpu
full gbdt-depthwise 1000000 "catboost-cpu,$XGB"
MOJOLEARN_SPEED_LGBM_PARAMS="$LGB_PARAMS" full gbdt-lossguide 1000000 "catboost-cpu,$XGB,$LGB"
mark PHASE_G_DONE

# FAST beside IDENTICAL, ours only, interleaved in one process.
unset MOJOLEARN_SPEED_OPPONENTS_FIRST
for L in rf et gbdt-symmetric gbdt-depthwise gbdt-lossguide; do
    echo "cell fast.$L $LEG $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/ab.txt"
    MOJOLEARN_SPEED_OURS_AB="numeric_mode='fast'" MOJOLEARN_SPEED_TAG=fastab \
        $AB speed baseline "$L" "$LEG" 1000000 5 ours
done
mark PHASE_FAST_DONE
echo "body end $(date -u +%T)" | tee -a "$OUT/ab.txt"
