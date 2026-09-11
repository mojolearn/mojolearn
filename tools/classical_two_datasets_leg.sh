#!/bin/sh
# tools/classical_two_datasets_leg.sh -- DEVIATION 2570, the classical
# opponent rows on the two benchmark datasets (ENGINEERING_RULES.md sections
# 9 and 10). Runs ON THE BOX from /root/mojolearn.
#
# NVIDIA (RunPod, DEVIATION 2571): driven phase by phase over tools/trees_leg.sh
# ssh, each phase under nohup, so a harness bug costs one phase:
#
#   TREES_LEG_NAME=mojolearn-ctd1 TREES_LEG_STATE=$HOME/mojolearn-evidence/classical-h100/pod1 \
#     sh tools/trees_leg.sh rent --gpu "NVIDIA H100 80GB HBM3" --minutes 70
#   ... ssh 'cd /root/mojolearn && nohup env MOJOLEARN_CTD_PHASES=setup \
#            MOJOLEARN_CTD_LANES=kmeans,pca,ols,knn sh tools/classical_two_datasets_leg.sh \
#            > /root/ctd_out/setup.console 2>&1 &'
#   then PHASES="prep races" with MOJOLEARN_CTD_SMOKE_ROWS=20000 first, then the
#   full shape, one MOJOLEARN_CTD_DATASETS at a time
#   (bench/results/classical_h100_2026-09-11/batch*.sh are the runs, in order).
#
# AMD (DigitalOcean): tools/do_extra_leg.sh's MOJOLEARN_GEMM_LEG_EXTRA body, all
# phases in one call (the default), from /root/mojolearn with pixi on PATH:
#
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/classical_two_datasets_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/classical-two-datasets-$(date -u +%Y-%m-%d_%H%M%S)-amd-leg1 \
#   bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
#
# AMD (Hot Aisle MI300X, DEVIATION 2572): tools/classical_hotaisle_leg.sh is the
# tools/hotaisle_leg.sh body; it calls this file once per phase group (setup,
# smoke, prep, leg 1, leg 2) under one deadline. Evidence:
# bench/results/classical_hotaisle_2026-09-11/.
#
# The DigitalOcean runner passes no environment to a body, so the lane list is this file's
# default (kmeans,pca,ols,knn) and the second lease runs
# tools/classical_two_datasets_leg2.sh (kde,svc), which sets
# MOJOLEARN_CTD_LANES and execs this file.
#
# PHASES (MOJOLEARN_CTD_PHASES, default "setup prep races"), each step with a
# row in status.tsv (name, exit, seconds); a red step does not stop the next,
# because a red step is a finding:
#   setup   env record; the opponents' Python (NVIDIA: the image's python3 with
#           torch 2.4.1+cu124, plus scikit-learn, pyarrow and cuml-cu12==26.8.0
#           from pypi.nvidia.com, the recipe of tools/trees_identical_remote.sh
#           and the version of the existing H100 rows; AMD: a venv with the
#           torch ROCm pin of tools/do_byte_lm_setup.sh and scikit-learn
#           1.5.2); the downloads (Istella in the BACKGROUND, it writes
#           istella.done; taxi after pyarrow); the IDENTICAL bindings
#           (bindings/build.sh always, build_estimators.sh for pca/ols/kde,
#           build_svm.sh for svc) with MOJOLEARN_GPU_ARCHS when set; the
#           ours import read back as identical
#   prep    tools/classical_two_datasets.py prep, one dataset at a time
#   races   one conductor per (lane, dataset), 1 warm-up + ROUNDS interleaved
#           rounds, quality; SKIPPED_DEADLINE when the body deadline is
#           closer than MOJOLEARN_CTD_MIN_RACE; summary.tsv after every race
#
# SMOKE: MOJOLEARN_CTD_SMOKE_ROWS=N preps capped blocks (prep --max-rows N)
# into $DATA-smoke, writes under $OUT/smoke and races 1 round by default.
#
# THREADS: OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and
# NUMEXPR_NUM_THREADS are UNSET for the whole body. Each race's JSON names the
# pools threadpoolctl reports for a scikit-learn arm.
#
# POSIX sh only: the box's /bin/sh is dash.
set -u
ROOT=${MOJOLEARN_CTD_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_CTD_OUT:-/root/gemm_leg_out/classical-two-datasets}
DATA=${MOJOLEARN_CTD_DATA:-/root/ctd-data}
WORK=${MOJOLEARN_CTD_WORK:-/root/ctd-work}
VENV=${MOJOLEARN_CTD_VENV:-/root/ctd-venv}
LANES=${MOJOLEARN_CTD_LANES:-kmeans,pca,ols,knn}
# Commas or spaces (DEVIATION 2572: a runner env value may not hold a space).
DATASETS=$(echo "${MOJOLEARN_CTD_DATASETS:-taxi istella}" | tr ',' ' ')
PHASES=$(echo "${MOJOLEARN_CTD_PHASES:-setup prep races}" | tr ',' ' ')
SMOKE=${MOJOLEARN_CTD_SMOKE_ROWS:-}
ROUNDS=${MOJOLEARN_CTD_ROUNDS:-5}
if [ -n "$SMOKE" ]; then
    DATA="$DATA-smoke"
    OUT="$OUT/smoke"
    ROUNDS=${MOJOLEARN_CTD_ROUNDS:-1}
fi
BODY_SECONDS=${MOJOLEARN_CTD_BODY_SECONDS:-2100}
MIN_RACE=${MOJOLEARN_CTD_MIN_RACE:-240}
RACE_SECONDS=${MOJOLEARN_CTD_RACE_SECONDS:-900}
JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
EXTRA_ARMS=${MOJOLEARN_CTD_EXTRA_ARMS:-}
# One deadline across several calls of this file in one body (DEVIATION 2572).
BODY_START=${MOJOLEARN_CTD_BODY_START:-$(date +%s)}
# The Mac's copies of the two TLC months (2026-09-11), checked after any fetch.
TAXI_SHA_2024_01=c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510
TAXI_SHA_2024_02=c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20
ISTELLA_TGZ_SHA=41b21116a3650cc043dbe16f02ee39f4467f9405b37fdbcc9a6a05e230a38981
TAXI_WAIT=${MOJOLEARN_CTD_TAXI_WAIT:-0}
UA="Mozilla/5.0 (X11; Linux x86_64) mojolearn-bench"

mkdir -p "$OUT" "$DATA" "$WORK"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
export MOJOLEARN_NUMERIC_MODE=identical
GBM_BENCH_DATA=${GBM_BENCH_DATA:-/root/datasets/gbm-bench}
export GBM_BENCH_DATA
COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
[ -n "$COMMIT" ] || COMMIT=$(cat "$ROOT/SHIPPED_COMMIT.txt" 2>/dev/null)
MOJOLEARN_REPO_COMMIT=${COMMIT:-unknown}
export MOJOLEARN_REPO_COMMIT
VENDOR=$(sed -n 's/^vendor=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
PROVIDER=$(sed -n 's/^provider=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
if [ -z "$VENDOR" ]; then
    if [ -e /dev/kfd ] && { command -v rocm-smi || command -v rocminfo || command -v amd-smi; } > /dev/null 2>&1; then VENDOR=amd
    elif command -v nvidia-smi > /dev/null 2>&1; then VENDOR=nvidia
    else VENDOR=unknown; fi
fi
if [ "$VENDOR" = nvidia ]; then
    PY=${MOJOLEARN_CTD_PYTHON:-python3}
    OURS_PY=$PY
    THEIRS_PY=$PY
else
    PY="$VENV/bin/python"
    OURS_PY="pixi run python3"
    THEIRS_PY="$VENV/bin/python"
fi
touch "$OUT/status.tsv"
{
    echo "lane=classical-two-datasets (DEVIATION 2570, 2571)"
    echo "invoked=$(date -u +%Y-%m-%dT%H:%M:%SZ) phases=$PHASES smoke_rows=${SMOKE:-none}"
    echo "vendor=$VENDOR provider=${PROVIDER:-unknown}"
    echo "commit=$MOJOLEARN_REPO_COMMIT"
    echo "lanes=$LANES"
    echo "datasets=$DATASETS"
    echo "rounds=$ROUNDS body_seconds=$BODY_SECONDS min_race=$MIN_RACE race_seconds=$RACE_SECONDS jobs=$JOBS"
    echo "gpu_archs=${MOJOLEARN_GPU_ARCHS:-}"
    echo "ours_python=$OURS_PY theirs_python=$THEIRS_PY extra_arms=$EXTRA_ARMS"
} >> "$OUT/gate.txt"

record() {
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -u +%H:%M:%S)" >> "$OUT/status.tsv"
}
run() {
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _rc=$?
    record "$_name" "$_rc" "$(( $(date +%s) - _t0 ))"
    return $_rc
}
wants() {  # <lane>...: true when any is in LANES
    for _w in "$@"; do
        case ",$LANES," in *,"$_w",*) return 0 ;; esac
    done
    return 1
}
has_phase() {
    case " $PHASES " in *" $1 "*) return 0 ;; esac
    return 1
}
arms_for() {  # <lane>: the arms a race runs on this vendor
    if [ "$VENDOR" = nvidia ]; then
        case "$1" in
            ols) _a="ours,cuml-gpu,torch-gpu,torch-gpu-eigh" ;;
            kmeans|pca|knn) _a="ours,cuml-gpu,torch-gpu" ;;
            *) _a="ours,cuml-gpu" ;;
        esac
    else
        case "$1" in
            ols) _a="ours,sklearn-cpu,torch-gpu,torch-gpu-eigh" ;;
            kmeans|pca|knn) _a="ours,sklearn-cpu,torch-gpu" ;;
            *) _a="ours,sklearn-cpu" ;;
        esac
    fi
    [ -n "$EXTRA_ARMS" ] && _a="$_a,$EXTRA_ARMS"
    echo "$_a"
}

# ============================================================================
# setup
# ============================================================================
if has_phase setup; then
    {
        echo "== date"; date -u
        echo "== uname"; uname -a
        echo "== nproc"; nproc
        echo "== lscpu"; lscpu 2>&1
        echo "== free -g"; free -g 2>&1
        if command -v rocm-smi > /dev/null 2>&1; then
            echo "== rocm-smi --showproductname --showdriverversion"
            rocm-smi --showproductname --showdriverversion 2>&1
        fi
        if command -v amd-smi > /dev/null 2>&1; then
            echo "== amd-smi version"; amd-smi version 2>&1
            echo "== amd-smi static --asic"; amd-smi static --asic 2>&1 | head -40
        fi
        if command -v rocminfo > /dev/null 2>&1; then
            echo "== rocminfo (names)"; rocminfo 2>&1 | grep -E 'Marketing Name|Name: +gfx|Compute Unit' | head -20
        fi
        if [ -f /opt/rocm/.info/version ]; then
            echo "== /opt/rocm/.info/version"; cat /opt/rocm/.info/version
        fi
        if [ -f /sys/module/amdgpu/version ]; then
            echo "== /sys/module/amdgpu/version"; cat /sys/module/amdgpu/version
        fi
        if command -v nvidia-smi > /dev/null 2>&1; then
            echo "== nvidia-smi"; nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv 2>&1
            nvidia-smi 2>&1 | head -12
        fi
        if command -v nvcc > /dev/null 2>&1; then
            echo "== nvcc"; nvcc --version 2>&1 | tail -2
        fi
    } > "$OUT/env.txt" 2>&1
    record env 0 0

    # ---- background, at once: the raw fetches (curl, no numpy) ----------------
    # DEVIATION 2572: the 472 MB Istella tarball and the two taxi months start at
    # t=0 beside pixi and the builds; only the decode waits for the venv. Each
    # file is checked against the Mac's sha256; a mismatch is deleted, so the
    # decode step below either fetches again or reports the miss.
    (
        mkdir -p "$GBM_BENCH_DATA/istella" "$GBM_BENCH_DATA/taxi"
        _t0=$(date +%s)
        _tgz="$GBM_BENCH_DATA/istella/istella-s-letor.tar.gz"
        # Resumed attempts, a clean restart from the third (the MI325X got HTTP
        # 504 on a resume; tools/trees_hotaisle_body.sh).
        _i=0
        while [ "$_i" -lt 4 ] && [ "$(sha256sum "$_tgz" 2>/dev/null | cut -c1-64)" != "$ISTELLA_TGZ_SHA" ]; do
            _i=$((_i + 1))
            [ "$_i" -ge 3 ] && rm -f "$_tgz"
            timeout -k 10 1500 curl -fsSL -C - --retry 3 --retry-delay 5 -A "$UA" -o "$_tgz" \
                http://library.istella.it/dataset/istella-s-letor.tar.gz >> "$OUT/istella_curl.log" 2>&1
            echo "attempt=$_i curl_exit=$? size=$(stat -c %s "$_tgz" 2>/dev/null || echo 0) $(date -u +%T)" >> "$OUT/istella_curl.log"
        done
        _got=$(sha256sum "$_tgz" 2>/dev/null | cut -c1-64)
        [ "$_got" = "$ISTELLA_TGZ_SHA" ] || rm -f "$_tgz"
        echo "istella_tgz sha256=${_got:-none} expected=$ISTELLA_TGZ_SHA seconds=$(( $(date +%s) - _t0 ))" >> "$OUT/fetch.txt"
        if [ -f "$_tgz" ]; then
            _t0=$(date +%s)
            tar -xzf "$_tgz" -C "$GBM_BENCH_DATA/istella" > "$OUT/istella_untar.log" 2>&1
            echo "istella_untar rc=$? seconds=$(( $(date +%s) - _t0 ))" >> "$OUT/fetch.txt"
        fi
        : > "$OUT/fetch_istella.done"
    ) &
    (
        _t0=$(date +%s)
        for m in 2024-01 2024-02; do
            _f="$GBM_BENCH_DATA/taxi/yellow_tripdata_$m.parquet"
            [ -f "$_f" ] || { timeout -k 10 600 curl -fsSL --retry 3 -A "$UA" -o "$_f.part" -w '%{http_code}' \
                "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$m.parquet" \
                > "$OUT/taxi_curl_$m.txt" 2>&1 && mv "$_f.part" "$_f"; }
            rm -f "$_f.part"
        done
        [ "$(sha256sum "$GBM_BENCH_DATA/taxi/yellow_tripdata_2024-01.parquet" 2>/dev/null | cut -c1-64)" = "$TAXI_SHA_2024_01" ] \
            && [ "$(sha256sum "$GBM_BENCH_DATA/taxi/yellow_tripdata_2024-02.parquet" 2>/dev/null | cut -c1-64)" = "$TAXI_SHA_2024_02" ] \
            || : > "$OUT/taxi_upload_wanted"
        # A CDN that blocks the box: MOJOLEARN_CTD_TAXI_WAIT seconds for the
        # Mac to upload the two months beside the cache (sha256 checked).
        while :; do
            _ok=1
            for m in 2024-01 2024-02; do
                _f="$GBM_BENCH_DATA/taxi/yellow_tripdata_$m.parquet"
                case $m in 2024-01) _want=$TAXI_SHA_2024_01 ;; *) _want=$TAXI_SHA_2024_02 ;; esac
                [ "$(sha256sum "$_f" 2>/dev/null | cut -c1-64)" = "$_want" ] || _ok=0
            done
            [ "$_ok" = 1 ] && break
            [ $(( $(date +%s) - _t0 )) -ge "$TAXI_WAIT" ] && break
            sleep 15
        done
        for m in 2024-01 2024-02; do
            _f="$GBM_BENCH_DATA/taxi/yellow_tripdata_$m.parquet"
            echo "taxi_$m curl_http=$(cat "$OUT/taxi_curl_$m.txt" 2>/dev/null) sha256=$(sha256sum "$_f" 2>/dev/null | cut -c1-64) sha_ok=$_ok seconds=$(( $(date +%s) - _t0 ))" >> "$OUT/fetch.txt"
        done
        : > "$OUT/fetch_taxi.done"
    ) &

    # ---- background: the opponents' Python, then the decodes -----------------
    # The decodes start only after pip has finished, so no process has numpy
    # loaded while pip replaces it underneath.
    (
        if [ "$VENDOR" = nvidia ]; then
            run pip-base timeout -k 30 900 "$PY" -m pip install --no-input --disable-pip-version-check \
                scikit-learn pyarrow threadpoolctl
            run pip-cuml timeout -k 30 1500 "$PY" -m pip install --no-input --disable-pip-version-check \
                --extra-index-url=https://pypi.nvidia.com cuml-cu12==26.8.0
        else
            BASE=/usr/bin/python3
            PIN="https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl#sha256=1d97c15798bf178299328032141a21d9777e7cdef59d5a7e3ac74e297c17198e https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl#sha256=6b141e1a03148b007c6217519cd9947d760123ded5caebadffec22cba7358d2d"
            # DEVIATION 2572: the pinned wheels are cp312. A box whose python3
            # is not 3.12 (or has no venv module) gets a 3.12 venv from uv.
            if "$BASE" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info[:2] == (3, 12) else 3)' > /dev/null 2>&1; then
                run venv "$BASE" -m venv "$VENV"
            else
                run uv-get sh -c 'curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/root/.local/uv sh'
                run venv /root/.local/uv/uv venv --seed --python 3.12 "$VENV"
            fi
            # The opponent as a user installs it today: scikit-learn, NumPy and
            # SciPy current from PyPI (the H100 recipe; the MI325X trees rows ran
            # 1.9.1), versions in versions.txt and pip_freeze.txt.
            # shellcheck disable=SC2086
            run wheels timeout -k 30 1200 "$VENV/bin/pip" install --disable-pip-version-check --no-input \
                --only-binary=:all: $PIN numpy scipy scikit-learn threadpoolctl joblib pyarrow
            # torch sees the GPU, or the second pin (PyTorch's own ROCm 6.2.4 wheel,
            # ROCm libraries bundled) replaces it; torch_source.txt says which.
            if "$VENV/bin/python" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)' > "$OUT/torch_probe1.log" 2>&1; then
                echo "torch_source=repo.radeon.com rocm-rel-6.4.1 torch 2.6.0 (cuda.is_available True)" > "$OUT/torch_source.txt"
            else
                run torch-fallback timeout -k 30 1200 "$VENV/bin/pip" install --disable-pip-version-check --no-input \
                    --force-reinstall --index-url https://download.pytorch.org/whl/rocm6.2.4 'torch==2.6.0+rocm6.2.4'
                "$VENV/bin/python" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)' > "$OUT/torch_probe2.log" 2>&1
                echo "torch_source=download.pytorch.org rocm6.2.4 torch 2.6.0 (probe2 rc=$?; the repo.radeon.com pin did not see the GPU, torch_probe1.log)" > "$OUT/torch_source.txt"
            fi
        fi
        "$PY" -m pip freeze > "$OUT/pip_freeze.txt" 2>&1
        "$PY" - > "$OUT/versions.txt" 2>&1 <<'PYV'
import numpy
print("numpy", numpy.__version__)
for name in ("torch", "cuml", "cupy", "sklearn", "scipy", "pyarrow"):
    try:
        mod = __import__(name)
        extra = ""
        if name == "torch":
            extra = " cuda %s hip %s available %s device %s" % (
                mod.version.cuda, getattr(mod.version, "hip", None), mod.cuda.is_available(),
                mod.cuda.get_device_name(0) if mod.cuda.is_available() else "-")
        if name == "cupy":
            extra = " cuda_runtime %s driver_api %s" % (
                mod.cuda.runtime.runtimeGetVersion(), mod.cuda.runtime.driverGetVersion())
        print(name, mod.__version__ + extra)
    except Exception as exc:  # noqa: BLE001
        print(name, "NOT IMPORTABLE", repr(exc))
PYV
        # The raw fetches first (a second writer on the same file is a torn file).
        while [ ! -f "$OUT/fetch_istella.done" ] || [ ! -f "$OUT/fetch_taxi.done" ]; do sleep 5; done
        ( run download-istella timeout -k 30 2700 "$PY" tools/speed_gbdt_arm.py --download istella
          echo "rc=$?" > "$OUT/istella.done" ) &
        run download-taxi timeout -k 30 1500 "$PY" tools/speed_gbdt_arm.py --download taxi
        echo "rc=$?" > "$OUT/taxi.done"
        wait
    ) &
    PID_BG=$!

    # ---- foreground: pixi and the IDENTICAL bindings the lanes need ------------
    if ! command -v pixi > /dev/null 2>&1; then
        run pixi-get sh -c 'curl -fsSL https://pixi.sh/install.sh | sh'
    fi
    run pixi-install timeout -k 30 1500 pixi install
    BUILDS="build.sh"
    wants pca ols kde && BUILDS="$BUILDS build_estimators.sh"
    wants svc && BUILDS="$BUILDS build_svm.sh"
    for b in $BUILDS; do
        run "build-${b%.sh}" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
            MOJOLEARN_COMPILE_JOBS="$JOBS" timeout -k 30 1500 sh "bindings/$b"
    done
    ls -la python/mojolearn/identical > "$OUT/identical_bindings.txt" 2>&1
    for so in python/mojolearn/identical/*.so; do
        [ -f "$so" ] || continue
        printf '%s sha256=%s sm_90a=%s gfx942=%s %s=%s\n' "$so" "$(sha256sum "$so" | cut -c1-16)" \
            "$(strings "$so" 2>/dev/null | grep -c sm_90a)" "$(strings "$so" 2>/dev/null | grep -c gfx942)" \
            "${MOJOLEARN_GPU_ARCHS:-box_arch_unset}" \
            "$( [ -n "${MOJOLEARN_GPU_ARCHS:-}" ] && strings "$so" 2>/dev/null | grep -c "${MOJOLEARN_GPU_ARCHS%%,*}")"
    done > "$OUT/identical_archs.txt" 2>&1
    record builds-done 0 "$(( $(date +%s) - BODY_START ))"
    # pip may still be replacing numpy; the import waits for the background
    # python work that precedes the downloads.
    while [ ! -f "$OUT/versions.txt" ] || ! grep -q '^pyarrow\|^scipy' "$OUT/versions.txt" 2>/dev/null; do
        sleep 10
        kill -0 "$PID_BG" 2>/dev/null || break
    done
    run ours-import env PYTHONPATH="$ROOT/python" MOJOLEARN_NUMERIC_MODE=identical timeout -k 10 300 \
        $OURS_PY -c 'import mojolearn; km = mojolearn.KMeans(n_clusters=2); print("mojolearn", mojolearn.__version__, km.numeric_mode_used(), km.vendor_used())'
    while [ ! -f "$OUT/taxi.done" ]; do
        sleep 10
        kill -0 "$PID_BG" 2>/dev/null || break
    done
    record setup-done 0 "$(( $(date +%s) - BODY_START ))"
fi

# ============================================================================
# prep, one dataset at a time (a failed download reds only its own rows)
# ============================================================================
if has_phase prep; then
    for ds in $DATASETS; do
        if [ -n "$SMOKE" ]; then
            run "prep-$ds" timeout -k 30 1500 "$PY" tools/classical_two_datasets.py prep \
                --data "$DATA" --lanes "$LANES" --datasets "$ds" --max-rows "$SMOKE"
        else
            run "prep-$ds" timeout -k 30 1500 "$PY" tools/classical_two_datasets.py prep \
                --data "$DATA" --lanes "$LANES" --datasets "$ds"
        fi
    done
fi

# ============================================================================
# races, lane order is priority order
# ============================================================================
if has_phase races; then
    for lane in $(echo "$LANES" | tr ',' ' '); do
        for ds in $DATASETS; do
            remaining=$(( BODY_START + BODY_SECONDS - $(date +%s) ))
            if [ "$remaining" -lt "$MIN_RACE" ]; then
                record "race-$lane-$ds" SKIPPED_DEADLINE 0
                continue
            fi
            bound=$RACE_SECONDS
            [ "$remaining" -lt "$bound" ] && bound=$remaining
            run "race-$lane-$ds" timeout -k 30 "$bound" "$PY" tools/classical_two_datasets.py race \
                --lane "$lane" --dataset "$ds" --data "$DATA" --out "$OUT" --work "$WORK" --root "$ROOT" \
                --rounds "$ROUNDS" --arms "$(arms_for "$lane")" \
                --ours-python "$OURS_PY" --theirs-python "$THEIRS_PY"
            # A conductor killed by the bound leaves no worker holding the GPU.
            pkill -9 -f 'classical_two_datasets.py worker' > /dev/null 2>&1
            "$PY" tools/classical_two_datasets.py summary --out "$OUT" > "$OUT/summary.log" 2>&1
        done
    done
fi

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) phases=$PHASES" >> "$OUT/gate.txt"
echo "---- status"
cat "$OUT/status.tsv"
echo "---- summary"
cat "$OUT/summary.tsv" 2>/dev/null
exit 0
