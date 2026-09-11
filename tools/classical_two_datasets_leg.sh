#!/bin/sh
# tools/classical_two_datasets_leg.sh -- DEVIATION 2570, the classical
# opponent rows on the two benchmark datasets (ENGINEERING_RULES.md sections
# 9 and 10). Runs ON THE DROPLET as tools/do_extra_leg.sh's
# MOJOLEARN_GEMM_LEG_EXTRA body, from /root/mojolearn with pixi on PATH and the
# default environment installed; /root/gemm_leg_out/classical-two-datasets/
# comes home with the runner's fetch (data, venv and arm outputs live outside
# it on purpose: the uplink is slow).
#
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/classical_two_datasets_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/classical-two-datasets-$(date -u +%Y-%m-%d_%H%M%S)-amd-leg1 \
#   bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
#
# The runner passes no environment to a body, so the lane list is this file's
# default (kmeans,pca,ols,knn: the lanes with a torch arm) and the second lease
# runs tools/classical_two_datasets_leg2.sh (kde,svc), which sets
# MOJOLEARN_CTD_LANES and execs this file.
#
# PHASES, each with a row in status.tsv (name, exit, seconds); a red phase does
# not stop the next, because a red phase is a finding:
#   env                  nproc, lscpu, free, rocm-smi / amd-smi / nvidia-smi,
#                        /opt/rocm/.info/version, the amdgpu module version
#   download-istella     BACKGROUND, the pixi python (numpy only):
#                        tools/speed_gbdt_arm.py --download istella
#   venv, wheels         BACKGROUND, a venv of the image's python3: on AMD the
#                        torch ROCm pin tools/do_byte_lm_setup.sh installed
#                        (torch 2.6.0+rocm6.4.1.git1ded221d, pytorch-triton-rocm
#                        3.2.0+rocm6.4.1.git6da9e660, sha256-pinned as in
#                        tools/torch_lm_step_opponent_leg.sh); on NVIDIA
#                        torch 2.4.1+cu124. Beside it numpy 1.26.4, scipy
#                        1.14.1, scikit-learn 1.5.2, threadpoolctl 3.5.0,
#                        joblib 1.4.2, pyarrow 17.0.0 (taxi's decode).
#   download-taxi        BACKGROUND after wheels: --download taxi
#   build-*              IDENTICAL bindings for the lanes asked for:
#                        bindings/build.sh (kmeans, knn), build_estimators.sh
#                        (pca, ols, kde), build_svm.sh (svc)
#   ours-import          the binding imports and reads back identical
#   prep-<dataset>       tools/classical_two_datasets.py prep
#   race-<lane>-<dataset>  one conductor each (1 warm-up + 5 interleaved
#                        rounds, quality), under timeout; SKIPPED_DEADLINE when
#                        the body deadline is closer than MOJOLEARN_CTD_MIN_RACE
#   summary.tsv          rewritten after every race, so a partial leg still
#                        carries a table
#
# THREADS: OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and
# NUMEXPR_NUM_THREADS are UNSET for the whole body. scikit-learn runs on every
# core as installed and each race's JSON names the pools threadpoolctl reports.
#
# POSIX sh only: the droplet's /bin/sh is dash.
set -u
ROOT=${MOJOLEARN_CTD_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_CTD_OUT:-/root/gemm_leg_out/classical-two-datasets}
DATA=${MOJOLEARN_CTD_DATA:-/root/ctd-data}
WORK=${MOJOLEARN_CTD_WORK:-/root/ctd-work}
VENV=${MOJOLEARN_CTD_VENV:-/root/ctd-venv}
LANES=${MOJOLEARN_CTD_LANES:-kmeans,pca,ols,knn}
DATASETS=${MOJOLEARN_CTD_DATASETS:-taxi istella}
ROUNDS=${MOJOLEARN_CTD_ROUNDS:-5}
BODY_SECONDS=${MOJOLEARN_CTD_BODY_SECONDS:-2100}
MIN_RACE=${MOJOLEARN_CTD_MIN_RACE:-240}
RACE_SECONDS=${MOJOLEARN_CTD_RACE_SECONDS:-900}
JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
BODY_START=$(date +%s)

mkdir -p "$OUT" "$DATA" "$WORK"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
export MOJOLEARN_NUMERIC_MODE=identical
GBM_BENCH_DATA=${GBM_BENCH_DATA:-/root/datasets/gbm-bench}
export GBM_BENCH_DATA
COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
MOJOLEARN_REPO_COMMIT=${COMMIT:-unknown}
export MOJOLEARN_REPO_COMMIT
VENDOR=$(sed -n 's/^vendor=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
if [ -z "$VENDOR" ]; then
    if [ -e /dev/kfd ] && command -v rocm-smi > /dev/null 2>&1; then VENDOR=amd
    elif command -v nvidia-smi > /dev/null 2>&1; then VENDOR=nvidia
    else VENDOR=unknown; fi
fi
: > "$OUT/status.tsv"
{
    echo "lane=classical-two-datasets (DEVIATION 2570)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "vendor=$VENDOR"
    echo "commit=$MOJOLEARN_REPO_COMMIT"
    echo "lanes=$LANES"
    echo "datasets=$DATASETS"
    echo "rounds=$ROUNDS body_seconds=$BODY_SECONDS min_race=$MIN_RACE race_seconds=$RACE_SECONDS jobs=$JOBS"
    echo "gpu_archs=${MOJOLEARN_GPU_ARCHS:-}"
} > "$OUT/gate.txt"

record() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$OUT/status.tsv"
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

# ---- env ------------------------------------------------------------------
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
    fi
    if [ -f /opt/rocm/.info/version ]; then
        echo "== /opt/rocm/.info/version"; cat /opt/rocm/.info/version
    fi
    if [ -f /sys/module/amdgpu/version ]; then
        echo "== /sys/module/amdgpu/version"; cat /sys/module/amdgpu/version
    fi
    if command -v nvidia-smi > /dev/null 2>&1; then
        echo "== nvidia-smi"; nvidia-smi --query-gpu=name,driver_version --format=csv 2>&1
    fi
} > "$OUT/env.txt" 2>&1
record env 0 0

# ---- background: Istella (numpy only, the pixi python) -----------------------
( run download-istella timeout -k 30 1500 pixi run python3 tools/speed_gbdt_arm.py --download istella ) &
PID_ISTELLA=$!

# ---- background: the opponents' venv, then taxi (needs pyarrow) ---------------
(
    if [ "$VENDOR" = amd ]; then
        BASE=/usr/bin/python3
        PIN="https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl#sha256=1d97c15798bf178299328032141a21d9777e7cdef59d5a7e3ac74e297c17198e https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl#sha256=6b141e1a03148b007c6217519cd9947d760123ded5caebadffec22cba7358d2d"
        INDEX=""
    else
        BASE=python3
        PIN="torch==2.4.1"
        INDEX="--index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple"
    fi
    if ! "$BASE" -m venv "$VENV" > "$OUT/venv-first.log" 2>&1; then
        DEBIAN_FRONTEND=noninteractive timeout -k 10 120 apt-get update -qq > "$OUT/venv-apt.log" 2>&1
        DEBIAN_FRONTEND=noninteractive timeout -k 10 300 apt-get install -y -qq python3-venv python3-pip >> "$OUT/venv-apt.log" 2>&1
    fi
    run venv "$BASE" -m venv "$VENV"
    # shellcheck disable=SC2086
    run wheels timeout -k 30 1200 "$VENV/bin/pip" install --disable-pip-version-check --no-input \
        --only-binary=:all: $INDEX $PIN numpy==1.26.4 scipy==1.14.1 scikit-learn==1.5.2 \
        threadpoolctl==3.5.0 joblib==1.4.2 pyarrow==17.0.0
    "$VENV/bin/pip" freeze > "$OUT/venv_freeze.txt" 2>&1
    "$VENV/bin/python" -c 'import torch, sklearn, scipy, numpy; print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None), "cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"); print("sklearn", sklearn.__version__, "scipy", scipy.__version__, "numpy", numpy.__version__)' > "$OUT/venv_versions.txt" 2>&1
    run download-taxi timeout -k 30 1200 "$VENV/bin/python" tools/speed_gbdt_arm.py --download taxi
) &
PID_VENV=$!

# ---- foreground: the IDENTICAL bindings the lanes need -------------------------
BUILDS=""
wants kmeans knn && BUILDS="$BUILDS build.sh"
wants pca ols kde && BUILDS="$BUILDS build_estimators.sh"
wants svc && BUILDS="$BUILDS build_svm.sh"
for b in $BUILDS; do
    run "build-${b%.sh}" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS="$JOBS" \
        timeout -k 30 1500 sh "bindings/$b"
done
ls -la python/mojolearn/identical > "$OUT/identical_bindings.txt" 2>&1
run ours-import env PYTHONPATH="$ROOT/python" MOJOLEARN_NUMERIC_MODE=identical timeout -k 10 300 \
    pixi run python3 -c 'import mojolearn; print("mojolearn", mojolearn.__version__)'

wait "$PID_ISTELLA"
wait "$PID_VENV"
record background-joined 0 "$(( $(date +%s) - BODY_START ))"

# ---- prep, one dataset at a time (a failed download reds only its own rows) ----
for ds in $DATASETS; do
    run "prep-$ds" timeout -k 30 1200 "$VENV/bin/python" tools/classical_two_datasets.py prep \
        --data "$DATA" --lanes "$LANES" --datasets "$ds"
done

# ---- races, lane order is priority order ----------------------------------------
for lane in $(echo "$LANES" | tr ',' ' '); do
    for ds in $DATASETS; do
        remaining=$(( BODY_START + BODY_SECONDS - $(date +%s) ))
        if [ "$remaining" -lt "$MIN_RACE" ]; then
            record "race-$lane-$ds" SKIPPED_DEADLINE 0
            continue
        fi
        bound=$RACE_SECONDS
        [ "$remaining" -lt "$bound" ] && bound=$remaining
        run "race-$lane-$ds" timeout -k 30 "$bound" "$VENV/bin/python" tools/classical_two_datasets.py race \
            --lane "$lane" --dataset "$ds" --data "$DATA" --out "$OUT" --work "$WORK" --root "$ROOT" \
            --rounds "$ROUNDS" --ours-python "pixi run python3" --theirs-python "$VENV/bin/python"
        # A conductor killed by the bound leaves no worker holding the GPU.
        pkill -9 -f 'classical_two_datasets.py worker' > /dev/null 2>&1
        "$VENV/bin/python" tools/classical_two_datasets.py summary --out "$OUT" > "$OUT/summary.log" 2>&1
    done
done

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
echo "---- status"
cat "$OUT/status.tsv"
echo "---- summary"
cat "$OUT/summary.tsv" 2>/dev/null
exit 0
