#!/usr/bin/env bash
# Root/main only, inside an already leased Linux CUDA/HIP host. Source authored;
# this file does not provision, infer GPU architecture, or claim qualification.
# Requires an installed Pixi environment and a system Torch for the rented GPU.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ $(uname -s) == Linux ]] || { echo 'Remote Linux only; no Apple execution' >&2; exit 2; }
vendor=${MOJOLEARN_BYTE_LM_EXPECT_VENDOR:?set cuda or hip}
arch=${MOJOLEARN_GPU_ARCHS:?supply the actual single rented GPU architecture}
case "$vendor:$arch" in
    cuda:sm_[0-9]*) guard=tools/nvidia_serial_guard.py ;;
    hip:gfx[0-9]*) guard=tools/amd_serial_guard.py ;;
    *) echo 'Expected cuda with sm_NN or hip with gfxNNN; architecture is never inferred' >&2; exit 2 ;;
esac
[[ "$arch" != *[!A-Za-z0-9_]* ]] || { echo 'One GPU architecture required' >&2; exit 2; }
export MOJOLEARN_GPU_ARCHS="$arch"
OUT=${MOJOLEARN_BYTE_LM_VALIDATION_OUT:?new absolute output directory required}
[[ "$OUT" = /* && ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'Output must be a new absolute path' >&2; exit 2; }
seconds=${MOJOLEARN_BYTE_LM_VALIDATION_SECONDS:-3000}
[[ "$seconds" =~ ^[0-9]+$ ]] && ((seconds >= 60 && seconds <= 3000)) || exit 2
commit=${MOJOLEARN_COMMIT:?frozen source commit required}
mkdir -p "$OUT"
deadline=$(($(date +%s) + seconds - 30))
trap 'status=$?; printf "%s\n" "$status" > "$OUT/exit_code"' EXIT
export PYTHONPATH="$ROOT/python:$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2 MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_CPU_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=${MOJOLEARN_PYTHON:-python3}
: > "$OUT/results.tsv"
printf '%s\n' "source=$commit" "vendor=$vendor" "gpu_arch=$arch" \
    'scope=single-vendor real-byte training and independent gradient gate; no cross-vendor certificate' \
    > "$OUT/provenance.txt"

run() {
    local name=$1 cap=$2 remaining status
    shift 2
    remaining=$((deadline - $(date +%s)))
    if ((remaining < 15)); then
        printf '%s\t124\tSKIPPED_DEADLINE\n' "$name" >> "$OUT/results.tsv"
        return 124
    fi
    ((cap <= remaining)) || cap=$remaining
    # Retain the exact guarded argv before launch, in shell-readable form.
    printf '%q ' python3 "$guard" --seconds "$cap" -- "$@" > "$OUT/$name.command.txt"
    printf '\n' >> "$OUT/$name.command.txt"
    status=0
    python3 "$guard" --seconds "$cap" -- "$@" > "$OUT/$name.log" 2>&1 || status=$?
    printf '%s\t%s\n' "$name" "$status" | tee -a "$OUT/results.tsv"
    return "$status"
}

receipt() {
    # Called only after run returned zero, including guard cleanup. Any receipt
    # refusal stops the campaign; a summary alone is never success evidence.
    local name=$1 kind=$2 result=$3
    "$PY" tools/root_job_receipt.py --vendor "$vendor" --exit-code 0 \
        --job-kind "$kind" --command-file "$OUT/$name.command.txt" \
        --guard-log "$OUT/$name.log" --result "$result" \
        --output "$OUT/$name.receipt.json"
}

VENV="$ROOT/.byte-lm-validation-venv"
[[ ! -e "$VENV" && ! -L "$VENV" ]] || { echo 'Refusing existing byte-LM environment' >&2; exit 2; }
run byte-venv 30 "$PY" -m venv --system-site-packages "$VENV"
PY="$VENV/bin/python"
export MOJOLEARN_PYTHON="$PY"
run byte-dependencies 120 "$PY" -m pip install --disable-pip-version-check \
    --no-input --only-binary=:all: numpy==1.26.4 pytest==8.3.5
run dependencies 30 "$PY" -c 'import numpy, pytest, torch; print(numpy.__version__, pytest.__version__, torch.__version__)'
run dependency-freeze 30 "$PY" -m pip freeze
run guard-checks 60 "$PY" -m unittest discover -s tools -p 'test_*serial_guard.py'
run comparator-fixtures 60 "$PY" -m pytest -q tools/tests/test_byte_lm_state_compare.py
# The build script refuses an existing destination. Use the public package
# location so the ordinary backend loader selects the newly built extension.
export MOJOLEARN_BYTE_LM_OUTDIR="$ROOT/python/mojolearn/identical"
run byte-build 900 bash bindings/build_byte_lm.sh
mkdir "$OUT/bindings"
run retain-binding 30 cp "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" "$OUT/bindings/"
run byte-host-mocks 120 "$PY" -m pytest -q python/mojolearn/tests/test_byte_lm_surface.py
run byte-step1 240 "$PY" tools/byte_lm_real_text_capture.py \
    --output "$OUT/step1" --expected-vendor "$vendor" --steps 1
receipt byte-step1 capture "$OUT/step1/summary.json"
run byte-gradient-oracle 240 "$PY" tools/byte_lm_gradient_oracle.py \
    "$OUT/step1/step000001" --expected-vendor "$vendor" --output "$OUT/gradient-oracle.json"
receipt byte-gradient-oracle oracle "$OUT/gradient-oracle.json"
# set -e prevents expansion after a failed numerical gate or receipt. The
# separate full run starts from the fixed initialization and actual text.
run byte-full128 1500 "$PY" tools/byte_lm_real_text_capture.py \
    --output "$OUT/full128" --expected-vendor "$vendor" --steps 128
receipt byte-full128 capture "$OUT/full128/summary.json"
echo 'Raw captures and root receipts retained. Learning requires summary review; head/resume/control and cross-vendor comparison are separate future runs.'
