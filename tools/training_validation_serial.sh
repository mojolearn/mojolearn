#!/usr/bin/env bash
# Root/main execution only. Run inside an already leased remote Linux host.
# This script does not provision hosts, install dependencies, or qualify Metal.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ $(uname -s) == Linux ]] || { echo 'Remote Linux only; no Apple execution' >&2; exit 2; }
vendor=${MOJOLEARN_TRAIN_EXPECT_VENDOR:?set cuda or hip}
case "$vendor" in
    cuda) guard=tools/nvidia_serial_guard.py ;;
    hip) guard=tools/amd_serial_guard.py ;;
    *) echo 'Expected vendor must be cuda or hip' >&2; exit 2 ;;
esac
OUT=${MOJOLEARN_TRAIN_VALIDATION_OUT:?new absolute artifact directory required}
[[ "$OUT" = /* && ! -e "$OUT" ]] || { echo 'Output must be a new absolute path' >&2; exit 2; }
mkdir -p "$OUT"
PY=${MOJOLEARN_PYTHON:-python3}
export PYTHONPATH="$ROOT/python:$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2 MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_CPU_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_RUN_SMALL_MLP_GPU=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
seconds=${MOJOLEARN_TRAIN_VALIDATION_SECONDS:-2400}
[[ "$seconds" =~ ^[0-9]+$ ]] && ((seconds >= 60 && seconds <= 3000)) || exit 2
deadline=$(($(date +%s) + seconds - 30))
trap 'status=$?; printf "%s\n" "$status" > "$OUT/exit_code"' EXIT
: > "$OUT/results.tsv"
run() {
    local name=$1 cap=$2 remaining status
    shift 2
    remaining=$((deadline - $(date +%s)))
    if ((remaining < 15)); then
        printf '%s\t124\tSKIPPED_DEADLINE\n' "$name" >> "$OUT/results.tsv"
        return 124
    fi
    ((cap <= remaining)) || cap=$remaining
    status=0
    python3 "$guard" --seconds "$cap" -- "$@" > "$OUT/$name.log" 2>&1 || status=$?
    printf '%s\t%s\n' "$name" "$status" | tee -a "$OUT/results.tsv"
    return "$status"
}
printf '%s\n' "source=${MOJOLEARN_COMMIT:?frozen source commit required}" \
    "vendor=$vendor" 'scope=training integration and independent numerical checks; not a cross-vendor certificate' \
    > "$OUT/provenance.txt"
VENV="$ROOT/.training-validation-venv"
[[ ! -e "$VENV" ]] || { echo 'Refusing an existing training environment' >&2; exit 2; }
run training-venv 30 "$PY" -m venv --system-site-packages "$VENV"
PY="$VENV/bin/python"
export MOJOLEARN_PYTHON="$PY"
run training-dependencies 120 "$PY" -m pip install --disable-pip-version-check \
    --no-input --only-binary=:all: numpy==1.26.4 pytest==8.3.5
run dependencies 30 "$PY" -c 'import numpy, pytest, torch; print(numpy.__version__, pytest.__version__, torch.__version__)'
run dependency-freeze 30 "$PY" -m pip freeze
run guard-checks 60 "$PY" -m unittest discover -s tools -p 'test_*serial_guard.py'
# First admit the historical nonlinear gradient composition independently.
run gradient-build 600 pixi run mojo build -j 2 --target-cpu x86-64-v3 \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/train_gradient_capture.mojo \
    -o "$OUT/train-gradient-capture"
run gradient-capture 180 env MOJOLEARN_TRAIN_CAPTURE_OUTPUT="$OUT/gradient-capture" \
    "$OUT/train-gradient-capture"
run gradient-oracle 180 "$PY" tools/transformer_training_gradient_oracle.py \
    "$OUT/gradient-capture" --expected-vendor "$vendor" --output "$OUT/gradient-oracle.json"
# A failed prerequisite stops here. Never broaden a run after a failed gate.
run training-build 600 bash bindings/build_training.sh
run linalg-build 600 bash bindings/build_linalg.sh
run mlp-surface-and-reference 240 "$PY" -m pytest -q \
    python/mojolearn/tests/test_small_mlp_surface.py
run mlp-numerical-edges 240 "$PY" -m pytest -q \
    python/mojolearn/tests/test_small_mlp_numerical_edges.py
run mlp-continuous 240 "$PY" tools/small_mlp_training_capture.py capture --out "$OUT/mlp-continuous"
run mlp-head 120 "$PY" tools/small_mlp_training_capture.py capture --head --out "$OUT/mlp-head"
run mlp-resume 180 "$PY" tools/small_mlp_training_capture.py capture \
    --resume "$OUT/mlp-head/checkpoint-0008.json" --out "$OUT/mlp-resume"
run mlp-compare 60 "$PY" tools/small_mlp_training_capture.py compare \
    --left "$OUT/mlp-continuous" --right "$OUT/mlp-head" "$OUT/mlp-resume" \
    --output "$OUT/mlp-comparison.json"
mkdir "$OUT/bindings"
run retain-training 30 cp python/mojolearn/identical/_mojolearn_training.so "$OUT/bindings/"
run retain-linalg 30 cp python/mojolearn/identical/_mojolearn_linalg.so "$OUT/bindings/"
echo 'Integration jobs finished. Raw multi-step/vendor-resume qualification is separate.'
