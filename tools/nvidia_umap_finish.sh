#!/usr/bin/env bash
# Root-owned targeted UMAP runtime, correctness, timing and real-data quality.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${MOJOLEARN_CAMPAIGN_OUT:?artifact directory required}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH="$ROOT/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export MOJOLEARN_CPU_THREADS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
deadline=$(($(date +%s) + ${MOJOLEARN_CAMPAIGN_SECONDS:-3000} - 30))
rc=0
[[ ! -e "$OUT/results.tsv" ]] || exit 9
: > "$OUT/results.tsv"
run() {
    local name=$1 cap=$2 remaining status
    shift 2
    remaining=$((deadline - $(date +%s)))
    if ((remaining < 15)); then
        printf '%s\t124\tSKIPPED_DEADLINE\n' "$name" >> "$OUT/results.tsv"
        rc=1
        return 124
    fi
    ((cap > remaining)) && cap=$remaining
    python3 tools/nvidia_serial_guard.py --seconds "$cap" -- "$@" > "$OUT/$name.log" 2>&1
    status=$?
    printf '%s\t%s\n' "$name" "$status" | tee -a "$OUT/results.tsv"
    ((status == 0)) || rc=1
    return "$status"
}
trap 'status=$?; printf "%s\n" "$status" > "$OUT/exit_code"' EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/nvidia.csv" || exit 9
printf '%s\n' "source=${MOJOLEARN_COMMIT:-unknown}" 'targeted NVIDIA UMAP; not wheel or cross-vendor certification' > "$OUT/provenance.txt"
run vendor-venv 30 python3 -m venv --system-site-packages .nvidia-feature-venv || exit 1
PY="$ROOT/.nvidia-feature-venv/bin/python"
export MOJOLEARN_PYTHON="$PY"
run vendor-wheels 300 "$PY" -m pip install --disable-pip-version-check --no-input --only-binary=:all: \
    numpy==2.4.6 scikit-learn==1.9.0 cupy-cuda12x==14.2.0 cuml-cu12==26.8.0 umap-learn --extra-index-url https://pypi.nvidia.com || exit 1
run vendor-freeze 30 "$PY" -m pip freeze
# Refuse incompatible external libraries before spending time on compilation.
run external-runtime 110 "$PY" tools/nvidia_public_compare.py --lane umap --probe-external --umap-rows 1024 --out "$OUT/external-runtime" || exit 1
run guard-checks 60 "$PY" -m unittest discover -s tools -p 'test_*serial_guard.py'
run admission-checks 60 "$PY" -m unittest discover -s tools -p 'test_nvidia_*finish_validate.py'
mkdir -p "$OUT/bindings"
for mode in identical fast; do
    export MOJOLEARN_NUMERIC_MODE=$mode
    run "build-metrics-$mode" 600 bash bindings/build_metrics.sh || continue
    run "umap-api-$mode" 240 "$PY" -m unittest mojolearn.tests.test_umap_surface mojolearn.tests.test_umap_transform
    binding="$ROOT/python/mojolearn/_mojolearn_metrics.so"
    [[ "$mode" != identical ]] || binding="$ROOT/python/mojolearn/identical/_mojolearn_metrics.so"
    run "retain-metrics-$mode" 30 cp "$binding" "$OUT/bindings/metrics-$mode.so"
done
run compare-umap 600 "$PY" tools/nvidia_public_compare.py --lane umap --umap-rows 1024 --umap-epochs 50 --out "$OUT/umap-three-arm"
# Capture and pin the real dataset BEFORE observing model quality.
if run digits-capture 30 "$PY" tools/umap_real_dataset_quality.py --capture-data-only --out "$OUT/digits-pin"; then
    data_sha=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_sha256"])' "$OUT/digits-pin/dataset.json")
    run digits-quality 1500 "$PY" tools/umap_real_dataset_quality.py --expected-data-sha256 "$data_sha" --out "$OUT/digits-quality"
else
    printf 'digits-quality\t124\tSKIPPED_DATASET\n' >> "$OUT/results.tsv"
    rc=1
fi
exit "$rc"
