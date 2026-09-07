#!/usr/bin/env bash
# Root-owned serial source qualification, exactly one CUDA comparator per lane.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${MOJOLEARN_CAMPAIGN_OUT:?artifact directory required}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH="$ROOT/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_CPU_THREADS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=python3
deadline=$(($(date +%s) + ${MOJOLEARN_CAMPAIGN_SECONDS:-3000} - 30))
rc=0
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
printf '%s\n' "source=${MOJOLEARN_COMMIT:-unknown}" 'source-built bindings; not release-wheel certification' > "$OUT/provenance.txt"
run vendor-venv 30 python3 -m venv --system-site-packages .nvidia-feature-venv || exit 1
PY="$ROOT/.nvidia-feature-venv/bin/python"
export MOJOLEARN_PYTHON="$PY"
run vendor-wheels 300 "$PY" -m pip install --disable-pip-version-check --no-input --only-binary=:all: \
    numpy pytest einops catboost cupy-cuda12x cuml-cu12 --extra-index-url https://pypi.nvidia.com
run vendor-freeze 30 "$PY" -m pip freeze
run vendor-cuda 30 "$PY" -c 'import torch; assert torch.cuda.is_available() and not torch.version.hip; print(torch.__version__, torch.version.cuda)' || exit 1
run mamba-corpus 240 "$PY" mamba/corpus/gen_corpus.py

for mode in identical fast; do
    export MOJOLEARN_NUMERIC_MODE=$mode
    for binding in gbdt metrics mamba; do
        run "build-$binding-$mode" 600 bash "bindings/build_$binding.sh" || continue
        case "$binding" in
            gbdt)
                run "gbdt-boundary-$mode" 120 "$PY" -m pytest -q python/mojolearn/tests/test_multiclass_ova_surface.py python/mojolearn/tests/test_gbdt_input_safety.py python/mojolearn/tests/test_gbdt_tree_metadata.py python/mojolearn/tests/test_gbdt_mode_serialization.py
                ;;
            metrics)
                run "umap-api-$mode" 240 "$PY" -m unittest mojolearn.tests.test_umap_surface mojolearn.tests.test_umap_transform
                ;;
            mamba)
                run "mamba-api-$mode" 240 "$PY" python/mojolearn/tests/test_mamba_surface.py
                if [[ "$mode" == identical ]]; then
                    for generation in 2 3; do
                        run "mamba$generation-native-dump" 240 env MOJOLEARN_MAMBA_GRAD_DUMP="$OUT/mamba$generation-native" pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "mamba/checks/mamba${generation}_backward_tail_dump.mojo"
                    done
                    run mamba23-backward 240 env MOJOLEARN_MAMBA23_BACKWARD_NVIDIA=1 MOJOLEARN_MAMBA2_BACKWARD_NATIVE_DUMP="$OUT/mamba2-native" MOJOLEARN_MAMBA3_BACKWARD_NATIVE_DUMP="$OUT/mamba3-native" "$PY" -m unittest mojolearn.tests.test_mamba23_backward_surface
                fi
                ;;
        esac
    done
done
run nan-forbidden 180 pixi run mojo run -I . checks/nan_forbidden_host_check.mojo
run umap-finite-params 180 pixi run check-umap-finite-params
run umap-controls-fast 180 pixi run mojo run -I . umap/checks/optimizer_controls_check.mojo
run umap-controls-identical 180 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . umap/checks/optimizer_controls_check.mojo
run umap-identity 240 pixi run check-umap-stage-identity
# Exactly one external for each case: CatBoost CUDA, cuML CUDA. All three
# arms rotate across seven rounds on this same GPU, including host transfers.
run compare-gbdt 300 "$PY" tools/nvidia_public_compare.py --lane gbdt --out "$OUT/gbdt-three-arm"
# 1024 reaches the native FAST GPU optimizer, unlike small serial fixtures.
run compare-umap 600 "$PY" tools/nvidia_public_compare.py --lane umap --umap-rows 1024 --umap-epochs 50 --out "$OUT/umap-three-arm"
exit "$rc"
