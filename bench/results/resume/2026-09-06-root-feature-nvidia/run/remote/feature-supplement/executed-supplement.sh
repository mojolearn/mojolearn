#!/usr/bin/env bash
set -uo pipefail
cd /root/mojolearn
OUT=/root/gemm_leg_out/feature-supplement
mkdir -p "$OUT"
# Never overlap the original root-owned campaign's compile/test/measurement.
while [[ ! -f /root/gemm_leg.done ]]; do
    if (( $(date +%s) >= 1788713548 )); then exit 124; fi
    sleep 2
done
export PYTHONPATH=/root/mojolearn/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MOJOLEARN_COMMIT=202c44caacd2ee56164da148662393d77a1840d3
PY=/root/mojolearn/.nvidia-feature-venv/bin/python
export MOJOLEARN_PYTHON="$PY"
rc=0
trap 'status=$?; printf "%s\n" "$status" > "$OUT/exit_code"' EXIT
: > "$OUT/results.tsv"
run() {
    local name=$1 cap=$2 remaining status
    shift 2
    remaining=$((1788714028 - 240 - $(date +%s)))
    if ((remaining < 15)); then rc=1; printf '%s\t124\n' "$name" >> "$OUT/results.tsv"; return 124; fi
    ((cap > remaining)) && cap=$remaining
    python3 tools/nvidia_serial_guard.py --seconds "$cap" -- "$@" > "$OUT/$name.log" 2>&1
    status=$?
    printf '%s\t%s\n' "$name" "$status" | tee -a "$OUT/results.tsv"
    ((status == 0)) || rc=1
    return "$status"
}
sha256sum bindings/build_gbdt.sh python/mojolearn/tests/test_mamba23_backward_surface.py > "$OUT/harness-before.sha256"
cp /root/mojolearn-supplement/build_gbdt.sh bindings/build_gbdt.sh
cp /root/mojolearn-supplement/test_mamba23_backward_surface.py python/mojolearn/tests/test_mamba23_backward_surface.py
sha256sum bindings/build_gbdt.sh python/mojolearn/tests/test_mamba23_backward_surface.py > "$OUT/harness-after.sha256"
cp /root/mojolearn-supplement/supplement.sh "$OUT/executed-supplement.sh"
printf '%s\n' 'Native/library sources unchanged from frozen checkpoint; only smoke harness and external oracle setting corrected.' > "$OUT/provenance.txt"
for mode in identical fast; do
    run "mamba-api-$mode" 240 env MOJOLEARN_NUMERIC_MODE="$mode" "$PY" python/mojolearn/tests/test_mamba_surface.py
done
for family in 2 3; do
    mkdir -p "$OUT/mamba$family-native"
    run "mamba$family-native-dump" 240 env MOJOLEARN_MAMBA_GRAD_DUMP="$OUT/mamba$family-native" \
      pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "mamba/checks/mamba${family}_backward_tail_dump.mojo"
done
run mamba23-backward 240 env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_MAMBA23_BACKWARD_NVIDIA=1 \
    MOJOLEARN_MAMBA2_BACKWARD_NATIVE_DUMP="$OUT/mamba2-native" MOJOLEARN_MAMBA3_BACKWARD_NATIVE_DUMP="$OUT/mamba3-native" \
    "$PY" -m unittest mojolearn.tests.test_mamba23_backward_surface -v
if run build-gbdt-fast 600 env MOJOLEARN_NUMERIC_MODE=fast bash bindings/build_gbdt.sh; then
    run gbdt-boundary-fast 120 env MOJOLEARN_NUMERIC_MODE=fast "$PY" -m pytest -q \
      python/mojolearn/tests/test_multiclass_ova_surface.py python/mojolearn/tests/test_gbdt_input_safety.py \
      python/mojolearn/tests/test_gbdt_tree_metadata.py python/mojolearn/tests/test_gbdt_mode_serialization.py
    run compare-gbdt 300 "$PY" tools/nvidia_public_compare.py --lane gbdt --out "$OUT/gbdt-three-arm"
fi
if [[ -f /root/mojolearn-supplement/extra.sh ]]; then source /root/mojolearn-supplement/extra.sh; fi
exit "$rc"
