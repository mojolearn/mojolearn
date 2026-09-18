#!/usr/bin/env bash
# Supplemental release checks on the exact installed Linux wheel. The rental
# controller bounds this whole script, including the existing surface gate.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[[ $# = 7 && $1 = qualify-release-linux3 ]] || { echo 'Expected qualify-release-linux3 WHEEL SHA VENDOR OUT PROOFS ARCH' >&2; exit 2; }
WHEEL=$(realpath "$2")
VENDOR=$4
OUT=$(realpath -m "$5")
ARCH=$7
case "$VENDOR/$ARCH" in
    cuda/sm_90a) COLUMN=nvidia-h100-sm_90a ;;
    cuda/sm_89) COLUMN=nvidia-l40s-sm_89 ;;
    hip/gfx942) COLUMN=amd-mi325x-gfx942 ;;
    *) echo 'Unsupported release device column' >&2; exit 2 ;;
esac
# Both the existing gate and the supplemental processes inherit this cap.
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
taskset -pc "$cores" $$
export MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_CPU_THREADS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_THREAD_LIMIT=1 OMP_MAX_ACTIVE_LEVELS=1 BLIS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
bash "$ROOT/tools/release_linux_surface_qualification.sh" "$@"
# A supplemental failure must invalidate the successful surface marker too.
printf '1\n' > "$OUT/exit_code"
VPY="$OUT/venv/bin/python"
# The installed harness runs outside the checkout. Carry the exact archived
# tools/source commit explicitly; an absent witness must still refuse.
if [[ -s "$ROOT/commit.txt" ]]; then
    MOJOLEARN_COMMIT=$(cat "$ROOT/commit.txt")
else
    MOJOLEARN_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
fi
[[ "$MOJOLEARN_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid qualification commit witness' >&2; exit 2; }
export MOJOLEARN_COMMIT
# Saved-model gates use their own provenance variable; archived source has no
# .git directory from which to recover it.
export MOJOLEARN_GATE_COMMIT="$MOJOLEARN_COMMIT"
cd "$OUT"
timeout -k 10 600 env -u PYTHONPATH -u PYTHONHOME \
    MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1 \
    "$VPY" -m mojolearn._identity_break \
    --lanes mamba1,mamba2,mamba2-dtlimit,mamba3 --repeats 2 \
    --require-backend "$VENDOR" --fail-on-refused --vendor "$COLUMN" \
    --json "$OUT/mamba-column.json" > "$OUT/mamba-column.log" 2>&1
# These properties lack current three-vendor columns in the historical CPU
# gate. A prediction-arithmetic fault need not alter a saved model's bytes;
# retain actual independent device references instead of weakening that gate.
EXTRA_LANES=transformer,transformer-window,samba,samba-untied-dropout-accum,radius,radius-chebyshev,radius-manhattan,radius-minkowski-p3,dbscan,dbscan-brute-l1,dbscan-weighted,minmax-scaler,minmax-scaler-clip,standard-scaler-no-std,optim-sgd,spectral-precomputed,metrics-homogeneity-completeness,ordered-gradient-sum,mamba1-bf16w,mamba1-int8w,mlp-bf16w,mlp-int8w,samba-bf16w,samba-int8w
timeout -k 10 900 env -u PYTHONPATH -u PYTHONHOME \
    MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1 \
    "$VPY" -m mojolearn._identity_break \
    --lanes "$EXTRA_LANES" --repeats 2 --require-backend "$VENDOR" \
    --fail-on-refused --vendor "$COLUMN" \
    --json "$OUT/property-column.json" > "$OUT/property-column.log" 2>&1
# UMAP's row-separable transform changed the inference hashes after the old
# saved-model recordings. Capture fresh evidence on each physical vendor from
# this exact installed wheel, then replay its saved GPU models on the CPU.
# These files are evidence to review for explicit reference supersession; this
# script never edits the historical references or admits its own new hashes.
timeout -k 10 600 env -u PYTHONPATH -u PYTHONHOME \
    MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1 \
    "$VPY" -m mojolearn._identity_break \
    --lanes umap --repeats 2 --require-backend "$VENDOR" \
    --fail-on-refused --vendor "$COLUMN" \
    --json "$OUT/umap-column.json" > "$OUT/umap-column.log" 2>&1
timeout -k 10 600 env -u PYTHONPATH -u PYTHONHOME \
    MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1 \
    "$VPY" "$ROOT/tools/classical_host_gate.py" --package-root '' \
    record "$OUT/umap-saved-models" --lanes umap \
    > "$OUT/umap-saved-models.log" 2>&1
timeout -k 10 300 env -u PYTHONPATH -u PYTHONHOME \
    MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1 \
    "$VPY" "$ROOT/tools/classical_host_gate.py" --package-root '' \
    check "$OUT/umap-saved-models" --gpu-column "$OUT/umap-column.json" \
    --report "$OUT/umap-cpu-replay.json" > "$OUT/umap-cpu-replay.log" 2>&1
timeout -k 10 900 "$VPY" "$ROOT/tools/qualify_verifier_wheel.py" "$WHEEL" \
    --python "$VPY" --output "$OUT/verifier-cli" > "$OUT/verifier-cli.log" 2>&1
printf '0\n' > "$OUT/exit_code"
