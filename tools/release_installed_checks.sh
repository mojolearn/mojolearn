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
bash "$ROOT/tools/linux_surface_qualification.sh" "$@"
# A supplemental failure must invalidate the successful surface marker too.
printf '1\n' > "$OUT/exit_code"
VPY="$OUT/venv/bin/python"
cd "$OUT"
timeout -k 10 600 env -u PYTHONPATH -u PYTHONHOME \
    MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1 \
    "$VPY" -m mojolearn._identity_break \
    --lanes mamba1,mamba2,mamba2-dtlimit,mamba3 --repeats 2 \
    --require-backend "$VENDOR" --fail-on-refused --vendor "$COLUMN" \
    --json "$OUT/mamba-column.json" > "$OUT/mamba-column.log" 2>&1
timeout -k 10 900 "$VPY" "$ROOT/tools/qualify_verifier_wheel.py" "$WHEEL" \
    --python "$VPY" --output "$OUT/verifier-cli" > "$OUT/verifier-cli.log" 2>&1
printf '0\n' > "$OUT/exit_code"
