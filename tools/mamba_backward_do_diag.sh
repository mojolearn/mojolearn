#!/usr/bin/env bash
# Serial backward-only corrective campaign; guarded by the parent DO leg.
set -uo pipefail
cd "${REPO:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$
rc=0
MOJOLEARN_MAMBA_CERT_VENDOR=amd MOJOLEARN_MAMBA_CERT_OUT="$OUT/diag/mamba-cert" \
    pixi run bash tools/mamba_backward_certify.sh > "$OUT/diag/mamba-console.log" 2>&1 || rc=1
MOJOLEARN_MAMBA_CERT_VENDOR=amd MOJOLEARN_MAMBA_CERT_PROFILE=long-sequence-v1 \
    MOJOLEARN_MAMBA_CERT_OUT="$OUT/diag/followup/mamba-long-cert" \
    pixi run bash tools/mamba_backward_certify.sh > "$OUT/diag/mamba-long-console.log" 2>&1 || rc=1
printf 'backward_exit=%s\n' "$rc" > "$OUT/diag/status.txt"
exit "$rc"
