#!/usr/bin/env bash
# Invoked by the guarded DigitalOcean phase-9 diagnostic route.
set -uo pipefail
cd "${REPO:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$
export MOJOLEARN_MAMBA_CERT_VENDOR=amd
export MOJOLEARN_MAMBA_CERT_OUT="$OUT/diag/mamba-cert"
export MOJOLEARN_FOLLOWUP_OUT="$OUT/diag/followup"
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . umap/checks/identity_check.mojo > "$OUT/diag/umap.identity.log" 2>&1
u=$?
pixi run mamba-backward-cert-amd > "$OUT/diag/mamba-console.log" 2>&1
m=$?
bash tools/umap_mamba_followup.sh
f=$?
printf 'umap_identity_exit=%s\nmamba_cert_exit=%s\nfollowup_exit=%s\n' "$u" "$m" "$f" > "$OUT/diag/status.txt"
[ "$u" = 0 ] && [ "$m" = 0 ] && [ "$f" = 0 ]
