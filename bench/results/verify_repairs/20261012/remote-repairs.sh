#!/bin/bash
set -uo pipefail
cd /root/ml-verify
A=/root/ml-verify/artifacts
export MOJOLEARN_COMMIT=87e6005eaad08b3f67a25d4e344a555bb2590e7e MOJOLEARN_NUMERIC_MODE=identical
backend=$(venv/bin/python -I -c 'import mojolearn; print(mojolearn.vendor())')
if [ "$backend" = cuda ]; then vendor=nvidia-rtx4090-sm_89; else vendor=amd-mi325x-gfx942; fi
timeout -k 30 900 venv/bin/python -I -u -m mojolearn._identity_break --lanes gbdt-feature-freq,gbdt-tensor-ctr-tables,linalg-eigh,trees-oob-cv-link,radius,radius-chebyshev,radius-manhattan,radius-minkowski-p3 --repeats 1 --require-backend "$backend" --vendor "$vendor" --json "$A/repaired-column.json" > "$A/repaired-column.log" 2>&1
rc=$?; echo "$rc" > "$A/repaired-column.exit"
printf 'REPAIRED_COLUMN_EXIT=%s\n' "$rc"
