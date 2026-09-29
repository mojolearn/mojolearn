#!/bin/bash
# Backward vectors of the 15-bit profile from a LATER training step
# (orchestrator, 2026-09-29): step 4000 of the trajectory of run
# e.fwdbwd.attn.s0, where the weight rows' exponents have spread, so that
# the two quantization rules give different bits in the backward cases.
export LOWBIT_QUALITY_NEED_MODEL=0
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" bench/lowbit_quality/backward_export.py --corpus "$CORPUS" --out "$WORK/out/backward_vectors" \
    --device cuda --steps "${LOWBIT_QUALITY_EXPORT_STEP:-4000}" --seed 0 --validate \
    --name "int15_backward_vectors_step${LOWBIT_QUALITY_EXPORT_STEP:-4000}" --commit "$COMMIT"
rc=$?
echo "backward_export exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
