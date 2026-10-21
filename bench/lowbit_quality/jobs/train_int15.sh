#!/bin/bash
# TRAINING UNDER THE 15-BIT PROFILE AS A COMPLETE CONFIGURATION (orchestrator,
# 2026-09-29, after the int15 GEMM was shown identical on four boxes): 15-bit
# codes on every projection AND every attention product, forward only, then
# forward and backward; FIVE seeds each against five baseline seeds. The
# zero-code record codes every backward operand at 15 bits, per product.
# No int8 arm, no bf16 arm, no mixed-profile arm is in this job.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ONLY="e:fwd:attn,e:fwdbwd:attn"
export LOWBIT_QUALITY_TRAIN_ARGS="--seeds 5 --baseline-seeds 5 --zero-code-widths int15 --only $ONLY"
rc=0
bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" || rc=$?
"$PY" bench/lowbit_quality/train_table.py --runs "$WORK/out/train" --out "$WORK/out/train" \
    --name training_int15 --only "$ONLY" || rc=$?
echo "train_int15 exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
