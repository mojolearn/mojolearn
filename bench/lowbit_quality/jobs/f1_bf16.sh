#!/bin/bash
# FINALIST F1 on both held-out texts, then the training arms in Andrew's order
# (2026-09-29): bf16 on both operands, forward and backward, FIRST; then 15-bit
# codes the same way; complete configurations (attention included) before the
# projections alone; five seeds each. No arm that codes an operand in 8 bits
# and no F2 arm is run: both are dropped (2026-09-29, Andrew).
# Training records go into the directory the earlier plan's runs are in; a run
# whose record exists is not repeated.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GITHUB=${LOWBIT_QUALITY_CORPUS2:-/root/mojolearn-wt/lowbit-quality/training/corpus/pile_github/input.txt}
[ -f "$GITHUB" ] || { echo "REFUSED: $GITHUB is not staged"; exit 3; }
rc=0
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$CORPUS" --corpus-key corpus/enwik8/input.txt \
    --out "$WORK/out/f1_enwik8" --commit "$COMMIT" --arms "a,floor,F1,c+attn" || rc=$?
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$GITHUB" --corpus-key corpus/pile_github/input.txt \
    --tail-from 96000000 --out "$WORK/out/f1_github" --commit "$COMMIT" --arms "a,floor,F1,c+attn" || rc=$?
echo "F1 inference exit=$rc"
export LOWBIT_QUALITY_TRAIN_ARGS="--seeds 5 --only c:fwdbwd:attn,c:fwd:attn,e:fwdbwd:attn,e:fwd:attn,c:fwdbwd:proj,c:fwd:proj,e:fwdbwd:proj,e:fwd:proj"
bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" || rc=$?
echo "f1_bf16 exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
