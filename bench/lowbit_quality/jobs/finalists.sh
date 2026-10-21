#!/bin/bash
# THE TWO FINALISTS (orchestrator, 2026-09-29 03:35Z), each one complete
# configuration, on both held-out texts, then in training.
#   F1  15-bit codes on every projection and every attention product
#   F2  15-bit codes on every projection, int8 codes on every attention product
# Training: forward only, then forward and backward, five seeds each, into the
# training directory the plan's runs are in (a run whose record exists is not
# repeated; F1 in training is arm e with the attention switch on).
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GITHUB=${LOWBIT_QUALITY_CORPUS2:-/root/mojolearn-wt/lowbit-quality/training/corpus/pile_github/input.txt}
[ -f "$GITHUB" ] || { echo "REFUSED: $GITHUB is not staged"; exit 3; }
rc=0
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$CORPUS" --corpus-key corpus/enwik8/input.txt \
    --out "$WORK/out/finalists_enwik8" --commit "$COMMIT" --arms "a,floor,F1,F2" || rc=$?
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$GITHUB" --corpus-key corpus/pile_github/input.txt \
    --tail-from 96000000 --out "$WORK/out/finalists_github" --commit "$COMMIT" --arms "a,floor,F1,F2" || rc=$?
echo "finalists inference exit=$rc"
export LOWBIT_QUALITY_TRAIN_ARGS="--seeds 5 --only F2:fwd:attn,F2:fwdbwd:attn,e:fwd:attn,e:fwdbwd:attn"
bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" || rc=$?
echo "finalists exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
