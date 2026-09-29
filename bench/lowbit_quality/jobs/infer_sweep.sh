#!/bin/bash
# The second held-out text and the width sweep (orchestrator, 2026-09-29).
#   1. the arms that matter on corpus/pile_github (code), the second text
#   2. the width sweep and the weight-side follow-ups on corpus/enwik8
#   3. the same on corpus/pile_github
# Each step writes its own record; bench/lowbit_quality/infer_table.py merges
# records of one text and refuses a merge whose baseline or ids moved.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GITHUB=${LOWBIT_QUALITY_CORPUS2:-/root/mojolearn-wt/lowbit-quality/training/corpus/pile_github/input.txt}
[ -f "$GITHUB" ] || { echo "REFUSED: $GITHUB is not staged (tools/dataset_store.sh stage ... corpus/pile_github/input.txt)"; exit 3; }
SWEEP=""
for w in 8 10 12 15; do for a in 15 12 10 8; do SWEEP="$SWEEP,sweep-w$w-a$a"; done; done
FOLLOW="int8mw-int15a,int8s1w-int15a,int8w-int15a-fp32head,int8w-int15a-int15head,int8w-int15a-int15mlp,int8w-int15a-int15attnproj,int8m-both"
rc=0
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$GITHUB" --corpus-key corpus/pile_github/input.txt \
    --tail-from 96000000 --out "$WORK/out/infer_github" --commit "$COMMIT" \
    --arms "a,floor,b,c,e,c+attn,e+attn,d,f,d+attn,f+attn,int8w-fp32a,fp32w-int8a" || rc=$?
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$CORPUS" --corpus-key corpus/enwik8/input.txt \
    --out "$WORK/out/infer_sweep_enwik8" --commit "$COMMIT" --arms "a$SWEEP,$FOLLOW" || rc=$?
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$GITHUB" --corpus-key corpus/pile_github/input.txt \
    --tail-from 96000000 --out "$WORK/out/infer_sweep_github" --commit "$COMMIT" --arms "a$SWEEP,$FOLLOW" || rc=$?
echo "infer_sweep exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
