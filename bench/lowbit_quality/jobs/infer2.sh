#!/bin/bash
# Follow-up arms of the inference table, each under its own name: the weight
# side with the activation held at 15 bits. Same evaluation set; the record
# carries arm a again and infer_table.py refuses a merge if it moved.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT="$WORK/out/infer2"
mkdir -p "$OUT"
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" \
    --arms "${LOWBIT_QUALITY_ARMS:-a,int10w-int15a,int12w-int15a,int8mw-int15a,int8s1w-int15a,int8w-int15a-fp32head,int8w-int15a-int15head,int8w-int15a-int15mlp,int8w-int15a-int15attnproj}"
rc=$?
echo "infer2 exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
