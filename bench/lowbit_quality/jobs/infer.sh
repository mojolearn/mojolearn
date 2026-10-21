#!/bin/bash
# The inference table: the quantizer cross-check first, then the six arms,
# the attention switch, the numerical floor and the follow-up arms (each under
# its own name), 200 windows of 512 ids.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT="$WORK/out/infer"
mkdir -p "$OUT"
rc=0
"$PY" bench/lowbit_quality/quantizer_check.py --model "$MODEL" --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" || rc=$?
echo "quantizer_check exit=$rc"
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" \
    --validate-hf --arms "${LOWBIT_QUALITY_ARMS:-a,floor,b,c,d,e,f,c+attn,d+attn,e+attn,f+attn,int8w-fp32a,fp32w-int8a,int15w-fp32a,fp32w-int15a,int8w-int15a,int8w-int12a,int8w-int10a,int8w-int8s1a,int8w-int8s2a,int8m-both,d-fp32head,d-fp32mlp,d-fp32down,d-fp32attnproj,d-int15down,d-int15head-down,attn-only-int8,attn-only-int15,d-qk,d-pv}" || rc=$?
echo "infer exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
