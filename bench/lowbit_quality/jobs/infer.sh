#!/bin/bash
# The inference table: the six arms, the attention switch, the numerical
# floor, 200 windows of 512 ids. Then the quantizer cross-check record.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT="$WORK/out/infer"
mkdir -p "$OUT"
rc=0
"$PY" bench/lowbit_quality/quantizer_check.py --model "$MODEL" --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" || rc=$?
echo "quantizer_check exit=$rc"
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" \
    --validate-hf --arms "${LOWBIT_QUALITY_ARMS:-a,floor,b,c,d,e,f,c+attn,d+attn,e+attn,f+attn}" || rc=$?
echo "infer exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
