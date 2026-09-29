#!/bin/bash
# The first job: the quantizer cross-check, this forward against transformers,
# and every primary arm on 8 windows. A smoke, labeled as one in its record.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT="$WORK/out/smoke"
mkdir -p "$OUT"
rc=0
"$PY" bench/lowbit_quality/quantizer_check.py --model "$MODEL" --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" || rc=$?
echo "quantizer_check exit=$rc"
"$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" \
    --validate-hf --limit-windows 8 || rc=$?
echo "smoke exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
