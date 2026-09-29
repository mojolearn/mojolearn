#!/bin/bash
# F1-pv32, THE CONFIGURATION THAT WOULD SHIP (orchestrator, 2026-09-29):
# 15-bit codes on both operands of every projection, of the head and of
# Q.K^T; P.V in fp32. Two jobs of one GPU each on pod nvc2.
#   part 0: inference on both held-out texts (the same windows and interval
#           as the other arms), then its half of the training runs
#   part 1: the other half of the training runs
# Training: forward only, then forward and backward, five seeds each,
# against the five baseline seeds already recorded in the same directory.
# Sourced by f1_pv32_a.sh and f1_pv32_b.sh, which set the part.
PART=${LOWBIT_QUALITY_PART:?set by f1_pv32_a.sh or f1_pv32_b.sh}
[ "$PART" = 0 ] || export LOWBIT_QUALITY_NEED_MODEL=0
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GITHUB=${LOWBIT_QUALITY_CORPUS2:-/root/mojolearn-wt/lowbit-quality/training/corpus/pile_github/input.txt}
PER=${LOWBIT_QUALITY_WORKERS_PER_GPU:-6}
TOTAL=$((2 * PER))
OUT="$WORK/out/train_int15"
ONLY="F1-pv32:fwd:attn,F1-pv32:fwdbwd:attn"
mkdir -p "$OUT/logs"
rc=0
pids=""
for i in $(seq 0 $((PER - 1))); do
    w=$((PART * PER + i))
    "$PY" bench/lowbit_quality/byte_lm_train.py --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" \
        --seeds 5 --baseline-seeds 5 --zero-code-widths int15 --only "$ONLY" \
        --worker $w --workers $TOTAL > "$OUT/logs/pv32_worker$w.log" 2>&1 &
    pids="$pids $!"
done
if [ "$PART" = 0 ]; then
    [ -f "$GITHUB" ] || { echo "REFUSED: $GITHUB is not staged"; exit 3; }
    "$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$CORPUS" --corpus-key corpus/enwik8/input.txt \
        --out "$WORK/out/f1_pv32_enwik8" --commit "$COMMIT" --arms "a,floor,F1-pv32" || rc=$?
    "$PY" bench/lowbit_quality/infer_eval.py --model "$MODEL" --corpus "$GITHUB" --corpus-key corpus/pile_github/input.txt \
        --tail-from 96000000 --out "$WORK/out/f1_pv32_github" --commit "$COMMIT" --arms "a,floor,F1-pv32" || rc=$?
    echo "F1-pv32 inference exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
for p in $pids; do wait "$p" || rc=$?; done
grep -h "^DONE\|Traceback\|Error" "$OUT"/logs/pv32_worker*.log | tail -60
"$PY" bench/lowbit_quality/train_table.py --runs "$OUT" --out "$OUT" --name "training_f1_pv32_after_part$PART" --only "$ONLY" || true
echo "f1_pv32 part $PART exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
