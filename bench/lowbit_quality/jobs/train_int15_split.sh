#!/bin/bash
# TRAINING UNDER THE 15-BIT PROFILE AS A COMPLETE CONFIGURATION, split over
# two jobs of one GPU each (pod nvc2, lane lowbit-quality-b; orchestrator,
# 2026-09-29 06:15Z). The plan is train_int15.sh's: five baseline seeds, then
# 15-bit codes on every projection and every attention product, forward only
# and forward and backward, five seeds each. The run list is dealt to
# 2 * PER workers; part A (LOWBIT_QUALITY_PART=0) starts the first PER of
# them and part B (=1) the rest, so no run is held by both parts.
# Sourced by train_int15_a.sh and train_int15_b.sh, which set the part.
export LOWBIT_QUALITY_NEED_MODEL=0
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PART=${LOWBIT_QUALITY_PART:?set by train_int15_a.sh or train_int15_b.sh}
PER=${LOWBIT_QUALITY_WORKERS_PER_GPU:-6}
TOTAL=$((2 * PER))
OUT="$WORK/out/train_int15"
ONLY="e:fwd:attn,e:fwdbwd:attn"
mkdir -p "$OUT/logs"
"$PY" bench/lowbit_quality/byte_lm_train.py --corpus "$CORPUS" --out "$OUT/selftest_part$PART" --commit "$COMMIT" \
    --self-test || { echo "self-test FAILED: no run is started"; exit 4; }
pids=""
for i in $(seq 0 $((PER - 1))); do
    w=$((PART * PER + i))
    "$PY" bench/lowbit_quality/byte_lm_train.py --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" \
        --seeds 5 --baseline-seeds 5 --zero-code-widths int15 --only "$ONLY" \
        --worker $w --workers $TOTAL > "$OUT/logs/worker$w.log" 2>&1 &
    pids="$pids $!"
done
rc=0
for p in $pids; do wait "$p" || rc=$?; done
grep -h "^DONE\|Traceback\|Error" "$OUT"/logs/worker*.log | tail -60
"$PY" bench/lowbit_quality/train_table.py --runs "$OUT" --out "$OUT" --name "training_int15_after_part$PART" --only "$ONLY" || true
echo "train_int15 part $PART exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
