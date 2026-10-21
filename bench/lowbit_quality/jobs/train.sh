#!/bin/bash
# The training table: the self-test of the custom products, then every run of
# the plan, LOWBIT_QUALITY_WORKERS_PER_GPU processes on each GPU the queue
# handed this job. A run whose record exists is not repeated.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT="$WORK/out/${LOWBIT_QUALITY_TRAIN_DIR:-train}"
mkdir -p "$OUT/logs"
PER=${LOWBIT_QUALITY_WORKERS_PER_GPU:-6}
GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' ' ')
N=0; for g in $GPUS; do N=$((N + PER)); done
CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d' ' -f1) "$PY" bench/lowbit_quality/byte_lm_train.py \
    --corpus "$CORPUS" --out "$OUT" --commit "$COMMIT" --self-test || { echo "self-test FAILED: no run is started"; exit 4; }
w=0; pids=""
for g in $GPUS; do
    for i in $(seq 1 "$PER"); do
        CUDA_VISIBLE_DEVICES=$g "$PY" bench/lowbit_quality/byte_lm_train.py --corpus "$CORPUS" --out "$OUT" \
            --commit "$COMMIT" --worker $w --workers $N ${LOWBIT_QUALITY_TRAIN_ARGS:-} > "$OUT/logs/worker$w.log" 2>&1 &
        pids="$pids $!"
        w=$((w + 1))
    done
done
rc=0
for p in $pids; do wait "$p" || rc=$?; done
grep -h "^DONE\|Traceback\|Error" "$OUT"/logs/worker*.log | tail -200
"$PY" bench/lowbit_quality/train_table.py --runs "$OUT" --out "$OUT" || rc=$?
echo "train exit=$rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
