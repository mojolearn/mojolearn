#!/bin/bash
# Alternating A/B of IsolationForest.score_samples between the two FAST svm
# bindings a previous tools/aft_ab.sh run left in its out dir (A.so, B.so),
# one process per round (lane apple-fast-trees-io, the IF_QUERY_RAW arm).
#   bash tools/aft_if_score_ab.sh <aft_ab out dir> <taxi|istella> [pairs]
set -u
out=$1; ds=$2; pairs=${3:-3}
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
py=${AFT_PY:-python3}
[ -z "${AFT_PY:-}" ] && [ -x "$HOME/board-0834/cache/venv/bin/python" ] && py=$HOME/board-0834/cache/venv/bin/python
so=python/mojolearn/_mojolearn_svm.so
export GBM_BENCH_DATA=${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}
echo "AFT-IFSCORE-AB head=$(git rev-parse --short HEAD) out=$out ds=$ds pairs=$pairs"
for f in A B; do [ -f "$out/$f.so" ] || { echo "AFT-IFSCORE-AB missing $out/$f.so"; exit 1; }; done
for i in $(seq 1 "$pairs"); do
    for arm in A B; do
        cp "$out/$arm.so" "$so.aft" && mv -f "$so.aft" "$so"
        AFT_IFSCORE_ROUNDS=1 MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SPEED_EXPECTED_VENDOR=metal \
        PYTHONPATH="$here/python${PYTHONPATH:+:$PYTHONPATH}" \
            "$py" -u tools/aft_if_score.py "$ds" "$arm" > "$out/score_${ds}_${arm}_$i.log" 2>&1
        echo "AFT-IFSCORE-RUN arm=$arm pair=$i rc=$?"
        grep -E '^AFT-IFSCORE ' "$out/score_${ds}_${arm}_$i.log" | sed "s/^/pair=$i /"
        grep -m 3 -iE 'Traceback|Error' "$out/score_${ds}_${arm}_$i.log"
    done
done
