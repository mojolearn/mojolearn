#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The trees family's Apple speed job: the timing command of ONE steward
# speed request (tools/apple_steward.py submit --kind speed), run in the
# steward's checkout at the submitted commit, after its builds.
#
#   TAP_CELLS="gbdt:gbdt-symmetric:taxi rf:taxi xt:adaboost:taxi" sh tools/trees_apple_speed.sh
#
# Cell spellings (one process each; GTP / FTRAIN / TAP lines on stdout):
#   gbdt:<cell>:<dataset>        tools/gbdt_train_probe.py, trees TAP_GBDT_TREES (10,100)
#   gbdtstage:<cell>:<dataset>   the same fit once at 100 trees under MOJOLEARN_STAGE_TIMES=1
#                                (a SPLIT: it drains per stage, never a timing)
#   rf:<dataset> / et:<dataset>  tools/forest_train_ab.py fit, TAP_ROWS rows, TAP_ROUNDS rounds
#   xt:<est>:<dataset>           bench/speed/trees_apple_profile.py (dt, bagging, adaboost,
#                                dart, embedding, iforest)
# Datasets are the board's (taxi, taxireg, istella, istellareg) from
# ~/datasets/gbm-bench, staged from R2 (tools/dataset_store.sh stage).
set -u
ROWS="${TAP_ROWS:-1000000}"
ROUNDS="${TAP_ROUNDS:-2}"
TREES="${TAP_GBDT_TREES:-10,100}"
OUT="${TAP_OUT:-/tmp/tap.$$}"
mkdir -p "$OUT"
PY="pixi run -e default python"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
echo "TAPRUN commit=$(git rev-parse --short HEAD) mode=${MOJOLEARN_NUMERIC_MODE:-unset} rows=$ROWS rounds=$ROUNDS cells=${TAP_CELLS:-}"
for cell in ${TAP_CELLS:-}; do
    kind=${cell%%:*}; rest=${cell#*:}
    echo "=== $cell"
    t0=$(date +%s)
    case "$kind" in
        gbdt)
            c=${rest%%:*}; d=${rest#*:}
            $PY tools/gbdt_train_probe.py fit --cell "$c" --dataset "$d" --rows "$ROWS" \
                --trees "$TREES" --reps "$ROUNDS" --json "$OUT/gbdt_${c}_${d}.json" 2>&1 | grep -E '^GTP|Error|error|Traceback' ;;
        gbdtstage)
            c=${rest%%:*}; d=${rest#*:}
            MOJOLEARN_STAGE_TIMES=1 $PY tools/gbdt_train_probe.py fit --cell "$c" --dataset "$d" --rows "$ROWS" \
                --trees 100 --reps 1 --no-predict 2>&1 | tail -n 120 ;;
        rf|et)
            $PY tools/forest_train_ab.py fit --lane "$kind" --dataset "$rest" --rows "$ROWS" \
                --rounds "$ROUNDS" --label tap --json "$OUT/${kind}_${rest}.json" --score 2>&1 \
                | grep -E '^FTRAIN|Error|error|Traceback' ;;
        xt)
            e=${rest%%:*}; d=${rest#*:}
            $PY bench/speed/trees_apple_profile.py --est "$e" --dataset "$d" --rows "$ROWS" \
                --rounds "$ROUNDS" 2>&1 | grep -E '^TAP|Error|error|Traceback' ;;
        *) echo "unknown cell $cell" ;;
    esac
    echo "=== $cell wall_s=$(( $(date +%s) - t0 ))"
done
