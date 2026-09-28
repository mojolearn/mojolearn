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
#   xtstage:<est>:<dataset>:<n>  the same fit with n estimators under MOJOLEARN_STAGE_TIMES=1
#   rfstage:<lane>:<dataset>     a forest fit under MOJOLEARN_STAGE_TIMES=1 (no launch clock)
# Datasets are the board's (taxi, taxireg, istella, istellareg) from
# ~/datasets/gbm-bench, staged from R2 (tools/dataset_store.sh stage).
set -u
ROWS="${TAP_ROWS:-1000000}"
ROUNDS="${TAP_ROUNDS:-2}"
TREES="${TAP_GBDT_TREES:-10,100}"
OUT="${TAP_OUT:-/tmp/tap.$$}"
mkdir -p "$OUT"
PY="pixi run -e default python"
# TAP_PROFILE: the xtrees timing script, for a checkout that predates it
# (a copy from a later commit; it finds the checkout through TAP_REPO).
PROFILE="${TAP_PROFILE:-bench/speed/trees_apple_profile.py}"
export TAP_REPO="$PWD"
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
            $PY "$PROFILE" --est "$e" --dataset "$d" --rows "$ROWS" \
                --rounds "$ROUNDS" 2>&1 | grep -E '^TAP|Error|error|Traceback' ;;
        xtprof)
            # xtprof:<est>:<dataset>:<n_estimators>  cProfile of one fit
            e=${rest%%:*}; r2=${rest#*:}; d=${r2%%:*}; n=${r2#*:}
            $PY "$PROFILE" --est "$e" --dataset "$d" --rows "$ROWS" \
                --rounds 1 --n-estimators "$n" --profile 2>&1 | grep -v -E '^\s*$' | head -n 120 ;;
        xtstage)
            # xtstage:<est>:<dataset>:<n_estimators>  one fit under MOJOLEARN_STAGE_TIMES=1
            # without the launch clock (a SPLIT: stage ends drain, never a timing)
            e=${rest%%:*}; r2=${rest#*:}; d=${r2%%:*}; n=${r2#*:}
            MOJOLEARN_STAGE_TIMES=1 $PY "$PROFILE" --est "$e" --dataset "$d" --rows "$ROWS" \
                --rounds 1 --n-estimators "$n" 2>&1 | grep -v -E '^\s*$' | tail -n "${TAP_STAGE_TAIL:-150}" ;;
        rfstage)
            # rfstage:<lane>:<dataset>  one forest fit under MOJOLEARN_STAGE_TIMES=1, no launch clock
            l=${rest%%:*}; d=${rest#*:}
            MOJOLEARN_STAGE_TIMES=1 $PY tools/forest_train_ab.py fit --lane "$l" --dataset "$d" --rows "$ROWS" \
                --rounds 1 --label stage --json "$OUT/rfstage_${l}_${d}.json" 2>&1 | tail -n 80 ;;
        rfclock)
            # rfclock:<lane>:<dataset>  RF_LAUNCH_LOG launch clock of one fit (a SPLIT)
            l=${rest%%:*}; d=${rest#*:}
            RF_LAUNCH_LOG="$OUT/rfclock_${l}_${d}.log" RF_LAUNCH_CLOCK=1 MOJOLEARN_STAGE_TIMES=1 \
                $PY tools/forest_train_ab.py fit --lane "$l" --dataset "$d" --rows "$ROWS" \
                --rounds 1 --label clock --json "$OUT/rfclock_${l}_${d}.json" 2>&1 | tail -n 80
            test -f "$OUT/rfclock_${l}_${d}.log" && awk -F'\t' 'NF==2{s[$1]+=$2; c[$1]++; t+=$2} END{for(k in s) printf "RFCLOCK %-48s n=%d ms=%.1f\n", k, c[k], s[k]/1e6; printf "RFCLOCK TOTAL ms=%.1f\n", t/1e6}' "$OUT/rfclock_${l}_${d}.log" | sort -t= -k3 -rn | head -n 60 ;;
        *) echo "unknown cell $cell" ;;
    esac
    echo "=== $cell wall_s=$(( $(date +%s) - t0 ))"
done
