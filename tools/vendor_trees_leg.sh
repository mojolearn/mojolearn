#!/bin/sh
# OUR TREES AGAINST THE LIBRARY AN NVIDIA USER WOULD ACTUALLY RUN, IN BOTH TIERS.
#
# The existing speed harness answers this for the FAST arm only -- its own
# docstring says so: "How fast is mojolearn's FAST path ... against what an
# NVIDIA user would actually run", and "We expect to lose the GPU columns,
# possibly by a lot. Recording how much is the entire point." What it has
# never been asked is the SECOND half, which is the one the identity claim
# lives or dies on: what does BITWISE REPRODUCIBILITY cost on top of that?
#
# WE ARE NOT BITWISE IDENTICAL TO CATBOOST OR LIGHTGBM AND THIS DOES NOT
# CLAIM TO BE. Our identity claim is that OUR OWN implementation returns the
# same bits on Apple, NVIDIA and AMD. Against a vendor library the comparison
# is SPEED and HELD-OUT ACCURACY and nothing else; no line here gates a hash
# against theirs, and none may ever be quoted as if it did.
#
# THE PAIRING IS THE REPOSITORY'S OWN STANDING RULE, not a choice made here:
#
#   gbdt-symmetric   CatBoost ONLY. Oblivious trees are CatBoost's design and
#                    the lineage this package ports; LightGBM must never
#                    appear beside this lane (standing order 2026-08-22).
#   gbdt-lossguide   LightGBM. Leaf-wise growth IS LightGBM's own algorithm,
#                    which is exactly why it belongs here and not above.
#   gbdt-depthwise   XGBoost and CatBoost under the same grow policy.
#   rf               cuML-RF. cuML has no ExtraTrees, and on the `et` lane it
#                    REFUSES BY NAME rather than going missing.
#
# THREE ARMS, TWO PROCESSES. The FAST process runs ours AND the opponents so
# they are interleaved on one box in one thermal window; the IDENTICAL
# process runs `--ours-only`, because the opponent numbers do not change with
# our build define and paying for them twice would buy nothing but drift.
#
#   MOJOLEARN_VT_LANE     gbdt-symmetric (default) | gbdt-lossguide | rf | et
#   MOJOLEARN_VT_ROWS     training rows (default 1000000, floor 1000000)
#   MOJOLEARN_VT_OUT      results directory
#   MOJOLEARN_VT_SKIP_BUILD  1 -> reuse what is already built
set -u

OUT="${MOJOLEARN_VT_OUT:-bench/results/vendor_trees/$(date +%Y-%m-%d_%H%M%S)}"
LANE="${MOJOLEARN_VT_LANE:-gbdt-symmetric}"
ROWS="${MOJOLEARN_VT_ROWS:-1000000}"
PY="${MOJOLEARN_VT_PYTHON:-pixi run -e gbmbench python}"
# shellcheck disable=SC2086  # $PY is deliberately word-split
py() { $PY "$@"; }
MOJOLEARN_PYTHON="$PY"; export MOJOLEARN_PYTHON

# THE ROW FLOOR IS ANDREW'S STANDING ORDER (2026-09-01) AND IS ENFORCED, NOT
# DOCUMENTED: never measure or decide tree performance below a million rows.
if [ "$ROWS" -lt 1000000 ]; then
    echo "!! MOJOLEARN_VT_ROWS=$ROWS is below the 1,000,000-row tree floor; refusing" >&2
    exit 2
fi

mkdir -p "$OUT/logs"
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host $(hostname)"
  echo "lane $LANE"
  echo "rows $ROWS"
  echo "store ${GBM_BENCH_DATA:-unset}"
  command -v nvidia-smi >/dev/null 2>&1 && echo "device $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

build_fail=0
if [ "${MOJOLEARN_VT_SKIP_BUILD:-0}" != "1" ]; then
    echo "== build FAST =="
    bash bindings/build_estimators.sh > "$OUT/logs/build.estimators.log" 2>&1 \
        || { echo "!! estimators FAST build failed"; tail -15 "$OUT/logs/build.estimators.log"; build_fail=1; }
    bash bindings/build_gbdt.sh > "$OUT/logs/build.gbdt.log" 2>&1 \
        || { echo "!! gbdt FAST build failed"; tail -15 "$OUT/logs/build.gbdt.log"; build_fail=1; }
    bash bindings/build_rf.sh > "$OUT/logs/build.rf.log" 2>&1 \
        || echo "!! rf FAST build failed (only the rf/et lanes need it)"

    echo "== build IDENTICAL =="
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_estimators.sh \
        > "$OUT/logs/build.ident.estimators.log" 2>&1 \
        || { echo "!! estimators IDENTICAL build failed"; build_fail=1; }
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_gbdt.sh \
        > "$OUT/logs/build.ident.gbdt.log" 2>&1 \
        || { echo "!! gbdt IDENTICAL build failed"; tail -15 "$OUT/logs/build.ident.gbdt.log"; build_fail=1; }
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_rf.sh \
        > "$OUT/logs/build.ident.rf.log" 2>&1 \
        || echo "!! rf IDENTICAL build failed (only the rf/et lanes need it)"
fi

echo
echo "== FAST: ours interleaved WITH the opponents, one thermal window =="
MOJOLEARN_NUMERIC_MODE=fast py bench/speed/forest_speed_arm.py \
    --lane "$LANE" --dataset higgs --rows "$ROWS" --devices gpu \
    > "$OUT/logs/fast.log" 2>&1
echo "fast_exit=$?" >> "$OUT/env.txt"
grep -E "^FSPEED" "$OUT/logs/fast.log" | tail -20

echo
echo "== IDENTICAL: ours only, because the opponents do not move with our define =="
MOJOLEARN_NUMERIC_MODE=identical py bench/speed/forest_speed_arm.py \
    --lane "$LANE" --dataset higgs --rows "$ROWS" --devices gpu --ours-only \
    > "$OUT/logs/ident.log" 2>&1
echo "ident_exit=$?" >> "$OUT/env.txt"
grep -E "^FSPEED" "$OUT/logs/ident.log" | tail -10

# --------------------------------------------------------------------------
# The table. Median over rounds per (arm, mode), with the accuracy beside it.
# --------------------------------------------------------------------------
med() {  # med <log> <arm>
    grep "^FSPEED " "$1" | grep "arm=$2 " \
      | awk '{for(i=1;i<=NF;i++) if(index($i,"ms=")==1) print substr($i,4)}' \
      | sort -n | awk '{v[n++]=$1} END{if(n==0)exit;
            if(n%2)printf "%.1f", v[(n-1)/2]; else printf "%.1f",(v[n/2-1]+v[n/2])/2}'
}
acc() {  # acc <log> <arm> <metric>
    grep "^FSPEED-ACC " "$1" | grep "arm=$2 " | grep "metric=$3 " \
      | awk '{for(i=1;i<=NF;i++) if(index($i,"value=")==1) v=substr($i,7)} END{if(v!="")printf "%s", v}'
}
shape() { grep "^FSPEED " "$1" | head -1 \
      | awk '{for(i=1;i<=NF;i++) if(index($i,"shape=")==1) printf "%s", substr($i,7)}'; }

SHAPE=$(shape "$OUT/logs/fast.log")
echo
echo "== VTREES lane=$LANE shape=${SHAPE:-none} =="
printf 'arm\tmode\tmed_ms\tauc\tlogloss\n' > "$OUT/table.tsv"
for spec in "ours:fast:$OUT/logs/fast.log" "ours:identical:$OUT/logs/ident.log"; do
    a=$(echo "$spec" | cut -d: -f1); m=$(echo "$spec" | cut -d: -f2); f=$(echo "$spec" | cut -d: -f3)
    t=$(med "$f" "$a"); [ -z "$t" ] && t="-"
    u=$(acc "$f" "$a" auc); [ -z "$u" ] && u="-"
    l=$(acc "$f" "$a" logloss); [ -z "$l" ] && l="-"
    echo "VTREES arm=$a mode=$m med_ms=$t auc=$u logloss=$l"
    printf '%s\t%s\t%s\t%s\t%s\n' "$a" "$m" "$t" "$u" "$l" >> "$OUT/table.tsv"
done
# Every opponent that produced a round, from the FAST log where they ran.
for a in $(grep "^FSPEED " "$OUT/logs/fast.log" \
            | awk '{for(i=1;i<=NF;i++) if(index($i,"arm=")==1) print substr($i,5)}' \
            | sort -u | grep -v '^ours$'); do
    t=$(med "$OUT/logs/fast.log" "$a"); [ -z "$t" ] && t="-"
    u=$(acc "$OUT/logs/fast.log" "$a" auc); [ -z "$u" ] && u="-"
    l=$(acc "$OUT/logs/fast.log" "$a" logloss); [ -z "$l" ] && l="-"
    echo "VTREES arm=$a mode=vendor med_ms=$t auc=$u logloss=$l"
    printf '%s\t%s\t%s\t%s\t%s\n' "$a" "vendor" "$t" "$u" "$l" >> "$OUT/table.tsv"
done

echo
echo "REFUSALS (an opponent that will not run must be VISIBLE, never absent):"
grep -h "^FSPEED-REFUSED" "$OUT/logs/fast.log" "$OUT/logs/ident.log" 2>/dev/null | sed 's/^/  /' | head -12
echo
echo "results in $OUT"

rc=0
[ "$build_fail" != "0" ] && rc=2
[ -z "$(med "$OUT/logs/fast.log" ours)" ] && { echo "!! our FAST arm produced no rounds"; rc=3; }
case "${SHAPE:-}" in
    higgs-*) ;;
    *) echo "!! NOT HIGGS (shape=${SHAPE:-none}): a synthetic fixture is not an"
       echo "   answer to a vendor comparison, and it is below the row floor"
       rc=4 ;;
esac
[ "$rc" != "0" ] && echo "!! vendor_trees_leg FAILED (rc=$rc)"
exit $rc
