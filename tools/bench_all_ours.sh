#!/bin/sh
# tools/bench_all_ours.sh -- ONE PASS over every classical ML and decision tree
# lane we ship, OUR ARM ONLY, to answer "how long does the whole stack take to
# benchmark, and did anything move".
#
# WHY OURS-ONLY BY DEFAULT. An opponent row is measured ONCE per
# (GPU, driver, library version, dataset) and cached in
# bench/OPPONENT_REFERENCE.md (ENGINEERING_RULES 9). Re-racing CatBoost,
# XGBoost, cuML and scikit-learn on every sweep is most of the cost and none of
# the information. `--opponents` turns them on for the first sweep on a new
# tuple; leave it off afterwards.
#
# THIS SCRIPT DECIDES NOTHING. It does not compute a flip verdict, it does not
# touch a default. It reports times and output hashes. Verdicts stay with
# tools/flip_verdict.py and the section 9 rule, on purpose: a sweep you run
# occasionally is the wrong place to move a default.
#
# PARALLELISM. Never within a pod. Two timed runs sharing one GPU corrupt both
# numbers, which is why every leg here serializes its cells. Use `--shard i/n`
# to split the lane list across n POdS and run those pods concurrently; each
# pod then runs its own shard one lane at a time.
#
#   sh tools/bench_all_ours.sh                               # all lanes, ours, 1M rows
#   sh tools/bench_all_ours.sh --rows full                   # taxi 5.75M, Istella-S 2.04M
#   sh tools/bench_all_ours.sh --lanes rf,et,iforest
#   sh tools/bench_all_ours.sh --datasets taxi,istella,criteo
#   sh tools/bench_all_ours.sh --shard 2/3                   # this pod takes lanes 2,5,8,...
#   sh tools/bench_all_ours.sh --opponents --rounds 5        # first sweep on a new tuple
#
# RUNS ON THE POD, from /root/mojolearn, after the datasets are staged
# (tools/dataset_store.sh stage) and the IDENTICAL set is built. POSIX sh.
set -u

OUT="${MOJOLEARN_BENCH_OUT:-/root/trees_out}"
ROWS=1000000
ROUNDS=3
OPPONENTS=0
SHARD=1; SHARDS=1
#: forest_speed_arm.py --lane, exactly (checked against spec.LANE_NAMES)
TREE_LANES="gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et iforest"
#: classical_two_datasets.py LANES, exactly
CLASSICAL_LANES="kmeans pca ols knn kde svc dbscan hdbscan"
LANES=""
DATASETS="taxi istella"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --rows)      ROWS="$2"; shift 2 ;;
        --rounds)    ROUNDS="$2"; shift 2 ;;
        --lanes)     LANES=$(echo "$2" | tr ',' ' '); shift 2 ;;
        --datasets)  DATASETS=$(echo "$2" | tr ',' ' '); shift 2 ;;
        --opponents) OPPONENTS=1; shift ;;
        --shard)     SHARD=$(echo "$2" | cut -d/ -f1); SHARDS=$(echo "$2" | cut -d/ -f2); shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        -h|--help)   sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; sed -n '2,36p' "$0"; exit 2 ;;
    esac
done

[ -n "$LANES" ] || LANES="$TREE_LANES $CLASSICAL_LANES"
mkdir -p "$OUT/bench_all"
TSV="$OUT/bench_all/sweep.tsv"
LOG="$OUT/bench_all/sweep.log"
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical

note() { echo "$* $(date -u +%T)" | tee -a "$LOG"; }
is_tree_lane() { case " $TREE_LANES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# full size, per dataset, from the loaders' own row counts
rows_for() {
    if [ "$ROWS" != full ]; then echo "$ROWS"; return; fi
    case "$1" in
        taxi)    echo 5750000 ;;   # NYC TLC Jan+Feb 2024, load_taxi
        istella) echo 2043304 ;;   # Istella-S train.txt, load_istella
        criteo)  echo 4000000 ;;   # a slice; the whole day file is ~195M rows
        *)       echo 1000000 ;;
    esac
}

note "sweep start rows=$ROWS rounds=$ROUNDS opponents=$OPPONENTS shard=$SHARD/$SHARDS"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$LOG" 2>&1 || true
[ -s "$TSV" ] || printf 'lane\tdataset\trows\tarm\tseconds\tstatus\n' > "$TSV"

SWEEP_T0=$(date +%s)
i=0
for lane in $LANES; do
    i=$((i+1))
    # round-robin shard so tree and classical lanes spread evenly over pods
    [ $(( (i - 1) % SHARDS + 1 )) -eq "$SHARD" ] || continue
    for ds in $DATASETS; do
        # criteo is the CATEGORICAL set and exists to exercise the CTR path
        # (DEVIATION 2634), which only trees have. classical_two_datasets.py
        # also hard-restricts `race --dataset` to choices=DATASETS, currently
        # ("taxi","istella"), so argparse would REJECT criteo there until that
        # tuple is extended. Skip it rather than send a call that cannot parse.
        if [ "$ds" = criteo ] && ! is_tree_lane "$lane"; then
            note "skip $lane on criteo (classical race takes taxi|istella only)"; continue
        fi
        n=$(rows_for "$ds")
        t0=$(date +%s)
        if is_tree_lane "$lane"; then
            # cmd_speed's 6th arg is a MODE (full|ours|stage), not a free tag:
            # `ours` races our arm alone, `full` brings the opponents in. The
            # tag rides on MOJOLEARN_SPEED_TAG, which only suffixes the log.
            mode=ours; arms=ours
            if [ "$OPPONENTS" = 1 ]; then mode=full; arms="ours,catboost-gpu,xgboost-gpu"; fi
            MOJOLEARN_SPEED_ARMS="$arms" MOJOLEARN_SPEED_TAG=sweep \
                sh tools/trees_identical_ab.sh speed all "$lane" "$ds" "$n" "$ROUNDS" "$mode" \
                >> "$LOG" 2>&1
            rc=$?
        else
            # race REQUIRES --data and --out and takes NO --rows: the lane
            # shapes come from the prepped blocks under --data, so row count is
            # fixed at prep time (MOJOLEARN_CTD_* / `prep --max-rows`), not here.
            arms=ours
            [ "$OPPONENTS" = 1 ] && arms="ours,cuml-gpu,sklearn-cpu"
            python3 tools/classical_two_datasets.py race \
                --lane "$lane" --dataset "$ds" \
                --data "${MOJOLEARN_CTD_DATA:-/root/ctd-data}" \
                --out "$OUT/bench_all/ctd-$lane-$ds" \
                --work "${MOJOLEARN_CTD_WORK:-/root/ctd-work}" \
                --arms "$arms" --rounds "$ROUNDS" \
                >> "$LOG" 2>&1
            rc=$?
            n="prepped"   # the shape is whatever prep built, not a flag here
        fi
        t1=$(date +%s)
        st=ok; [ "$rc" -eq 0 ] || st="FAILED_rc$rc"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$ds" "$n" "$arms" "$((t1-t0))" "$st" >> "$TSV"
        note "  $lane $ds rows=$n $((t1-t0))s $st"
    done
done
SWEEP_T1=$(date +%s)

note "sweep done total=$((SWEEP_T1-SWEEP_T0))s shard=$SHARD/$SHARDS"
echo ""
echo "=== $TSV ==="
cat "$TSV"
echo ""
echo "TOTAL WALL CLOCK for this shard: $((SWEEP_T1-SWEEP_T0))s"
awk -F'\t' 'NR>1 && $6!="ok" {n++} END {if (n) printf "FAILED CELLS: %d (see %s)\n", n, "'"$LOG"'"; else print "all cells ok"}' "$TSV"
