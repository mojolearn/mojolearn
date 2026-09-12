#!/bin/sh
# tools/bench_all_ours.sh -- ONE PASS over every classical ML and decision tree
# lane we ship, on ONE POD, to produce an INTERNALLY CONSISTENT board: one
# commit, one GPU, one driver, one set of library versions, arms interleaved
# within each cell.
#
# WHY CONSISTENCY IS THE PRODUCT. Before 2026-09-12 the board was a patchwork of
# rows from different pods, dates, commits and drivers, and that patchwork
# produced a WRONG HEADLINE: the published CatBoost symmetric-taxi row (709.0 ms,
# OPPONENT_REFERENCE.md:2109) turned out to be a slow sample against 623.8 ms
# re-measured, so a "0.437x" claim was really 0.525x. Opponent columns drift
# about 10% pod to pod. A cross-lane sentence ("KDE is our worst gap") is only
# meaningful if every cell in it came off the same machine in one heat window.
#
# THIS SCRIPT DECIDES NOTHING. It does not compute a flip verdict and it does
# not touch a default. It reports times, quality and hashes. Verdicts stay with
# tools/flip_verdict.py and the ENGINEERING_RULES.md section 9 rule, on purpose:
# a sweep you run occasionally is the wrong place to move a default.
#
# PARALLELISM. Never within a pod: two timed runs sharing one GPU corrupt both
# numbers, so every cell here serializes. `--shard i/n` splits lanes across
# POdS and is REFUSED BY DEFAULT (see --shard below), because sharding a
# consistency sweep reintroduces the exact cross-machine patchwork it exists to
# remove.
#
#   sh tools/bench_all_ours.sh --rows full --rounds 5 --opponents   # THE BOARD
#   sh tools/bench_all_ours.sh --lanes rf,et,iforest
#   sh tools/bench_all_ours.sh --datasets taxi,istella,criteo
#
# RUNS ON THE POD, from /root/mojolearn, after the datasets are staged
# (tools/dataset_store.sh stage) and the IDENTICAL set is built. POSIX sh.
set -u

OUT="${MOJOLEARN_BENCH_OUT:-/root/trees_out}"
ROWS=1000000
ROUNDS=3
OPPONENTS=0
SHARD=1; SHARDS=1
PREP=1
#: forest_speed_arm.py --lane, exactly (checked against spec.LANE_NAMES)
TREE_LANES="gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et iforest"
#: classical_two_datasets.py LANES, exactly
CLASSICAL_LANES="kmeans pca ols knn kde svc dbscan hdbscan"
LANES=""
DATASETS="taxi istella"

# DBSCAN's eps and min_samples, PER DATASET, committed here rather than
# exported by hand (FIXED 2026-09-12). `_dbscan_params` RAISES when the
# variable is unset -- there is no default -- so every dbscan cell of every
# previous sweep died at `ready` unless the operator happened to export these.
# The values are the ones the published rows were measured at
# (OPPONENT_REFERENCE.md:1902-1908, lane linear-cluster-istella): eps is the
# p75 quantile of the 10th-nearest-neighbour distance on the standardized
# block, min_samples 10. They lived ONLY in a pod shell, which is precisely the
# kind of unversioned input this sweep exists to eliminate.
: "${MOJOLEARN_CTD_DBSCAN_TAXI:=0.177,10}"
: "${MOJOLEARN_CTD_DBSCAN_ISTELLA:=4.17,10}"
export MOJOLEARN_CTD_DBSCAN_TAXI MOJOLEARN_CTD_DBSCAN_ISTELLA

while [ "$#" -gt 0 ]; do
    case "$1" in
        --rows)      ROWS="$2"; shift 2 ;;
        --rounds)    ROUNDS="$2"; shift 2 ;;
        --lanes)     LANES=$(echo "$2" | tr ',' ' '); shift 2 ;;
        --datasets)  DATASETS=$(echo "$2" | tr ',' ' '); shift 2 ;;
        --opponents) OPPONENTS=1; shift ;;
        --no-prep)   PREP=0; shift ;;
        --shard)     SHARD=$(echo "$2" | cut -d/ -f1); SHARDS=$(echo "$2" | cut -d/ -f2); shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        -h|--help)   sed -n '2,34p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; sed -n '2,34p' "$0"; exit 2 ;;
    esac
done

# A SHARDED CONSISTENCY SWEEP IS A CONTRADICTION. Splitting lanes across pods
# puts "our KDE" and "our kNN" on two machines whose opponent columns differ by
# about 10%, so no cross-lane sentence drawn from the board is safe. Refused
# rather than clamped, so the operator has to mean it.
if [ "$SHARDS" != 1 ] && [ "${MOJOLEARN_SWEEP_ALLOW_SHARD:-0}" != 1 ]; then
    echo "REFUSING --shard $SHARD/$SHARDS: this sweep's product is a board whose" >&2
    echo "  cells share one pod, one driver and one heat window. Sharding puts" >&2
    echo "  lanes on different machines and reintroduces the patchwork that" >&2
    echo "  produced the wrong CatBoost headline (OPPONENT_REFERENCE.md:2109)." >&2
    echo "  Set MOJOLEARN_SWEEP_ALLOW_SHARD=1 if you truly want per-lane pods." >&2
    exit 2
fi

[ -n "$LANES" ] || LANES="$TREE_LANES $CLASSICAL_LANES"
mkdir -p "$OUT/bench_all" "$OUT/bench_all/ctd"
TSV="$OUT/bench_all/sweep.tsv"
LOG="$OUT/bench_all/sweep.log"
CTD_OUT="$OUT/bench_all/ctd"
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical
# The loaders read GBM_BENCH_DATA; dataset_store.sh stages under /root/datasets.
export GBM_BENCH_DATA="${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"
# A thread cap inherited from a shell would silently throttle the scikit-learn
# arms and nothing in the table would say so.
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
# Drivers below 580 need the CUDA ptxas rather than the bundled one, or our
# kernels fail to compile at run time (lane linear-cluster-istella).
_drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ "${_drv:-999}" -lt 580 ] 2>/dev/null && [ -x /usr/local/cuda/bin/ptxas ]; then
    export MODULAR_NVPTX_COMPILER_PATH="${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}"
fi

note() { echo "$* $(date -u +%T)" | tee -a "$LOG"; }
is_tree_lane() { case " $TREE_LANES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# THE OPPONENT ROSTER IS PER LANE, and it is not negotiable by a global list.
# BROKEN UNTIL 2026-09-12: `--opponents` sent "catboost-gpu,xgboost-gpu" to
# EVERY tree lane. Neither name is in the rf, et or iforest roster
# (speed_gbdt_arm.opponent_builders), so all three lanes refused every opponent
# BY NAME, the opponent list went EMPTY, and the cell ran ours-only while the
# TSV still recorded three arms. A silently ours-only cell that reads as a race
# is exactly the "asserted green" failure this sweep exists to remove.
tree_arms_for() {
    case "$1" in
        gbdt-symmetric) echo "catboost-gpu" ;;              # CatBoost only (DEVIATION 1831)
        gbdt-depthwise) echo "catboost-gpu,xgboost-gpu" ;;
        gbdt-lossguide) echo "catboost-gpu,xgboost-gpu,lightgbm-cuda" ;;
        rf)             echo "cuml-rf-gpu" ;;
        et)             echo "sklearn-et-cpu" ;;            # cuML ships no ExtraTrees
        iforest)        echo "cuml-iforest-gpu" ;;
        *)              echo "" ;;
    esac
}

# `et` is the ONE lane where an NVIDIA box admits a CPU arm by name: no GPU
# ExtraTrees exists anywhere, so sklearn on all cores is its only legal
# opponent (resolve_devices, ENGINEERING_RULES.md section 10). Every other lane
# stays GPU-only. The arm name carries -cpu so the table cannot hide it.
tree_devices_for() {
    case "$1" in
        et) echo "cpu,gpu" ;;
        *)  echo "gpu" ;;
    esac
}

# Classical rosters. `ours` is always first; the GPU library on the box is the
# opponent (DEVIATION 2571). hdbscan has NO `ours` arm -- this library ships no
# HDBSCAN -- and dbscan has no scikit-learn arm, so a single global list makes
# `race` exit by name on both. BROKEN UNTIL 2026-09-12.
classical_arms_for() {
    _l="$1"
    if [ "$OPPONENTS" = 0 ]; then
        case "$_l" in hdbscan) echo "" ;; *) echo "ours" ;; esac
        return
    fi
    case "$_l" in
        kmeans|pca|knn) echo "ours,cuml-gpu,torch-gpu" ;;
        ols)            echo "ours,cuml-gpu,torch-gpu,torch-gpu-eigh" ;;
        kde|svc)        echo "ours,cuml-gpu" ;;             # no torch arm exists
        dbscan)         echo "ours,cuml-gpu" ;;             # cuml-gpu-rbc overflows at 1M
        hdbscan)        echo "cuml-gpu" ;;                  # opponent-only, by construction
        *)              echo "ours" ;;
    esac
}

# A DBSCAN round on Istella-S takes about 337 s and the harness default is 300,
# which killed the first published attempt at round 1 (lane README 197-201).
classical_round_seconds_for() {
    case "$1-$2" in
        dbscan-istella) echo 1800 ;;
        dbscan-*)       echo 900 ;;
        *)              echo 300 ;;
    esac
}

# full size, per dataset, from the loaders' own row counts.
# NOTE THE CAP DOES NOT BIND ON TAXI and the TSV must not pretend it does: the
# taxi npz holds 5,750,086 rows, so `taxireg` has 5,250,086 train rows and
# `taxi` (card-filtered for the tip task) has 4,110,786. 5,750,000 is therefore
# a no-op ceiling on both. The REAL shape is the `shape=` tag on every FSPEED
# line, which is what tools/bench_all_summarize.py reports.
rows_for() {
    if [ "$ROWS" != full ]; then echo "$ROWS"; return; fi
    case "$1" in
        taxi)    echo 5750000 ;;   # no-op ceiling; real train rows 4,110,786
        istella) echo 2043304 ;;   # exact: the whole cached train split
        criteo)  echo 4000000 ;;
        *)       echo 1000000 ;;
    esac
}

note "sweep start rows=$ROWS rounds=$ROUNDS opponents=$OPPONENTS prep=$PREP"
note "commit $(cat /root/mojolearn/SHIPPED_COMMIT.txt 2>/dev/null || git -C /root/mojolearn rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$LOG" 2>&1 || true
[ -s "$TSV" ] || printf 'lane\tdataset\trows_requested\tarm\tseconds\tstatus\n' > "$TSV"

# ---------------------------------------------------------------------------
# PREP. The classical racer reads prebuilt blocks from --data and opens
# <block>-<dataset>.json BEFORE anything else; with no blocks every classical
# cell dies with FileNotFoundError. bench_all_ours.sh never called prep, so the
# classical half of this sweep could never have run. Untimed, once per box.
# ---------------------------------------------------------------------------
CTD_DATA="${MOJOLEARN_CTD_DATA:-/root/ctd-data}"
if [ "$PREP" = 1 ]; then
    _cl=""
    for lane in $LANES; do
        is_tree_lane "$lane" || _cl="$_cl,$lane"
    done
    _cl=$(echo "$_cl" | sed 's/^,//')
    if [ -n "$_cl" ]; then
        _pds=$(echo "$DATASETS" | tr ' ' ',' | sed 's/,*criteo,*/,/g; s/^,//; s/,$//')
        note "prep classical blocks lanes=$_cl datasets=$_pds -> $CTD_DATA"
        t0=$(date +%s)
        python3 tools/classical_two_datasets.py prep \
            --data "$CTD_DATA" --lanes "$_cl" --datasets "$_pds" >> "$LOG" 2>&1
        note "  prep rc=$? $(( $(date +%s) - t0 ))s"
    fi
fi

SWEEP_T0=$(date +%s)
for lane in $LANES; do
    for ds in $DATASETS; do
        # criteo is the CATEGORICAL set and exists to exercise the CTR path
        # (DEVIATION 2634), which only trees have. classical_two_datasets.py
        # hard-restricts `race --dataset` to ("taxi","istella"), so argparse
        # would REJECT criteo there. Skip rather than send a call that cannot parse.
        if [ "$ds" = criteo ] && ! is_tree_lane "$lane"; then
            note "skip $lane on criteo (classical race takes taxi|istella only)"; continue
        fi
        n=$(rows_for "$ds")
        t0=$(date +%s)
        if is_tree_lane "$lane"; then
            mode=ours; arms=ours
            if [ "$OPPONENTS" = 1 ]; then
                _opp=$(tree_arms_for "$lane")
                if [ -n "$_opp" ]; then mode=full; arms="ours,$_opp"; fi
            fi
            MOJOLEARN_SPEED_ARMS="$arms" MOJOLEARN_SPEED_TAG=sweep \
            MOJOLEARN_SPEED_DEVICES=$(tree_devices_for "$lane") \
                sh tools/trees_identical_ab.sh speed all "$lane" "$ds" "$n" "$ROUNDS" "$mode" \
                >> "$LOG" 2>&1
            rc=$?
        else
            arms=$(classical_arms_for "$lane")
            if [ -z "$arms" ]; then
                note "  skip $lane $ds (no arm: this library ships no $lane, and --opponents is off)"
                continue
            fi
            # ONE output directory, so `summary` and the board harvester can
            # see every cell; the old per-cell directory made aggregation
            # impossible.
            python3 tools/classical_two_datasets.py race \
                --lane "$lane" --dataset "$ds" \
                --data "$CTD_DATA" --out "$CTD_OUT" \
                --work "${MOJOLEARN_CTD_WORK:-/root/ctd-work}" \
                --arms "$arms" --rounds "$ROUNDS" \
                --round-seconds "$(classical_round_seconds_for "$lane" "$ds")" \
                --ours-python "${MOJOLEARN_CTD_OURS_PY:-python3}" \
                --theirs-python "${MOJOLEARN_CTD_THEIRS_PY:-python3}" \
                >> "$LOG" 2>&1
            rc=$?
            # A worker that outlived its conductor holds GPU memory and would
            # contend with the next cell's timing. Every real classical leg
            # does this between lanes.
            pkill -9 -f 'classical_two_datasets.py worker' > /dev/null 2>&1
            n="prepped"   # the shape is whatever prep built, not a flag here
        fi
        t1=$(date +%s)
        st=ok; [ "$rc" -eq 0 ] || st="FAILED_rc$rc"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$ds" "$n" "$arms" "$((t1-t0))" "$st" >> "$TSV"
        note "  $lane $ds rows=$n $((t1-t0))s $st"
    done
done
SWEEP_T1=$(date +%s)

note "sweep done total=$((SWEEP_T1-SWEEP_T0))s"
echo ""
echo "=== $TSV (WALL SECONDS PER CELL, not the measurement) ==="
cat "$TSV"
echo ""
echo "TOTAL WALL CLOCK: $((SWEEP_T1-SWEEP_T0))s"
awk -F'\t' 'NR>1 && $6!="ok" {n++} END {if (n) printf "FAILED CELLS: %d (see %s)\n", n, "'"$LOG"'"; else print "all cells ok"}' "$TSV"
echo ""
echo "=== THE BOARD (the measurement; UNKNOWN is not a pass) ==="
python3 tools/bench_all_summarize.py --out "$OUT/bench_all" --rounds "$ROUNDS" \
    --lanes "$(echo "$LANES" | tr ' ' ',')" \
    --datasets "$(echo "$DATASETS" | tr ' ' ',')" \
    --tsv "$OUT/bench_all/board.tsv" --md "$OUT/bench_all/board.md"
