#!/bin/sh
# THREE FIXTURE SIZES PER LANE, one price run each.
#
# WHY A SWEEP AND NOT MORE ROUNDS. The first two priced columns returned
# results one fixture cannot explain: `nt` read 0.69x on an MI325X and 1.31x
# on an H100 -- opposite signs -- and on AMD the two modes returned
# BIT-IDENTICAL output while taking different times, which is the fast path
# walking a slower route to the same answer rather than identity being free.
# `kmeans` on the H100 did the same at 1.04x.
#
# A ratio is either a property of the CONTRACT or of the SHAPE. If identity's
# cost grows with the fixture, the small readings were launch overhead. If the
# sign holds across three size points, it is real and the FAST path is the
# thing to look at. More rounds cannot tell those apart; more sizes can.
#
# THE TREE LANES KEEP THE 1M-ROW FLOOR at their smallest point. A tree
# measured below 1M rows is not admissible in this repository at all.
#
#   MOJOLEARN_SWEEP_OUT=<dir> sh tools/lanes_price_sweep.sh
#
# Sizes are the harness's own documented steps (bench/lanes_price_main.mojo's
# knob table): column 1 is its default, column 2 its "datacenter step".
set -u
OUT="${MOJOLEARN_SWEEP_OUT:-bench/results/lanes_price/sweep_$(date +%Y-%m-%d_%H%M%S)}"
ROUNDS="${MOJOLEARN_SWEEP_ROUNDS:-3}"
LANES="${MOJOLEARN_SWEEP_LANES:-kmeans knn dbscan gram nt gemv gbdt rf et}"

rc_all=0
for tier in S M L; do
    case "$tier" in
        S) K=100000;  N=20000;  D=20000;  G=1000000; T=4096;   V=128;  B=1000000; R=1000000; E=1000000 ;;
        M) K=500000;  N=100000; D=50000;  G=4000000; T=65536;  V=2048; B=1500000; R=1500000; E=1500000 ;;
        L) K=2000000; N=400000; D=100000; G=8000000; T=262144; V=8192; B=2000000; R=2000000; E=2000000 ;;
    esac
    # PER-TIER LANE LIST. `lanes_price.sh` ABORTS the whole run when a lane
    # raises, and that is correct -- a hash that moves under IDENTICAL is a
    # FINDING and must stop the run rather than be averaged into a table.
    # But it also means one lane's finding costs every lane after it: on
    # 2026-09-03 dbscan raised at 100,000 rows and took gram, nt, gemv and
    # all three tree lanes down with it, so the datacenter step has no
    # numbers at all. dbscan is therefore excluded from tier L BY NAME, with
    # its defect recorded in bench/results/lanes_price/SWEEP_2026-09-03/,
    # rather than left in to re-discover the same thing at the cost of six
    # other lanes. Put it back the moment the gfx942 defect is fixed.
    TIER_LANES="$LANES"
    if [ "$tier" = "L" ]; then
        TIER_LANES="$(printf '%s\n' $LANES | grep -v '^dbscan$' | tr '\n' ' ')"
    fi
    echo "===== SWEEP TIER $tier ($TIER_LANES) ====="
    MOJOLEARN_LANES_PRICE_OUT="$OUT/$tier" \
    MOJOLEARN_LANES_PRICE_LANES="$TIER_LANES" \
    MOJOLEARN_LANES_PRICE_ROUNDS="$ROUNDS" \
    MOJOLEARN_LANES_PRICE_KMEANS_ROWS="$K" \
    MOJOLEARN_LANES_PRICE_KNN_INDEX="$N" \
    MOJOLEARN_LANES_PRICE_DBSCAN_ROWS="$D" \
    MOJOLEARN_LANES_PRICE_GRAM_ROWS="$G" \
    MOJOLEARN_LANES_PRICE_NT_ROWS="$T" \
    MOJOLEARN_LANES_PRICE_GEMV_DIM="$V" \
    MOJOLEARN_LANES_PRICE_GBDT_ROWS="$B" \
    MOJOLEARN_LANES_PRICE_RF_ROWS="$R" \
    MOJOLEARN_LANES_PRICE_ET_ROWS="$E" \
    sh tools/lanes_price.sh
    rc=$?
    [ "$rc" != 0 ] && { echo "SWEEP TIER $tier FAILED rc=$rc"; rc_all=$rc; }
done

echo "===== SWEEP SUMMARY ====="
for tier in S M L; do
    [ -f "$OUT/$tier/ratio.tsv" ] || { echo "$tier: no ratio.tsv"; continue; }
    awk -v t="$tier" 'NR>1 {printf "%s\t%s\t%s\t%s\t%s\n", t, $1, $5, $6"-"$7, $14}' "$OUT/$tier/ratio.tsv"
done
exit "$rc_all"
