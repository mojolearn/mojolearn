#!/bin/sh
# Run tools/fast_replication_ab.sh once per candidate define, SEQUENTIALLY.
#
# WHY ONE ARM PER RUN. DEVIATION 2040 was measured inert on an MI325X --
# gbdt 1.001, bits equal, and the two binaries verifiably different so reach
# was not in doubt. That null result is worth something ONLY because 2040 was
# the sole thing changed. Combine two defines and a null tells you nothing
# about either, and a win tells you nothing about which.
#
# THE QUESTION ALL THESE ARMS SERVE: identity is not supposed to be free, and
# on the gbdt lane IDENTICAL BEATS FAST at every fixture on both devices
# (0.731 on a 304-CU MI325X, 0.938 on a 132-SM H100) while ALSO paying a
# software Cephes expf and a barrier-per-step reduction. So the FAST arm is
# what is under suspicion, and these are the two candidates still standing
# after 2040 was refuted.
set -u
DEFINES="${MOJOLEARN_AB_DEFINES:-MOJOLEARN_2041_FAST_CHUNKS_PIN MOJOLEARN_2042_FAST_NO_LOOKBACK}"
BASE="${MOJOLEARN_AB_OUT_BASE:-bench/results/fast_replication_ab/$(date +%Y-%m-%d_%H%M%S)}"
rc_all=0
for d in $DEFINES; do
    echo "===== A/B ARM $d ====="
    MOJOLEARN_AB_DEFINE="$d" MOJOLEARN_AB_OUT="$BASE/$d" sh tools/fast_replication_ab.sh
    rc=$?
    [ "$rc" != 0 ] && { echo "ARM $d FAILED rc=$rc"; rc_all=$rc; }
done
echo "===== ALL ARMS ====="
for d in $DEFINES; do
    [ -f "$BASE/$d/ab.tsv" ] || { echo "$d: no table"; continue; }
    awk -v a="$d" 'NR>1 {printf "%s\t%s\tratio %s\t%s\n", a, $1, $4, $7}' "$BASE/$d/ab.tsv"
done
echo
echo "ratio = PIN / BASE. Below 1.0 means the arm made FAST faster."
echo "A ratio is only a speed result on a row whose two hashes are EQUAL."
exit "$rc_all"
