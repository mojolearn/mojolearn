#!/bin/bash
# tools/lowbit_units/gate_job.sh -- lane/lowbit-units, step 0: the existing
# low-bit gate (tools/lowbit_mma_leg.sh: check-gemm-lowbit, the forced-flat
# run and the two sabotage arms) in the lane's OWN tree on a shared box.
# Runs from the tree the queue or the steward starts it in; everything it
# writes lands under bench/results/lowbit_units/<box>/gate/ in that tree.
#
#   sh tools/nvidia_central.sh submit lowbit-units /root/mojolearn-lowbit-units/tools/lowbit_units/gate_job.sh
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
export MOJOLEARN_LOWBIT_LEG_ROOT="$PWD"
export MOJOLEARN_LOWBIT_LEG_OUT="$PWD/bench/results/lowbit_units/$BOX/gate"
rm -rf "$MOJOLEARN_LOWBIT_LEG_OUT"
sh tools/lowbit_mma_leg.sh
rc=$?
cat "$MOJOLEARN_LOWBIT_LEG_OUT/status.tsv" "$MOJOLEARN_LOWBIT_LEG_OUT/gate.txt" 2>/dev/null
for f in lowbit lowbit-force-flat; do
    echo "== $f (tail)"; tail -25 "$MOJOLEARN_LOWBIT_LEG_OUT/$f.log" 2>/dev/null
done
exit $rc
