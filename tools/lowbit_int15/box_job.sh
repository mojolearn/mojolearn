#!/bin/bash
# tools/lowbit_int15/box_job.sh -- lane/lowbit-int15, the lane's phases on
# ONE box as one job (one job per lane at a time on a shared box):
#
#   bash tools/lowbit_int15/box_job.sh <box name> <phase> [<phase> ...]
#
#   gate    tools/lowbit_int15/gate_job.sh    the fifteen-bit gate and its three arms
#   sim     tools/lowbit_int15/sim_job.sh     the cross-check against the simulation and its three arms
#   price   tools/lowbit_int15/price_job.sh   the timing, AFTER the builds are warm; it must run alone
#   int8_gate tools/lowbit_int15/int8_gate_job.sh  the EXISTING low-bit gate (the int8 profile runs through the fragment loads this lane edited)
#   harness tools/lowbit_int15/harness_job.sh the repo's verification harness on the lane gemm-int15
#
# Every phase runs even when an earlier one failed (a red phase is a
# finding); the exit is non-zero when any phase's was. The box name is what
# the results are filed under (h100, mi325x, m3ultra, m2pro).
set -u
cd "$(dirname "$0")/../.." || exit 9
[ $# -ge 2 ] || { echo "box_job.sh <box> <phase> [<phase> ...]" >&2; exit 2; }
export MOJOLEARN_LOWBIT_BOX=$1
shift
red=0
summary=""
for phase in "$@"; do
    case "$phase" in
        gate|sim|price|int8_gate|harness) ;;
        *) echo "box_job.sh: unknown phase $phase" >&2; exit 2 ;;
    esac
    echo "######## phase $phase on $MOJOLEARN_LOWBIT_BOX, started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash "tools/lowbit_int15/${phase}_job.sh"
    rc=$?
    echo "######## phase $phase on $MOJOLEARN_LOWBIT_BOX exit=$rc, finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    summary="$summary $phase=$rc"
    [ "$rc" -eq 0 ] || red=1
done
echo "box_job: box=$MOJOLEARN_LOWBIT_BOX phases:$summary red=$red"
exit "$red"
