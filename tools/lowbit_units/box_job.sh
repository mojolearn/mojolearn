#!/bin/bash
# tools/lowbit_units/box_job.sh -- lane/lowbit-units, every phase of the
# lane on ONE box, as one steward speed job (one job at a time per lane):
#
#   bash tools/lowbit_units/box_job.sh <box name> <phase> [<phase> ...]
#
#   gate     tools/lowbit_units/gate_job.sh    the existing low-bit gate and its sabotage arms
#   price    tools/lowbit_units/price_job.sh   the timing harness and its sabotage arm
#   chunk    tools/lowbit_units/chunk_job.sh   the Apple exact-chunk gate and its sabotage arms (Apple only)
#   vendor   tools/lowbit_units/vendor_job.sh  the vendor library, COMPARISON ONLY
#
# Every phase runs even when an earlier one failed (a red phase is a
# finding); the exit is non-zero when any phase's was. The box name is what
# the results are filed under (h100, mi325x, m2pro, and a second Apple box by
# changing only this name and the steward target).
set -u
cd "$(dirname "$0")/../.." || exit 9
[ $# -ge 2 ] || { echo "box_job.sh <box> <phase> [<phase> ...]" >&2; exit 2; }
export MOJOLEARN_LOWBIT_BOX=$1
shift
red=0
summary=""
for phase in "$@"; do
    case "$phase" in
        gate|price|chunk|vendor) ;;
        *) echo "box_job.sh: unknown phase $phase" >&2; exit 2 ;;
    esac
    echo "######## phase $phase on $MOJOLEARN_LOWBIT_BOX, started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash "tools/lowbit_units/${phase}_job.sh"
    rc=$?
    echo "######## phase $phase on $MOJOLEARN_LOWBIT_BOX exit=$rc, finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    summary="$summary $phase=$rc"
    [ "$rc" -eq 0 ] || red=1
done
echo "box_job: box=$MOJOLEARN_LOWBIT_BOX phases:$summary red=$red"
exit "$red"
