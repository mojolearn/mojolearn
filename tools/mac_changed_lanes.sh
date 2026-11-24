#!/bin/bash
# mac_changed_lanes.sh <outdir>
#
# AFTER A MAC COMPILE (the orchestrator's local compile through ~/mojolearn-evidence/compile_slot.sh, or
# packaging/macos/build_release_wheel.sh), run the changed lanes once, Metal and the CPU column in ONE
# verify process (device="gpu" and device="cpu", 2026-11-24), and print the comparison against the
# committed table (Andrew, 2026-10-13):
#
#   bash tools/mac_changed_lanes.sh ~/mojolearn-evidence/<round>/changed-lanes-mac
#
# The run holds the Mac's Metal lock (tools/mac_slot.py metal), like every other Apple GPU job, for both
# sides. Each column gets its own lane list (tools/identity_columns.py changed-lanes --backend metal|cpu),
# unless LANES is set for both. Output: <outdir>/summary.txt, one line per lane per
# column (tools/changed_lanes_check.sh). Informational: Apple and host digests are recorded, never required.
# Exit: non-zero only when a column could not be checked.
set -uo pipefail
[ $# -eq 1 ] || { echo "usage: mac_changed_lanes.sh <outdir>" >&2; exit 2; }
OUT=$1
case $OUT in /*) ;; *) echo "mac_changed_lanes: <outdir> must be absolute" >&2; exit 2 ;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
[ "$(uname -s)" = Darwin ] || { echo "mac_changed_lanes: Macs only (Metal + the host column)" >&2; exit 2; }
mkdir -p "$OUT" || exit 2
rc=0
python3 tools/mac_slot.py --wait-timeout 3600 metal -- bash tools/changed_lanes_check.sh "$OUT" metal \
    > "$OUT/metal.log" 2>&1 || rc=1
grep -E '^(CHANGED-LANES|NOTHING-CHANGED)|state=DIVERGENT' "$OUT/summary.txt" 2>/dev/null | cut -c1-240 | head -n 40
echo "MAC-CHANGED-LANES rc=$rc summary=$OUT/summary.txt (log $OUT/metal.log: Metal and the CPU column)"
exit $rc
