#!/bin/bash
# changed_lanes_check.sh <outdir> <column>
#
# THE CHANGED LANES, CHECKED WHERE THE BINDINGS WERE JUST BUILT (Andrew, 2026-10-13: fold light
# verification of what changed into compiling). Runs in a BUILT tree (the bindings of <column> built in
# python/mojolearn/), once: `python -m mojolearn verify --lanes <changed lanes> --json` on <column>, the
# default fixture, one run, against the table committed in this tree, then one line per lane
# (tools/changed_lanes_summary.py): VERIFIED, DIVERGENT with the first differing cell's hashes, or
# NO-REFERENCE. On the Mac, tools/mac_changed_lanes.sh runs it for Metal and the CPU column after a
# compile. Apple and host digests are recorded, never required (identity is NVIDIA == AMD; those
# columns are recorded through lq, tools/record_changed_lanes.sh). A changed lane reads DIVERGENT
# against the old table whenever its bits moved; that is information, never a reason to rerun.
#
#   <outdir>   absolute; created. Writes summary.txt (appended), verify-<column>.json and .log,
#              lanes-<column>.txt
#   <column>   cuda | hip | metal | cpu   (cpu sets MOJOLEARN_VENDOR=cpu: the host column)
#
# Environment:
#   LANES=a,b        the changed lanes. Unset: computed here, `tools/identity_columns.py changed-lanes
#                    --scope routine --backend <column>` (needs git history; honors
#                    MOJOLEARN_RELEASE_BUILD_ONLY through tools/lane_select.py). Empty or `none`:
#                    prints NOTHING-CHANGED and exits 0
#   PYTHON=...       the interpreter (default: the tree's .pixi default env python, else python3)
#   CPU_THREADS=3    verify --cpu-threads
#
# Exit: 0 when every step ran (whatever the verdicts say; the summary carries them), 1 when a step could
# not run (no lane list, no report). Never compiles, never rents, never reruns: identity runs ONCE.
set -uo pipefail

die() { echo "changed_lanes_check: $*" >&2; exit 2; }
[ $# -eq 2 ] || die "usage: changed_lanes_check.sh <outdir> <cuda|hip|metal|cpu>"
OUT=$1; COL=$2
case $COL in cuda|hip|metal|cpu) ;; *) die "column $COL: one of cuda, hip, metal, cpu" ;; esac
case $OUT in /*) ;; *) die "<outdir> must be absolute" ;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || die "cannot enter $ROOT"
mkdir -p "$OUT" || die "cannot create $OUT"
if [ -n "${PYTHON:-}" ]; then
    read -r -a PY <<< "$PYTHON"
elif [ -x "$ROOT/.pixi/envs/default/bin/python" ]; then
    PY=("$ROOT/.pixi/envs/default/bin/python")
else
    PY=(python3)
fi
T=${CPU_THREADS:-3}
SUMMARY=$OUT/summary.txt

# --- which lanes ---------------------------------------------------------------
if [ -z "${LANES+x}" ]; then
    LANES=$("${PY[@]}" tools/identity_columns.py changed-lanes --scope routine --backend "$COL" \
            2> "$OUT/lanes-$COL.err" | tail -n 1) || { echo "CHANGED-LANES column=$COL LANE-LIST-FAILED (see $OUT/lanes-$COL.err)" | tee -a "$SUMMARY"; exit 1; }
    grep -m 1 'CHANGED-LANES' "$OUT/lanes-$COL.err"
fi
[ "$LANES" = none ] && LANES=
printf '%s\n' "$LANES" > "$OUT/lanes-$COL.txt"
if [ -z "$LANES" ]; then
    echo "NOTHING-CHANGED column=$COL: no lane's bits can have moved since the admitted table" | tee -a "$SUMMARY"
    exit 0
fi
n=$(tr ',' '\n' <<< "$LANES" | grep -c .)
echo "# changed lanes on $COL: $n (list $OUT/lanes-$COL.txt)"
rc=0

# --- verify against the committed table ---------------------------------------------
venv=(env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python")
[ "$COL" = cpu ] && venv+=(MOJOLEARN_VENDOR=cpu)
"${venv[@]}" "${PY[@]}" -m mojolearn verify --lanes "$LANES" --json --cpu-threads "$T" \
    > "$OUT/verify-$COL.json" 2> "$OUT/verify-$COL.log"
vrc=$?
echo "# verify --lanes ($n) on $COL exited $vrc (log $OUT/verify-$COL.log)"

# --- one line per lane ----------------------------------------------------------------
"${PY[@]}" tools/changed_lanes_summary.py "$OUT/verify-$COL.json" --column "$COL" --lanes "$LANES" --out "$SUMMARY" \
    || rc=1
exit $rc
