#!/bin/bash
# changed_lanes_check.sh <outdir> <column>
#
# THE CHANGED LANES, CHECKED WHERE THE BINDINGS WERE JUST BUILT (Andrew, 2026-10-13: fold light
# verification of what changed into compiling, so a release needs no separate GPU rental for
# re-records). Runs in a BUILT tree (every binding of <column> built in python/mojolearn/), once:
#
#   1. RECORD_LABEL set (the release's NVIDIA and AMD build legs): one identity record of the changed
#      lanes, `LANES=<lanes> REPEATS=1 tools/record_identity_column.sh <label> <outdir>/record`, which
#      tools/release.py's identity-admit step admits at home (NVIDIA == AMD, 0 conflicts);
#   2. `python -m mojolearn verify --lanes <lanes> --json` on <column>, the default fixture, one run,
#      against the table committed in this tree;
#   3. one line per lane (tools/changed_lanes_summary.py): VERIFIED, DIVERGENT with the first differing
#      cell's hashes, or NO-REFERENCE. Before the admit, a changed lane reads DIVERGENT against the old
#      table whenever its bits moved; that is what the record is for, never a reason to rerun.
#
#   <outdir>   absolute, outside the tree; created. Writes summary.txt (appended), verify-<column>.json
#              and .log, lanes-<column>.txt, record/ (with RECORD_LABEL)
#   <column>   cuda | hip | metal | cpu   (cpu sets MOJOLEARN_VENDOR=cpu: the host column)
#
# Environment:
#   LANES=a,b        the changed lanes. Unset: computed here, `tools/identity_columns.py changed-lanes
#                    --scope routine --backend <column>` (needs git history; honors
#                    MOJOLEARN_RELEASE_BUILD_ONLY through tools/lane_select.py). Empty or `none`:
#                    prints NOTHING-CHANGED and exits 0
#   RECORD_LABEL=x   also record (step 1); nvidia-*/amd-* labels only
#   PYTHON=...       the interpreter (default: the tree's .pixi default env python, else python3)
#   CPU_THREADS=3    verify --cpu-threads and the record's thread pools
#
# Exit: 0 when every step ran (whatever the verdicts say; the summary carries them), 1 when a step could
# not run (no report, no record). Never compiles, never rents, never reruns: identity runs ONCE.
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

# --- 1. the record (GPU build legs) ------------------------------------------------
if [ -n "${RECORD_LABEL:-}" ]; then
    case $RECORD_LABEL in nvidia-*|amd-*) ;; *) die "RECORD_LABEL $RECORD_LABEL: only nvidia-* or amd-* columns are recorded here" ;; esac
    LANES="$LANES" REPEATS=1 CPU_THREADS="$T" PYTHON="${PY[*]}" \
        bash tools/record_identity_column.sh "$RECORD_LABEL" "$OUT/record" > "$OUT/record.log" 2>&1
    rrc=$?
    line=$(grep -m 1 '^RECORDED' "$OUT/record.log" | cut -c1-300)
    echo "CHANGED-LANES-RECORD column=$COL label=$RECORD_LABEL rc=$rrc ${line:-NO RECORDED LINE (see $OUT/record.log)}" | tee -a "$SUMMARY"
    [ $rrc = 0 ] || rc=1
fi

# --- 2. verify against the committed table ---------------------------------------------
venv=(env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python")
[ "$COL" = cpu ] && venv+=(MOJOLEARN_VENDOR=cpu)
"${venv[@]}" "${PY[@]}" -m mojolearn verify --lanes "$LANES" --json --cpu-threads "$T" \
    > "$OUT/verify-$COL.json" 2> "$OUT/verify-$COL.log"
vrc=$?
echo "# verify --lanes ($n) on $COL exited $vrc (log $OUT/verify-$COL.log)"

# --- 3. one line per lane ----------------------------------------------------------------
"${PY[@]}" tools/changed_lanes_summary.py "$OUT/verify-$COL.json" --column "$COL" --lanes "$LANES" --out "$SUMMARY" \
    || rc=1
exit $rc
