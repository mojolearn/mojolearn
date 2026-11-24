#!/bin/bash
# changed_lanes_check.sh <outdir> <column>
#
# THE CHANGED LANES, CHECKED WHERE THE BINDINGS WERE JUST BUILT (Andrew, 2026-10-13: fold light
# verification of what changed into compiling). Runs in a BUILT tree (the bindings of <column> built in
# python/mojolearn/), once: `python -m mojolearn verify --lanes <changed lanes> --json` on <column>, the
# default fixture, one run, against the table committed in this tree, then one line per lane
# (tools/changed_lanes_summary.py): VERIFIED, DIVERGENT with the first differing cell's hashes, or
# NO-REFERENCE. ONE PROCESS, BOTH SIDES (2026-11-24): verify runs each lane through device="gpu" and
# device="cpu", so a GPU <column> also prints the cpu column's lines (its own changed-lane list,
# `--backend cpu`), from the same report. On the Mac, tools/mac_changed_lanes.sh runs it once for
# Metal and the CPU column after a compile. Apple and host digests are recorded, never required (identity is NVIDIA == AMD; those
# columns are recorded through lq, tools/record_changed_lanes.sh). A changed lane reads DIVERGENT
# against the old table whenever its bits moved; that is information, never a reason to rerun.
#
#   <outdir>   absolute; created. Writes summary.txt (appended), verify-<column>.json and .log,
#              lanes-<column>.txt
#   <column>   cuda | hip | metal | cpu   (cpu: the CPU side only, on a CPU-only tree or any other)
#
# Environment:
#   LANES=a,b        the changed lanes, for every column printed. Unset: computed here per column,
#                    `tools/identity_columns.py changed-lanes --scope routine --backend <column>` (and
#                    `--backend cpu` for the cpu column of a GPU tree; needs git history; honors
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

# --- which lanes, per column ---------------------------------------------------
# A GPU column prints its own lines and the cpu column's (side cpu of the same run).
# Plain variables, no associative array: macOS ships bash 3.2.
lanes_for() {  # $1 column -> stdout: its lane list ("" when none)
    local c=$1 l
    if [ -z "${LANES+x}" ]; then
        l=$("${PY[@]}" tools/identity_columns.py changed-lanes --scope routine --backend "$c" \
                2> "$OUT/lanes-$c.err" | tail -n 1) || return 1
        grep -m 1 'CHANGED-LANES' "$OUT/lanes-$c.err" >&2
    else
        l=$LANES
    fi
    [ "$l" = none ] && l=
    printf '%s\n' "$l" > "$OUT/lanes-$c.txt"
    printf '%s' "$l"
}
COL_L=$(lanes_for "$COL") \
    || { echo "CHANGED-LANES column=$COL LANE-LIST-FAILED (see $OUT/lanes-$COL.err)" | tee -a "$SUMMARY"; exit 1; }
CPU_L=
if [ "$COL" != cpu ]; then
    CPU_L=$(lanes_for cpu) \
        || { echo "CHANGED-LANES column=cpu LANE-LIST-FAILED (see $OUT/lanes-cpu.err)" | tee -a "$SUMMARY"; exit 1; }
fi
ALL=$(printf '%s,%s\n' "$COL_L" "$CPU_L" | tr ',' '\n' | grep . | sort -u | paste -sd, -)
summarize_col() {  # $1 column, $2 its lanes
    local c=$1 l=$2 side=gpu
    [ "$c" = cpu ] && side=cpu
    if [ -z "$l" ]; then
        echo "NOTHING-CHANGED column=$c: no lane's bits can have moved since the admitted table" | tee -a "$SUMMARY"
        return 0
    fi
    "${PY[@]}" tools/changed_lanes_summary.py "$OUT/verify-$COL.json" --column "$c" --side "$side" \
        --lanes "$l" --out "$SUMMARY"
}
if [ -z "$ALL" ]; then
    summarize_col "$COL" ""
    [ "$COL" = cpu ] || summarize_col cpu ""
    exit 0
fi
n=$(tr ',' '\n' <<< "$ALL" | grep -c .)
echo "# changed lanes on $COL$([ "$COL" = cpu ] || echo ' and cpu'): $n in one verify run (lists $OUT/lanes-*.txt)"
rc=0

# --- verify against the committed table: one process, both sides through device= --------
venv=(env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python")
"${venv[@]}" "${PY[@]}" -m mojolearn verify --lanes "$ALL" --json --cpu-threads "$T" \
    > "$OUT/verify-$COL.json" 2> "$OUT/verify-$COL.log"
vrc=$?
echo "# verify --lanes ($n) on $COL (both sides) exited $vrc (log $OUT/verify-$COL.log)"

# --- one line per lane, per column ----------------------------------------------------
summarize_col "$COL" "$COL_L" || rc=1
if [ "$COL" != cpu ]; then summarize_col cpu "$CPU_L" || rc=1; fi
exit $rc
