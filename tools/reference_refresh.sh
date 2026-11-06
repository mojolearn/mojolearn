#!/bin/bash
# tools/reference_refresh.sh [<sha>] [columns] -- THE REFERENCE REFRESH after merges to main (2026-11-04): record
# the identity lanes of <sha> (default: the tip of origin/main) whose closure differs from their admitted record on
# each column of the table (nvidia, amd, cpu, apple; the old ptx column is history) over every lane the harness ships, admit every fetched
# record at once into python/mojolearn/verify_reference/table.json and commit the admission on branch
# ref/<sha12>, pushed for the orchestrator to merge. So the release freeze finds every row already admitted.
# Pending refreshes of older commits are superseded: N merges cost one refresh, at the newest commit.
#
#   tools/reference_refresh.sh --plan [<sha>] [columns]    lanes OWED / REUSED and the GAPS per column; submits nothing
#   tools/reference_refresh.sh [<sha>] [columns]           the refresh (idempotent: rerun after any crash)
#   tools/reference_refresh.sh --no-wait [<sha>] [columns] submit what is owed, exit; rerun to fetch and admit
#   tools/reference_refresh.sh --status <sha>              the saved state
#   tools/reference_refresh.sh --resubmit COL [<sha>]      forget a dead box's job, submit it again
#   ... --scope routine|neural-training --reason TEXT      narrow the lanes (fewer columns also need --reason)
#
# Exit codes: 0 done, 2 refused, 3 still in flight, 4 admitted with GAPS (owed lanes no box or column records).
# Orchestrator only: it submits to the NVIDIA pod and the AMD droplet that are already up (it rents nothing).
# Details: the docstring of tools/reference_refresh.py and docs/release/REFERENCE_REFRESH.md.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${REFRESH_PYTHON:-}
if [ -z "$PY" ]; then
    for p in "$ROOT/.pixi/envs/default/bin/python" "$HOME/CascadeProjects/mojolearn/.pixi/envs/default/bin/python"; do
        [ -x "$p" ] && { PY=$p; break; }
    done
fi
PY=${PY:-python3}
export REFRESH_PYTHON=$PY
exec "$PY" "$ROOT/tools/reference_refresh.py" "$@"
