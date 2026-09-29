#!/bin/sh
# The full bench board WITH the parameter check in force (Andrew, 2026-09-29),
# its own output directory, our FAST and IDENTICAL against each opponent,
# mojolearn 0.8.25 from PyPI. Pass --families to choose what runs.
# Run by the apple steward (speed kind) from the repo root. Resumable: every
# finished race is in board.json, and a rerun of this script skips it.
#
#   sh bench/results/bench_board/2026-09-29_m3ultra_checked/run.sh [--skip-failed]
#
# The steward kills a speed command at 3 hours, which would orphan the board's
# drivers (they run in their own sessions). So this script stops the board
# itself at 2 h 50 min, kills every process of this run, and renders BOARD.md
# from what is finished. The next queued copy of this job resumes. (The first
# job also stopped at 12:50Z, when the box was to end at 13:00Z; Andrew then
# kept the box for the whole board, so that stop is gone.)
set -u
OUT="$HOME/mojolearn-evidence/bench-board/2026-09-29_m3ultra_checked"
CACHE="$HOME/bench-board-cache"
LOG="$OUT/run.log"
PY="$HOME/mojolearn/.pixi/envs/default/bin/python3.13"
mkdir -p "$OUT" "$CACHE"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "run.sh start at $(git rev-parse HEAD) args: $*"

# data: staged from R2 by a script holding presigned GET URLs (no credential on
# the box); once per box
STAGE="$HOME/mojolearn-evidence/bench-board-m3ultra/stage_m3ultra.sh"
if [ ! -f "$HOME/datasets/.board-staged-ok" ]; then
    say "staging data"
    sh "$STAGE" >> "$LOG" 2>&1 || { say "STAGING FAILED"; exit 1; }
    touch "$HOME/datasets/.board-staged-ok"
    say "staged"
fi

start=$(date -u +%s)
stop=$((start + 10200))
say "board runs until $(date -u -r "$stop" +%FT%TZ) at most"

"$PY" tools/bench_board.py --vendor apple --mojolearn-version 0.8.25 \
    --base-python "$PY" --out "$OUT" --cache "$CACHE" --no-cpu-arm "$@" \
    >> "$LOG" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date -u +%s)" -ge "$stop" ]; then
        say "STOP: time limit reached; killing the board and its drivers"
        kill -9 "$pid" 2>/dev/null
        pkill -9 -f "$PWD/tools/" 2>/dev/null
        pkill -9 -f "$PWD/bench/" 2>/dev/null
        pkill -9 -f "$CACHE/" 2>/dev/null
        sleep 5
        break
    fi
    sleep 20
done
wait "$pid" 2>/dev/null; rc=$?
say "board exit $rc"
"$PY" tools/bench_board.py --out "$OUT" --render-only >> "$LOG" 2>&1
say "rendered: $(ls -la "$OUT/BOARD.md" 2>&1)"
tail -60 "$LOG"
exit 0
