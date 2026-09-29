#!/bin/sh
# The M2 Pro's classical/dbscan races, LAST, on the newest released wheel
# (Andrew, 2026-09-29): the M2 Pro board's hour-long copies could not finish
# dbscan/istella on 0.8.25, so every copy started it again. Here they run on
# mojolearn 0.8.31 (it carries the int64 edge-count fix, merge 71b7af311) in
# their own output directory and venv (a board is one box and one wheel), in
# one job with the longest stop the steward allows (it kills a speed command
# at 3 hours). The board's --stop-at is the job's stop: a race still running
# then is killed by the board and recorded as a TIMEOUT by name, never
# started again. Submitted only when the M2 Pro is otherwise idle.
#
#   sh bench/results/bench_board/2026-09-29_m2pro_dbscan/run.sh --no-smoke-gate
set -u
OUT="$HOME/mojolearn-evidence/bench-board/2026-09-29_m2pro_dbscan"
CACHE="$HOME/bench-board-cache"
VENV="$CACHE/venv-0831"
LOG="$OUT/run.log"
PY="$HOME/mojolearn/.pixi/envs/default/bin/python3.13"
mkdir -p "$OUT" "$CACHE"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "run.sh start at $(git rev-parse HEAD) args: $*"

if [ ! -f "$HOME/datasets/.board-staged-ok" ]; then
    say "REFUSED: the board data is not staged on this box (~/datasets/.board-staged-ok)"
    exit 1
fi

start=$(date -u +%s)
stop=$((start + ${BOARD_STOP_S:-10500}))
say "board runs until $(date -u -r "$stop" +%FT%TZ) at most"

"$PY" tools/bench_board.py --vendor apple --mojolearn-version 0.8.31 \
    --base-python "$PY" --out "$OUT" --cache "$CACHE" --venv "$VENV" --no-cpu-arm \
    --families classical --lanes dbscan --skip-failed --stop-at "$((stop - 120))" "$@" \
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
