#!/bin/sh
# The M3 Ultra's races whose fix is in mojolearn 0.8.29 and not in 0.8.25
# (Andrew, 2026-09-29): classical/dbscan (the int64 edge count, merge
# 71b7af311) and algos/bayesian-gmm (the moments through the identical GEMM,
# merge 7f970402d), on taxi and Istella. Both merges are in 0.8.29's source
# commit fc735663f (bench/results/release_verification/2026-09-29_pypi_0829).
# Every other race of the Ultra board stays on 0.8.25 in
# 2026-09-29_m3ultra_checked and is not measured again.
#
# A board is one box and one wheel, so these races have their own output
# directory, and their own venv, so the 0.8.25 board's venv is never touched.
# Each race times ours and its opponents together, so it stands alone.
# Run by the apple steward (speed kind) from the repo root, between copies of
# the 0.8.25 board. Resumable like that board's run.sh.
#
#   sh bench/results/bench_board/2026-09-29_m3ultra_0829/run.sh --no-smoke-gate
set -u
OUT="$HOME/mojolearn-evidence/bench-board/2026-09-29_m3ultra_0829"
CACHE="$HOME/bench-board-cache"
VENV="$CACHE/venv-0829"
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
stop=$((start + 10200))
say "board runs until $(date -u -r "$stop" +%FT%TZ) at most"

"$PY" tools/bench_board.py --vendor apple --mojolearn-version 0.8.29 \
    --base-python "$PY" --out "$OUT" --cache "$CACHE" --venv "$VENV" --no-cpu-arm \
    --families classical,algos --lanes dbscan,bayesian-gmm "$@" \
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
