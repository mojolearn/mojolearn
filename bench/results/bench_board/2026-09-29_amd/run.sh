#!/bin/sh
# The full bench board on AMD (Andrew, 2026-09-29): the DigitalOcean MI325X
# (do-amd), IDENTICAL only (the board plans it), our GPU arm against the AMD
# opponents (XGBoost's GPU arm where a ROCm build exists, otherwise the CPU
# learners on every core; torch ROCm 6.4.1), mojolearn 0.8.25 from PyPI (the
# core and its mojolearn-amd plugin, the version the Apple and NVIDIA runs
# use), the parameter check in force, no smoke gate, no ours-cpu arm (as the
# Apple and NVIDIA runs).
# Submitted to the AMD steward as a speed job (exclusive on the GPU):
#
#   python3 tools/apple_steward.py submit --kind speed --lane bench-board-amd \
#       --commit <pushed sha> --target do-amd \
#       --cmd 'sh bench/results/bench_board/2026-09-29_amd/run.sh [bench_board.py args, e.g. --rerun <prefixes>]'
#
# Resumable: every finished race is in board.json, and the next queued copy of
# this job skips it. The steward ends a speed command at 3 hours, so this
# script stops the board itself at 2 h 50 min, kills every process of this
# run, and renders BOARD.md from what is finished.
#
# THE INTERPRETER: the pinned torch ROCm wheels are cp312, so the board's venv
# is built from the box's Python 3.12 (/usr/bin/python3, 3.12.3 on do-amd),
# without the system site packages (the image carries no torch or learners).
#
# DATA (staged from R2 by the laptop, sha256 verified on the box by
# tools/dataset_store.sh stage): taxi_speed.npz, istella_speed.npz and
# istella_rank.npz under /root/datasets/gbm-bench; the corpora under
# /root/r2-stage/corpus (tools/bench_board_algos.py corpus_path).
set -u
OUT="$HOME/mojolearn-evidence/bench-board/2026-09-29_amd"
CACHE="$HOME/bench-board-cache"
DATA="$HOME/datasets/gbm-bench"
LOG="$OUT/run.log"
PY=/usr/bin/python3
STOP_S=${BOARD_STOP_S:-10200}          # 2 h 50 min: under the steward's 3-hour cap
mkdir -p "$OUT" "$CACHE"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "run.sh start at $(git rev-parse HEAD 2>/dev/null) on $(hostname) args: $*"

for f in taxi/taxi_speed.npz istella/istella_speed.npz istella/istella_rank.npz; do
    [ -s "$DATA/$f" ] || { say "REFUSING: $DATA/$f is not staged"; exit 1; }
done
for k in corpus/enwik8/input.txt corpus/pile_github/input.txt; do
    [ -s "$HOME/r2-stage/$k" ] || { say "REFUSING: $HOME/r2-stage/$k is not staged"; exit 1; }
done

start=$(date -u +%s)
stop=$((start + STOP_S))
say "board runs until $(date -u -d "@$stop" +%FT%TZ) at most"

"$PY" tools/bench_board.py --vendor amd --mojolearn-version 0.8.25 \
    --base-python "$PY" --out "$OUT" --cache "$CACHE" --data-root "$DATA" \
    --no-cpu-arm --no-smoke-gate "$@" >> "$LOG" 2>&1 &
pid=$!
killrun() {   # every process of this run but this script and its parent
    for pat in "$PWD/tools/" "$PWD/bench/speed/" "$CACHE/"; do
        for p in $(pgrep -f "$pat"); do
            [ "$p" = $$ ] || [ "$p" = "$PPID" ] || kill -9 "$p" 2>/dev/null
        done
    done
}
while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date -u +%s)" -ge "$stop" ]; then
        say "STOP: time limit reached; killing the board and its drivers"
        kill -9 "$pid" 2>/dev/null
        killrun
        sleep 5
        break
    fi
    sleep 20
done
wait "$pid" 2>/dev/null; rc=$?
say "board exit $rc"
"$PY" tools/bench_board.py --out "$OUT" --render-only >> "$LOG" 2>&1
say "rendered: $(ls -la "$OUT/BOARD.md" 2>&1)"
tail -40 "$LOG"
exit 0
