#!/bin/bash
# The full bench board on NVIDIA (Andrew, 2026-09-29): IDENTICAL only (the
# board plans it), our GPU arm against the NVIDIA opponents (cuML, cuVS,
# cuGraph, XGBoost/CatBoost/LightGBM GPU arms, torch CUDA), mojolearn 0.8.25
# from PyPI (the core and its mojolearn-nvidia plugin, the same version the
# Apple run used), the parameter check in force, no smoke gate.
# Submitted through the shared pod queue from the lane tree:
#
#   tools/nvidia_central.sh submit bench-board-nvidia --cap 240 bench/results/bench_board/2026-09-29_nvidia/run.sh
#
# Extra arguments for bench_board.py (e.g. --rerun <prefixes>) are read from
# $ARGS_FILE when it exists (the queue takes one script path, no arguments).
#
# Resumable: every finished race is in board.json, and the next queued copy of
# this job skips it. The queue kills a job at its 240-minute cap, which would
# orphan the board's drivers (their own sessions), so this script stops the
# board itself first, kills every process of this run and renders BOARD.md
# from what is finished. The Mac fetches $OUT_ROOT after every job
# (~/mojolearn-evidence/bench-board/2026-09-29_nvidia/).
set -u
OUT_ROOT="/root/bench-board"
OUT="$OUT_ROOT/2026-09-29_nvidia"
CACHE="/root/bench-board-cache"
ARGS_FILE="$OUT_ROOT/next-args"
LOG="$OUT/run.log"
PY=/usr/bin/python3
STOP_S=${BOARD_STOP_S:-13200}          # 220 min: under the 240-minute job cap
mkdir -p "$OUT" "$CACHE"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "run.sh start at $(git rev-parse HEAD 2>/dev/null) (the pod's base; patch sync: $(tr '\n' ' ' < "$(git rev-parse --absolute-git-dir)/devpod_synced" 2>/dev/null || echo none)) job ${NVQ_JOB_ID:-?} GPU ${CUDA_VISIBLE_DEVICES:-?} on $(hostname)"

# data: staged from R2 by the Mac (tools/dataset_store.sh stage, sha256
# verified); the Mac writes the marker when every key is verified
MARK=/root/datasets/.board-nvidia-staged-ok
waited=0
while [ ! -f "$MARK" ]; do
    [ $waited -lt 5400 ] || { say "REFUSING: data not staged ($MARK missing after 90 min)"; exit 1; }
    [ $waited -gt 0 ] || say "waiting for the R2 staging marker $MARK"
    sleep 30; waited=$((waited + 30))
done

extra=()
if [ -s "$ARGS_FILE" ]; then
    read -r -a extra < "$ARGS_FILE"
    say "extra args from $ARGS_FILE: ${extra[*]}"
    mv "$ARGS_FILE" "$ARGS_FILE.used-$(date -u +%Y%m%dT%H%M%SZ)"
fi

start=$(date -u +%s)
stop=$((start + STOP_S))
say "board runs until $(date -u -d "@$stop" +%FT%TZ) at most"

"$PY" tools/bench_board.py --vendor nvidia --mojolearn-version 0.8.25 \
    --base-python "$PY" --system-site-packages --out "$OUT" --cache "$CACHE" \
    --no-cpu-arm --no-smoke-gate "${extra[@]}" >> "$LOG" 2>&1 &
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
