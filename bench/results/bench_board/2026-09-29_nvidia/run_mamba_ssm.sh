#!/bin/bash
# The mamba-ssm opponent on the NVIDIA board (orchestrator, 2026-09-29): the
# three Mamba forward lanes with mamba_ssm's fused kernels (mamba-ssm-fp32,
# mamba-ssm-tf32) beside ours and the torch arms, in their OWN --out and venv:
# the mamba-ssm pins would change the main board's resume key (run.sh passes
# --no-mamba-ssm). The shared opponent store (/root/bench-board/
# opponent-store.jsonl) supplies the torch arms already measured on this box.
#
#   tools/nvidia_central.sh submit bench-board-l40s --cap 240 bench/results/bench_board/2026-09-29_nvidia/run_mamba_ssm.sh
set -u
OUT="/root/bench-board/2026-09-29_nvidia_mamba-ssm"
CACHE="/root/bench-board-cache-mamba-ssm"   # its own clean venv (mamba-ssm builds from source here)
LOG="$OUT/run.log"
PY=/usr/bin/python3
STOP_S=${BOARD_STOP_S:-13200}
mkdir -p "$OUT" "$CACHE"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "run_mamba_ssm.sh start (patch sync: $(tr '\n' ' ' < "$(git rev-parse --absolute-git-dir)/devpod_synced" 2>/dev/null || echo none)) job ${NVQ_JOB_ID:-?} GPU ${CUDA_VISIBLE_DEVICES:-?} on $(hostname)"
[ -f /root/datasets/.board-nvidia-staged-ok ] || { say "REFUSING: data not staged"; exit 1; }
start=$(date -u +%s); stop=$((start + STOP_S))
"$PY" tools/bench_board.py --vendor nvidia --mojolearn-version 0.8.25 \
    --base-python "$PY" --out "$OUT" --cache "$CACHE" --no-cpu-arm --no-smoke-gate \
    --opponent-store /root/bench-board/opponent-store.jsonl \
    --families neural --lanes mamba1-forward,mamba2-forward,mamba3-forward >> "$LOG" 2>&1 &
pid=$!
killrun() {
    for pat in "$PWD/tools/" "$PWD/bench/speed/" "$CACHE/"; do
        for p in $(pgrep -f "$pat"); do
            [ "$p" = $$ ] || [ "$p" = "$PPID" ] || kill -9 "$p" 2>/dev/null
        done
    done
}
while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date -u +%s)" -ge "$stop" ]; then
        say "STOP: time limit reached; killing the board and its drivers"
        kill -9 "$pid" 2>/dev/null; killrun; sleep 5; break
    fi
    sleep 20
done
wait "$pid" 2>/dev/null; rc=$?
say "board exit $rc"
"$PY" tools/bench_board.py --out "$OUT" --render-only >> "$LOG" 2>&1
tail -40 "$LOG"
exit 0
