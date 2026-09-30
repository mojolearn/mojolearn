#!/bin/sh
# tools/amd_step_time_leg4.sh -- lane/amd-step-time (2026-09-24), the fourth
# AMD leg: the trial arm `-D MOJOLEARN_FTZ_CLASS_AMD=1` (checks/numerics.mojo:
# `ftz` spelled as one class compare and a select in AMD device code, the
# same function for every input) against the branch head, on top of the
# launch bound and the leaf split. Needs /root/urls and /root/amd_in (pushed
# by the lane after the VM is up).
#   1. base binding with and without the arm; byte LM: lean and timers, each
#      with and without the arm; estimators and embedding with the arm
#   2. B4 itemization (timers) and lean step, both arms
#   3. the replay of steps 101..103 from ckpt 100 with the arm, held to the
#      H100 chain; the lanes byte-lm, byte-lm-resident, ols, ridge, pca,
#      kmeans, logistic, tsvd, embedding, gemm-pinned on the arm's bindings
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
BIN=/root/amd_bin; mkdir -p "$BIN"
S="sh tools/amd_step_time_session.sh"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg4 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
FC="-D MOJOLEARN_FTZ_CLASS_AMD=1"
TM="-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1"

blm() {  # tag, defines
    mkdir -p "$BIN/out_$1"; rm -f "$BIN/out_$1/_mojolearn_byte_lm.so"
    t0=$(date +%s)
    MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_$1" MOJOLEARN_BUILD_EXTRA_DEFINES="$2" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.$1.log" 2>&1
    say "byte_lm $1 exit=$? secs=$(( $(date +%s) - t0 ))"
    cp "$BIN/out_$1/_mojolearn_byte_lm.so" "$BIN/byte_lm.$1.so"
}
t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_BUILD_EXTRA_DEFINES="$FC" sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "base (ftzclass) exit=$? secs=$(( $(date +%s) - t0 ))"
( blm lean "" ) & ( blm ftz "$FC" ) & ( blm timers "$TM" ) & ( blm timersftz "$TM $FC" ) &
( MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_BUILD_EXTRA_DEFINES="$FC" sh bindings/build_estimators.sh > "$OUT/builds/estimators.log" 2>&1; say "estimators (ftzclass) exit=$?" ) &
( MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_BUILD_EXTRA_DEFINES="$FC" sh bindings/build_embedding.sh > "$OUT/builds/embedding.log" 2>&1; say "embedding (ftzclass) exit=$?" ) &
( MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build_linalg.sh > "$OUT/builds/linalg.log" 2>&1; say "linalg exit=$?" ) &
wait

item() {  # tag
    $S use "$1" > /dev/null
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/item-$1" --shape 4 2048 768 12 12 64 2048 12 50257 \
        --steps 1 --resident-lean --component-timing --component-timing-steps 2 --budget-seconds 600 > "$OUT/item-$1.log" 2>&1
    grep -h '^timing ' "$OUT/item-$1.log" "$OUT/item-$1"/*.log 2>/dev/null > "$OUT/item-$1.timing.txt"
    python3 tools/amd_step_timing_summary.py "$OUT/item-$1.timing.txt" --skip-shards 1 --tsv "$OUT/item-$1.summary.tsv" > /dev/null 2>&1
    say "item $1: $(tail -1 "$OUT/item-$1.summary.tsv")"
}
lean() {
    $S use "$1" > /dev/null
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/lean-$1" --shape 4 2048 768 12 12 64 2048 12 50257 \
        --steps 3 --resident-lean --witness-every-step --budget-seconds 600 > "$OUT/lean-$1.log" 2>&1
    say "lean $1: $(grep -o '"steady_median_seconds": [0-9.]*' "$OUT/lean-$1/result.json")"
}
item timers; item timersftz; lean lean; lean ftz; lean lean; lean ftz

while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
$S replay ftz ckpt_00000100.blm A-1.chain.partial.jsonl 3 > /dev/null 2>&1
$S use ftz > /dev/null
pixi run python -m mojolearn verify --lanes "byte-lm,byte-lm-resident,ols,ridge,pca,kmeans,logistic,tsvd,embedding,gemm-pinned" > "$OUT/verify-ftz.log" 2>&1
say "verify ftz exit=$?: $(grep RESULT "$OUT/verify-ftz.log" | tail -1 | cut -c1-260)"

touch /root/amd_step_ready
say "leg4 scripted part done; holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
