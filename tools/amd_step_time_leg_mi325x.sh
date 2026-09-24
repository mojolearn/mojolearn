#!/bin/sh
# tools/amd_step_time_leg_mi325x.sh -- lane/amd-step-time (2026-09-24): the
# MI325X confirmation on the T3 run's own GPU model (DigitalOcean
# gpu-mi325x1-256gb through tools/do_extra_leg.sh amd, gates on). Needs
# /root/urls and /root/amd_in pushed by the lane after the droplet is up.
#   1. base binding; byte LM: baseline (the lane's four revert defines plus
#      the attention trial build run on the pre-lane AMD attention word, so
#      the baseline is main before the lane), the branch head (origin/main),
#      the branch's timers build
#   2. replays held to the H100 chain: baseline steps 101..102 from ckpt 100,
#      branch steps 101..103 from ckpt 100 and 1999..2000 from ckpt 1998
#   3. lean B4 step (baseline and branch) and the timed B4 itemization
#   4. every device binding; gemm device, backward and workspace checks;
#      python -m mojolearn verify over the 201 GEMM-reaching non-par lanes
#      (bench/results/amd_step_time_2026-09-24/lanes_gemm_nonpar.txt) in
#      chunks of 25, against the shipped reference table
# Then holds until /root/amd_step_done (at most 15 minutes).
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds" "$OUT/verify"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
BIN=/root/amd_bin; mkdir -p "$BIN"
S="sh tools/amd_step_time_session.sh"
ST="$OUT/session.txt"
# the AMD attention default before the lane (checks/kernel_matrix.mojo at 5a3804a45)
PRE_ARM=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "mi325x leg started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
rocm-smi --showproductname --showuniqueid --showclocks > "$OUT/gpu.txt" 2>&1
{ cat /opt/rocm/.info/version 2>/dev/null; ls -d /opt/rocm* 2>/dev/null; } > "$OUT/rocm.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

blm() {  # tag, defines
    mkdir -p "$BIN/out_$1"; rm -f "$BIN/out_$1/_mojolearn_byte_lm.so"
    t0=$(date +%s)
    MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_$1" MOJOLEARN_BUILD_EXTRA_DEFINES="$2" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.$1.log" 2>&1
    say "byte_lm $1 exit=$? secs=$(( $(date +%s) - t0 ))"
    cp "$BIN/out_$1/_mojolearn_byte_lm.so" "$BIN/byte_lm.$1.so"
}
lean() {  # tag
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/lean-$1" --shape 4 2048 768 12 12 64 2048 12 50257 \
        --steps 3 --resident-lean --witness-every-step --budget-seconds 900 > "$OUT/lean-$1.log" 2>&1
    say "lean $1: $(python3 -c "import json;r=json.load(open('$OUT/lean-$1/result.json'));print(r['steady_median_seconds'], [w['sha256']['parameters'][:12] for w in r['step_witnesses']])" 2>&1 | tail -1)"
}
t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "base exit=$? secs=$(( $(date +%s) - t0 ))"
( blm baseline "-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1 -D MOJOLEARN_GEMM_NO_LEAF_SPLIT=1 -D MOJOLEARN_GEMM_NO_MFMA=1 -D MOJOLEARN_FTZ_NO_CLASS=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1" ) &
( blm branch "" ) &
( blm timers "-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1" ) &
while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ] || [ ! -s /root/amd_in/A-2.chain.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
say "fetch exit=$?"
wait

# ---- the step: baseline (pre-lane attention arm) and branch ----
MOJOLEARN_ATTN_ARM=$PRE_ARM $S replay baseline ckpt_00000100.blm A-1.chain.partial.jsonl 2 > /dev/null 2>&1
$S replay branch ckpt_00000100.blm A-1.chain.partial.jsonl 3 > /dev/null 2>&1
$S replay branch ckpt_00001998.blm A-2.chain.jsonl 2 > /dev/null 2>&1
$S use branch > /dev/null
lean branch
$S use baseline > /dev/null
MOJOLEARN_ATTN_ARM=$PRE_ARM lean baseline
$S use timers > /dev/null
pixi run python tools/lm_step_memory_probe.py --out "$OUT/item-timers" --shape 4 2048 768 12 12 64 2048 12 50257 \
    --steps 1 --resident-lean --component-timing --component-timing-steps 2 --budget-seconds 600 > "$OUT/item-timers.log" 2>&1
grep -h '^timing ' "$OUT/item-timers.log" "$OUT/item-timers"/*.log 2>/dev/null > "$OUT/item-timers.timing.txt"
python3 tools/amd_step_timing_summary.py "$OUT/item-timers.timing.txt" --skip-shards 1 --tsv "$OUT/item-timers.summary.tsv" > "$OUT/item-timers.summary.txt" 2>&1
say "item timers: $(tail -1 "$OUT/item-timers.summary.tsv")"

# ---- identity: every device binding, the GEMM checks, the GEMM-reaching lanes ----
$S use branch > /dev/null
n=0
for f in estimators linalg training transformer embedding kernel_methods gp svm mixture metrics preprocessing resample solver rf gbdt trees hdbscan ivf mamba arima tsa; do
    ( t0=$(date +%s); MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=2 sh bindings/build_$f.sh > "$OUT/builds/$f.log" 2>&1
      say "build $f exit=$? secs=$(( $(date +%s) - t0 ))" ) &
    n=$((n + 1))
    if [ $((n % 4)) -eq 0 ]; then wait; fi
done
wait
say "builds done"
for c in gemm_device_check gemm_backward_check gemm_workspace_check; do
    extra=""
    [ "$c" = gemm_workspace_check ] && extra="-D MOJOLEARN_STEP_PHASE_TIMERS=1"
    t0=$(date +%s)
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" $extra \
        -I . gemm/checks/$c.mojo > "$OUT/$c.log" 2>&1
    say "$c exit=$? secs=$(( $(date +%s) - t0 )): $(grep -E 'all green|PASS|FAIL' "$OUT/$c.log" | tail -1 | cut -c1-200)"
done
split -l 25 -d bench/results/amd_step_time_2026-09-24/lanes_gemm_nonpar.txt "$BIN/lanechunk"
for f in "$BIN"/lanechunk*; do
    k=$(basename "$f" | sed 's/lanechunk//')
    t0=$(date +%s)
    pixi run python -m mojolearn verify --lanes "$(tr '\n' ',' < "$f" | sed 's/,$//')" --json-out "$OUT/verify/chunk$k.json" > "$OUT/verify/chunk$k.log" 2>&1
    say "verify chunk$k exit=$? secs=$(( $(date +%s) - t0 )): $(grep RESULT "$OUT/verify/chunk$k.log" | tail -1 | cut -c1-300)"
done
gzip -9 -f "$OUT"/verify/*.json

touch /root/amd_step_ready
say "MI325X leg scripted part done; holding (at most 15 min)"
n=0
while [ ! -e /root/amd_step_done ] && [ $n -lt 45 ]; do sleep 20; n=$((n + 1)); done
exit 0
