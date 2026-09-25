#!/bin/sh
# tools/release_0819/nvidia_leg.sh -- lane/release-0819 (2026-09-25): the
# NVIDIA proof of the MERGED source (lane/nvidia-step-time +
# lane/amd-step-time-2-proof) on one RunPod H100 80GB HBM3, through
# tools/gemm_remote_leg.sh nvidia (whose own gates run gemm_device_check and
# the card first). Every binary built on the box from this commit, sm_90a,
# IDENTICAL, column nvidia:
#   1. every device binding; the byte LM binding (merged defaults)
#   2. GEMM A/B at the T3 shapes: ref (-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1
#      -D MOJOLEARN_GEMM_NO_KPACK_NARROW=1: 0.8.18's NVIDIA GEMM) and merged,
#      kinds ordinary,tiny,mixed (the lane's 36 cases), then skew,border,sparse
#      (36 more); the admission sabotage on the first three kinds (must DIFFER)
#   3. gemm_backward_check, gemm_workspace_check
#   4. lean B4 witnesses against the known digests; replays held to the H100
#      chain (101..103 from ckpt 100, 1999..2000 from ckpt 1998)
#   5. verify over the 201 non-par lanes that reach gemm_identical.mojo
# Needs /root/urls and /root/amd_in pushed after the pod is up. Holds until
# /root/nv_step_done (at most 4 minutes).
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/nv-step-time
mkdir -p "$OUT/builds" "$OUT/ab" "$OUT/verify"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia
: "${MOJOLEARN_GPU_ARCHS:=sm_90a}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
S="sh tools/nvidia_step_time/session.sh"
ST="$OUT/session.txt"
BIN=/root/nv_bin; mkdir -p "$BIN"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "release-0819 nvidia leg started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
{ nvidia-smi; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv; } > "$OUT/gpu.txt" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "build base exit=$? secs=$(( $(date +%s) - t0 ))"
n=0
for f in estimators linalg training transformer embedding kernel_methods gp svm mixture metrics preprocessing resample solver rf gbdt trees hdbscan ivf mamba arima tsa; do
    ( t0=$(date +%s); MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=4 sh bindings/build_$f.sh > "$OUT/builds/$f.log" 2>&1
      say "build $f exit=$? secs=$(( $(date +%s) - t0 ))" ) &
    n=$((n + 1))
    if [ $((n % 8)) -eq 0 ]; then wait; fi
done
( $S bind merged ) &
wait
cp "$BIN/byte_lm.merged.so" python/mojolearn/identical/_mojolearn_byte_lm.so

AB_REF=ref $S ab ref MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1 MOJOLEARN_GEMM_NO_KPACK_NARROW=1
AB_REF=ref $S ab merged
AB_REF=ref $S ab sabotage MOJOLEARN_GEMM_SABOTAGE_ADMIT_ALWAYS=1
say "sabotage lines differing from ref: $(diff "$OUT/ab/ref.hashes" "$OUT/ab/sabotage.hashes" | grep -c '^>')"
AB_KINDS=skew,border,sparse AB_REF=ref3 $S ab ref3 MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1 MOJOLEARN_GEMM_NO_KPACK_NARROW=1
AB_KINDS=skew,border,sparse AB_REF=ref3 $S ab merged3
t0=$(date +%s)
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
    -I . gemm/checks/gemm_backward_check.mojo > "$OUT/gemm_backward_check.log" 2>&1
say "gemm_backward_check exit=$? secs=$(( $(date +%s) - t0 )): $(tail -1 "$OUT/gemm_backward_check.log" | cut -c1-200)"
t0=$(date +%s)
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_STEP_PHASE_TIMERS=1 --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
    -I . gemm/checks/gemm_workspace_check.mojo > "$OUT/gemm_workspace_check.log" 2>&1
say "gemm_workspace_check exit=$? secs=$(( $(date +%s) - t0 )): $(tail -1 "$OUT/gemm_workspace_check.log" | cut -c1-200)"

while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ] || [ ! -s /root/amd_in/A-2.chain.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
$S lean merged 3
say "lean witnesses vs known: $(python3 -c "
import json
r=json.load(open('$OUT/lean-merged/result.json'))
p=[w['sha256']['parameters'][:12] for w in r['step_witnesses']]
l=[w['sha256'].get('loss','')[:12] for w in r['step_witnesses']]
print('EQUAL' if p==['5516ffe5f550','77477af42588','4e439a8a9751'] and l==['676298dabb30','afc46227a372','34fe4c49b0dd'] else 'DIFFER')" 2>&1 | tail -1)"
$S replay merged ckpt_00000100.blm A-1.chain.partial.jsonl 3
$S replay merged ckpt_00001998.blm A-2.chain.jsonl 2

cp "$BIN/byte_lm.merged.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sh tools/nvidia_step_time/verify_lanes.sh > "$OUT/verify_run.log" 2>&1
say "verify: $(tail -1 "$OUT/verify/summary.txt" 2>/dev/null)"

touch /root/nv_step_ready
say "release-0819 nvidia leg scripted part done; holding (at most 4 min)"
n=0
while [ ! -e /root/nv_step_done ] && [ $n -lt 12 ]; do sleep 20; n=$((n + 1)); done
exit 0
