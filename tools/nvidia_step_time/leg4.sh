#!/bin/sh
# tools/nvidia_step_time/leg4.sh -- lane/nvidia-step-time (2026-09-25), the
# fourth H100 leg (leg 3 plus the attention launch bounds and the embedding
# run-start scan; the AMD column compiled): the branch head's NVIDIA defaults (GEMM window admission +
# the narrow 128x64 kpack tile under a 512 bound) proven and timed, then the
# identity lanes. Needs /root/urls and /root/amd_in pushed after the pod is up.
#   1. every device binding from this commit (four at a time); byte LM:
#      head (default), ref (-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1
#      -D MOJOLEARN_GEMM_NO_KPACK_NARROW=1: main's GEMM), timers
#   2. GEMM A/B, three operand kinds: ref, head, and the admission sabotage
#   3. gemm_backward_check, gemm_workspace_check (the runner already ran
#      gemm_device_check and the card on this commit)
#   4. lean B4 (ref, head); replays on head held to the H100 chain
#      (101..103 from ckpt 100, 1999..2000 from ckpt 1998); the timed
#      itemization; an nsys trace
#   5. python -m mojolearn verify over the 201 non-par lanes that reach
#      gemm/checks/gemm_identical.mojo, chunks of 25, against the shipped
#      reference table
# Then holds until /root/nv_step_done.
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
say "leg4 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
{ nvidia-smi; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv; } > "$OUT/gpu.txt" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
( apt-get update -qq > /dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nsight-systems-12-4 > "$OUT/nsys_install.log" 2>&1
  python3 -m pip install -q nvidia-cuda-nvcc-cu12 > "$OUT/ptxas_install.log" 2>&1
  P=$(python3 -c 'import nvidia.cuda_nvcc, os; print(os.path.join(list(nvidia.cuda_nvcc.__path__)[0], "bin", "ptxas"))' 2>/dev/null)
  [ -x "$P" ] && ln -sf "$P" /root/ptxas_new ) &
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
( $S bind head ) & ( $S bind ref MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1 MOJOLEARN_GEMM_NO_KPACK_NARROW=1 MOJOLEARN_ATTN_NO_LAUNCH_BOUND=1 MOJOLEARN_EMB_SERIAL_RUN_BEGIN=1 ) &
( $S bind timers MOJOLEARN_STEP_PHASE_TIMERS=1 MOJOLEARN_ATTN_PHASE_TIMERS=1 ) &
wait
cp "$BIN/byte_lm.head.so" python/mojolearn/identical/_mojolearn_byte_lm.so

AB_REF=ref $S ab ref MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1 MOJOLEARN_GEMM_NO_KPACK_NARROW=1
AB_REF=ref $S ab head
AB_REF=ref $S ab sabotage MOJOLEARN_GEMM_SABOTAGE_ADMIT_ALWAYS=1
AB_REF=ref $S ab narrowonly MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1
$S ptx head
for c in gemm_backward_check; do
    t0=$(date +%s)
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
        -I . gemm/checks/$c.mojo > "$OUT/$c.log" 2>&1
    say "$c exit=$? secs=$(( $(date +%s) - t0 )): $(tail -1 "$OUT/$c.log" | cut -c1-200)"
done
t0=$(date +%s)
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_STEP_PHASE_TIMERS=1 --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
    -I . gemm/checks/gemm_workspace_check.mojo > "$OUT/gemm_workspace_check.log" 2>&1
say "gemm_workspace_check exit=$? secs=$(( $(date +%s) - t0 )): $(tail -1 "$OUT/gemm_workspace_check.log" | cut -c1-200)"

while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
$S lean ref 3
$S lean head 3
$S replay head ckpt_00000100.blm A-1.chain.partial.jsonl 3
$S replay head ckpt_00001998.blm A-2.chain.jsonl 2
$S item timers
$S nsys head

cp "$BIN/byte_lm.head.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sh tools/nvidia_step_time/verify_lanes.sh > "$OUT/verify_run.log" 2>&1
say "verify: $(tail -1 "$OUT/verify/summary.txt" 2>/dev/null)"

# the AMD column still compiles the shared sources (build only; its bits are owed)
t0=$(date +%s)
mkdir -p "$BIN/out_amd"
MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_amd" \
    sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.amd_gfx942.log" 2>&1
say "byte_lm gfx942 (AMD column) build exit=$? secs=$(( $(date +%s) - t0 ))"

touch /root/nv_step_ready
say "leg4 scripted part done; holding"
while [ ! -e /root/nv_step_done ]; do sleep 20; done
exit 0
