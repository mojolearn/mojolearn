#!/bin/sh
# tools/nvidia_step_time/leg2.sh -- lane/nvidia-step-time (2026-09-25), the
# second H100 leg: the GEMM WINDOW ADMISSION (`lib_gemm_window_admit_for`,
# NVIDIA) and the finer staging decomposition. Needs /root/urls and
# /root/amd_in pushed after the pod is up (as leg 1).
#   1. PTX resources: admission on (default) and off
#   2. GEMM A/B at the T3 shapes, all three operand kinds: the reference
#      (-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1, main's kernel), the default
#      (admission), and the sabotage (every window admitted: must DIFFER on
#      the subnormal-forcing kinds)
#   3. the DIAG decomposition at T3 with the no-barrier / no-prefetch /
#      no-store variants
#   4. byte LM bindings: noadm (main's GEMM), adm (default); lean B4 steps
#      of both (witnesses must be equal); replays on adm held to the H100
#      chain (101..103 from ckpt 100, 1999..2000 from ckpt 1998); timed
#      itemization on an adm timers binding
# Then holds until /root/nv_step_done.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/nv-step-time
mkdir -p "$OUT/builds" "$OUT/ab"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia
: "${MOJOLEARN_GPU_ARCHS:=sm_90a}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
S="sh tools/nvidia_step_time/session.sh"
ST="$OUT/session.txt"
BIN=/root/nv_bin; mkdir -p "$BIN"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg2 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv > "$OUT/gpu.txt" 2>&1
( python3 -m pip install -q nvidia-cuda-nvcc-cu12 > "$OUT/ptxas_install.log" 2>&1
  P=$(python3 -c 'import nvidia.cuda_nvcc, os; print(os.path.join(list(nvidia.cuda_nvcc.__path__)[0], "bin", "ptxas"))' 2>/dev/null)
  [ -x "$P" ] && ln -sf "$P" /root/ptxas_new ) &
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "build base exit=$? secs=$(( $(date +%s) - t0 ))"

( $S bind adm ) & ( $S bind noadm MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1 ) & ( $S bind timers MOJOLEARN_STEP_PHASE_TIMERS=1 MOJOLEARN_ATTN_PHASE_TIMERS=1 ) &
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_GEMM_DIAG=1 -D MOJOLEARN_GEMM_STEP_T3=1 \
    --target-accelerator sm_90a -I . bench/gemm_step_diag_main.mojo -o "$BIN/diag_t3" > "$OUT/diag_build.log" 2>&1
say "diag build exit=$?"
wait
AB_REF=noadm $S ab noadm MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1
AB_REF=noadm $S ab adm
AB_REF=noadm $S ab sabotage MOJOLEARN_GEMM_SABOTAGE_ADMIT_ALWAYS=1
$S ptx adm
$S ptx noadm MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1
MOJOLEARN_GEMM_STEP_ROUNDS=3 MOJOLEARN_GEMM_STEP_WARMUPS=1 "$BIN/diag_t3" > "$OUT/diag_t3.log" 2>&1
say "diag: $(grep '^DIAGSTEP' "$OUT/diag_t3.log" | sed 's/ (per-call.*//' | tr '\n' '|')"

while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
$S lean noadm 3
$S lean adm 3
$S replay adm ckpt_00000100.blm A-1.chain.partial.jsonl 3
$S replay adm ckpt_00001998.blm A-2.chain.jsonl 2
$S item timers

touch /root/nv_step_ready
say "leg2 scripted part done; holding"
while [ ! -e /root/nv_step_done ]; do sleep 20; done
exit 0
