#!/bin/sh
# tools/nvidia_step_time/leg1.sh -- lane/nvidia-step-time (2026-09-25), the
# first H100 leg: MAP and PROFILE one T3 step on sm_90a from origin/main's
# source (the first NVIDIA timing of the GEMM launch bound), and the same-box
# BEFORE on 0.8.17's GEMM source. Runs as MOJOLEARN_GEMM_LEG_EXTRA after the
# runner's gates. Needs, pushed by the lane over ssh after the pod is up:
#   /root/urls/{tokens.json,ckpt.json,recipe.url}   (tools/amd_step_time_urls.py)
#   /root/amd_in/{A-1.chain.partial.jsonl,A-2.chain.jsonl}
#   /root/base0817.tgz   (git archive 4795b0d54 of the three GEMM-path files
#                         main changed since 0.8.17: checks/kernel_matrix.mojo,
#                         checks/numerics.mojo, gemm/checks/gemm_identical.mojo)
# Steps: host facts; nsys and a current ptxas in the background; base
# bindings; byte LM bindings branch + timers; the 0.8.17 byte LM binding from
# the overlay; GEMM A/B at the T3 shapes; PTX resources of the GEMM kernels;
# replays held to the H100 chain (0.8.17: 101..102; branch: 101..103 and
# 1999..2000); lean B4 steps; the timed itemization; an nsys trace.
# Then holds until /root/nv_step_done (the lane works on over ssh).
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/nv-step-time
mkdir -p "$OUT/builds"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia
: "${MOJOLEARN_GPU_ARCHS:=sm_90a}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
S="sh tools/nvidia_step_time/session.sh"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg1 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
{ nvidia-smi; nvidia-smi -q | grep -E 'Product Name|Driver Version|CUDA Version|VBIOS|Max Clocks' -A0; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.current --format=csv; } > "$OUT/gpu.txt" 2>&1
{ nvcc --version; ls -d /usr/local/cuda*; } > "$OUT/cuda.txt" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

# tools in the background: nsys (apt, the image's CUDA repo) and a ptxas that
# reads the PTX version Mojo emits (pip nvidia-cuda-nvcc)
( t0=$(date +%s)
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems-cli \
      || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nsight-systems-12-4 \
      || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems-2023.4.4; } > "$OUT/nsys_install.log" 2>&1
  say "nsys install: $(command -v nsys || ls /opt/nvidia/nsight-systems*/bin/nsys 2>/dev/null | head -1) secs=$(( $(date +%s) - t0 ))"
  python3 -m pip install -q nvidia-cuda-nvcc-cu12 > "$OUT/ptxas_install.log" 2>&1
  P=$(python3 -c 'import nvidia.cuda_nvcc, os; print(os.path.join(list(nvidia.cuda_nvcc.__path__)[0], "bin", "ptxas"))' 2>/dev/null)
  [ -x "$P" ] && ln -sf "$P" /root/ptxas_new
  say "ptxas: $(/root/ptxas_new --version 2>/dev/null | tail -1)" ) &

pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "build base exit=$? secs=$(( $(date +%s) - t0 ))"

( $S bind branch ) & ( $S bind timers MOJOLEARN_STEP_PHASE_TIMERS=1 MOJOLEARN_ATTN_PHASE_TIMERS=1 ) &
AB_REF=branch $S ab branch
$S ptx branch
$S ptx cap1024 MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1
wait

# the 0.8.17 GEMM path, built from an overlay of its three files, then restored
while [ ! -s /root/base0817.tgz ]; do sleep 10; done
mkdir -p /root/main_keep && tar cf /root/main_keep/files.tar checks/kernel_matrix.mojo checks/numerics.mojo gemm/checks/gemm_identical.mojo
tar xzf /root/base0817.tgz -C "$ROOT"
sha256sum checks/kernel_matrix.mojo checks/numerics.mojo gemm/checks/gemm_identical.mojo > "$OUT/base0817.files.sha256"
$S bind base0817
tar xf /root/main_keep/files.tar -C "$ROOT"
sha256sum checks/kernel_matrix.mojo checks/numerics.mojo gemm/checks/gemm_identical.mojo > "$OUT/main.files.sha256"
say "overlay restored"

while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
$S lean branch 3
$S lean base0817 3
$S replay base0817 ckpt_00000100.blm A-1.chain.partial.jsonl 2
$S replay branch ckpt_00000100.blm A-1.chain.partial.jsonl 3
$S replay branch ckpt_00001998.blm A-2.chain.jsonl 2
$S item timers
$S nsys branch

touch /root/nv_step_ready
say "leg1 scripted part done; holding"
while [ ! -e /root/nv_step_done ]; do sleep 20; done
exit 0
