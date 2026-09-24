#!/bin/sh
# tools/amd_step_time_body.sh -- lane/amd-step-time (2026-09-24). The body of
# ONE guarded AMD leg (tools/do_extra_leg.sh amd, or tools/hotaisle_leg.sh)
# that prepares a working box for the step-time lane and then HOLDS it for
# the lane's own ssh session until /root/amd_step_done appears or the leg's
# work bound ends the body. The runner's caps, dead-men and verified delete
# are unchanged; this body only decides what the box does meanwhile.
#
# On start (each step logged to status.txt, a failure is a result):
#   1. the box: rocminfo agent name, rocm-smi product, ROCm version,
#      rocprofv3/rocprof availability, uname, CPU;
#   2. the base binding and the byte LM binding built from this commit
#      (MOJOLEARN_TARGET_COLUMN=amd, IDENTICAL), the byte LM .so kept as
#      bin/byte_lm.branch.so;
#   3. gemm/checks/amd_excp_probe.mojo built and run (probe.log): the device
#      facts the EXCP seam rests on, before anything else relies on them;
#   4. /root/amd_step_ready is written; then the body waits.
# Everything under /root/gemm_leg_out comes home with the leg's fetch.
#
# POSIX sh (the runner ships it as /root/gemm_leg_extra.sh).
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/bin"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) archs=$MOJOLEARN_GPU_ARCHS"

{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
rocm-smi --showproductname --showuniqueid --showclocks --showmeminfo vram > "$OUT/gpu.txt" 2>&1
rocminfo 2>/dev/null | grep -E 'Marketing Name|Name: +gfx|Compute Unit|Max Clock' > "$OUT/rocminfo.txt" 2>&1
{ cat /opt/rocm/.info/version 2>/dev/null; ls -d /opt/rocm* 2>/dev/null; command -v rocprofv3; command -v rocprof; command -v omniperf; } > "$OUT/rocm.txt" 2>&1
say "gpu: $(grep -m1 -i 'card series\|product name' "$OUT/gpu.txt" | cut -c1-120)"

_t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1; _rc=$?
say "build base exit=$_rc secs=$(( $(date +%s) - _t0 ))"
rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
_t0=$(date +%s)
MOJOLEARN_BUILD_EXTRA_DEFINES="" sh bindings/build_byte_lm.sh > "$OUT/build_byte_lm.log" 2>&1; _rc=$?
say "build byte_lm (branch) exit=$_rc secs=$(( $(date +%s) - _t0 ))"
[ "$_rc" -eq 0 ] && cp python/mojolearn/identical/_mojolearn_byte_lm.so "$OUT/bin/byte_lm.branch.so"
pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1 || { pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1; say "numpy pip exit=$?"; }

_t0=$(date +%s)
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
    -I . gemm/checks/amd_excp_probe.mojo -o "$OUT/bin/amd_excp_probe" > "$OUT/probe_build.log" 2>&1; _rc=$?
say "probe build exit=$_rc secs=$(( $(date +%s) - _t0 ))"
[ "$_rc" -eq 0 ] && { "$OUT/bin/amd_excp_probe" > "$OUT/probe.log" 2>&1; say "probe exit=$?: $(grep EXCP_CHAIN "$OUT/probe.log" | tr '\n' ' ' | cut -c1-300)"; }

touch /root/amd_step_ready
say "ready; holding for the lane's session"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
say "done flag seen; finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
