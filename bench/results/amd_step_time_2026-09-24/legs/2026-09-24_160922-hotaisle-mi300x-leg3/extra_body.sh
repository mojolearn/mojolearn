#!/bin/sh
# tools/amd_step_time_leg3.sh -- lane/amd-step-time (2026-09-24), the third
# AMD leg: the identity checks of the changed GEMM path on the branch (launch
# bound + AMD leaf split), scripted. The runner's gates (gemm_device_check,
# the card) run before this body when the leg is started without
# --skip-gates.
#   1. every device binding built from this commit (four at a time)
#   2. gemm/checks/gemm_backward_check.mojo and gemm_workspace_check.mojo
#   3. python -m mojolearn verify over the 201 non-par lanes that reach
#      gemm/checks/gemm_identical.mojo (tools/lane_select.py
#      --lanes-for-paths; the list is bench/results/amd_step_time_2026-09-24/
#      lanes_gemm_nonpar.txt), against the shipped reference table
# Then holds until /root/amd_step_done.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg3 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; free -g; } > "$OUT/host.txt" 2>&1
rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "build base exit=$? secs=$(( $(date +%s) - t0 ))"
n=0
for f in byte_lm estimators linalg training transformer embedding kernel_methods gp svm mixture metrics preprocessing resample solver rf gbdt trees hdbscan ivf mamba arima tsa; do
    ( t0=$(date +%s); MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=2 sh bindings/build_$f.sh > "$OUT/builds/$f.log" 2>&1
      say "build $f exit=$? secs=$(( $(date +%s) - t0 ))" ) &
    n=$((n + 1))
    if [ $((n % 4)) -eq 0 ]; then wait; fi
done
wait

for c in gemm_backward_check gemm_workspace_check; do
    t0=$(date +%s)
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
        -I . gemm/checks/$c.mojo > "$OUT/$c.log" 2>&1
    say "$c exit=$? secs=$(( $(date +%s) - t0 )): $(tail -1 "$OUT/$c.log" | cut -c1-200)"
done

LANES=$(tr '\n' ',' < bench/results/amd_step_time_2026-09-24/lanes_gemm_nonpar.txt | sed 's/,$//')
t0=$(date +%s)
pixi run python -m mojolearn verify --lanes "$LANES" --json-out "$OUT/verify-all.json" > "$OUT/verify-all.log" 2>&1
say "verify 201 lanes exit=$? secs=$(( $(date +%s) - t0 )): $(grep RESULT "$OUT/verify-all.log" | tail -1 | cut -c1-300)"
gzip -9 -f "$OUT/verify-all.json"

touch /root/amd_step_ready
say "leg3 scripted part done; holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
