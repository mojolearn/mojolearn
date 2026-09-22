#!/bin/sh
# Since the 2026-09-14 flip, arm 1 (scalar accumulators) is the default and the
# second AOT build below carries -D MOJOLEARN_2711_GRAM_STRIDED_DEAD=1, the OLD
# arm 0; the output names ("arm1") are kept so the two 5090 records read alike.
# tools/jacobi_sm120a_probe_leg.sh: the on-box body of the DEVIATION 2711 leg
# (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh nvidia --payload gemm).
# Mac reference:
# bench/results/identity_break/2026-09-14_rtx5090-sm_120a/jacobi_probe.apple-m4.txt.
#
# What it does, on the box, from /root/mojolearn with pixi on PATH:
#   1. resolves MOJOLEARN_GPU_ARCHS from the device exactly as
#      tools/identity_three_columns_leg.sh does (12.0 -> sm_120a), so the
#      probe is compiled for the architecture the refusing bindings were;
#   2. regenerates the `odd` fixture bytes with the SAME generator the
#      identity leg used (tools/identity_break.py fixture("odd"), numpy
#      default_rng(0), 12345 x 17 float32; sha256 595dda3a45cf8a3e...);
#   3. AOT-builds decomposition/checks/jacobi_sm120a_probe.mojo with the
#      binding build's own flags (--target-accelerator, the IDENTICAL define)
#      and runs it, then also runs it JIT (`mojo run`, the box's own device
#      target) so a difference between the two compilations is visible;
#   4. leaves everything under /root/gemm_leg_out/jacobi_probe, which the
#      leg fetches home. Compare `grep ^PROBE` against the Mac reference.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/jacobi_probe
mkdir -p "$OUT"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "commit=$(cat /root/mojolearn/commit.txt 2>/dev/null | head -1 || git -C /root/mojolearn rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/gpu.txt")"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
say "mojo=$(head -1 "$OUT/mojo_version.txt")"
say "MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-<none>}"

if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;; 8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;; 12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs_resolved=$MOJOLEARN_GPU_ARCHS"

# 2. The fixture bytes, from the identity leg's own generator.
cat > "$OUT/gen_odd.py" <<'PYEOF'
import sys, hashlib, numpy as np
sys.path.insert(0, "tools")
from identity_break import fixture
X, _, _ = fixture("odd")
X.tofile(sys.argv[1])
print("shape", X.shape, X.dtype, "sha256", hashlib.sha256(X.tobytes()).hexdigest(), "bytes", X.nbytes)
PYEOF
PYTHONPATH=/root/mojolearn/python pixi run python "$OUT/gen_odd.py" "$OUT/odd_x.f32" > "$OUT/gen_odd.log" 2>&1
say "fixture=$(cat "$OUT/gen_odd.log" | tr '\n' ' ')"

# 3a. AOT, the binding build's flags. `-D MOJOLEARN_NUMERIC_IDENTICAL=1` is
# what bindings/build.sh passes for MOJOLEARN_NUMERIC_MODE=identical.
_t0=$(date +%s)
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-4}" \
    --target-accelerator "$MOJOLEARN_GPU_ARCHS" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
    -o "$OUT/jacobi_probe_aot" decomposition/checks/jacobi_sm120a_probe.mojo > "$OUT/build_aot.log" 2>&1
say "build_aot_exit=$? seconds=$(( $(date +%s) - _t0 ))"
if [ -x "$OUT/jacobi_probe_aot" ]; then
    "$OUT/jacobi_probe_aot" "$OUT/odd_x.f32" > "$OUT/probe_aot.txt" 2> "$OUT/probe_aot.err"
    say "probe_aot_exit=$?"
fi

# 3a'. AOT again with DEVIATION 2711's define, so `compute_covariance` and
# `gemm_tn` themselves (the pca, tsvd and ols cases) run through arm 1 on
# this target, not only the STAGE lines' by-name launch.
_t0=$(date +%s)
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-4}" \
    --target-accelerator "$MOJOLEARN_GPU_ARCHS" -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    -D MOJOLEARN_2711_GRAM_STRIDED_DEAD=1 -I . \
    -o "$OUT/jacobi_probe_aot_arm1" decomposition/checks/jacobi_sm120a_probe.mojo > "$OUT/build_aot_arm1.log" 2>&1
say "build_aot_arm1_exit=$? seconds=$(( $(date +%s) - _t0 ))"
if [ -x "$OUT/jacobi_probe_aot_arm1" ]; then
    "$OUT/jacobi_probe_aot_arm1" "$OUT/odd_x.f32" > "$OUT/probe_aot_arm1.txt" 2> "$OUT/probe_aot_arm1.err"
    say "probe_aot_arm1_exit=$?"
fi

# 3b. JIT, the box's own device target (what `mojo run` picks).
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
    decomposition/checks/jacobi_sm120a_probe.mojo "$OUT/odd_x.f32" > "$OUT/probe_jit.txt" 2> "$OUT/probe_jit.err"
say "probe_jit_exit=$?"

# 4. The one-screen summary the orchestrator reads first.
for f in probe_aot probe_aot_arm1 probe_jit; do
    [ -f "$OUT/$f.txt" ] || continue
    say "== $f"
    grep -E "^PROBE_DEVICE|^MATRIX|^STAGE|^PROBE case=[^ ]+ n=|^PROBE_DONE" "$OUT/$f.txt" >> "$G"
    grep -E "^PROBE case=.* sweep=" "$OUT/$f.txt" | awk '{print $2, $3, $5, $6}' >> "$G"
    grep -i -E "error|abort|fault" "$OUT/$f.err" | head -5 >> "$G"
done
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
