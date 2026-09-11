#!/bin/sh
# tools/byte_lm_host_leg.sh. THE CPU INFERENCE LEG BODY (DEVIATIONS 2610-2614).
# Runs on a CPU-only droplet through tools/do_extra_leg.sh cpu-intel|cpu-amd
# with MOJOLEARN_DO_EXTRA_PAYLOAD carrying the capture subset. cwd is
# /root/mojolearn, the leg directory is /root/gemm_leg_out, pixi is on PATH.
#
#   1. production build, no accelerator target, x86-64-v3 on x86_64
#   2. gate: every held-out loss byte plus every 8th training step's, against
#      the retained Metal/CUDA/HIP capture
#   3. sabotage build (DEVIATION 2612) and the same gate, which must FAIL
#
# POSIX sh, `set -u` and not `set -e`: a red gate is a result and its logs come
# home. Exit 0 only when the production gate passes AND the sabotage gate
# found a mismatch.
set -u
OUT=/root/gemm_leg_out
mkdir -p "$OUT"
CAPTURE=bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple
STEPS="${MOJOLEARN_BYTE_LM_HOST_STEPS:-every:8}"
# The box has no .git; the runner already wrote the pinned commit to leg.txt.
MOJOLEARN_GATE_COMMIT=$(sed -n 's/^commit=//p' "$OUT/leg.txt" | head -1)
export MOJOLEARN_GATE_COMMIT

lscpu > "$OUT/lscpu.txt" 2>&1 || true
grep -m1 -E '^(flags|Features)' /proc/cpuinfo > "$OUT/cpu_flags.txt" 2>&1 || true
nproc > "$OUT/nproc.txt" 2>&1 || true

if ! command -v cc > /dev/null 2>&1; then
    { DEBIAN_FRONTEND=noninteractive apt-get update -qq \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential; } > "$OUT/apt.log" 2>&1
    echo "apt_build_essential_exit=$?" >> "$OUT/leg.txt"
fi

# The production binding goes to its installed path, python/mojolearn/host/,
# because that file is how `_backend` recognizes a CPU-only install
# (DEVIATION 2615). No GPU set is built on this box, so the gate's
# `import mojolearn` takes exactly the path a CPU-only user's does.
sh bindings/build_byte_lm_host.sh > "$OUT/build_prod.log" 2>&1
prod_build=$?
echo "build_prod_exit=$prod_build" >> "$OUT/leg.txt"

prod_gate=9
if [ "$prod_build" = 0 ]; then
    cp python/mojolearn/host/_mojolearn_byte_lm_host.so "$OUT/_mojolearn_byte_lm_host.prod.so"
    python3 tools/byte_lm_host_gate.py --capture "$CAPTURE" --steps "$STEPS" \
        --report "$OUT/gate_prod.json" > "$OUT/gate_prod.log" 2>&1
    prod_gate=$?
fi
echo "gate_prod_exit=$prod_gate" >> "$OUT/leg.txt"

MOJOLEARN_BYTE_LM_HOST_OUTDIR=/root/host-sab \
    MOJOLEARN_BUILD_EXTRA_DEFINES='-D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1' \
    sh bindings/build_byte_lm_host.sh > "$OUT/build_sab.log" 2>&1
sab_build=$?
echo "build_sab_exit=$sab_build" >> "$OUT/leg.txt"

sab_gate=9
if [ "$sab_build" = 0 ]; then
    MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1 \
        MOJOLEARN_BYTE_LM_HOST_BINARY=/root/host-sab/_mojolearn_byte_lm_host.so \
        python3 tools/byte_lm_host_gate.py --capture "$CAPTURE" --steps "$STEPS" \
        --expect-mismatch --report "$OUT/gate_sab.json" > "$OUT/gate_sab.log" 2>&1
    sab_gate=$?
fi
echo "gate_sab_exit=$sab_gate" >> "$OUT/leg.txt"

[ "$prod_gate" = 0 ] && [ "$sab_gate" = 0 ]
