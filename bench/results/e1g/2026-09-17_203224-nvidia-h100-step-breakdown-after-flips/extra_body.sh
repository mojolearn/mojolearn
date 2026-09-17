#!/bin/sh
# tools/step_breakdown_leg.sh -- DEVIATION 2630: a complete per-component
# timing of the SHIPPED byte LM training step, with the untimed lean step on
# the same pod beside it. VENDOR-AGNOSTIC body, written for the NVIDIA H100
# column: it runs ON THE BOX as the MOJOLEARN_GEMM_LEG_EXTRA body of
# tools/gemm_remote_leg.sh (RunPod), from /root/mojolearn with pixi on PATH;
# everything under /root/gemm_leg_out/step-breakdown/ comes home with the
# leg's fetch. Brief: docs/lanes/BRIEF_step_breakdown_2026-09-11.md.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/step_breakdown_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-step-breakdown \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" --local-card <card>
#
# tools/gemm_remote_leg.sh has no extra-env plumbing and exports neither the
# target column nor the GPU arch to this body, so the body sets both itself:
# the vendor from the box (a working nvidia-smi, else /dev/kfd or an AMD tool;
# /dev/dri alone is not AMD evidence), MOJOLEARN_TARGET_COLUMN from the vendor
# (it selects the kernel-matrix rows, the attention default arm and the GEMM
# ksplit default among them, so a build without it is not the shipped build),
# and the arch from nvidia-smi compute_cap (9.0 is spelled sm_90a, DEVIATION
# 2293; AMD must be given MOJOLEARN_GPU_ARCHS). Optional knobs for a hand run:
#   MOJOLEARN_STEP_LEG_TIMED_STEPS  timed steps per run (default 3; one warmup
#                                   step, which includes setup, precedes them)
#   MOJOLEARN_STEP_LEG_DEADLINE     seconds per build-free item (default 300)
#   MOJOLEARN_STEP_LEG_SKIP_REPEAT  1 skips the shipped rerun at the end
#   MOJOLEARN_COMPILE_JOBS          compiler workers (default 2)
#
# PHASES, each with its own exit code in status.tsv; a later phase runs even
# when an earlier one fails, because a red phase is a finding:
#
#   build-binding-base      bindings/build.sh (no extra defines; the base
#                           package import needs it).
#   build-byte-lm-shipped   bindings/build_byte_lm.sh as shipped, into bin/shipped.
#   build-byte-lm-timers    the same with MOJOLEARN_BUILD_EXTRA_DEFINES=
#                           "-D MOJOLEARN_STEP_PHASE_TIMERS=1
#                            -D MOJOLEARN_ATTN_PHASE_TIMERS=1", into bin/timers.
#                           No trial define: both builds run the column's
#                           shipped attention arm and GEMM dispatch.
#   corpus-*                enwik8 and the Pile GitHub component
#                           (ENGINEERING_RULES section 9), fetched on the box
#                           and verified against their manifests.
#   lean-shipped-<corpus>   the shipped binding installed;
#                           tools/lm_step_memory_probe.py --target --resident-lean
#                           --witness-every-step --steps 1+N: the untimed lean
#                           step (steady median over the N steps after the
#                           first) and a witness per step.
#   lean-timers-<corpus>    the timers binding installed, the switch OFF, the
#                           same command: what compiling the timers in costs.
#   timing-<corpus>         the timers binding, --steps 1 --component-timing
#                           --component-timing-steps N --witness-every-step:
#                           one warmup step, then N steps under
#                           MOJOLEARN_TRANSFORMER_TIMING=1, every line folded
#                           into result.json as a per-step median.
#   lean-shipped2-enwik8    the shipped binding again, last (pod drift).
#   summary                 tools/step_breakdown_summary.py OUT: breakdown.tsv
#                           (the tree of every timed interval with its
#                           remainders, categories, GEMM kinds, counts, lean
#                           medians and the timer overhead) and witnesses.tsv
#                           (every run's per-step hashes against the shipped
#                           run's; bits_identical must be True).
#
# nvidia_smi.txt (one csv line) and host.txt (CPU model, cores) name the pod:
# pods differ in speed and only same-pod numbers compare. Everything here is
# OUR IDENTICAL arm against itself; no opponent runs.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_STEP_LEG_OUT:-/root/gemm_leg_out/step-breakdown}
TIMED=${MOJOLEARN_STEP_LEG_TIMED_STEPS:-3}
DEADLINE=${MOJOLEARN_STEP_LEG_DEADLINE:-300}
JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
mkdir -p "$OUT/bin/shipped" "$OUT/bin/timers"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical
unset MOJOLEARN_TRANSFORMER_TIMING MOJOLEARN_ATTN_ARM MOJOLEARN_GEMM_ARM

case "$TIMED" in
    ''|*[!0-9]*|0) echo "MOJOLEARN_STEP_LEG_TIMED_STEPS=$TIMED is not a positive integer; nothing run" > "$OUT/gate.txt"; exit 9 ;;
esac
STEPS_LEAN=$((TIMED + 1))

# ---- the vendor, the column, the arch ------------------------------------
case "${MOJOLEARN_TARGET_COLUMN:-}" in
    amd|nvidia) VENDOR=$MOJOLEARN_TARGET_COLUMN ;;
    *) if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
           VENDOR=nvidia
       elif [ -e /dev/kfd ] || command -v rocm-smi > /dev/null 2>&1 || command -v amd-smi > /dev/null 2>&1; then
           VENDOR=amd
       else
           VENDOR=unknown
       fi ;;
esac
if [ "$VENDOR" = unknown ]; then
    echo "vendor=unknown: no working nvidia-smi, no /dev/kfd, no rocm-smi or amd-smi; nothing run" > "$OUT/gate.txt"
    exit 9
fi
export MOJOLEARN_TARGET_COLUMN="$VENDOR"

if [ "$VENDOR" = nvidia ]; then
    # MAX's bundled CUDA 13 assembler needs driver 580. Older-driver pods use
    # their installed assembler at BOTH build and runtime.
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] && [ "$VENDOR" = nvidia ]; then
    cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    case "$cap" in
        9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
        [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
        *) MOJOLEARN_GPU_ARCHS="" ;;
    esac
fi
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    echo "vendor=$VENDOR gpu_archs=MISSING: one mojo build is one GPU arch; set MOJOLEARN_GPU_ARCHS; nothing run" > "$OUT/gate.txt"
    exit 9
fi
export MOJOLEARN_GPU_ARCHS

rc=0
run() {
    # run <name> <command...>: log to $OUT/<name>.log, record the exit code.
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    printf '%s\t%s\t%ss\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    [ "$_code" -eq 0 ] || rc=1
    return "$_code"
}

gpu_snapshot() {  # <file>
    if [ "$VENDOR" = nvidia ]; then
        nvidia-smi > "$1" 2>&1
    else
        { rocm-smi --showproductname --showdriverversion --showuse --showmemuse --showtemp 2>&1 \
            || echo "rocm-smi did not answer"; } > "$1"
    fi
}

{
    echo "deviation=2630"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT timed_steps=$TIMED lean_steps=$STEPS_LEAN deadline=$DEADLINE jobs=$JOBS"
    echo "vendor=$VENDOR gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN"
    echo "timers_defines=-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1 (no trial define)"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor|provider|size)=' /root/gemm_leg_out/leg.txt
    git -C "$ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/checkout_head=/'
} > "$OUT/gate.txt"
if [ "$VENDOR" = nvidia ]; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current,power.limit,temperature.gpu,clocks.sm,uuid \
        --format=csv,noheader > "$OUT/nvidia_smi.txt" 2>&1
else
    rocm-smi --showproductname --showdriverversion > "$OUT/nvidia_smi.txt" 2>&1 || echo "rocm-smi did not answer" > "$OUT/nvidia_smi.txt"
fi
{
    grep -m1 'model name' /proc/cpuinfo 2>/dev/null
    echo "nproc=$(nproc 2>/dev/null)"
    uname -r
} > "$OUT/host.txt"
gpu_snapshot "$OUT/gpu_before.txt"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

# ---- the builds ------------------------------------------------------------
BIND=python/mojolearn/identical
TIMERS="-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1"
rm -f "$BIND/_mojolearn.so" "$BIND/_mojolearn_byte_lm.so" \
    "$OUT/bin/shipped/_mojolearn_byte_lm.so" "$OUT/bin/timers/_mojolearn_byte_lm.so"
run build-binding-base env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
    MOJOLEARN_COMPILE_JOBS="$JOBS" sh bindings/build.sh
run build-byte-lm-shipped env MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_BYTE_LM_OUTDIR="$OUT/bin/shipped" sh bindings/build_byte_lm.sh
run build-byte-lm-timers env MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_BYTE_LM_OUTDIR="$OUT/bin/timers" MOJOLEARN_BUILD_EXTRA_DEFINES="$TIMERS" \
    sh bindings/build_byte_lm.sh
for f in "$BIND/_mojolearn.so" "$OUT/bin/shipped/_mojolearn_byte_lm.so" "$OUT/bin/timers/_mojolearn_byte_lm.so"; do
    [ -f "$f" ] && sha256sum "$f" >> "$OUT/bindings.sha256"
done
if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
fi

install_binding() {  # <shipped|timers>: the one byte LM binding the next probe imports
    rm -f "$BIND/_mojolearn_byte_lm.so"
    [ -f "$OUT/bin/$1/_mojolearn_byte_lm.so" ] || return 1
    cp "$OUT/bin/$1/_mojolearn_byte_lm.so" "$BIND/_mojolearn_byte_lm.so" || return 1
    echo "$(date -u +%H:%M:%SZ) installed=$1" >> "$OUT/installs.txt"
}

probe() {  # <run name> <corpus path> <probe args...>
    _pname=$1
    _pcorpus=$2
    shift 2
    PYTHONPATH="$ROOT/python:$ROOT" run "$_pname" timeout "$DEADLINE" pixi run python \
        tools/lm_step_memory_probe.py --out "$OUT/$_pname" --target --resident-lean \
        --witness-every-step --budget-seconds "$DEADLINE" --corpus "$_pcorpus" "$@"
}

# ---- the two corpora (section 9) -------------------------------------------
CORPORA=""
run corpus-enwik8 sh tools/fetch_corpus_enwik8.sh \
    && CORPORA="$CORPORA enwik8=training/corpus/enwik8/input.txt"
run corpus-pile-github sh tools/fetch_corpus_pile_github.sh \
    && CORPORA="$CORPORA pilegithub=training/corpus/pile_github/input.txt"
echo "corpora=$CORPORA" >> "$OUT/gate.txt"

# ---- per corpus: shipped lean, timers lean (switch off), timed steps -------
for spec in $CORPORA; do
    name=${spec%%=*}
    path=${spec#*=}
    if install_binding shipped; then
        probe "lean-shipped-$name" "$path" --steps "$STEPS_LEAN"
    else
        printf 'lean-shipped-%s\t9\t0s\n' "$name" >> "$OUT/status.tsv"
        rc=1
    fi
    if install_binding timers; then
        probe "lean-timers-$name" "$path" --steps "$STEPS_LEAN"
        probe "timing-$name" "$path" --steps 1 --component-timing --component-timing-steps "$TIMED"
    else
        printf 'lean-timers-%s\t9\t0s\ntiming-%s\t9\t0s\n' "$name" "$name" >> "$OUT/status.tsv"
        rc=1
    fi
done

# ---- the shipped build again, last: how far the pod drifted ---------------
if [ "${MOJOLEARN_STEP_LEG_SKIP_REPEAT:-0}" != "1" ]; then
    for spec in $CORPORA; do
        name=${spec%%=*}
        path=${spec#*=}
        [ "$name" = enwik8 ] || continue
        install_binding shipped && probe "lean-shipped2-$name" "$path" --steps "$STEPS_LEAN"
    done
fi
rm -f "$BIND/_mojolearn_byte_lm.so"

# ---- the breakdown and the witnesses --------------------------------------
run summary pixi run python tools/step_breakdown_summary.py "$OUT"
cp "$OUT/summary.log" "$OUT/summary.txt" 2>/dev/null

gpu_snapshot "$OUT/gpu_after.txt"
# Binaries stay on the box: they are not evidence and the blob fences refuse them.
rm -rf "${OUT:?}/bin"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
[ -f "$OUT/summary.txt" ] && cat "$OUT/summary.txt"
cat "$OUT/status.tsv"
exit "$rc"
