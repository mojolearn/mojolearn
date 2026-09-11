#!/bin/sh
# tools/step_glue_leg.sh -- DEVIATIONS 2645 to 2648: the byte LM step glue
# arms against the shipped step, lean step on both corpora in one pod, every
# step witness compared, timers for attribution, a verdict per arm
# (ENGINEERING_RULES section 9). VENDOR-AGNOSTIC body, written for the
# NVIDIA H100 column: it runs ON THE BOX as the MOJOLEARN_GEMM_LEG_EXTRA body
# of tools/gemm_remote_leg.sh (RunPod), from /root/mojolearn with pixi on
# PATH; everything under /root/gemm_leg_out/step-glue/ comes home with the
# leg's fetch. Brief: docs/lanes/BRIEF_step_glue_2026-09-11.md.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/step_glue_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-step-glue \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" --local-card <card>
#
# tools/gemm_remote_leg.sh exports neither the target column nor the GPU arch
# to this body, so the body sets both itself (the step breakdown leg's
# logic): the vendor from the box, MOJOLEARN_TARGET_COLUMN from the vendor
# (a build without it is not the shipped build), the arch from nvidia-smi
# compute_cap (9.0 is spelled sm_90a, DEVIATION 2293). Knobs for a hand run:
#   MOJOLEARN_STEP_GLUE_LEG_ARMS         comma list of glue arm names (default
#                                        optskip_noshadow,rows16,rows8,
#                                        optskip_noshadow_rows16,optskip_noshadow_rows8)
#   MOJOLEARN_STEP_GLUE_LEG_TIMING_ARM   the arm timed beside `shipped` for
#                                        attribution (default optskip_noshadow_rows16)
#   MOJOLEARN_STEP_GLUE_LEG_TIMED_STEPS  timed steps per run (default 3, after
#                                        one warmup step that includes setup)
#   MOJOLEARN_STEP_GLUE_LEG_DEADLINE     seconds per build-free item (default 300)
#   MOJOLEARN_STEP_GLUE_LEG_SKIP_TIMERS  1 skips the timers build and its runs
#   MOJOLEARN_STEP_GLUE_LEG_SKIP_CHECK   1 skips the on-box check build and run
#   MOJOLEARN_COMPILE_JOBS               compiler workers (default 2)
#
# PHASES, each with its own exit code in status.tsv; a later phase runs even
# when an earlier one fails, because a red phase is a finding:
#
#   build-binding-base       bindings/build.sh (the base package import needs it).
#   build-byte-lm-shipped    bindings/build_byte_lm.sh as shipped, into bin/shipped.
#   build-byte-lm-glue       the same with -D MOJOLEARN_STEP_GLUE_TRIAL=1, into bin/glue.
#   build-byte-lm-gluetimers the trial define plus -D MOJOLEARN_STEP_PHASE_TIMERS=1
#                            -D MOJOLEARN_ATTN_PHASE_TIMERS=1, into bin/gluetimers.
#   build-glue-check, glue-check
#                            training/checks/step_glue_check.mojo with the trial
#                            define, then run: names, the RMSNorm row geometries
#                            forward and backward, the update kernels, the step
#                            end to end, the refusal and the rollback, with reach
#                            by sabotage (brief section 6). Small shapes, bits only.
#   corpus-*                 enwik8 and the Pile GitHub component (section 9).
#   lean-shipped-<corpus>    the SHIPPED binding: the witnesses every arm must equal.
#   lean-glue-shipped-<corpus>
#                            the trial binding with MOJOLEARN_STEP_GLUE_ARM=shipped:
#                            THE REFERENCE of every verdict (same binary as the arms).
#   lean-glue-<arm>-<corpus> the trial binding under each arm.
#   lean-glue-shipped2-enwik8 the reference again, last (pod drift).
#   timing-<arm>-enwik8      the timers binding, `shipped` and the timing arm,
#                            1 warmup then N steps under MOJOLEARN_TRANSFORMER_TIMING=1.
#   summary                  summary.tsv (medians, ratios, witnesses, verdicts) and
#                            attribution.tsv (the brief's section 1 rows, both arms).
#
# Every lean run is tools/lm_step_memory_probe.py --target --resident-lean
# --witness-every-step --steps 1+N. Only same-pod numbers compare;
# nvidia_smi.txt and host.txt name the pod. OUR IDENTICAL step against itself;
# no opponent runs.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_STEP_GLUE_LEG_OUT:-/root/gemm_leg_out/step-glue}
ARMS=${MOJOLEARN_STEP_GLUE_LEG_ARMS:-optskip_noshadow,rows16,rows8,optskip_noshadow_rows16,optskip_noshadow_rows8}
TIMING_ARM=${MOJOLEARN_STEP_GLUE_LEG_TIMING_ARM:-optskip_noshadow_rows16}
TIMED=${MOJOLEARN_STEP_GLUE_LEG_TIMED_STEPS:-3}
DEADLINE=${MOJOLEARN_STEP_GLUE_LEG_DEADLINE:-300}
JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
mkdir -p "$OUT/bin/shipped" "$OUT/bin/glue" "$OUT/bin/gluetimers"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical
unset MOJOLEARN_TRANSFORMER_TIMING MOJOLEARN_ATTN_ARM MOJOLEARN_GEMM_ARM \
    MOJOLEARN_STEP_GLUE_ARM MOJOLEARN_STEP_GLUE_ARM_SABOTAGE

case "$TIMED" in
    ''|*[!0-9]*|0) echo "MOJOLEARN_STEP_GLUE_LEG_TIMED_STEPS=$TIMED is not a positive integer; nothing run" > "$OUT/gate.txt"; exit 9 ;;
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
    echo "deviations=2645-2648"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT timed_steps=$TIMED lean_steps=$STEPS_LEAN deadline=$DEADLINE jobs=$JOBS"
    echo "vendor=$VENDOR gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN"
    echo "arms=$ARMS timing_arm=$TIMING_ARM"
    echo "glue_define=-D MOJOLEARN_STEP_GLUE_TRIAL=1 timers_defines=-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1"
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
GLUE="-D MOJOLEARN_STEP_GLUE_TRIAL=1"
TIMERS="-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1"
rm -f "$BIND/_mojolearn.so" "$BIND/_mojolearn_byte_lm.so" \
    "$OUT/bin/shipped/_mojolearn_byte_lm.so" "$OUT/bin/glue/_mojolearn_byte_lm.so" \
    "$OUT/bin/gluetimers/_mojolearn_byte_lm.so"
run build-binding-base env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
    MOJOLEARN_COMPILE_JOBS="$JOBS" sh bindings/build.sh
run build-byte-lm-shipped env MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_BYTE_LM_OUTDIR="$OUT/bin/shipped" sh bindings/build_byte_lm.sh
run build-byte-lm-glue env MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_BYTE_LM_OUTDIR="$OUT/bin/glue" MOJOLEARN_BUILD_EXTRA_DEFINES="$GLUE" \
    sh bindings/build_byte_lm.sh
if [ "${MOJOLEARN_STEP_GLUE_LEG_SKIP_TIMERS:-0}" != "1" ]; then
    run build-byte-lm-gluetimers env MOJOLEARN_NUMERIC_MODE=identical \
        MOJOLEARN_BYTE_LM_OUTDIR="$OUT/bin/gluetimers" MOJOLEARN_BUILD_EXTRA_DEFINES="$GLUE $TIMERS" \
        sh bindings/build_byte_lm.sh
fi
for f in "$BIND/_mojolearn.so" "$OUT/bin/shipped/_mojolearn_byte_lm.so" \
    "$OUT/bin/glue/_mojolearn_byte_lm.so" "$OUT/bin/gluetimers/_mojolearn_byte_lm.so"; do
    [ -f "$f" ] && sha256sum "$f" >> "$OUT/bindings.sha256"
done
if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
fi

# ---- the small-shape identity check on this box -----------------------------
if [ "${MOJOLEARN_STEP_GLUE_LEG_SKIP_CHECK:-0}" != "1" ]; then
    # shellcheck disable=SC2086
    run build-glue-check pixi run mojo build -j "$JOBS" -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 $GLUE \
        training/checks/step_glue_check.mojo -o "$OUT/bin/glue-check"
    if [ -x "$OUT/bin/glue-check" ]; then
        run glue-check timeout "$DEADLINE" "$OUT/bin/glue-check"
        grep -h '^REACH\|^FAIL\|^VACUOUS\|^step_glue_check:' "$OUT/glue-check.log" \
            | sed 's/^/glue_check: /' >> "$OUT/gate.txt"
    fi
fi

install_binding() {  # <shipped|glue|gluetimers>: the one byte LM binding the next probe imports
    rm -f "$BIND/_mojolearn_byte_lm.so"
    [ -f "$OUT/bin/$1/_mojolearn_byte_lm.so" ] || return 1
    cp "$OUT/bin/$1/_mojolearn_byte_lm.so" "$BIND/_mojolearn_byte_lm.so" || return 1
    echo "$(date -u +%H:%M:%SZ) installed=$1" >> "$OUT/installs.txt"
}

probe() {  # <run name> <glue arm, or - for none> <corpus path> <probe args...>
    _pname=$1
    _parm=$2
    _pcorpus=$3
    shift 3
    if [ "$_parm" = - ]; then
        PYTHONPATH="$ROOT/python:$ROOT" run "$_pname" timeout "$DEADLINE" pixi run python \
            tools/lm_step_memory_probe.py --out "$OUT/$_pname" --target --resident-lean \
            --witness-every-step --budget-seconds "$DEADLINE" --corpus "$_pcorpus" "$@"
    else
        PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_STEP_GLUE_ARM="$_parm" run "$_pname" \
            timeout "$DEADLINE" pixi run python \
            tools/lm_step_memory_probe.py --out "$OUT/$_pname" --target --resident-lean \
            --witness-every-step --budget-seconds "$DEADLINE" --corpus "$_pcorpus" "$@"
    fi
}

# ---- the two corpora (ENGINEERING_RULES section 9) --------------------------
CORPORA=""
run corpus-enwik8 sh tools/fetch_corpus_enwik8.sh \
    && CORPORA="$CORPORA enwik8=training/corpus/enwik8/input.txt"
run corpus-pile-github sh tools/fetch_corpus_pile_github.sh \
    && CORPORA="$CORPORA pilegithub=training/corpus/pile_github/input.txt"
echo "corpora=$CORPORA" >> "$OUT/gate.txt"

# ---- per corpus: shipped, the reference, every arm --------------------------
for spec in $CORPORA; do
    name=${spec%%=*}
    path=${spec#*=}
    if install_binding shipped; then
        probe "lean-shipped-$name" - "$path" --steps "$STEPS_LEAN"
    else
        printf 'lean-shipped-%s\t9\t0s\n' "$name" >> "$OUT/status.tsv"
        rc=1
    fi
    if install_binding glue; then
        probe "lean-glue-shipped-$name" shipped "$path" --steps "$STEPS_LEAN"
        for arm in $(echo "$ARMS" | tr ',' ' '); do
            probe "lean-glue-$arm-$name" "$arm" "$path" --steps "$STEPS_LEAN"
        done
    else
        printf 'lean-glue-%s\t9\t0s\n' "$name" >> "$OUT/status.tsv"
        rc=1
    fi
done

# ---- the reference again, last: how far the pod drifted ---------------------
for spec in $CORPORA; do
    name=${spec%%=*}
    path=${spec#*=}
    [ "$name" = enwik8 ] || continue
    install_binding glue && probe "lean-glue-shipped2-$name" shipped "$path" --steps "$STEPS_LEAN"
done

# ---- attribution: the timers binding, shipped and the timing arm ------------
if [ "${MOJOLEARN_STEP_GLUE_LEG_SKIP_TIMERS:-0}" != "1" ]; then
    for spec in $CORPORA; do
        name=${spec%%=*}
        path=${spec#*=}
        [ "$name" = enwik8 ] || continue
        if install_binding gluetimers; then
            for arm in shipped "$TIMING_ARM"; do
                probe "timing-$arm-$name" "$arm" "$path" --steps 1 \
                    --component-timing --component-timing-steps "$TIMED"
            done
        fi
    done
fi
rm -f "$BIND/_mojolearn_byte_lm.so"

# ---- the summary: medians, ratios, witnesses, verdicts, attribution ---------
run summary pixi run python - "$OUT" "$ARMS" "$TIMING_ARM" <<'PY'
import json, math, pathlib, sys

out = pathlib.Path(sys.argv[1])
arms = [a for a in sys.argv[2].split(',') if a]
timing_arm = sys.argv[3]
corpora = ['enwik8', 'pilegithub']


def load(name):
    r = out / name / 'result.json'
    return json.loads(r.read_text()) if r.exists() else None


def witnesses(j):
    if not j:
        return None
    return [json.dumps(w.get('sha256'), sort_keys=True) for w in (j.get('step_witnesses') or [])]


rows = ['run\tcorpus\tarm_requested\tarm_ran\ttrial_build\tsteady_median_s\tratio_vs_glue_shipped'
        '\tratio_vs_shipped_build\twitness_steps\twitnesses_equal_shipped']
verdicts = []
for arm in ['shipped'] + arms:
    ratios = {}
    all_equal = True
    all_named = True
    complete = True
    for corpus in corpora:
        ship = load(f'lean-shipped-{corpus}')
        ref = load(f'lean-glue-shipped-{corpus}')
        run_name = f'lean-glue-{arm}-{corpus}'
        run = load(run_name)
        if run is None or ref is None or ship is None:
            rows.append(f'{run_name}\t{corpus}\t{arm}\tMISSING')
            complete = False
            continue
        med = run.get('steady_median_seconds')
        ref_med = ref.get('steady_median_seconds')
        ship_med = ship.get('steady_median_seconds')
        w_run = witnesses(run)
        w_ship = witnesses(ship)
        equal = bool(w_run) and w_run == w_ship
        ran = run.get('step_glue_arm')
        all_equal = all_equal and equal
        all_named = all_named and ran == arm
        if med and ref_med:
            ratios[corpus] = med / ref_med
        else:
            complete = False
        r_ref = f'{med / ref_med:.4f}' if med and ref_med else 'NA'
        r_ship = f'{med / ship_med:.4f}' if med and ship_med else 'NA'
        rows.append(f'{run_name}\t{corpus}\t{arm}\t{ran}\t{run.get("step_glue_trial_build")}\t{med}'
                    f'\t{r_ref}\t{r_ship}\t{len(w_run or [])}\t{equal}')
    if arm == 'shipped':
        continue
    if complete and len(ratios) == 2:
        g = math.sqrt(ratios['enwik8'] * ratios['pilegithub'])
        flip = g < 1 and all_equal and all_named
        verdicts.append(f'verdict {arm} {"FLIP" if flip else "NO FLIP"} geomean={g:.4f} '
                        f'enwik8={ratios["enwik8"]:.4f} pilegithub={ratios["pilegithub"]:.4f} '
                        f'witnesses_equal_shipped={all_equal} arm_named={all_named} '
                        f'(vs lean-glue-shipped, same pod)')
    else:
        verdicts.append(f'verdict {arm} NOT RUN (a corpus or the reference is missing)')

drift_a = load('lean-glue-shipped-enwik8')
drift_b = load('lean-glue-shipped2-enwik8')
if drift_a and drift_b and drift_a.get('steady_median_seconds') and drift_b.get('steady_median_seconds'):
    rows.append(f'drift\tenwik8\tshipped\tshipped\t\t{drift_b["steady_median_seconds"]}'
                f'\t{drift_b["steady_median_seconds"] / drift_a["steady_median_seconds"]:.4f}\t\t\t'
                f'{witnesses(drift_b) == witnesses(load("lean-shipped-enwik8"))}')
(out / 'summary.tsv').write_text('\n'.join(rows + verdicts) + '\n')

# Attribution: the brief's section 1 rows from the two timing runs.
keys = ['envelope.native_call', 'fwd.norm1', 'fwd.norm2', 'grad.norm1_kernels', 'grad.norm2_kernels',
        'step.validate_grads_scan', 'step.shadow_copy', 'step.opt_refuse_scan', 'step.optimizer',
        'step.validate_after_scan', 'step.embedding_backward', 'step.unpack_weights', 'step.pack_grads']
t_ship = load('timing-shipped-enwik8') or {}
t_arm = load(f'timing-{timing_arm}-enwik8') or {}
ms_ship = t_ship.get('component_timing_ms') or {}
ms_arm = t_arm.get('component_timing_ms') or {}
lines = [f'component\tshipped_ms\t{timing_arm}_ms\tdelta_ms']
for k in keys:
    a = ms_ship.get(k)
    b = ms_arm.get(k)
    d = f'{b - a:.3f}' if isinstance(a, (int, float)) and isinstance(b, (int, float)) else 'NA'
    lines.append(f'{k}\t{a}\t{b}\t{d}')
lines.append(f'timing_witnesses_equal\t{t_ship.get("component_timing_witnesses") == t_arm.get("component_timing_witnesses")}')
(out / 'attribution.tsv').write_text('\n'.join(lines) + '\n')
print('\n'.join(rows + verdicts))
print('\n'.join(lines))
PY

gpu_snapshot "$OUT/gpu_after.txt"
# Binaries stay on the box: they are not evidence and the blob fences refuse them.
rm -rf "${OUT:?}/bin"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
[ -f "$OUT/summary.tsv" ] && cat "$OUT/summary.tsv"
[ -f "$OUT/attribution.tsv" ] && cat "$OUT/attribution.tsv"
cat "$OUT/status.tsv"
exit "$rc"
