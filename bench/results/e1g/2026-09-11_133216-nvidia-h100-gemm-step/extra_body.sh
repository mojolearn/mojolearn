#!/bin/sh
# tools/gemm_step_leg.sh -- DEVIATION 2544, the GEMM step lane's on-box work:
# the arms of DEVIATIONS 2540 and 2541 behind the 2542 hook, checked and
# priced by the 2543 harnesses, then the lean LM step under each arm on both
# benchmark corpora. VENDOR-AGNOSTIC: runs ON THE BOX as the
# MOJOLEARN_GEMM_LEG_EXTRA body of tools/do_extra_leg.sh (DigitalOcean, AMD
# first, ENGINEERING_RULES 10) or of tools/gemm_remote_leg.sh (RunPod, the
# NVIDIA confirmation column), from /root/mojolearn with pixi on PATH;
# everything it writes under /root/gemm_leg_out/gemm-step/ comes home with
# the leg's fetch. Structure mirrors tools/attention_step_leg.sh.
#
# AMD (the deciding column), from a `git worktree add --detach` checkout:
#
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_step_leg.sh \
#   MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,lfold,half,half_ks16,quarter,head,half_head MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=auto" \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-gemm-step \
#   bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
#
# NVIDIA confirmation:
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_step_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-step \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# The vendor comes from MOJOLEARN_TARGET_COLUMN when the runner exports it,
# else from the box (a working nvidia-smi, or /dev/kfd / rocm-smi / amd-smi;
# /dev/dri alone is not AMD evidence); no CUDA command runs on the AMD path;
# the arch comes from MOJOLEARN_GPU_ARCHS (required on AMD, derived from
# nvidia-smi on NVIDIA). Everything here is OUR IDENTICAL arm against OUR
# shipped kernel; no opponent runs.
#
# PHASES, each with its own exit code in status.tsv; a later phase runs even
# when an earlier one fails, because a red phase is a finding:
#
#   build-check, build-price, build-resources
#                 gemm/checks/gemm_step_arms_check.mojo,
#                 bench/gemm_step_price_main.mojo and
#                 bench/gemm_step_resources_main.mojo under IDENTICAL with
#                 -D MOJOLEARN_GEMM_ARM_TRIAL=1.
#   step-check    the gate: every arm geometry bit-equal to the shipped
#                 128x128 plan and to FLAT on ragged and adversarial
#                 controls, reach by sabotage (exactly one moved cell per
#                 block), the selector raising on an unknown arm, then the
#                 twelve LM calls (including the three vocab-sized head
#                 calls) through identical_gemm_into under every arm, bits
#                 and reach per call. Its own deadline (900 s default).
#   resources     an INSTRUMENT: compile_function of the shipped
#                 specialization, the trimmed control and every arm
#                 geometry, printing registers, local, shared, const, max
#                 threads per block and blocks per SM, each geometry in its
#                 own try. Attempted on every vendor; an AMD raise is
#                 recorded, not hidden. ptxas is not used.
#   price-<arm>   one process per arm: per LM call digest equality, then 2
#                 warmups and 7 alternated rounds of shipped against the arm;
#                 BITS, PRICE, TABLE lines and one STEP line weighting each
#                 call by its per-step count. Hashed ordinary operands; a
#                 per-call kernel price, not the flip input (the LM step on
#                 the two corpora is).
#   build-binding-*  the base binding and the byte LM binding with the trial
#                 hook riding MOJOLEARN_BUILD_EXTRA_DEFINES.
#   corpus-*      enwik8 (tools/fetch_corpus_enwik8.sh) and the Pile's GitHub
#                 component (tools/fetch_corpus_pile_github.sh), verified
#                 against their manifests; run names enwik8 and pilegithub.
#   lm-<arm>-<corpus>  tools/lm_step_memory_probe.py --target --resident-lean
#                 --witness-every-step --corpus <corpus>, under
#                 MOJOLEARN_GEMM_ARM=<arm>, bracketed per corpus by
#                 lm-shipped-<corpus> first and lm-shippedclose-<corpus> last.
#   lmtiming-<arm>-<corpus>  only with MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1: the
#                 same with --component-timing --steps 1 (a breakdown).
#   lm_summary.tsv  medians, per-step witnesses, witnesses_equal_baseline
#                 against lm-shipped-<corpus>, each arm's ratio against the
#                 mean of the two shipped brackets, and one verdict line per
#                 arm (ENGINEERING_RULES 9: geometric mean of the two corpus
#                 ratios below 1, and every step witness equal to shipped on
#                 both corpora, else NO FLIP with the reason).
#
# KNOBS (all through MOJOLEARN_DO_EXTRA_ENV on DigitalOcean):
#   MOJOLEARN_GEMM_STEP_LEG_ARMS     price arms, default
#                                    shipped,lfold,half,half_ks16,quarter,head,half_head
#                                    (shipped is the shipped-against-shipped
#                                    noise control)
#   MOJOLEARN_GEMM_STEP_LEG_LM_ARMS  LM probe arms, default auto: the arm
#                                    with the lowest STEP ratio among price
#                                    runs that exited 0 (bits equal)
#   MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS  restricts the check's LM section
#   MOJOLEARN_GEMM_STEP_LEG_ROUNDS / _WARMUPS / _DEADLINE / _CHECK_DEADLINE
#   MOJOLEARN_GEMM_STEP_LEG_LM_STEPS (default 3)
#   MOJOLEARN_GEMM_STEP_LEG_SKIP_RESOURCES=1, MOJOLEARN_GEMM_STEP_LEG_SKIP_LM=1,
#   MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1
# Arm names pass through unchecked beyond their spelling; the native
# selector raises on an unknown one.
#
# A 60-minute lease fits: three harness builds (about 3 min each on a
# 2-worker box), the check (a few minutes, three vocab-sized calls per arm),
# the resources instrument, seven price runs (about 1 min each), two binding
# builds (about 6 min), then six probes (shipped, one arm, shipped close, on
# two corpora, about 1 to 2 min each).
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_GEMM_STEP_LEG_OUT:-/root/gemm_leg_out/gemm-step}
ARMS=${MOJOLEARN_GEMM_STEP_LEG_ARMS:-shipped,lfold,half,half_ks16,quarter,head,half_head}
LM_ARMS=${MOJOLEARN_GEMM_STEP_LEG_LM_ARMS:-auto}
CHECK_ARMS=${MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS:-}
ROUNDS=${MOJOLEARN_GEMM_STEP_LEG_ROUNDS:-7}
WARMUPS=${MOJOLEARN_GEMM_STEP_LEG_WARMUPS:-2}
JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
DEADLINE=${MOJOLEARN_GEMM_STEP_LEG_DEADLINE:-300}
CHECK_DEADLINE=${MOJOLEARN_GEMM_STEP_LEG_CHECK_DEADLINE:-900}
LM_STEPS=${MOJOLEARN_GEMM_STEP_LEG_LM_STEPS:-3}
mkdir -p "$OUT/bin"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical

case "$ARMS,$LM_ARMS,$CHECK_ARMS" in
    *[!A-Za-z0-9_,]*)
        echo "arm lists: letters, digits, _ and , only; nothing run" > "$OUT/gate.txt"
        exit 9 ;;
esac

# THE VENDOR. tools/do_extra_leg.sh exports MOJOLEARN_TARGET_COLUMN (amd or
# nvidia); tools/gemm_remote_leg.sh exports nothing, so the box is read.
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

gpu_snapshot() {  # <file>
    if [ "$VENDOR" = nvidia ]; then
        nvidia-smi --query-gpu=name,driver_version,uuid,clocks.sm,temperature.gpu --format=csv > "$1" 2>&1
    else
        { rocm-smi --showproductname --showdriverversion --showuse --showmemuse --showtemp 2>&1 \
            || echo "rocm-smi did not answer"; } > "$1"
    fi
}

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

# One mojo build is one GPU arch; read it from the leg's environment first.
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
    # Knobs go through `env NAME=value` inside the command, never as
    # assignments in front of this function (their scope differs by shell).
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    printf '%s\t%s\t%ss\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    [ "$_code" -eq 0 ] || rc=1
    return "$_code"
}

{
    echo "deviations=2540-2544"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "arms=$ARMS"
    echo "lm_arms=$LM_ARMS"
    echo "check_arms=${CHECK_ARMS:-all}"
    echo "rounds=$ROUNDS warmups=$WARMUPS deadline=$DEADLINE check_deadline=$CHECK_DEADLINE lm_steps=$LM_STEPS"
    echo "vendor=$VENDOR gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN jobs=$JOBS"
    echo "skip_resources=${MOJOLEARN_GEMM_STEP_LEG_SKIP_RESOURCES:-0} skip_lm=${MOJOLEARN_GEMM_STEP_LEG_SKIP_LM:-0} lmtiming=${MOJOLEARN_GEMM_STEP_LEG_LMTIMING:-0}"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor|provider|size)=' /root/gemm_leg_out/leg.txt
} > "$OUT/gate.txt"
gpu_snapshot "$OUT/gpu_before.txt"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
TRIAL="-D MOJOLEARN_GEMM_ARM_TRIAL=1"

# ---- builds ----------------------------------------------------------------
# shellcheck disable=SC2086
run build-check pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
    gemm/checks/gemm_step_arms_check.mojo -o "$OUT/bin/step-check"
# shellcheck disable=SC2086
run build-price pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
    bench/gemm_step_price_main.mojo -o "$OUT/bin/step-price"
if [ "${MOJOLEARN_GEMM_STEP_LEG_SKIP_RESOURCES:-0}" != "1" ]; then
    # shellcheck disable=SC2086
    run build-resources pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
        bench/gemm_step_resources_main.mojo -o "$OUT/bin/step-resources"
fi

# ---- the gate ----------------------------------------------------------------
if [ -x "$OUT/bin/step-check" ]; then
    run step-check env MOJOLEARN_GEMM_STEP_CHECK_LM=1 MOJOLEARN_GEMM_STEP_CHECK_ARMS="$CHECK_ARMS" \
        MOJOLEARN_GEMM_ARM= MOJOLEARN_GEMM_ARM_SABOTAGE= \
        timeout "$CHECK_DEADLINE" "$OUT/bin/step-check"
fi

# ---- the resources instrument (every vendor; a raise is a finding) ---------
if [ -x "$OUT/bin/step-resources" ]; then
    run resources timeout "$DEADLINE" "$OUT/bin/step-resources"
    grep -h '^GEMM_STEP_RESOURCES' "$OUT/resources.log" > "$OUT/resources_lines.txt" 2>/dev/null
fi

# ---- the per-call price, one arm per process --------------------------------
for arm in $(echo "$ARMS" | tr ',' ' '); do
    [ -x "$OUT/bin/step-price" ] || break
    run "price-$arm" env MOJOLEARN_GEMM_ARM="$arm" MOJOLEARN_GEMM_ARM_SABOTAGE=0 \
        MOJOLEARN_GEMM_STEP_ROUNDS="$ROUNDS" MOJOLEARN_GEMM_STEP_WARMUPS="$WARMUPS" \
        timeout "$DEADLINE" "$OUT/bin/step-price"
done
grep -h '^STEP\|^PRICE\|^TABLE\|^BITS' "$OUT"/price-*.log > "$OUT/price_tables.txt" 2>/dev/null
grep -h '^STEP' "$OUT"/price-*.log > "$OUT/price_step.txt" 2>/dev/null

# THE AUTO PICK: the lowest STEP ratio among price runs that exited 0 (a run
# whose bits moved raises, so exit 0 is bits equal on every call).
pick_auto() {
    for f in "$OUT"/price-*.log; do
        [ -f "$f" ] || continue
        _run=$(basename "$f" .log)
        _arm=${_run#price-}
        [ "$_arm" = shipped ] && continue
        _code=$(awk -F '\t' -v n="$_run" '$1 == n { c = $2 } END { print c }' "$OUT/status.tsv")
        [ "$_code" = 0 ] || continue
        awk -v arm="$_arm" '$1 == "STEP" && $2 == "gemm" {
            for (i = 3; i <= NF; i++) if ($i ~ /^ratio=/) { v = $i; sub(/^ratio=/, "", v); print v, arm }
        }' "$f"
    done | sort -g | head -1 | awk '{ print $2 }'
}
if [ "$LM_ARMS" = auto ]; then
    LM_ARMS=$(pick_auto)
fi
# `shipped` is always the bracket; drop it from the arm list.
LM_ARMS=$(echo "$LM_ARMS" | tr ',' '\n' | grep -v '^shipped$' | grep -v '^$' | paste -sd, -)
echo "lm_arms_resolved=${LM_ARMS:-none}" >> "$OUT/gate.txt"

# ---- the bindings, with the trial hook ---------------------------------------
LM_OK=0
if [ "${MOJOLEARN_GEMM_STEP_LEG_SKIP_LM:-0}" != "1" ] && [ -n "$LM_ARMS" ]; then
    rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
    run build-binding-base env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" \
        MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL" sh bindings/build.sh
    run build-binding-byte-lm env MOJOLEARN_NUMERIC_MODE=identical \
        MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL" sh bindings/build_byte_lm.sh
    for f in python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so; do
        [ -f "$f" ] && sha256sum "$f" >> "$OUT/bindings.sha256"
    done
    if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
        pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
    fi
    [ -f python/mojolearn/identical/_mojolearn_byte_lm.so ] && LM_OK=1
else
    echo "LM probes NOT RUN (skip_lm=${MOJOLEARN_GEMM_STEP_LEG_SKIP_LM:-0}, lm_arms=${LM_ARMS:-none})" >> "$OUT/gate.txt"
fi

# ---- the two corpora and the LM step under the arms -------------------------
if [ "$LM_OK" = 1 ]; then
    CORPORA=""
    run corpus-enwik8 sh tools/fetch_corpus_enwik8.sh \
        && CORPORA="$CORPORA enwik8=training/corpus/enwik8/input.txt"
    run corpus-pile-github sh tools/fetch_corpus_pile_github.sh \
        && CORPORA="$CORPORA pilegithub=training/corpus/pile_github/input.txt"
    echo "corpora=$CORPORA" >> "$OUT/gate.txt"
    for spec in $CORPORA; do
        name=${spec%%=*}
        path=${spec#*=}
        for arm in shipped $(echo "$LM_ARMS" | tr ',' ' ') shippedclose; do
            env_arm=$arm
            [ "$arm" = shippedclose ] && env_arm=shipped
            run "lm-$arm-$name" env PYTHONPATH="$ROOT/python:$ROOT" \
                MOJOLEARN_GEMM_ARM="$env_arm" MOJOLEARN_GEMM_ARM_SABOTAGE=0 \
                timeout "$DEADLINE" pixi run python tools/lm_step_memory_probe.py \
                --out "$OUT/lm-$arm-$name" --target --resident-lean --witness-every-step \
                --steps "$LM_STEPS" --budget-seconds "$DEADLINE" --corpus "$path"
            if [ "${MOJOLEARN_GEMM_STEP_LEG_LMTIMING:-0}" = "1" ] && [ "$arm" != shippedclose ]; then
                run "lmtiming-$arm-$name" env PYTHONPATH="$ROOT/python:$ROOT" \
                    MOJOLEARN_GEMM_ARM="$env_arm" MOJOLEARN_GEMM_ARM_SABOTAGE=0 \
                    timeout "$DEADLINE" pixi run python tools/lm_step_memory_probe.py \
                    --out "$OUT/lmtiming-$arm-$name" --target --resident-lean --component-timing \
                    --steps 1 --budget-seconds "$DEADLINE" --corpus "$path"
            fi
        done
    done
    pixi run python - "$OUT" <<'PY' > "$OUT/lm_summary.tsv" 2>> "$OUT/lm_summary.err"
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
rows, med = {}, {}
for d in sorted(out.glob('lm-*')):
    r = d / 'result.json'
    if not r.exists():
        print(f"{d.name}\tno result.json")
        continue
    j = json.loads(r.read_text())
    med[d.name] = j.get('steady_median_seconds')
    corpus = (j.get('corpus') or {}).get('sha256')
    print(f"{d.name}\tsteady_median_seconds={med[d.name]}\tlimited={j.get('limited')}"
          f"\tgemm_arm={j.get('gemm_arm')}\tattention_arm={j.get('attention_arm')}\tcorpus_sha256={corpus}")
    for w in j.get('step_witnesses') or []:
        print(f"{d.name}\tstep={w.get('step')}\tsha256={json.dumps(w.get('sha256'), sort_keys=True)}")
    rows[d.name] = [json.dumps(w.get('sha256'), sort_keys=True) for w in j.get('step_witnesses') or []]
ratios, equal = {}, {}
for name, hashes in rows.items():
    parts = name.split('-')
    if len(parts) != 3:
        continue
    arm, corpus = parts[1], parts[2]
    base = f'lm-shipped-{corpus}'
    if name == base:
        continue
    eq = base in rows and len(hashes) > 0 and hashes == rows[base]
    print(f"{name}\twitnesses_equal_baseline={eq}")
    if arm == 'shippedclose':
        continue
    equal.setdefault(arm, {})[corpus] = eq
    brackets = [med.get(base), med.get(f'lm-shippedclose-{corpus}')]
    brackets = [x for x in brackets if x]
    if med.get(name) and brackets:
        ratios.setdefault(arm, {})[corpus] = med[name] / (sum(brackets) / len(brackets))
        print(f"{name}\tratio_vs_shipped={ratios[arm][corpus]:.4f}\tshipped_brackets={len(brackets)}")
need = ('enwik8', 'pilegithub')
for arm in sorted(set(ratios) | set(equal)):
    per = ratios.get(arm, {})
    if not all(c in per for c in need):
        print(f"verdict\t{arm}\tNO FLIP\ta corpus is unmeasured (measured: {sorted(per)})")
        continue
    if not all(equal.get(arm, {}).get(c) for c in need):
        print(f"verdict\t{arm}\tNO FLIP\ta step witness differs from shipped: not the shipped step's bits")
        continue
    g = math.sqrt(per['enwik8'] * per['pilegithub'])
    word = 'FLIP' if g < 1 else 'NO FLIP'
    print(f"verdict\t{arm}\t{word}\tgeomean={g:.4f}\tenwik8={per['enwik8']:.4f}"
          f"\tpilegithub={per['pilegithub']:.4f}\tquality=identical (every step witness equal to shipped)")
PY
fi

gpu_snapshot "$OUT/gpu_after.txt"
# Binaries stay on the box: they are not evidence and the blob fences refuse them.
rm -rf "${OUT:?}/bin"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
[ -f "$OUT/price_step.txt" ] && cat "$OUT/price_step.txt"
[ -f "$OUT/resources_lines.txt" ] && cat "$OUT/resources_lines.txt"
[ -f "$OUT/lm_summary.tsv" ] && grep '^verdict\|witnesses_equal_baseline\|ratio_vs_shipped' "$OUT/lm_summary.tsv"
cat "$OUT/status.tsv"
exit "$rc"
