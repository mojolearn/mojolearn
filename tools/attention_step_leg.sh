#!/bin/sh
# tools/attention_step_leg.sh -- DEVIATIONS 2525 to 2527, the attention
# step lane's on-box work. Runs ON THE POD as tools/gemm_remote_leg.sh's
# MOJOLEARN_GEMM_LEG_EXTRA hook (after the leg's own IDENTICAL device check
# and card), from /root/mojolearn with pixi on PATH; everything it writes
# under /root/gemm_leg_out/attention-step/ comes home with the leg's fetch.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-step \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# The H100 is named because the baseline this lane attacks is an H100
# number (bench/results/e1g/2026-09-11_004220-nvidia/remote/lm-step-memory/
# target-lean-timing/result.json: attn.core 81.3 ms, bwd.attention 262.5 ms
# of a 564 ms lean step). Everything here is OUR IDENTICAL arm against OUR
# shipped kernels; no opponent runs.
#
# PHASES, each with its own exit code in status.tsv; a later phase runs
# even when an earlier one fails, because a red phase is a finding:
#
#   build-*       three builds of bench/attention_step_price_main.mojo under
#                 IDENTICAL: the trial hook alone (the price), the trial
#                 hook plus -D MOJOLEARN_ATTN_PHASE_TIMERS=1 (the per-kernel
#                 breakdown), and transformer/checks/
#                 transformer_attention_arms_check.mojo (small-shape bits).
#   arms-check    the small-shape gate: every arm equals eager on the fused
#                 check's cases, statuses as expected, reach at hd 64.
#   smoke-<arm>   the harness on the hashed and adversarial generator kinds,
#                 bits, eager oracle (first arm) and reach only, no timing
#                 (ENGINEERING_RULES section 9: an edge-case fixture is never
#                 a timing input).
#   build-binding-*  the base binding and the byte LM binding (bindings/
#                 build.sh, then bindings/build_byte_lm.sh) with the trial
#                 hook, the phase timers and the operand dump riding
#                 MOJOLEARN_BUILD_EXTRA_DEFINES, as tools/lm_step_memory_probe.sh
#                 builds them.
#   corpus-*      the two ORDINARY corpora: English text
#                 (training/corpus/tinyshakespeare, committed, sha256 checked)
#                 and source code (training/corpus/cpython312_lib, fetched by
#                 tools/fetch_corpus_cpython312_lib.sh from the versioned
#                 tarball and verified against its manifest).
#   dump-<corpus> one lean target step on the corpus with
#                 MOJOLEARN_ATTN_OPERAND_DUMP_DIR set: the backward launcher's
#                 first call writes the last layer's q, k, v, dctx and meta.txt
#                 under operands-<corpus>/ (real activations of the real
#                 training path).
#   price-<arm>   the harness on file:operands-shakespeare and
#                 file:operands-cpython, candidate <arm> against baseline:
#                 eager oracle (first arm only), bit equality, reach by
#                 sabotage, then 2 warmups and 7 alternated rounds; PRICE and
#                 TABLE lines in the log. THESE are the timing inputs.
#   timers-<arm>  the timer build under MOJOLEARN_TRANSFORMER_TIMING=1 on the
#                 same real activations, one round, no oracle, no reach: the
#                 `timing attn.<kernel>` lines per launch, summed by name into
#                 timers_summary.tsv. Serialized launches; a breakdown, never
#                 a price.
#   lm-<arm>-<corpus>  tools/lm_step_memory_probe.py --target --resident-lean
#                 --witness-every-step --corpus <corpus>, 3 steps, 300 s
#                 budget, under MOJOLEARN_ATTN_ARM=<arm>: the lean step price
#                 on real bytes and the per-step witnesses; lm_summary.tsv
#                 compares every arm's witnesses with the baseline arm's on
#                 the same corpus (witnesses_equal_baseline).
#   lmtiming-<arm>-<corpus>  the same with --component-timing --steps 1: the
#                 phase timers plus the attn.* per-kernel lines folded into
#                 result.json component_timing_ms.
#
# THE ARMS (MOJOLEARN_ATTN_LEG_ARMS, default
# bwd_stash,fwd_sstash,bwd_stash_tiled,stash_tiled): names pass through to
# the harness unchecked; the native side raises on an unknown one. The LM
# probe runs MOJOLEARN_ATTN_LEG_LM_ARMS (default baseline,stash_tiled) on
# both corpora; add `stash` when the lease allows (two probes per arm and
# corpus). MOJOLEARN_ATTN_LEG_SKIP_LM=1 skips the bindings, the dumps and
# the probes (then only the smoke runs and the price is NOT RUN).
#
# Every item runs under `timeout 300`; a 124 in status.tsv is the deadline.
# The whole file is meant to fit a 60-minute lease with the leg's own
# device check and card in front of it: three harness builds (about 3 min
# each on the H100 box with 2 workers), two binding builds (about 6 min),
# four smoke runs, two dumps, four price runs and five timer runs (about
# 1 min each), then eight probes (about 1 to 2 min each).
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_ATTN_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_ATTN_LEG_OUT:-/root/gemm_leg_out/attention-step}
ARMS=${MOJOLEARN_ATTN_LEG_ARMS:-bwd_stash,fwd_sstash,bwd_stash_tiled,stash_tiled}
LM_ARMS=${MOJOLEARN_ATTN_LEG_LM_ARMS:-baseline,stash_tiled}
ROUNDS=${MOJOLEARN_ATTN_LEG_ROUNDS:-7}
WARMUPS=${MOJOLEARN_ATTN_LEG_WARMUPS:-2}
JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
DEADLINE=${MOJOLEARN_ATTN_LEG_DEADLINE:-300}
mkdir -p "$OUT/bin"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_TARGET_COLUMN="${MOJOLEARN_TARGET_COLUMN:-nvidia}"

# MAX's bundled CUDA 13 assembler needs driver 580. Older-driver pods use
# their installed assembler at BOTH build and runtime.
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
case "$driver_major" in
    ''|*[!0-9]*) ;;
    *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
           export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
       fi ;;
esac

# One mojo build is one GPU arch; read it from the leg's environment first,
# else from the device (9.0 is spelled sm_90a, DEVIATION 2293).
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    case "$cap" in
        9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
        [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
        *) MOJOLEARN_GPU_ARCHS="" ;;
    esac
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

{
    echo "deviations=2525-2527"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "arms=$ARMS"
    echo "lm_arms=$LM_ARMS"
    echo "rounds=$ROUNDS warmups=$WARMUPS deadline=$DEADLINE"
    echo "gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor)=' /root/gemm_leg_out/leg.txt
} > "$OUT/gate.txt"
nvidia-smi --query-gpu=name,driver_version,uuid,clocks.sm,temperature.gpu --format=csv > "$OUT/gpu_before.csv" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
TRIAL="-D MOJOLEARN_ATTN_ARM_TRIAL=1"
TIMERS="-D MOJOLEARN_ATTN_PHASE_TIMERS=1"

# ---- builds ----------------------------------------------------------------
# shellcheck disable=SC2086
run build-price pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
    bench/attention_step_price_main.mojo -o "$OUT/bin/attn-price"
# shellcheck disable=SC2086
run build-timers pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL $TIMERS \
    bench/attention_step_price_main.mojo -o "$OUT/bin/attn-timers"
# shellcheck disable=SC2086
run build-arms-check pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
    transformer/checks/transformer_attention_arms_check.mojo -o "$OUT/bin/arms-check"

# ---- the small-shape gate --------------------------------------------------
if [ -x "$OUT/bin/arms-check" ]; then
    run arms-check timeout "$DEADLINE" "$OUT/bin/arms-check"
fi

# ---- the smoke: hashed and adversarial kinds, bits and reach only -----------
# (ENGINEERING_RULES section 9: never a timing input; MOJOLEARN_ATTN_TIMING=0)
first=1
for arm in $(echo "$ARMS" | tr ',' ' '); do
    [ -x "$OUT/bin/attn-price" ] || break
    oracle=0
    [ "$first" = 1 ] && oracle=1
    first=0
    MOJOLEARN_ATTN_ARM="$arm" MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 \
    MOJOLEARN_ATTN_ORACLE="$oracle" MOJOLEARN_ATTN_REACH=1 \
    run "smoke-$arm" timeout "$DEADLINE" "$OUT/bin/attn-price"
done

# ---- the bindings, with the trial hook, the timers and the operand dump ----
# The probe's own wrapper builds the two bindings the same way; the defines
# ride MOJOLEARN_BUILD_EXTRA_DEFINES so the flags live in one place (target
# cpu, linker floor are build.sh's). The operand dump define costs nothing
# unless MOJOLEARN_ATTN_OPERAND_DUMP_DIR is set for a run.
DUMP="-D MOJOLEARN_ATTN_OPERAND_DUMP=1"
LM_OK=0
if [ "${MOJOLEARN_ATTN_LEG_SKIP_LM:-0}" != "1" ]; then
    rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
    run build-binding-base env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" \
        MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL $TIMERS $DUMP" sh bindings/build.sh
    run build-binding-byte-lm env MOJOLEARN_NUMERIC_MODE=identical \
        MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL $TIMERS $DUMP" sh bindings/build_byte_lm.sh
    for f in python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so; do
        [ -f "$f" ] && sha256sum "$f" >> "$OUT/bindings.sha256"
    done
    if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
        pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
    fi
    [ -f python/mojolearn/identical/_mojolearn_byte_lm.so ] && LM_OK=1
fi

# ---- the two corpora (section 9: two ordinary corpora of different kinds) --
# English text is committed and pinned; source code is fetched on the box
# from the versioned tarball and verified against its manifest.
run corpus-tinyshakespeare sh -c 'test -f training/corpus/tinyshakespeare/input.txt && python3 -c "
import hashlib, json, sys
m = json.load(open(\"training/corpus/tinyshakespeare/manifest.json\"))
raw = open(\"training/corpus/tinyshakespeare/input.txt\", \"rb\").read()
ok = hashlib.sha256(raw).hexdigest() == m[\"sha256\"] and len(raw) == m[\"bytes\"]
print(\"tinyshakespeare\", \"ok\" if ok else \"MISMATCH\", len(raw))
sys.exit(0 if ok else 1)"'
run corpus-cpython312-lib sh tools/fetch_corpus_cpython312_lib.sh
CORPORA=""
[ -f training/corpus/tinyshakespeare/input.txt ] && CORPORA="$CORPORA shakespeare=training/corpus/tinyshakespeare/input.txt"
[ -f training/corpus/cpython312_lib/input.txt ] && CORPORA="$CORPORA cpython=training/corpus/cpython312_lib/input.txt"
echo "corpora=$CORPORA" >> "$OUT/gate.txt"

# ---- real activations: one lean step per corpus dumps the last layer's ----
# q, k, v and dctx (the first backward launcher call of the process).
FILE_KINDS=""
if [ "$LM_OK" = 1 ]; then
    for spec in $CORPORA; do
        name=${spec%%=*}
        path=${spec#*=}
        mkdir -p "$OUT/operands-$name"
        PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_ATTN_ARM=baseline \
        MOJOLEARN_ATTN_OPERAND_DUMP_DIR="$OUT/operands-$name" \
        run "dump-$name" timeout "$DEADLINE" pixi run python tools/lm_step_memory_probe.py \
            --out "$OUT/dump-$name" --target --resident-lean --steps 1 \
            --budget-seconds "$DEADLINE" --corpus "$path"
        if [ -f "$OUT/operands-$name/meta.txt" ]; then
            FILE_KINDS="${FILE_KINDS:+$FILE_KINDS,}file:$OUT/operands-$name"
            (cd "$OUT/operands-$name" && sha256sum q.bin k.bin v.bin dctx.bin > sha256.txt)
        fi
    done
fi
echo "file_kinds=$FILE_KINDS" >> "$OUT/gate.txt"

# ---- the price on the real activations, one candidate arm per run ---------
if [ -n "$FILE_KINDS" ]; then
    first=1
    for arm in $(echo "$ARMS" | tr ',' ' '); do
        [ -x "$OUT/bin/attn-price" ] || break
        oracle=0
        [ "$first" = 1 ] && oracle=1
        first=0
        MOJOLEARN_ATTN_ARM="$arm" MOJOLEARN_ATTN_KINDS="$FILE_KINDS" \
        MOJOLEARN_ATTN_ORACLE="$oracle" MOJOLEARN_ATTN_REACH=1 \
        MOJOLEARN_ATTN_ROUNDS="$ROUNDS" MOJOLEARN_ATTN_WARMUPS="$WARMUPS" \
        run "price-$arm" timeout "$DEADLINE" "$OUT/bin/attn-price"
    done
    # ---- the per-kernel breakdown (serialized; never a price) --------------
    for arm in baseline $(echo "$ARMS" | tr ',' ' '); do
        [ -x "$OUT/bin/attn-timers" ] || break
        MOJOLEARN_ATTN_ARM="$arm" MOJOLEARN_ATTN_KINDS="$FILE_KINDS" \
        MOJOLEARN_ATTN_ORACLE=0 MOJOLEARN_ATTN_REACH=0 \
        MOJOLEARN_ATTN_ROUNDS=1 MOJOLEARN_ATTN_WARMUPS=1 MOJOLEARN_TRANSFORMER_TIMING=1 \
        run "timers-$arm" timeout "$DEADLINE" "$OUT/bin/attn-timers"
    done
else
    echo "no real-activation dump; price and timers NOT RUN (smoke only)" >> "$OUT/gate.txt"
    printf 'price\t9\t0s\n' >> "$OUT/status.tsv"
    rc=1
fi
# One TABLE block per arm and kind, for the brief.
grep -h '^TABLE' "$OUT"/price-*.log > "$OUT/price_tables.txt" 2>/dev/null
grep -h '^REACH\|^BITS .*MOVED\|^FAIL\|PASS' "$OUT"/smoke-*.log "$OUT"/price-*.log > "$OUT/price_verdicts.txt" 2>/dev/null
# Sum the `timing attn.<kernel> <ms> ms` lines by name per run. Both arms
# print in every run (the harness alternates them), so the per-arm split
# is by the arm's own kernel names (bwd_zdot vs bwd_zdot_stash, and so on).
for f in "$OUT"/timers-*.log; do
    [ -f "$f" ] || continue
    awk -v run="$(basename "$f" .log)" '
        $1 == "timing" && $4 == "ms" { ms[$2] += $3; n[$2] += 1 }
        END { for (k in ms) printf "%s\t%s\t%.3f\t%d\n", run, k, ms[k], n[k] }
    ' "$f"
done | sort > "$OUT/timers_summary.tsv"

# ---- the LM step under the arms, on both corpora ----------------------------
if [ "$LM_OK" = 1 ]; then
    for spec in $CORPORA; do
        name=${spec%%=*}
        path=${spec#*=}
        for arm in $(echo "$LM_ARMS" | tr ',' ' '); do
            PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_ATTN_ARM="$arm" \
            run "lm-$arm-$name" timeout "$DEADLINE" pixi run python tools/lm_step_memory_probe.py \
                --out "$OUT/lm-$arm-$name" --target --resident-lean --witness-every-step \
                --steps 3 --budget-seconds "$DEADLINE" --corpus "$path"
            PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_ATTN_ARM="$arm" \
            run "lmtiming-$arm-$name" timeout "$DEADLINE" pixi run python tools/lm_step_memory_probe.py \
                --out "$OUT/lmtiming-$arm-$name" --target --resident-lean --component-timing \
                --steps 1 --budget-seconds "$DEADLINE" --corpus "$path"
        done
    done
    # The lean step medians and the per-step witnesses, one line each: the
    # baseline arm's witnesses on a corpus are the reference every other arm
    # on that corpus must equal (compared here and again at home).
    pixi run python - "$OUT" <<'PY' > "$OUT/lm_summary.tsv" 2>> "$OUT/lm_summary.err"
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
rows = {}
for d in sorted(out.glob('lm-*')):
    r = d / 'result.json'
    if not r.exists():
        print(f"{d.name}\tno result.json")
        continue
    j = json.loads(r.read_text())
    med = j.get('steady_median_seconds')
    lim = j.get('limited')
    corpus = (j.get('corpus') or {}).get('sha256')
    print(f"{d.name}\tsteady_median_seconds={med}\tlimited={lim}\tarm={j.get('attention_arm')}\tcorpus_sha256={corpus}")
    for w in j.get('step_witnesses') or []:
        print(f"{d.name}\tstep={w.get('step')}\tsha256={json.dumps(w.get('sha256'), sort_keys=True)}")
    rows[d.name] = [json.dumps(w.get('sha256'), sort_keys=True) for w in j.get('step_witnesses') or []]
for name, hashes in rows.items():
    parts = name.split('-')
    if len(parts) < 3:
        continue
    base = 'lm-baseline-' + '-'.join(parts[2:])
    if base in rows and base != name:
        print(f"{name}\twitnesses_equal_baseline={hashes == rows[base] and len(hashes) > 0}")
for d in sorted(out.glob('lmtiming-*')):
    r = d / 'result.json'
    if not r.exists():
        print(f"{d.name}\tno result.json")
        continue
    j = json.loads(r.read_text())
    ms = j.get('component_timing_ms') or {}
    for k in sorted(ms, key=lambda k: -ms[k]):
        if k.startswith(('attn.', 'bwd.attention', 'block.attention_total', 'envelope.native_call')):
            print(f"{d.name}\t{k}\t{ms[k]:.3f}")
PY
fi

nvidia-smi --query-gpu=name,driver_version,uuid,clocks.sm,temperature.gpu --format=csv > "$OUT/gpu_after.csv" 2>&1
# Binaries stay on the box: they are not evidence and the blob fences refuse them.
rm -rf "$OUT/bin"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
[ -f "$OUT/price_tables.txt" ] && cat "$OUT/price_tables.txt"
[ -f "$OUT/price_verdicts.txt" ] && cat "$OUT/price_verdicts.txt"
cat "$OUT/status.tsv"
exit "$rc"
