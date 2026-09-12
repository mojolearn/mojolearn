#!/bin/sh
# THE AMD SHIPPED BRANCH GATE for both flips of 2026-09-12.
#
# The two AMD legs earlier today measured TRIAL arms against the OLD shipped
# defaults and said those arms win (attention estash geomean 0.9716, step glue
# 0.9854, bits unmoved on both). The kernel matrix now routes both winners
# into the AMD column. NOTHING HAS EVER COMPILED THAT: a shipped AMD build
# with no trial define has never reached `ATTN_SHIPPED_BWD_ESTASH` or
# `STEP_GLUE_SHIPPED_ROWS` / `STEP_GLUE_SHIPPED_UPDATE`, because until this
# commit the AMD column carried neither bit. That gap is exactly where the
# NVIDIA side hid a correctness hole (a refusal assert stranded inside
# `comptime if STEP_GLUE_TRIAL`), so it is gated here before the flip merges.
#
# EVERY BUILD BELOW IS A SHIPPED BUILD unless the phase name says trial. No
# -D MOJOLEARN_ATTN_ARM_TRIAL, no -D MOJOLEARN_STEP_GLUE_TRIAL, and no
# EVERY_COLUMN knob: the point is that the AMD column's OWN routing rows now
# resolve to the winners, so the knobs would prove nothing here.
#
# A MOJOLEARN_GEMM_LEG_EXTRA body for tools/hotaisle_leg.sh. POSIX sh only.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/amd-ship
mkdir -p "$OUT/bin"
JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
G="$OUT/gate.txt"
ok=1

say() { echo "$@" >> "$G"; }
run() { _n=$1; shift; "$@" > "$OUT/$_n.log" 2>&1; _e=$?; echo "$_n	$_e" >> "$OUT/status.tsv"; [ "$_e" = 0 ] || ok=0; return "$_e"; }

say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "vendor=${MOJOLEARN_TARGET_COLUMN:-unset} archs=${MOJOLEARN_GPU_ARCHS:-unset} jobs=$JOBS"
say "commit=$(cat /root/mojolearn/COMMIT 2>/dev/null || echo unknown)"

# ---- gate 1: the shipped fused check on AMD, with the flipped default ------
# This is the bits question for DEVIATION 2657 on AMD. The DEFAULT line must
# now name the estash word, and the check must still be bit-identical to eager.
if run build-fused pixi run mojo build -j "$JOBS" -I . \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
        transformer/checks/transformer_fused_check.mojo -o "$OUT/bin/fused"; then
    run fused "$OUT/bin/fused"
    grep -hE "^DEFAULT |transformer_fused_check:" "$OUT/fused.log" | sed 's/^/fused: /' >> "$G"
    grep -q "transformer_fused_check: PASS" "$OUT/fused.log" || { say "GATE 1 FAIL"; ok=0; }
fi

# ---- gate 2: the shipped backward check on AMD -----------------------------
# The bits question for DEVIATION 2649's rows half: llama_rms_norm and
# bwd_rms_norm now launch 16 threads per block on a shipped AMD build.
if run build-bwd pixi run mojo build -j "$JOBS" -I . \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
        transformer/checks/transformer_backward_check.mojo -o "$OUT/bin/bwd"; then
    run bwd "$OUT/bin/bwd"
    grep -hE "clause \(a\)|PASS|FAIL" "$OUT/bwd.log" | tail -3 | sed 's/^/bwd: /' >> "$G"
    grep -q "FAIL" "$OUT/bwd.log" && { say "GATE 2 FAIL"; ok=0; }
fi

# ---- gate 3: the trial arms are unchanged by the flip ----------------------
if run build-glue-check pixi run mojo build -j "$JOBS" -I . \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_GLUE_TRIAL=1 \
        training/checks/step_glue_check.mojo -o "$OUT/bin/glue-check"; then
    run glue-check "$OUT/bin/glue-check"
    grep -hE "^REACH|step_glue_check:" "$OUT/glue-check.log" | sed 's/^/glue_check: /' >> "$G"
    grep -q "step_glue_check: PASS" "$OUT/glue-check.log" || { say "GATE 3 FAIL"; ok=0; }
fi

# ---- gate 4: a SHIPPED AMD binding resolves both winners (reach) -----------
# The readback is the runtime observation that the comptime constants resolved
# the way the routing rows say, on this vendor, in a binary with no trial hook.
run build-binding-base env MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS="$JOBS" sh bindings/build.sh
if run build-byte-lm env MOJOLEARN_NUMERIC_MODE=identical \
        MOJOLEARN_BYTE_LM_OUTDIR="$OUT/bin/shipped" sh bindings/build_byte_lm.sh; then
    run readback env PYTHONPATH="$OUT/bin/shipped:/root/mojolearn/python:/root/mojolearn" \
        pixi run python -c "
import _mojolearn_byte_lm as b
arm, trial = b.byte_lm_step_glue_arm()
print('READBACK arm=' + str(arm) + ' trial=' + str(trial))
assert str(arm) == 'optskip_noshadow_rows16', 'want the flipped AMD word, got ' + str(arm)
assert int(trial) == 0, 'want a shipped binding, got trial=' + str(trial)
print('READBACK OK')
"
    grep -hE "READBACK|AssertionError" "$OUT/readback.log" | sed 's/^/binding: /' >> "$G"
    grep -q "READBACK OK" "$OUT/readback.log" || { say "GATE 4 FAIL"; ok=0; }
fi

say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "AMD_SHIP_GATES_OK=$ok"
cat "$OUT/status.tsv"
cat "$G"
[ "$ok" = 1 ]
