#!/bin/sh
# tools/amd_step_time_leg5.sh -- lane/amd-step-time (2026-09-24), the fifth
# AMD leg: where the remaining shard time goes, and the attention arms NVIDIA
# already runs.
#   1. base binding; byte LM: the branch head (lean) and the attention TRIAL
#      build (-D MOJOLEARN_ATTN_ARM_TRIAL=1)
#   2. rocprofv3 kernel trace + stats of one lean B4 step on the branch: every
#      kernel of a shard, its launches and its seconds
#   3. rocprofv3 counters on three GEMM calls (bench/gemm_excp_ab_main.mojo):
#      VALU busy/issue, LDS waits, waves
#   4. the lean B4 step under each attention arm below, step witnesses kept
#      (every arm must write the same witnesses as the default)
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds" "$OUT/prof"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
BIN=/root/amd_bin; mkdir -p "$BIN"
S="sh tools/amd_step_time_session.sh"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg5 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

blm() {  # tag, defines
    mkdir -p "$BIN/out_$1"; rm -f "$BIN/out_$1/_mojolearn_byte_lm.so"
    t0=$(date +%s)
    MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_$1" MOJOLEARN_BUILD_EXTRA_DEFINES="$2" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.$1.log" 2>&1
    say "byte_lm $1 exit=$? secs=$(( $(date +%s) - t0 ))"
    cp "$BIN/out_$1/_mojolearn_byte_lm.so" "$BIN/byte_lm.$1.so"
}
t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "base exit=$? secs=$(( $(date +%s) - t0 ))"
( blm branch "" ) & ( blm trial "-D MOJOLEARN_ATTN_ARM_TRIAL=1" ) &
( pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator gfx942 -I . \
    bench/gemm_excp_ab_main.mojo -o "$BIN/ab_branch" > "$OUT/builds/ab.log" 2>&1; say "ab build exit=$?" ) &
wait

lean() {  # label, tag, env...
    label=$1; tag=$2; shift 2
    $S use "$tag" > /dev/null
    env "$@" pixi run python tools/lm_step_memory_probe.py --out "$OUT/lean-$label" --shape 4 2048 768 12 12 64 2048 12 50257 \
        --steps 3 --resident-lean --witness-every-step --budget-seconds 600 > "$OUT/lean-$label.log" 2>&1
    say "lean $label: $(python3 -c "import json;r=json.load(open('$OUT/lean-$label/result.json'));print(r['steady_median_seconds'], r.get('attention_arm'), [w['sha256']['parameters'][:12] for w in r['step_witnesses']])" 2>&1 | tail -1)"
}
lean branch branch

# ---- the kernel trace of one lean step (the probe's worker is the child) ----
$S use branch > /dev/null
t0=$(date +%s)
rocprofv3 --kernel-trace --stats -d "$OUT/prof/trace" -o lean -- pixi run python tools/lm_step_memory_probe.py --out "$OUT/lean-traced" \
    --shape 4 2048 768 12 12 64 2048 12 50257 --steps 2 --resident-lean --budget-seconds 600 > "$OUT/prof/trace.log" 2>&1
say "kernel trace exit=$? secs=$(( $(date +%s) - t0 ))"
find "$OUT/prof/trace" -name '*kernel_stats*' -exec cp {} "$OUT/prof/" \; 2>/dev/null
find "$OUT/prof/trace" -name '*kernel_trace*.csv' -exec gzip -9 {} \; 2>/dev/null

# ---- counters on three GEMM calls ----
for call in proj_fwd gateup_dA head_dA; do
    for set in "SQ_WAVES SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_ACTIVE_INST_VALU" "SQ_INSTS_LDS SQ_WAIT_INST_LDS SQ_WAIT_ANY SQ_WAVE_CYCLES" "SQ_INSTS_SALU SQ_ACTIVE_INST_LDS SQ_INST_CYCLES_VMEM SQ_INSTS_VMEM"; do
        tag=$(echo "$set" | cut -d' ' -f1)
        MOJOLEARN_EXCP_AB_CALLS=$call MOJOLEARN_EXCP_AB_KINDS=ordinary MOJOLEARN_EXCP_AB_ROUNDS=1 \
            rocprofv3 --pmc $set -d "$OUT/prof/pmc-$call-$tag" -o pmc -- "$BIN/ab_branch" > "$OUT/prof/pmc-$call-$tag.log" 2>&1
        say "pmc $call $tag exit=$?"
    done
done

# ---- attention arms (trial build; the default first, then NVIDIA's) ----
for arm in stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32 \
           stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32_bswz \
           stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32 \
           stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r64 \
           stash_tiled_fgrid_r64_pf_estash_dres_kvgrid_r32 \
           stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32_kvsplit \
           stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r64_bswz; do
    lean "arm-$arm" trial MOJOLEARN_ATTN_ARM=$arm
done
lean branch2 branch

touch /root/amd_step_ready
say "leg5 scripted part done; holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
