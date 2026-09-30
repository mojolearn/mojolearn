#!/bin/sh
# tools/amd_step_time_leg7.sh -- lane/amd-step-time (2026-09-24): the
# matrix-core GEMM's group launch (the packed kernel's GROUP mode on the
# MFMA kernel, the group size from the leaf split's rule with a floor) priced
# against the VALU kernels at the T3 shapes, every build's hashes held to the
# VALU build's; then the step (lean and replay) on the branch head. Needs
# /root/urls and /root/amd_in pushed after the VM is up.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds" "$OUT/ab"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
BIN=/root/amd_bin; mkdir -p "$BIN"
S="sh tools/amd_step_time_session.sh"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg7 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

abb() {  # tag, defines...
    tag=$1; shift
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator gfx942 \
        $(for d in "$@"; do printf ' -D %s' "$d"; done) -I . bench/gemm_excp_ab_main.mojo -o "$BIN/ab_$tag" > "$OUT/ab/$tag.build.log" 2>&1
    say "ab build $tag exit=$?"
}
blm() {  # tag, defines
    mkdir -p "$BIN/out_$1"; rm -f "$BIN/out_$1/_mojolearn_byte_lm.so"
    MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_$1" MOJOLEARN_BUILD_EXTRA_DEFINES="$2" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.$1.log" 2>&1
    say "byte_lm $1 exit=$?"
    cp "$BIN/out_$1/_mojolearn_byte_lm.so" "$BIN/byte_lm.$1.so"
}
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "base exit=$?"
( blm g1 "" ) & ( blm g4 "-D MOJOLEARN_GEMM_MFMA_GMIN4=1" ) &
abb valu MOJOLEARN_GEMM_NO_MFMA=1
abb g1
abb g2 MOJOLEARN_GEMM_MFMA_GMIN2=1
abb g4 MOJOLEARN_GEMM_MFMA_GMIN4=1
abb g16 MOJOLEARN_GEMM_MFMA_GMIN16=1
abb nogroups MOJOLEARN_GEMM_MFMA_NO_GROUPS=1
for t in valu g1 g2 g4 g16 nogroups; do
    MOJOLEARN_EXCP_AB_ROUNDS=3 "$BIN/ab_$t" > "$OUT/ab/leg7-$t.log" 2>&1
    grep '^EXCP_AB call' "$OUT/ab/leg7-$t.log" | sed 's/ ms=.*//' > "$OUT/ab/leg7-$t.hashes"
    if cmp -s "$OUT/ab/leg7-valu.hashes" "$OUT/ab/leg7-$t.hashes"; then v=IDENTICAL; else v=DIFFER; fi
    say "ab $t hashes_vs_valu=$v: $(grep '^EXCP_AB call' "$OUT/ab/leg7-$t.log" | grep ordinary | awk '{split($2,a,"=");split($9,b,"="); printf "%s=%s ", a[2], b[2]}')"
done
wait
while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
for t in g1 g4; do
    $S use $t > /dev/null
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/lean-$t" --shape 4 2048 768 12 12 64 2048 12 50257 \
        --steps 3 --resident-lean --witness-every-step --budget-seconds 600 > "$OUT/lean-$t.log" 2>&1
    say "lean $t: $(python3 -c "import json;r=json.load(open('$OUT/lean-$t/result.json'));print(r['steady_median_seconds'], [w['sha256']['parameters'][:12] for w in r['step_witnesses']])" 2>&1 | tail -1)"
done
$S replay g1 ckpt_00000100.blm A-1.chain.partial.jsonl 3 > /dev/null 2>&1
touch /root/amd_step_ready
say "leg7 scripted part done; holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
