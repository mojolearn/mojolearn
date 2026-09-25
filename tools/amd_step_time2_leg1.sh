#!/bin/sh
# tools/amd_step_time2_leg1.sh -- lane/amd-step-time-2 (2026-09-25), pass 2
# leg 1 on a Hot Aisle 1x MI300X (tools/hotaisle_leg.sh --skip-gates): the
# matrix-core GEMM's EXACT ADMISSION (gemm/checks/gemm_identical.mojo,
# `GEMM_MFMA_ADMIT`), priced and proven, fully scripted:
#   1. GEMM A/B harness (bench/gemm_excp_ab_main.mojo) built three ways:
#      valu (-D MOJOLEARN_GEMM_NO_MFMA=1), noadmit (0.8.18's matrix-core
#      kernel, -D MOJOLEARN_GEMM_MFMA_NO_ADMIT=1) and admit (the branch);
#      every kind (ordinary, tiny, mixed, skew, border, sparse); every hash
#      line of every build must be equal
#   2. byte LM binding: admit and noadmit; lean B4 step (witnesses) on both
#   3. replays held to the H100 chain on admit: steps 101..103 from ckpt 100
#      (A-1 chain) and 1999..2000 from ckpt 1998 (A-2 chain)
#   4. rocprofv3 kernel trace of one lean B4 step on admit
#   5. every device binding rebuilt (admit is the default); gemm device,
#      backward and workspace checks; verify over the 201 GEMM-reaching
#      non-par lanes in chunks of 25 against the shipped reference table
# Needs /root/urls (tools/amd_step_time_urls.py) and /root/amd_in
# (A-1.chain.partial.jsonl, A-2.chain.jsonl, lanes_gemm_nonpar.txt) pushed
# after the VM is up. Holds at most 10 minutes for /root/amd_step_done.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds" "$OUT/ab" "$OUT/verify" "$OUT/prof"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
BIN=/root/amd_bin; mkdir -p "$BIN"
S="sh tools/amd_step_time_session.sh"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "pass2 leg1 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; free -g; } > "$OUT/host.txt" 2>&1
rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

abb() {  # tag, defines...
    tag=$1; shift
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
        $(for d in "$@"; do printf ' -D %s' "$d"; done) -I . bench/gemm_excp_ab_main.mojo -o "$BIN/ab_$tag" > "$OUT/ab/$tag.build.log" 2>&1
    say "ab build $tag exit=$?"
}
blm() {  # tag, defines
    mkdir -p "$BIN/out_$1"; rm -f "$BIN/out_$1/_mojolearn_byte_lm.so"
    t0=$(date +%s)
    MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_$1" MOJOLEARN_BUILD_EXTRA_DEFINES="$2" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.$1.log" 2>&1
    say "byte_lm $1 exit=$? secs=$(( $(date +%s) - t0 ))"
    cp "$BIN/out_$1/_mojolearn_byte_lm.so" "$BIN/byte_lm.$1.so"
}
lean() {  # tag
    $S use "$1" > /dev/null
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/lean-$1" --shape 4 2048 768 12 12 64 2048 12 50257 \
        --steps 3 --resident-lean --witness-every-step --budget-seconds 600 > "$OUT/lean-$1.log" 2>&1
    say "lean $1: $(python3 -c "import json;r=json.load(open('$OUT/lean-$1/result.json'));print(r['steady_median_seconds'], [w['sha256']['parameters'][:12] for w in r['step_witnesses']], [w['sha256'].get('loss','')[:12] for w in r['step_witnesses']])" 2>&1 | tail -1)"
}

t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "base exit=$? secs=$(( $(date +%s) - t0 ))"
( blm admit "" ) &
( blm noadmit "-D MOJOLEARN_GEMM_MFMA_NO_ADMIT=1" ) &
abb valu MOJOLEARN_GEMM_NO_MFMA=1
abb noadmit MOJOLEARN_GEMM_MFMA_NO_ADMIT=1
abb admit
wait

# ---- 1. the GEMM hashes: every kind, every build ----
for t in valu noadmit admit; do
    MOJOLEARN_EXCP_AB_ROUNDS=3 timeout 900 "$BIN/ab_$t" > "$OUT/ab/p2l1-$t.log" 2>&1
    grep '^EXCP_AB call' "$OUT/ab/p2l1-$t.log" | sed 's/ ms=.*//' > "$OUT/ab/p2l1-$t.hashes"
    if cmp -s "$OUT/ab/p2l1-valu.hashes" "$OUT/ab/p2l1-$t.hashes"; then v=IDENTICAL; else v=DIFFER; fi
    say "ab $t lines=$(wc -l < "$OUT/ab/p2l1-$t.hashes") hashes_vs_valu=$v: $(grep '^EXCP_AB call' "$OUT/ab/p2l1-$t.log" | grep 'kind=ordinary' | awk '{split($2,a,"=");split($9,b,"="); printf "%s=%s ", a[2], b[2]}')"
done

# ---- 2 and 3. the step ----
while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ] || [ ! -s /root/amd_in/A-2.chain.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
say "fetch exit=$?"
lean admit
lean noadmit
$S replay admit ckpt_00000100.blm A-1.chain.partial.jsonl 3 > /dev/null 2>&1
$S replay admit ckpt_00001998.blm A-2.chain.jsonl 2 > /dev/null 2>&1

# ---- 4. the kernel trace of one lean step on admit ----
$S use admit > /dev/null
t0=$(date +%s)
timeout 600 rocprofv3 --kernel-trace --stats -d "$OUT/prof/trace" -o lean -- "$ROOT/.pixi/envs/default/bin/python" \
    tools/lm_step_memory_probe.py --out "$OUT/lean-traced" --shape 4 2048 768 12 12 64 2048 12 50257 --steps 2 \
    --resident-lean --budget-seconds 500 > "$OUT/prof/trace.log" 2>&1
say "kernel trace exit=$? secs=$(( $(date +%s) - t0 ))"
find "$OUT/prof/trace" -name '*stats*' -exec cp {} "$OUT/prof/" \; 2>/dev/null
find "$OUT/prof/trace" -name '*kernel_trace*.csv' -exec gzip -9 {} \; 2>/dev/null

# ---- 5. identity: every device binding, the GEMM checks, the lanes ----
n=0
for f in estimators linalg training transformer embedding kernel_methods gp svm mixture metrics preprocessing resample solver rf gbdt trees hdbscan ivf mamba arima tsa; do
    ( t0=$(date +%s); MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=2 sh bindings/build_$f.sh > "$OUT/builds/$f.log" 2>&1
      say "build $f exit=$? secs=$(( $(date +%s) - t0 ))" ) &
    n=$((n + 1))
    if [ $((n % 4)) -eq 0 ]; then wait; fi
done
wait
$S use admit > /dev/null
say "builds done"
for c in gemm_device_check gemm_backward_check gemm_workspace_check; do
    extra=""
    [ "$c" = gemm_workspace_check ] && extra="-D MOJOLEARN_STEP_PHASE_TIMERS=1"
    t0=$(date +%s)
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" $extra \
        -I . gemm/checks/$c.mojo > "$OUT/$c.log" 2>&1
    say "$c exit=$? secs=$(( $(date +%s) - t0 )): $(grep -E 'all green|PASS|FAIL' "$OUT/$c.log" | tail -1 | cut -c1-200)"
done
LIST=/root/amd_in/lanes_gemm_nonpar.txt
[ -s "$LIST" ] || LIST=bench/results/amd_step_time_2026-09-24/lanes_gemm_nonpar.txt
split -l 25 -d "$LIST" "$BIN/lanechunk"
for f in "$BIN"/lanechunk*; do
    [ -s "$f" ] || continue
    k=$(basename "$f" | sed 's/lanechunk//')
    t0=$(date +%s)
    pixi run python -m mojolearn verify --lanes "$(tr '\n' ',' < "$f" | sed 's/,$//')" --json-out "$OUT/verify/chunk$k.json" > "$OUT/verify/chunk$k.log" 2>&1
    say "verify chunk$k exit=$? secs=$(( $(date +%s) - t0 )): $(grep RESULT "$OUT/verify/chunk$k.log" | tail -1 | cut -c1-300)"
done
gzip -9 -f "$OUT"/verify/*.json

touch /root/amd_step_ready
say "pass2 leg1 scripted part done; holding (at most 10 min)"
n=0
while [ ! -e /root/amd_step_done ] && [ $n -lt 30 ]; do sleep 20; n=$((n + 1)); done
exit 0
