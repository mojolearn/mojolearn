#!/bin/sh
# tools/amd_step_time_leg2.sh -- lane/amd-step-time (2026-09-24), the second
# AMD leg, scripted. Needs /root/urls (tools/amd_step_time_urls.py) and
# /root/amd_in/*.jsonl (the expected chains), placed by the lane over ssh
# right after the box is up; waits for them.
#   1. base binding; GEMM A/B binaries: the branch (launch bound), the trial
#      arm build (MOJOLEARN_GEMM_ARM_TRIAL), the two AMD k=768 dispatch arms
#   2. A/B prices and hashes, ordinary operands, every arm against the branch
#   3. byte LM bindings: branch, and baseline (-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND)
#   4. the replays: baseline 2 steps from ckpt 100 (the same-box BEFORE), then
#      the branch from ckpt 100 (3 steps) and ckpt 1998 (2 steps), all held to
#      the H100 chain
#   5. the identity lanes on the branch binding
# Then holds until /root/amd_step_done.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/ab"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
S="sh tools/amd_step_time_session.sh"
BIN=/root/amd_bin; mkdir -p "$BIN"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg2 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
rocm-smi --showproductname --showuniqueid --showclocks > "$OUT/gpu.txt" 2>&1
{ cat /opt/rocm/.info/version 2>/dev/null; ls -d /opt/rocm* 2>/dev/null; } > "$OUT/rocm.txt" 2>&1

t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
say "build base exit=$? secs=$(( $(date +%s) - t0 ))"
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

abbuild() {  # tag, defines...
    tag=$1; shift
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
        $(for d in "$@"; do printf ' -D %s' "$d"; done) -I . bench/gemm_excp_ab_main.mojo -o "$BIN/ab_$tag" > "$OUT/ab/$tag.build.log" 2>&1
    say "ab build $tag exit=$?"
}
abbuild branch
abbuild trial MOJOLEARN_GEMM_ARM_TRIAL=1
abbuild k768kpack MOJOLEARN_GEMM_AMD_NO_K768_TUNED=1
abbuild k768p64 MOJOLEARN_GEMM_AMD_K768_PLAN64=1
# the byte LM bindings build on the CPU while the A/B runs on the GPU
( $S bind branch > "$OUT/bind_branch.out" 2>&1; cp "$BIN/byte_lm.branch.so" "$BIN/keep.branch.so" 2>/dev/null
  rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
  $S bind baseline MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1 > "$OUT/bind_baseline.out" 2>&1; say "bindings built: $(ls $BIN/byte_lm.*.so | tr '\n' ' ')" ) &
BINDPID=$!

abrun() {  # label, binary, env...
    label=$1; bin=$2; shift 2
    env MOJOLEARN_EXCP_AB_KINDS=ordinary MOJOLEARN_EXCP_AB_ROUNDS=3 "$@" "$BIN/ab_$bin" > "$OUT/ab/$label.log" 2>&1
    rc=$?
    grep '^EXCP_AB call' "$OUT/ab/$label.log" | sed 's/ ms=.*//' > "$OUT/ab/$label.hashes"
    if cmp -s "$OUT/ab/branch.hashes" "$OUT/ab/$label.hashes"; then v=IDENTICAL; else v=DIFFER; fi
    say "ab $label exit=$rc hashes_vs_branch=$v sum_ms=$(grep '^EXCP_AB call' "$OUT/ab/$label.log" | sed 's/.* ms=\([0-9.]*\).*/\1/' | awk '{s+=$1} END {print s}')"
}
abrun branch branch
abrun k768kpack k768kpack
abrun k768p64 k768p64
for arm in shipped tuned128 lfold half half_ks16 quarter kpack kpack_wide kpack_hg ksplit ksplit_leaf kfoldv; do
    abrun "arm-$arm" trial MOJOLEARN_GEMM_ARM=$arm
done
wait $BINDPID

while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
say "fetch exit=$?"
$S replay baseline ckpt_00000100.blm A-1.chain.partial.jsonl 2 > /dev/null 2>&1
$S replay branch ckpt_00000100.blm A-1.chain.partial.jsonl 3 > /dev/null 2>&1
$S replay branch ckpt_00001998.blm A-2.chain.jsonl 2 > /dev/null 2>&1
$S verify branch "byte-lm,byte-lm-resident,language-model-config,gemm-pinned,gemm-transposed,gemm-bf16,gemm-int8,embedding,cross-entropy-arms,training-primitives,grad-accumulation,optim-adam-clip,transformer,mlp,ols,ridge,pca,kmeans,tsvd,logistic" > /dev/null 2>&1

touch /root/amd_step_ready
say "leg2 scripted part done; holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
