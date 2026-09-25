#!/bin/sh
# tools/nvidia_step_time/session.sh -- lane/nvidia-step-time (2026-09-25):
# the lane's measurements ON the rented H100, one subcommand at a time (the
# NVIDIA sibling of tools/amd_step_time_session.sh):
#
#   bind   <tag> [defines...]          byte LM binding (sm_90a, IDENTICAL, column nvidia) -> $BIN/byte_lm.<tag>.so
#   use    <tag>                       install that binding into python/mojolearn/identical/
#   ab     <tag> [defines...]          bench/gemm_excp_ab_main.mojo build + run -> ab/<tag>.{log,hashes}
#   ptx    <tag> [defines...]          GEMM kernels' PTX + ptxas -v resources -> ptx/<tag>/
#   fetch                              tokens, recipe, checkpoints (URLs in /root/urls/*)
#   replay <tag> <ckpt> <chain> [n]    lm_segment run --expect-chain on binding <tag>
#   lean   <tag> [steps]               lm_step_memory_probe lean B4 step at the T3 shape -> lean-<tag>/
#   item   <tag>                       timed B4 itemization (needs a timers binding) -> item-<tag>.summary.tsv
#   nsys   <tag>                       nsys trace of the lean B4 step -> nsys-<tag>/*.csv
#   verify <tag> <lanes>               python -m mojolearn verify --lanes on binding <tag>
#
# Results under /root/gemm_leg_out/nv-step-time (fetched home by the leg);
# binaries in /root/nv_bin. POSIX sh.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/nv-step-time
BIN=/root/nv_bin
IN=/root/amd_in
mkdir -p "$OUT" "$BIN" "$IN"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia
: "${MOJOLEARN_GPU_ARCHS:=sm_90a}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
SHAPE="4 2048 768 12 12 64 2048 12 50257"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$ST"; }
defs() { for d in "$@"; do printf ' -D %s' "$d"; done; }

cmd=${1:-}; shift || true
case "$cmd" in
bind)
    tag=$1; shift
    mkdir -p "$BIN/out_$tag" "$OUT/builds"; rm -f "$BIN/out_$tag/_mojolearn_byte_lm.so"
    t0=$(date +%s)
    MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_$tag" MOJOLEARN_BUILD_EXTRA_DEFINES="$(defs "$@")" \
        sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.$tag.log" 2>&1; rc=$?
    [ "$rc" -eq 0 ] && cp "$BIN/out_$tag/_mojolearn_byte_lm.so" "$BIN/byte_lm.$tag.so"
    say "bind $tag defines=[$*] exit=$rc secs=$(( $(date +%s) - t0 ))" ;;
use)
    tag=$1
    cp "$BIN/byte_lm.$tag.so" python/mojolearn/identical/_mojolearn_byte_lm.so && say "binding now $tag" ;;
ab)
    tag=$1; shift
    mkdir -p "$OUT/ab"
    t0=$(date +%s)
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA --target-accelerator "$MOJOLEARN_GPU_ARCHS" $(defs "$@") \
        -I . bench/gemm_excp_ab_main.mojo -o "$BIN/ab_$tag" > "$OUT/ab/$tag.build.log" 2>&1 || { say "ab $tag build FAILED"; tail -30 "$OUT/ab/$tag.build.log"; exit 1; }
    MOJOLEARN_EXCP_AB_KINDS="${AB_KINDS:-ordinary,tiny,mixed}" MOJOLEARN_EXCP_AB_ROUNDS="${AB_ROUNDS:-3}" \
        "$BIN/ab_$tag" > "$OUT/ab/$tag.log" 2>&1; rc=$?
    grep '^EXCP_AB call' "$OUT/ab/$tag.log" | sed 's/ ms=.*//' > "$OUT/ab/$tag.hashes"
    v=""; [ -s "$OUT/ab/${AB_REF:-branch}.hashes" ] && { cmp -s "$OUT/ab/${AB_REF:-branch}.hashes" "$OUT/ab/$tag.hashes" && v=IDENTICAL || v=DIFFER; }
    say "ab $tag exit=$rc cases=$(wc -l < "$OUT/ab/$tag.hashes") vs_${AB_REF:-branch}=$v secs=$(( $(date +%s) - t0 )): $(grep '^EXCP_AB call' "$OUT/ab/$tag.log" | grep ordinary | awk '{split($2,a,"=");for(i=1;i<=NF;i++) if($i ~ /^ms=/){split($i,b,"=")}; printf "%s=%s ", a[2], b[2]}')" ;;
ptx)
    tag=$1; shift
    mkdir -p "$OUT/ptx/$tag"
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA $(defs "$@") \
        -I . tools/nvidia_step_time/probe_gemm_ptx.mojo -o "$BIN/ptx_$tag" > "$OUT/ptx/$tag/build.log" 2>&1 || { say "ptx $tag build FAILED"; exit 1; }
    "$BIN/ptx_$tag" > "$BIN/ptx_$tag.all" 2>&1
    head -1 "$BIN/ptx_$tag.all" > "$OUT/ptx/$tag/config.txt"
    PTXAS=$(ls /root/ptxas_new 2>/dev/null || command -v ptxas || echo /usr/local/cuda/bin/ptxas)
    "$PTXAS" --version > "$OUT/ptx/$tag/ptxas_version.txt" 2>&1
    for k in tuned128 kpack_all kpack_grp; do
        awk -v K="$k" '/^### /{on=($2==K); next} on' "$BIN/ptx_$tag.all" > "$BIN/ptx_$tag.$k.ptx"
        grep -E '^\s*\.(maxntid|minnctapersm|maxnreg|reqntid)|^\.version|^\.target' "$BIN/ptx_$tag.$k.ptx" > "$OUT/ptx/$tag/$k.directives.txt"
        "$PTXAS" -v -arch=sm_90a -O3 "$BIN/ptx_$tag.$k.ptx" -o "$BIN/ptx_$tag.$k.cubin" > "$OUT/ptx/$tag/$k.ptxas.txt" 2>&1
        { printf '%s' "$k"; for pat in 'fma.rn.f32' 'mul.rn.ftz.f32' 'ld.shared' 'st.shared' 'ld.global' 'st.global' 'bar.sync' 'ld.local' 'st.local'; do
            printf ' %s=%s' "$pat" "$(grep -c "$pat" "$BIN/ptx_$tag.$k.ptx")"; done; echo; } >> "$OUT/ptx/$tag/census.txt"
        if command -v cuobjdump > /dev/null 2>&1 && [ -s "$BIN/ptx_$tag.$k.cubin" ]; then
            cuobjdump -sass "$BIN/ptx_$tag.$k.cubin" > "$BIN/ptx_$tag.$k.sass" 2>&1
            { printf '%s sass' "$k"; for pat in 'FFMA' 'FMUL' 'LDS' 'STS' 'LDG' 'STG' 'LDL' 'STL' 'BAR'; do
                printf ' %s=%s' "$pat" "$(grep -cw "$pat[.A-Z0-9]*" "$BIN/ptx_$tag.$k.sass")"; done; echo; } >> "$OUT/ptx/$tag/census.txt"
        fi
    done
    say "ptx $tag: $(grep -h 'registers\|spill' "$OUT/ptx/$tag"/*.ptxas.txt | tr '\n' ' ' | cut -c1-600)" ;;
fetch)
    TOK=/root/tokens_stream; mkdir -p "$TOK"
    sed -n '/^cat > "\$OUT\/fetch_ranges.py"/,/^PY$/p' tools/lm_controls_body.sh | sed '1d;$d' > "$BIN/fetch_ranges.py"
    t0=$(date +%s)
    if [ ! -s "$TOK/tokens.i32" ]; then
        pixi run python "$BIN/fetch_ranges.py" /root/urls/tokens.json "$TOK" > "$OUT/tokens_fetch.log" 2>&1 || { say "tokens fetch FAILED"; exit 3; }
        cat "$TOK"/tokens.i32.part?? > "$TOK/tokens.i32" && rm -f "$TOK"/tokens.i32.part??
    fi
    say "tokens ready in $(( $(date +%s) - t0 )) s: $(wc -c < "$TOK/tokens.i32") bytes"
    curl -fsS --retry 3 -o "$IN/recipe.json" "$(cat /root/urls/recipe.url)" && sha256sum "$IN/recipe.json" | tee "$OUT/recipe.sha256"
    t0=$(date +%s)
    pixi run python "$BIN/fetch_ranges.py" /root/urls/ckpt.json "$IN" > "$OUT/ckpt_fetch.log" 2>&1 || { say "ckpt fetch FAILED"; exit 4; }
    sha256sum "$IN"/*.blm | tee "$OUT/ckpt.sha256"
    say "checkpoints ready in $(( $(date +%s) - t0 )) s" ;;
replay)
    tag=$1; ckpt=$2; chain=$3; steps=${4:-3}
    sh "$0" use "$tag" || exit 1
    name="replay-$tag-$(basename "$ckpt" .blm)"
    rm -rf "$OUT/$name"
    t0=$(date +%s)
    pixi run python tools/lm_segment.py run --recipe "$IN/recipe.json" --tokens /root/tokens_stream --from "$IN/$ckpt" \
        --steps "$steps" --devices 0 --route A --segment "nvstep-$name" --label "nv-step-time-$name" --no-checkpoints \
        --expect-chain "$IN/$chain" --out "$OUT/$name" > "$OUT/$name.log" 2>&1; rc=$?
    grep -v 'mbind memory' "$OUT/$name.log" > "$OUT/$name.clean.log"
    say "$name exit=$rc secs=$(( $(date +%s) - t0 )): $(grep -E '^\[.*\] step |PASS|FAIL|DISAGREE|REFUSED|Error' "$OUT/$name.clean.log" | cut -c1-160 | tr '\n' '|' | cut -c1-900)" ;;
lean)
    tag=$1; steps=${2:-3}
    sh "$0" use "$tag" > /dev/null || exit 1
    rm -rf "$OUT/lean-$tag"
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/lean-$tag" --shape $SHAPE \
        --steps "$steps" --resident-lean --witness-every-step --budget-seconds 600 > "$OUT/lean-$tag.log" 2>&1
    say "lean $tag: $(python3 -c "import json;r=json.load(open('$OUT/lean-$tag/result.json'));print(r['steady_median_seconds'], [w['sha256']['parameters'][:12] for w in r['step_witnesses']], [w['sha256']['loss'][:12] for w in r['step_witnesses']] if 'loss' in r['step_witnesses'][0]['sha256'] else '')" 2>&1 | tail -1)" ;;
item)
    tag=$1
    sh "$0" use "$tag" > /dev/null || exit 1
    rm -rf "$OUT/item-$tag"
    t0=$(date +%s)
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/item-$tag" --shape $SHAPE \
        --steps 1 --resident-lean --component-timing --component-timing-steps 2 --budget-seconds 900 > "$OUT/item-$tag.log" 2>&1
    grep -h '^timing ' "$OUT/item-$tag.log" "$OUT/item-$tag"/*.log 2>/dev/null > "$OUT/item-$tag.timing.txt"
    python3 tools/amd_step_timing_summary.py "$OUT/item-$tag.timing.txt" --skip-shards 1 --tsv "$OUT/item-$tag.summary.tsv" > "$OUT/item-$tag.summary.txt" 2>&1
    gzip -f "$OUT/item-$tag.timing.txt"
    say "item $tag secs=$(( $(date +%s) - t0 )): $(head -5 "$OUT/item-$tag.summary.txt" | tr '\n' ' ' | cut -c1-300)" ;;
nsys)
    tag=$1
    NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
    [ -n "$NSYS" ] || { say "nsys: not on this box"; exit 5; }
    sh "$0" use "$tag" > /dev/null || exit 1
    D="$OUT/nsys-$tag"; rm -rf "$D" "$BIN/nsys-$tag"*; mkdir -p "$D"
    t0=$(date +%s)
    "$NSYS" profile -t cuda --sample=none --cpuctxsw=none -o "$BIN/nsys-$tag" -f true \
        pixi run python tools/lm_step_memory_probe.py --out "$BIN/nsys-lean-$tag" --shape $SHAPE \
        --steps 2 --resident-lean --budget-seconds 600 > "$D/profile.log" 2>&1
    "$NSYS" stats -r cuda_gpu_kern_sum -f csv -o "$D/k" "$BIN/nsys-$tag.nsys-rep" > "$D/stats.log" 2>&1
    "$NSYS" stats -r cuda_gpu_trace -f csv -o "$BIN/nsys-$tag-trace" "$BIN/nsys-$tag.nsys-rep" >> "$D/stats.log" 2>&1
    "$NSYS" stats -r cuda_gpu_mem_time_sum,cuda_api_sum -f csv -o "$D/m" "$BIN/nsys-$tag.nsys-rep" >> "$D/stats.log" 2>&1
    python3 tools/nvidia_step_time/nsys_fold.py "$BIN/nsys-$tag-trace_cuda_gpu_trace.csv" > "$D/kernels.tsv" 2>> "$D/stats.log"
    say "nsys $tag secs=$(( $(date +%s) - t0 )): $(head -8 "$D/kernels.tsv" | cut -c1-150 | tr '\n' '|')" ;;
verify)
    tag=$1; lanes=$2
    sh "$0" use "$tag" > /dev/null || exit 1
    t0=$(date +%s)
    pixi run python -m mojolearn verify --lanes "$lanes" --json-out "$OUT/verify-$tag.json" > "$OUT/verify-$tag.log" 2>&1; rc=$?
    say "verify $tag exit=$rc secs=$(( $(date +%s) - t0 )): $(tail -3 "$OUT/verify-$tag.log" | tr '\n' ' ' | cut -c1-300)" ;;
*)
    sed -n 2,18p "$0"; exit 2 ;;
esac
