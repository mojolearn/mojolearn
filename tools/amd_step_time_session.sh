#!/bin/sh
# tools/amd_step_time_session.sh -- lane/amd-step-time (2026-09-24): the
# measurements of the lane, run ON the AMD box held by
# tools/amd_step_time_body.sh, one subcommand at a time, over the lane's own
# ssh session:
#
#   sh tools/amd_step_time_session.sh asm   <tag> [defines...]   gfx942 asm of the step GEMM kernels -> asm/<tag>/
#   sh tools/amd_step_time_session.sh bind  <tag> [defines...]   byte LM binding with extra defines -> /root/amd_bin/byte_lm.<tag>.so
#   sh tools/amd_step_time_session.sh ab    <tag> [defines...]   bench/gemm_excp_ab_main.mojo -> ab/<tag>.log
#   sh tools/amd_step_time_session.sh fetch                      tokens, recipe, checkpoints (URLs in /root/urls/*.json)
#   sh tools/amd_step_time_session.sh replay <tag> <ckpt> <chain> [steps]   lm_segment run --expect-chain on binding <tag>
#   sh tools/amd_step_time_session.sh timed  <tag> <ckpt> <chain>           the same, one step, MOJOLEARN_TRANSFORMER_TIMING=1
#   sh tools/amd_step_time_session.sh verify <tag> <lanes>                  python -m mojolearn verify on binding <tag>
#
# Every result lands under /root/gemm_leg_out/amd-step-time (fetched home by
# the leg). Binaries stay in /root/amd_bin. POSIX sh.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
BIN=/root/amd_bin
IN=/root/amd_in
mkdir -p "$OUT" "$BIN" "$IN"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$ST"; }
defs() { for d in "$@"; do printf ' -D %s' "$d"; done; }

cmd=${1:-}; shift || true
case "$cmd" in
asm)
    tag=$1; shift
    mkdir -p "$OUT/asm/$tag"
    t0=$(date +%s)
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD $(defs "$@") \
        -I . tools/amd_codegen/probe_step_gemm.mojo -o "$BIN/asm_$tag" > "$OUT/asm/$tag/build.log" 2>&1 || { say "asm $tag build FAILED"; exit 1; }
    "$BIN/asm_$tag" > "$BIN/asm_$tag.s" 2>&1
    head -1 "$BIN/asm_$tag.s" > "$OUT/asm/$tag/config.txt"
    # the resource lines of each kernel, and the instruction census of the file per kernel
    awk '/^### /{k=$2} /\.vgpr_count|\.sgpr_count|vgpr_spill|sgpr_spill|private_segment_fixed_size|group_segment_fixed_size|; NumVgprs|; NumAgprs|; TotalNumVgprs|; ScratchSize|; Occupancy|; LDSByteSize|\.max_flat_workgroup_size|amdhsa_next_free_vgpr|amdhsa_accum_offset/{print k": "$0}' "$BIN/asm_$tag.s" > "$OUT/asm/$tag/resources.txt"
    for k in tuned128 kpack_all kpack_grp; do
        awk -v K="$k" '/^### /{on=($2==K)} on' "$BIN/asm_$tag.s" > "$BIN/asm_$tag.$k.s"
        printf '%s' "$k" >> "$OUT/asm/$tag/census.txt"
        for pat in 'v_fmac_f32' 'v_fma_f32' 'v_pk_fma_f32' 'v_cmp_class_f32' 's_nop' 'v_cndmask_b32' 'ds_read' 'ds_write' 'scratch_' 'buffer_' 'global_load' 's_getreg' 's_setreg'; do
            printf ' %s=%s' "$pat" "$(grep -c "$pat" "$BIN/asm_$tag.$k.s")" >> "$OUT/asm/$tag/census.txt"
        done
        echo >> "$OUT/asm/$tag/census.txt"
    done
    say "asm $tag done in $(( $(date +%s) - t0 )) s"; cat "$OUT/asm/$tag/census.txt"; cat "$OUT/asm/$tag/resources.txt" ;;
bind)
    tag=$1; shift
    rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
    t0=$(date +%s)
    MOJOLEARN_BUILD_EXTRA_DEFINES="$(defs "$@")" sh bindings/build_byte_lm.sh > "$OUT/build_byte_lm.$tag.log" 2>&1; rc=$?
    [ "$rc" -eq 0 ] && cp python/mojolearn/identical/_mojolearn_byte_lm.so "$BIN/byte_lm.$tag.so"
    say "bind $tag defines=[$*] exit=$rc secs=$(( $(date +%s) - t0 ))" ;;
use)
    tag=$1
    cp "$BIN/byte_lm.$tag.so" python/mojolearn/identical/_mojolearn_byte_lm.so && say "binding now $tag" ;;
ab)
    tag=$1; shift
    mkdir -p "$OUT/ab"
    t0=$(date +%s)
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" $(defs "$@") \
        -I . bench/gemm_excp_ab_main.mojo -o "$BIN/ab_$tag" > "$OUT/ab/$tag.build.log" 2>&1 || { say "ab $tag build FAILED"; tail -30 "$OUT/ab/$tag.build.log"; exit 1; }
    say "ab $tag built in $(( $(date +%s) - t0 )) s"
    "$BIN/ab_$tag" > "$OUT/ab/$tag.log" 2>&1; say "ab $tag exit=$? in $(( $(date +%s) - t0 )) s"
    grep EXCP_AB "$OUT/ab/$tag.log" | cut -c1-200 ;;
fetch)
    # /root/urls/tokens.json, ckpt.json ({name: url}); /root/urls/recipe.url
    TOK=/root/tokens_stream; mkdir -p "$TOK"
    cp tools/lm_controls_body.sh /dev/null
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
replay|timed)
    tag=$1; ckpt=$2; chain=$3; steps=${4:-3}
    sh "$0" use "$tag" || exit 1
    name="$cmd-$tag-$(basename "$ckpt" .blm)"
    rm -rf "$OUT/$name"
    extra=""
    [ "$cmd" = timed ] && { export MOJOLEARN_TRANSFORMER_TIMING=1; steps=1; }
    t0=$(date +%s)
    pixi run python tools/lm_segment.py run --recipe "$IN/recipe.json" --tokens /root/tokens_stream --from "$IN/$ckpt" \
        --steps "$steps" --devices 0 --route A --segment "amdstep-$name" --label "amd-step-time-$name" --no-checkpoints \
        --expect-chain "$IN/$chain" --out "$OUT/$name" > "$OUT/$name.log" 2>&1; rc=$?
    say "$name exit=$rc secs=$(( $(date +%s) - t0 )): $(grep -E 'PASS|FAIL|DISAGREE|REFUSED|Error' "$OUT/$name.log" | grep -v '^timing' | tail -2 | tr '\n' ' ' | cut -c1-300)"
    if [ "$cmd" = timed ]; then
        grep '^timing ' "$OUT/$name.log" > "$OUT/$name.timing.txt"
        python3 tools/amd_step_timing_summary.py "$OUT/$name.timing.txt" --skip-shards 1 --tsv "$OUT/$name.summary.tsv" | tail -40
        gzip -f "$OUT/$name.log"
    fi ;;
verify)
    tag=$1; lanes=$2
    sh "$0" use "$tag" || exit 1
    t0=$(date +%s)
    pixi run python -m mojolearn verify --lanes "$lanes" --json-out "$OUT/verify-$tag.json" > "$OUT/verify-$tag.log" 2>&1; rc=$?
    say "verify $tag exit=$rc secs=$(( $(date +%s) - t0 )): $(tail -3 "$OUT/verify-$tag.log" | tr '\n' ' ' | cut -c1-300)" ;;
*)
    sed -n 2,20p "$0"; exit 2 ;;
esac
