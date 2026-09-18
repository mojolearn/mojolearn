#!/bin/sh
# lane/attention-replay-vendors: everything a column owes beside the 700-step
# pair. Run through a wrapper that passes the mode (the runners pass no
# environment). The AFTER commit witness is the runner's own commit= line in
# /root/gemm_leg_out/leg.txt; BEFORE is the base recorded in
# tools/lm_attention_replay_vendors_before.json.
#
#   sh tools/lm_attention_vendor_extra_body.sh <gates|identity|all>
#
# gates     native HD64 repair/preservation gates on the SHIPPED default build
#           (no REPAIR define: the matrix row itself must turn replay on), the
#           row-off control and four sabotage/corruption arms that must FAIL by
#           name, the dk/dv tail-guard gate and its sabotage, the broad fused
#           gate; then the reduced-shape HD64 training witness (default and
#           legacy builds) that the Apple column must equal.
# identity  tools/identity_break.py on the byte-LM / transformer / Samba lanes,
#           AFTER (this tree) and BEFORE (a copy with
#           tools/lm_attention_replay_vendors_before.py apply, i.e. main's
#           source), then --diff. Before and after build in the SAME path.
set -u
MODE=${1:?gates, identity or all}
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/attn-replay-extra
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
AFTER_SHA=$(sed -n 's/^commit=\([0-9a-f]*\).*/\1/p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
[ -n "$AFTER_SHA" ] || AFTER_SHA=$(sed -n 's/^commit_sha=\([0-9a-f]*\).*/\1/p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
BEFORE_SHA=$(sed -n 's/.*"_base": "\([0-9a-f]*\)".*/\1/p' tools/lm_attention_replay_vendors_before.json)
export MOJOLEARN_NUMERIC_MODE=identical
if [ -e /dev/kfd ]; then
    export MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942; COL=MOJOLEARN_COLUMN_AMD; LABEL=amd-gfx942
else
    export MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a; COL=MOJOLEARN_COLUMN_NVIDIA; LABEL=nvidia-sm_90a
    [ ! -x /usr/local/cuda/bin/ptxas ] || export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
fi
echo "mode=$MODE column=$MOJOLEARN_TARGET_COLUMN before=$BEFORE_SHA after=$AFTER_SHA" > "$OUT/leg.txt"
run() {
    name=$1; shift
    start=$(date +%s)
    if "$@" > "$OUT/$name.log" 2>&1; then code=0; else code=$?; fi
    printf '%s\t%s\t%s\n' "$name" "$code" "$(( $(date +%s)-start ))" >> "$OUT/status.tsv"
    return "$code"
}
# A check that must FAIL, and fail with the named line (printed, not counted).
must_fail() {
    name=$1; pattern=$2; shift 2
    if run "$name" "$@"; then echo "BLIND: $name passed" >> "$OUT/gates_verdict.txt"; return 1; fi
    if grep "$pattern" "$OUT/$name.log" >> "$OUT/gates_verdict.txt"; then
        echo "EXPECTED FAIL $name" >> "$OUT/gates_verdict.txt"
    else
        echo "WRONG FAILURE $name (no '$pattern')" >> "$OUT/gates_verdict.txt"; return 1
    fi
}
MR="pixi run mojo run -j 4 --target-accelerator $MOJOLEARN_GPU_ARCHS -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D $COL -I ."
M=transformer/checks/attention_masked_tail_check.mojo
T=transformer/checks/attention_tail_guard_check.mojo

gates() {
    must_fail g_rowoff 'masked-tail repair disabled: INERT' $MR -D MOJOLEARN_ATTN_LEGACY_CORNER=1 $M
    must_fail g_sab_z 'zdot repair differs' $MR -D MOJOLEARN_ATTN_REPAIR_SAB_Z=1 $M
    must_fail g_sab_dq 'dq repair differs' $MR -D MOJOLEARN_ATTN_REPAIR_SAB_DQ=1 $M
    must_fail g_corrupt_z 'zdot preserve differs' $MR -D MOJOLEARN_CHECK_CORRUPT_Z_PRESERVE=1 $M
    must_fail g_corrupt_dq 'dq preserve differs' $MR -D MOJOLEARN_CHECK_CORRUPT_DQ_PRESERVE=1 $M
    must_fail g_tail_sab 'accepted dk differs' $MR -D MOJOLEARN_ATTN_TAIL_GUARD_SABOTAGE=1 $T
    run g_masked_clean $MR $M && echo "PASS g_masked_clean" >> "$OUT/gates_verdict.txt"
    run g_tail_clean $MR $T && echo "PASS g_tail_clean" >> "$OUT/gates_verdict.txt"
    run g_fused_default $MR transformer/checks/transformer_fused_check.mojo && echo "PASS g_fused_default" >> "$OUT/gates_verdict.txt"
    # Reduced HD64 training witness: the shape the Apple column also runs.
    export PYTHONPATH="$ROOT/python:$ROOT"
    run corpus sh tools/fetch_corpus_enwik8.sh --check
    rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
    run r_build_base env MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 sh bindings/build.sh
    for arm in default legacy; do
        rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
        defines=""
        [ "$arm" != legacy ] || defines="-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1"
        run "r_build_$arm" env MOJOLEARN_BUILD_EXTRA_DEFINES="$defines" MOJOLEARN_COMPILE_JOBS=8 sh bindings/build_byte_lm.sh
        run "r_train_$arm" pixi run python tools/lm_ce_alias_probe.py --out "$OUT/reduced/$arm" \
            --shape 1 256 128 2 2 64 256 2 256 --steps 100 --tail 0 --smi-every 0 --witness-every 25 \
            --corpus training/corpus/enwik8/input.txt
    done
}

NEURAL=byte-lm,byte-lm-resident,byte-lm-host-infer,byte-lm-host-infer-threaded,byte-lm-host-train,transformer,transformer-window,samba,samba-untied-dropout-accum
build_all() {
    tag=$1
    rm -f python/mojolearn/identical/*.so
    for s in build build_byte_lm build_transformer build_training build_mamba; do
        run "${tag}_$s" env MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 sh "bindings/$s.sh"
    done
    for s in build_core_host build_byte_lm_host build_transformer_host build_training_host build_neural_host build_mamba_host; do
        run "${tag}_$s" env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 sh "bindings/$s.sh"
    done
    sha256sum python/mojolearn/identical/*.so > "$OUT/${tag}_bindings.sha256" 2>&1
}
identity() {
    export PYTHONPATH="$ROOT/python"
    build_all after
    run id_after env MOJOLEARN_COMMIT="$AFTER_SHA" pixi run python tools/identity_break.py --lanes "$NEURAL" \
        --vendor "$LABEL" --json "$OUT/identity_break.after.$LABEL.json"
    # BEFORE: main's source, built at the SAME path (sections are path dependent).
    run before_patch pixi run python tools/lm_attention_replay_vendors_before.py apply tools/lm_attention_replay_vendors_before.json || return 1
    grep -n "return column == COLUMN_NVIDIA$" checks/kernel_matrix.mojo >> "$OUT/before_patch.log"
    build_all before
    run id_before env MOJOLEARN_COMMIT="$BEFORE_SHA" pixi run python tools/identity_break.py --lanes "$NEURAL" \
        --vendor "$LABEL" --json "$OUT/identity_break.before.$LABEL.json"
    run id_diff pixi run python tools/identity_break.py --diff "$OUT/identity_break.before.$LABEL.json" \
        "$OUT/identity_break.after.$LABEL.json"
    pixi run python tools/lm_attention_replay_vendors_before.py restore tools/lm_attention_replay_vendors_before.json > "$OUT/after_restore.log" 2>&1
}
case "$MODE" in
    gates) gates ;;
    identity) identity ;;
    all) gates; identity ;;
    *) exit 9 ;;
esac
echo COMPLETE >> "$OUT/status.tsv"
