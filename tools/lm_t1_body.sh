#!/bin/sh
# tools/lm_t1_body.sh -- T1 of docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md: the
# segment runner at the TARGET shape on one rented box, as a
# MOJOLEARN_GEMM_LEG_EXTRA body. Placeholders are substituted by the launcher
# (RunPod passes no environment to a body):
#
#   @ARM@         nvidia | amd
#   @DEVICES@     "0" or "0,1": the devices the two-device arm may use
#   @UPLOADS@     the JSON {file name: presigned PUT URL} for the NVIDIA arm's
#                 checkpoints and chain, written to a file here; "" for none
#   @CKPT0_URL@ @CKPT1_URL@ @CHAIN_URL@   AMD arm only: presigned GETs of the
#                 NVIDIA arm's seed checkpoint, boundary-minus-two checkpoint
#                 and chain, and their pinned sha256s @CKPT0_SHA@ @CKPT1_SHA@
#   @ENDURANCE@   AMD arm: batch-4 steps of the endurance probe (0 skips it)
#   @ENDURANCE_BUDGET@  seconds it may run
#
# What the NVIDIA arm produces: the seed checkpoint (step 0), three K=64
# optimizer steps on ONE device with the boundary at 3 (so checkpoints 1 and
# 3), every step hashed, the two checkpoints pushed to R2; the same three
# steps on TWO devices held to the one-device chain; an arrival replay from
# checkpoint 1 on the same box. Times for the step, the hashing, the save
# and the upload are in segment.json.
#
# What the AMD arm produces: the NVIDIA arm's checkpoint 0 and 1 and chain,
# fetched and verified; three steps from checkpoint 0 held to the NVIDIA
# chain (the cross-vendor comparison at the target shape); two steps from
# checkpoint 1 held to the same chain (the arrival replay across vendors);
# then the batch-4 endurance probe on the repaired attention path (E7).
#
# POSIX sh. Never `set -e`: a failure is a result and its log comes home.
set -u
ARM="@ARM@"
DEVICES="@DEVICES@"
ENDURANCE="@ENDURANCE@"
ENDURANCE_BUDGET="@ENDURANCE_BUDGET@"
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-t1-$ARM
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH="$ROOT/python:$ROOT"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "arm=$ARM devices=$DEVICES started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
uname -a > "$OUT/uname.txt" 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1 || rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1

case "$ARM" in
    nvidia)
        export MOJOLEARN_TARGET_COLUMN=nvidia
        if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
            cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
            case "$cap" in
                9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
                [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
                *) say "arch unknown (compute_cap='$cap')"; exit 2 ;;
            esac
        fi ;;
    amd)
        export MOJOLEARN_TARGET_COLUMN=amd
        : "${MOJOLEARN_GPU_ARCHS:=gfx942}" ;;
    *) say "unknown arm $ARM"; exit 2 ;;
esac
export MOJOLEARN_GPU_ARCHS
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"

# ---- build the base and byte-LM bindings from this commit ----
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
_t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1; _rc=$?
say "build base exit=$_rc secs=$(( $(date +%s) - _t0 ))"
[ "$_rc" -eq 0 ] || { say "base build failed; nothing run"; exit 1; }
_t0=$(date +%s)
MOJOLEARN_BUILD_EXTRA_DEFINES="" sh bindings/build_byte_lm.sh > "$OUT/build_byte_lm.log" 2>&1; _rc=$?
say "build byte_lm exit=$_rc secs=$(( $(date +%s) - _t0 ))"
[ "$_rc" -eq 0 ] || { say "byte_lm build failed; nothing run"; exit 1; }
pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1 || { pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1; say "numpy pip exit=$?"; }
pixi run python -c 'import mojolearn, sys; b = mojolearn._backend.binding("_mojolearn_byte_lm", "identical"); print("vendor", b.byte_lm_vendor(), "set_lr", callable(getattr(b, "byte_lm_parallel_set_lr", None)))' > "$OUT/binding.txt" 2>&1
say "binding: $(cat "$OUT/binding.txt" | tail -1)"

# ---- the token stream (enwik8 ids under the 50,257 vocabulary, staged from R2) ----
TOK=$(dirname "$(find /root -name tokens.i32 -path '*tokens*' 2>/dev/null | head -1)")
[ -f "$TOK/manifest.json" ] || { say "no staged token stream with a manifest under /root"; exit 3; }
say "tokens=$TOK"
S=tools/lm_segment.py
R="$OUT/recipe.json"
pixi run python $S recipe --out "$R" --shape 4 2048 768 12 12 64 2048 12 0 --tokens "$TOK" --shards 64 --steps 10 \
    --peak-lr 6e-4 --warmup 2 --checkpoint-every 100 > "$OUT/recipe.log" 2>&1; say "recipe exit=$?"

run() {  # $1 name, rest: lm_segment run args
    _n="$1"; shift
    _t0=$(date +%s)
    pixi run python $S run --recipe "$R" --tokens "$TOK" --out "$OUT/$_n" --label "$ARM-$_n" "$@" > "$OUT/$_n.log" 2>&1
    say "$_n exit=$? secs=$(( $(date +%s) - _t0 ))"
    grep -E "PASS|FAIL|REFUSED|DISAGREE" "$OUT/$_n.log" | tail -2 >> "$ST"
}

case "$ARM" in
nvidia)
    _t0=$(date +%s)
    pixi run python $S init --recipe "$R" --tokens "$TOK" --out "$OUT/ckpt_00000000.blm" > "$OUT/init.log" 2>&1
    say "init exit=$? secs=$(( $(date +%s) - _t0 )): $(tail -1 "$OUT/init.log")"
    cat > "$OUT/uploads.json" <<'UPLOADS'
@UPLOADS@
UPLOADS
    _up=""; [ -s "$OUT/uploads.json" ] && [ "$(head -c 1 "$OUT/uploads.json")" = "{" ] && _up="--upload-urls $OUT/uploads.json"
    # shellcheck disable=SC2086
    run one --from "$OUT/ckpt_00000000.blm" --steps 3 --boundary 3 --devices 0 --route A --segment t1 --record-window 0:1 $_up
    if [ "$DEVICES" != "0" ]; then
        run two --from "$OUT/ckpt_00000000.blm" --steps 3 --devices "$DEVICES" --route A --segment t1-two --no-checkpoints --expect-chain "$OUT/one/chain.jsonl"
    fi
    run replay --from "$OUT/one/ckpt_00000001.blm" --steps 2 --devices 0 --route A --segment t1-replay --no-checkpoints --expect-chain "$OUT/one/chain.jsonl"
    # the seed checkpoint is uploaded too (its URL is under the file's name)
    if [ -n "$_up" ]; then
        _u=$(pixi run python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ckpt_00000000.blm",""))' "$OUT/uploads.json")
        if [ -n "$_u" ]; then curl -fsS --retry 3 -T "$OUT/ckpt_00000000.blm" "$_u" > "$OUT/upload_seed.log" 2>&1; say "upload seed exit=$?"; fi
    fi
    sha256sum "$OUT"/ckpt_00000000.blm "$OUT"/one/ckpt_*.blm > "$OUT/checkpoints.sha256" 2>&1
    rm -f "$OUT"/ckpt_00000000.blm "$OUT"/one/ckpt_*.blm   # 1.95 GB each; pinned above and in R2, never fetched home
    ;;
amd)
    mkdir -p "$OUT/from-nvidia"
    _t0=$(date +%s)
    curl -fsS --retry 3 -o "$OUT/from-nvidia/ckpt_00000000.blm" '@CKPT0_URL@' > "$OUT/fetch.log" 2>&1; say "fetch ckpt0 exit=$?"
    curl -fsS --retry 3 -o "$OUT/from-nvidia/ckpt_00000001.blm" '@CKPT1_URL@' >> "$OUT/fetch.log" 2>&1; say "fetch ckpt1 exit=$?"
    curl -fsS --retry 3 -o "$OUT/from-nvidia/chain.jsonl" '@CHAIN_URL@' >> "$OUT/fetch.log" 2>&1; say "fetch chain exit=$? secs=$(( $(date +%s) - _t0 ))"
    printf '%s  %s\n%s  %s\n' '@CKPT0_SHA@' "$OUT/from-nvidia/ckpt_00000000.blm" '@CKPT1_SHA@' "$OUT/from-nvidia/ckpt_00000001.blm" \
        | sha256sum -c - > "$OUT/from-nvidia/verified.log" 2>&1; _rc=$?
    say "nvidia checkpoints verified exit=$_rc"
    [ "$_rc" -eq 0 ] || { say "the NVIDIA checkpoints did not verify; nothing compared"; exit 4; }
    run xvendor --from "$OUT/from-nvidia/ckpt_00000000.blm" --steps 3 --devices 0 --route B --segment t1 --no-checkpoints --expect-chain "$OUT/from-nvidia/chain.jsonl" --record-window 0:1
    run arrival --from "$OUT/from-nvidia/ckpt_00000001.blm" --steps 2 --devices 0 --route B --segment t1-arrival --no-checkpoints --expect-chain "$OUT/from-nvidia/chain.jsonl"
    rm -f "$OUT"/from-nvidia/*.blm
    if [ "$ENDURANCE" -gt 0 ] 2>/dev/null; then
        CORPUS=training/corpus/enwik8/input.txt
        if [ -f "$CORPUS" ]; then
            _t0=$(date +%s)
            timeout -k 30 "$ENDURANCE_BUDGET" pixi run python tools/lm_ce_alias_probe.py --out "$OUT/endurance" \
                --shape 4 2048 768 12 12 64 2048 12 50257 --steps "$ENDURANCE" --tail 0 --corpus "$CORPUS" \
                --smi-every 10 --witness-every 0 > "$OUT/endurance.log" 2>&1
            say "endurance exit=$? secs=$(( $(date +%s) - _t0 ))"
            pixi run python tools/lm_attention_endurance_check.py "$OUT/endurance/result.json" > "$OUT/endurance_verdict.log" 2>&1
            say "endurance verdict exit=$?: $(tail -3 "$OUT/endurance_verdict.log" | tr '\n' ' ' | cut -c1-300)"
        else
            say "endurance skipped: no enwik8 text staged"
        fi
    fi
    ;;
esac
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
