#!/bin/sh
# tools/byte_lm_export_fast/leg_2gpu_amd.sh -- lane/lm-export-overlap-amd:
# the MOJOLEARN_GEMM_LEG_EXTRA body of ONE Hot Aisle 2x MI300X VM
# (tools/hotaisle_leg.sh amd --rent --one-body, MOJOLEARN_HOTAISLE_SPEC=2gpu),
# column amd, gfx942, IDENTICAL. It proves the merged branch
# lane/lm-export-overlap (the byte LM export straight into the caller's
# memory, plus the overlapped per-step hash of tools/lm_segment.py run) on AMD
# and on TWO devices, before and after on the same box:
#
#   OLD  the tree with overlap_change.patch reversed: origin/main's binding,
#        package and runner (the five files of both lanes), its byte LM
#        binding built here from source
#   NEW  this commit, its byte LM binding built here from source
#
#   1. box_export.py OLD and NEW, on devices 0,1 (the pooled optimizer, T3's
#      configuration: m and v from two contexts) and on device 0 alone: the
#      sha256 of every exported array and the export seconds
#   2. replays of steps 101..103 from runs/t3/2026-09-22/A/1/ckpt_00000100.blm
#      held to A-1's witness chain on devices 0,1: NEW overlapped, NEW
#      --sync-hash, OLD (all --no-checkpoints), then NEW overlapped and NEW
#      --sync-hash WITH checkpoints (--boundary 103: checkpoints at 101 and
#      103, the one at 101 written while step 102 computes)
#   3. NEW --control split, one step, device 0
#
# Every replay takes the same --label and --segment so the chain lines can be
# compared field by field. Checkpoints stay in /root/xf_runs (never fetched);
# their sha256 comes home. Needs /root/urls/{tokens.json,ckpt.json,recipe.url}
# (tools/amd_step_time_urls.py) and /root/amd_in/A-1.chain.partial.jsonl
# pushed over ssh after the VM is up. POSIX sh, never set -e: a failure is a
# result and its log comes home.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/export-overlap-amd
BIN=/root/xf_bin
IN=/root/amd_in
RUNS=/root/xf_runs
EV=tools/byte_lm_export_fast
PATCH="$EV/overlap_change.patch"
mkdir -p "$OUT/builds" "$BIN" "$IN" "$RUNS"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
ST="$OUT/status.txt"
FILES="bindings/_mojolearn_byte_lm.mojo training/byte_lm_parallel.mojo training/checks/train_loop.mojo python/mojolearn/parallel_training.py tools/lm_segment.py"
LABEL=export-overlap-amd
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) archs=$MOJOLEARN_GPU_ARCHS"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; echo "sha_ni=$(grep -c sha_ni /proc/cpuinfo)"; free -g; } > "$OUT/host.txt" 2>&1
{ rocm-smi --showproductname --showuniqueid --showdriverversion; rocminfo | grep -E '^\s+Name:\s+gfx'; } > "$OUT/gpu.txt" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
sha256sum $FILES > "$OUT/files.new.sha256"

apply_patch() {  # $1 = -R for the old tree, empty for the new
    if command -v git > /dev/null 2>&1; then
        git apply $1 "$PATCH" && return 0
    fi
    command -v patch > /dev/null 2>&1 || { apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq patch > /dev/null 2>&1; }
    patch -p1 $1 --no-backup-if-mismatch < "$PATCH"
}

bind() {  # $1 tag
    mkdir -p "$BIN/out_$1"; rm -f "$BIN/out_$1/_mojolearn_byte_lm.so"
    t0=$(date +%s)
    MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_$1" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.$1.log" 2>&1; rc=$?
    [ "$rc" -eq 0 ] && cp "$BIN/out_$1/_mojolearn_byte_lm.so" "$BIN/byte_lm.$1.so" && sha256sum "$BIN/byte_lm.$1.so" > "$OUT/byte_lm.$1.so.sha256"
    say "bind $1 exit=$rc secs=$(( $(date +%s) - t0 ))"
    return $rc
}

use() { cp "$BIN/byte_lm.$1.so" python/mojolearn/identical/_mojolearn_byte_lm.so && say "binding now $1"; }

replay() {  # $1 tag, $2 name, $3 steps, $4 devices, rest: extra args
    tag=$1; name=$2; steps=$3; devs=$4; shift 4
    rm -rf "$RUNS/$name" "$OUT/$name"
    t0=$(date +%s)
    pixi run python tools/lm_segment.py run --recipe "$IN/recipe.json" --tokens /root/tokens_stream --from "$IN/ckpt_00000100.blm" \
        --steps "$steps" --devices "$devs" --route A --segment "$LABEL" --label "$LABEL" \
        --expect-chain "$IN/A-1.chain.partial.jsonl" --out "$RUNS/$name" "$@" > "$OUT/$name.log" 2>&1; rc=$?
    grep -v 'mbind memory' "$OUT/$name.log" > "$OUT/$name.clean.log"
    mkdir -p "$OUT/$name"
    for f in "$RUNS/$name"/*; do
        [ -f "$f" ] || continue
        case "$f" in *.blm) sha256sum "$f" >> "$OUT/$name/checkpoints.sha256"; ls -l "$f" >> "$OUT/$name/checkpoints.ls" ;; *) cp "$f" "$OUT/$name/" ;; esac
    done
    say "$name ($tag, devices $devs) exit=$rc secs=$(( $(date +%s) - t0 )): $(grep -E '^\[.*\] step |PASS|FAIL|DISAGREE|REFUSED|Error' "$OUT/$name.clean.log" | cut -c1-170 | tr '\n' '|' | cut -c1-900)"
}

measure() {  # $1 tag, $2 devices, $3 name
    t0=$(date +%s)
    pixi run python "$EV/box_export.py" "$IN/recipe.json" /root/tokens_stream "$IN/ckpt_00000100.blm" \
        "$IN/A-1.chain.partial.jsonl" "$OUT/export_$3.json" --devices "$2" --label "$3" > "$OUT/export_$3.log" 2>&1; rc=$?
    say "measure $3 ($1, devices $2) exit=$rc secs=$(( $(date +%s) - t0 )): $(python3 -c "import json;r=json.load(open('$OUT/export_$3.json'));print({k:v for k,v in r.items() if k.endswith('median_s')}, 'agree', r['forms_agree'], 'expected', r['equals_expected'])" 2>&1 | tail -1)"
}

# ---- inputs, in the background as soon as the URLs are pushed (at most 30 minutes)
fetch() {
    i=0
    while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/urls/ckpt.json ] || [ ! -s /root/urls/recipe.url ] || [ ! -s "$IN/A-1.chain.partial.jsonl" ]; do
        i=$((i + 1)); [ $i -gt 180 ] && { say "inputs never pushed"; return 4; }
        sleep 10
    done
    TOK=/root/tokens_stream; mkdir -p "$TOK"
    sed -n '/^cat > "\$OUT\/fetch_ranges.py"/,/^PY$/p' tools/lm_controls_body.sh | sed '1d;$d' > "$BIN/fetch_ranges.py"
    t0=$(date +%s)
    python3 "$BIN/fetch_ranges.py" /root/urls/tokens.json "$TOK" > "$OUT/tokens_fetch.log" 2>&1 || { say "tokens fetch FAILED"; return 4; }
    cat "$TOK"/tokens.i32.part?? > "$TOK/tokens.i32" && rm -f "$TOK"/tokens.i32.part??
    say "tokens ready in $(( $(date +%s) - t0 )) s: $(wc -c < "$TOK/tokens.i32") bytes"
    curl -fsS --retry 3 -o "$IN/recipe.json" "$(cat /root/urls/recipe.url)" && sha256sum "$IN/recipe.json" > "$OUT/recipe.sha256"
    python3 "$BIN/fetch_ranges.py" /root/urls/ckpt.json "$IN" > "$OUT/ckpt_fetch.log" 2>&1 || { say "ckpt fetch FAILED"; return 4; }
    sha256sum "$IN"/*.blm > "$OUT/ckpt.sha256"
    sha256sum "$IN/A-1.chain.partial.jsonl" > "$OUT/expect_chain.sha256"
    say "inputs ready: $(cut -c1-16 "$OUT/ckpt.sha256")"
    : > /root/xf_inputs
}
fetch &
FETCH=$!

# ---- the OLD tree, then the base bindings (the package imports) beside its byte LM binding
apply_patch -R > "$OUT/patch_reverse.log" 2>&1; rc=$?
sha256sum $FILES > "$OUT/files.old.sha256"
say "reverse patch exit=$rc"
[ "$rc" -eq 0 ] || { say "the old tree could not be made; stopping"; exit 3; }
t0=$(date +%s)
( MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1; say "build base exit=$? secs=$(( $(date +%s) - t0 ))" ) &
BASE=$!
bind old
wait $BASE
wait $FETCH
[ -e /root/xf_inputs ] || { say "inputs missing; stopping"; exit 4; }

# ---- OLD: origin/main's binding, package and runner
if [ -s "$BIN/byte_lm.old.so" ]; then
    use old
    measure old 0,1 old_2dev
    measure old 0 old_1dev
    replay old replay-old-2dev 3 0,1 --no-checkpoints
fi

# ---- NEW: this commit
apply_patch "" > "$OUT/patch_forward.log" 2>&1; rc=$?
sha256sum $FILES > "$OUT/files.new_again.sha256"
cmp -s "$OUT/files.new.sha256" "$OUT/files.new_again.sha256" && same=yes || same=NO
say "forward patch exit=$rc; tree back to the commit: $same"
[ "$same" = yes ] || { say "the new tree is not the commit; stopping"; exit 5; }
bind new || exit 6
use new
measure new 0,1 new_2dev
measure new 0 new_1dev
replay new replay-new-2dev 3 0,1 --no-checkpoints
replay new replay-new-sync-2dev 3 0,1 --no-checkpoints --sync-hash
replay new ckpt-new-2dev 3 0,1 --boundary 103
replay new ckpt-new-sync-2dev 3 0,1 --boundary 103 --sync-hash
replay new split-new 1 0 --control split

say "done"
exit 0
