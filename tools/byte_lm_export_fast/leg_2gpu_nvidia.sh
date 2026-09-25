#!/bin/sh
# tools/byte_lm_export_fast/leg_2gpu_nvidia.sh -- lane/lm-export-overlap-nvidia:
# the MOJOLEARN_GEMM_LEG_EXTRA body of one rented 2x H100 pod
# (MOJOLEARN_GEMM_LEG_GPU_COUNT=2 tools/gemm_remote_leg.sh nvidia). Proves the
# merge of lane/byte-lm-export-fast and lane/lm-segment-hash-overlap on TWO
# devices (the pooled optimizer, m and v read back from two contexts):
#
#   OLD  change.patch reversed on the binding and package (four files) and
#        origin/main's runner (runner_old_2gpu.patch applied), i.e. the five
#        files of the merge base; byte LM binding built here from source
#   NEW  this commit; byte LM binding built here from source
#
#   1. box_export.py --devices 0,1, OLD and NEW: sha256 of every exported
#      array (parameters, m, v, flags, gradient) and the export timings
#   2. replay 101..103 from runs/t3/2026-09-22/A/1/ckpt_00000100.blm on
#      devices 0,1 held to A-1's chain (--no-checkpoints --expect-chain):
#      NEW overlapped (default), NEW --sync-hash, OLD. Same route, segment
#      and label in all three so the lines compare field by field.
#      Then NEW with checkpoints (recipe with checkpoint_every 1, steps
#      101..102), overlapped and --sync-hash: checkpoint sha256 must match.
#   3. NEW --control split, one step (a one-device control by definition).
#
# Needs /root/urls/{tokens.json,ckpt.json,recipe.url} and
# /root/amd_in/A-1.chain.partial.jsonl pushed over ssh after the pod is up.
# POSIX sh, never set -e: a failure is a result and its log comes home.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/overlap-2gpu
BIN=/root/xf_bin
IN=/root/amd_in
CK=/root/xf_ckpt   # checkpoints stay OFF the fetched tree (about 2 GB each)
EV=tools/byte_lm_export_fast
DEV=0,1
SEG=overlap-2gpu
mkdir -p "$OUT/builds" "$BIN" "$IN" "$CK"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia
: "${MOJOLEARN_GPU_ARCHS:=sm_90a}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
ST="$OUT/status.txt"
FILES="bindings/_mojolearn_byte_lm.mojo training/byte_lm_parallel.mojo training/checks/train_loop.mojo python/mojolearn/parallel_training.py tools/lm_segment.py"
# The five files of the merge base (origin/main when both lanes branched),
# the same digests as bench/results/byte_lm_export_fast_2026-09-25/h100/files.old.sha256.
OLD_SHA="e969c5f1455ec8f4574f1a572d0dd4a68ee4944d54213d93fa1fec9b01779297
23544b014d4efb8bf7c839bbc512e92a09fa1714bb4bbc389ef12adc46e494d1
38920b21203be78853c6ba3dcee443e7197ddf11c49ab6cd302e316356074ad7
b85b3ea1ba0d09a0bd65e7c4c348fcf77b446bacad7e151557733a524ca4bdfe
7f649b2ddadff1759920177e12de04fb14880847126005d61da23658782e1d8f"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; echo "sha_ni=$(grep -c sha_ni /proc/cpuinfo)"; free -g; df -h /root; } > "$OUT/host.txt" 2>&1
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
nvidia-smi topo -m > "$OUT/gpu_topo.txt" 2>&1
say "gpus: $(wc -l < "$OUT/gpu.txt")"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
sha256sum $FILES > "$OUT/files.new.sha256"

gapply() {  # git apply, or patch(1) when the image has no git
    if command -v git > /dev/null 2>&1; then git apply "$@"; return $?; fi
    rev=""; ex=""; p=""
    for a in "$@"; do case "$a" in -R) rev=-R ;; --exclude=*) ex="${a#--exclude=}" ;; *) p="$a" ;; esac; done
    if [ -n "$ex" ]; then
        python3 - "$p" "$ex" "$BIN/filtered.patch" <<'PY'
import re, sys
src, ex, dst = sys.argv[1:]
parts = re.split(r'(?m)^(?=diff --git )', open(src).read())
open(dst, "w").write("".join(x for x in parts if not x.startswith("diff --git a/%s " % ex)))
PY
        p="$BIN/filtered.patch"
    fi
    patch -p1 $rev --no-backup-if-mismatch < "$p"
}

to_old() {
    gapply -R --exclude=tools/lm_segment.py "$EV/change.patch" && gapply "$EV/runner_old_2gpu.patch"
}
to_new() {
    gapply -R "$EV/runner_old_2gpu.patch" && gapply --exclude=tools/lm_segment.py "$EV/change.patch"
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

run_seg() {  # $1 tag, $2 name, $3 out dir, $4 recipe, $5 steps, $6 devices, rest: extra args
    tag=$1; name=$2; dir=$3; rec=$4; steps=$5; dev=$6; shift 6
    rm -rf "$dir"
    t0=$(date +%s)
    pixi run python tools/lm_segment.py run --recipe "$rec" --tokens /root/tokens_stream --from "$IN/ckpt_00000100.blm" \
        --steps "$steps" --devices "$dev" --route A --segment "$SEG" --label "$SEG" \
        --expect-chain "$IN/A-1.chain.partial.jsonl" --out "$dir" "$@" > "$OUT/$name.log" 2>&1; rc=$?
    grep -v 'mbind memory' "$OUT/$name.log" > "$OUT/$name.clean.log"
    say "$name ($tag) exit=$rc secs=$(( $(date +%s) - t0 )): $(grep -E '^\[.*\] step |PASS|FAIL|DISAGREE|REFUSED|Error' "$OUT/$name.clean.log" | cut -c1-170 | tr '\n' '|' | cut -c1-1000)"
}

measure() {  # $1 tag
    t0=$(date +%s)
    pixi run python "$EV/box_export.py" "$IN/recipe.json" /root/tokens_stream "$IN/ckpt_00000100.blm" \
        "$IN/A-1.chain.partial.jsonl" "$OUT/export_$1.json" --devices "$DEV" --label "$1" > "$OUT/export_$1.log" 2>&1; rc=$?
    say "measure $1 exit=$rc secs=$(( $(date +%s) - t0 )): $(python3 -c "import json;r=json.load(open('$OUT/export_$1.json'));print(r['devices'], {k:v for k,v in r.items() if k.endswith('median_s')}, 'agree', r['forms_agree'], 'expected', r['equals_expected'])" 2>&1 | tail -1)"
}

# ---- inputs, in the background as soon as the URLs are pushed
fetch() {
    while [ ! -s /root/urls/tokens.json ] || [ ! -s "$IN/A-1.chain.partial.jsonl" ]; do sleep 10; done
    TOK=/root/tokens_stream; mkdir -p "$TOK"
    sed -n '/^cat > "\$OUT\/fetch_ranges.py"/,/^PY$/p' tools/lm_controls_body.sh | sed '1d;$d' > "$BIN/fetch_ranges.py"
    t0=$(date +%s)
    python3 "$BIN/fetch_ranges.py" /root/urls/tokens.json "$TOK" > "$OUT/tokens_fetch.log" 2>&1 || { say "tokens fetch FAILED"; return 4; }
    cat "$TOK"/tokens.i32.part?? > "$TOK/tokens.i32" && rm -f "$TOK"/tokens.i32.part??
    say "tokens ready in $(( $(date +%s) - t0 )) s: $(wc -c < "$TOK/tokens.i32") bytes"
    curl -fsS --retry 3 -o "$IN/recipe.json" "$(cat /root/urls/recipe.url)" && sha256sum "$IN/recipe.json" > "$OUT/recipe.sha256"
    python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); r["checkpoint_every"] = 1; open(sys.argv[2], "w").write(json.dumps(r, indent=1) + "\n")' \
        "$IN/recipe.json" "$IN/recipe_every1.json" && sha256sum "$IN/recipe_every1.json" > "$OUT/recipe_every1.sha256"
    python3 "$BIN/fetch_ranges.py" /root/urls/ckpt.json "$IN" > "$OUT/ckpt_fetch.log" 2>&1 || { say "ckpt fetch FAILED"; return 4; }
    sha256sum "$IN"/*.blm > "$OUT/ckpt.sha256"
    sha256sum "$IN/A-1.chain.partial.jsonl" > "$OUT/expect_chain.sha256"
    say "inputs ready: $(cut -c1-16 "$OUT/ckpt.sha256")"
    : > /root/xf_inputs
}
fetch &
FETCH=$!

# ---- the OLD tree (the merge base's five files), then the base bindings beside its byte LM binding
to_old > "$OUT/patch_to_old.log" 2>&1; rc=$?
sha256sum $FILES > "$OUT/files.old.sha256"
if [ "$(cut -d' ' -f1 "$OUT/files.old.sha256")" = "$OLD_SHA" ]; then same=yes; else same=NO; fi
say "to_old exit=$rc; tree is the merge base's five files: $same"
[ "$rc" -eq 0 ] && [ "$same" = yes ] || { say "the old tree could not be made; stopping"; exit 3; }
t0=$(date +%s)
( MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1; say "build base exit=$? secs=$(( $(date +%s) - t0 ))" ) &
BASE=$!
bind old
wait $BASE
wait $FETCH
[ -e /root/xf_inputs ] || { say "inputs missing; stopping"; exit 4; }

# ---- OLD: the merge base's binding, package and (synchronous) runner, two devices
if [ -s "$BIN/byte_lm.old.so" ]; then
    use old
    measure old
    run_seg old replay-old "$OUT/replay-old" "$IN/recipe.json" 3 "$DEV" --no-checkpoints
fi

# ---- NEW: this commit
to_new > "$OUT/patch_to_new.log" 2>&1; rc=$?
sha256sum $FILES > "$OUT/files.new_again.sha256"
cmp -s "$OUT/files.new.sha256" "$OUT/files.new_again.sha256" && same=yes || same=NO
say "to_new exit=$rc; tree back to the commit: $same"
[ "$same" = yes ] || { say "the new tree is not the commit; stopping"; exit 5; }
bind new || exit 6
use new
measure new
run_seg new replay-new-overlap "$OUT/replay-new-overlap" "$IN/recipe.json" 3 "$DEV" --no-checkpoints
run_seg new replay-new-sync "$OUT/replay-new-sync" "$IN/recipe.json" 3 "$DEV" --no-checkpoints --sync-hash

# checkpoints at 101 and 102 (checkpoint_every 1), overlapped then synchronous; files kept off the fetch
for mode in overlap sync; do
    extra=""; [ "$mode" = sync ] && extra=--sync-hash
    run_seg new "ckpt-$mode" "$CK/ckpt-$mode" "$IN/recipe_every1.json" 2 "$DEV" $extra
    mkdir -p "$OUT/ckpt-$mode"
    cp "$CK/ckpt-$mode"/chain.jsonl "$CK/ckpt-$mode"/manifest.tsv "$CK/ckpt-$mode"/segment.json "$CK/ckpt-$mode"/log.txt "$OUT/ckpt-$mode/" 2>/dev/null
    ( cd "$CK/ckpt-$mode" && for f in *.blm; do echo "$(stat -c %s "$f") $(sha256sum "$f")"; done ) > "$OUT/ckpt-$mode.sha256" 2>&1
done
if cmp -s "$OUT/ckpt-overlap.sha256" "$OUT/ckpt-sync.sha256"; then ck=EQUAL; else ck=DIFFER; fi
say "checkpoints overlapped vs sync-hash: $ck: $(tr '\n' ' ' < "$OUT/ckpt-overlap.sha256")"
[ "$ck" = EQUAL ] && rm -f "$CK"/ckpt-*/*.blm

run_seg new split-new "$OUT/split-new" "$IN/recipe.json" 1 0 --control split

# ---- the chains, field by field (everything but the measured seconds and the prev line digest)
pixi run python - "$OUT" > "$OUT/chains_compare.txt" 2>&1 <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
WALL = {"seconds", "hash_seconds", "prev"}
def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
ref_name = "replay-new-overlap"
ref = lines(out / ref_name / "chain.jsonl")
ok = True
for name in ("replay-new-sync", "replay-old", "ckpt-overlap", "ckpt-sync"):
    other = lines(out / name / "chain.jsonl")
    n = min(len(ref), len(other))
    diffs = []
    for a, b in zip(ref[:n], other[:n]):
        if set(a) != set(b):
            diffs.append((a.get("step"), "key set", sorted(set(a) ^ set(b))))
        for k in sorted(set(a) | set(b)):
            if k in WALL:
                continue
            if a.get(k) != b.get(k):
                diffs.append((a.get("step"), k, (a.get(k), b.get(k))))
    want = 2 if name.startswith("ckpt") else 3
    if len(other) != want or len(ref) != 3:
        diffs.append(("-", "line count", (len(ref), len(other))))
    ok = ok and not diffs
    fields = sorted(set(ref[0]) - WALL)
    print("%s vs %s: %d line(s), %d field(s) each (%s), %s" % (ref_name, name, n, len(fields), ",".join(fields),
          "EQUAL" if not diffs else "DIFFER %r" % diffs))
for r in ref:
    print("step %s state %s grad %s lr %s" % (r["step"], r["state_sha256"], r["gradient_sha256"], r["lr_f32_hex"]))
print("ALL EQUAL" if ok else "SOME DIFFER")
PY
say "chains: $(tail -1 "$OUT/chains_compare.txt")"
pixi run python tools/lm_segment.py compare "$OUT/replay-new-overlap/chain.jsonl" "$OUT/replay-new-sync/chain.jsonl" "$OUT/replay-old/chain.jsonl" > "$OUT/lm_segment_compare.txt" 2>&1
say "lm_segment compare exit=$?: $(tail -2 "$OUT/lm_segment_compare.txt" | tr '\n' ' ' | cut -c1-300)"

say "done"
exit 0
