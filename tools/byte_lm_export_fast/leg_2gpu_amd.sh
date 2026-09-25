#!/bin/sh
# tools/byte_lm_export_fast/leg_2gpu_amd.sh -- lane/lm-export-overlap-amd:
# the MOJOLEARN_GEMM_LEG_EXTRA body of ONE Hot Aisle 2x MI300X VM
# (tools/hotaisle_leg.sh amd --rent --one-body, MOJOLEARN_HOTAISLE_SPEC=2gpu),
# column amd, gfx942, IDENTICAL. LIGHT validation of the merged branch
# lane/lm-export-overlap (byte LM export straight into the caller's memory,
# overlapped per-step hash, progress PUTs for a resume), everything once, on
# TWO devices, NEW only (the OLD digests are the H100's,
# bench/results/byte_lm_export_fast_2026-09-25/h100/export_old.json):
#
#   1. box_export.py on devices 0,1 (pooled optimizer, T3's configuration:
#      m and v from two contexts): the sha256 of every exported array after
#      step 101 from runs/t3/2026-09-22/A/1/ckpt_00000100.blm, and one timing
#   2. ONE run of `lm_segment.py run` (default overlapped loop) on devices
#      0,1 from ckpt 100, steps 101..110 held to A-1's witness chain, under
#      the T3 recipe with checkpoint_every set to 2 (checkpoint_every is not
#      part of the data schedule a checkpoint carries), uploading to presigned
#      PUTs under a SCRATCH R2 prefix (never runs/t3). After the progress PUT
#      that follows the second checkpoint (chain.progress.jsonl to step 105)
#      the process is killed with SIGKILL, so no final chain.jsonl reaches R2.
#      The Mac then runs lm_run_driver.record_partial against that prefix.
#
# Checkpoints stay in /root/xf_runs (never fetched); their sha256 comes home.
# Needs /root/urls/{tokens.json,ckpt.json,recipe.url,uploads.json}
# (tools/amd_step_time_urls.py; uploads.json minted on the Mac) and
# /root/amd_in/A-1.chain.partial.jsonl pushed over ssh after the VM is up.
# POSIX sh, never set -e: a failure is a result and its log comes home.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/export-overlap-amd
BIN=/root/xf_bin
IN=/root/amd_in
RUNS=/root/xf_runs
EV=tools/byte_lm_export_fast
mkdir -p "$OUT/builds" "$BIN" "$IN" "$RUNS"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
ST="$OUT/status.txt"
# the runner's processes by command line, from /proc (the image may have no pgrep)
runner_pids() {
    for d in /proc/[0-9]*; do
        tr '\0' ' ' < "$d/cmdline" 2>/dev/null | grep -q 'tools/lm_segment.py run --recipe' && echo "${d#/proc/}"
    done
}
FILES="bindings/_mojolearn_byte_lm.mojo training/byte_lm_parallel.mojo training/checks/train_loop.mojo python/mojolearn/parallel_training.py tools/lm_segment.py"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) archs=$MOJOLEARN_GPU_ARCHS"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; echo "sha_ni=$(grep -c sha_ni /proc/cpuinfo)"; free -g; } > "$OUT/host.txt" 2>&1
{ rocm-smi --showproductname --showuniqueid --showdriverversion; rocminfo | grep -E '^ +Name: +gfx'; } > "$OUT/gpu.txt" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
sha256sum $FILES > "$OUT/files.new.sha256"
# two devices whenever the box has two (the point of the leg); a one-GPU box
# (a DigitalOcean fallback) runs the same steps on device 0 and says so
NGPU=$(rocminfo 2>/dev/null | grep -c -E '^ +Name: +gfx')
if [ "$NGPU" -ge 2 ]; then DEVS=0,1; else DEVS=0; fi
say "gpu agents=$NGPU devices=$DEVS"

# ---- inputs, in the background as soon as the URLs are pushed (at most 30 minutes)
fetch() {
    i=0
    while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/urls/ckpt.json ] || [ ! -s /root/urls/recipe.url ] \
          || [ ! -s /root/urls/uploads.json ] || [ ! -s "$IN/A-1.chain.partial.jsonl" ]; do
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
    python3 -c "import json,sys;r=json.load(open(sys.argv[1]));r['checkpoint_every']=2;open(sys.argv[2],'w').write(json.dumps(r))" \
        "$IN/recipe.json" "$IN/recipe_ck2.json" && sha256sum "$IN/recipe_ck2.json" >> "$OUT/recipe.sha256"
    python3 "$BIN/fetch_ranges.py" /root/urls/ckpt.json "$IN" > "$OUT/ckpt_fetch.log" 2>&1 || { say "ckpt fetch FAILED"; return 4; }
    sha256sum "$IN"/*.blm > "$OUT/ckpt.sha256"
    sha256sum "$IN/A-1.chain.partial.jsonl" > "$OUT/expect_chain.sha256"
    say "inputs ready: $(cut -c1-16 "$OUT/ckpt.sha256")"
    : > /root/xf_inputs
}
fetch &
FETCH=$!

# ---- the base bindings (the package imports) beside the NEW byte LM binding
t0=$(date +%s)
( MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1; say "build base exit=$? secs=$(( $(date +%s) - t0 ))" ) &
BASE=$!
mkdir -p "$BIN/out_new"
t1=$(date +%s)
MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_new" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.new.log" 2>&1; rc=$?
say "bind new exit=$rc secs=$(( $(date +%s) - t1 ))"
wait $BASE
[ "$rc" -eq 0 ] || exit 6
cp "$BIN/out_new/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum python/mojolearn/identical/_mojolearn_byte_lm.so > "$OUT/byte_lm.new.so.sha256"
wait $FETCH
[ -e /root/xf_inputs ] || { say "inputs missing; stopping"; exit 4; }

# ---- 1. the export, two devices, once
t0=$(date +%s)
pixi run python "$EV/box_export.py" "$IN/recipe.json" /root/tokens_stream "$IN/ckpt_00000100.blm" \
    "$IN/A-1.chain.partial.jsonl" "$OUT/export_new_2dev.json" --devices "$DEVS" --label "new_$DEVS" > "$OUT/export_new_2dev.log" 2>&1; rc=$?
say "export new devices $DEVS exit=$rc secs=$(( $(date +%s) - t0 )): $(python3 -c "import json;r=json.load(open('$OUT/export_new_2dev.json'));print({k:v for k,v in r.items() if k.endswith('median_s')}, 'agree', r['forms_agree'], 'expected', r['equals_expected'])" 2>&1 | tail -1)"

# ---- 2. one overlapped run, two devices, checkpoints every 2, progress PUTs, SIGKILL after step 105's
NAME=run-new-2dev
rm -rf "$RUNS/$NAME"; mkdir -p "$RUNS/$NAME"
t0=$(date +%s)
pixi run python tools/lm_segment.py run --recipe "$IN/recipe_ck2.json" --tokens /root/tokens_stream --from "$IN/ckpt_00000100.blm" \
    --steps 10 --devices "$DEVS" --route A --segment 4 --label export-overlap-amd \
    --expect-chain "$IN/A-1.chain.partial.jsonl" --upload-urls /root/urls/uploads.json --out "$RUNS/$NAME" > "$OUT/$NAME.log" 2>&1 &
RUNPID=$!
killed=no
while kill -0 $RUNPID 2>/dev/null; do
    if grep -q 'progress: chain.progress.jsonl to step 105' "$RUNS/$NAME/log.txt" 2>/dev/null; then
        for p in $(runner_pids); do kill -9 "$p" 2>/dev/null; done
        killed="yes $(date -u +%H:%M:%S)"
        break
    fi
    [ $(( $(date +%s) - t0 )) -gt 2400 ] && { say "$NAME never reached step 105's progress PUT in 40 minutes"; break; }
    sleep 1
done
wait $RUNPID 2>/dev/null; rc=$?
sleep 2
say "$NAME exit=$rc killed=$killed secs=$(( $(date +%s) - t0 )) survivors=$(runner_pids | wc -l | tr -d ' ')"
grep -v 'mbind memory' "$OUT/$NAME.log" > "$OUT/$NAME.clean.log"
mkdir -p "$OUT/$NAME"
for f in "$RUNS/$NAME"/*; do
    [ -f "$f" ] || continue
    case "$f" in *.blm) sha256sum "$f" >> "$OUT/$NAME/checkpoints.sha256"; ls -l "$f" >> "$OUT/$NAME/checkpoints.ls" ;; *) cp "$f" "$OUT/$NAME/" ;; esac
done
say "$NAME: $(grep -E 'step |checkpoint |uploaded|progress:|PASS|FAIL|DISAGREE|REFUSED|Error' "$OUT/$NAME/log.txt" | cut -c1-150 | tr '\n' '|' | cut -c1-2000)"
say "done"
exit 0
