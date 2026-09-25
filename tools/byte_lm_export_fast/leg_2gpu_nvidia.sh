#!/bin/sh
# tools/byte_lm_export_fast/leg_2gpu_nvidia.sh -- lane/lm-export-overlap-nvidia:
# the MOJOLEARN_GEMM_LEG_EXTRA body of one rented 2x H100 pod
# (MOJOLEARN_GEMM_LEG_GPU_COUNT=2 tools/gemm_remote_leg.sh nvidia). A LIGHT
# proof of the merge of lane/byte-lm-export-fast and
# lane/lm-segment-hash-overlap on TWO devices (the pooled optimizer, m and v
# read back from two contexts). Only this commit's byte LM binding is built.
# Everything runs once.
#
#   1. box_export.py --devices 0,1: the sha256 of every exported array after
#      step 101 (held on the Mac to the OLD binding's one-H100 export_old.json)
#      and the export timings.
#   2. ONE replay from runs/t3/2026-09-22/A/1/ckpt_00000100.blm on devices 0,1,
#      the default (overlapped) loop, held to A-1's chain (--expect-chain),
#      under the recipe with checkpoint_every 2, with presigned PUTs to a
#      SCRATCH R2 prefix (/root/urls/upload.json, minted on the Mac, never
#      under runs/t3/). Checkpoints 102 and 104 upload, and after each the
#      progress chain and manifest. Once the log shows the progress chain PUT
#      for step 105 (after the second checkpoint), the run is killed with
#      SIGKILL, so no final chain.jsonl or segment.json is ever uploaded. The
#      Mac then resumes it from R2 alone (lm_run_driver.record_partial).
#
# Needs /root/urls/{tokens.json,ckpt.json,recipe.url,upload.json} and
# /root/amd_in/A-1.chain.partial.jsonl pushed over ssh after the pod is up.
# POSIX sh, never set -e: a failure is a result and its log comes home.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/overlap-2gpu
BIN=/root/xf_bin
IN=/root/amd_in
CK=/root/xf_ckpt   # the run's checkpoints stay OFF the fetched tree (about 2 GB each)
EV=tools/byte_lm_export_fast
DEV=0,1
STEPS=9            # 101..109; killed after step 105's progress PUT, well before the end
KILL_AT=105
mkdir -p "$OUT/builds" "$BIN" "$IN" "$CK"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia
: "${MOJOLEARN_GPU_ARCHS:=sm_90a}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
ST="$OUT/status.txt"
FILES="bindings/_mojolearn_byte_lm.mojo training/byte_lm_parallel.mojo training/checks/train_loop.mojo python/mojolearn/parallel_training.py tools/lm_segment.py"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; echo "sha_ni=$(grep -c sha_ni /proc/cpuinfo)"; free -g; df -h /root; } > "$OUT/host.txt" 2>&1
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
nvidia-smi topo -m > "$OUT/gpu_topo.txt" 2>&1
say "gpus: $(wc -l < "$OUT/gpu.txt")"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
sha256sum $FILES > "$OUT/files.new.sha256"

# ---- inputs, in the background as soon as the URLs are pushed
fetch() {
    while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/urls/upload.json ] || [ ! -s "$IN/A-1.chain.partial.jsonl" ]; do sleep 10; done
    TOK=/root/tokens_stream; mkdir -p "$TOK"
    sed -n '/^cat > "\$OUT\/fetch_ranges.py"/,/^PY$/p' tools/lm_controls_body.sh | sed '1d;$d' > "$BIN/fetch_ranges.py"
    t0=$(date +%s)
    python3 "$BIN/fetch_ranges.py" /root/urls/tokens.json "$TOK" > "$OUT/tokens_fetch.log" 2>&1 || { say "tokens fetch FAILED"; return 4; }
    cat "$TOK"/tokens.i32.part?? > "$TOK/tokens.i32" && rm -f "$TOK"/tokens.i32.part??
    say "tokens ready in $(( $(date +%s) - t0 )) s: $(wc -c < "$TOK/tokens.i32") bytes"
    curl -fsS --retry 3 -o "$IN/recipe.json" "$(cat /root/urls/recipe.url)" && sha256sum "$IN/recipe.json" > "$OUT/recipe.sha256"
    python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); r["checkpoint_every"] = 2; open(sys.argv[2], "w").write(json.dumps(r, indent=1) + "\n")' \
        "$IN/recipe.json" "$IN/recipe_every2.json" && sha256sum "$IN/recipe_every2.json" > "$OUT/recipe_every2.sha256"
    python3 "$BIN/fetch_ranges.py" /root/urls/ckpt.json "$IN" > "$OUT/ckpt_fetch.log" 2>&1 || { say "ckpt fetch FAILED"; return 4; }
    sha256sum "$IN"/*.blm > "$OUT/ckpt.sha256"
    sha256sum "$IN/A-1.chain.partial.jsonl" > "$OUT/expect_chain.sha256"
    python3 -c 'import json, sys; print(" ".join(sorted(json.load(open(sys.argv[1])))))' /root/urls/upload.json > "$OUT/upload_keys.txt"
    say "inputs ready: $(cut -c1-16 "$OUT/ckpt.sha256"); upload keys: $(cat "$OUT/upload_keys.txt")"
    : > /root/xf_inputs
}
fetch &
FETCH=$!

# ---- the base bindings (the package imports) beside this commit's byte LM binding
t0=$(date +%s)
( MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1; say "build base exit=$? secs=$(( $(date +%s) - t0 ))" ) &
BASE=$!
mkdir -p "$BIN/out_new"
t1=$(date +%s)
MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_new" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.new.log" 2>&1; rc=$?
say "bind new exit=$rc secs=$(( $(date +%s) - t1 ))"
wait $BASE
[ "$rc" -eq 0 ] || { say "the byte LM binding did not build; stopping"; exit 6; }
cp "$BIN/out_new/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum python/mojolearn/identical/_mojolearn_byte_lm.so > "$OUT/byte_lm.new.so.sha256"
wait $FETCH
[ -e /root/xf_inputs ] || { say "inputs missing; stopping"; exit 4; }

# ---- 1. the export on two devices
t0=$(date +%s)
pixi run python "$EV/box_export.py" "$IN/recipe.json" /root/tokens_stream "$IN/ckpt_00000100.blm" \
    "$IN/A-1.chain.partial.jsonl" "$OUT/export_new_2gpu.json" --devices "$DEV" --label new-2gpu > "$OUT/export_new_2gpu.log" 2>&1; rc=$?
say "export new 2gpu exit=$rc secs=$(( $(date +%s) - t0 )): $(python3 -c "import json;r=json.load(open('$OUT/export_new_2gpu.json'));print(r['devices'], {k:v for k,v in r.items() if k.endswith('median_s')}, 'agree', r['forms_agree'], 'expected', r['equals_expected'])" 2>&1 | tail -1)"

# ---- 2. one overlapped replay with checkpoints and progress PUTs, killed after step 105's progress PUT
RUN="$CK/run"
rm -rf "$RUN"
t0=$(date +%s)
pixi run python tools/lm_segment.py run --recipe "$IN/recipe_every2.json" --tokens /root/tokens_stream --from "$IN/ckpt_00000100.blm" \
    --steps "$STEPS" --devices "$DEV" --route A --segment 5 --label overlap-2gpu-nvidia \
    --expect-chain "$IN/A-1.chain.partial.jsonl" --upload-urls /root/urls/upload.json --out "$RUN" > "$OUT/run.log" 2>&1 &
RUNPID=$!
killed=no
while kill -0 "$RUNPID" 2> /dev/null; do
    if grep -q "progress: chain.progress.jsonl to step $KILL_AT in" "$RUN/log.txt" "$OUT/run.log" 2> /dev/null; then
        pids=$(pgrep -f "tools/lm_segment.py run --recipe $IN/recipe_every2.json")
        say "progress PUT for step $KILL_AT seen; SIGKILL to $pids"
        kill -9 $pids
        killed=yes
        break
    fi
    sleep 1
done
wait "$RUNPID" 2> /dev/null; rc=$?
grep -v 'mbind memory' "$OUT/run.log" > "$OUT/run.clean.log"
say "run exit=$rc killed=$killed secs=$(( $(date +%s) - t0 )): $(grep -E '^\[.*\] (step |checkpoint|uploaded|progress)|PASS|FAIL|DISAGREE|REFUSED|Error' "$OUT/run.clean.log" | cut -c1-150 | tr '\n' '|' | cut -c1-2400)"
mkdir -p "$OUT/run"
for f in chain.jsonl manifest.tsv log.txt segment.json; do [ -e "$RUN/$f" ] && cp "$RUN/$f" "$OUT/run/"; done
( cd "$RUN" && for f in *.blm; do [ -e "$f" ] && echo "$(stat -c %s "$f") $(sha256sum "$f")"; done ) > "$OUT/run_checkpoints.sha256" 2>&1
say "checkpoints on the box: $(tr '\n' ' ' < "$OUT/run_checkpoints.sha256")"
say "done"
exit 0
