# Third leg: every shard's loss of steps 101 and 701 (all 64 of each) on the host's threaded
# forward, four checks at a time, each held to its chain line's losses_f32_hex[s]. The host
# training step is too slow for a whole-step gradient (gradient.cmd: one shard did not finish in
# 147 minutes), so this is the largest check of the recorded step that fits a lease. Presigned
# GET URLs replace the @..@ placeholders on the Mac and are not committed.
set -u
mkdir -p "$LEG_OUT"; G="$LEG_OUT/gate.txt"; say() { echo "[$(date -u +%T)] $*" | tee -a "$G"; }
T0=$(date +%s)
say "started=$(date -u +%FT%TZ) commit=${MOJOLEARN_COMMIT:-unknown}"
lscpu > "$LEG_OUT/lscpu.txt" 2>&1; nproc >> "$LEG_OUT/lscpu.txt"
say "cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//') nproc=$(nproc)"
W=/root/witness; mkdir -p $W/ckpt; cd /root/mojolearn
cp bench/results/lm_t3_cpu_witness_2026-09-23/chain_excerpt.jsonl $W/chain.jsonl
( curl -fsS --retry 5 -o $W/ckpt/ckpt_00000100.blm '@URL_CKPT100@'; echo "ckpt100_exit=$?" >> "$G"
  curl -fsS --retry 5 -o $W/ckpt/ckpt_00000700.blm '@URL_CKPT700@'; echo "ckpt700_exit=$?" >> "$G" ) &
DL=$!
curl -fsS --retry 5 -o $W/recipe.json '@URL_RECIPE@'
curl -fsS --retry 5 -o $W/manifest.json '@URL_MANIFEST@'
cat > $W/parts.json <<'JSON'
{"tokens.i32.part00": "@URL_P00@", "tokens.i32.part01": "@URL_P01@", "tokens.i32.part02": "@URL_P02@",
 "tokens.i32.part03": "@URL_P03@", "tokens.i32.part04": "@URL_P04@", "tokens.i32.part05": "@URL_P05@",
 "tokens.i32.part06": "@URL_P06@"}
JSON
sha256sum $W/recipe.json $W/manifest.json $W/chain.jsonl | tee "$LEG_OUT/inputs.sha256" >> "$G"
python3 -m venv /root/wv > "$LEG_OUT/venv.log" 2>&1
/root/wv/bin/pip download --disable-pip-version-check --quiet --no-deps --only-binary=:all: -d /root/wheel "mojolearn==0.8.15" > "$LEG_OUT/pip_download.log" 2>&1
sha256sum /root/wheel/*.whl | tee "$LEG_OUT/wheel.sha256" >> "$G"
/root/wv/bin/pip install --disable-pip-version-check --quiet numpy /root/wheel/*.whl > "$LEG_OUT/pip_install.log" 2>&1; say "pip_install_exit=$?"
wait $DL
sha256sum $W/ckpt/*.blm | tee "$LEG_OUT/checkpoints.sha256" >> "$G"
R="$LEG_OUT/records"; L="$LEG_OUT/logs"; mkdir -p "$R" "$L"
one() {  # STEP CKPT SHARD
    timeout 900 env PYTHONPATH= /root/wv/bin/python tools/lm_cpu_witness.py loss --recipe $W/recipe.json \
        --manifest $W/manifest.json --tokens $W/parts.json --chain $W/chain.jsonl --checkpoint "$2" \
        --step "$1" --shard "$3" --threaded --out "$R/loss_step$1_shard$(printf %02d $3).json" \
        > "$L/loss_step$1_shard$(printf %02d $3).log" 2>&1
}
export -f one; export W R L
for s in $(seq 0 63); do echo "101 $W/ckpt/ckpt_00000100.blm $s"; done > $W/jobs.txt
for s in $(seq 0 63); do echo "701 $W/ckpt/ckpt_00000700.blm $s"; done >> $W/jobs.txt
xargs -P 4 -L 1 bash -c 'one "$0" "$1" "$2"' < $W/jobs.txt
say "PASS=$(grep -l '"verdict": "PASS"' $R/*.json | wc -l) FAIL=$(grep -l '"verdict": "FAIL"' $R/*.json | wc -l) records=$(ls $R | wc -l)"
say "finished=$(date -u +%FT%TZ) seconds=$(( $(date +%s) - T0 ))"
