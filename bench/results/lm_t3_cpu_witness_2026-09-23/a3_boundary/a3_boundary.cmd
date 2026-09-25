# The A/2 to A/3 boundary CPU witness (tools/lm_cpu_witness.py), one RunPod CPU pod.
# Checks: (a) all 64 shard losses of step 2001 from A/2's final checkpoint (ckpt_00002000.blm)
# against A/3's live chain line 2001; (b) the held-out loss (shard 013, batch 0) from A/3's
# final checkpoint (ckpt_00002400.blm), recorded with its bits; (c) one negative control
# (perturbed token) on the step 2001 shard 0 check, which must FAIL; (d) an extra cheap check,
# the A/1 step-101 shard-0 loss (route A segment 1, ckpt_00000100.blm) against its own chain
# line 101, on the PUBLISHED wheel mojolearn==0.8.18 (the previous run of this same check was
# on 0.8.15 and read 40de1f6a). Presigned GET URLs replace the @..@ placeholders on the Mac
# (they are not committed) and are not read from R2 credentials on this box.
set -u
mkdir -p "$LEG_OUT"; G="$LEG_OUT/gate.txt"; say() { echo "[$(date -u +%T)] $*" | tee -a "$G"; }
T0=$(date +%s)
say "started=$(date -u +%FT%TZ) commit=${MOJOLEARN_COMMIT:-unknown}"
lscpu > "$LEG_OUT/lscpu.txt" 2>&1; nproc >> "$LEG_OUT/lscpu.txt"
say "cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//') nproc=$(nproc)"
W=/root/witness; mkdir -p $W/ckpt; cd /root/mojolearn
cp bench/results/lm_t3_cpu_witness_2026-09-23/a3_boundary/chain_excerpt_a3.jsonl $W/chain_a3.jsonl
cp bench/results/lm_t3_cpu_witness_2026-09-23/chain_excerpt.jsonl $W/chain_a1.jsonl

( curl -fsS --retry 5 -o $W/ckpt/ckpt_00000100.blm '@URL_CKPT100@'; echo "ckpt100_exit=$?" >> "$G"
  curl -fsS --retry 5 -o $W/ckpt/ckpt_00002000.blm '@URL_CKPT2000@'; echo "ckpt2000_exit=$?" >> "$G"
  curl -fsS --retry 5 -o $W/ckpt/ckpt_00002400.blm '@URL_CKPT2400@'; echo "ckpt2400_exit=$?" >> "$G" ) &
DL=$!
curl -fsS --retry 5 -o $W/recipe.json '@URL_RECIPE@'
curl -fsS --retry 5 -o $W/manifest.json '@URL_MANIFEST@'
cat > $W/parts.json <<'JSON'
{"tokens.i32.part00": "@URL_P00@", "tokens.i32.part01": "@URL_P01@", "tokens.i32.part02": "@URL_P02@",
 "tokens.i32.part03": "@URL_P03@", "tokens.i32.part04": "@URL_P04@", "tokens.i32.part05": "@URL_P05@",
 "tokens.i32.part06": "@URL_P06@"}
JSON
sha256sum $W/recipe.json $W/manifest.json $W/chain_a3.jsonl $W/chain_a1.jsonl | tee "$LEG_OUT/inputs.sha256" >> "$G"

python3 -m venv /root/wv > "$LEG_OUT/venv.log" 2>&1 || { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv && python3 -m venv /root/wv; } >> "$LEG_OUT/venv.log" 2>&1
/root/wv/bin/pip download --disable-pip-version-check --quiet --no-deps --only-binary=:all: -d /root/wheel "mojolearn==0.8.18" > "$LEG_OUT/pip_download.log" 2>&1
sha256sum /root/wheel/*.whl | tee "$LEG_OUT/wheel.sha256" >> "$G"
/root/wv/bin/pip install --disable-pip-version-check --quiet numpy /root/wheel/*.whl > "$LEG_OUT/pip_install.log" 2>&1; say "pip_install_exit=$?"
/root/wv/bin/pip freeze > "$LEG_OUT/pip_freeze.txt" 2>&1
PY="env PYTHONPATH= /root/wv/bin/python"
$PY -c "import sys, mojolearn; print(sys.version.split()[0], mojolearn.__version__, mojolearn.__file__)" >> "$G" 2>&1
wait $DL
sha256sum $W/ckpt/*.blm | tee "$LEG_OUT/checkpoints.sha256" >> "$G"

R="$LEG_OUT/records"; L="$LEG_OUT/logs"; mkdir -p "$R" "$L"
C3="--recipe $W/recipe.json --manifest $W/manifest.json --tokens $W/parts.json --chain $W/chain_a3.jsonl"
C1="--recipe $W/recipe.json --manifest $W/manifest.json --tokens $W/parts.json --chain $W/chain_a1.jsonl"

# (a) all 64 shard losses of step 2001 from ckpt_00002000.blm, four at a time
one() {  # SHARD
    timeout 900 env PYTHONPATH= /root/wv/bin/python tools/lm_cpu_witness.py loss $C3 \
        --checkpoint $W/ckpt/ckpt_00002000.blm --step 2001 --shard "$1" --threaded \
        --out "$R/loss_step2001_shard$(printf %02d "$1").json" \
        > "$L/loss_step2001_shard$(printf %02d "$1").log" 2>&1
}
export -f one; export W R L C3
seq 0 63 | xargs -P 4 -I{} bash -c 'one "$0"' {}
say "step2001: PASS=$(grep -l '\"verdict\": \"PASS\"' $R/loss_step2001_shard*.json 2>/dev/null | wc -l) FAIL=$(grep -l '\"verdict\": \"FAIL\"' $R/loss_step2001_shard*.json 2>/dev/null | wc -l) records=$(ls $R/loss_step2001_shard*.json 2>/dev/null | wc -l)"

run3() {  # NAME ARGS...
    name=$1; shift
    timeout 900 env PYTHONPATH= /root/wv/bin/python tools/lm_cpu_witness.py "$@" > "$L/$name.log" 2>&1; rc=$?
    say "$name exit=$rc $(grep -E 'PASS|FAIL|RECORDED' "$L/$name.log" | tail -1)"
}
# (c) negative control: one token id + 1 on step 2001 shard 0
run3 control_token_step2001_shard0 loss $C3 --checkpoint $W/ckpt/ckpt_00002000.blm --step 2001 --shard 0 --threaded --perturb-token --out "$R/control_token_step2001_shard0.json"
# (b) held-out loss from ckpt_00002400.blm, recorded (no reference to hold it to yet)
run3 heldout0_ckpt2400_threaded heldout $C3 --checkpoint $W/ckpt/ckpt_00002400.blm --heldout-index 0 --threaded --out "$R/heldout0_ckpt2400_threaded.json"
# (d) extra cheap check: A/1 step 101 shard 0 on 0.8.18 (was 40de1f6a on 0.8.15)
run3 loss_step101_shard0_threaded_0818 loss $C1 --checkpoint $W/ckpt/ckpt_00000100.blm --step 101 --shard 0 --threaded --out "$R/loss_step101_shard0_threaded_0818.json"

say "finished=$(date -u +%FT%TZ) seconds=$(( $(date +%s) - T0 ))"
