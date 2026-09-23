# The CPU witness of T3 route A segment 1 (tools/lm_cpu_witness.py), on one RunPod CPU pod.
# Presigned GET URLs are substituted for the @..@ placeholders on the Mac (6 h expiry) and are
# not committed. Inputs: the recipe, the stream's manifest, checkpoints 100 and 700, and byte
# ranges of the token parts, all from R2; the chain lines 100, 101, 700, 701 ship in the repo.
set -u
mkdir -p "$LEG_OUT"; G="$LEG_OUT/gate.txt"; say() { echo "[$(date -u +%T)] $*" | tee -a "$G"; }
T0=$(date +%s); HARD=$((T0 + 195 * 60))   # everything ends 195 min after the start; the lease is longer
left() { echo $(( HARD - $(date +%s) )); }
say "started=$(date -u +%FT%TZ) commit=${MOJOLEARN_COMMIT:-unknown}"
lscpu > "$LEG_OUT/lscpu.txt" 2>&1; nproc >> "$LEG_OUT/lscpu.txt"; free -g > "$LEG_OUT/free.txt" 2>&1
say "cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//') nproc=$(nproc) mem=$(free -g | awk '/Mem:/{print $2}')G"
W=/root/witness; mkdir -p $W/ckpt; cd /root/mojolearn
EX=bench/results/lm_t3_cpu_witness_2026-09-23/chain_excerpt.jsonl
cp "$EX" $W/chain.jsonl

# checkpoints in the background while the wheel installs
( t=$(date +%s); curl -fsS --retry 5 -o $W/ckpt/ckpt_00000100.blm '@URL_CKPT100@'; echo "ckpt100_exit=$? seconds=$(( $(date +%s) - t ))" >> "$G"
  t=$(date +%s); curl -fsS --retry 5 -o $W/ckpt/ckpt_00000700.blm '@URL_CKPT700@'; echo "ckpt700_exit=$? seconds=$(( $(date +%s) - t ))" >> "$G" ) &
DL=$!
curl -fsS --retry 5 -o $W/recipe.json '@URL_RECIPE@'
curl -fsS --retry 5 -o $W/manifest.json '@URL_MANIFEST@'
cat > $W/parts.json <<'JSON'
{"tokens.i32.part00": "@URL_P00@", "tokens.i32.part01": "@URL_P01@", "tokens.i32.part02": "@URL_P02@",
 "tokens.i32.part03": "@URL_P03@", "tokens.i32.part04": "@URL_P04@", "tokens.i32.part05": "@URL_P05@",
 "tokens.i32.part06": "@URL_P06@"}
JSON
sha256sum $W/recipe.json $W/manifest.json $W/chain.jsonl | tee "$LEG_OUT/inputs.sha256" >> "$G"

# the PUBLISHED wheel, hashed; no build
python3 -m venv /root/wv > "$LEG_OUT/venv.log" 2>&1 || { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv && python3 -m venv /root/wv; } >> "$LEG_OUT/venv.log" 2>&1
/root/wv/bin/pip download --disable-pip-version-check --quiet --no-deps --only-binary=:all: -d /root/wheel "mojolearn==0.8.15" > "$LEG_OUT/pip_download.log" 2>&1
sha256sum /root/wheel/*.whl | tee "$LEG_OUT/wheel.sha256" >> "$G"
/root/wv/bin/pip install --disable-pip-version-check --quiet numpy /root/wheel/*.whl > "$LEG_OUT/pip_install.log" 2>&1; say "pip_install_exit=$?"
/root/wv/bin/pip freeze > "$LEG_OUT/pip_freeze.txt" 2>&1
PY="env PYTHONPATH= /root/wv/bin/python"
$PY -c "import sys, mojolearn; print(sys.version.split()[0], mojolearn.__version__, mojolearn.__file__)" >> "$G" 2>&1
wait $DL
sha256sum $W/ckpt/*.blm | tee "$LEG_OUT/checkpoints.sha256" >> "$G"

WIT="$PY tools/lm_cpu_witness.py"
C="--recipe $W/recipe.json --manifest $W/manifest.json --tokens $W/parts.json --chain $W/chain.jsonl"
R="$LEG_OUT/records"; mkdir -p "$R"
TIMEV=/usr/bin/time; [ -x $TIMEV ] || { apt-get install -y time > /dev/null 2>&1 || true; }
tv() { if [ -x /usr/bin/time ]; then /usr/bin/time -v "$@"; else "$@"; fi; }

# the gradient replay: one host step per shard, one core, until its deadline (it writes its record after every shard)
( tv timeout $(( $(left) - 300 )) $WIT gradient $C --checkpoint $W/ckpt/ckpt_00000100.blm --step 101 --shards 0:64 \
      --deadline-seconds $(( $(left) - 900 )) --out "$R/gradient_step101.json" > "$LEG_OUT/gradient_step101.log" 2>&1
  echo "gradient_exit=$?" >> "$G" ) &
GR=$!

run() {  # NAME ARGS...: one check, bounded by the time left
    name=$1; shift
    [ "$(left)" -gt 120 ] || { say "$name SKIPPED: out of time"; return; }
    tv timeout $(( $(left) - 60 )) $WIT "$@" > "$LEG_OUT/$name.log" 2>&1; rc=$?
    say "$name exit=$rc $(grep -E 'PASS|FAIL|RECORDED' "$LEG_OUT/$name.log" | tail -1)"
}
run loss_step101_shard0_threaded  loss $C --checkpoint $W/ckpt/ckpt_00000100.blm --step 101 --shard 0 --threaded --out "$R/loss_step101_shard0_threaded.json"
run control_token_step101_shard0  loss $C --checkpoint $W/ckpt/ckpt_00000100.blm --step 101 --shard 0 --threaded --perturb-token --out "$R/control_token_step101_shard0.json"
run heldout0_ckpt100_threaded     heldout $C --checkpoint $W/ckpt/ckpt_00000100.blm --heldout-index 0 --threaded --out "$R/heldout0_ckpt100_threaded.json"
run loss_step701_shard0_threaded  loss $C --checkpoint $W/ckpt/ckpt_00000700.blm --step 701 --shard 0 --threaded --out "$R/loss_step701_shard0_threaded.json"
run heldout0_ckpt700_threaded     heldout $C --checkpoint $W/ckpt/ckpt_00000700.blm --heldout-index 0 --threaded --out "$R/heldout0_ckpt700_threaded.json"
run control_param_step101_shard0  loss $C --checkpoint $W/ckpt/ckpt_00000100.blm --step 101 --shard 0 --threaded --perturb-param-bit 38597376 --out "$R/control_param_step101_shard0.json"
run loss_step101_shard63_threaded loss $C --checkpoint $W/ckpt/ckpt_00000100.blm --step 101 --shard 63 --threaded --out "$R/loss_step101_shard63_threaded.json"
run loss_step101_shard0_reference loss $C --checkpoint $W/ckpt/ckpt_00000100.blm --step 101 --shard 0 --out "$R/loss_step101_shard0_reference.json"
wait $GR
say "finished=$(date -u +%FT%TZ) seconds=$(( $(date +%s) - T0 ))"
