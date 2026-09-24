# The gradient witness alone (tools/lm_cpu_witness.py gradient), second leg: the first leg's
# gradient process was killed by the pod's memory limit (exit 137) during shard 0 while a
# reference-path loss ran beside it, so this leg runs it alone on a memory-optimized pod and
# samples its resident memory every 30 s. Presigned GET URLs replace the @..@ placeholders on the
# Mac (6 h expiry) and are not committed.
set -u
mkdir -p "$LEG_OUT"; G="$LEG_OUT/gate.txt"; say() { echo "[$(date -u +%T)] $*" | tee -a "$G"; }
T0=$(date +%s); HARD=$((T0 + 150 * 60))   # the leg's lease maximum is 180 min
left() { echo $(( HARD - $(date +%s) )); }
say "started=$(date -u +%FT%TZ) commit=${MOJOLEARN_COMMIT:-unknown}"
lscpu > "$LEG_OUT/lscpu.txt" 2>&1; nproc >> "$LEG_OUT/lscpu.txt"; free -g > "$LEG_OUT/free.txt" 2>&1
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes > "$LEG_OUT/cgroup_memory.txt" 2>&1
say "cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//') nproc=$(nproc) cgroup_memory=$(head -1 "$LEG_OUT/cgroup_memory.txt")"
W=/root/witness; mkdir -p $W/ckpt; cd /root/mojolearn
cp bench/results/lm_t3_cpu_witness_2026-09-23/chain_excerpt.jsonl $W/chain.jsonl
( t=$(date +%s); curl -fsS --retry 5 -o $W/ckpt/ckpt_00000100.blm '@URL_CKPT100@'; echo "ckpt100_exit=$? seconds=$(( $(date +%s) - t ))" >> "$G" ) &
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
/root/wv/bin/pip freeze > "$LEG_OUT/pip_freeze.txt" 2>&1
wait $DL
sha256sum $W/ckpt/*.blm | tee "$LEG_OUT/checkpoints.sha256" >> "$G"
PY="env PYTHONPATH= /root/wv/bin/python"
mkdir -p "$LEG_OUT/records"
timeout $(( $(left) - 120 )) $PY tools/lm_cpu_witness.py gradient --recipe $W/recipe.json --manifest $W/manifest.json \
    --tokens $W/parts.json --chain $W/chain.jsonl --checkpoint $W/ckpt/ckpt_00000100.blm --step 101 --shards 0:64 \
    --deadline-seconds $(( $(left) - 600 )) --out "$LEG_OUT/records/gradient_step101.json" > "$LEG_OUT/gradient_step101.log" 2>&1 &
GP=$!
while kill -0 $GP 2>/dev/null; do
    echo "$(date -u +%T) $(ps -o rss= -p "$(pgrep -f "^/root/wv/bin/python tools/lm_cpu_witness.py gradient" | head -1)" 2>/dev/null | awk '{printf "%.2f", $1/1048576}') GiB $(free -g | awk '/Mem:/{print "used="$3"G"}')" >> "$LEG_OUT/rss.tsv"
    sleep 30
done
wait $GP; say "gradient_exit=$? $(tail -1 "$LEG_OUT/gradient_step101.log")"
say "peak_rss_gib=$(sort -k2 -n "$LEG_OUT/rss.tsv" | tail -1 | awk '{print $2}') finished=$(date -u +%FT%TZ) seconds=$(( $(date +%s) - T0 ))"
