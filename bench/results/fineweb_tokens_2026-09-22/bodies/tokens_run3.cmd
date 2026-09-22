# FineWeb-Edu shards 000..003 (train) + 013 (held out) -> one token stream through the pinned vocabulary, run1
set -u
mkdir -p "$LEG_OUT"; G="$LEG_OUT/gate.txt"; say() { echo "$@" >> "$G"; }
say "run=run3"; say "started=$(date -u +%FT%TZ)"; say "commit=$MOJOLEARN_COMMIT"
lscpu > "$LEG_OUT/lscpu.txt" 2>&1; nproc >> "$LEG_OUT/lscpu.txt"; say "cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//')"
V=$(find /root -name vocab.ranks.tsv -o -name ranks.tsv 2>/dev/null | grep -v mojolearn/python | head -1)
D=$(dirname "$(find /root -name 000_00000.parquet 2>/dev/null | head -1)")
say "vocab=$V"; say "shards_dir=$D"
[ -n "$V" ] && [ -n "$D" ] || { say "STAGED FILES NOT FOUND"; exit 3; }
for s in 000 001 002 003 013; do [ -f "$D/${s}_00000.parquet" ] || { say "MISSING SHARD $s"; exit 3; }; done
python3 -m venv /root/tokvenv > "$LEG_OUT/venv.log" 2>&1 || { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv; python3 -m venv /root/tokvenv; } >> "$LEG_OUT/venv.log" 2>&1
/root/tokvenv/bin/pip install --disable-pip-version-check --quiet numpy pyarrow "mojolearn==0.8.14" > "$LEG_OUT/pip_install.log" 2>&1; say "pip_install_exit=$?"
/root/tokvenv/bin/pip freeze > "$LEG_OUT/pip_freeze.txt" 2>&1
sha256sum "$V" "$D"/000_00000.parquet "$D"/001_00000.parquet "$D"/002_00000.parquet "$D"/003_00000.parquet "$D"/013_00000.parquet > "$LEG_OUT/inputs.sha256"
mkdir -p /root/tokens
t0=$(date +%s)
PYTHONPATH= /root/tokvenv/bin/python tools/fineweb_tokens.py --vocab "$V" --out /root/tokens --held-out-shard \
    "$D"/000_00000.parquet "$D"/001_00000.parquet "$D"/002_00000.parquet "$D"/003_00000.parquet "$D"/013_00000.parquet > "$LEG_OUT/tokens.log" 2>&1
say "tokens_exit=$? seconds=$(( $(date +%s) - t0 ))"
tail -1 "$LEG_OUT/tokens.log" >> "$G"
cp /root/tokens/manifest.json "$LEG_OUT/manifest.json"
sha256sum /root/tokens/tokens.i32 /root/tokens/manifest.json > "$LEG_OUT/tokens.sha256"; cat "$LEG_OUT/tokens.sha256" >> "$G"
ls -l /root/tokens >> "$G"
t1=$(date +%s)
cd /root/tokens && split -b 2000000000 -d -a 2 tokens.i32 tokens.i32.part && ls -l >> "$G"
sha256sum tokens.i32.part* > "$LEG_OUT/parts.sha256"; cat "$LEG_OUT/parts.sha256" >> "$G"
cat tokens.i32.part* | sha256sum | cut -c1-64 > "$LEG_OUT/parts_joined.sha256"; say "parts_joined_sha256=$(cat "$LEG_OUT/parts_joined.sha256")"
up() { curl -fsS --retry 3 -T "$1" "$2" > "$LEG_OUT/upload_$1.log" 2>&1; say "upload $1 exit=$? secs=$(( $(date +%s) - t1 ))"; }
up tokens.i32.part00 '<presigned PUT URL>'
up tokens.i32.part01 '<presigned PUT URL>'
up tokens.i32.part02 '<presigned PUT URL>'
up tokens.i32.part03 '<presigned PUT URL>'
up tokens.i32.part04 '<presigned PUT URL>'
up tokens.i32.part05 '<presigned PUT URL>'
up tokens.i32.part06 '<presigned PUT URL>'
up manifest.json '<presigned PUT URL>'
say "finished=$(date -u +%FT%TZ)"
