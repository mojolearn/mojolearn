#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA body: train the GPT-3 Small run's BPE vocabulary on
# this box from the PUBLISHED pip wheel mojolearn==@VERSION@ and bring the two
# files and their sha256 home. The trainer is host code (tokenizer/train/
# bpe_train.mojo): it runs on this box's CPU and touches no GPU. The GPU
# inventory is recorded only to say which rented box this was.
#
# Placeholders substituted by make_body.sh (RunPod passes no environment):
#   @VERSION@  the wheel version
#   @SLUG@     this lease's output slug
#   @URL@      a short-lived presigned GET for the sample text in R2
#   @SHA@      the sample's pinned sha256
#   @BUDGET@   seconds the trainer may run
# set -u and NOT set -e: a failure is a result and its log has to come home.
set -u
VERSION="@VERSION@"
SLUG="@SLUG@"
URL="@URL@"
SHA="@SHA@"
BUDGET="@BUDGET@"

OUT="/root/gemm_leg_out/$SLUG"
mkdir -p "$OUT"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
say "slug=$SLUG"
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
uname -a > "$OUT/uname.txt" 2>&1
lscpu > "$OUT/lscpu.txt" 2>&1
nproc >> "$OUT/lscpu.txt" 2>&1
if command -v nvidia-smi > /dev/null 2>&1; then nvidia-smi -L > "$OUT/gpu_inventory.txt" 2>&1; fi
if command -v rocm-smi > /dev/null 2>&1; then rocm-smi --showproductname > "$OUT/gpu_inventory.txt" 2>&1; fi

PY=""
for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" > /dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY=$(command -v "$c"); break
    fi
done
if [ -z "$PY" ]; then
    ( apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip ) \
        > "$OUT/apt_python.log" 2>&1
    PY=$(command -v python3 || true)
fi
say "system_python=$PY ($("$PY" --version 2>&1))"
VENV=/root/vocab-venv
rm -rf "$VENV"
"$PY" -m venv "$VENV" > "$OUT/venv.log" 2>&1
if [ ! -x "$VENV/bin/pip" ]; then
    ( apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv ) >> "$OUT/venv.log" 2>&1
    rm -rf "$VENV"; "$PY" -m venv "$VENV" >> "$OUT/venv.log" 2>&1
fi
[ -x "$VENV/bin/pip" ] || { say "VENV FAILED; see venv.log. Nothing was run."; exit 12; }
"$VENV/bin/pip" install --disable-pip-version-check --quiet numpy "mojolearn==$VERSION" > "$OUT/pip_install.log" 2>&1
say "pip_install_exit=$?"
"$VENV/bin/pip" freeze > "$OUT/pip_freeze.txt" 2>&1

mkdir -p /root/vocabrun && cd /root/vocabrun
curl -fsS --retry 5 -o sample.txt "$URL" > "$OUT/curl.log" 2>&1
say "curl_exit=$?"
GOT=$(sha256sum sample.txt | cut -c1-64)
say "sample_sha256=$GOT"
[ "$GOT" = "$SHA" ] || { say "SAMPLE SHA256 DOES NOT MATCH THE PIN $SHA. Nothing was trained."; exit 3; }

cat > door.py <<'PYEOF'
import hashlib, os, sys, time
import mojolearn as ml
assert "vocab-venv" in ml.__file__, ml.__file__
src, out = sys.argv[1], sys.argv[2]
docs = [open(src, "rb").read()]
t0 = time.time()
v = ml.tokenizer.BpeVocabularyTrainer(vocab_size=50256, min_frequency=2, backend="mojo").train(docs)
wall = time.time() - t0
v.write_ranks(out + ".ranks.tsv"); v.write_tokenizer_json(out + ".tokenizer.json")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(f"mojolearn={ml.__version__} input_sha256={hashlib.sha256(docs[0]).hexdigest()} input_bytes={len(docs[0])} "
      f"backend={v.stats['backend']} tokens={v.n_tokens} merges={len(v.merges)} ties={v.n_ties_broken} "
      f"groups={v.stats['n_groups']} train_wall_s={wall:.1f} ranks_sha256={sha(out + '.ranks.tsv')} "
      f"tokenizer_json_sha256={sha(out + '.tokenizer.json')}", flush=True)
PYEOF
timeout -k 15 "$BUDGET" "$VENV/bin/python" door.py sample.txt vocab > "$OUT/train.out" 2> "$OUT/train.log"
say "train_exit=$?"
cat "$OUT/train.out" >> "$G"
cp vocab.ranks.tsv vocab.tokenizer.json "$OUT/" 2>/dev/null
sha256sum vocab.ranks.tsv vocab.tokenizer.json > "$OUT/vocab.sha256" 2>&1
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$OUT/BODY_DONE"
