#!/bin/sh
# tools/tokdet_pod_cmd.sh -- the body of the tokenizer-determinism leg, run on
# one RunPod CPU pod by tools/runpod_cpu_leg.sh --cmd-file.
#
# THIS IS A TEMPLATE. Two placeholders are substituted on the Mac just before
# renting, and the rendered copy is written to the scratchpad, never committed:
#
#   @CORPUS_URL@   a presigned GET for corpus/enwik8/input.txt, minted by
#                  tools/dataset_store.sh presign. The leg uploads this file
#                  over ssh stdin, so the URL lands in a file on the box and
#                  never appears in the box's process list. No credential
#                  leaves the Mac.
#   @CORPUS_SHA@   the pinned sha256 from bench/results/dataset_store/manifest.tsv.
#
# WHY THE POD EXISTS AT ALL. The thread-count axis is the whole reason. Andrew
# allows one core per agent on the shared Mac, so a trainer spinning up every
# core cannot run there. The other axes are run here too, so that every axis
# gets one coherent x86/Linux column instead of being split across machines;
# the Mac contributes an independent arm64 column at one core.
#
# WHAT COMES BACK. Only summary.json and the per-comparison JSON. The trained
# vocabularies themselves are deleted on the box: about a hundred runs of
# half-megabyte artifacts is an oversized blob, and every artifact's sha256 is
# already recorded inside summary.json, which is the thing the verdict rests
# on.
set -eu

R=/root/mojolearn
W=/root/tokdet_work
OUT="$LEG_OUT"
mkdir -p "$OUT" "$W"

echo "=== box ==="
nproc > "$OUT/nproc.txt"
(lscpu 2>/dev/null || true) > "$OUT/lscpu.txt"
cat "$OUT/nproc.txt"

echo "=== corpus from R2 ==="
curl -fsSL --retry 3 --retry-delay 2 -o "$W/enwik8.txt" '@CORPUS_URL@'
GOT=$(sha256sum "$W/enwik8.txt" | cut -d' ' -f1)
if [ "$GOT" != "@CORPUS_SHA@" ]; then
    echo "corpus sha256 MISMATCH: got $GOT want @CORPUS_SHA@" >&2
    exit 1
fi
echo "corpus ok $GOT" | tee "$OUT/corpus_sha256.txt"

echo "=== throwaway env (never a mojolearn dependency) ==="
python3 -m venv "$W/venv"
"$W/venv/bin/pip" install -q --disable-pip-version-check \
    'tokenizers==0.23.2' 'sentencepiece==0.2.2' 'protobuf'
PY="$W/venv/bin/python"
"$PY" - > "$OUT/versions.txt" <<'PYEOF'
import platform, sys
import tokenizers, sentencepiece
print("python", sys.version.split()[0])
print("platform", platform.platform(), platform.machine())
print("tokenizers", tokenizers.__version__)
print("sentencepiece", sentencepiece.__version__)
PYEOF
cat "$OUT/versions.txt"

echo "=== corpus shards ==="
"$PY" "$R/tools/tokdet_corpus.py" --source "$W/enwik8.txt" --out "$W/corpus16" \
    --bytes 16777216 --shards 4 > "$OUT/corpus_manifest.json"

echo "=== BPE matrix (all axes, threads 1..16) ==="
"$PY" "$R/tools/tokdet_matrix.py" --corpus-dir "$W/corpus16" --out "$W/bpe" \
    --python "$PY" --threads 1,2,4,8,16 --reps 5 --axis-reps 2 \
    --base-vocab 8000 --vocabs 1000,32000 --orders rev,rot \
    --kinds hf,sp --models bpe 2>&1 | tee "$OUT/bpe_console.txt"

echo "=== Unigram matrix (reduced: it costs ~5x BPE per run) ==="
"$PY" "$R/tools/tokdet_matrix.py" --corpus-dir "$W/corpus16" --out "$W/unigram" \
    --python "$PY" --threads 1,4,16 --reps 3 --axis-reps 2 \
    --base-vocab 8000 --vocabs 32000 --orders rev \
    --kinds hf,sp --models unigram 2>&1 | tee "$OUT/unigram_console.txt"

echo "=== collect summaries, leave the vocabularies on the box ==="
for m in bpe unigram; do
    mkdir -p "$OUT/$m"
    cp "$W/$m/summary.json" "$OUT/$m/summary.json" 2>/dev/null || true
    cp -r "$W/$m/compare" "$OUT/$m/compare" 2>/dev/null || true
done
du -sh "$OUT" | tee "$OUT/leg_out_size.txt"
echo "=== done ==="
