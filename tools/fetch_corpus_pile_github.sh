#!/bin/sh
# tools/fetch_corpus_pile_github.sh -- rebuild the pinned source-code corpus
# training/corpus/pile_github/input.txt (ENGINEERING_RULES section 9's code
# kind for every neural timing and quality claim) and verify it against
# manifest.json. The bytes are not committed; this script and the manifest
# are the pinned artifact.
#
# The Pile's GitHub component is ordinary public GitHub code across many
# languages, with published bits per byte for GPT-2, GPT-3 and byte-level
# transformers (Gao et al. 2020, arXiv 2101.00027; SpaceByte, arXiv
# 2404.14408). The source is the validation file of monology/pile-uncopyrighted
# at a pinned Hugging Face revision (no login, no terms gate).
#
# SELECTION: every record of val.jsonl.zst whose meta.pile_set_name is
# "Github", in file order, its "text" encoded UTF-8 and followed by one "\n"
# byte, the concatenation truncated to CAP bytes.
#
#   sh tools/fetch_corpus_pile_github.sh            # writes input.txt
#   sh tools/fetch_corpus_pile_github.sh --check    # verifies an existing input.txt only
#
# MOJOLEARN_CORPUS_SOURCE_DIR=<dir> reuses a val.jsonl.zst already there (its
# sha256 is still checked). Needs curl, python3, one of sha256sum / shasum,
# and a zstd decoder: the zstd tool, else `apt-get install zstd` when root
# on Debian or Ubuntu, else python's zstandard module. POSIX sh only.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/training/corpus/pile_github"
REVISION="3be90335b66f24456a5d6659d9c8d208c0357119"
URL="https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/$REVISION/val.jsonl.zst"
SOURCE_BYTES=338045152
SOURCE_SHA="db5e5d1532bf8dc33a6589b50ecba1a8c96f7b4b9cb343d168e603c393007c26"
CAP=100000000
# Computed on the Mac 2026-09-11: all 18,337 GitHub records of the file,
# 97,124,565 bytes, under the cap.
CORPUS_SHA="52a5b4c36ab9119c15505331c10e3b23690377d40fbac3e598c7fafe13a324df"
CORPUS_BYTES=97124565

sha256_of() {
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

check() {
    [ -f "$DIR/input.txt" ] || { echo "no $DIR/input.txt" >&2; return 1; }
    _sha=$(sha256_of "$DIR/input.txt")
    _len=$(wc -c < "$DIR/input.txt" | tr -d ' ')
    if [ "$_sha" != "$CORPUS_SHA" ] || [ "$_len" != "$CORPUS_BYTES" ]; then
        echo "corpus mismatch: sha256 $_sha bytes $_len (manifest $CORPUS_SHA $CORPUS_BYTES)" >&2
        return 1
    fi
    echo "ok: $DIR/input.txt sha256 $_sha bytes $_len"
    return 0
}

if [ "${1:-}" = "--check" ]; then
    check
    exit $?
fi
if check 2> /dev/null; then
    exit 0
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/pile_github.XXXXXX") || exit 1
trap 'rm -rf "$WORK"' EXIT
SRC="$WORK/val.jsonl.zst"
if [ -n "${MOJOLEARN_CORPUS_SOURCE_DIR:-}" ] && [ -f "$MOJOLEARN_CORPUS_SOURCE_DIR/val.jsonl.zst" ]; then
    SRC="$MOJOLEARN_CORPUS_SOURCE_DIR/val.jsonl.zst"
else
    curl -sSL --retry 3 -o "$SRC" "$URL" || { echo "download failed" >&2; exit 1; }
fi
_ssha=$(sha256_of "$SRC")
[ "$_ssha" = "$SOURCE_SHA" ] || { echo "source sha256 $_ssha, expected $SOURCE_SHA (bytes $(wc -c < "$SRC" | tr -d ' '), expected $SOURCE_BYTES)" >&2; exit 1; }

if ! command -v zstd > /dev/null 2>&1 && [ "$(id -u)" = 0 ] && command -v apt-get > /dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd > "$WORK/apt.log" 2>&1 \
        || { DEBIAN_FRONTEND=noninteractive apt-get update -qq > /dev/null 2>&1 \
             && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd >> "$WORK/apt.log" 2>&1; }
fi
mkdir -p "$DIR"
SELECT='
import json, sys
cap = int(sys.argv[1])
out = open(sys.argv[2], "wb")
left = cap
records = 0
for line in sys.stdin.buffer:
    rec = json.loads(line)
    if rec.get("meta", {}).get("pile_set_name") != "Github":
        continue
    piece = rec["text"].encode("utf-8") + b"\n"
    out.write(piece[:left])
    left -= min(left, len(piece))
    records += 1
    if left == 0:
        break
out.close()
print("github records %d bytes %d" % (records, cap - left), file=sys.stderr)
'
if command -v zstd > /dev/null 2>&1; then
    zstd -dc "$SRC" | python3 -c "$SELECT" "$CAP" "$DIR/input.txt.part"
    _rc=$?
else
    python3 -c 'import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(open(sys.argv[1], "rb"), sys.stdout.buffer)' "$SRC" \
        | python3 -c "$SELECT" "$CAP" "$DIR/input.txt.part"
    _rc=$?
fi
# A decoder that stops early after the cap closes the pipe; only the
# selection's own status and the pinned hash decide.
[ "$_rc" -eq 0 ] || { echo "selection failed (exit $_rc; no zstd tool and no zstandard module?)" >&2; exit 1; }
mv "$DIR/input.txt.part" "$DIR/input.txt"
check
