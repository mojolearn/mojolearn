#!/bin/sh
# tools/fetch_corpus_enwik8.sh -- rebuild the pinned English corpus
# training/corpus/enwik8/input.txt (ENGINEERING_RULES section 9's English
# kind for every neural timing and quality claim) and verify it against
# manifest.json. The bytes are not committed; this script and the manifest
# are the pinned artifact.
#
# enwik8 is the first 10^8 bytes of the English Wikipedia XML dump of
# 2006-03-03, the Hutter Prize file and the standard byte-level language
# modeling benchmark (bits per character; 90M/5M/5M train/valid/test).
#
#   sh tools/fetch_corpus_enwik8.sh            # writes input.txt
#   sh tools/fetch_corpus_enwik8.sh --check    # verifies an existing input.txt only
#
# MOJOLEARN_CORPUS_SOURCE_DIR=<dir> reuses an enwik8.zip already there (its
# sha256 is still checked). Exit 0 only when input.txt exists and its sha256,
# md5 and length equal the pins. POSIX sh only (dash on the boxes); needs
# curl, python3 and one of sha256sum / shasum.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/training/corpus/enwik8"
URL="http://mattmahoney.net/dc/enwik8.zip"
ZIP_BYTES=36445475
ZIP_SHA="547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc"
# md5 as published by the benchmark's scripts; sha256 computed 2026-09-11.
CORPUS_MD5="a1fa5ffddb56f4953e226637dabbb36a"
CORPUS_SHA="2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
CORPUS_BYTES=100000000

sha256_of() {
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

check() {
    [ -f "$DIR/input.txt" ] || { echo "no $DIR/input.txt" >&2; return 1; }
    _len=$(wc -c < "$DIR/input.txt" | tr -d ' ')
    [ "$_len" = "$CORPUS_BYTES" ] || { echo "corpus length $_len, expected $CORPUS_BYTES" >&2; return 1; }
    _sha=$(sha256_of "$DIR/input.txt")
    _md5=$(python3 -c 'import hashlib, sys; print(hashlib.md5(open(sys.argv[1], "rb").read()).hexdigest())' "$DIR/input.txt")
    if [ "$_sha" != "$CORPUS_SHA" ] || [ "$_md5" != "$CORPUS_MD5" ]; then
        echo "corpus mismatch: sha256 $_sha md5 $_md5 (pinned $CORPUS_SHA $CORPUS_MD5)" >&2
        return 1
    fi
    echo "ok: $DIR/input.txt sha256 $_sha md5 $_md5 bytes $_len"
    return 0
}

if [ "${1:-}" = "--check" ]; then
    check
    exit $?
fi
if check 2> /dev/null; then
    exit 0
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/enwik8.XXXXXX") || exit 1
trap 'rm -rf "$WORK"' EXIT
ZIP="$WORK/enwik8.zip"
if [ -n "${MOJOLEARN_CORPUS_SOURCE_DIR:-}" ] && [ -f "$MOJOLEARN_CORPUS_SOURCE_DIR/enwik8.zip" ]; then
    ZIP="$MOJOLEARN_CORPUS_SOURCE_DIR/enwik8.zip"
else
    curl -sSL --retry 3 -o "$ZIP" "$URL" || { echo "download failed" >&2; exit 1; }
fi
_zsha=$(sha256_of "$ZIP")
[ "$_zsha" = "$ZIP_SHA" ] || { echo "zip sha256 $_zsha, expected $ZIP_SHA (bytes $(wc -c < "$ZIP" | tr -d ' '), expected $ZIP_BYTES)" >&2; exit 1; }
mkdir -p "$DIR"
python3 -c '
import shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z, z.open("enwik8") as src, open(sys.argv[2], "wb") as dst:
    shutil.copyfileobj(src, dst, 1 << 20)
' "$ZIP" "$DIR/input.txt.part" || { echo "extract failed" >&2; exit 1; }
mv "$DIR/input.txt.part" "$DIR/input.txt"
check
