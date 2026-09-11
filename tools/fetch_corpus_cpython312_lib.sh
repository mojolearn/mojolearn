#!/bin/sh
# tools/fetch_corpus_cpython312_lib.sh -- rebuild the pinned source-code
# corpus training/corpus/cpython312_lib/input.txt from the CPython 3.12.0
# release tarball and verify it against manifest.json (DEVIATIONS 2525 to
# 2527; ENGINEERING_RULES section 9's second neural kind). The bytes are not
# committed; this script and the manifest are the pinned artifact.
#
#   sh tools/fetch_corpus_cpython312_lib.sh            # writes input.txt
#   sh tools/fetch_corpus_cpython312_lib.sh --check    # verifies an existing input.txt only
#
# Exit 0 only when input.txt exists and its sha256 and length equal the
# manifest's. POSIX sh only (dash on the pods); needs curl, tar and one of
# sha256sum / shasum.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/training/corpus/cpython312_lib"
URL="https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz"
TARBALL_SHA="51412956d24a1ef7c97f1cb5f70e185c13e3de1f50d131c0aac6338080687afb"
CORPUS_SHA="f08d783cac53829be0da6def7ac74947f3e339915dfee7ca8d8db5ee344fc956"
CORPUS_BYTES=4522096

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

WORK=$(mktemp -d "${TMPDIR:-/tmp}/cpython312_lib.XXXXXX") || exit 1
trap 'rm -rf "$WORK"' EXIT
curl -sSL -o "$WORK/Python-3.12.0.tgz" "$URL" || { echo "download failed" >&2; exit 1; }
_tsha=$(sha256_of "$WORK/Python-3.12.0.tgz")
[ "$_tsha" = "$TARBALL_SHA" ] || { echo "tarball sha256 $_tsha, expected $TARBALL_SHA" >&2; exit 1; }
# Top-level Lib/*.py only, no subdirectories; GNU and BSD tar both accept
# the member pattern.
tar -xzf "$WORK/Python-3.12.0.tgz" -C "$WORK" 'Python-3.12.0/Lib/*.py' 2> /dev/null \
    || tar -xzf "$WORK/Python-3.12.0.tgz" -C "$WORK" --wildcards 'Python-3.12.0/Lib/*.py' \
    || { echo "extract failed" >&2; exit 1; }
mkdir -p "$DIR"
( cd "$WORK/Python-3.12.0/Lib" && ls *.py | LC_ALL=C sort | tr '\n' '\0' | xargs -0 cat ) > "$DIR/input.txt.part" \
    || { echo "concatenation failed" >&2; exit 1; }
mv "$DIR/input.txt.part" "$DIR/input.txt"
check
