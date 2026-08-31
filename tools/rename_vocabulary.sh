#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# THE 2026-08-31 VOCABULARY CHANGE, recorded as a script so the rename is
# auditable rather than a wall of untraceable edits.
#
#   mojo_only/       -> original/            no upstream file corresponds
#   ported/          -> derived/             Apache-2.0 section 4's own word
#   PORTED_MAP.tsv   -> DERIVATION_MAP.tsv
#   UNPORTED.tsv     -> NOT_IMPLEMENTED.tsv
#
# WHY NOT `reimplemented/`. It is the more flattering word and in many cases
# the more accurate one, but it cannot be asserted for a whole DIRECTORY
# before a file-by-file audit says so for every file in it. One genuine
# transcription inside a directory named `reimplemented` is a greppable
# overclaim. That judgement belongs per row in DERIVATION_MAP, beside its
# evidence. `derived` is true of every file in there today.
#
# WHY THE NESTING IS NOT FLATTENED. The boundary is what makes the 84.9%
# measurable at all, and the path underneath `derived/` is a path-for-path
# mirror of the upstream tree (spectral/derived/cuvs/preprocessing/spectral/
# detail/ against cuVS's cpp/src/preprocessing/spectral/detail/), which is
# what lets a reviewer diff file for file. Renaming fixes the words. Removing
# the boundary would delete the evidence.
#
# THE SUBSTITUTIONS ARE ORDERED AND TARGETED. A blind `ported -> derived`
# would mangle every sentence containing "not ported", "unported upstream"
# or "the port is transliterated", which are prose and must survive. Only
# PATH and IMPORT forms are touched, and the two filenames go first so they
# are not caught by the path rules.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY="${1:-}"
run() { if [ "$DRY" = "--dry-run" ]; then echo "  would: $*"; else "$@"; fi; }

echo "== 1. directories =="
for d in $(find . -maxdepth 2 -type d -name ported -not -path './.pixi/*' | sort); do
    run git mv "$d" "${d%/ported}/derived"
done
for d in $(find . -maxdepth 2 -type d -name mojo_only -not -path './.pixi/*' | sort); do
    run git mv "$d" "${d%/mojo_only}/original"
done

echo "== 2. the two ledger filenames =="
for f in $(find . -maxdepth 2 -name PORTED_MAP.tsv -not -path './.pixi/*' | sort); do
    run git mv "$f" "$(dirname "$f")/DERIVATION_MAP.tsv"
done
for f in $(find . -maxdepth 2 -name UNPORTED.tsv -not -path './.pixi/*' | sort); do
    run git mv "$f" "$(dirname "$f")/NOT_IMPLEMENTED.tsv"
done

echo "== 3. references, in this order =="
FILES=$(git ls-files -- '*.mojo' '*.py' '*.sh' '*.toml' '*.md' '*.tsv' '*.yml' '*.yaml' '*.cff' \
        | grep -v '^bench/results/' | grep -v '^upstream/')
if [ "$DRY" = "--dry-run" ]; then
    echo "  would rewrite references in $(echo "$FILES" | wc -l | tr -d ' ') files"
    exit 0
fi
printf '%s\n' "$FILES" | while IFS= read -r f; do
    [ -f "$f" ] || continue
    # 3a. the ledger names FIRST, so the path rules below cannot catch them
    perl -pi -e 's/\bPORTED_MAP\b/DERIVATION_MAP/g' "$f"
    perl -pi -e 's/\bUNPORTED\.tsv/NOT_IMPLEMENTED.tsv/g' "$f"
    # 3b. mojo_only is a whole token and only ever names this directory
    perl -pi -e 's/\bmojo_only\b/original/g' "$f"
    # 3c. `ported` ONLY in import and path position. `\b` before `ported`
    #     does not match inside `unported`, so prose survives.
    perl -pi -e 's/\.ported\./.derived./g' "$f"
    perl -pi -e 's{\bported/}{derived/}g' "$f"
done
echo "done"
