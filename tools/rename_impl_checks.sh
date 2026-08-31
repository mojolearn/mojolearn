#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# THE SECOND VOCABULARY PASS, 2026-08-31. `derived/` -> `impl/`,
# `original/` -> `checks/`.
#
# WHY, AND IT IS A MEASUREMENT RATHER THAN A PREFERENCE. The split was never
# original-versus-borrowed. Measured:
#
#     derived/    287 files, 65,464 lines, and 20 of 23 estimator.mojo files
#                 import it -- this is the SHIPPED ALGORITHM, the thing that
#                 runs when a user calls .fit()
#     original/   236 files, of which 97 are _check, 25 _oracle and 16
#                 _fixture -- the VERIFICATION APPARATUS
#
# So the axis is IMPLEMENTATION versus VERIFICATION, and it had been labelled
# with provenance words that made a claim the layout was never fit to make.
# `impl/` says what the directory is and nothing about whose it is; `checks/`
# says what the other one is. Provenance is carried per file and per
# DERIVATION_MAP row, where it belongs and where it is per-file rather than
# in bulk.
#
# NOT FLATTENED TO THE LANE ROOT. `spectral/` has two different
# `spectral_embedding.mojo` at different depths; flattening collides. Only the
# root word changes.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY="${1:-}"
run() { if [ "$DRY" = "--dry-run" ]; then echo "  would: $*"; else "$@"; fi; }
echo "== directories =="
for d in $(find . -maxdepth 2 -type d -name derived -not -path './.pixi/*' | sort); do
    run git mv "$d" "${d%/derived}/impl"
done
for d in $(find . -maxdepth 2 -type d -name original -not -path './.pixi/*' | sort); do
    run git mv "$d" "${d%/original}/checks"
done
[ "$DRY" = "--dry-run" ] && exit 0
echo "== references =="
git ls-files -- '*.mojo' '*.py' '*.sh' '*.toml' '*.md' '*.tsv' '*.yml' '*.cff' \
  | grep -v '^bench/results/' | while IFS= read -r f; do
    [ -f "$f" ] || continue
    # Import and path position only. `\b` before `derived` does not match
    # inside a word, and the prose senses of "derived" and "original" have to
    # survive: this file's own comment above uses both.
    perl -pi -e 's/\.derived\./.impl./g; s{\bderived/}{impl/}g' "$f"
    perl -pi -e 's/\.original\./.checks./g; s{\boriginal/}{checks/}g' "$f"
done
echo done
