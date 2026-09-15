#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# Write tokenizer/impl/unicode_table_generated.mojo before anything compiles
# tokenizer/ (bindings/build_host_family.sh tokenizer, pixi run
# check-tokenizer). The table comes from Python's standard unicodedata at the
# version tokenizer/impl/unicode_class.mojo pins; see
# tokenizer/tools/gen_unicode_categories.py.
#
# Which Python: python3 on PATH when its unicodedata is the pinned version,
# else a pinned CPython fetched by `pixi exec` (Unicode 16.0.0 is CPython
# 3.14's). A pin change must move PINNED_PYTHON with it.
set -eu
cd "$(dirname -- "$0")/../.."
PINNED_PYTHON="python=3.14"
want=$(sed -n 's/^comptime UNICODE_VERSION_PINNED = "\([^"]*\)"$/\1/p' tokenizer/impl/unicode_class.mojo)
[ -n "$want" ] || { echo "gen_unicode_table: no UNICODE_VERSION_PINNED in tokenizer/impl/unicode_class.mojo" >&2; exit 2; }
if command -v python3 >/dev/null 2>&1 \
    && python3 -c "import sys, unicodedata; sys.exit(unicodedata.unidata_version != '$want')" 2>/dev/null; then
    exec python3 tokenizer/tools/gen_unicode_categories.py
fi
if ! command -v pixi >/dev/null 2>&1; then
    echo "gen_unicode_table: python3 on PATH does not have unicodedata $want and pixi is not installed to fetch $PINNED_PYTHON" >&2
    exit 2
fi
echo "gen_unicode_table: python3 on PATH is not unicodedata $want; using pixi exec --spec $PINNED_PYTHON" >&2
exec pixi exec --spec "$PINNED_PYTHON" -- python3 tokenizer/tools/gen_unicode_categories.py
