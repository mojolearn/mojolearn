#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# batchv_quality_sk.sh: after tools/batchv_quality.sh, sklearn's float64
# PowerTransformer on the same fixtures, and each arm's lambdas / transform
# against it (BATCHV-Q sk-<arm> lines). Quality only; times nothing.
set -u
cd "$(dirname "$0")/.."
out=$HOME/batchv-q
PY=${BATCHV_PY:-$HOME/board-0834/cache/venv/bin/python}
$PY tools/batchv_quality.py dump sk "$out/prep_sk.npz" || exit 1
for a in off pt ptns; do $PY tools/batchv_quality.py cmp "$out/prep_sk.npz" "$out/prep_$a.npz" "sk-$a"; done
