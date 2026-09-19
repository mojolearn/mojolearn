#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Regenerate bench/border_types_oracle.txt: CatBoost's own borders for every
`feature_border_type` on columns built to exercise each binarizer.

    python3 tools/border_types_oracle.py            # needs catboost==1.2.10

`checks/border_types_check.mojo` (`pixi run check-border-types`) holds
`gbdt/grid_creator/binarization.mojo::select_borders` and `best_split` to
these borders EXACTLY (float32 bits). CatBoost is only needed to regenerate
the file; the check reads the committed file.

The borders come from `Pool.quantize(border_count=b, feature_border_type=t)`
and `Pool.save_quantization_borders`, which is the dense-float
`NSplitSelection::BestSplit` path both of their learners quantize with
(`libs/data/quantization.cpp`), below the 200,000-row border subsample, with
no NaN (so nan_mode adds no border). Every column is float32 and no value is
subnormal, so the flush `select_borders` applies is inert here.

File format (whitespace separated, one token per line after each header;
floats as the DECIMAL of their float32 bit pattern, so nothing is parsed
through a decimal float):

    columns <n>
    column <id> <rows>
    <bits> ...
    cases <n>
    case <column> <type> <budget> <n_borders>
    <bits> ...
"""
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

TYPES = ("GreedyLogSum", "Median", "Uniform", "UniformAndQuantiles",
         "MaxLogSum", "MinEntropy", "GreedyMinEntropy")
BUDGETS = (1, 7, 15, 32, 63, 128, 254)
ROOT = Path(__file__).resolve().parents[1]


def columns():
    rng = np.random.default_rng(20260919)
    cols = [
        rng.integers(0, 256, 2048).astype(np.float32),                  # heavy ties, a grid
        rng.normal(size=3000).astype(np.float32),                      # smooth, distinct
        np.exp(rng.normal(0.0, 2.0, 2500)).astype(np.float32),         # skewed, wide magnitudes
        rng.choice(np.array([-3, -1, 0, 0.5, 2, 7, 11], np.float32), 1500),  # seven levels
        np.round(rng.normal(size=2000) * 4.0).astype(np.float32) / 4,  # quarter grid, signs, zeros
        np.concatenate([np.zeros(900, np.float32),
                        rng.exponential(3.0, 1100).astype(np.float32)]),  # a dominant value
    ]
    for c in cols:
        rng.shuffle(c)
    return cols


def catboost_borders(col, border_type, budget):
    import catboost
    pool = catboost.Pool(col.reshape(-1, 1))
    pool.quantize(border_count=budget, feature_border_type=border_type)
    fd, path = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    try:
        pool.save_quantization_borders(path)
        values = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                feature, value = line.split("\t")[:2]
                assert int(feature) == 0
                values.append(np.float32(float(value)))
    finally:
        os.unlink(path)
    return np.sort(np.asarray(values, dtype=np.float32))


def bits(a):
    return [int(v) for v in np.asarray(a, dtype=np.float32).view(np.uint32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "border_types_oracle.txt")
    args = ap.parse_args()
    import catboost
    cols = columns()
    out = [f"columns {len(cols)}"]
    for i, c in enumerate(cols):
        out.append(f"column {i} {len(c)}")
        out.extend(str(b) for b in bits(c))
    cases = []
    for i, c in enumerate(cols):
        for t in TYPES:
            for b in BUDGETS:
                borders = catboost_borders(c, t, b)
                cases.append((i, t, b, borders))
    out.append(f"cases {len(cases)}")
    for i, t, b, borders in cases:
        out.append(f"case {i} {t} {b} {len(borders)}")
        out.extend(str(x) for x in bits(borders))
    args.out.write_text("\n".join(out) + "\n")
    print(f"wrote {args.out}: {len(cols)} columns, {len(cases)} cases, catboost {catboost.__version__}")


if __name__ == "__main__":
    main()
