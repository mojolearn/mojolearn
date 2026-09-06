# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Sliced epsilon fixtures for the shape sweep (checks/shape_sweep.mojo).

    pixi run -e bench python tools/shape_slices.py

Cuts (rows, feats) sub-fixtures out of the cached epsilon binning at 128
borders (their GPU default) so `t = a + b*rows + c*feats + d*rows*feats`
can be fitted from measured points. Slices are PREFIXES on both axes --
first N rows of the first F features -- so every sliced cell is
bit-identical to the same cell of the full fixture and nothing is
re-quantized: the grid stays CatBoost's own, per feature, unchanged.

Requires `tools/epsilon_prep.py` to have run (fixtures in
~/.cache/mojolearn). Emits eps<rows>k_<feats>f_{bins_128.u8,folds_128.txt,
y.f32} in the same cache dir. The full 400k x 2000 point reuses the
existing epsilon files; the sweep lists it separately.
"""
import os

import numpy as np

CACHE = os.path.expanduser("~/.cache/mojolearn")
ROWS = 400_000
FEATS = 2_000

# (rows, feats) grid: corners + a middle point for lack-of-fit visibility.
SLICES = [
    (100_000, 500),
    (100_000, 2_000),
    (400_000, 500),
    (200_000, 1_000),
]


def main():
    bins = np.fromfile(
        os.path.join(CACHE, "epsilon_bins_128.u8"), dtype=np.uint8
    ).reshape(FEATS, ROWS)
    folds = [
        int(line)
        for line in open(os.path.join(CACHE, "epsilon_folds_128.txt"))
        if line.strip()
    ]
    y = np.fromfile(
        os.path.join(CACHE, "epsilon_y.f32"), dtype=np.float32
    )
    assert len(folds) == FEATS and len(y) == ROWS

    for r, f in SLICES:
        stem = os.path.join(CACHE, "eps%dk_%df" % (r // 1000, f))
        np.ascontiguousarray(bins[:f, :r]).tofile(stem + "_bins_128.u8")
        with open(stem + "_folds_128.txt", "w") as fh:
            for c in folds[:f]:
                fh.write("%d\n" % c)
        y[:r].tofile(stem + "_y.f32")
        print("wrote", stem, r, "rows x", f, "feats")


if __name__ == "__main__":
    main()
