# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Non-finite UMAP hyperparameters must refuse before numerical work."""

from std.memory import bitcast
from umap.curve import fit_umap_curve
from umap.graph import fuzzy_simplicial_graph
from umap.params import UMAPParams


def main() raises:
    var patterns: List[UInt32] = [0x7FC00001, 0x7F800000, 0xFF800000]
    var idx: List[UInt32] = [0, 1, 1, 0]
    var dist: List[Float32] = [0, 1, 0, 1]
    for pattern in patterns:
        var value = bitcast[DType.float32](pattern)
        for field in range(6):
            var refused = False
            try:
                if field == 0:
                    UMAPParams(n_neighbors=2, min_dist=value).validate(2)
                elif field == 1:
                    UMAPParams(n_neighbors=2, spread=value).validate(2)
                elif field == 2:
                    UMAPParams(n_neighbors=2, set_op_mix_ratio=value).validate(2)
                elif field == 3:
                    _ = fit_umap_curve(value, Float32(1.0))
                elif field == 4:
                    _ = fit_umap_curve(Float32(0.1), value)
                else:
                    _ = fuzzy_simplicial_graph(idx, dist, 2, 2, value)
            except:
                refused = True
            if not refused:
                raise Error("UMAP admitted a non-finite hyperparameter")
    print("UMAP non-finite parameter refusal PASS: 18 cases")
