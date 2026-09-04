# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from max.gpu.host import DeviceContext
from std.memory import bitcast

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from umap.estimator import fit_transform
from umap.params import UMAPParams


def main() raises:
    var ctx = DeviceContext()
    # Irregular one-dimensional spacing avoids k-NN ties while exercising a
    # two-dimensional public output.
    var x: List[Float32] = [0, 1, 2.2, 4, 6.5, 10, 14.5, 20]
    var params = UMAPParams(
        n_neighbors=3, n_components=2, n_epochs=4, random_seed=UInt64(19)
    )
    var first = fit_transform(ctx, x, 8, 1, params)
    var second = fit_transform(ctx, x, 8, 1, params)
    if len(first) != 16 or len(second) != 16:
        raise Error("UMAP fit_transform returned the wrong layout shape")
    var changed = False
    for i in range(16):
        if first[i] != first[i]:
            raise Error("UMAP fit_transform returned a non-finite layout")
        if first[i] != Float32(0.0):
            changed = True
    if not changed:
        raise Error("UMAP fit_transform returned a collapsed layout")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        for i in range(16):
            if bitcast[DType.uint32](first[i]) != bitcast[DType.uint32](
                second[i]
            ):
                raise Error("UMAP IDENTICAL end-to-end layout changed bits")
    var refused = False
    try:
        _ = fit_transform(
            ctx, x, 8, 1,
            UMAPParams(n_neighbors=3, n_components=4, n_epochs=4),
        )
    except:
        refused = True
    if not refused:
        raise Error("UMAP fit_transform admitted an unsupported dimension")
    print("UMAP end-to-end estimator PASS")

