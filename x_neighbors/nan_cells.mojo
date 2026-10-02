# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`op_nan_cells` (lane/neural-pass71, 2026-10-01): the NaN cells of an
n x d float32 matrix, flat indices ascending, with the NaN count of every
column and the total: KNNImputer's missing-cell list and column flags,
which its Python layer built by walking every cell of the matrix as a list
(the board's taxi block: 52 of its 85 ms; Istella's 22 million cells,
seconds). One host pass, no arithmetic: a cell is missing when it is not
equal to itself."""
from x_neighbors.items import FP, IP


def nan_cells_host(x: FP, cells: IP, colmiss: IP, info: IP, n: Int, d: Int):
    for f in range(d):
        colmiss.unsafe_store(f, Int32(0))
    var c = 0
    var i = 0
    for r in range(n):
        for f in range(d):
            var v = x.unsafe_load(i)
            if v != v:
                cells.unsafe_store(c, Int32(i))
                c += 1
                colmiss.unsafe_store(f, colmiss.unsafe_load(f) + Int32(1))
            i += 1
    info.unsafe_store(0, Int32(c))
