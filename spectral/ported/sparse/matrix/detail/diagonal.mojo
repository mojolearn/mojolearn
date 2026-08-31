# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`raft/sparse/matrix/detail/diagonal.cuh`, the COO overloads that
`laplacian_normalized` reaches on the 26.08 path: `diagonal` (`:156-177`),
`scale_by_diagonal_symmetric` (`:194-216`) and `set_diagonal` (the COO
arm). All three are `map_offset` lambdas over the nnz entries -- one thread
per entry, no fold -- and are transliterated as one-thread-per-entry kernels.

`diagonal` in theirs is `diag_ptr[rows[idx]] = values[idx]` whenever `rows ==
cols`: with one diagonal entry per row (which `compute_graph_laplacian`
guarantees by inserting a zero where none existed, and DEVIATION 775's
refusal of repeated keys guarantees there is never a second) it is a plain
scatter with no race.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined

from mojo_only.numerics import ftz

#: SABOTAGE. Reassociate seam L6 to `row_scale * (value * col_scale)`.
#: Theirs is `row_scale * value * col_scale`, C++ left to right
#: (`diagonal.cuh:216`), TWO roundings in THAT order. This arm keeps two
#: roundings and moves the parenthesis, so it is a pure ASSOCIATIVITY
#: change with no fusion in it -- the smallest perturbation the
#: normalization seam admits. Reaches the DEVICE arm only: the oracle
#: spells L6 inline in `host_laplacian`. Must FAIL device == oracle at
#: `spectral.L.vals`. Contract section 9.
comptime SAB_LAPLACIAN_SEAM = is_defined[
    "MOJOLEARN_SPECTRAL_SABOTAGE_LAPLACIAN_SEAM"
]()


def coo_diagonal_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    diag: MutPointer[Float32, MutAnyOrigin],
    nnz_in: Int32,
):
    """`diagonal.cuh:173-176`: `if (rows[idx] == cols[idx]) diag[rows[idx]] =
    values[idx]`. The caller zero-fills `diag` first (`:169`)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(nnz_in):
        return
    var r = rows.unsafe_load(idx)
    if r == cols.unsafe_load(idx):
        diag.unsafe_store(Int(r), vals.unsafe_load(idx))


def coo_scale_by_diagonal_symmetric_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    diag: MutPointer[Float32, MutAnyOrigin],
    nnz_in: Int32,
):
    """`diagonal.cuh:209-216`: `row_scale = d[row] == 0 ? 0 : 1/d[row]`,
    `col_scale` likewise, `value = row_scale * value * col_scale` -- TWO
    multiplies in C++'s left-to-right order, `(row_scale * value) *
    col_scale`, each a single rounding; `1.0f / d` is a correctly rounded
    division (row 10: correct on normals on every column measured). Seams
    flushed. The `== 0` arms are unreachable after `zero_to_one_functor`
    (`laplacian.cuh:272-273`) but are transliterated, not removed."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(nnz_in):
        return
    var r = Int(rows.unsafe_load(idx))
    var c = Int(cols.unsafe_load(idx))
    var dr = diag.unsafe_load(r)
    var dc = diag.unsafe_load(c)
    var row_scale = Float32(0.0)
    if dr != Float32(0.0):
        row_scale = ftz(Float32(1.0) / dr)
    var col_scale = Float32(0.0)
    if dc != Float32(0.0):
        col_scale = ftz(Float32(1.0) / dc)
    comptime if SAB_LAPLACIAN_SEAM:
        var t = ftz(vals.unsafe_load(idx) * col_scale)
        vals.unsafe_store(idx, ftz(row_scale * t))
    else:
        var t = ftz(row_scale * vals.unsafe_load(idx))
        vals.unsafe_store(idx, ftz(t * col_scale))


def coo_set_diagonal_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    nnz_in: Int32,
    scalar: Float32,
):
    """The COO `set_diagonal`: every `rows == cols` entry becomes `scalar`."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(nnz_in):
        return
    if rows.unsafe_load(idx) == cols.unsafe_load(idx):
        vals.unsafe_store(idx, scalar)
