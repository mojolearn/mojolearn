# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane sk-weq-trees: the host column's `core/weight_compact_device.mojo`.
A weighted tree fit without bootstrap drops its zero-weight rows (scikit-
learn's splitter keeps only positively weighted samples); the CPU-only host
bindings keep the same rows in the same ascending order as the device's
flag + exclusive scan + emit, so the host column fits the same trees."""


def host_nonzero_weight_rows(w: List[Float32]) -> List[Int]:
    """The rows `i` with `w[i] != 0`, ascending."""
    var rows = List[Int]()
    for i in range(len(w)):
        if w[i] != Float32(0):
            rows.append(i)
    return rows^


def host_gather_colmajor(
    x: List[Float32], n: Int, n_cols: Int, rows: List[Int]
) -> List[Float32]:
    """Column-major `len(rows) x n_cols`: the kept rows of the column-major
    `n x n_cols` x."""
    var m = len(rows)
    var out = List[Float32](length=m * n_cols, fill=Float32(0))
    for c in range(n_cols):
        for k in range(m):
            out[c * m + k] = x[c * n + rows[k]]
    return out^


def host_gather_f32(v: List[Float32], rows: List[Int]) -> List[Float32]:
    var out = List[Float32](length=len(rows), fill=Float32(0))
    for k in range(len(rows)):
        out[k] = v[rows[k]]
    return out^


def host_gather_i32(v: List[Int32], rows: List[Int]) -> List[Int32]:
    var out = List[Int32](length=len(rows), fill=Int32(0))
    for k in range(len(rows)):
        out[k] = v[rows[k]]
    return out^


def host_all_unit(w: List[Float32]) -> Bool:
    for i in range(len(w)):
        if w[i] != Float32(1):
            return False
    return True
