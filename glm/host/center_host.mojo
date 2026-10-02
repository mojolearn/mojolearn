# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column of glm/impl/center_device.mojo (lane hr-small-passes,
2026-10-02): the same items (glm/impl/center_items.mojo), so the same
words. The column sums are exact integer sums, so the host's one pass in
row order is the device's blocked fold's total."""
from glm.impl.center_items import CS_WORDS, exact_add, exact_finish, center_cell, scale_cell

comptime _F32P = MutPointer[Float32, MutAnyOrigin]
comptime _U64P = MutPointer[UInt64, MutAnyOrigin]


def col_sums_on_cpu(x: Int, dst: Int, rows: Int, cols: Int) raises:
    if rows <= 0 or cols <= 0:
        raise Error("col_sums: rows and cols must be positive")
    var xp = _F32P(unsafe_from_address=x)
    var op = _U64P(unsafe_from_address=dst)
    for c in range(cols):
        var acc = InlineArray[Int64, CS_WORDS](fill=Int64(0))
        for r in range(rows):
            exact_add(acc, xp.unsafe_load(r * cols + c))
        op.unsafe_store(c, exact_finish(acc))


def center_on_cpu(x: Int, mu: Int, dst: Int, rows: Int, cols: Int):
    var xp = _F32P(unsafe_from_address=x)
    var mp = _F32P(unsafe_from_address=mu)
    var op = _F32P(unsafe_from_address=dst)
    for i in range(rows * cols):
        op.unsafe_store(i, center_cell(xp.unsafe_load(i), mp.unsafe_load(i % cols)))


def scale_rows_on_cpu(x: Int, w: Int, dst: Int, rows: Int, cols: Int):
    var xp = _F32P(unsafe_from_address=x)
    var wp = _F32P(unsafe_from_address=w)
    var op = _F32P(unsafe_from_address=dst)
    for i in range(rows * cols):
        op.unsafe_store(i, scale_cell(xp.unsafe_load(i), wp.unsafe_load(i // cols)))
