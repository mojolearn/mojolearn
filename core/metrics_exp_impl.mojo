# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host row helpers of the classification, ranking and clustering metrics'
native owner (bindings/metrics_exp_native.mojo): the label input step's
remap of dense class codes onto a caller's label order (what
`_metrics_impl._label_map` gathered through the base binding's
`gather_i32`), and the contingency matrix's dtype cast (what
`Array.astype` did). Pure functions of their inputs, no Python."""


def mexp_gather_i32(
    table: MutPointer[Int32, MutUntrackedOrigin], nt: Int,
    codes: MutPointer[Int32, MutUntrackedOrigin], n: Int,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) -> Int:
    """`dst[i] = table[codes[i]]` for the n rows (codes in [0, nt)). Returns
    the first row whose mapped value is -1 (a label outside the caller's
    order), or -1 when there is none."""
    var first = -1
    for i in range(n):
        var c = Int(codes[i])
        var v = Int32(-1)
        if c >= 0 and c < nt:
            v = table[c]
        dst[i] = v
        if v < 0 and first < 0:
            first = i
    return first


def mexp_cast_i64(
    src: MutPointer[Int64, MutUntrackedOrigin], n: Int, dst_addr: Int, code: Int,
):
    """The n Int64 counts as int32 (code 0), float64 (code 1) or float32
    (code 2), the conversions `Array.astype` made."""
    if code == 0:
        var d = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
        for i in range(n):
            d[i] = src[i].cast[DType.int32]()
    elif code == 1:
        var d = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=dst_addr)
        for i in range(n):
            d[i] = src[i].cast[DType.float64]()
    else:
        var d = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
        for i in range(n):
            d[i] = src[i].cast[DType.float32]()
