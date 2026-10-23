# SPDX-License-Identifier: Apache-2.0
"""Host steps of the graph layers' native forward
(bindings/xcnn_graph_native.mojo): the edge-index admission check and the
bias rows. No arithmetic on the layer's values."""

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def graph_edges_in_range(edges: Int, count: Int, n: Int) -> Bool:
    """Every one of the `count` int64 node ids at `edges` is in [0, n)
    (`ei.min() >= 0 and ei.max() < n`)."""
    var p = _I64P(unsafe_from_address=edges)
    for i in range(count):
        var v = p[i]
        if v < 0 or v >= Int64(n):
            return False
    return True


def graph_bias_rows(bias: Int, out: Int, rows: Int, cols: Int):
    """`np.broadcast_to(bias, (rows, cols))` written C-contiguous."""
    var src = _F32P(unsafe_from_address=bias)
    var dst = _F32P(unsafe_from_address=out)
    for r in range(rows):
        for c in range(cols):
            dst[r * cols + c] = src[c]
