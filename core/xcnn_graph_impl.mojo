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


def graph_bias_rows(bias: Int, out_buf: Int, rows: Int, cols: Int):
    """`np.broadcast_to(bias, (rows, cols))` written C-contiguous."""
    var src = _F32P(unsafe_from_address=bias)
    var dst = _F32P(unsafe_from_address=out_buf)
    for r in range(rows):
        for c in range(cols):
            dst[r * cols + c] = src[c]


def graph_admit_edges(base: Int, row_step: Int, edge_step: Int, count: Int, item: Int, kind: Int,
                      little: Bool, nodes: Int, source: Int, target: Int) -> Bool:
    # Native input conversion, before graph execution. Logical (2,E) strides
    # may be negative or noncontiguous. Preserve integer ids exactly and the
    # former float->int64 truncation for finite values, then range-check before
    # narrowing to the CSR engine's int32 index representation.
    from std.memory import bitcast
    from core.cast_in import cast_w64_bits, CAST_KIND_FLOAT, CAST_KIND_UNSIGNED
    if nodes < 0 or nodes > 2147483647:
        return False
    var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=source)
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=target)
    for row in range(2):
        for i in range(count):
            var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=base + row * row_step + i * edge_step)
            var raw = UInt64(0)
            for byte in range(item):
                raw |= UInt64(bytes[byte if little else item - 1 - byte]) << UInt64(byte * 8)
            var word = cast_w64_bits(raw, kind, item)
            var value = Int64(0)
            if kind == CAST_KIND_FLOAT:
                var fp = bitcast[DType.float64](word)
                if not (fp > -1 and fp < Float64(nodes)):
                    return False
                value = Int64(fp)
            else:
                if kind == CAST_KIND_UNSIGNED and word >= UInt64(nodes):
                    return False
                value = bitcast[DType.int64](word)
                if value < 0 or value >= Int64(nodes):
                    return False
            if row == 0:
                src[i] = Int32(value)
            else:
                dst[i] = Int32(value)
    return True
