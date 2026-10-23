# SPDX-License-Identifier: Apache-2.0
"""The column moves of `mojolearn.parallel_forecasting`'s Holt-Winters
shards (bindings/parallel_forecast_native.mojo): copies, no arithmetic."""

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def forecast_gather_cols(dst_addr: Int, src_addr: Int, rows: Int, batch: Int, lo: Int, w: Int):
    """dst[r * w + j] = src[r * batch + lo + j] for r < rows, j < w."""
    var dst = _F32P(unsafe_from_address=dst_addr)
    var src = _F32P(unsafe_from_address=src_addr)
    for r in range(rows):
        for j in range(w):
            dst[r * w + j] = src[r * batch + lo + j]


def forecast_scatter_cols(dst_addr: Int, src_addr: Int, rows: Int, batch: Int, lo: Int, w: Int):
    """dst[r * batch + lo + j] = src[r * w + j] for r < rows, j < w."""
    var dst = _F32P(unsafe_from_address=dst_addr)
    var src = _F32P(unsafe_from_address=src_addr)
    for r in range(rows):
        for j in range(w):
            dst[r * batch + lo + j] = src[r * w + j]
