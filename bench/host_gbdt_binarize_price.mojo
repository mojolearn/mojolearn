# SPDX-License-Identifier: Apache-2.0
"""A/B price receipt for the production host GBDT compressed-index build.

Run the current lower-bound implementation and its exact legacy arm with:

    mojo run -I . bench/host_gbdt_binarize_price.mojo
    mojo run -D MOJOLEARN_GBDT_HOST_BINARIZE_LINEAR=1 -I . \
        bench/host_gbdt_binarize_price.mojo
"""

from std.time import perf_counter_ns

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.host.gbdt_oracle import (
    GBDT_HOST_BINARIZE_LINEAR,
    GbdtHostGrid,
    _binarize_columns,
)
from gbdt.data.quantization import NAN_TREATMENT_AS_IS


def _hash(x: List[UInt32]) -> UInt64:
    var h = UInt64(1469598103934665603)
    for i in range(len(x)):
        var v = UInt64(x[i])
        for b in range(4):
            h = (h ^ ((v >> UInt64(8 * b)) & UInt64(255))) * UInt64(
                1099511628211
            )
    return h


def main() raises:
    # Large enough to represent the host fitting path while staying suitable
    # for a local gate: 4M values, 128 borders per feature.
    var rows = 250_000
    var features = 16
    var x = List[Float32](
        length=rows * features, fill=Float32(0.0)
    )
    for f in range(features):
        for r in range(rows):
            # Repeated equality cases are deliberate: lower_bound must retain
            # the strict `value > border` convention at every border.
            var code = (r * 73 + f * 29) % 1024
            x[f * rows + r] = Float32(code - 512) / Float32(64.0)

    var borders = List[List[Float32]]()
    var folds = List[Int]()
    var nan = List[Int]()
    for _ in range(features):
        var bs = List[Float32]()
        for b in range(128):
            bs.append(Float32(b - 64) / Float32(8.0))
        borders.append(bs^)
        folds.append(128)
        nan.append(NAN_TREATMENT_AS_IS)
    var grid = GbdtHostGrid(borders^, folds.copy(), nan^)
    var layout = build_layout(folds)

    # Warm allocation and code pages before five independent prices.
    _ = _binarize_columns(x, rows, features, grid, layout)
    for rep in range(5):
        var t0 = perf_counter_ns()
        var out = _binarize_columns(x, rows, features, grid, layout)
        var elapsed = Float64(perf_counter_ns() - t0) / 1.0e6
        print(
            "GBDT_HOST_BINARIZE_PRICE",
            "linear" if GBDT_HOST_BINARIZE_LINEAR else "lower_bound",
            rep,
            elapsed,
            _hash(out),
        )
