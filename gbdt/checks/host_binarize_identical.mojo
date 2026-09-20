# SPDX-License-Identifier: Apache-2.0
"""Exact gate for the host GBDT lower-bound compressed-index builder."""

from gbdt.data.quantization import NAN_TREATMENT_AS_IS
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.host.gbdt_oracle import GbdtHostGrid, _binarize_columns


def main() raises:
    var folds: List[Int] = [1, 2, 3, 4, 7, 15, 31, 63, 128, 255]
    var features = len(folds)
    var rows = 4096
    var borders = List[List[Float32]]()
    var nan = List[Int](length=features, fill=NAN_TREATMENT_AS_IS)
    for f in range(features):
        var bs = List[Float32]()
        for b in range(folds[f]):
            bs.append(Float32(2 * b - folds[f]) / Float32(16.0))
        borders.append(bs^)

    var x = List[Float32](length=rows * features, fill=Float32(0.0))
    for f in range(features):
        for r in range(rows):
            # Includes values equal to borders, between borders, outside both
            # ends, and signed zero.  Equality is the important strictness
            # case: bin = count(border < value), not count(border <= value).
            var q = (r * 37 + f * 11) % (4 * folds[f] + 9)
            var v = Float32(q - 2 * folds[f] - 4) / Float32(32.0)
            if r % 257 == 0:
                v = Float32(-0.0)
            x[f * rows + r] = v

    var layout = build_layout(folds)
    var grid = GbdtHostGrid(borders^, folds.copy(), nan^)
    var got = _binarize_columns(x, rows, features, grid, layout)

    var want = List[UInt32](
        length=rows * layout.columns, fill=UInt32(0)
    )
    for f in range(features):
        ref cf = layout.features[f]
        var base = Int(cf.offset) * rows
        for r in range(rows):
            var index = UInt32(0)
            var v = x[f * rows + r]
            for b in range(folds[f]):
                if v > grid.borders[f][b]:
                    index += 1
            want[base + r] = want[base + r] | (
                (index & cf.mask) << cf.shift
            )

    if len(got) != len(want):
        raise Error("host binarize output length changed")
    for i in range(len(got)):
        if got[i] != want[i]:
            raise Error(
                "host lower_bound differs from linear reference at cell "
                + String(i) + ": got " + String(got[i])
                + ", expected " + String(want[i])
            )
    print(
        "host GBDT binarize lower_bound: exact over",
        rows * features,
        "values and every packing policy",
    )
