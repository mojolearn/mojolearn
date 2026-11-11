# SPDX-License-Identifier: Apache-2.0
"""Native NMF output metadata and private fully-overwritten scratch owners."""
from core.native_array_layout import NativeArrayLayout


def nmf_output_shape(rows: Int, cols: Int) -> List[Int]:
    # Both dimensions are retained, including a zero second dimension.
    var shape = List[Int]()
    shape.append(rows)
    shape.append(cols)
    return shape^


struct NMFScratch(Movable):
    var values: List[Float32]

    def __init__(out self, rows: Int, cols: Int) raises:
        var layout = NativeArrayLayout(nmf_output_shape(rows, cols), 4)
        # nmf_run writes every output word before successful return. These
        # discarded outputs never need Python Array packing or a zero fill;
        # finite refusals/errors publish neither this storage nor its bytes.
        self.values = List[Float32](unsafe_uninit_length=layout.count)

    def address(self) -> Int:
        return Int(self.values.unsafe_ptr()) if len(self.values) != 0 else 0
