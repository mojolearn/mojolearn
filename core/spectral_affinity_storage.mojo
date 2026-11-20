# SPDX-License-Identifier: Apache-2.0
"""Native generated-affinity storage for standalone spectral embedding."""


struct SpectralAffinityStorage(Movable):
    var rows: Int
    var values: List[Float32]

    def __init__(out self, rows: Int):
        self.rows = rows
        self.values = List[Float32](length=rows * rows, fill=0)

    def address(self) -> Int:
        return Int(self.values.unsafe_ptr())
