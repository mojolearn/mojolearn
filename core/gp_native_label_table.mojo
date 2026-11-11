# SPDX-License-Identifier: Apache-2.0
"""Native numeric gather table; arbitrary-width/object classes stay public pins."""

comptime GP_LABEL_OBJECT = 0
comptime GP_LABEL_I64 = 1
comptime GP_LABEL_F64 = 2


struct GPLabelTable(Movable):
    var kind: Int
    var available: Bool
    var integers: List[Int64]
    var reals: List[Float64]

    def __init__(out self, kind: Int, count: Int) raises:
        if count < 0 or kind < 0 or kind > 2:
            raise Error("GP label table: invalid count or kind")
        self.kind = kind
        self.available = kind != GP_LABEL_OBJECT and count > 0
        self.integers = List[Int64](length=count if kind == GP_LABEL_I64 else 0, fill=0)
        self.reals = List[Float64](length=count if kind == GP_LABEL_F64 else 0, fill=0.0)

    def address(self) -> Int:
        if not self.available:
            return 0
        return Int(self.integers.unsafe_ptr()) if self.kind == GP_LABEL_I64 else Int(self.reals.unsafe_ptr())

    def refuse_numeric(mut self):
        # A class outside int64 retains the existing object-label output.
        self.available = False
