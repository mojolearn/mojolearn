# SPDX-License-Identifier: Apache-2.0
"""Owned scalar-label UNPACK words for the existing numeric label kernels."""


struct GPLabelInput(Copyable, Movable):
    var wide: Bool
    var as_bool: Bool
    var values: List[UInt64]

    def __init__(out self, count: Int, wide: Bool, as_bool: Bool) raises:
        if count < 0 or (wide and as_bool):
            raise Error("GP label input: invalid count or scalar kind")
        self.wide = wide
        self.as_bool = as_bool
        self.values = List[UInt64](length=count, fill=0)

    def code(self) -> Int:
        return 1 if self.wide else 3

    def address(self) -> Int:
        return Int(self.values.unsafe_ptr())

    def count(self) -> Int:
        return len(self.values)
