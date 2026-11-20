# SPDX-License-Identifier: Apache-2.0
"""Owned dense label codes; public class objects stay at the boundary."""


struct NativeLabelCodes(Copyable, Movable):
    var values: List[Int32]

    def __init__(out self, count: Int) raises:
        if count < 0:
            raise Error("native label codes require a nonnegative count")
        self.values = List[Int32](length=count, fill=0)

    def __init__(out self, var values: List[Int32]):
        self.values = values^

    def address(self) -> Int:
        return Int(self.values.unsafe_ptr())

    def count(self) -> Int:
        return len(self.values)
