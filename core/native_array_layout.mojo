# SPDX-License-Identifier: Apache-2.0
"""Native output shape, stride and allocation metadata."""


def native_array_itemsize(typestr: String) raises -> Int:
    if typestr == "<f8" or typestr == "<i8":
        return 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<f2" or typestr == "<u2":
        return 2
    if typestr == "<u1" or typestr == "<i1":
        return 1
    raise Error("TypeError: " + typestr + " is not an Array dtype")


def native_array_code(typestr: String) raises -> String:
    if typestr == "<f4":
        return "f"
    if typestr == "<f8":
        return "d"
    if typestr == "<i8":
        return "q"
    if typestr == "<i4":
        return "i"
    if typestr == "<u4":
        return "I"
    if typestr == "<u1":
        return "B"
    if typestr == "<i1":
        return "b"
    if typestr == "<f2" or typestr == "<u2":
        return "H"
    raise Error("TypeError: " + typestr + " is not an Array dtype")


struct NativeArrayLayout(Copyable, Movable):
    var shape: List[Int]
    var strides: List[Int]
    var count: Int
    var itemsize: Int
    var nbytes: Int
    var order: String

    def __init__(out self, var shape: List[Int], itemsize: Int, var order: String = "C") raises:
        self.count = 1
        for dimension in shape:  # driver-loop: over native output rank metadata
            if dimension < 0:
                raise Error("ValueError: negative dimensions are not allowed")
            if dimension != 0 and self.count > 9223372036854775807 // dimension:
                raise Error("ValueError: array element count exceeds native storage")
            self.count *= dimension
        if itemsize <= 0 or self.count > 9223372036854775807 // itemsize:
            raise Error("ValueError: array byte count exceeds native storage")
        self.itemsize = itemsize
        self.nbytes = self.count * itemsize
        self.strides = List[Int](length=len(shape), fill=0)
        var step = itemsize
        for k in range(len(shape)):  # driver-loop: over native output stride metadata
            var axis = len(shape) - 1 - k if order == "C" else k
            self.strides[axis] = step
            step *= shape[axis]
        self.shape = shape^
        self.order = order^


def native_matrix_shape(rows: Int, cols: Int) -> List[Int]:
    var shape = List[Int]()
    shape.append(rows)
    if cols != 0:
        shape.append(cols)
    return shape^
