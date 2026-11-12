# SPDX-License-Identifier: Apache-2.0
"""RUN of the native Array owner (bindings/array_native.mojo): rank and stride
arithmetic, dtype predicates and the helper-missing refusal. No Python import
(tools/python_free_dirs.sh). Lane rs-shared-infra-b (2026-11-12): moved word
for word; rank-sized metadata, nothing to split across cores."""


def _rebuild(key: String) -> Error:
    return Error("RuntimeError: mojolearn: this binary has no `" + key
                 + "` helper (an older build); rebuild it")


def _prod(values: List[Int]) -> Int:
    var n = 1
    for i in range(len(values)):  # driver-loop: over array rank metadata
        n *= values[i]
    return n


def _c_strides(shape: List[Int], itemsize: Int) -> List[Int]:
    var out = List[Int](length=len(shape), fill=0)
    var acc = itemsize
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        var axis = len(shape) - 1 - k
        out[axis] = acc
        acc *= shape[axis]
    return out^


def _f_strides(shape: List[Int], itemsize: Int) -> List[Int]:
    var out = List[Int](length=len(shape), fill=0)
    var acc = itemsize
    for axis in range(len(shape)):  # driver-loop: over array rank metadata
        out[axis] = acc
        acc *= shape[axis]
    return out^


def _both_orders(shape: List[Int]) -> Bool:
    var big = 0
    for i in range(len(shape)):  # driver-loop: over array rank metadata
        if shape[i] != 1:
            big += 1
    return big <= 1


def _is_int(dtype: String) -> Bool:
    return (dtype == "<i4" or dtype == "<i8" or dtype == "<u4" or dtype == "<u1"
            or dtype == "<u2" or dtype == "<i1")


def _is_float(dtype: String) -> Bool:
    return dtype == "<f4" or dtype == "<f8" or dtype == "<f2"


def _address_in(address: Int, members: List[Int]) -> Bool:
    for i in range(len(members)):  # driver-loop: over the few connected class-set members
        if members[i] == address:
            return True
    return False
