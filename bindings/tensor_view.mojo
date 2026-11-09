# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The buffer facts one admitted tensor carries (`TensorView`) and the pure
# metadata helpers that read them, with no Python import: moved word for word
# from bindings/optimizer_public_door.mojo (lane a6-device-x-family-classical,
# 2026-11-09) so the one admission path (bindings/admitted_input.mojo
# `AdmittedTyped`) and the optimizer door share them without the admission
# path importing the optimizer door.


def shape_text(dims: List[Int]) -> String:
    """A shape as Python prints a tuple: `(3, 4)`, `(5,)`, `()`."""
    var text = String("(")
    for i in range(len(dims)):  # driver-loop: over shape words of a refusal message
        if i > 0:
            text += ", "
        text += String(dims[i])
    if len(dims) == 1:
        text += ","
    return text + ")"


struct TensorView(Copyable, Movable):
    """The facts `_bufcheck.probe` read, the address `_buffer.addr` read, and
    the dtype name a refusal prints."""
    var addr: Int
    var nbytes: Int
    var itemsize: Int
    var readonly: Bool
    var c_contig: Bool
    var fmt: String
    var dtype: String
    var shape: List[Int]
    var strides: List[Int]
    var has_strides: Bool
    var is_array: Bool

    def __init__(out self):
        self.addr = 0
        self.nbytes = 0
        self.itemsize = 0
        self.readonly = False
        self.c_contig = True
        self.fmt = String("B")
        self.dtype = String("")
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.has_strides = False
        self.is_array = False

    def count(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):  # driver-loop: over shape words
            n *= self.shape[i]
        return n


def _contiguous(shape: List[Int], strides: List[Int], has_strides: Bool, itemsize: Int) -> Bool:
    """`_buffer._contiguity`'s C half (NumPy's rule)."""
    if len(shape) == 0:
        return True
    var n = 1
    for i in range(len(shape)):  # driver-loop: over shape words
        n *= shape[i]
    if n <= 1 or not has_strides:
        return True
    var expect = itemsize
    for k in range(len(shape) - 1, -1, -1):  # driver-loop: over one stride word per axis
        if shape[k] != 1 and strides[k] != expect:
            return False
        expect *= shape[k]
    return True


def base_format(fmt: String) -> String:
    """`_bufcheck.base_format`: a byte-order mark naming the host's order
    removed (this library's hosts are little-endian)."""
    if fmt.byte_length() > 1 and (fmt.startswith("@") or fmt.startswith("=") or fmt.startswith("<")):
        return String(fmt[byte=1:])
    return fmt


def _format_name(code: String, itemsize: Int, fmt: String) -> String:
    if code == "f":
        return "float32"
    if code == "d":
        return "float64"
    if code == "e":
        return "float16"
    if code == "g":
        return "longdouble"
    if code == "?":
        return "bool"
    if code == "b":
        return "int8"
    if code == "B":
        return "uint8"
    if code == "h":
        return "int16"
    if code == "H":
        return "uint16"
    if code == "i":
        return "int32"
    if code == "I":
        return "uint32"
    if code == "q":
        return "int64"
    if code == "Q":
        return "uint64"
    if code == "n":
        return "intp"
    if code == "N":
        return "uintp"
    if code == "c":
        return "bytes"
    if code == "l":
        return "int" + String(8 * itemsize)
    if code == "L":
        return "uint" + String(8 * itemsize)
    return "'" + fmt + "'"


def _typestr_name(t: String) -> String:
    if t == "<f4" or t == ">f4":
        return "float32"
    if t == "<f8" or t == ">f8":
        return "float64"
    if t == "<f2":
        return "float16"
    if t == "<i4" or t == ">i4":
        return "int32"
    if t == "<i8" or t == ">i8":
        return "int64"
    if t == "<u4" or t == ">u4":
        return "uint32"
    if t == "<u1" or t == "|u1":
        return "uint8"
    if t == "<i2":
        return "int16"
    if t == "<u2":
        return "uint16"
    if t == "|i1":
        return "int8"
    return ""


def _array_code(t: String) -> String:
    """`_array._CODE`: an Array dtype's struct code."""
    if t == "<f4":
        return "f"
    if t == "<f8":
        return "d"
    if t == "<i4":
        return "i"
    if t == "<i8":
        return "q"
    if t == "<u4":
        return "I"
    if t == "<u1":
        return "B"
    if t == "<f2" or t == "<u2":
        return "H"
    if t == "<i1":
        return "b"
    return "B"
