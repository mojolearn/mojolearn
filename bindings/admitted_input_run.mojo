# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of the one admission path (bindings/admitted_input_unpack.mojo):
the Python-free AdmittedXView and the typed admission helpers. No Python
import (tools/python_free_dirs.sh).

AdmittedXView is the composed typed view of one admitted float array
(lane admitted-x-view, 2026-11-12, the orchestrator's decision: a composed
view as resample's RPLayout, not an out-pin pass over every site). The door
keeps AdmittedX, whose `array`, `source`, `rows_base` and `rows_indices`
owners keep `address` valid; it builds the view once (`AdmittedX.view()`)
and every typed read takes the view. The door names the owner again after
the last call that reads `address` (AGENTS.md: Mojo frees a local at its
last named use). Field names match AdmittedX's typed fields, so a body that
read `x.address` / `x.rows` reads the view unchanged."""

from bindings.device_rows_door_run import DoorRowsStore
from bindings.admitted_finite_wording import admitted_finite_refuse
from bindings.tensor_view import TensorView, base_format
from core.admitted_convert import admitted_convert
from core.cast_in import CAST_KIND_UNSIGNED, CAST_KIND_SIGNED, CAST_KIND_FLOAT, CAST_KIND_BOOL


struct AdmittedXView(ImplicitlyCopyable, Movable):
    """The typed view of one admitted float32 array: what a kernel reads.

    Fields:
        address: the first float32 word (0 for a device-only fold).
        rows, cols, ndim: the shape and rank (0 is one stored word).
        row_stride, col_stride: element strides of the words (C order:
            cols and 1; F order: 1 and rows). The words are dense.
        dtype: the element type of the words (always float32 today).
        c_order: C-order words; False for an `order="F"` admission.
        contiguous: the words are one dense block (always True for an
            admitted X; the field names the contract a RUN routine checks).
        borrowed, finite, position, device_rows, store: as on AdmittedX.
    """
    var address: Int
    var rows: Int
    var cols: Int
    var ndim: Int
    var row_stride: Int
    var col_stride: Int
    var dtype: DType
    var c_order: Bool
    var contiguous: Bool
    var borrowed: Bool
    var finite: Bool
    var position: Int
    var device_rows: Bool
    var store: DoorRowsStore

    def __init__(out self, address: Int, rows: Int, cols: Int, ndim: Int, c_order: Bool, borrowed: Bool,
                 finite: Bool, position: Int, device_rows: Bool, store: DoorRowsStore):
        self.address = address
        self.rows = rows
        self.cols = cols
        self.ndim = ndim
        self.row_stride = cols if c_order else 1
        self.col_stride = 1 if c_order else rows
        self.dtype = DType.float32
        self.c_order = c_order
        self.contiguous = True
        self.borrowed = borrowed
        self.finite = finite
        self.position = position
        self.device_rows = device_rows
        self.store = store

    def count(self) -> Int:
        """The number of float32 words (rows x cols)."""
        return self.rows * self.cols

    def at(self, row: Int, col: Int = 0) -> Int:
        """The address of word (row, col)."""
        return self.address + (row * self.row_stride + col * self.col_stride) * 4


comptime RESHAPE_HINT = "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."


def _admitted_storage_key(typestr: String) raises -> String:
    if typestr == "<f4":
        return "float32"
    if typestr == "<f8":
        return "float64"
    if typestr == "<i8":
        return "int64"
    raise Error("TypeError: admitted storage holds <f4, <f8 or <i8, not " + typestr)


def _entry_nonfinite(text: String, name: String) -> Bool:
    """True when `text` is a kernel entry's own refusal of a non-finite
    `name` ("X" or "y"): the x_linear fit's "mojolearn: X contains NaN or
    infinity", the forest's "X contains NaN or infinity; the forest has no
    missing-value arm", IsolationForest's "mojolearn IsolationForest: X
    contains NaN or infinity" and the resident forest's "resident forest
    requires finite Float32 values" (X only). The admission's own wording
    ("Input X contains ...") is not an entry's: it is left as it is."""
    if text.startswith("Input ") or text.startswith("ValueError: Input "):
        return False
    if name == "X" and "resident forest requires finite Float32 values" in text:
        return True
    return (name + " contains NaN or infinity") in text and "metrics" not in text


def admitted_finite_entry_refuse(error: Error, est: String) raises:
    """The refusal of a deferred finite test that the kernel entry raised in
    its own words instead of returning the status word (the fit entries of
    x_linear, the forests and IsolationForest; lane a7c): raises
    scikit-learn's wording through `admitted_finite_refuse`, the one
    wording of every deferred refusal. Returns for any other error: the
    caller re-raises it."""
    var text = String(error)
    if _entry_nonfinite(text, "X"):
        admitted_finite_refuse(1, est, "X")
    if _entry_nonfinite(text, "y"):
        admitted_finite_refuse(1, est, "y")


def admitted_width(typestr: String) raises -> Int:
    """The item width of an Array typestr; refuses a dtype no Array holds."""
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<f8" or typestr == "<i8":
        return 8
    if typestr == "<u2" or typestr == "<f2":
        return 2
    if typestr == "<i1" or typestr == "<u1":
        return 1
    raise Error("TypeError: " + typestr + " is not an Array dtype")


def _typed_kind(v: TensorView, mut item: Int) raises -> Int:
    """The CastIn element kind of a viewed buffer (its item size in `item`);
    -1 for a format no Array dtype holds."""
    item = v.itemsize
    if v.is_array:
        var t = v.dtype
        if t == "<f4" or t == "<f8" or t == "<f2":
            return CAST_KIND_FLOAT
        if t == "<i1" or t == "<i4" or t == "<i8":
            return CAST_KIND_SIGNED
        return CAST_KIND_UNSIGNED
    var code = base_format(v.fmt)
    if code == "f" or code == "d" or code == "e":
        return CAST_KIND_FLOAT
    if code == "?":
        return CAST_KIND_BOOL
    if code == "b" or code == "h" or code == "i" or code == "l" or code == "q" or code == "n":
        return CAST_KIND_SIGNED
    if code == "B" or code == "H" or code == "I" or code == "L" or code == "Q" or code == "N":
        return CAST_KIND_UNSIGNED
    return -1


def _typed_target(typestr: String, mut item: Int) raises -> Int:
    item = admitted_width(typestr)
    if typestr == "<f4" or typestr == "<f8" or typestr == "<f2":
        return CAST_KIND_FLOAT
    if typestr == "<i1" or typestr == "<i4" or typestr == "<i8":
        return CAST_KIND_SIGNED
    return CAST_KIND_UNSIGNED


def admitted_typed_convert(v: TensorView, typestr: String, dst: Int, name: String) raises:
    """Every element of the viewed buffer, in C order, into the C-order
    `typestr` store at `dst`: a copy of each contiguous run for the same
    dtype, else one conversion per element by Array.astype's rules. The
    elements are split into contiguous C-order ranges across the host's
    cores (core/admitted_convert.mojo; each element writes only its own cell,
    so no word depends on the split); a refusal is the one the first failing
    element in C order raises."""
    var item = 0
    var kind = _typed_kind(v, item)
    if kind < 0:
        raise Error("TypeError: mojolearn: " + name + " has buffer format '" + v.fmt + "', which is not a numeric array")
    var titem = 0
    var tkind = _typed_target(typestr, titem)
    var nd = len(v.shape)
    var strides = List[Int](length=max(nd, 1), fill=0)
    var shape = List[Int](length=max(nd, 1), fill=1)
    if v.has_strides:
        for k in range(nd):  # driver-loop: over stride words
            strides[k] = v.strides[k]
    else:
        var step = v.itemsize
        for k in range(nd - 1, -1, -1):  # driver-loop: over stride words
            strides[k] = step
            step *= v.shape[k]
    for k in range(nd):  # driver-loop: over shape words
        shape[k] = v.shape[k]
    admitted_convert(v.addr, shape, strides, v.count(), item, kind, titem, tkind, typestr, dst)


def admitted_view_kind(v: TensorView, name: String) raises -> Tuple[Int, Int]:
    """(CastIn element kind, item size) of a captured view; refused for a
    format no numeric Array dtype holds."""
    var item = 0
    var kind = _typed_kind(v, item)
    if kind < 0:
        raise Error("TypeError: mojolearn: " + name + " has buffer format '" + v.fmt + "', which is not a numeric array")
    return (kind, item)
