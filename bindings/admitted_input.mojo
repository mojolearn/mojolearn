# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE ONE ADMISSION PATH for X, y and sample_weight (plan
# docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.2, lane A0
# consol-a0-admission, 2026-11-08).
#
#   admit_x(mut call, source, est, context, position, admitted) -> AdmittedX
#                     X for fit, predict or score: scikit-learn's refusals
#                     (sparse, complex, rank, empty, non-finite) with their
#                     wording, then the C-order float32 words the kernels
#                     read, recorded on the call's AdmittedCall at `position`
#   admit_x_unrecorded(source, est, context, admitted) -> AdmittedX
#                     the same for a door without an admission record
#   admit_x(source, ndim, name, context, column_target, admitted) -> array
#                     the C-order float32 array of an input of rank `ndim`
#                     (the former classical_host_f32, same signature)
#   Options on every admit_x form (keywords, defaults keep the old
#   behaviour): `defer_finite` (the entry tests a borrowed X), `device_x`
#   (a DeviceRows fold stays a device handle), `order` ("C" or "F": one
#   parallel cast-and-transpose pass), `as_array` (a mojolearn Array, never
#   an ndarray), `finite=False` (the no-check mode: NaN and infinity pass,
#   for imputers and the NaN-accepting doors; the result's `finite` stays
#   False; integ-w3 folds lane a1b's `allow_nonfinite` into lane a6's option). A matrix held in bindings/admitted_registry.mojo by the
#   running operation is returned as it is (C order, not as_array).
#   admit_f32(source, ndim, name, context, ...) -> AdmittedX
#                     the array form as an AdmittedX (ndim -1: any rank up
#                     to 2): targets, fitted arrays, published predictions
#   admit_weight_words(source, rows, context) -> AdmittedX
#                     admit_weights as an AdmittedX; None is `rows` ones
#   admit_c_view(source, name, context) -> array
#                     any numeric input in its own dtype, C order (a view, or
#                     one C-order copy of a strided export)
#   admitted_dtype(view, context) -> String
#                     the Array typestr of a view's items
#   admit_ids(source, name, context, wide) -> Array
#                     a 1-D integer input as fresh int64 / int32 words
#   admit_y(...)      the float fit target (one column, finite, as long as X)
#   admit_weights(...)  sample_weight as `rows` float32 words (None stays None)
#   admitted_storage(shape, typestr, context)
#                     THE uninitialised allocator ("<f4", "<f8", "<i8") of a
#                     destination written in full before it is read
#   admitted_storage_f32(rows, cols, context)
#                     its float32 rows (cols == 0) or rows x cols form
#   admitted_array(shape, typestr, context, order) / admitted_vector(count,
#                     typestr, context)
#                     THE zero-filled mojolearn Array (any Array dtype) of a
#                     destination that is partly written or accumulated into
#                     (lane a1b: labels_empty / labels_vector / trees_empty)
#   admitted_words(array, address, rows, cols, c_order)
#                     an AdmittedX over float32 words the door made itself
#                     (a kernel's output, a joined vector: lane a1b, the
#                     former LabelsMatrix / TreesMatrix constructor)
#   admitted_finite_refuse(status, est)
#                     the refusal of a deferred finite test (defer_finite)
#   admitted_finite_entry_refuse(error, est)
#                     the same refusal when the entry raised it in its own
#                     words (one scikit-learn wording for every door)
#   admit_rows_host(value, context)
#                     the host rows of a DeviceRows fold, gathered in
#                     parallel out of the host base (no device round trip)
#
# Rules this module keeps (lesson 8: one helper per concern):
# - Metadata is read once per argument: the exporter's view (classical_view:
#   ndarray, Array, buffer, pandas, Torch, list) and its shape, strides and
#   item kind; the result is recorded in `call.inputs`.
# - A float32 C-order input is BORROWED: no copy and no host pass but the
#   finiteness scan; the caller's array is never frozen or written
#   (scikit-learn semantics). Anything else is ONE typed cast into
#   uninitialised storage, row tasks across the host's cores
#   (core/admitted_input.mojo), the finite test riding in the same pass.
# - The finiteness test of a borrowed input belongs to the kernel entry that
#   reads X anyway (lane a5-admit-core-followup; REGRESSIONS R1): a door
#   whose entry returns the finite status word passes `defer_finite=True`,
#   and admit_x then makes NO host pass over a float32 C-order X (it returns
#   `finite = False`). The GPU entry enqueues `enqueue_finite_status` on the
#   upload it makes in any case and reads the word back with its result; the
#   host entry (CPU column) calls `host_finite_status` (core/host_finite_scan.mojo)
#   (core/admitted_finite_status.mojo); the door refuses through
#   `admitted_finite_refuse`. A door that does not opt in keeps the parallel
#   host scan here (core/host_finite_scan.mojo). A cast input is tested in
#   the cast pass itself, so it is never deferred.
# - A DeviceRows fold (REGRESSIONS R13) is never downloaded: AdmittedX
#   carries its handle (`device_rows`, `rows_base`, `rows_indices`). A door
#   whose entry gathers on the device passes `device_x=True` and gets the
#   handle with no host words (address 0); any other door gets the rows
#   gathered on the host straight out of model selection's host base, in
#   row tasks across the cores with the finite test in the copy
#   (core/admitted_rows_gather.mojo). The device gather, its download and
#   the host scan after it are gone.
# - Numerics: none. The bytes reaching a kernel are the same float32 C-order
#   bytes as before; the finite and sign tests are predicates.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.admitted_call import AdmittedCall, admitted_input_dense, ADMITTED_ITEM_F32
from bindings.admitted_finite_wording import admitted_finite_wording
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays import (
    classical_wrap_array, _store_code, classical_input, classical_kind, classical_address, classical_cast_kind,
    classical_is_array, classical_is_f32, classical_c_order, classical_numpy, classical_dtype_str,
)
from bindings.native_buffer_owner import native_buffer_view
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, classical_admit_value
from bindings.classical_warnings import classical_warn_column
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.native_array_storage import native_scratch_bytes, native_array_zeros, native_array_from_values, native_array_from_bytes
from bindings.py_attr import py_attr
from bindings.admitted_registry import admitted_held_index, admitted_held_address, admitted_held_rows, admitted_held_cols
from bindings.device_rows_door import door_is_device_rows, DoorRows, DoorRowsStore, door_rows_store
from core.admitted_input import admitted_check_f32, admitted_check_weights_f32, admitted_cast_f32
from core.admitted_ids import admitted_ids_i64, admitted_ids_i32
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_NONFINITE, ARG_OUT_OF_RANGE
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from std.python._cpython import PyObjectPtr
from std.reflection import reflect
from std.sys.compile import is_defined
from bindings.py_balanced import py_setattr, py_get, py_setitem
from bindings.py_attr import py_attr
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_buffer_owner import native_storage_allocate
from bindings.native_owner_lifetime import native_release_admit, _released_dtype
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.tensor_view import TensorView, shape_text, _contiguous, base_format, _array_code
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from providers.python.array_metadata_pyapi import array_owner_once
from core.admitted_convert import admitted_convert
from core.cast_in import CAST_KIND_UNSIGNED, CAST_KIND_SIGNED, CAST_KIND_FLOAT, CAST_KIND_BOOL

comptime RESHAPE_HINT = "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."

comptime _ADM_F32P = MutPointer[Float32, MutUntrackedOrigin]


struct AdmittedX(Copyable, Movable):
    """One admitted float array of a public call, as the kernels read it.

    This is the contract every admission lane builds on (A1-A4 of the
    executor consolidation plan); a door that needs more adds a field here,
    never a private copy.

    Fields:
        array: the C-order float32 array the kernels read. It IS the
            caller's own array when `borrowed`; otherwise one fresh array
            the door owns. Keep it alive while any kernel reads `address`.
        source: the caller's object, as the door received it.
        address: the data address of `array` (its first float32 word).
        rows: the number of rows (1 for a scalar).
        cols: the number of columns (1 for a rank-0 or rank-1 input).
        ndim: the rank of `array` (0 is stored as a one-word vector).
        borrowed: True when `array` is the caller's float32 C-order array,
            read in place and never written.
        finite: True once every word was tested finite (a non-finite
            input is refused). False only when the door passed
            `defer_finite=True` and X was borrowed: its kernel entry tests
            the words it uploads and returns the status word, and the door
            refuses through `admitted_finite_refuse`.
        position: the row of this input in `call.inputs`, -1 when the
            door passed no AdmittedCall.
        device_rows: True when the caller passed a model-selection fold
            (`_arena_io.DeviceRows`); `rows_base` is then its float32
            C-order base and `rows_indices` its int64 row indices: the
            handle a device entry gathers from (model selection keeps the
            base resident). With `device_x=True` no host words exist
            (`array` is None, `address` 0, `finite` False).
        c_order: True for C-order words; False when the door asked for
            `order="F"` (then `address` holds F-order words).
        store: with `device_x=True`, the fold's resident store ids in
            model selection's store (`store.active()`): the kernel entry
            gathers the rows on the device out of them
            (core/msel_device_x.mojo msel_device_x_f32). Inactive
            otherwise (a door outside model selection, or a host column
            whose store is not on a device: then the rows are gathered on
            the host and `address` holds them).
    """
    var array: PythonObject
    var source: PythonObject
    var address: Int
    var rows: Int
    var cols: Int
    var ndim: Int
    var borrowed: Bool
    var finite: Bool
    var position: Int
    var device_rows: Bool
    var rows_base: PythonObject
    var rows_indices: PythonObject
    var c_order: Bool
    var store: DoorRowsStore

    def __init__(out self, array: PythonObject, source: PythonObject, address: Int, rows: Int, cols: Int,
                 ndim: Int, borrowed: Bool, finite: Bool = True):
        self.array = array
        self.source = source
        self.address = address
        self.rows = rows
        self.cols = cols
        self.ndim = ndim
        self.borrowed = borrowed
        self.finite = finite
        self.position = -1
        self.device_rows = False
        self.rows_base = PythonObject(None)
        self.rows_indices = PythonObject(None)
        self.c_order = True
        self.store = DoorRowsStore.none()


def admitted_words(array: PythonObject, address: Int, rows: Int, cols: Int, c_order: Bool = True) -> AdmittedX:
    """An AdmittedX over `rows` x `cols` float32 words at `address` that the
    door made itself (a fresh vector, a kernel's output, a joined block):
    `array` owns them. Finite is not claimed for anything but the words
    admit_x tested, so `finite` is False; nothing is borrowed."""
    var x = AdmittedX(array, array, address, rows, cols, 2 if cols > 1 else 1, False, False)
    x.c_order = c_order
    return x^


def _array_itemsize(typestr: String) raises -> Int:
    """The item width of an Array dtype (refuses any other typestr)."""
    if typestr == "<f8" or typestr == "<i8":
        return 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<f2" or typestr == "<u2":
        return 2
    if typestr == "<u1" or typestr == "<i1":
        return 1
    raise Error("TypeError: " + typestr + " is not an Array dtype")


def admitted_array(shape: PythonObject, typestr: String, context: PythonObject, order: String = "C") raises -> PythonObject:
    """A new ZERO-FILLED mojolearn Array of `shape` (a tuple of native ints)
    and any Array dtype, assembled from its slots; never an ndarray. THE
    zero-filled allocator of a destination that is partly written or
    accumulated into (or whose zeros are read)."""
    # Lane a1b-doors-opt-in-trees-linear: labels_empty / labels_vector (and
    # their trees_empty alias) moved here word for word: one zero store
    # (native_array_zeros), the same zero bytes as before.
    # TOMBSTONE: labels_empty / labels_vector / trees_empty (bindings/classical_labels.mojo) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
    # Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    _ = _array_itemsize(typestr)  # refuses a non-Array dtype
    return classical_wrap_array(native_array_zeros(_store_code(typestr), count), shape, typestr, order, context)


def admitted_vector(count: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    """`admitted_array((count,), typestr, context)`: a zero-filled vector."""
    return admitted_array(Python.tuple(count), typestr, context)


def _repr(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").repr, value))


def admitted_storage_f32(rows: Int, cols: Int, context: PythonObject, order: String = "C",
                         as_array: Bool = False) raises -> PythonObject:
    """A new C-order float32 array of `rows` (cols == 0) or rows x cols
    words whose contents are NOT initialised: the destination of one cast
    that writes every word before anything reads it. With NumPy it is an
    ndarray over an uninitialised bytearray (no fill pass); without NumPy it
    is a mojolearn Array (its array.array store has no uninitialised
    constructor, so that install keeps the zero-filled store)."""
    # REGRESSIONS F3 / C3: the zero fill of a destination the cast then
    # overwrote in full was a second pass over X.
    return admitted_storage(Python.tuple(rows) if cols == 0 else Python.tuple(rows, cols), "<f4", context,
                            order, as_array)


def _admitted_storage_key(typestr: String) raises -> String:
    if typestr == "<f4":
        return "float32"
    if typestr == "<f8":
        return "float64"
    if typestr == "<i8":
        return "int64"
    raise Error("TypeError: admitted storage holds <f4, <f8 or <i8, not " + typestr)


def admitted_storage(shape: PythonObject, typestr: String, context: PythonObject, order: String = "C",
                     as_array: Bool = False) raises -> PythonObject:
    """THE uninitialised allocator (lane a5-admit-core-followup): a new
    array in `order` ("C" or "F") of `shape` (a tuple of native ints) holding "<f4", "<f8"
    or "<i8" words whose contents are NOT initialised. Only for a
    destination one native call writes in FULL before anything reads it or
    it escapes (a cast, a gather, a kernel's download); a destination that
    is partly written, or accumulated into, keeps a zero-filled allocator.
    With NumPy it is an ndarray over an uninitialised bytearray
    (`native_scratch_bytes`: no fill pass). Without NumPy it is a mojolearn
    Array over the zero-filled array.array store (array.array has no
    uninitialised constructor); `as_array` asks for that Array whatever is
    installed (a door whose result must never be an ndarray)."""
    # The zero-fill-then-overwrite sites that move here are listed by owning
    # lane in ~/mojolearn-evidence/a5-zero-fill-sites.md.
    if order != "C" and order != "F":
        raise Error("ValueError: admitted storage order must be 'C' or 'F', not " + order)
    if as_array or not classical_numpy(context):
        return admitted_array(shape, typestr, context, order)
    var key = _admitted_storage_key(typestr)
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var width = 4 if typestr == "<f4" else 8
    var storage = native_scratch_bytes(count * width)
    return py_call(context["ndarray"], shape, dtype=context[PythonObject(key)], buffer=storage, order=PythonObject(order))


def admit_x(source: PythonObject, ndim: Int, name: String, context: PythonObject, column_target: Bool = False,
            admitted: Bool = False, defer_finite: Bool = False, order: String = "C",
            as_array: Bool = False, finite: Bool = True) raises -> PythonObject:
    """The float32 array of an input of rank `ndim` in `order` ("C", or "F"
    for a 2-D input): the input itself when it already is one (borrowed),
    otherwise one cast copy into uninitialised storage. `as_array`: the
    result is a mojolearn Array, never an ndarray (an ndarray input is then
    copied). Refuses a rank mismatch, an empty input, a non-native byte
    order and NaN or infinity ("Input <name> contains NaN or infinity").
    `finite=False`: the no-check mode of a door that accepts NaN (the words
    are cast or borrowed, never tested)."""
    # Order "F" is the cast with the view's axes swapped: the C-order
    # cols x rows destination of the transposed view IS the F-order
    # rows x cols array, so cast and transpose are one parallel pass
    # (core/admitted_input.mojo admitted_cast_f32, row tasks over the
    # columns; no staging store, no second pass). The words are the cast's
    # words at their F-order cells: no bit changes.
    # `defer_finite`: a borrowed input is not scanned here; the caller's
    # kernel entry tests the words it uploads (core/admitted_finite_status.mojo)
    # and the caller refuses through admitted_finite_refuse. A cast input is
    # still tested here: the test rides in the cast pass, at no extra pass.
    # `admitted`: `source` is an array view classical_view returned in the
    # same native call (an ndarray, or an Array without NumPy). It was
    # admitted and its pandas/Torch storage read there, so the admission walk
    # and the two storage tests are not repeated for the same value.
    # A matrix the running meta-estimator already admitted (held in
    # bindings/admitted_registry.mojo) is the array its members read as it
    # is: no view, no check, no scan, in this form too (lane a1b: the
    # array-form doors, classical_host_f32's callers among them, skipped
    # the registry and re-checked every member's X).
    if ndim == 2 and order == "C" and not as_array and admitted_held_index(source) >= 0:
        return source
    var b = native_loaded_module("builtins")
    var prepared = source
    if not admitted:
        classical_admit_value(source, context)
        if door_is_device_rows(source, context):
            # A model-selection fold (lane a6): its rows gathered on the host
            # out of the host base in row tasks (no device round trip); the
            # borrowed float32 C-order rows are then tested below as any X.
            prepared = admit_rows_host_unchecked(source, context)
        elif native_pandas_exact(source):
            # integ-w3: the no-check mode reaches the pandas cast too (a
            # NaN-accepting door given a DataFrame was refused there).
            prepared = native_pandas_f32(source, context, allow_nonfinite=not finite)
        elif _ordinary_torch(py_type(source), context):
            prepared = native_torch_numpy(source)
    var value = classical_input(prepared, context, name)
    var nd = Int(py=value.ndim)
    var column = column_target and ndim == 1 and nd == 2 and Int(py=value.shape[1]) == 1 and py_type(source) is not b.list and py_type(source) is not b.tuple and py_type(source) is not b.memoryview
    if column:
        classical_warn_column(context)
    if ndim >= 0 and nd != ndim and not column:
        raise Error(name + " must be " + String(ndim) + "-D")
    if nd > 2:
        raise Error(name + " must be at most 2-D")
    # ndim -1 (lane a1b): any rank up to 2, kept as it is (a fitted array).
    var rank = nd if ndim < 0 else ndim
    var rows = 1 if nd == 0 else Int(py=value.shape[0])
    var cols = 1 if nd < 2 else Int(py=value.shape[1])
    if rows < 1 or cols < 1:
        raise Error(name + " is empty")
    # An Array is always in native (little-endian) byte order.
    if not classical_is_array(value, context) and not Bool(py=value.dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, name, context)
    var s0 = 0 if nd == 0 else Int(py=value.strides[0])
    var s1 = 0 if nd < 2 else Int(py=value.strides[1])
    var addr = Int(py=classical_address(value))
    if order != "C" and order != "F":
        raise Error("ValueError: order must be 'C' or 'F', not " + order)
    var want_f = order == "F" and nd == 2 and not column
    var in_order: Bool
    if want_f:
        # F-contiguous float32 words: unit row stride, column stride of a
        # whole column (a single row or column is both orders).
        in_order = (s0 == 4 or rows == 1) and (s1 == rows * 4 or cols == 1)
    else:
        in_order = classical_c_order(value, context)
    var borrowed = (not column and classical_is_f32(value, context) and in_order
                    and (not as_array or classical_is_array(value, context)))
    var result = value if borrowed else admitted_storage_f32(rows, cols if rank == 2 else 0, context,
                                                             "F" if want_f else "C", as_array)
    var out = _ADM_F32P(unsafe_from_address=Int(py=classical_address(result)))
    if not finite:
        # The no-check mode (lane a6): a NaN-accepting door. A borrowed X
        # is read as is; a cast X is cast and its verdict ignored.
        if not borrowed:
            if want_f:
                _ = admitted_cast_f32(addr, s1, s0, cols, rows, item, kind, out)
            else:
                _ = admitted_cast_f32(addr, s0, s1, rows, cols, item, kind, out)
    elif not (borrowed and defer_finite):
        if want_f:
            admitted_check_f32(addr, s1, s0, cols, rows, item, kind, name, borrowed, out)
        else:
            admitted_check_f32(addr, s0, s1, rows, cols, item, kind, name, borrowed, out)
    _ = value^
    _ = prepared^
    return result


def admit_f32(source: PythonObject, ndim: Int, name: String, context: PythonObject, order: String = "C",
              defer_finite: Bool = False, as_array: Bool = False, finite: Bool = True,
              admitted: Bool = False) raises -> AdmittedX:
    """`admit_x(source, ndim, name, ...)` as an AdmittedX: the float32 words
    of an input of rank `ndim` (-1: any rank up to 2) in `order`, with its
    rows and columns (a rank-0 or rank-1 input has one column). For the
    inputs that are not a public X (a target, a fitted array, a published
    prediction); a public X goes through admit_x_unrecorded / admit_x."""
    # Lane a1b-doors-opt-in-trees-linear: the former labels_f32 /
    # trees_f32 (bindings/classical_labels.mojo, a second cast path with its
    # own view struct) is this call.
    var array = admit_x(source, ndim, name, context, False, admitted, defer_finite, order, as_array,
                        finite)
    var nd = Int(py=array.ndim)
    var rows = 1 if nd == 0 else Int(py=array.shape[0])
    var cols = 1 if nd < 2 else Int(py=array.shape[1])
    var x = AdmittedX(array, source, Int(py=classical_address(array)), rows, cols, nd, array is source,
                      finite and not defer_finite)
    x.c_order = order == "C" or rows == 1 or cols == 1
    return x^


def admit_weight_words(source: PythonObject, rows: Int, context: PythonObject,
                       admitted: Bool = False) raises -> AdmittedX:
    """`admit_weights(source, rows)` as an AdmittedX of `rows` words; None
    is `rows` ones (the unweighted fit's weights, as a fresh vector)."""
    # Lane a1b: the former labels_check_sample_weight / trees_check_sample_weight.
    var w: PythonObject
    if source is None:
        # One float32 1.0 repeated in C (`array.array.__mul__`), the store
        # labels_full_f32 made.
        var one = native_array_from_values(String("f"), Python.list(PythonObject(1.0)))
        w = classical_wrap_array(py_call(py_attr(one, "__mul__"), PythonObject(rows)), Python.tuple(rows), "<f4",
                                 "C", context)
    else:
        w = admit_weights(source, rows, context, admitted)
    var x = AdmittedX(w, source, Int(py=classical_address(w)), rows, 1, 1, w is source, True)
    return x^


def admitted_dtype(view: PythonObject, context: PythonObject) raises -> String:
    """The Array typestr of an admitted view's items ("<f4", "<i8", "<u1",
    ...): a one-byte "|" type and a native "=" one read as "<", a bool as
    its "<u1" bytes (the reading of `_buffer_typestr`). Refuses a
    non-native byte order."""
    var typestr = classical_dtype_str(view, context)
    if typestr.startswith("|") or typestr.startswith("="):
        typestr = "<" + String(typestr[byte=1:typestr.byte_length()])
    if typestr == "<b1":
        typestr = "<u1"
    if not typestr.startswith("<"):
        raise Error("TypeError: mojolearn: an input of dtype " + typestr + " must be in native byte order")
    return typestr


def admit_c_view(source: PythonObject, name: String, context: PythonObject, admitted: Bool = False) raises -> PythonObject:
    """The exporter's view of an input of any numeric dtype, in C order: the
    view itself (classical_view: an ndarray, or an Array without NumPy) when
    its words are already C-contiguous, else one C-order copy of its bytes
    (`memoryview.tobytes('C')`, in C) as an Array of its own dtype. For the
    doors that read a label, prior or flag vector in its own dtype; X goes
    through admit_x."""
    # Lane a1b-doors-opt-in-trees-linear: the former labels_view /
    # trees_view (a LabelsView struct over the same exporters, which
    # linearized a strided export the same way).
    var view = classical_view(source, context, name, admitted)
    if classical_c_order(view, context):
        return view
    if classical_is_array(view, context):
        # An F-ordered 2-D Array (an install without NumPy): its store is in
        # storage order, so its bytes are not the C-order words.
        raise Error("TypeError: mojolearn: " + name + " must be a C-ordered Array")
    var typestr = admitted_dtype(view, context)
    var exported = native_buffer_view(view)
    var store = native_array_from_bytes(String(_store_code(typestr)), py_call(exported.tobytes, "C"))
    var copy = classical_wrap_array(store, view.shape, typestr, "C", context)
    _ = view^
    return copy


def admit_ids(source: PythonObject, name: String, context: PythonObject, wide: Bool = True,
              admitted: Bool = False) raises -> PythonObject:
    """A 1-D integer (or bool) input as a fresh C-order int64 (`wide`) or
    int32 mojolearn Array: one parallel read of the exporter's own strides
    (core/admitted_ids.mojo). Refuses another rank, a float or text input,
    and an id that does not fit."""
    # Lane a1b-doors-opt-in-trees-linear: index vectors, fold ids and LU
    # pivots were read through labels_view + labels_cast (a second view
    # struct, a strided export linearized first).
    var view = classical_view(source, context, name, admitted)
    if Int(py=view.ndim) != 1:
        raise Error("mojolearn: " + name + " must be 1-D, got " + String(Int(py=view.ndim)) + "-D")
    var kind = classical_kind(view, context)
    if kind != "i" and kind != "u" and kind != "b":
        raise Error("TypeError: mojolearn: " + name + " must hold integers")
    var count = Int(py=view.shape[0])
    var out = admitted_array(Python.tuple(count), "<i8" if wide else "<i4", context)
    if count > 0:
        var addr = Int(py=classical_address(view))
        var step = Int(py=view.strides[0])
        var item = Int(py=view.itemsize)
        var dst = Int(py=classical_address(out))
        var fits: Bool
        if wide:
            fits = admitted_ids_i64(addr, step, 0, count, 1, item, kind == "i", False,
                                    MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst))
        else:
            fits = admitted_ids_i32(addr, step, 0, count, 1, item, kind == "i", False,
                                    MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst))
        if not fits:
            raise Error("ValueError: mojolearn: " + name + " holds an id outside int" + ("64" if wide else "32"))
    _ = view^
    return out


def classical_view(source: PythonObject, context: PythonObject, name: String = "input",
                   admitted: Bool = False) raises -> PythonObject:
    """An array view of an admitted input (pandas and Torch storage read
    natively), before any cast: an ndarray with NumPy, an Array without."""
    # `admitted`: the caller's native entry already ran classical_admit_value
    # over this very argument (classical_c admits every operation argument
    # before dispatch), so the walk is not repeated; storage is still read.
    if not admitted:
        classical_admit_value(source, context)
    if door_is_device_rows(source, context):
        # A model-selection fold reaching a door that views X directly
        # (lane a6): its rows gathered on the host straight out of the host
        # base (row tasks across the cores, no device round trip), not
        # finite-tested here: the door's own test runs on the view, as for
        # any array. This is what lets every family and classical estimator
        # opt into DeviceRows folds (no fold is downloaded).
        return admit_rows_host_unchecked(source, context)
    return _classical_view_storage(source, context, name)


def _classical_view_storage(source: PythonObject, context: PythonObject, name: String) raises -> PythonObject:
    """`classical_view`'s storage read (pandas and Torch natively), with no
    DeviceRows branch. `_admitted_rows` views a fold's base and indices
    through this, so classical_view -> admit_rows_host_unchecked ->
    _admitted_rows never calls back into classical_view: that mutual
    recursion deadlocked the Mojo compiler (integ-w3, 2026-11-09)."""
    var prepared = source
    if native_pandas_exact(source):
        prepared = native_pandas_f32(source, context)
    elif _ordinary_torch(py_type(source), context):
        prepared = native_torch_numpy(source)
    return classical_input(prepared, context, name)


def classical_refuse_sparse(source: PythonObject, est: String) raises:
    # scipy.sparse containers are recognized by their defining module, so no
    # attribute or property of the container runs.
    # p3-labels-metrics: this runs twice per classical call (the door and
    # the X check); an exact list or tuple is never sparse, and the module
    # name is tested as a native string (no Python startswith call).
    var b = native_loaded_module("builtins")
    var cls = py_type(source)
    if cls is b.list or cls is b.tuple:
        return
    var module = cls.__module__
    if py_type(module) is b.str and String(py=module).startswith("scipy.sparse"):
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + est + " takes dense X only.)")


def _numeric(view: PythonObject, context: PythonObject) raises -> Bool:
    var kind = classical_kind(view, context)
    return kind == "b" or kind == "i" or kind == "u" or kind == "f"


def _cast_x(view: PythonObject, est: String, context: PythonObject, defer_finite: Bool, order: String,
           as_array: Bool, finite: Bool = True) raises -> PythonObject:
    var nd = Int(py=view.ndim)
    if nd != 2:
        raise Error("mojolearn: X must be 2-D, got " + String(nd) + "-D shape " + _repr(view.shape))
    if not _numeric(view, context):
        raise Error("TypeError: mojolearn: X has buffer format " + _repr(py_call(native_loaded_module("builtins").memoryview, view).format) + ", which is not a numeric array")
    try:
        return admit_x(view, 2, "X", context, admitted=True, defer_finite=defer_finite, order=order,
                       as_array=as_array, finite=finite)
    except error:
        if String(error) == "Input X contains NaN or infinity":
            raise Error("Input X contains NaN or infinity (" + est + ").")
        raise Error(String(error))


def _check_x_view(view: PythonObject, est: String, context: PythonObject, defer_finite: Bool = False,
                  order: String = "C", as_array: Bool = False, finite: Bool = True) raises -> PythonObject:
    # The one place for scikit-learn's X refusal text.
    var kind = classical_kind(view, context)
    if kind == "c":
        raise Error("Complex data not supported: " + est + " got a complex X.")
    if _numeric(view, context):
        var nd = Int(py=view.ndim)
        if nd < 2:
            raise Error("Expected 2D array, got " + ("scalar" if nd == 0 else "1D") + " array instead (shape=" + _repr(view.shape) + "). " + RESHAPE_HINT)
        if nd == 2 and Int(py=view.shape[0]) == 0:
            raise Error("Found array with 0 sample(s) (shape=" + _repr(view.shape) + ") while a minimum of 1 is required by " + est + ".")
        if nd == 2 and Int(py=view.shape[1]) == 0:
            raise Error("Found array with 0 feature(s) (shape=" + _repr(view.shape) + ") while a minimum of 1 is required by " + est + ".")
    return _cast_x(view, est, context, defer_finite, order, as_array, finite)


def _admitted_rows(source: PythonObject, est: String, context: PythonObject, device_x: Bool,
                   finite: Bool = True) raises -> AdmittedX:
    """A model-selection fold (`_arena_io.DeviceRows`) admitted without its
    device gather ever being downloaded (REGRESSIONS R13; plan section 3.2
    step 4). The handle (base, indices) always rides on the result. With
    `device_x` the door's entry gathers on the device out of the resident
    base: no host word is read or written here. Otherwise the rows are
    gathered on the host straight out of the host base, in row tasks across
    the cores, finite-tested in the copy (core/admitted_rows_gather.mojo).
    The same rows in the same order as model selection's device gather, so
    the same bytes reach the kernel. `finite=False` (a NaN-accepting door):
    a NaN fold is not refused (lane a6). With `device_x` the fold's store
    ids ride on `store` (lane a6); when they cannot be resolved (no store
    connection) the rows are gathered on the host as without `device_x`."""
    # TOMBSTONE: door_rows_host (bindings/device_rows_door.mojo, the DeviceRows
    # materialize: device gather, download, host scan, kernel re-upload) deleted
    # 2026-11-08 by lane a5-admit-core-followup; code recoverable at f7d10d2b09.
    # Restore: experiments/removed/A5_DOOR_ROWS_HOST.patch.
    var door = DoorRows(source, context)
    var base = _classical_view_storage(door.base, context, "X")
    if Int(py=base.ndim) != 2 or not classical_is_f32(base, context) or not classical_c_order(base, context):
        raise Error("TypeError: DeviceRows requires a float32 C-order 2-D base")
    var indices = _classical_view_storage(door.indices, context, "indices")
    if (Int(py=indices.ndim) != 1 or classical_kind(indices, context) != "i" or Int(py=indices.itemsize) != 8
            or not classical_c_order(indices, context)):
        raise Error("TypeError: DeviceRows requires C-contiguous int64 row indices")
    var base_rows = Int(py=base.shape[0])
    var columns = Int(py=base.shape[1])
    var rows = door.rows
    if rows == 0:
        raise Error("Found array with 0 sample(s) (shape=(0, " + String(columns) + ")) while a minimum of 1 is required by " + est + ".")
    if columns == 0:
        raise Error("Found array with 0 feature(s) (shape=(" + String(rows) + ", 0)) while a minimum of 1 is required by " + est + ".")
    var array = PythonObject(None)
    var address = 0
    var tested = False
    var store = DoorRowsStore.none()
    if device_x:
        store = door_rows_store(source, context)
        if store.active() and (store.rows != rows or store.row_words != columns or store.base_rows != base_rows):
            raise Error("TypeError: DeviceRows store entries differ from the fold's base and indices")
    if not store.active():
        array = admitted_storage_f32(rows, columns, context)
        address = Int(py=classical_address(array))
        var status = admitted_gather_rows_f32(Int(py=classical_address(base)), base_rows, columns,
                                              Int(py=classical_address(indices)), rows, address)
        if status == ARG_OUT_OF_RANGE:
            raise Error("IndexError: DeviceRows row index out of range for a base of " + String(base_rows) + " rows")
        if status == ARG_NONFINITE and finite:
            raise Error("Input X contains NaN or infinity (" + est + ").")
        tested = finite
    var x = AdmittedX(array, source, address, rows, columns, 2, False, tested)
    x.store = store
    x.device_rows = True
    x.rows_base = door.base
    x.rows_indices = door.indices
    _ = base^
    _ = indices^
    return x^


def _admitted_x(source: PythonObject, est: String, context: PythonObject, admitted: Bool,
                defer_finite: Bool = False, device_x: Bool = False, order: String = "C",
                as_array: Bool = False, finite: Bool = True) raises -> AdmittedX:
    """X refused with scikit-learn's wording and cast (or borrowed) as
    float32 in `order`; no admission record."""
    if order == "C" and not as_array:
        # A matrix the running operation already admitted (a meta-estimator
        # holds it for its members: bindings/admitted_registry.mojo) is
        # returned as it is: no view, no check, no scan.
        var held = admitted_held_index(source)
        if held >= 0:
            return AdmittedX(source, source, admitted_held_address(held), admitted_held_rows(held),
                             admitted_held_cols(held), 2, True)
    if door_is_device_rows(source, context):
        if order != "C" or as_array:
            # The gather writes C-order fresh storage; an F-order or Array
            # request re-admits those rows through the cast below.
            var rows_x = _admitted_rows(source, est, context, False, finite)
            var ordered = _admitted_x(rows_x.array, est, context, True, defer_finite, False, order, as_array,
                                      finite)
            ordered.source = source
            ordered.device_rows = True
            ordered.rows_base = rows_x.rows_base
            ordered.rows_indices = rows_x.rows_indices
            return ordered^
        return _admitted_rows(source, est, context, device_x, finite)
    classical_refuse_sparse(source, est)
    var view = classical_view(source, context, "X", admitted)
    var x = _check_x_view(view, est, context, defer_finite, order, as_array, finite)
    # _cast_x refused every rank but 2; the cast keeps the view's shape.
    var rows = Int(py=view.shape[0])
    var columns = Int(py=view.shape[1])
    # Borrowed exactly when the kernels read the view's own words.
    var address = Int(py=classical_address(x))
    var borrowed = address == Int(py=classical_address(view))
    _ = view^
    # A borrowed X under defer_finite was not scanned: its entry tests it.
    var result = AdmittedX(x, source, address, rows, columns, 2, borrowed,
                           finite and not (borrowed and defer_finite))
    result.c_order = order == "C" or rows == 1 or columns == 1
    return result^


def admit_x_unrecorded(source: PythonObject, est: String, context: PythonObject,
                       admitted: Bool = False, defer_finite: Bool = False,
                       device_x: Bool = False, order: String = "C",
                       as_array: Bool = False, finite: Bool = True) raises -> AdmittedX:
    """X for fit, predict or score: refused with scikit-learn's wording (a
    sparse, complex, empty or non-2-D X; NaN or infinity), then the C-order
    float32 words the kernels read (the input itself when it already is
    one). For a door without an admission record.

    `defer_finite`: the door's kernel entry tests the words of a borrowed X
    and returns the status word (`finite` is then False; refuse through
    `admitted_finite_refuse`). `device_x`: the door's entry reads a
    DeviceRows fold from its device handle (`device_rows`), so no host rows
    are made. `order`: "C" (default) or "F" words. `as_array`: the array
    is a mojolearn Array, never an ndarray. `finite=False`: the no-check
    mode of a door that accepts NaN (a NaN fold or X is not refused;
    `finite` on the result is then False)."""
    return _admitted_x(source, est, context, admitted, defer_finite, device_x, order, as_array, finite)


def admit_x(mut call: AdmittedCall, source: PythonObject, est: String, context: PythonObject,
            position: Int, admitted: Bool = False, defer_finite: Bool = False,
            device_x: Bool = False, order: String = "C", as_array: Bool = False,
            finite: Bool = True) raises -> AdmittedX:
    """As above, recording the admitted X on the door's admission record at
    `position` (its rows, columns, address and the array that owns it), so
    the door reads the shape from the record instead of admitting X again
    (labels_shape_of; rows N254, N279). A DeviceRows fold under `device_x`
    is recorded with address 0 and no owning array; so is an F-order X
    (the record's readers take C-order words)."""
    var x = _admitted_x(source, est, context, admitted, defer_finite, device_x, order, as_array, finite)
    x.position = call.add_input(admitted_input_dense(position, source, x.array, x.address if x.c_order else 0,
                                                     x.rows, x.cols, ADMITTED_ITEM_F32, False))
    return x^


def admitted_finite_refuse(status: Int, est: String, name: String = "X") raises:
    """The refusal of a deferred finite test: `status` is the word the
    kernel entry returned (core/admitted_finite_status.mojo; 0 = every word
    finite). Raises scikit-learn's wording, as admit_x would have."""
    if status != 0:
        raise Error(admitted_finite_wording(est, name))


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


def admit_rows_host(value: PythonObject, est: String, context: PythonObject) raises -> PythonObject:
    """The host rows of a DeviceRows fold as one fresh C-order float32
    array, gathered in parallel out of model selection's host base (no
    device gather, no download, the finite test in the copy). For a door
    that still hands its owners an array object instead of an AdmittedX."""
    return _admitted_rows(value, est, context, False).array


def admit_rows_host_unchecked(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`admit_rows_host` with no finite refusal: the view of a fold for a
    door that tests (or accepts) NaN itself (classical_view)."""
    return _admitted_rows(value, "", context, False, False).array


def admit_y(source: PythonObject, rows: Int, est: String, context: PythonObject,
            admitted: Bool = False) raises -> PythonObject:
    """The float fit target: present, real, one column (a column vector warns
    and is raveled), finite and as long as X; the C-order float32 vector the
    kernels read."""
    if source is None:
        raise Error(est + " requires y to be passed, but the target y is None.")
    var b = native_loaded_module("builtins")
    var view = classical_view(source, context, "y", admitted)
    if classical_kind(view, context) == "c":
        raise Error("Complex data not supported: " + est + " got a complex y.")
    var nd = Int(py=view.ndim)
    var column = nd == 2 and Int(py=view.shape[1]) == 1 and py_type(source) is not b.list and py_type(source) is not b.tuple and py_type(source) is not b.memoryview
    var y = PythonObject(None)
    if nd == 1 or column:
        try:
            y = admit_x(view, 1, "y", context, column_target=column, admitted=True)
        except error:
            if String(error) == "Input y contains NaN or infinity":
                raise Error("Input y contains NaN or infinity (" + est + ").")
            raise Error(String(error))
    var length = 1 if nd == 0 else Int(py=view.shape[0])
    if length != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(length) + "]")
    if y is None:
        raise Error("mojolearn " + est + " currently requires one target")
    _ = view^
    return y


def classical_weight_refusal(source: PythonObject, error: Error) raises -> Error:
    """The refusal of a sample_weight the shared admission refused. That
    refusal names no accepted type; say what a weight may be. An object that
    only defines __array__ (scikit-learn's _NotAnArray) is never converted:
    no Python protocol runs in a fit."""
    if String(error).startswith("TypeError: Sequential requires native input types"):
        return Error("TypeError: sample_weight must be None, a number, a NumPy array, a buffer "
                     + "(memoryview, array.array, bytearray), a list or tuple of numbers, a pandas "
                     + "Series or a Torch tensor; got " + String(py_type(source).__name__)
                     + ", which is not read because it only converts through Python-defined "
                     + "methods such as __array__")
    return Error(String(error))


def admit_weights(source: PythonObject, rows: Int, context: PythonObject,
                  admitted: Bool = False) raises -> PythonObject:
    """sample_weight as `rows` float32 words: None stays None, a scalar fills
    every row; scikit-learn's shape, finiteness, sign and all-zero refusals."""
    if source is None:
        return PythonObject(None)
    var view = PythonObject(None)
    try:
        view = classical_view(source, context, "sample_weight", admitted)
    except error:
        var refusal = classical_weight_refusal(source, error)
        raise refusal
    var nd = Int(py=view.ndim)
    if nd > 1:
        raise Error("Sample weights must be 1D array or scalar")
    var scalar = nd == 0
    if not scalar and Int(py=view.shape[0]) != rows:
        raise Error("sample_weight.shape == (" + String(Int(py=view.shape[0])) + ",), expected (" + String(rows) + ",)!")
    var value = PythonObject(None)
    try:
        value = admit_x(view, 0 if scalar else 1, "sample_weight", context, admitted=True)
    except error:
        if String(error) == "Input sample_weight contains NaN or infinity":
            raise Error("Input sample_weight contains NaN or infinity.")
        raise Error(String(error))
    # A scalar weight fills every one of the `rows` words of a fresh vector,
    # so the vector needs no zero fill first.
    var out = admitted_storage_f32(rows, 0, context) if scalar else value
    var input = _ADM_F32P(unsafe_from_address=Int(py=classical_address(value)))
    var output = _ADM_F32P(unsafe_from_address=Int(py=classical_address(out)))
    var any_positive = admitted_check_weights_f32(input, output, rows, scalar)
    if not any_positive:
        raise Error("Sample weights must contain at least one non-zero number.")
    _ = value^
    _ = view^
    return out


# ---------------------------------------------------------------------------
# THE MULTI-DTYPE ADMISSION (lane a6-device-x-family-classical, 2026-11-09).
#
# The float32 path above is `admit_x`; this is the same path for an input the
# kernel reads in another Array dtype ("<f8", "<i8", "<i4", "<u2", "<i1",
# ... any Array typestr) or in the dtype it already has (metrics, linalg ops,
# low-bit weights, resampling):
#
#   AdmittedTyped(value, context, admitted)
#                     metadata once: a mojolearn Array read off its
#                     import-pinned slots (its storage owners retained), any
#                     other native exporter pinned through one
#                     PyObject_GetBuffer (AdmittedBorrowedBuffer) for the call
#   admitted_typed_c(var input, typestr, name, context, force_copy) -> AdmittedBuf
#                     borrow when the dtype and C order already match
#                     (zero copy; a foreign exporter becomes a pinned Array
#                     over the same memory), else ONE conversion into a new
#                     C-order store, split across the host's cores
#                     (core/admitted_convert.mojo: typed loads, Array.astype's
#                     rules, the first failing element's refusal)
#   admitted_typed_output(var input) -> AdmittedBuf
#                     a caller-supplied output, pinned for the call
#   admit_typed(source, typestr, ndim, name, context, admitted) -> AdmittedBuf
#                     the three above for one argument with the rank and
#                     empty refusals
#   admitted_snapshot(value, memo, context)
#                     an operation-private snapshot of a metric argument
#                     (containers walked, aliases kept, dtypes kept)
#
# These were LinAdmission / lin_admitted_c / lin_borrow_array / lin_convert /
# lin_typed / lin_admitted_output / lin_admit_exporter / LinBuf / LinBufferPin
# (bindings/linalg_ops_buffers.mojo) and metric_input_snapshot
# (providers/python/metric_input_snapshot_pyapi.mojo), moved here word for
# word: the same bytes reach every kernel (no bit change).


struct AdmittedBuf(Movable):
    """A C-ordered buffer the kernels read: the object that owns it (kept
    alive across the call), its address and its shape (and, for a borrowed
    exporter, the pin and storage owner that keep the address valid)."""
    var keep: PythonObject
    var addr: Int
    var shape: List[Int]
    var pin: Optional[AdmittedBorrowedBuffer]
    var owner: PythonObject

    def __init__(out self, keep: PythonObject, addr: Int, var shape: List[Int]):
        # Python's None is a runtime object. Constructing it in a default
        # argument asks the compiler to initialize CPython at compile time.
        self.keep = keep
        self.addr = addr
        self.shape = shape^
        self.pin = None
        self.owner = PythonObject(None)

    def __init__(out self, keep: PythonObject, addr: Int, var shape: List[Int],
                 var pin: Optional[AdmittedBorrowedBuffer], owner: PythonObject):
        self.keep = keep
        self.addr = addr
        self.shape = shape^
        self.pin = pin^
        self.owner = owner

    def dim(self, k: Int) -> Int:
        return self.shape[k]

    def rank(self) -> Int:
        return len(self.shape)

    def count(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):  # driver-loop: over shape words
            n *= self.shape[i]
        return n

    def text(self) -> String:
        return shape_text(self.shape)


def admitted_shape_tuple(shape: List[Int]) raises -> PythonObject:
    """A shape as a Python tuple."""
    var dims = Python.list()
    for i in range(len(shape)):  # driver-loop: over shape words
        _ = py_call(dims.append, PythonObject(shape[i]))
    return py_call(native_loaded_module("builtins").tuple, dims)


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


def admitted_zeros(shape: List[Int], typestr: String, context: PythonObject) raises -> AdmittedBuf:
    """A new zero-filled C-order Array of any Array dtype. Array stores have
    no uninitialised constructor (admitted_storage keeps the same zeros
    without NumPy), so the one fill pass stays."""
    var count = 1
    for i in range(len(shape)):  # driver-loop: over shape words
        count *= shape[i]
    _ = admitted_width(typestr)  # its refusal for a dtype no Array holds
    var out = classical_wrap_array(native_array_zeros(_store_code(typestr), count), admitted_shape_tuple(shape),
                                   typestr, "C", context)
    var addr = Int(py=classical_address(out))
    return AdmittedBuf(out, addr, shape.copy())


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


def admitted_typed_exporter(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    # Preserve the native mmap/array owners already connected by the classical
    # boundary. The context-free provider helper intentionally knows fewer
    # import-time owners; no Python protocol is used to broaden that set here.
    if depth > 64:
        raise Error("TypeError: native linalg buffer owner nesting exceeds its bound")
    var cls = py_type(value)
    var b = native_loaded_module("builtins")
    if cls is b.memoryview:
        var owner = value.obj
        if owner is not None:
            admitted_typed_exporter(owner, context, depth + 1)
        return
    var owners = context["buffer_owner_types"]
    for i in range(len(owners)):  # small-loop(import-connected native storage classes)
        if cls is owners[i]:
            return
    admit_native_buffer_exporter(value)


struct AdmittedTyped(Movable):
    # One operation-local metadata capture. Exact native exporter admission
    # precedes PyObject_GetBuffer; the pin lives through the numerical call.
    var keep: PythonObject
    var view: TensorView
    var pin: Optional[AdmittedBorrowedBuffer]
    var owner: PythonObject

    def __init__(out self, value: PythonObject, context: PythonObject, admitted: Bool = False) raises:
        # `admitted`: the caller ran classical_admit_value over this very
        # value in the same call (admit_typed, row N237).
        if not admitted:
            classical_admit_value(value, context)
        self.keep = value
        var view = TensorView()
        self.pin = None
        self.owner = PythonObject(None)
        if classical_is_array(value, context):
            var slots = context["array_fields"]
            var cls = context["array_class"]
            # Keep the admitted storage owners themselves, so replacement of
            # the input Array's slots while native code releases the GIL cannot
            # free the captured address. Each slot is read once through its
            # import-pinned native descriptor rather than a mutable class hook.
            self.owner = array_owner_once(value, context)
            var address = py_call(slots["_addr"], value, cls)
            var item = py_call(slots["itemsize"], value, cls)
            var readonly = py_call(slots["_readonly"], value, cls)
            var dtype = py_call(slots["dtype"], value, cls)
            var shape = py_call(slots["shape"], value, cls)
            var strides = py_call(slots["strides"], value, cls)
            if (py_type(address) is not py_type(PythonObject(0)) or py_type(item) is not py_type(PythonObject(0))
                or py_type(readonly) is not py_type(PythonObject(True)) or py_type(dtype) is not py_type(PythonObject(""))
                or py_type(shape) is not py_type(Python.tuple()) or py_type(strides) is not py_type(Python.tuple())):
                raise Error("TypeError: Array metadata must contain native integer, boolean, string and tuple values")
            if len(shape) != len(strides):
                raise Error("TypeError: Array shape and stride ranks differ")
            view.is_array = True
            view.addr = Int(py=address)
            view.itemsize = Int(py=item)
            view.readonly = Bool(py=readonly)
            view.dtype = String(py=dtype)
            view.fmt = _array_code(view.dtype)
            for axis in range(len(shape)):  # small-loop(Array rank metadata admitted once)
                if py_type(shape[axis]) is not py_type(PythonObject(0)) or py_type(strides[axis]) is not py_type(PythonObject(0)):
                    raise Error("TypeError: Array dimensions and strides must be native integers")
                view.shape.append(Int(py=shape[axis]))
                view.strides.append(Int(py=strides[axis]))
            view.nbytes = view.count() * view.itemsize
            view.has_strides = True
            view.c_contig = _contiguous(view.shape, view.strides, True, view.itemsize)
            self.view = view^
            return
        admitted_typed_exporter(value, context)
        self.pin = AdmittedBorrowedBuffer(value)
        ref layout = self.pin.value().layout.value()
        view.addr = layout.address
        view.itemsize = layout.itemsize
        view.readonly = layout.readonly
        view.nbytes = layout.count * layout.itemsize
        view.shape = layout.shape.copy()
        view.strides = layout.strides.copy()
        view.has_strides = True
        view.fmt = self.pin.value().format
        if view.fmt.startswith(">") or view.fmt.startswith("!"):
            raise Error("TypeError: mojolearn: big-endian buffers are not supported")
        view.c_contig = _contiguous(view.shape, view.strides, True, layout.itemsize)
        for axis in range(len(layout.suboffsets)):  # small-loop(buffer rank)
            if layout.suboffsets[axis] >= 0:
                raise Error("TypeError: mojolearn: indirect buffers are not supported")

        self.view = view^


# One Python type per numeric mode: a process may load both builds of a binding.
comptime ADMITTED_PIN_MODE: Int = 1 if is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]() else 0


struct AdmittedBufferPin[MODE: Int = ADMITTED_PIN_MODE](Movable, Writable):
    var pin: Optional[AdmittedBorrowedBuffer]

    def __init__(out self, var pin: Optional[AdmittedBorrowedBuffer]):
        self.pin = pin^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("AdmittedBufferPin")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def register_admitted_buffer_pin(mut module: PythonModuleBuilder) raises:
    # The module that owns the linalg public operations registers the pin
    # once; _typed_borrow_array allocates it with PythonObject(alloc=...).
    # Once per process: a second initialization of the owner attaches the
    # existing type object (bindings/py_type_registry.mojo).
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[AdmittedBufferPin[]]()
    except error:
        if not py_type_absent(error, String(reflect[AdmittedBufferPin[]].name())):
            raise error
        _ = module.add_type[AdmittedBufferPin[]]("_AdmittedBufferPin")
        return
    py_type_attach(module.module, "_AdmittedBufferPin", existing)


def _typed_borrow_array(var input: AdmittedTyped, typestr: String, context: PythonObject) raises -> AdmittedBuf:
    # Array.from_buffer's zero-copy ownership semantics from metadata already
    # admitted once. The public CPython raw-memory view does not ask the exporter
    # for its metadata again; the owned pin keeps storage valid for its lifetime.
    var v = input.view.copy()
    var array = native_storage_allocate(context["array_class"])
    var held_pin = Optional[AdmittedBorrowedBuffer]()
    swap(held_pin, input.pin)
    var pin = PythonObject(alloc=AdmittedBufferPin[](held_pin^))
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyMemoryView_FromMemory")(
        MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=v.addr), v.nbytes,
        Int32(0x100 if v.readonly else 0x200))
    if not raw:
        raise cpy.unsafe_get_error()
    var bytes_view = PythonObject(from_owned=raw)
    var view = py_call(bytes_view.cast, _store_code(typestr))
    py_setattr(array, "_store", input.keep)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", pin)
    py_setattr(array, "_mv", view)
    py_setattr(array, "_addr", PythonObject(v.addr))
    py_setattr(array, "_readonly", PythonObject(v.readonly))
    py_setattr(array, "shape", admitted_shape_tuple(v.shape))
    var canonical = List[Int](length=len(v.shape), fill=0)
    var step = v.itemsize
    for axis in range(len(v.shape) - 1, -1, -1):  # small-loop(known output view shape)
        canonical[axis] = step
        step *= v.shape[axis]
    py_setattr(array, "strides", admitted_shape_tuple(canonical))
    py_setattr(array, "dtype", PythonObject(typestr))
    py_setattr(array, "ndim", PythonObject(len(v.shape)))
    py_setattr(array, "size", PythonObject(v.count()))
    py_setattr(array, "itemsize", PythonObject(v.itemsize))
    py_setattr(array, "nbytes", PythonObject(v.nbytes))
    py_setattr(array, "order", PythonObject("C"))
    return AdmittedBuf(array, v.addr, v.shape.copy())


def admitted_typed_c(var input: AdmittedTyped, typestr: String, name: String,
                     context: PythonObject, force_copy: Bool = False) raises -> AdmittedBuf:
    """The admitted input as C-order `typestr` words: borrowed when it
    already is (an Array as it is, a foreign exporter as a pinned Array over
    the same memory), else one parallel conversion into a new Array.
    `force_copy`: always the converted copy (an operation-private input)."""
    var item = 0
    var kind = _typed_kind(input.view, item)
    var target_item = 0
    var target_kind = _typed_target(typestr, target_item)
    if not force_copy and kind == target_kind and item == target_item and input.view.c_contig:
        if input.view.is_array:
            var held_pin = Optional[AdmittedBorrowedBuffer]()
            swap(held_pin, input.pin)
            return AdmittedBuf(input.keep, input.view.addr, input.view.shape.copy(), held_pin^, input.owner)
        return _typed_borrow_array(input^, typestr, context)
    # Every cell is written by the conversion; Array stores have no
    # uninitialised constructor, so the destination is the zeroed Array.
    var result = admitted_zeros(input.view.shape, typestr, context)
    admitted_typed_convert(input.view, typestr, result.addr, name)
    _ = input^
    return result^


def admitted_typed_output(var input: AdmittedTyped) -> AdmittedBuf:
    # Preserve the supplied output object's identity while its native buffer
    # export and Array storage owners stay pinned through the complete call.
    var pin = Optional[AdmittedBorrowedBuffer]()
    swap(pin, input.pin)
    return AdmittedBuf(input.keep, input.view.addr, input.view.shape.copy(), pin^, input.owner)


def admit_typed(source: PythonObject, typestr: String, ndim: Int, name: String,
                context: PythonObject, admitted: Bool = False) raises -> AdmittedBuf:
    """One argument as C-order `typestr` words of rank `ndim` (-1: any
    rank), refused when empty."""
    # Native exporters cross the metadata boundary directly. Existing native
    # list/frame/Torch/scalar admission may first materialize supported values;
    # no arbitrary Python conversion or buffer hook is a fallback here.
    # `admitted`: the family door already ran classical_admit_value over this
    # very argument. Row N237: the value is admitted once here (or by the
    # door), not again by classical_view and AdmittedTyped; a view that
    # classical_view materialized is a new object and is still admitted.
    if not admitted:
        classical_admit_value(source, context)
    var prepared = source
    if not classical_is_array(source, context):
        try:
            admitted_typed_exporter(source, context)
        except:
            prepared = classical_view(source, context, name, True)
    var input = AdmittedTyped(prepared, context, prepared is source)
    var nd = len(input.view.shape)
    if ndim >= 0 and nd != ndim:
        var hint = String("")
        if ndim == 2 and nd < 2:
            hint = ". " + String(RESHAPE_HINT)
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(nd)
                    + "-D shape " + shape_text(input.view.shape) + hint)
    if input.view.count() == 0:
        var what = String("sample(s)")
        if nd == 2 and input.view.shape[0] != 0:
            what = "feature(s)"
        var shape = shape_text(input.view.shape)
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    return admitted_typed_c(input^, typestr, name, context)


def admit_view(source: PythonObject, name: String, context: PythonObject, admitted: Bool = False) raises -> AdmittedTyped:
    """One argument's capture at its own dtype, no conversion: the metadata
    (address, shape, byte strides, item size, format) and the pinned owners
    `admit_typed` reads, for a caller that casts the words itself
    (`admitted_view_kind` + core/admitted_input_nd.mojo). An Array whose
    storage is the library's native ctypes output store (an astype or
    converter result, a fitted field) is read from its slots with its
    storage owners retained (`array_owner_once`): the ctypes-backed Array
    support the TSA owners' tsa_buffer used to repack by hand (lane
    m1-followups-misc)."""
    if not admitted:
        classical_admit_value(source, context)
    var prepared = source
    if not classical_is_array(source, context):
        try:
            admitted_typed_exporter(source, context)
        except:
            prepared = classical_view(source, context, name, True)
    return AdmittedTyped(prepared, context, prepared is source)


def admitted_view_kind(v: TensorView, name: String) raises -> Tuple[Int, Int]:
    """(CastIn element kind, item size) of a captured view; refused for a
    format no numeric Array dtype holds."""
    var item = 0
    var kind = _typed_kind(v, item)
    if kind < 0:
        raise Error("TypeError: mojolearn: " + name + " has buffer format '" + v.fmt + "', which is not a numeric array")
    return (kind, item)


def admitted_dtype_admit(dtype: PythonObject, context: PythonObject) raises:
    """Refuse a NumPy dtype whose storage cannot be copied as raw bytes
    without callbacks; NumPy's StringDType is admitted with its owners."""
    if _released_dtype(dtype, context, 0):
        return
    # NumPy's native variable-length StringDType cannot be copied as raw
    # allocator bytes. Accept its exact builtin class, with callback-free
    # missing-value/metadata owners, and use native scalar storage copying.
    var b = native_loaded_module("builtins")
    var string_dtype = PythonObject(None)
    try:
        string_dtype = py_call(native_numpy_dtype(context), "T")
    except:
        pass
    if string_dtype is None or py_type(dtype) is not py_type(string_dtype):
        raise Error("TypeError: resample requires a native NumPy dtype")
    native_release_admit(py_call(b.getattr, dtype, "na_object", PythonObject(None)), context)
    var metadata = dtype.metadata
    if metadata is not None:
        var pairs = py_call(b.list, metadata.items())
        for i in range(len(pairs)):  # driver-loop: native dtype metadata cannot retain callback owners
            native_release_admit(pairs[i][0], context)
            native_release_admit(pairs[i][1], context)


def _snapshot_flat_leaves(value: PythonObject, context: PythonObject) raises -> Bool:
    """True when no element of the list/tuple `value` is a container the
    snapshot copies (a mojolearn Array, array.array, ndarray, list, tuple or
    bytearray): the exact types are read once with `set(map(type, value))`
    and each distinct type is tested, not each element."""
    var b = native_loaded_module("builtins")
    var types = py_call(b.list, py_call(b.set, py_call(b.map, b.type, value)))
    var ndarray = py_get(context, "ndarray")
    var array_class = context["array_class"]
    var native_array = context["array_type"]
    for i in range(len(types)):  # driver-loop: over the distinct exact element types (a handful)
        var cls = types[i]
        if (cls is array_class or cls is native_array or cls is ndarray or cls is b.list
                or cls is b.tuple or cls is b.bytearray):
            return False
    return True


def _snapshot_array_alias(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """A private Array sharing `value`'s storage: every slot (storage owners,
    address and metadata) is read once through the import-pinned native slot
    descriptors and written to a fresh Array of the connected class. The GIL
    is held throughout, so the slots form one consistent capture."""
    var cls = context["array_class"]
    var fields = context["array_fields"]
    var array_alias = native_storage_allocate(cls)
    var names = py_call(native_loaded_module("builtins").list, fields)
    for i in range(len(names)):  # driver-loop: over the fixed Array slot names
        var name = names[i]
        py_setattr(array_alias, String(py=name), py_call(fields[name], value, cls))
    return array_alias


def admitted_snapshot(value: PythonObject, memo: PythonObject,
                      context: PythonObject, depth: Int = 0) raises -> PythonObject:
    """An operation-private snapshot of a metric argument: an Array through
    AdmittedTyped (a C-order one aliased with its storage owners retained, a
    strided one copied in its own dtype), array.array / bytearray / ndarray
    copied or viewed natively, containers walked with their aliases (`memo`)
    and scalar leaf dtypes kept."""
    if depth > 64:
        raise Error("TypeError: metric input nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var array = classical_is_array(value, context)
    var ndarray = py_get(context, "ndarray")
    var native_array = cls is context["array_type"]
    if not array and not native_array and cls is not ndarray and cls is not b.list and cls is not b.tuple and cls is not b.bytearray:
        # Exact scalar leaves retain their dtype inside containers. Pandas and
        # Torch retain their existing native materialization semantics.
        return value
    var key = py_call(b.id, value)
    var known = py_get(memo, key)
    if known is not None:
        return known
    var result = PythonObject(None)
    if array:
        var input = AdmittedTyped(value, context)
        if input.view.c_contig:
            # Metrics only read their inputs. A private Array that shares the
            # admitted storage owners freezes the metadata (a caller replacing
            # its slots cannot change what later readers see, and the owners
            # keep the address alive); the data itself is not copied.
            _ = input^
            result = _snapshot_array_alias(value, context)
        else:
            var dtype = input.view.dtype
            # Copy the same dtype, with the original pin/slot owners retained until
            # copying finishes. Later readers see this private Array's metadata.
            var output = admitted_typed_c(input^, dtype, "metric input", context, True)
            result = output.keep
            _ = output^
    elif native_array:
        # Exact array.array construction preserves the native typecode and
        # copies its C storage without caller iteration/conversion hooks.
        result = py_call(native_loaded_module("array").array, value.typecode, value)
    elif cls is b.bytearray:
        result = py_call(b.bytearray, value)
    elif cls is ndarray:
        # Exact native ndarray.view captures the caller's metadata once. C
        # ndarray.copy only copies storage/references in the same dtype; no
        # NumPy computation, cast, __array__, or user __copy__ hook runs.
        var view = py_call(py_attr(ndarray, "view"), value)
        var dtype = view.dtype
        admitted_dtype_admit(dtype, context)
        var flags = view.flags
        if (not Bool(py=dtype.hasobject)
                and (Bool(py=flags.c_contiguous) or Bool(py=flags.f_contiguous))):
            # Metrics only read their inputs (scikit-learn takes no snapshot).
            # The private view fixes shape/strides/dtype once and holds the
            # base storage alive; a concurrent writer can change values but
            # never the memory the kernel reads, so no data copy is needed.
            result = view
        else:
            result = py_call(py_attr(ndarray, "copy"), view, order="K")
            if Bool(py=dtype.hasobject):
                # A caller may replace its object cells while a kernel runs. The
                # private snapshot must never become the final callback owner.
                native_release_admit(result, context)
        _ = view^
    elif _snapshot_flat_leaves(value, context):
        # Every element is a leaf, which the walk below returns unchanged and
        # never memoizes: one C-level copy of the container is the same
        # snapshot, without a recursive call per element.
        result = py_call(b.tuple, value) if cls is b.tuple else py_call(b.list, value)
    else:
        var items = Python.list()
        for i in range(len(value)):  # driver-loop: stable nested native metric containers, preserving scalar leaf dtypes
            py_call(items.append, admitted_snapshot(value[i], memo, context, depth + 1))
        result = py_call(b.tuple, items) if cls is b.tuple else items
    py_setitem(memo, key, result)
    return result
