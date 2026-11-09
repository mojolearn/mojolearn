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
#   an ndarray), `allow_nonfinite` (NaN and infinity pass: imputers and
#   the NaN-accepting doors; `finite` stays False). A matrix held in bindings/admitted_registry.mojo by the
#   running operation is returned as it is (C order, not as_array).
#   admit_f32(source, ndim, name, context, ...) -> AdmittedX
#                     the array form as an AdmittedX (ndim -1: any rank up
#                     to 2): targets, fitted arrays, published predictions
#   admit_weight_words(source, rows, context) -> AdmittedX
#                     admit_weights as an AdmittedX; None is `rows` ones
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
#   host entry (CPU column) calls `host_finite_status`
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
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays import (
    classical_wrap_array, _store_code, classical_input, classical_kind, classical_address, classical_cast_kind,
    classical_is_array, classical_is_f32, classical_c_order, classical_numpy,
)
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, classical_admit_value
from bindings.classical_warnings import classical_warn_column
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.native_array_storage import native_scratch_bytes, native_array_zeros
from bindings.admitted_registry import admitted_held_index, admitted_held_address, admitted_held_rows, admitted_held_cols
from bindings.device_rows_door import door_is_device_rows, DoorRows
from core.admitted_input import admitted_check_f32, admitted_check_weights_f32, admitted_cast_f32
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_NONFINITE, ARG_OUT_OF_RANGE

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
            as_array: Bool = False, allow_nonfinite: Bool = False) raises -> PythonObject:
    """The float32 array of an input of rank `ndim` in `order` ("C", or "F"
    for a 2-D input): the input itself when it already is one (borrowed),
    otherwise one cast copy into uninitialised storage. `as_array`: the
    result is a mojolearn Array, never an ndarray (an ndarray input is then
    copied). Refuses a rank mismatch, an empty input, a non-native byte
    order and NaN or infinity ("Input <name> contains NaN or infinity")."""
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
    var b = native_loaded_module("builtins")
    var prepared = source
    if not admitted:
        classical_admit_value(source, context)
        if native_pandas_exact(source):
            prepared = native_pandas_f32(source, context)
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
    if allow_nonfinite:
        # A NaN-accepting door (imputers, missing-value encoders): the cast
        # still runs, its finite flag is not a refusal; a borrowed input is
        # read as it is (no pass at all).
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
              defer_finite: Bool = False, as_array: Bool = False, allow_nonfinite: Bool = False,
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
                        allow_nonfinite)
    var nd = Int(py=array.ndim)
    var rows = 1 if nd == 0 else Int(py=array.shape[0])
    var cols = 1 if nd < 2 else Int(py=array.shape[1])
    var x = AdmittedX(array, source, Int(py=classical_address(array)), rows, cols, nd, array is source,
                      not allow_nonfinite and not defer_finite)
    x.c_order = order == "C" or rows == 1 or cols == 1
    return x^


def admit_weight_words(source: PythonObject, rows: Int, context: PythonObject,
                       admitted: Bool = False) raises -> AdmittedX:
    """`admit_weights(source, rows)` as an AdmittedX of `rows` words; None
    is `rows` ones (the unweighted fit's weights, as a fresh vector)."""
    # Lane a1b: the former labels_check_sample_weight / trees_check_sample_weight.
    var w: PythonObject
    if source is None:
        w = admitted_storage_f32(rows, 0, context, "C", True)
        var p = _ADM_F32P(unsafe_from_address=Int(py=classical_address(w)))
        for i in range(rows):  # native input marshaling: the ones of an unweighted fit, one store per row
            p[i] = 1.0
    else:
        w = admit_weights(source, rows, context, admitted)
    var x = AdmittedX(w, source, Int(py=classical_address(w)), rows, 1, 1, w is source, True)
    return x^


def classical_view(source: PythonObject, context: PythonObject, name: String = "input",
                   admitted: Bool = False) raises -> PythonObject:
    """An array view of an admitted input (pandas and Torch storage read
    natively), before any cast: an ndarray with NumPy, an Array without."""
    # `admitted`: the caller's native entry already ran classical_admit_value
    # over this very argument (classical_c admits every operation argument
    # before dispatch), so the walk is not repeated; storage is still read.
    if not admitted:
        classical_admit_value(source, context)
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
           as_array: Bool, allow_nonfinite: Bool = False) raises -> PythonObject:
    var nd = Int(py=view.ndim)
    if nd != 2:
        raise Error("mojolearn: X must be 2-D, got " + String(nd) + "-D shape " + _repr(view.shape))
    if not _numeric(view, context):
        raise Error("TypeError: mojolearn: X has buffer format " + _repr(py_call(native_loaded_module("builtins").memoryview, view).format) + ", which is not a numeric array")
    try:
        return admit_x(view, 2, "X", context, admitted=True, defer_finite=defer_finite, order=order,
                       as_array=as_array, allow_nonfinite=allow_nonfinite)
    except error:
        if String(error) == "Input X contains NaN or infinity":
            raise Error("Input X contains NaN or infinity (" + est + ").")
        raise Error(String(error))


def _check_x_view(view: PythonObject, est: String, context: PythonObject, defer_finite: Bool = False,
                  order: String = "C", as_array: Bool = False, allow_nonfinite: Bool = False) raises -> PythonObject:
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
    return _cast_x(view, est, context, defer_finite, order, as_array, allow_nonfinite)


def _admitted_rows(source: PythonObject, est: String, context: PythonObject, device_x: Bool) raises -> AdmittedX:
    """A model-selection fold (`_arena_io.DeviceRows`) admitted without its
    device gather ever being downloaded (REGRESSIONS R13; plan section 3.2
    step 4). The handle (base, indices) always rides on the result. With
    `device_x` the door's entry gathers on the device out of the resident
    base: no host word is read or written here. Otherwise the rows are
    gathered on the host straight out of the host base, in row tasks across
    the cores, finite-tested in the copy (core/admitted_rows_gather.mojo).
    The same rows in the same order as model selection's device gather, so
    the same bytes reach the kernel."""
    # TOMBSTONE: door_rows_host (bindings/device_rows_door.mojo, the DeviceRows
    # materialize: device gather, download, host scan, kernel re-upload) deleted
    # 2026-11-08 by lane a5-admit-core-followup; code recoverable at f7d10d2b09.
    # Restore: experiments/removed/A5_DOOR_ROWS_HOST.patch.
    var door = DoorRows(source, context)
    var base = classical_view(door.base, context, "X", True)
    if Int(py=base.ndim) != 2 or not classical_is_f32(base, context) or not classical_c_order(base, context):
        raise Error("TypeError: DeviceRows requires a float32 C-order 2-D base")
    var indices = classical_view(door.indices, context, "indices", True)
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
    var finite = False
    if not device_x:
        array = admitted_storage_f32(rows, columns, context)
        address = Int(py=classical_address(array))
        var status = admitted_gather_rows_f32(Int(py=classical_address(base)), base_rows, columns,
                                              Int(py=classical_address(indices)), rows, address)
        if status == ARG_OUT_OF_RANGE:
            raise Error("IndexError: DeviceRows row index out of range for a base of " + String(base_rows) + " rows")
        if status == ARG_NONFINITE:
            raise Error("Input X contains NaN or infinity (" + est + ").")
        finite = True
    var x = AdmittedX(array, source, address, rows, columns, 2, False, finite)
    x.device_rows = True
    x.rows_base = door.base
    x.rows_indices = door.indices
    _ = base^
    _ = indices^
    return x^


def _admitted_x(source: PythonObject, est: String, context: PythonObject, admitted: Bool,
                defer_finite: Bool = False, device_x: Bool = False, order: String = "C",
                as_array: Bool = False, allow_nonfinite: Bool = False) raises -> AdmittedX:
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
            var rows_x = _admitted_rows(source, est, context, False)
            var ordered = _admitted_x(rows_x.array, est, context, True, defer_finite, False, order, as_array,
                                      allow_nonfinite)
            ordered.source = source
            ordered.device_rows = True
            ordered.rows_base = rows_x.rows_base
            ordered.rows_indices = rows_x.rows_indices
            return ordered^
        return _admitted_rows(source, est, context, device_x)
    classical_refuse_sparse(source, est)
    var view = classical_view(source, context, "X", admitted)
    var x = _check_x_view(view, est, context, defer_finite, order, as_array, allow_nonfinite)
    # _cast_x refused every rank but 2; the cast keeps the view's shape.
    var rows = Int(py=view.shape[0])
    var columns = Int(py=view.shape[1])
    # Borrowed exactly when the kernels read the view's own words.
    var address = Int(py=classical_address(x))
    var borrowed = address == Int(py=classical_address(view))
    _ = view^
    # A borrowed X under defer_finite was not scanned: its entry tests it.
    # Under allow_nonfinite nothing was refused.
    var result = AdmittedX(x, source, address, rows, columns, 2, borrowed,
                           not allow_nonfinite and not (borrowed and defer_finite))
    result.c_order = order == "C" or rows == 1 or columns == 1
    return result^


def admit_x_unrecorded(source: PythonObject, est: String, context: PythonObject,
                       admitted: Bool = False, defer_finite: Bool = False,
                       device_x: Bool = False, order: String = "C",
                       as_array: Bool = False, allow_nonfinite: Bool = False) raises -> AdmittedX:
    """X for fit, predict or score: refused with scikit-learn's wording (a
    sparse, complex, empty or non-2-D X; NaN or infinity), then the C-order
    float32 words the kernels read (the input itself when it already is
    one). For a door without an admission record.

    `defer_finite`: the door's kernel entry tests the words of a borrowed X
    and returns the status word (`finite` is then False; refuse through
    `admitted_finite_refuse`). `device_x`: the door's entry reads a
    DeviceRows fold from its device handle (`device_rows`), so no host rows
    are made. `order`: "C" (default) or "F" words. `as_array`: the array
    is a mojolearn Array, never an ndarray. `allow_nonfinite`: NaN and
    infinity are admitted as they are (a NaN-accepting door)."""
    return _admitted_x(source, est, context, admitted, defer_finite, device_x, order, as_array, allow_nonfinite)


def admit_x(mut call: AdmittedCall, source: PythonObject, est: String, context: PythonObject,
            position: Int, admitted: Bool = False, defer_finite: Bool = False,
            device_x: Bool = False, order: String = "C", as_array: Bool = False,
            allow_nonfinite: Bool = False) raises -> AdmittedX:
    """As above, recording the admitted X on the door's admission record at
    `position` (its rows, columns, address and the array that owns it), so
    the door reads the shape from the record instead of admitting X again
    (labels_shape_of; rows N254, N279). A DeviceRows fold under `device_x`
    is recorded with address 0 and no owning array; so is an F-order X
    (the record's readers take C-order words)."""
    var x = _admitted_x(source, est, context, admitted, defer_finite, device_x, order, as_array, allow_nonfinite)
    x.position = call.add_input(admitted_input_dense(position, source, x.array, x.address if x.c_order else 0,
                                                     x.rows, x.cols, ADMITTED_ITEM_F32, False))
    return x^


def admitted_finite_refuse(status: Int, est: String, name: String = "X") raises:
    """The refusal of a deferred finite test: `status` is the word the
    kernel entry returned (core/admitted_finite_status.mojo; 0 = every word
    finite). Raises scikit-learn's wording, as admit_x would have."""
    if status != 0:
        raise Error("Input " + name + " contains NaN or infinity (" + est + ").")


def admit_rows_host(value: PythonObject, est: String, context: PythonObject) raises -> PythonObject:
    """The host rows of a DeviceRows fold as one fresh C-order float32
    array, gathered in parallel out of model selection's host base (no
    device gather, no download, the finite test in the copy). For a door
    that still hands its owners an array object instead of an AdmittedX."""
    return _admitted_rows(value, est, context, False).array


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
