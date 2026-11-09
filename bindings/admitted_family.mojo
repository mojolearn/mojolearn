# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The family doors' face of THE ONE ADMISSION PATH (plan
# docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.3 row A2, lane
# a2-admit-family, 2026-11-08): clustering, neighbors, SVM, GP, mixture,
# kernel methods, manifold, model selection and parallel_shard admit their
# arrays here, on bindings/admitted_input.mojo's AdmittedX and the A0 core
# (core/admitted_input.mojo cast, core/host_finite_scan.mojo scan,
# core/admitted_weights.mojo weight status).
#
#   admit_family_meta(source, ndim, name, context, dense, finite, admitted)
#                     -> ArrayMetadata: metadata read once, the family
#                     refusals in `_buffer.as_f32_c`'s wording, then the
#                     float32 C-order words: the input itself when it already
#                     is one (borrowed), otherwise ONE parallel typed cast
#   admit_family(...) -> AdmittedX: the same as the AdmittedX contract
#   admit_family_f64(view, name, context) -> the float64 C-order words
#   admit_family_owned(x, context) -> an array the model may keep
#   admit_family_weights(source, n, allow_negative, context, wide)
#                     -> ArrayMetadata of the n weight words
#
# Replaces (deleted with this lane; TOMBSTONE in bindings/family_common.mojo
# and bindings/svc_common.mojo): family_f32, family_cast_f32,
# family_input_snapshot, family_snapshot_f32, family_f32_snapshot,
# family_as_f32, family_as_f32_dense, family_cast_view, family_cast_view_f64,
# family_owned, family_owned_snapshot, family_sample_weight,
# family_sample_weight_snapshot, _family_weight_scalar, _family_weight_buffer,
# svc_f32, svc_snapshot and core/family_weights_impl.mojo.
#
# What changed against those helpers (no output bit):
# - family_f32 read the metadata twice (family_view, then admit_x re-ran the
#   admission walk and the pandas/Torch tests on the view); it is read once.
# - `finite=True` takes the finiteness verdict from the cast itself (the cast
#   computed it and the snapshot path threw it away, so the door scanned the
#   copy a second time); a borrowed input is scanned once, in parallel.
# - The weight status (finite, sign, all-zero) is one parallel pass instead of
#   two serial ones, and a scalar weight is filled in parallel.
#
# A DeviceRows fold reaching a family door is handed to admit_x unchanged
# (admit_family_meta); the family side keeps no download or second scan.
#
# Numerics: none. The bytes reaching a kernel are the same float32 (or
# float64) C-order bytes, written by the same classical_f32_bits word cast;
# the finiteness and sign tests are predicates.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem, py_setattr
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.native_array_storage import native_scratch_bytes, native_scratch_address
from bindings.classical_arrays import (
    classical_address, classical_is_array, classical_cast_kind, classical_dtype_str, classical_c_order,
)
from bindings.admitted_input import AdmittedX, RESHAPE_HINT, classical_view, admit_x_unrecorded
from bindings.device_rows_door import door_is_device_rows
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.family_common import family_array, family_repr
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.admitted_input import admitted_cast_f32, admitted_cast_f64
from core.admitted_weights import (
    admitted_fill, admitted_weight_status, WEIGHTS_NONFINITE, WEIGHTS_NEGATIVE, WEIGHTS_ALL_ZERO,
)
from core.host_finite_scan import host_all_finite_f32_strided
from core.host_parallel_copy import host_parallel_copy_bytes
from core.cast_in import CastIn, CAST_KIND_FLOAT, NPY_FLOAT, NPY_DOUBLE

comptime _AF_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _AF_F64P = MutPointer[Float64, MutUntrackedOrigin]

comptime FAMILY_ZERO_WEIGHTS = "Sample weights must contain at least one non-zero number."
comptime FAMILY_NEGATIVE_WEIGHTS = "Negative values in data passed to `sample_weight`"
comptime FAMILY_NONFINITE_WEIGHTS = "Input sample_weight contains NaN or infinity."


def _shape_tuple(shape: List[Int]) raises -> PythonObject:
    """`tuple(shape)` of admitted native dimensions, for a refusal message."""
    var shape_words = Python.list()
    for word in shape:  # small-loop(admitted rank for an error message)
        _ = py_call(shape_words.append, word)
    return py_call(native_loaded_module("builtins").tuple, shape_words)


def admitted_array(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """A new C-order mojolearn Array of `shape` ("<f4" or "<f8") whose words
    are NOT initialised: the destination of a cast, a copy or a fill that
    writes every word before anything reads it. It is an ordinary owned
    Array that may be kept as a fitted field or returned: its store is an
    uninitialised bytearray (native_scratch_bytes, no fill pass) and its
    `_mv` the typed memoryview of that store, the layout every Array reader
    and the save path (bindings/native_array_snapshot.mojo) accept. An empty
    shape takes the zero-length family_array."""
    # Regression audit R6 / REGRESSIONS C3: family_array zero-filled every
    # destination (native_array_zeros) that the cast or copy then overwrote
    # in full, a second pass over X.
    var b = native_loaded_module("builtins")
    var rank = len(shape)
    var count = 1
    for k in range(rank):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    if count == 0:
        return family_array(shape, typestr, context)
    var item = 8 if typestr == "<f8" else 4
    var store = native_scratch_bytes(count * item)
    var view = py_call(native_buffer_view(store).cast, PythonObject("d" if item == 8 else "f"))
    var steps = List[Int](length=rank, fill=0)
    var step = item
    for k in range(rank):  # driver-loop: over array rank metadata
        steps[rank - 1 - k] = step
        step *= Int(py=shape[rank - 1 - k])
    var strides = Python.list()
    for k in range(rank):  # driver-loop: over array rank metadata
        _ = py_call(strides.append, steps[k])
    var array = native_storage_allocate(context["array_class"])
    py_setattr(array, "_store", store)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", PythonObject(None))
    py_setattr(array, "_mv", view)
    py_setattr(array, "_addr", PythonObject(native_scratch_address(store)))
    py_setattr(array, "_readonly", PythonObject(False))
    py_setattr(array, "shape", py_call(b.tuple, shape))
    py_setattr(array, "dtype", PythonObject(typestr))
    py_setattr(array, "ndim", PythonObject(rank))
    py_setattr(array, "size", PythonObject(count))
    py_setattr(array, "nbytes", PythonObject(count * item))
    py_setattr(array, "itemsize", PythonObject(item))
    py_setattr(array, "strides", py_call(b.tuple, strides))
    py_setattr(array, "order", PythonObject("C"))
    return array


def admit_family_input(source: PythonObject, ndim: Int, name: String, context: PythonObject,
                       dense: Bool = False, admitted: Bool = False, empty: Bool = False) raises -> ArrayMetadata:
    """The exporter's metadata, read once, with `_buffer._as_typed`'s rank,
    emptiness and rank-above-2 refusals in their wording, before any cast
    (`admit_family_cast` is the second step); `ndim` < 0 accepts any rank up
    to 2. `dense`: a scipy.sparse input is densified exactly first (its
    implicit entries are zeros)."""
    var prepared = source
    var view_admitted = admitted
    if dense:
        var sparse = host_sparse_format(source)
        if sparse.byte_length() > 0:
            if not py_contains(context, PythonObject("output_context")):
                py_setitem(context, "output_context", context)
            prepared = host_sparse_dense(source, sparse, context)
            view_admitted = False
    var view = classical_view(prepared, context, name, view_admitted)
    var shape = array_shape_once(view, context)
    var nd = len(shape)
    var count = 1
    for word in shape:  # driver-loop: over captured native array rank metadata
        count *= word
    if ndim >= 0 and nd != ndim:
        var hint = String("")
        if ndim == 2 and nd < 2:
            hint = ". " + String(RESHAPE_HINT)
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(nd)
                    + "-D shape " + family_repr(_shape_tuple(shape)) + hint)
    if count == 0 and not empty:
        var what = String("sample(s)")
        if nd == 2 and shape[0] != 0:
            what = "feature(s)"
        var text = family_repr(_shape_tuple(shape))
        raise Error("mojolearn: " + name + " is empty, shape " + text + ": Found array with 0 " + what
                    + " (shape=" + text + ") while a minimum of 1 is required.")
    if nd > 2:
        raise Error("mojolearn: " + name + " must be at most 2-D, got " + String(nd) + "-D")
    var snapshot = array_metadata_once(view, shape^, name, context)
    _ = view^
    _ = prepared^
    return snapshot^


def _fold_rows(source: PythonObject, name: String, context: PythonObject, refusal: String) raises -> AdmittedX:
    """admit_x of a DeviceRows fold, its non-finite refusal in the door's
    wording."""
    try:
        return admit_x_unrecorded(source, name, context)
    except error:
        if String(error).find("contains NaN or infinity") >= 0:
            if refusal.byte_length() > 0:
                raise Error(refusal)
            raise Error("Input " + name + " contains NaN or infinity")
        raise error^


def admit_family_meta(source: PythonObject, ndim: Int, name: String, context: PythonObject,
                      dense: Bool = False, finite: Bool = False, admitted: Bool = False,
                      empty: Bool = False, refusal: String = "") raises -> ArrayMetadata:
    """The admitted float32 C-order words of a family door's array input, as
    its metadata record: `keep` is the input itself when it already is
    float32 in C order (borrowed, never written), otherwise one cast copy
    (`copied`). `finite` refuses a NaN or an infinity ("Input <name>
    contains NaN or infinity", which a caller may reword); without it the
    words are not tested (as `_buffer.as_f32_c`). `admitted`: `source` is a
    view classical_view returned in the same native call. `empty` admits a
    view with no words (the cast of an already checked view, which
    `_buffer.as_f32_c`'s cast step never refused). `refusal`, when given,
    is the door's own wording of the non-finite refusal."""
    if door_is_device_rows(source, context):
        # A model-selection fold (`_arena_io.DeviceRows`) goes to the one
        # admission path unchanged: admit_x resolves it and tests it finite
        # once; this door neither downloads it nor scans it again. Its rows
        # are float32 C-order, so the snapshot below borrows them.
        var fold = _fold_rows(source, name, context, refusal)
        var rows = admit_family_input(fold.array, ndim, name, context, False, True, empty)
        admit_family_cast(rows, name, context)
        return rows^
    var snapshot = admit_family_input(source, ndim, name, context, dense, admitted, empty)
    admit_family_cast(snapshot, name, context, finite, refusal)
    return snapshot^


def admit_family_cast(mut snapshot: ArrayMetadata, name: String, context: PythonObject, finite: Bool = False,
                      refusal: String = "") raises:
    """The second step of `admit_family_meta` on an `admit_family_input`
    record: borrow a float32 C-order input, else cast it once (row tasks
    across the host's cores) into an owned Array recorded on `snapshot`;
    `finite` refuses a NaN or an infinity, from the cast's own verdict (or
    one parallel scan of a borrowed input)."""
    var ok = True
    var src = snapshot.source
    if snapshot.f32_c():
        if finite:
            # parallel: one finiteness scan of a borrowed input across host cores
            ok = host_all_finite_f32_strided(src.addr, src.s0, src.s1, src.rows, src.cols)
    else:
        # The destination is an owned mojolearn Array (the type the family
        # outputs and fitted fields always had), uninitialised: the cast
        # writes every word.
        var result = admitted_array(snapshot.python_shape(), "<f4", context)
        var address = Int(py=classical_address(result))
        # parallel: one typed cast over row tasks, the finiteness verdict in the same pass
        ok = admitted_cast_f32(src.addr, src.s0, src.s1, src.rows, src.cols, src.itemsize, src.kind,
                               _AF_F32P(unsafe_from_address=address))
        snapshot.replace_f32(result, address)
    if finite and not ok:
        if refusal.byte_length() > 0:
            raise Error(refusal)
        raise Error("Input " + name + " contains NaN or infinity")


def admitted_family_x(snapshot: ArrayMetadata, source: PythonObject, finite: Bool) -> AdmittedX:
    """The AdmittedX of an admitted family snapshot."""
    var borrowed = not snapshot.copied
    var x = AdmittedX(snapshot.keep, source, snapshot.source.addr, snapshot.source.rows, snapshot.source.cols,
                      len(snapshot.shape), borrowed)
    x.finite = finite
    return x^


def admit_family(source: PythonObject, ndim: Int, name: String, context: PythonObject,
                 dense: Bool = False, finite: Bool = False, admitted: Bool = False,
                 empty: Bool = False, refusal: String = "") raises -> AdmittedX:
    """`admit_family_meta` as the AdmittedX contract: `.array` the float32
    C-order array the kernels read, `.borrowed` when it is the input's own
    storage."""
    var snapshot = admit_family_meta(source, ndim, name, context, dense, finite, admitted, empty, refusal)
    return admitted_family_x(snapshot, source, finite)


def admit_family_copied(snapshot: ArrayMetadata, source: PythonObject) raises -> Bool:
    """The `input_copied_` flag of a fitted SVM: the admission copied the
    words, or the input was a Python list, tuple or range (always built
    anew)."""
    var b = native_loaded_module("builtins")
    var cls = py_type(source)
    return snapshot.copied or cls is b.list or cls is b.tuple or cls is b.range


def admit_family_f64(view: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_f64_c`'s cast of a checked view of rank 0, 1 or 2: the
    view itself when it is already float64 in C order, otherwise one
    parallel cast into a new mojolearn Array of its shape. No finiteness
    refusal (as_f64_c made none)."""
    var nd = Int(py=view.ndim)
    if nd > 2:
        raise Error("mojolearn: " + name + " must be at most 2-D, got " + String(nd) + "-D")
    if not classical_is_array(view, context) and not Bool(py=view.dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    if classical_dtype_str(view, context) == "<f8" and classical_c_order(view, context):
        return view
    var b = native_loaded_module("builtins")
    var shape = view.shape
    var rows = 1 if nd == 0 else Int(py=shape[0])
    var cols = 1 if nd < 2 else Int(py=shape[1])
    var item = Int(py=view.itemsize)
    var kind = classical_cast_kind(view, item, name, context)
    var strides = view.strides if nd > 0 else PythonObject(None)
    var s0 = 0 if nd == 0 else Int(py=strides[0])
    var s1 = 0 if nd < 2 else Int(py=strides[1])
    var result = admitted_array(py_call(b.tuple, shape), "<f8", context)
    admitted_cast_f64(Int(py=classical_address(view)), s0, s1, rows, cols, item, kind,
                      _AF_F64P(unsafe_from_address=Int(py=classical_address(result))))
    _ = result  # keep the Python object alive through the native call above
    return result


def admit_family_owned(value: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """An admitted float32 array the model may keep: the admitted array
    itself when this call owns it (the cast result, or an Array the exporter
    made and nobody else holds), else one parallel copy of the caller's
    words, so a caller editing its array after fit changes nothing."""
    if value.owned_writable:
        return value.keep
    var result = admitted_array(value.python_shape(), "<f4", context)
    # parallel: the owned float32 copy across host cores (0.8.66 regression C3).
    host_parallel_copy_bytes(Int(py=classical_address(result)), value.source.addr, value.count * 4)
    _ = result
    return result


def admit_family_owned(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`value` (a C-order float32 array) as an owned mojolearn Array: the
    array itself when it is one the owner made, else one parallel copy."""
    if classical_is_array(value, context) and value._base is None and not Bool(py=value._readonly):
        return value
    var b = native_loaded_module("builtins")
    var result = admitted_array(py_call(b.tuple, value.shape), "<f4", context)
    # parallel: the owned float32 copy across host cores (0.8.66 regression C3).
    host_parallel_copy_bytes(Int(py=classical_address(result)), Int(py=classical_address(value)),
                             Int(py=value.size) * 4)
    _ = result  # keep the Python object alive through the native call above
    _ = value
    return result


def _weight_scalar(value: Float64, n: Int, wide: Bool, context: PythonObject) raises -> ArrayMetadata:
    var item = 8 if wide else 4
    var weights = admitted_array(Python.tuple(n), "<f8" if wide else "<f4", context)
    var address = Int(py=classical_address(weights))
    if wide:
        admitted_fill[DType.float64](_AF_F64P(unsafe_from_address=address), n, value)
    else:
        admitted_fill[DType.float32](_AF_F32P(unsafe_from_address=address), n, Float32(value))
    var descriptor = CastIn(address, CAST_KIND_FLOAT, item, n, 1, item, 0)
    return ArrayMetadata(weights, List[Int]([n]), descriptor, n, NPY_DOUBLE if wide else NPY_FLOAT,
                         True, True, True, weights, True)


def _weight_buffer(source: PythonObject, n: Int, wide: Bool, context: PythonObject) raises -> ArrayMetadata:
    # Admit supported native storage before reading any user attributes. One
    # shape and dtype/stride snapshot is retained through validation and use.
    var view = classical_view(source, context, "sample_weight")
    var shape = array_shape_once(view, context)
    var count = 1
    for dimension in shape:  # small-loop(admitted sample-weight rank)
        count *= dimension
    if count == 0:
        raise Error("sample_weight.shape == (0,), expected (" + String(n) + ",)!")
    var nd = len(shape)
    if nd == 0:
        # A zero-dimensional buffer is a broadcast scalar, read with the
        # widening cast (never float(Array)).
        if py_type(source) is py_type(PythonObject(True)):
            raise Error("Sample weights must be 1D array or scalar")
        var scalar = array_metadata_once(view, shape^, "sample_weight", context, True)
        var one = family_array(Python.tuple(1), "<f8", context)
        var ptr = _AF_F64P(unsafe_from_address=Int(py=classical_address(one)))
        var raw = scalar.source
        var scalar_address = raw.addr
        var reversed = family_array(Python.tuple(raw.itemsize), "<u1", context)
        if not scalar.native_byte_order:
            # float(zero_dim_ndarray) accepted a byte-swapped scalar; keep
            # that scalar-only behavior before the unchanged cast.
            var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=raw.addr)
            var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(reversed)))
            for byte in range(raw.itemsize):  # small-loop(one admitted scalar word)
                dst[byte] = src[raw.itemsize - 1 - byte]
            scalar_address = Int(py=classical_address(reversed))
        admitted_cast_f64(scalar_address, 0, 0, 1, 1, raw.itemsize, raw.kind, ptr)
        var result = _weight_scalar(ptr[0], n, wide, context)
        _ = scalar
        _ = one
        _ = reversed
        return result^
    if nd != 1:
        raise Error("Sample weights must be 1D array or scalar")
    if shape[0] != n:
        raise Error("sample_weight.shape == " + family_repr(_shape_tuple(shape)) + ", expected (" + String(n) + ",)!")
    var snapshot = array_metadata_once(view, shape^, "sample_weight", context)
    var ready = snapshot.c_order and snapshot.dtype_number == (NPY_DOUBLE if wide else NPY_FLOAT)
    if not ready:
        var weights = admitted_array(Python.tuple(n), "<f8" if wide else "<f4", context)
        var address = Int(py=classical_address(weights))
        var raw = snapshot.source
        if wide:
            admitted_cast_f64(raw.addr, raw.s0, raw.s1, raw.rows, raw.cols, raw.itemsize, raw.kind,
                              _AF_F64P(unsafe_from_address=address))
            snapshot.replace_f64(weights, address)
        else:
            _ = admitted_cast_f32(raw.addr, raw.s0, raw.s1, raw.rows, raw.cols, raw.itemsize, raw.kind,
                                  _AF_F32P(unsafe_from_address=address))
            snapshot.replace_f32(weights, address)
    return snapshot^


def admit_family_weights(source: PythonObject, n: Int, allow_negative: Bool, context: PythonObject,
                         wide: Bool = False) raises -> ArrayMetadata:
    """sample_weight as `n` float32 (or, `wide`, float64) words: an int or a
    float fills every row; prepare_targets' refusals in their order (NaN or
    infinity, a negative weight unless `allow_negative`, all zero). `.keep`
    is the weight array the kernels read."""
    var b = native_loaded_module("builtins")
    var snapshot: ArrayMetadata
    if py_type(source) is b.int or py_type(source) is b.float:
        snapshot = _weight_scalar(Float64(py=source), n, wide, context)
    else:
        snapshot = _weight_buffer(source, n, wide, context)
    var address = snapshot.source.addr
    var status: Int
    if wide:
        status = admitted_weight_status[DType.float64](_AF_F64P(unsafe_from_address=address), n, allow_negative)
    else:
        status = admitted_weight_status[DType.float32](_AF_F32P(unsafe_from_address=address), n, allow_negative)
    if status == WEIGHTS_NONFINITE:
        raise Error(FAMILY_NONFINITE_WEIGHTS)
    if status == WEIGHTS_NEGATIVE:
        raise Error(FAMILY_NEGATIVE_WEIGHTS)
    if status == WEIGHTS_ALL_ZERO:
        raise Error(FAMILY_ZERO_WEIGHTS)
    return snapshot^
