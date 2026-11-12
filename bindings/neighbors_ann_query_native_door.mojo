# SPDX-License-Identifier: Apache-2.0
# Public IVFPQ/IVFSQ/IVFRaBitQ/Cagra search ownership. The caller admits the
# estimator layout and selects its import-connected x_ann binding natively.
# Input/fitted metadata is captured once here, with storage owners retained
# through prepare/search/filter/result conversion. The numerical operations
# are the ANN operations table's entries (core/neighbors_entry_table_ann.mojo,
# lane calltable-cn-ann): the binding's key is read once at UNPACK
# (`ann_ops_key`) and every prepare, search, filter, code export and release
# is a native call with plain words, no CPython in between. The resident
# operations run with the GIL held (their registry, x_ann/resident.mojo, is
# guarded by it); the per-call searches and the filter with it released.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_ann.neighbors_ann_query_run import (
    ANN_SEARCH_CAGRA, AnnSearchRequest, ann_search_owner, ann_check_search_dims,
    ann_check_query_features, cagra_check_search, ivf_check_search, pq_check_search, pq_check_code_dims,
    cagra_seed_word, ann_check_stored_shape, ann_check_stored_strides, ann_mask_refusal, ann_mask_fill_ones,
    ann_mask_from_bytes, ann_search_plan, ann_handle_addresses, ann_full_addresses, ann_filter_params,
)
from bindings.admitted_family_door import admit_family_meta
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.admitted_input import admitted_finite_refuse
from bindings.admitted_finite_wording import admitted_finite_wording
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put, prepared_drop
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.mode_metadata_native_door import _fields
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable, fitted_freeze
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_address, classical_is_array
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_validate_unpack import classical_view
from bindings.family_common_unpack import FAMILY_I32P
from bindings.family_common_pack import family_array, family_call_kw
from bindings.parallel_public_result_door import parallel_public_result
from providers.python.array_metadata_pyapi_door import (
    array_shape_once, array_slot_once, array_owner_once, array_metadata_once,
)
from bindings.native_ops_key_ann_unpack import ann_ops_key
from core.native_descriptor_abi import NativeDescriptorCall
from core.neighbors_entry_table_ann import (
    ANN_OP_IVF_PQ_SEARCH, ANN_OP_IVF_SQ_SEARCH, ANN_OP_IVF_RABITQ_SEARCH, ANN_OP_CAGRA_SEARCH,
    ANN_OP_FILTER_TOPK, ann_ops_entry, ann_index_prepare_call, ann_index_search_call,
    ann_index_release_call, ann_index_export_codes_call, ann_search_call,
)
from core.cast_in import CAST_KIND_BOOL


struct _StoredArray(Copyable, Movable):
    # ArrayMetadata's numerical CastIn currently supports rank <= 2. PQ
    # codebooks have rank three, so retain their exact storage metadata here
    # without flattening through Python or reading the source shape twice.
    var value: PythonObject
    var owner: PythonObject
    var shape: List[Int]
    var address: Int

    def __init__(out self, value: PythonObject, owner: PythonObject,
                 var shape: List[Int], address: Int):
        self.value = value
        self.owner = owner
        self.shape = shape^
        self.address = address

    def key(self) raises -> PythonObject:
        var shape = Python.list()
        for axis in self.shape:  # driver-loop: over (saved array rank)
            _ = py_call(shape.append, axis)
        return Python.tuple(py_call(native_loaded_module("builtins").id, self.value),
                            self.address, py_call(native_loaded_module("builtins").tuple, shape))


def _int(fields: PythonObject, name: String, owner: String) raises -> Int:
    var value = py_get(fields, name)
    if py_type(value) is not py_type(PythonObject(0)):
        raise Error("TypeError: mojolearn " + owner + ": " + name + " must be an exact native int")
    return Int(py=value)


def _stored(fields: PythonObject, name: String, expected: List[Int], floating: Bool,
            context: PythonObject) raises -> _StoredArray:
    var value = py_get(fields, name)
    classical_admit_value(value, context)
    var is_array = classical_is_array(value, context)
    if not is_array and py_type(value) is not py_get(context, "ndarray"):
        raise Error("TypeError: " + name + " requires exact native array storage")
    var shape = array_shape_once(value, context)
    ann_check_stored_shape(name, shape, expected)
    var dtype = array_slot_once(value, "dtype", context) if is_array else value.dtype
    var item = array_slot_once(value, "itemsize", context) if is_array else value.itemsize
    if py_type(item) is not py_type(PythonObject(0)) or Int(py=item) != 4:
        raise Error("TypeError: fitted " + name + " requires 32-bit storage")
    if is_array:
        if py_type(dtype) is not py_type(PythonObject("")) or String(dtype) != ("<f4" if floating else "<i4"):
            raise Error("TypeError: invalid fitted " + name + " dtype")
    elif String(dtype.kind) != ("f" if floating else "i") or not Bool(py=dtype.isnative):
        raise Error("TypeError: invalid fitted " + name + " dtype or byte order")
    var strides = array_slot_once(value, "strides", context) if is_array else value.strides
    if py_type(strides) is not py_type(Python.tuple()) or len(strides) != len(shape):
        raise Error("TypeError: fitted array strides must be an exact native tuple")
    # The words up to the last axis whose stride is not an exact int; RUN
    # refuses that axis when its walk from the last axis reaches it.
    var steps = List[Int](length=len(shape), fill=0)
    var non_int_axis = -1
    for offset in range(len(shape)):  # driver-loop: over (saved array rank)
        var axis = len(shape) - offset - 1
        if py_type(strides[axis]) is not py_type(PythonObject(0)):
            non_int_axis = axis
            break
        steps[axis] = Int(py=strides[axis])
    ann_check_stored_strides(name, shape, steps, non_int_axis)
    var address = array_slot_once(value, "_addr", context) if is_array else classical_address(value)
    if py_type(address) is not py_type(PythonObject(0)):
        raise Error("TypeError: fitted array address must be an exact native int")
    return _StoredArray(value, array_owner_once(value, context), shape^, Int(py=address))


comptime _ANN_STORED_SLOT = "neighbors_ann_stored_v1"


def ann_stored_drop(model: PythonObject) raises:
    """Drop the stored-array admission stamp of `model` (and the arrays it
    retains). For the fit owners: a refit replaces the fitted arrays."""
    var mods = NativeModules()
    prepared_drop(mods, _ANN_STORED_SLOT, model)


def _stored_all(model: PythonObject, fields: PythonObject, names: List[String], shapes: List[List[Int]],
                floating: List[Bool], context: PythonObject) raises -> List[_StoredArray]:
    """`_stored` of every named fitted array, once while they are unchanged.

    Perf (perf2-neighbors): the admitted arrays' identity stamp lives in the
    native prepared registry, outside `__dict__`, weakly keyed by the model
    (as the IVF, radius, KDE and local-neighbor stamps). Each record entry
    is (array, owner, address, shape); while every slot holds the same
    object and the fitted dimensions give the same shapes, a search reuses
    the captured owners and addresses instead of re-admitting each array and
    re-reading its shape, dtype, itemsize, strides and address slots."""
    var mods = NativeModules()
    var b = native_loaded_module("builtins")
    var record = prepared_get(mods, _ANN_STORED_SLOT, model)
    var count = len(names)
    var hit = record is not None and py_type(record) is b.tuple and len(record) == 4 * count
    if hit:
        for i in range(count):  # driver-loop: over (fixed saved-array schema)
            if record[4 * i] is not py_get(fields, names[i]):
                hit = False
                break
            var shape = record[4 * i + 3]
            if len(shape) != len(shapes[i]):
                hit = False
                break
            for axis in range(len(shapes[i])):  # driver-loop: over (saved array rank)
                if Int(py=shape[axis]) != shapes[i][axis]:
                    hit = False
                    break
            if not hit:
                break
    var arrays = List[_StoredArray]()
    if hit:
        for i in range(count):  # driver-loop: over (fixed saved-array schema)
            arrays.append(_StoredArray(record[4 * i], record[4 * i + 1], shapes[i].copy(), Int(py=record[4 * i + 2])))
        return arrays^
    var stamp = Python.list()
    for i in range(count):  # driver-loop: over (fixed saved-array schema)
        arrays.append(_stored(fields, names[i], shapes[i].copy(), floating[i], context))
        var words = Python.list()
        for axis in arrays[i].shape:  # driver-loop: over (saved array rank)
            _ = py_call(words.append, axis)
        _ = py_call(stamp.append, arrays[i].value)
        _ = py_call(stamp.append, arrays[i].owner)
        _ = py_call(stamp.append, arrays[i].address)
        _ = py_call(stamp.append, py_call(b.tuple, words))
    prepared_put(mods, _ANN_STORED_SLOT, model, py_call(b.tuple, stamp))
    return arrays^


def _mask(source: PythonObject, n: Int, owner: String, context: PythonObject) raises -> PythonObject:
    var result = family_array(Python.tuple(n), "<i4", context)
    var dest_address = Int(py=classical_address(result))
    var b = native_loaded_module("builtins")
    var refusal = ann_mask_refusal(owner, n)
    if source is None:
        ann_mask_fill_ones(dest_address, n)
        return result
    classical_admit_value(source, context)
    if py_type(source) is b.list or py_type(source) is b.tuple:
        if len(source) != n:
            raise Error(refusal)
        var dest = FAMILY_I32P(unsafe_from_address=dest_address)
        for i in range(n):  # cpu-route: explicit boolean input admission and copy
            if py_type(source[i]) is not b.bool:
                raise Error(refusal)
            dest[i] = 1 if Bool(py=source[i]) else 0
        return result
    var view = classical_view(source, context, "filter")
    var shape = array_shape_once(view, context)
    if len(shape) != 1 or shape[0] != n:
        raise Error(refusal)
    var pin = ArrayPin()
    var snapshot = array_metadata_once(view, shape^, "filter", context, pin)
    if snapshot.source.kind != CAST_KIND_BOOL or snapshot.source.itemsize != 1:
        raise Error(refusal)
    ann_mask_from_bytes(dest_address, snapshot.source.addr, n, snapshot.source.s0)
    _ = snapshot
    _ = pin^
    return result


def _key_equal(a: PythonObject, b: PythonObject) raises -> Bool:
    # Keys contain only tuples, exact ints and the fixed "device" marker.
    # Never dispatch tuple equality onto mutable scalar protocols.
    var kind = py_type(a)
    if kind is not py_type(b):
        return False
    if kind is py_type(PythonObject(0)):
        return Int(py=a) == Int(py=b)
    if kind is py_type(PythonObject("")):
        return String(a) == String(b)
    if kind is not py_type(Python.tuple()):
        raise Error("TypeError: resident ANN key requires exact native tuple/int/string metadata")
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(fixed resident-key metadata)
        if not _key_equal(a[i], b[i]):
            return False
    return True


def _resident(fields: PythonObject) raises -> PythonObject:
    var cached = py_get(fields, "_resident")
    if cached is None:
        return cached
    var b = native_loaded_module("builtins")
    if (py_type(cached) is not b.tuple or len(cached) != 3
            or py_type(cached[1]) is not b.int or py_type(cached[2]) is not py_type(b)):
        raise Error("TypeError: resident ANN state requires its native handle and module owner")
    # Validate the complete key even when its first differing cell would make
    # comparison short-circuit. Nothing in an unvisited cell may run on release.
    _ = _key_equal(cached[0], cached[0])
    # The minting image must be a device column (the only one with a
    # resident index); its key travels with the handle as the module.
    if ann_ops_key(cached[2]).vendor == 0:
        raise Error("TypeError: resident ANN state requires its native handle and module owner")
    return cached


def _release(cached: PythonObject):
    """Best-effort release of a resident handle through its minting image's
    table entry (GIL held: the registry is guarded by it)."""
    try:
        var entry = ann_ops_entry(ann_ops_key(cached[2]))
        ann_index_release_call(entry, Int(py=cached[1]))
    except:
        pass  # Preserve the previous best-effort old-handle release.


def _key(arrays: List[_StoredArray], device_codes: Bool, binding: PythonObject) raises -> PythonObject:
    var values = Python.list()
    for i in range(len(arrays)):  # driver-loop: over (fixed saved-array schema)
        _ = py_call(values.append, arrays[i].key())
    if device_codes:
        _ = py_call(values.append, Python.tuple("device"))
    var b = native_loaded_module("builtins")
    return Python.tuple(py_call(b.tuple, values), py_call(b.id, binding))


def _addresses(arrays: List[_StoredArray]) -> List[Int]:
    var result = List[Int](capacity=len(arrays))
    for i in range(len(arrays)):  # driver-loop: over (fixed saved-array schema)
        result.append(arrays[i].address)
    return result^


def _export_codes(fields: PythonObject, cached: PythonObject, n: Int, pq_dim: Int,
                  context: PythonObject) raises -> _StoredArray:
    if cached is None:
        raise Error("RuntimeError: mojolearn IVFPQIndex: the resident index was released before its codes were read")
    var codes = family_array(Python.tuple(n, pq_dim), "<i4", context)
    var address = Int(py=classical_address(codes))
    var entry = ann_ops_entry(ann_ops_key(cached[2]))
    # GIL held: the resident registry (see the module note).
    ann_index_export_codes_call(entry, Int(py=cached[1]), address, n * pq_dim)
    py_setitem(fields, "codes_", codes)
    _ = py_call(fields.pop, "_device_codes", PythonObject(None))
    # The resident handle keeps these codes on the device (hazard class A).
    _ = fitted_freeze(codes, context)
    # Known allocation metadata; do not rediscover shape/dtype from output.
    return _StoredArray(codes, codes, [n, pq_dim], address)


def ann_pq_materialize(model: PythonObject, context: PythonObject) raises -> PythonObject:
    # Shared by the parent's native lazy attribute and pickle-state owners.
    # This returns the stored native Array; the outer public owner supplies
    # its usual completed ndarray view when this is an attribute read.
    classical_connect_numpy(context)
    var fields = _fields(model)
    var current = py_get(fields, "codes_")
    if current is not None:
        return current
    if py_get(fields, "_device_codes") is not PythonObject(True):
        raise Error("AttributeError: 'IVFPQIndex' object has no attribute 'codes_'")
    var n = _int(fields, "n_rows_", "IVFPQIndex")
    var pq_dim = _int(fields, "pq_dim_", "IVFPQIndex")
    pq_check_code_dims(n, pq_dim)
    var cached = _resident(fields)
    if cached is None:
        raise Error("RuntimeError: mojolearn IVFPQIndex: the resident index was released before its codes were read")
    var key = cached[0]
    var tuple_type = py_type(Python.tuple())
    if py_type(key) is not tuple_type or len(key) != 2 or py_type(key[0]) is not tuple_type or len(key[0]) != 5:
        raise Error("TypeError: resident IVFPQIndex requires its five-array native key")
    var exported = _export_codes(fields, cached, n, pq_dim, context)
    # Replace only the device-code marker. If the caller changed a different
    # fitted array, retaining that array's old key makes the next search
    # prepare a new handle; lazy export must not bless stale device arrays.
    var arrays = Python.list()
    for i in range(4):  # driver-loop: over (PQ's four non-code array keys)
        _ = py_call(arrays.append, key[0][i])
    _ = py_call(arrays.append, exported.key())
    var replacement = Python.tuple(py_call(native_loaded_module("builtins").tuple, arrays), key[1])
    py_setitem(fields, "_resident", Python.tuple(replacement, cached[1], cached[2]))
    return exported.value


def _handle(fields: PythonObject, kind: Int, binding: PythonObject, entry: NativeDescriptorCall,
            resident: Bool, mut arrays: List[_StoredArray], params: List[Int], n: Int,
            pq_dim: Int, device_codes: Bool, context: PythonObject) raises -> PythonObject:
    # `resident`: the binding's table is a device column (the CPU column
    # keeps no resident index and searches per call).
    var cached = _resident(fields)
    var key = _key(arrays, device_codes, binding)
    if resident and cached is not None and _key_equal(cached[0], key):
        return cached[1]
    if device_codes:
        arrays.append(_export_codes(fields, cached, n, pq_dim, context))
        key = _key(arrays, False, binding)
    if not resident:
        return PythonObject(None)
    # p3-neighbors-cluster (hazard class A): the key holds identity, address
    # and shape, never values, so the fitted arrays become read-only when the
    # device copy is taken; storage that cannot be frozen (a caller-assigned
    # ndarray) keeps the per-call search (bindings/p3nc_fitted_freeze.mojo).
    for i in range(len(arrays)):  # driver-loop: over (fixed saved-array schema)
        if not fitted_is_freezable(arrays[i].value, context):
            if cached is not None:
                _ = py_call(fields.pop, "_resident", PythonObject(None))
                _release(cached)
            return PythonObject(None)
    for i in range(len(arrays)):  # small-loop(fixed saved-array schema)
        _ = fitted_freeze(arrays[i].value, context)
    if cached is not None:
        _ = py_call(fields.pop, "_resident", PythonObject(None))
        _release(cached)
    # GIL held: the resident registry (see the module note). The addressed
    # arrays are retained by `arrays` (the caller's) through the call.
    var handle = PythonObject(ann_index_prepare_call(entry, kind, _addresses(arrays), params))
    py_setitem(fields, "_resident", Python.tuple(key, handle, binding))
    return handle


def _cagra_seed(fields: PythonObject) raises -> Int:
    var value = py_get(fields, "random_state")
    if value is None:
        return 0
    if py_type(value) is not py_type(PythonObject(0)):
        raise Error("TypeError: mojolearn CagraIndex: random_state must be an exact native int")
    # Export the admitted exact integer's bytes without decimal conversion's
    # Python digit limit. Only scalar representation crosses this boundary;
    # the int31 modular arithmetic runs in Mojo, including large seed values.
    var b = native_loaded_module("builtins")
    var count = (Int(py=py_call(b.int.bit_length, value)) + 8) // 8
    var keywords = Python.dict()
    py_setitem(keywords, "signed", PythonObject(True))
    var bytes = family_call_kw(b.int.to_bytes, keywords, value, PythonObject(count), PythonObject("big"))
    var seed_bytes = List[Int](capacity=count)
    for i in range(count):  # driver-loop: over the exact integer's bytes (UNPACK of one scalar)
        seed_bytes.append(Int(py=bytes[i]))
    return cagra_seed_word(seed_bytes)


def neighbors_ann_search(model: PythonObject, kind: Int, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = _fields(model)
    var binding = args[0]
    var owner = ann_search_owner(kind)
    var device_codes = kind == 0 and py_contains(fields, "_device_codes") and not py_contains(fields, "codes_")
    if device_codes and fields["_device_codes"] is not PythonObject(True):
        raise Error("TypeError: resident ANN codes marker must be the native true value")
    if not py_contains(fields, "graph_" if kind == ANN_SEARCH_CAGRA else "codes_") and not device_codes:
        raise Error("ValueError: mojolearn " + owner + ": call fit before search")
    var n = _int(fields, "n_rows_", owner)
    var dim = _int(fields, "n_features_in_", owner)
    var k = _int(fields, "n_neighbors", owner)
    ann_check_search_dims(n, dim, k)
    # Lane a8b-device-x-neighbors-rest: the queries are refused when they hold
    # a NaN or an infinity (they were not tested at all before). A cast is
    # refused by its own verdict; borrowed float32 queries are tested by the
    # search entry behind its query upload on the GPU binding, by one parallel
    # host scan on the CPU column, and the word comes back with the result.
    var q_pin = ArrayPin()
    var q = admit_family_meta(args[1], 2, "queries", context, q_pin, False, True,
                              refusal=admitted_finite_wording(owner, "queries"), defer_finite=True)
    var status = List[Int32](length=1, fill=Int32(0))
    var status_word = 0 if q.copied else Int(status.unsafe_ptr())
    var m = q.shape[0]
    ann_check_query_features(owner, q.shape[1], dim)
    # A resident index already owns its all-ones filter. Allocate/copy a
    # host mask only for an explicit filter or the per-call numerical route.
    var mask = PythonObject(None) if args[2] is None else _mask(args[2], n, owner, context)
    var request = AnnSearchRequest(kind, n, dim, k, m, status_word, kind == ANN_SEARCH_CAGRA and args[2] is not None,
                                   device_codes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    if kind == ANN_SEARCH_CAGRA:
        request.graph_degree = _int(fields, "graph_degree_", owner)
        request.itopk = _int(fields, "itopk_size", owner)
        request.width = _int(fields, "search_width", owner)
        request.iterations = _int(fields, "max_iterations", owner)
        request.seeds = _int(fields, "n_seeds", owner)
        cagra_check_search(request.graph_degree, request.itopk, request.width, request.iterations, request.seeds)
        request.seed = _cagra_seed(fields)
    else:
        request.n_lists = _int(fields, "n_lists_", owner)
        request.probes = _int(fields, "n_probes", owner)
        ivf_check_search(n, request.n_lists, request.probes)
        if kind == 0:
            request.pq_dim = _int(fields, "pq_dim_", owner)
            request.pq_bits = _int(fields, "pq_bits_", owner)
            request.pq_len = _int(fields, "pq_len_", owner)
            pq_check_search(dim, request.pq_dim, request.pq_bits, request.pq_len)
        elif kind == 2:
            request.seed = _int(fields, "seed_", owner)
    var plan = ann_search_plan(request)
    var selected = plan.selected
    var arrays = _stored_all(model, fields, plan.names, plan.shapes, plan.floating, context)
    var distances = family_array(Python.tuple(m, selected), "<f4", context)
    var indices = family_array(Python.tuple(m, selected), "<i4", context)
    var candidates = family_array(Python.tuple(m), "<i4", context)
    # The binding's ANN table, read once (UNPACK); every numerical call below
    # is a native call through it.
    var ops = ann_ops_key(binding)
    var entry = ann_ops_entry(ops)
    var handle = _handle(fields, kind, binding, entry, ops.vendor != 0, arrays, plan.prepare_params, n, request.pq_dim,
                         device_codes, context)
    var q_address = q.source.addr
    var distances_address = Int(py=classical_address(distances))
    var indices_address = Int(py=classical_address(indices))
    var candidates_address = Int(py=classical_address(candidates))
    if handle is not None:
        var mask_address = 0 if mask is None else Int(py=classical_address(mask))
        var addrs = ann_handle_addresses(kind, q_address, distances_address, indices_address, candidates_address,
                                         mask_address, args[2] is not None)
        var handle_word = Int(py=handle)
        # GIL held: the resident registry (see the module note).
        _ = ann_index_search_call(entry, handle_word, addrs, plan.search_params)
    else:
        if kind != ANN_SEARCH_CAGRA and mask is None:
            mask = _mask(PythonObject(None), n, owner, context)
        var mask_address = 0 if mask is None else Int(py=classical_address(mask))
        var addrs = ann_full_addresses(kind, _addresses(arrays), mask_address, q_address, distances_address,
                                       indices_address, candidates_address)
        var operation = ANN_OP_IVF_PQ_SEARCH if kind == 0 else (ANN_OP_IVF_SQ_SEARCH if kind == 1 else (ANN_OP_IVF_RABITQ_SEARCH if kind == 2 else ANN_OP_CAGRA_SEARCH))
        with GILReleased(Python()):
            _ = ann_search_call(entry, operation, addrs, plan.full_params)
    # The queries' finite status word (0 when a cast tested them already).
    admitted_finite_refuse(Int(status[0]), owner, "queries")
    _ = status^
    if request.filtered:
        var filtered_distances = family_array(Python.tuple(m, k), "<f4", context)
        var filtered_indices = family_array(Python.tuple(m, k), "<i4", context)
        var filter_addrs: List[Int] = [distances_address, indices_address, Int(py=classical_address(mask)),
                                       Int(py=classical_address(filtered_distances)),
                                       Int(py=classical_address(filtered_indices))]
        var filter_words = ann_filter_params(m, n, selected, k)
        with GILReleased(Python()):
            _ = ann_search_call(entry, ANN_OP_FILTER_TOPK, filter_addrs, filter_words)
        # These owners are replaced below. Their raw addresses in the call's
        # argument list do not retain the original candidate buffers; keep
        # both alive until native filtering has consumed every candidate.
        _ = distances
        _ = indices
        distances = filtered_distances
        indices = filtered_indices
    if kind != ANN_SEARCH_CAGRA:
        py_setitem(fields, "n_candidates_", candidates)
    _ = q
    _ = q_pin
    _ = arrays
    _ = mask
    _ = candidates
    return parallel_public_result(Python.tuple(distances, indices), "search", PythonObject(None), context)
