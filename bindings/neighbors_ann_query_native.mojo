# SPDX-License-Identifier: Apache-2.0
# Public IVFPQ/IVFSQ/IVFRaBitQ/Cagra search ownership. The caller admits the
# estimator layout and selects its import-connected x_ann binding natively.
# Input/fitted metadata is captured once here, with storage owners retained
# through prepare/search/filter/result conversion. Numerical exports still
# use the existing packed CPython ABI: this repairs the whole-operation
# Python wrapper but is NOT completion of the x_ann provider migration.
from std.memory import MutPointer
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.mode_metadata_native import _fields
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_is_array
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_view
from bindings.family_common import family_array, family_f32_snapshot, family_call_kw
from bindings.parallel_public_result import parallel_public_result
from providers.python.array_metadata_pyapi import (
    array_shape_once, array_slot_once, array_owner_once, array_metadata_once,
)
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
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
        for axis in self.shape:  # small-loop(saved array rank)
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
    if len(shape) != len(expected):
        raise Error("ValueError: invalid fitted " + name + " rank")
    for axis in range(len(shape)):  # small-loop(saved array rank)
        if shape[axis] != expected[axis]:
            raise Error("ValueError: invalid fitted " + name + " shape")
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
    var step = 4
    for offset in range(len(shape)):  # small-loop(saved array rank)
        var axis = len(shape) - offset - 1
        if py_type(strides[axis]) is not py_type(PythonObject(0)):
            raise Error("TypeError: fitted array strides must be exact native ints")
        if shape[axis] > 1 and Int(py=strides[axis]) != step:
            raise Error("ValueError: fitted " + name + " requires C-contiguous storage")
        step *= shape[axis]
    var address = array_slot_once(value, "_addr", context) if is_array else classical_address(value)
    if py_type(address) is not py_type(PythonObject(0)):
        raise Error("TypeError: fitted array address must be an exact native int")
    return _StoredArray(value, array_owner_once(value, context), shape^, Int(py=address))


def _mask(source: PythonObject, n: Int, owner: String, context: PythonObject) raises -> PythonObject:
    var result = family_array(Python.tuple(n), "<i4", context)
    var dest = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result)))
    var b = native_loaded_module("builtins")
    var refusal = "ValueError: mojolearn " + owner + ": filter must be a boolean array of shape (" + String(n) + ",), one flag per indexed row"
    if source is None:
        for i in range(n):  # cpu-route: explicit filter preparation, no numerical algorithm
            dest[i] = 1
        return result
    classical_admit_value(source, context)
    if py_type(source) is b.list or py_type(source) is b.tuple:
        if len(source) != n:
            raise Error(refusal)
        for i in range(n):  # cpu-route: explicit boolean input admission and copy
            if py_type(source[i]) is not b.bool:
                raise Error(refusal)
            dest[i] = 1 if Bool(py=source[i]) else 0
        return result
    var view = classical_view(source, context, "filter")
    var shape = array_shape_once(view, context)
    if len(shape) != 1 or shape[0] != n:
        raise Error(refusal)
    var snapshot = array_metadata_once(view, shape^, "filter", context)
    if snapshot.source.kind != CAST_KIND_BOOL or snapshot.source.itemsize != 1:
        raise Error(refusal)
    var source_bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=snapshot.source.addr)
    for i in range(n):  # cpu-route: explicit strided boolean-to-int32 filter preparation
        dest[i] = 1 if source_bytes[i * snapshot.source.s0] != 0 else 0
    _ = snapshot
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
    _ = neural_descriptor_entry(cached[2], "x_ann_index_release")
    return cached


def _key(arrays: List[_StoredArray], device_codes: Bool, binding: PythonObject) raises -> PythonObject:
    var values = Python.list()
    for i in range(len(arrays)):  # small-loop(fixed saved-array schema)
        _ = py_call(values.append, arrays[i].key())
    if device_codes:
        _ = py_call(values.append, Python.tuple("device"))
    var b = native_loaded_module("builtins")
    return Python.tuple(py_call(b.tuple, values), py_call(b.id, binding))


def _addresses(arrays: List[_StoredArray]) raises -> PythonObject:
    var result = Python.list()
    for i in range(len(arrays)):  # small-loop(fixed saved-array schema)
        _ = py_call(result.append, arrays[i].address)
    return result


def _export_codes(fields: PythonObject, cached: PythonObject, n: Int, pq_dim: Int,
                  context: PythonObject) raises -> _StoredArray:
    if cached is None:
        raise Error("RuntimeError: mojolearn IVFPQIndex: the resident index was released before its codes were read")
    var codes = family_array(Python.tuple(n, pq_dim), "<i4", context)
    var address = Int(py=classical_address(codes))
    _ = py_call(neural_descriptor_entry(cached[2], "x_ann_index_export_codes"),
                cached[1], address, n * pq_dim)
    py_setitem(fields, "codes_", codes)
    _ = py_call(fields.pop, "_device_codes", PythonObject(None))
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
    if n < 1 or pq_dim < 1:
        raise Error("ValueError: invalid fitted IVFPQIndex code dimensions")
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
    for i in range(4):  # small-loop(PQ's four non-code array keys)
        _ = py_call(arrays.append, key[0][i])
    _ = py_call(arrays.append, exported.key())
    var replacement = Python.tuple(py_call(native_loaded_module("builtins").tuple, arrays), key[1])
    py_setitem(fields, "_resident", Python.tuple(replacement, cached[1], cached[2]))
    return exported.value


def _handle(fields: PythonObject, kind: Int, binding: PythonObject,
            mut arrays: List[_StoredArray], params: PythonObject, n: Int,
            pq_dim: Int, device_codes: Bool, context: PythonObject) raises -> PythonObject:
    var cached = _resident(fields)
    var key = _key(arrays, device_codes, binding)
    var prepare = neural_descriptor_entry(binding, "x_ann_index_prepare", False)
    if prepare is not None and cached is not None and _key_equal(cached[0], key):
        return cached[1]
    if device_codes:
        arrays.append(_export_codes(fields, cached, n, pq_dim, context))
        key = _key(arrays, False, binding)
    if prepare is None:
        return PythonObject(None)
    if cached is not None:
        _ = py_call(fields.pop, "_resident", PythonObject(None))
        try:
            _ = py_call(neural_descriptor_entry(cached[2], "x_ann_index_release"), cached[1])
        except:
            pass  # Preserve the previous best-effort old-handle release.
    var handle = py_call(prepare, kind, _addresses(arrays), params)
    if py_type(handle) is not py_type(PythonObject(0)):
        raise Error("TypeError: resident ANN prepare must return an exact native handle")
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
    if Int(py=bytes[0]) >= 128:
        raise Error("ValueError: mojolearn CagraIndex: random_state must be None or >= 0")
    var word = 0
    for i in range(count):  # cpu-route: seed-word admission over exact integer bytes
        word = (word * 256 + Int(py=bytes[i])) % 2147483646
    return word + 1


def neighbors_ann_search(model: PythonObject, kind: Int, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = _fields(model)
    var binding = args[0]
    var owner = String("IVFPQIndex") if kind == 0 else (String("IVFSQIndex") if kind == 1 else (String("IVFRaBitQIndex") if kind == 2 else String("CagraIndex")))
    if kind < 0 or kind > 3:
        raise Error("RuntimeError: unknown native ANN search kind")
    var device_codes = kind == 0 and py_contains(fields, "_device_codes") and not py_contains(fields, "codes_")
    if device_codes and fields["_device_codes"] is not PythonObject(True):
        raise Error("TypeError: resident ANN codes marker must be the native true value")
    if not py_contains(fields, "graph_" if kind == 3 else "codes_") and not device_codes:
        raise Error("ValueError: mojolearn " + owner + ": call fit before search")
    var n = _int(fields, "n_rows_", owner)
    var dim = _int(fields, "n_features_in_", owner)
    var k = _int(fields, "n_neighbors", owner)
    if n < 1 or dim < 1 or k < 1:
        raise Error("ValueError: ANN rows, features and n_neighbors must be positive")
    var q = family_f32_snapshot(args[1], 2, "queries", context)
    var m = q.shape[0]
    if q.shape[1] != dim:
        raise Error("ValueError: mojolearn " + owner + ": queries have " + String(q.shape[1]) + " features, the index has " + String(dim))
    # A resident index already owns its all-ones filter. Allocate/copy a
    # host mask only for an explicit filter or the per-call numerical route.
    var mask = PythonObject(None) if args[2] is None else _mask(args[2], n, owner, context)
    var arrays = List[_StoredArray]()
    var prepare_params = Python.list()
    var search_params = Python.list()
    var full_params = Python.list()
    var n_lists = 0
    var pq_dim = 0
    var graph_degree = 0
    var selected = k
    var filtered = kind == 3 and args[2] is not None
    if kind == 3:
        graph_degree = _int(fields, "graph_degree_", owner)
        var length = _int(fields, "itopk_size", owner)
        var width = _int(fields, "search_width", owner)
        var iterations = _int(fields, "max_iterations", owner)
        var seeds = _int(fields, "n_seeds", owner)
        if graph_degree < 1 or length < 1 or width < 1 or iterations < 0 or seeds < 0:
            raise Error("ValueError: invalid CagraIndex search configuration")
        iterations = iterations if iterations != 0 else length
        seeds = min(seeds if seeds != 0 else length + width * graph_degree, n)
        var seed = _cagra_seed(fields)
        selected = length if filtered else k
        arrays.append(_stored(fields, "dataset_", [n, dim], True, context))
        arrays.append(_stored(fields, "graph_", [n, graph_degree], False, context))
        prepare_params = Python.list(n, dim, graph_degree)
        search_params = Python.list(m, selected, length, width, iterations, seeds, seed)
        full_params = Python.list(n, dim, graph_degree, m, selected, length, width, iterations, seeds, seed)
    else:
        n_lists = _int(fields, "n_lists_", owner)
        var probes = _int(fields, "n_probes", owner)
        if n_lists < 1 or n_lists > n or probes < 1 or probes > n_lists:
            raise Error("ValueError: ANN n_lists must be in [1, n_rows] and n_probes in [1, n_lists]")
        arrays.append(_stored(fields, "centers_", [n_lists, dim], True, context))
        arrays.append(_stored(fields, "list_offsets_", [n_lists + 1], False, context))
        arrays.append(_stored(fields, "list_indices_", [n], False, context))
        if kind == 0:
            pq_dim = _int(fields, "pq_dim_", owner)
            var bits = _int(fields, "pq_bits_", owner)
            var length = _int(fields, "pq_len_", owner)
            if pq_dim < 1 or pq_dim > dim or bits < 1 or bits > 8 or length != (dim + pq_dim - 1) // pq_dim:
                raise Error("ValueError: invalid fitted IVFPQIndex dimensions")
            arrays.append(_stored(fields, "codebooks_", [pq_dim, 1 << bits, length], True, context))
            if not device_codes:
                arrays.append(_stored(fields, "codes_", [n, pq_dim], False, context))
            prepare_params = Python.list(n, dim, n_lists, pq_dim, bits)
            full_params = Python.list(n, dim, n_lists, pq_dim, bits, m, k, probes)
        elif kind == 1:
            arrays.append(_stored(fields, "sq_vmin_", [dim], True, context))
            arrays.append(_stored(fields, "sq_delta_", [dim], True, context))
            arrays.append(_stored(fields, "codes_", [n, dim], False, context))
            prepare_params = Python.list(n, dim, n_lists)
            full_params = Python.list(n, dim, n_lists, m, k, probes)
        else:
            var seed = _int(fields, "seed_", owner)
            var rotated = 1
            while rotated < dim:  # small-loop(RaBitQ's existing power-of-two padded feature count)
                rotated *= 2
            arrays.append(_stored(fields, "codes_", [n, (rotated + 31) // 32], False, context))
            arrays.append(_stored(fields, "norms_", [n], True, context))
            arrays.append(_stored(fields, "ip_factors_", [n], True, context))
            prepare_params = Python.list(n, dim, n_lists, seed)
            full_params = Python.list(n, dim, n_lists, seed, m, k, probes)
        search_params = Python.list(m, k, probes)
    var distances = family_array(Python.tuple(m, selected), "<f4", context)
    var indices = family_array(Python.tuple(m, selected), "<i4", context)
    var candidates = family_array(Python.tuple(m), "<i4", context)
    var handle = _handle(fields, kind, binding, arrays, prepare_params, n, pq_dim, device_codes, context)
    var addrs = Python.list(q.source.addr, classical_address(distances), classical_address(indices))
    if handle is not None:
        if kind != 3:
            _ = py_call(addrs.append, classical_address(candidates))
            if args[2] is not None:
                _ = py_call(addrs.append, classical_address(mask))
        _ = py_call(neural_descriptor_entry(binding, "x_ann_index_search"), handle, addrs, search_params)
    else:
        if kind != 3 and mask is None:
            mask = _mask(PythonObject(None), n, owner, context)
        addrs = _addresses(arrays)
        if kind == 1 or kind == 2:
            _ = py_call(addrs.append, classical_address(mask))
        _ = py_call(addrs.append, q.source.addr)
        _ = py_call(addrs.append, classical_address(distances))
        _ = py_call(addrs.append, classical_address(indices))
        if kind != 3:
            _ = py_call(addrs.append, classical_address(candidates))
        if kind == 0:
            _ = py_call(addrs.append, classical_address(mask))
        var entry = String("x_ann_ivf_pq_search") if kind == 0 else (String("x_ann_ivf_sq_search") if kind == 1 else (String("x_ann_ivf_rabitq_search") if kind == 2 else String("x_ann_cagra_search")))
        _ = py_call(neural_descriptor_entry(binding, entry), addrs, full_params)
    if filtered:
        var filtered_distances = family_array(Python.tuple(m, k), "<f4", context)
        var filtered_indices = family_array(Python.tuple(m, k), "<i4", context)
        _ = py_call(neural_descriptor_entry(binding, "x_ann_filter_topk"),
                    Python.list(classical_address(distances), classical_address(indices), classical_address(mask),
                                classical_address(filtered_distances), classical_address(filtered_indices)),
                    Python.list(m, n, selected, k))
        # These owners are replaced below. Their raw addresses in the call's
        # argument list do not retain the original candidate buffers; keep
        # both alive until native filtering has consumed every candidate.
        _ = distances
        _ = indices
        distances = filtered_distances
        indices = filtered_indices
    if kind != 3:
        py_setitem(fields, "n_candidates_", candidates)
    _ = q
    _ = arrays
    _ = mask
    _ = candidates
    return parallel_public_result(Python.tuple(distances, indices), "search", PythonObject(None), context)
