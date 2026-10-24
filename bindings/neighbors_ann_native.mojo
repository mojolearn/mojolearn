# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): the
# approximate indexes' `fit` (IVFIndex in python/mojolearn/_ivf_impl.py;
# IVFPQIndex, IVFSQIndex, IVFRaBitQIndex and CagraIndex in
# python/mojolearn/_expansion_ann.py) and the function `refine`.
# Each `family_xn_ann_*` operation validates, prepares, runs the family's
# compiled kernels and returns, with the former Python method's refusals,
# operations and order, so the bits do not move: the same int parameter
# refusals (`_ann_int` / `_int_param`), the same build entry of the binding
# the shell resolved with the same address and parameter lists, the same
# fitted attributes (Arrays shaped as `Array.reshape` views over the build's
# outputs) and the same resident-handle key. The shell hands over the binding
# `_bind` (or `_backend.binding` for `refine`) resolved.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_storage_allocate
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem, py_setattr
from bindings.classical_arrays import classical_address, classical_kind, classical_is_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_validate import classical_view
from bindings.family_common import (
    FAMILY_I32P, family_repr, family_str, family_is_str, family_isinstance, family_call, family_array,
    family_as_f32, family_field, family_type_name,
)
from core.neighbors_ann_impl import ann_candidates_i32


# ---------------------------------------------------------------- shared steps


def _ann_int(owner: String, name: String, value: PythonObject) raises -> PythonObject:
    """`_expansion_ann._ann_int` (and `_ivf_impl._int_param`, the same rule
    under owner "IVFIndex"): an int that is not a bool, as `int(v)`."""
    var b = native_loaded_module("builtins")
    if family_isinstance(value, b.bool) or not family_isinstance(value, b.int):
        raise Error("TypeError: mojolearn " + owner + ": " + name + " must be an int, got " + family_type_name(value))
    return family_call(b.int, value)


def _ann_param(fields: PythonObject, owner: String, name: String) raises -> PythonObject:
    """`self._p(name)`: the constructor parameter `name` through `_ann_int`."""
    return _ann_int(owner, name, family_field(fields, name))


def _ann_empty(count: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.empty((count,), typestr)`: a zero-filled C-order Array, with
    `Array`'s refusal of a negative dimension."""
    if count < 0:
        raise Error("mojolearn: negative dimension in shape (" + String(count) + ",)")
    return family_array(Python.tuple(count), typestr, context)


def _ann_view(other: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`other.reshape(shape)` of a C-order Array (`Array._view_of`): the
    same memory under a new shape, `other` kept alive through `_base`."""
    var b = native_loaded_module("builtins")
    var rank = len(shape)
    var size = 1
    for k in range(rank):  # driver-loop: over array rank metadata
        size *= Int(py=shape[k])
    if size != Int(py=other.size):
        raise Error("mojolearn: cannot reshape size " + String(Int(py=other.size)) + " into " + family_repr(shape))
    var item = Int(py=other.itemsize)
    var steps = List[Int](length=rank, fill=0)
    var step = item
    for k in range(rank):  # driver-loop: over array rank metadata
        var axis = rank - 1 - k
        steps[axis] = step
        step *= Int(py=shape[axis])
    var strides = Python.list()
    for k in range(rank):  # driver-loop: over array rank metadata
        _ = py_call(strides.append, PythonObject(steps[k]))
    var view = native_storage_allocate(context["array_class"])
    py_setattr(view, "_store", other._store)
    py_setattr(view, "_base", other)
    py_setattr(view, "_pin", other._pin)
    py_setattr(view, "_mv", other._mv)
    py_setattr(view, "_addr", other._addr)
    py_setattr(view, "_readonly", other._readonly)
    py_setattr(view, "shape", py_call(b.tuple, shape))
    py_setattr(view, "dtype", other.dtype)
    py_setattr(view, "ndim", PythonObject(rank))
    py_setattr(view, "size", PythonObject(size))
    py_setattr(view, "itemsize", PythonObject(item))
    py_setattr(view, "nbytes", PythonObject(size * item))
    py_setattr(view, "order", PythonObject("C"))
    py_setattr(view, "strides", py_call(b.tuple, strides))
    return view


def _ann_as_array(x: PythonObject, context: PythonObject) raises -> PythonObject:
    """The float32 C-order X as the mojolearn Array `as_f32_c` returned: the
    converted copy or the caller's Array as is, or a zero-copy Array view of
    the caller's buffer (`Array.from_buffer`)."""
    if classical_is_array(x, context):
        return x
    var b = native_loaded_module("builtins")
    var pin = py_call(b.memoryview, x)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var array = native_storage_allocate(context["array_class"])
    py_setattr(array, "_store", x)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", pin)
    py_setattr(array, "_mv", py_call(py_call(pin.cast, "B").cast, "f"))
    py_setattr(array, "_addr", classical_address(x))
    py_setattr(array, "_readonly", pin.readonly)
    py_setattr(array, "shape", Python.tuple(rows, cols))
    py_setattr(array, "dtype", PythonObject("<f4"))
    py_setattr(array, "ndim", PythonObject(2))
    py_setattr(array, "size", PythonObject(rows * cols))
    py_setattr(array, "nbytes", PythonObject(rows * cols * 4))
    py_setattr(array, "itemsize", PythonObject(4))
    py_setattr(array, "strides", Python.tuple(cols * 4, 4))
    py_setattr(array, "order", PythonObject("C"))
    return array


def _ann_entry(binding: PythonObject, name: String) raises -> PythonObject:
    """`binding.<name>`: the attribute lookup the Python method made (a
    by-name refusal of a host binding is raised unchanged)."""
    return family_call(native_loaded_module("builtins").getattr, binding, PythonObject(name))


def _ann_release(fields: PythonObject, release: String) raises:
    """`_release_resident`: pop the held `(key, handle, binding)`, if any,
    and release it through the minting binding's `release` entry. Quiet on a
    binding that cannot be reached any more or a handle already gone."""
    var cached = py_call(fields.pop, PythonObject("_resident"), PythonObject(None))
    if cached is None:
        return
    try:
        _ = py_call(py_call(native_loaded_module("builtins").getattr, cached[2], PythonObject(release)), cached[1])
    except:
        pass


def _ann_key_entry(fields: PythonObject, name: String) raises -> PythonObject:
    """One element of `_resident_key`: ("device",) for an array not in the
    instance dict, else (id, address, shape) of the stored array."""
    var b = native_loaded_module("builtins")
    var a = py_get(fields, PythonObject(name), PythonObject(None))
    if a is None:
        return Python.tuple(PythonObject("device"))
    return Python.tuple(py_call(b.id, a), a._addr, py_call(b.tuple, a.shape))


# ---------------------------------------------------------------- IVFIndex


def _ivf_metric(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IVFIndex._metric_code`."""
    var b = native_loaded_module("builtins")
    var metric = family_field(fields, "metric")
    var table = context["family_xn_ann_ivf_metrics"]
    if family_is_str(metric):
        if not py_contains(table, metric):
            raise Error("mojolearn IVFIndex: metric must be one of " + family_repr(py_call(b.sorted, table))
                        + ", got " + family_repr(metric))
        return table[metric]
    if family_isinstance(metric, b.bool) or not py_contains(Python.tuple(0, 1), metric):
        raise Error("mojolearn IVFIndex: the metric codes are 0 (L2Expanded) and 1 (L2SqrtExpanded), got "
                    + family_repr(metric))
    return family_call(b.int, metric)


def ann_keep_array(source: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """What `as_f32_c(source)` returned for the converted `x`: the caller's
    own C-order float32 Array itself, else `_ann_as_array(x)`."""
    if (classical_is_array(source, context) and String(source.dtype) == "<f4"
            and String(source.order) == "C"):
        return source
    return _ann_as_array(x, context)


def _ivf_extension(fields: PythonObject, binding: PythonObject, context: PythonObject) raises:
    """`IVFIndex._extension`: the binary's own compile-time mode must be the
    one requested (`int(fn()) != _MODE_CODE.get(want)`)."""
    var b = native_loaded_module("builtins")
    var operator = Python.import_module("operator")
    var want = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    if want is None or not Bool(py=py_call(b.bool, want)):
        want = context["backend_fields"]["_DEFAULT_MODE"]
    var getter = family_call(b.getattr, binding, PythonObject("ivf_numeric_mode"), PythonObject(None))
    if getter is not None:
        var got = family_call(b.int, family_call(getter))
        var wanted = py_get(context["family_xn_ann_ivf_mode_codes"], want, PythonObject(None))
        if Bool(py=family_call(operator.ne, got, wanted)):
            raise Error("RuntimeError: mojolearn IVFIndex: numeric_mode=" + family_repr(want) + " was requested but "
                        + family_str(family_call(b.getattr, binding, PythonObject("__name__")))
                        + " reports another compile-time mode; rebuild it with bash bindings/build_ivf.sh")


def _ivf_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IVFIndex.fit(X)`: `ivf_flat_build_resident` when the binding has it
    (the list vectors stay in its handle), else `ivf_flat_build`."""
    var b = native_loaded_module("builtins")
    var binding = args[0]
    var fields = py_fields(model)
    _ = py_call(fields.pop, PythonObject("_device_list_data"), PythonObject(None))
    _ann_release(fields, "ivf_flat_index_release")
    var x = family_as_f32(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var dim = Int(py=x.shape[1])
    var n_lists_o = _ann_param(fields, "IVFIndex", "n_lists")
    var iters = _ann_param(fields, "IVFIndex", "kmeans_n_iters")
    var seed = _ann_param(fields, "IVFIndex", "random_state")
    var metric = _ivf_metric(fields, context)
    var n_lists = Int(py=n_lists_o)
    if n_lists < 1:
        raise Error("mojolearn IVFIndex: n_lists must be at least 1, got " + String(n_lists))
    var centers = _ann_empty(n_lists * dim, "<f4", context)
    var norms = _ann_empty(n_lists, "<f4", context)
    var offsets = _ann_empty(n_lists + 1, "<i4", context)
    var indices = _ann_empty(n, "<i4", context)
    _ivf_extension(fields, binding, context)
    var params = Python.list(PythonObject(n), PythonObject(dim), n_lists_o, iters, metric, seed)
    var resident_build = family_call(b.getattr, binding, PythonObject("ivf_flat_build_resident"), PythonObject(None))
    if resident_build is not None:
        # ORDER MATCHES bindings/_mojolearn_ivf.mojo (ivf_flat_build_resident):
        # x, centers_out, center_norms_out, offsets_out, indices_out;
        # n, dim, n_lists, kmeans_n_iters, metric, seed.
        var handle = family_call(b.int, family_call(
            resident_build,
            Python.list(classical_address(x), classical_address(centers), classical_address(norms),
                        classical_address(offsets), classical_address(indices)),
            params))
        _ = x  # keep the Python object alive through the native call above
        _ = centers  # keep the Python object alive through the native call above
        _ = norms  # keep the Python object alive through the native call above
        _ = offsets  # keep the Python object alive through the native call above
        _ = indices  # keep the Python object alive through the native call above
        _ = py_call(fields.pop, PythonObject("list_data_"), PythonObject(None))
        py_setitem(fields, "centers_", _ann_view(centers, Python.tuple(n_lists, dim), context))
        py_setitem(fields, "center_norms_", norms)
        py_setitem(fields, "list_offsets_", offsets)
        py_setitem(fields, "list_indices_", indices)
        py_setitem(fields, "n_features_in_", PythonObject(dim))
        py_setitem(fields, "n_rows_", PythonObject(n))
        py_setitem(fields, "n_lists_", n_lists_o)
        py_setitem(fields, "metric_code_", metric)
        py_setitem(fields, "_device_list_data", PythonObject(True))
        # `self._resident_key(False) + (id(native),)`
        var arrays = Python.tuple(_ann_key_entry(fields, "centers_"), _ann_key_entry(fields, "center_norms_"),
                                  _ann_key_entry(fields, "list_offsets_"), _ann_key_entry(fields, "list_indices_"),
                                  _ann_key_entry(fields, "list_data_"))
        var key = Python.tuple(arrays, fields["n_rows_"], fields["n_features_in_"], fields["n_lists_"],
                               fields["metric_code_"], PythonObject(False), py_call(b.id, binding))
        py_setitem(fields, "_resident", Python.tuple(key, handle, binding))
        return model
    var data = _ann_empty(n * dim, "<f4", context)
    var build = family_call(b.getattr, binding, PythonObject("ivf_flat_build"), PythonObject(None))
    if build is None:
        raise Error("ImportError: mojolearn IVFIndex: "
                    + family_str(family_call(b.getattr, binding, PythonObject("__name__"), binding))
                    + " exports no ivf_flat_build; this binary predates the build and search split, "
                    + "rebuild it with bash bindings/build_ivf.sh")
    # ORDER MATCHES bindings/ivf_index_arrays.mojo (ivf_flat_build):
    # x, centers_out, center_norms_out, offsets_out, indices_out, list_data_out.
    _ = family_call(build,
                    Python.list(classical_address(x), classical_address(centers), classical_address(norms),
                                classical_address(offsets), classical_address(indices), classical_address(data)),
                    params)
    _ = x  # keep the Python object alive through the native call above
    _ = centers  # keep the Python object alive through the native call above
    _ = norms  # keep the Python object alive through the native call above
    _ = offsets  # keep the Python object alive through the native call above
    _ = indices  # keep the Python object alive through the native call above
    _ = data  # keep the Python object alive through the native call above
    py_setitem(fields, "centers_", _ann_view(centers, Python.tuple(n_lists, dim), context))
    py_setitem(fields, "center_norms_", norms)
    py_setitem(fields, "list_offsets_", offsets)
    py_setitem(fields, "list_indices_", indices)
    py_setitem(fields, "list_data_", _ann_view(data, Python.tuple(n, dim), context))
    py_setitem(fields, "n_features_in_", PythonObject(dim))
    py_setitem(fields, "n_rows_", PythonObject(n))
    py_setitem(fields, "n_lists_", n_lists_o)
    py_setitem(fields, "metric_code_", metric)
    return model


# ---------------------------------------------------------------- IVFPQIndex


def _pq_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IVFPQIndex.fit(X)`: `x_ann_ivf_pq_build_resident` when the binding
    has it (`codes_` stays in its handle), else `x_ann_ivf_pq_build`."""
    var b = native_loaded_module("builtins")
    var binding = args[0]
    var fields = py_fields(model)
    # `self._NAME`, the class's name in `_p`'s refusals (read off the class).
    var owner = String(py_call(b.getattr, py_type(model), PythonObject("_NAME")))
    _ = py_call(fields.pop, PythonObject("_device_codes"), PythonObject(None))
    _ann_release(fields, "x_ann_index_release")
    var x = family_as_f32(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var dim = Int(py=x.shape[1])
    var n_lists_o = _ann_param(fields, owner, "n_lists")
    var pq_dim_o = _ann_param(fields, owner, "pq_dim")
    var pq_bits_o = _ann_param(fields, owner, "pq_bits")
    var n_lists = Int(py=n_lists_o)
    var pq_dim = Int(py=pq_dim_o)
    var pq_bits = Int(py=pq_bits_o)
    if not (1 <= pq_dim and pq_dim <= dim):
        raise Error("mojolearn IVFPQIndex: pq_dim must be in [1, " + String(dim) + "], got " + String(pq_dim))
    if not (1 <= pq_bits and pq_bits <= 8):
        raise Error("mojolearn IVFPQIndex: pq_bits must be in [1, 8], got " + String(pq_bits))
    if not (1 <= n_lists and n_lists <= n):
        raise Error("mojolearn IVFPQIndex: n_lists must be in [1, " + String(n) + "], got " + String(n_lists))
    # `-(-dim // pq_dim)`: the ceiling of dim / pq_dim (both positive here).
    var pq_len = (dim + pq_dim - 1) // pq_dim
    var n_codes = 1 << pq_bits
    var centers = _ann_empty(n_lists * dim, "<f4", context)
    var offsets = _ann_empty(n_lists + 1, "<i4", context)
    var indices = _ann_empty(n, "<i4", context)
    var codebooks = _ann_empty(pq_dim * n_codes * pq_len, "<f4", context)
    var resident_build = family_call(b.getattr, binding, PythonObject("x_ann_ivf_pq_build_resident"), PythonObject(None))
    if resident_build is not None:
        # x, centers, offsets, list_indices, codebooks (bindings/_mojolearn_x_ann.mojo);
        # n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters.
        var addrs = Python.list(classical_address(x), classical_address(centers), classical_address(offsets),
                                classical_address(indices), classical_address(codebooks))
        var iters = _ann_param(fields, owner, "kmeans_n_iters")
        var seed = _ann_param(fields, owner, "random_state")
        var pq_iters = _ann_param(fields, owner, "pq_kmeans_n_iters")
        var handle = family_call(b.int, family_call(
            resident_build, addrs,
            Python.list(PythonObject(n), PythonObject(dim), n_lists_o, iters, seed, pq_dim_o, pq_bits_o, pq_iters)))
        _ = x  # keep the Python object alive through the native call above
        _ = centers  # keep the Python object alive through the native call above
        _ = offsets  # keep the Python object alive through the native call above
        _ = indices  # keep the Python object alive through the native call above
        _ = codebooks  # keep the Python object alive through the native call above
        _ = py_call(fields.pop, PythonObject("codes_"), PythonObject(None))
        py_setitem(fields, "centers_", _ann_view(centers, Python.tuple(n_lists, dim), context))
        py_setitem(fields, "list_offsets_", offsets)
        py_setitem(fields, "list_indices_", indices)
        py_setitem(fields, "codebooks_", _ann_view(codebooks, Python.tuple(pq_dim, n_codes, pq_len), context))
        py_setitem(fields, "n_features_in_", PythonObject(dim))
        py_setitem(fields, "n_rows_", PythonObject(n))
        py_setitem(fields, "n_lists_", n_lists_o)
        py_setitem(fields, "pq_dim_", pq_dim_o)
        py_setitem(fields, "pq_bits_", pq_bits_o)
        py_setitem(fields, "pq_len_", PythonObject(pq_len))
        py_setitem(fields, "_device_codes", PythonObject(True))
        # `self._resident_key(native)` over `_SAVE_ARRAYS`
        var arrays = Python.tuple(_ann_key_entry(fields, "centers_"), _ann_key_entry(fields, "list_offsets_"),
                                  _ann_key_entry(fields, "list_indices_"), _ann_key_entry(fields, "codebooks_"),
                                  _ann_key_entry(fields, "codes_"))
        var key = Python.tuple(arrays, py_call(b.id, binding))
        py_setitem(fields, "_resident", Python.tuple(key, handle, binding))
        return model
    var codes = _ann_empty(n * pq_dim, "<i4", context)
    var build = _ann_entry(binding, "x_ann_ivf_pq_build")
    # x, centers, offsets, list_indices, codebooks, codes;
    # n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters.
    var addrs = Python.list(classical_address(x), classical_address(centers), classical_address(offsets),
                            classical_address(indices), classical_address(codebooks), classical_address(codes))
    var iters = _ann_param(fields, owner, "kmeans_n_iters")
    var seed = _ann_param(fields, owner, "random_state")
    var pq_iters = _ann_param(fields, owner, "pq_kmeans_n_iters")
    _ = family_call(build, addrs,
                    Python.list(PythonObject(n), PythonObject(dim), n_lists_o, iters, seed, pq_dim_o, pq_bits_o, pq_iters))
    _ = x  # keep the Python object alive through the native call above
    _ = centers  # keep the Python object alive through the native call above
    _ = offsets  # keep the Python object alive through the native call above
    _ = indices  # keep the Python object alive through the native call above
    _ = codebooks  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    py_setitem(fields, "centers_", _ann_view(centers, Python.tuple(n_lists, dim), context))
    py_setitem(fields, "list_offsets_", offsets)
    py_setitem(fields, "list_indices_", indices)
    py_setitem(fields, "codebooks_", _ann_view(codebooks, Python.tuple(pq_dim, n_codes, pq_len), context))
    py_setitem(fields, "codes_", _ann_view(codes, Python.tuple(n, pq_dim), context))
    py_setitem(fields, "n_features_in_", PythonObject(dim))
    py_setitem(fields, "n_rows_", PythonObject(n))
    py_setitem(fields, "n_lists_", n_lists_o)
    py_setitem(fields, "pq_dim_", pq_dim_o)
    py_setitem(fields, "pq_bits_", pq_bits_o)
    py_setitem(fields, "pq_len_", PythonObject(pq_len))
    return model


# ---------------------------------------------------------------- IVFSQIndex


def _sq_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IVFSQIndex.fit(X)` through `x_ann_ivf_sq_build`."""
    var binding = args[0]
    var fields = py_fields(model)
    var owner = String("IVFSQIndex")
    _ann_release(fields, "x_ann_index_release")
    var x = family_as_f32(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var dim = Int(py=x.shape[1])
    var n_lists_o = _ann_param(fields, owner, "n_lists")
    var n_lists = Int(py=n_lists_o)
    if not (1 <= n_lists and n_lists <= n):
        raise Error("mojolearn IVFSQIndex: n_lists must be in [1, " + String(n) + "], got " + String(n_lists))
    var train_per_list = PythonObject(0)  # 0: the binding's default training rows
    if family_field(fields, "max_train_points_per_cluster") is not None:
        train_per_list = _ann_param(fields, owner, "max_train_points_per_cluster")
        if Int(py=train_per_list) < 1:
            raise Error("mojolearn IVFSQIndex: max_train_points_per_cluster must be positive or None, "
                        + "got " + String(Int(py=train_per_list)))
    var centers = _ann_empty(n_lists * dim, "<f4", context)
    var offsets = _ann_empty(n_lists + 1, "<i4", context)
    var indices = _ann_empty(n, "<i4", context)
    var vmin = _ann_empty(dim, "<f4", context)
    var delta = _ann_empty(dim, "<f4", context)
    var codes = _ann_empty(n * dim, "<i4", context)
    var build = _ann_entry(binding, "x_ann_ivf_sq_build")
    # x, centers, offsets, list_indices, vmin, delta, codes;
    # n, dim, n_lists, kmeans_n_iters, seed, train_rows_per_list.
    var addrs = Python.list(classical_address(x), classical_address(centers), classical_address(offsets),
                            classical_address(indices), classical_address(vmin), classical_address(delta),
                            classical_address(codes))
    var iters = _ann_param(fields, owner, "kmeans_n_iters")
    var seed = _ann_param(fields, owner, "random_state")
    _ = family_call(build, addrs, Python.list(PythonObject(n), PythonObject(dim), n_lists_o, iters, seed, train_per_list))
    _ = x  # keep the Python object alive through the native call above
    _ = centers  # keep the Python object alive through the native call above
    _ = offsets  # keep the Python object alive through the native call above
    _ = indices  # keep the Python object alive through the native call above
    _ = vmin  # keep the Python object alive through the native call above
    _ = delta  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    py_setitem(fields, "centers_", _ann_view(centers, Python.tuple(n_lists, dim), context))
    py_setitem(fields, "list_offsets_", offsets)
    py_setitem(fields, "list_indices_", indices)
    py_setitem(fields, "sq_vmin_", vmin)
    py_setitem(fields, "sq_delta_", delta)
    py_setitem(fields, "codes_", _ann_view(codes, Python.tuple(n, dim), context))
    py_setitem(fields, "n_features_in_", PythonObject(dim))
    py_setitem(fields, "n_rows_", PythonObject(n))
    py_setitem(fields, "n_lists_", n_lists_o)
    return model


# ---------------------------------------------------------------- IVFRaBitQIndex


def _rabitq_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IVFRaBitQIndex.fit(X)` through `x_ann_ivf_rabitq_build`."""
    var binding = args[0]
    var fields = py_fields(model)
    var owner = String("IVFRaBitQIndex")
    _ann_release(fields, "x_ann_index_release")
    var x = family_as_f32(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var dim = Int(py=x.shape[1])
    var n_lists_o = _ann_param(fields, owner, "n_lists")
    var n_lists = Int(py=n_lists_o)
    if not (1 <= n_lists and n_lists <= n):
        raise Error("mojolearn IVFRaBitQIndex: n_lists must be in [1, " + String(n) + "], got " + String(n_lists))
    var D = 1
    while D < dim:  # small-loop(D: the power-of-two ceiling of the feature count, at most 63 doublings)
        D *= 2
    var words = (D + 31) // 32
    var centers = _ann_empty(n_lists * dim, "<f4", context)
    var offsets = _ann_empty(n_lists + 1, "<i4", context)
    var indices = _ann_empty(n, "<i4", context)
    var codes = _ann_empty(n * words, "<i4", context)
    var norms = _ann_empty(n, "<f4", context)
    var ips = _ann_empty(n, "<f4", context)
    var build = _ann_entry(binding, "x_ann_ivf_rabitq_build")
    # x, centers, offsets, list_indices, codes, norms, ips; n, dim, n_lists, kmeans_n_iters, seed.
    var addrs = Python.list(classical_address(x), classical_address(centers), classical_address(offsets),
                            classical_address(indices), classical_address(codes), classical_address(norms),
                            classical_address(ips))
    var iters = _ann_param(fields, owner, "kmeans_n_iters")
    var seed = _ann_param(fields, owner, "random_state")
    _ = family_call(build, addrs, Python.list(PythonObject(n), PythonObject(dim), n_lists_o, iters, seed))
    _ = x  # keep the Python object alive through the native call above
    _ = centers  # keep the Python object alive through the native call above
    _ = offsets  # keep the Python object alive through the native call above
    _ = indices  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    _ = norms  # keep the Python object alive through the native call above
    _ = ips  # keep the Python object alive through the native call above
    py_setitem(fields, "centers_", _ann_view(centers, Python.tuple(n_lists, dim), context))
    py_setitem(fields, "list_offsets_", offsets)
    py_setitem(fields, "list_indices_", indices)
    py_setitem(fields, "codes_", _ann_view(codes, Python.tuple(n, words), context))
    py_setitem(fields, "norms_", norms)
    py_setitem(fields, "ip_factors_", ips)
    py_setitem(fields, "n_features_in_", PythonObject(dim))
    py_setitem(fields, "n_rows_", PythonObject(n))
    py_setitem(fields, "n_lists_", n_lists_o)
    py_setitem(fields, "seed_", _ann_param(fields, owner, "random_state"))
    return model


# ---------------------------------------------------------------- CagraIndex


def _cagra_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`CagraIndex.fit(X)` through `x_ann_cagra_build`; the index keeps X."""
    var binding = args[0]
    var fields = py_fields(model)
    var owner = String("CagraIndex")
    _ann_release(fields, "x_ann_index_release")
    var x = family_as_f32(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    # `min(self._p("intermediate_graph_degree"), n - 1)`, then
    # `min(self._p("graph_degree"), kdeg)`
    var kdeg = min(Int(py=_ann_param(fields, owner, "intermediate_graph_degree")), n - 1)
    var deg = min(Int(py=_ann_param(fields, owner, "graph_degree")), kdeg)
    var graph = _ann_empty(n * deg, "<i4", context)
    var build = _ann_entry(binding, "x_ann_cagra_build")
    # x, graph_out; n, d, intermediate_graph_degree, graph_degree
    _ = family_call(build, Python.list(classical_address(x), classical_address(graph)),
                    Python.list(PythonObject(n), PythonObject(d), PythonObject(kdeg), PythonObject(deg)))
    _ = x  # keep the Python object alive through the native call above
    _ = graph  # keep the Python object alive through the native call above
    py_setitem(fields, "dataset_", ann_keep_array(args[1], x, context))
    py_setitem(fields, "graph_", _ann_view(graph, Python.tuple(n, deg), context))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "n_rows_", PythonObject(n))
    py_setitem(fields, "graph_degree_", PythonObject(deg))
    return model


# ---------------------------------------------------------------- refine


def _refine(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`refine(dataset, queries, candidates, k, numeric_mode, metric)`:
    args are (binding, dataset, queries, candidates, k, metric)."""
    var b = native_loaded_module("builtins")
    try:
        _ = Python.import_module("numpy")
    except:
        raise Error("ImportError: mojolearn: _expansion_ann requires optional NumPy support. "
                    + "Install it with: python -m pip install \"mojolearn[numpy]\" "
                    + "(\"mojolearn[verify]\" also includes NumPy).")
    classical_connect_numpy(context)
    var metric = args[5]
    if not py_contains(Python.tuple("sqeuclidean", "euclidean"), metric):
        raise Error("mojolearn refine: metric must be 'sqeuclidean' or 'euclidean' "
                    + "(inner product is not implemented), got " + family_repr(metric))
    var x = family_as_f32(args[1], 2, "dataset", context)
    var q = family_as_f32(args[2], 2, "queries", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var m = Int(py=q.shape[0])
    if Int(py=q.shape[1]) != d:
        raise Error("mojolearn refine: queries have " + String(Int(py=q.shape[1])) + " features, the dataset has "
                    + String(d))
    # `np.asarray(candidates)`, read natively
    var c = family_call(Python.import_module("numpy").asarray, args[3])
    var kind = classical_kind(c, context)
    if Int(py=c.ndim) != 2 or Int(py=c.shape[0]) != m or not (kind == "i" or kind == "u"):
        raise Error("mojolearn refine: candidates must be an integer array of shape (" + String(m) + ", k0)")
    var k0 = Int(py=c.shape[1])
    var k_o = _ann_int("refine", "k", args[4])
    var k = Int(py=k_o)
    var native = args[0]
    # The kernel skips every id outside [0, n) (refine_core.refine_cell); an
    # id beyond int32 is refused here, then the ids are narrowed to int32.
    var c32 = _ann_empty(m * k0, "<i4", context)
    if m * k0 > 0:
        var swap = False if classical_is_array(c, context) else not Bool(py=c.dtype.isnative)
        var fits = ann_candidates_i32(Int(py=classical_address(c)), Int(py=c.strides[0]), Int(py=c.strides[1]), m, k0,
                                      Int(py=c.itemsize), kind == "i", swap,
                                      FAMILY_I32P(unsafe_from_address=Int(py=classical_address(c32))))
        _ = c  # keep the Python object alive through the native call above
        _ = c32  # keep the Python object alive through the native call above
        if not fits:
            raise Error("mojolearn refine: a candidate index exceeds int32")
    var dist = _ann_empty(m * k, "<f4", context)
    var idx = _ann_empty(m * k, "<i4", context)
    var root = Bool(py=metric == PythonObject("euclidean"))
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(m), PythonObject(k0), k_o)
    if root:
        _ = py_call(params.append, PythonObject(1))
    var entry = _ann_entry(native, "x_ann_refine")
    # dataset, queries, candidates, out_d, out_i; n, d, m, k0, k (+ root:
    # the binding takes the euclidean roots)
    _ = family_call(entry,
                    Python.list(classical_address(x), classical_address(q), classical_address(c32),
                                classical_address(dist), classical_address(idx)),
                    params)
    _ = x  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = c32  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    _ = idx  # keep the Python object alive through the native call above
    return Python.tuple(_ann_view(dist, Python.tuple(m, k), context), _ann_view(idx, Python.tuple(m, k), context))


def neighbors_ann_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    # NumPy's connected types first, so an ndarray input is read as one
    # buffer whichever door ran first in the process.
    classical_connect_numpy(context)
    if operation == "family_xn_ann_ivf_fit":
        return _ivf_fit(model, args, context)
    if operation == "family_xn_ann_pq_fit":
        return _pq_fit(model, args, context)
    if operation == "family_xn_ann_sq_fit":
        return _sq_fit(model, args, context)
    if operation == "family_xn_ann_rabitq_fit":
        return _rabitq_fit(model, args, context)
    if operation == "family_xn_ann_cagra_fit":
        return _cagra_fit(model, args, context)
    if operation == "family_xn_ann_refine":
        return _refine(args, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)
