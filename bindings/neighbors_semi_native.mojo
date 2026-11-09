# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): LabelPropagation / LabelSpreading and the AdditiveChi2Sampler /
# SkewedChi2Sampler public methods (python/mojolearn/_expansion_neighbors.py).
# Each `family_xn_semi_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python method's refusals, operations
# and order, so the bits do not move.
#
# The shell hands over the binding `_bind()` resolved (the x_neighbors
# binding of the model's numeric mode) and the method's arguments. Every
# primitive is the binding's `xn_<name>(addresses, ints, floats)` entry the
# Python `_XNeighbors._op` called, with the same lists in the same order;
# the outputs are zero-filled mojolearn Arrays as `_buffer.empty` /
# `_lazy_out._empty_out` made them (the opt-in MOJOLEARN_XN_LAZY_OUT
# anonymous-map backing is a memory placement only: the same zeros, the
# same bits). Import-time switches the Python module read once
# (MOJOLEARN_XN_OLD_ITEMS, MOJOLEARN_XN_UNFUSED_KNN) are read from the
# context, where the module put them at its import.
from core.p3nc_scan import p3nc_dense_nnz_f32, p3nc_dense_to_csr_f32
from providers.python.x_neighbors_descriptor_pyapi import xn_admit_arguments, xn_execute, xn_module, xn_is_module
from providers.python.x_neighbors_admission_pyapi import xn_capabilities
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from providers.python.pcs_rng_pyapi import pcs_seed, _pcs_connect
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.admitted_family import admit_family
from bindings.admitted_input import admitted_finite_refuse
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_is_int, family_is_str, family_isinstance, family_call,
    family_call_kw, family_entry, family_array, family_field, family_min,
)
from providers.python.estimator_binding_pyapi import estimator_arguments
from bindings.classical_labels import classical_encode_labels, classical_decode_labels, classical_accuracy
from providers.python.family_public_lifecycle_pyapi import family_lifecycle_spec, family_lifecycle_begin, family_lifecycle_finish
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_end, neighbors_public_refusal
from core.shared_slot import SharedSlot

#: `_expansion_neighbors._U_LOG`, the unary op code of `log(a * x + b)`.
comptime _SEMI_U_LOG = 1


# ---------------------------------------------------------------- shared steps


def _semi_b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _semi_truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(_semi_b().bool, value))


def _semi_eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`, Python's own comparison."""
    return _semi_truth(family_call(native_loaded_module("operator").eq, a, b))


def _semi_int(value: PythonObject) raises -> Int:
    """`int(value)`."""
    return Int(py=family_call(_semi_b().int, value))


def _semi_float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=family_call(_semi_b().float, value))


def _semi_f32_scalar(value: Float64) -> Float64:
    """`_f32_scalar`: a Python double rounded once to float32."""
    return Float64(Float32(value))


def _semi_out(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.empty(shape, typestr)` (and `_empty_out`): a zero-filled
    C-order Array, with `_check_shape`'s negative-dimension refusal."""
    for k in range(len(shape)):  # small-loop(shape: array rank metadata): at most two dimensions
        if Int(py=shape[k]) < 0:
            raise Error("mojolearn: negative dimension in shape " + family_repr(shape))
    return family_array(shape, typestr, context)


def _semi_x(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_f32(X)`: a scipy.sparse matrix densified, then
    `as_f32_c(ndim=2, name="X")`."""
    return admit_family(source, 2, "X", context, dense=True).array


def _semi_addr(value: PythonObject) raises -> PythonObject:
    return classical_address(value)


def _semi_op(binding: PythonObject, name: String, addrs: PythonObject, ints: PythonObject, floats: PythonObject,
             context: PythonObject) raises:
    """`_XNeighbors._op`: the binding's `xn_<name>(addrs, ints, floats)`."""
    xn_execute(binding, name, addrs, ints, floats)


def _semi_numeric_mode_used(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NumericModeMixin.numeric_mode_used` on the resolved binding."""
    if not xn_is_module(binding):
        return PythonObject("identical")
    var b = _semi_b()
    var full = String(py_call(b.getattr, xn_module(binding), PythonObject("__name__")))
    var dotted = full.split(".")
    var name = String(dotted[len(dotted) - 1])
    var stem = String("mojolearn")
    if name != "_mojolearn":
        stem = String(name.removeprefix("_mojolearn_"))
    var getter = family_call(b.getattr, xn_module(binding), PythonObject(stem + "_numeric_mode"), PythonObject(None))
    if getter is not None:
        var code = family_call(getter)
        var table = context["backend_fields"]["_CODE_MODE"]
        if not py_contains(table, code):
            raise Error("RuntimeError: mojolearn: " + name + " returned unknown numeric mode " + family_repr(code))
        return table[code]
    if len(dotted) < 2:
        raise Error("IndexError: list index out of range")
    return PythonObject(String(dotted[len(dotted) - 2]))


def _semi_fast_tier(binding: PythonObject, context: PythonObject) -> Bool:
    """`_XNeighbors._fast_tier`: True on the FAST tier; any failure of the
    read-back answers False."""
    try:
        return _semi_eq(_semi_numeric_mode_used(binding, context), PythonObject("fast"))
    except:
        return False


# Perf (perf3-nbr-msel-trees): the flags are a property of the loaded
# compiled module (its numeric-mode read-back and its kfeat entry), so they
# are read once per module and mode table, not by a getattr chain per fit.
# The cache retains the module and table, so their identities are not reused.
struct _SemiKfeatCache(Defaultable, Movable):
    var module: PythonObject
    var table: PythonObject
    var flags: Int

    def __init__(out self):
        self.module = PythonObject(None)
        self.table = PythonObject(None)
        self.flags = 0


comptime _SEMI_KFEAT_CACHE = SharedSlot[_SemiKfeatCache, "MojolearnSemiKfeatFlagsV2"]


def _semi_kfeat_flags_read(binding: PythonObject, context: PythonObject) raises -> Int:
    if not _semi_fast_tier(binding, context):
        return 0
    var b = _semi_b()
    var fn_obj = family_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_kfeat_flags"), PythonObject(None))
    if fn_obj is None:
        return 0
    return _semi_int(family_call(fn_obj))


def _semi_kfeat_flags(binding: PythonObject, context: PythonObject) raises -> Int:
    """`_kfeat_flags`: the chi2 samplers' FAST + Apple fit entries compiled
    into the bound binary; 0 off the FAST tier or without the read-back."""
    if not xn_is_module(binding):
        return _semi_kfeat_flags_read(binding, context)
    var module = xn_module(binding)
    var table = PythonObject(None)
    try:
        table = context["backend_fields"]["_CODE_MODE"]
    except:
        return _semi_kfeat_flags_read(binding, context)
    var cache = _SEMI_KFEAT_CACHE.get_or_create_ptr()
    if cache[].module is not None and cache[].module is module and cache[].table is table:
        return cache[].flags
    var flags = _semi_kfeat_flags_read(binding, context)
    cache[].module = module
    cache[].table = table
    cache[].flags = flags
    return flags


def _semi_seed(seed: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_neighbors._seed_word`: an int as given; None (numpy's
    global RandomState) or an exact numpy.random.RandomState gives one draw
    in [0, 2^31), drawn natively through NumPy's BitGenerator capsule
    (providers/python/pcs_rng_pyapi.mojo `pcs_seed`: the same word as
    `randint(0, 2**31, size=1)[0]`, no RandomState method runs); anything
    else is refused before any of its methods is looked up."""
    # TOMBSTONE: P1_SEMI_SEED_RANDOMSTATE (removed: numpy's global RandomState / caller RandomState .randint drawn through Python) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
    # Restore: experiments/removed/P1_SEMI_SEED_RANDOMSTATE.patch.
    if family_is_int(seed):
        return seed
    if seed is not None:
        # Without a loaded NumPy no RandomState exists; with it, only the
        # exact class (an identity check, no attribute hook) is drawn from.
        if py_get(native_loaded_module("sys").modules, "numpy.random.mtrand") is None:
            raise Error(family_repr(seed) + " cannot be used to seed a numpy.random.RandomState instance")
        var record = py_get(context, "family_pcs_rng")
        if record is None:
            record = _pcs_connect(context)
        if py_type(seed) is not record["randomstate"]:
            raise Error(family_repr(seed) + " cannot be used to seed a numpy.random.RandomState instance")
    return pcs_seed(seed, context)


# ---------------------------------------------------------------- AdditiveChi2Sampler


def _achi2_interval(fields: PythonObject) raises -> Float64:
    """`AdditiveChi2Sampler._interval`."""
    var given = family_field(fields, "sample_interval")
    if given is not None:
        return _semi_float(given)
    var table = Python.dict()
    py_setitem(table, PythonObject(1), PythonObject(0.8))
    py_setitem(table, PythonObject(2), PythonObject(0.5))
    py_setitem(table, PythonObject(3), PythonObject(0.4))
    var steps = family_field(fields, "sample_steps")
    if not Bool(py=family_call(native_loaded_module("builtins").bool, family_call(native_loaded_module("operator").contains, table, steps))):
        raise Error("If sample_steps is not in [1, 2, 3], you need to provide sample_interval")
    return Float64(py=table[steps])


def _achi2_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`AdditiveChi2Sampler.fit(X)`."""
    var b = _semi_b()
    var fields = py_fields(model)
    var x = _semi_x(args[1], context)
    var size = Int(py=x.size)
    var first = -2
    if size != 0 and (_semi_kfeat_flags(args[0], context) & 1) != 0:
        # XN_FAST_ACHI2_DEVSCAN: the device scan; -2 is below its size gate.
        var scan = family_call(b.getattr, args[0], PythonObject("x_neighbors_kfeat_first_negative"))
        first = _semi_int(family_call(scan, _semi_addr(x), PythonObject(size)))
        _ = x  # keep the Python object alive through the native call above
    var neg = False
    if first != -2:
        neg = first >= 0
    elif size != 0:
        var lo = Float64(family_min(x))
        if lo < 0.0:
            neg = True
    if neg:
        raise Error("Negative values in data passed to AdditiveChi2Sampler")
    _ = _achi2_interval(fields)
    py_setitem(fields, "n_features_in_", PythonObject(Int(py=x.shape[1])))


def _achi2_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`AdditiveChi2Sampler.transform(X)`: the dense map; a sparse input
    gives the map's nonzeros as a csr matrix, built by SciPy as before."""
    var b = _semi_b()
    var fields = py_fields(model)
    var source = args[1]
    var sparse = (Bool(py=py_call(b.hasattr, source, PythonObject("toarray")))
                  and Bool(py=py_call(b.hasattr, source, PythonObject("nnz"))))
    var x = _semi_x(source, context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var steps = _semi_int(family_field(fields, "sample_steps"))
    var width = d * (2 * steps - 1)
    var out = _semi_out(Python.tuple(n, width), "<f4", context)
    if Int(py=x.size) != 0:
        var lo = Float64(family_min(x))
        if lo < 0.0:
            raise Error("Negative values in data passed to AdditiveChi2Sampler")
    var interval = _semi_f32_scalar(_achi2_interval(fields))
    _semi_op(args[0], "achi2", Python.list(_semi_addr(x), _semi_addr(out)),
             Python.list(PythonObject(n), PythonObject(d), PythonObject(steps)), Python.list(PythonObject(interval)),
             context)
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    if sparse:
        # p3-neighbors-cluster: the CSR arrays are built natively (count, then
        # fill: the entries SciPy's dense conversion keeps, row-major, columns
        # ascending; `eliminate_zeros` had nothing left to drop), and SciPy is
        # handed the three finished buffers. A sparse input implies SciPy is
        # already loaded (the input is its matrix), so no import runs here.
        var sp = py_get(native_loaded_module("sys").modules, "scipy.sparse")
        if sp is None:
            _ = family_call(b.__import__, PythonObject("scipy.sparse"))
            sp = native_loaded_module("sys").modules["scipy.sparse"]
        var dense = FAMILY_F32P(unsafe_from_address=Int(py=_semi_addr(out)))
        var nnz = p3nc_dense_nnz_f32(dense, n * width)
        var data = _semi_out(Python.tuple(nnz), "<f4", context)
        var indices = _semi_out(Python.tuple(nnz), "<i4", context)
        var indptr = _semi_out(Python.tuple(n + 1), "<i4", context)
        p3nc_dense_to_csr_f32(dense, n, width, FAMILY_F32P(unsafe_from_address=Int(py=_semi_addr(data))),
                              MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=_semi_addr(indices))),
                              MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=_semi_addr(indptr))))
        _ = out  # keep the Python object alive through the native passes above
        var keywords = Python.dict()
        py_setitem(keywords, "shape", Python.tuple(n, width))
        return family_call_kw(family_call(b.getattr, sp, PythonObject("csr_matrix")), keywords,
                              Python.tuple(data, indices, indptr))
    return out


# ---------------------------------------------------------------- SkewedChi2Sampler


def _schi2_set_weights(fields: PythonObject, w: PythonObject) raises:
    """The `random_weights_` setter: drop a pending z, store the weights."""
    _ = py_call(fields.pop, PythonObject("_schi2_z"), PythonObject(None))
    py_setitem(fields, "_random_weights", w)


def _schi2_weights(fields: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """The `random_weights_` getter: weights made from a pending z on the
    first read (`x_neighbors_kfeat_schi2_weights`), else the stored ones."""
    var b = _semi_b()
    var z = py_get(fields, PythonObject("_schi2_z"), PythonObject(None))
    if z is not None:
        var w = _semi_out(py_call(b.tuple, z.shape), "<f4", context)
        var entry = family_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_kfeat_schi2_weights"))
        _ = family_call(entry, _semi_addr(z), _semi_addr(w), PythonObject(Int(py=z.size)))
        _ = z  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        py_setitem(fields, "_random_weights", w)
        _ = py_call(fields.pop, PythonObject("_schi2_z"))
    if py_contains(fields, PythonObject("_random_weights")):
        return fields["_random_weights"]
    if py_contains(fields, PythonObject("random_weights_")):
        return fields["random_weights_"]
    raise Error("AttributeError: 'SkewedChi2Sampler' object has no attribute 'random_weights_'")


def schi2_public_weights(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The public `random_weights_` getter as one native call: the model's
    x_neighbors binding is resolved only when a pending z must be expanded
    (as `self._bind()` was called only then)."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var binding = PythonObject(None)
    if py_get(fields, PythonObject("_schi2_z"), PythonObject(None)) is not None:
        var reserved = Python.list(PythonObject(None))
        binding = xn_admit_arguments(estimator_arguments(model, "_mojolearn_x_neighbors", reserved, context),
                                     context)[0]
    return _schi2_weights(fields, binding, context)


def schi2_public_set_weights(model: PythonObject, value: PythonObject) raises:
    """The public `random_weights_` setter as one native call."""
    _schi2_set_weights(py_fields(model), value)


def _schi2_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`SkewedChi2Sampler.fit(X)`: the weights and offsets from the device
    counter-based draw (or the legacy-stream binding call)."""
    var b = _semi_b()
    var operator = native_loaded_module("operator")
    var fields = py_fields(model)
    var x = _semi_x(args[1], context)
    var d = Int(py=x.shape[1])
    var nc = _semi_int(family_field(fields, "n_components"))
    var seed = _semi_seed(family_field(fields, "random_state"), context)
    var usable = family_is_int(seed) and d > 0 and nc > 0
    var flags = 0
    if usable:
        flags = _semi_kfeat_flags(args[0], context)
    _ = py_call(fields.pop, PythonObject("_schi2_z"), PythonObject(None))
    var fit_idn = PythonObject(None)
    if usable:
        # Admission may return a native capability snapshot. Optional exports
        # belong to its retained module, not the snapshot object's attributes.
        fit_idn = neural_descriptor_entry(xn_module(args[0]), "x_neighbors_kfeat_schi2_fit_idn", False)
    var entry = PythonObject(None)
    if fit_idn is not None:
        entry = fit_idn
    elif (flags & 2) != 0:
        entry = neural_descriptor_entry(xn_module(args[0]), "x_neighbors_kfeat_schi2_fit")
    if entry is not None:
        var w = _semi_out(Python.tuple(d, nc), "<f4", context)
        var off = _semi_out(Python.tuple(nc), "<f4", context)
        var word = family_call(operator.and_, seed, PythonObject(0xFFFFFFFF))
        _ = family_call(entry, Python.list(word, PythonObject(d), PythonObject(nc)), _semi_addr(w), _semi_addr(off))
        _ = w  # keep the Python object alive through the native call above
        _ = off  # keep the Python object alive through the native call above
        _schi2_set_weights(fields, w)
        py_setitem(fields, "random_offset_", off)
        py_setitem(fields, "n_features_in_", PythonObject(d))
        return
    if d < 1 or nc < 1:
        raise Error("mojolearn SkewedChi2Sampler: X needs at least one feature and n_components >= 1")
    raise Error("NotImplementedError: mojolearn SkewedChi2Sampler: this binary lacks x_neighbors_kfeat_schi2_fit_idn "
                + "(rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")


def _schi2_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SkewedChi2Sampler.transform(X)`."""
    var b = _semi_b()
    var fields = py_fields(model)
    var binding = args[0]
    var x = _semi_x(args[1], context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if Int(py=x.size) != 0:
        var lo = Float64(family_min(x))
        if lo <= -_semi_float(family_field(fields, "skewedness")):
            raise Error("X may not contain entries smaller than -skewedness.")
    var nc = _semi_int(family_field(fields, "n_components"))
    var z = py_get(fields, PythonObject("_schi2_z"), PythonObject(None))
    var lazy = z is not None
    if not lazy:
        lazy = (_semi_kfeat_flags(binding, context) & 4) != 0
    if lazy and _semi_eq(PythonObject(d), py_get(fields, PythonObject("n_features_in_"), PythonObject(None))):
        # SCHI2_LAZYW: log, the pending weights, the map: one call.
        var w: PythonObject
        if z is not None:
            w = _semi_out(Python.tuple(d, nc), "<f4", context)
        else:
            w = _schi2_weights(fields, binding, context)
        var out = _semi_out(Python.tuple(n, nc), "<f4", context)
        var entry = family_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_kfeat_schi2_transform"))
        var pending = 0 if z is None else 1
        var weights_in = w if z is None else z
        var weights_out = PythonObject(0) if z is None else _semi_addr(w)
        var offset = family_field(fields, "random_offset_")
        var skew = _semi_f32_scalar(_semi_float(family_field(fields, "skewedness")))
        _ = family_call(entry, Python.list(PythonObject(n), PythonObject(d), PythonObject(nc), PythonObject(pending)),
                        Python.list(_semi_addr(x), _semi_addr(weights_in), weights_out, _semi_addr(offset),
                                    _semi_addr(out)),
                        Python.list(PythonObject(skew)))
        _ = x  # keep the Python object alive through the native call above
        _ = weights_in  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = offset  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        if z is not None:
            _schi2_set_weights(fields, w)
        return out
    # `_unary(X, _U_LOG, 1.0, skewedness)`, then the map.
    var skew = _semi_f32_scalar(_semi_float(family_field(fields, "skewedness")))
    var lx = _semi_out(py_call(b.tuple, x.shape), "<f4", context)
    _semi_op(binding, "unary", Python.list(_semi_addr(x), _semi_addr(lx)),
             Python.list(PythonObject(Int(py=x.size)), PythonObject(_SEMI_U_LOG)),
             Python.list(PythonObject(1.0), PythonObject(skew)), context)
    _ = x  # keep the Python object alive through the native call above
    var out = _semi_out(Python.tuple(n, nc), "<f4", context)
    var w = _schi2_weights(fields, binding, context)
    var offset = family_field(fields, "random_offset_")
    _semi_op(binding, "skew_transform",
             Python.list(_semi_addr(lx), _semi_addr(w), _semi_addr(offset), _semi_addr(out)),
             Python.list(PythonObject(n), PythonObject(d), PythonObject(nc)), Python.list(), context)
    _ = lx  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = offset  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- label propagation


def _semi_knn_idx(binding: PythonObject, q: PythonObject, r: PythonObject, k: Int,
                  context: PythonObject, status: Int) raises -> PythonObject:
    """`_XNeighbors._knn_sq(Q, R, k, False)`'s indices: the fused
    `knn_sq_tiled` (`knn_sq` under MOJOLEARN_XN_OLD_ITEMS), or `sqdist` then
    `knn_select` under MOJOLEARN_XN_UNFUSED_KNN. `status` (lane
    a8b-device-x-neighbors-rest): the host address of Q's finite status word,
    answered by the first op's status slot."""
    var n = Int(py=q.shape[0])
    var d = Int(py=q.shape[1])
    var m = Int(py=r.shape[0])
    if _semi_truth(context["family_xn_semi_unfused_knn"]):
        var dmat = _semi_out(Python.tuple(n, m), "<f4", context)
        _semi_op(binding, "sqdist", Python.list(_semi_addr(q), _semi_addr(r), _semi_addr(dmat)),
                 Python.list(PythonObject(n), PythonObject(m), PythonObject(d), PythonObject(status)), Python.list(),
                 context)
        _ = q  # keep the Python object alive through the native call above
        _ = r  # keep the Python object alive through the native call above
        var dist = _semi_out(Python.tuple(n, k), "<f4", context)
        var idx = _semi_out(Python.tuple(n, k), "<i4", context)
        _semi_op(binding, "knn_select", Python.list(_semi_addr(dmat), _semi_addr(dist), _semi_addr(idx)),
                 Python.list(PythonObject(n), PythonObject(m), PythonObject(k), PythonObject(0)), Python.list(), context)
        _ = dmat  # keep the Python object alive through the native call above
        _ = dist  # keep the Python object alive through the native call above
        return idx
    var dist = _semi_out(Python.tuple(n, k), "<f4", context)
    var idx = _semi_out(Python.tuple(n, k), "<i4", context)
    var name = String("knn_sq") if _semi_truth(context["family_svc_xn_old_items"]) else String("knn_sq_tiled")
    _semi_op(binding, name, Python.list(_semi_addr(q), _semi_addr(r), _semi_addr(dist), _semi_addr(idx)),
             Python.list(PythonObject(n), PythonObject(m), PythonObject(d), PythonObject(k), PythonObject(0),
                         PythonObject(status)),
             Python.list(), context)
    _ = q  # keep the Python object alive through the native call above
    _ = r  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    return idx


def _lp_compact(binding: PythonObject, idx: PythonObject, n_reference: Int, variant: Int,
                context: PythonObject) raises -> PythonObject:
    """`_compact_graph`: (cols, vals) of the compact kNN graph."""
    var n = Int(py=idx.shape[0])
    var k = Int(py=idx.shape[1])
    var cols = _semi_out(Python.tuple(n, k), "<i4", context)
    var vals = _semi_out(Python.tuple(n, k), "<f4", context)
    _semi_op(binding, "lp_knn_graph", Python.list(_semi_addr(idx), _semi_addr(cols), _semi_addr(vals)),
             Python.list(PythonObject(n), PythonObject(n_reference), PythonObject(k), PythonObject(variant)),
             Python.list(), context)
    _ = idx  # keep the Python object alive through the native call above
    _ = cols  # keep the Python object alive through the native call above
    _ = vals  # keep the Python object alive through the native call above
    return Python.tuple(cols, vals)


def _lp_knn_product(binding: PythonObject, graph: PythonObject, m: Int, labels: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`_graph_product` over the compact graph (`lp_knn_product`)."""
    var cols = graph[0]
    var vals = graph[1]
    var n = Int(py=cols.shape[0])
    var k = Int(py=cols.shape[1])
    var c = Int(py=labels.shape[1])
    var out = _semi_out(Python.tuple(n, c), "<f4", context)
    _semi_op(binding, "lp_knn_product",
             Python.list(_semi_addr(cols), _semi_addr(vals), _semi_addr(labels), _semi_addr(out)),
             Python.list(PythonObject(n), PythonObject(m), PythonObject(k), PythonObject(c)), Python.list(), context)
    _ = cols  # keep the Python object alive through the native call above
    _ = vals  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    return out


def _semi_matmul(binding: PythonObject, a: PythonObject, bm: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_XNeighbors._matmul(A, B)`."""
    var n = Int(py=a.shape[0])
    var k = Int(py=a.shape[1])
    var m = Int(py=bm.shape[1])
    var out = _semi_out(Python.tuple(n, m), "<f4", context)
    _semi_op(binding, "matmul", Python.list(_semi_addr(a), _semi_addr(bm), _semi_addr(out)),
             Python.list(PythonObject(n), PythonObject(k), PythonObject(m)), Python.list(), context)
    _ = a  # keep the Python object alive through the native call above
    _ = bm  # keep the Python object alive through the native call above
    return out


def _semi_rbf(binding: PythonObject, a: PythonObject, bq: PythonObject, gamma: PythonObject,
              context: PythonObject, status: Int) raises -> PythonObject:
    """`_XNeighbors._kernel(A, B, "rbf", gamma, 0.0, 0)`; `status`: A's
    finite status word from the kernel op's status slot (0 not asked)."""
    var n = Int(py=a.shape[0])
    var d = Int(py=a.shape[1])
    var m = Int(py=bq.shape[0])
    var out = _semi_out(Python.tuple(n, m), "<f4", context)
    var kind = context["family_svc_xn_kernels"]["rbf"]
    var g = _semi_f32_scalar(_semi_float(gamma))
    _semi_op(binding, "kernel", Python.list(_semi_addr(a), _semi_addr(bq), _semi_addr(out)),
             Python.list(PythonObject(n), PythonObject(m), PythonObject(d), PythonObject(_semi_int(kind)), PythonObject(0),
                         PythonObject(status)),
             Python.list(PythonObject(g), PythonObject(_semi_f32_scalar(0.0))), context)
    _ = a  # keep the Python object alive through the native call above
    _ = bq  # keep the Python object alive through the native call above
    return out


def _semi_row_normalize(binding: PythonObject, p: PythonObject, context: PythonObject) raises -> PythonObject:
    """`row_normalize` of a 2-D float32 matrix into a new one."""
    var b = _semi_b()
    var out = _semi_out(py_call(b.tuple, p.shape), "<f4", context)
    _semi_op(binding, "row_normalize", Python.list(_semi_addr(p), _semi_addr(out)),
             Python.list(PythonObject(Int(py=p.shape[0])), PythonObject(Int(py=p.shape[1]))), Python.list(), context)
    _ = p  # keep the Python object alive through the native call above
    return out


def _semi_class_array(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_class_array`: int or bool classes decode as ints, real classes as
    floats, anything else as the labels themselves."""
    var b = _semi_b()
    var all_int = True
    var all_real = True
    # The class tuples are built once per call, not twice per class.
    var int_kinds = Python.tuple(b.bool, b.int)
    var real_kinds = Python.tuple(b.int, b.float)
    for i in range(len(classes)):  # driver-loop: over the k class labels (type test)
        var v = classes[i]
        if all_int and not family_isinstance(v, int_kinds):
            all_int = False
        if all_real and not family_isinstance(v, real_kinds):
            all_real = False
    if all_int or all_real:
        var values = Python.list()
        for i in range(len(classes)):  # driver-loop: over the k class values
            _ = py_call(values.append, family_call(b.int if all_int else b.float, classes[i]))
        return classical_decode_labels(values, codes, context)
    return classical_decode_labels(classes, codes, context)


def _semi_argmax_labels(binding: PythonObject, classes: PythonObject, m: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """`_argmax_labels`: each row's first maximum (`row_argmax`), decoded."""
    var n = Int(py=m.shape[0])
    var c = Int(py=m.shape[1])
    var codes = _semi_out(Python.tuple(n), "<i4", context)
    if n > 0:
        _semi_op(binding, "row_argmax", Python.list(_semi_addr(m), _semi_addr(codes)),
                 Python.list(PythonObject(n), PythonObject(c)), Python.list(), context)
        _ = m  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
    return _semi_class_array(classes, codes, context)


def _semi_tol_words(fields: PythonObject) raises -> Tuple[Int, Int]:
    """`float(tol)` as its IEEE double bits, (high word, low word)."""
    var bits = bitcast[DType.uint64](Float64(_semi_float(family_field(fields, "tol"))))
    return (Int(bits >> 32), Int(bits & 0xFFFFFFFF))


def _lp_alpha_f32(fields: PythonObject, spreading: Bool) raises -> Float64:
    """`_f32_scalar(self.alpha)` for spreading, 0.0 for propagation."""
    if not spreading:
        return 0.0
    return _semi_f32_scalar(_semi_float(family_field(fields, "alpha")))


def _lp_build_graph(model: PythonObject, fields: PythonObject, binding: PythonObject, x: PythonObject,
                    spreading: Bool, context: PythonObject, status: Int) raises -> PythonObject:
    """`_build_graph` of the dense kernel: the RBF affinity, then
    LabelPropagation's row normalization or LabelSpreading's normalized
    Laplacian (the raw affinity under the classical-normalization switch)."""
    var b = _semi_b()
    var kernel = family_field(fields, "kernel")
    if not _semi_eq(kernel, PythonObject("rbf")):
        # 'knn' never reaches here (fit takes the compact graph first).
        raise Error("NotImplementedError: " + String(py_type(model).__name__) + ": kernel=" + family_repr(kernel)
                    + " is not implemented ('rbf' or 'knn')")
    var a = _semi_rbf(binding, x, x, family_field(fields, "gamma"), context, status)
    # Lane a8b-device-x-neighbors-rest: X's finite status, read with the
    # affinity the kernel op downloaded, refused before the graph is used.
    admitted_finite_refuse(Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=status)[0]),
                           String(py_type(model).__name__))
    var flag = family_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_classical_graph_normalization"), PythonObject(None))
    var raw = flag is not None and _semi_truth(family_call(flag))
    py_setitem(fields, "_classical_raw_graph", PythonObject(raw))
    if raw:
        return a
    if not spreading:
        return _semi_row_normalize(binding, a, context)
    var n = Int(py=a.shape[0])
    var g = _semi_out(Python.tuple(n, n), "<f4", context)
    if _semi_truth(context["family_svc_xn_old_items"]):
        _semi_op(binding, "ls_laplacian", Python.list(_semi_addr(a), _semi_addr(g)), Python.list(PythonObject(n)),
                 Python.list(), context)
    else:
        var deg = _semi_out(Python.tuple(n), "<f4", context)
        _semi_op(binding, "col_degree", Python.list(_semi_addr(a), _semi_addr(deg)), Python.list(PythonObject(n)),
                 Python.list(), context)
        _semi_op(binding, "ls_laplacian_deg", Python.list(_semi_addr(a), _semi_addr(deg), _semi_addr(g)),
                 Python.list(PythonObject(n)), Python.list(), context)
        _ = deg  # keep the Python object alive through the native calls above
    _ = a  # keep the Python object alive through the native calls above
    return g


def _lp_finish(fields: PythonObject, binding: PythonObject, x: PythonObject, classes: PythonObject,
               ld: PythonObject, n_iter: Int, context: PythonObject) raises:
    """`_finish_fit`: the final row normalization and the fitted state, set
    in the Python method's order."""
    var final = _semi_row_normalize(binding, ld, context)
    py_setitem(fields, "X_", x)
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "label_distributions_", final)
    py_setitem(fields, "n_iter_", PythonObject(n_iter))
    py_setitem(fields, "transduction_", _semi_argmax_labels(binding, classes, final, context))
    py_setitem(fields, "n_features_in_", PythonObject(Int(py=x.shape[1])))


def _lp_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`_LabelPropagationBase.fit(X, y)` (and `LabelSpreading.fit`'s alpha
    refusal first)."""
    var b = _semi_b()
    var binding = args[0]
    var fields = py_fields(model)
    var variant = py_call(b.getattr, py_type(model), PythonObject("_variant"))
    var spreading = not _semi_eq(variant, PythonObject("propagation"))
    if spreading:
        var alpha = _semi_float(family_field(fields, "alpha"))
        if not (0.0 < alpha and alpha < 1.0):
            raise Error("alpha must be in (0, 1)")
    var x = _semi_x(args[1], context)
    var n = Int(py=x.shape[0])
    var encoded = classical_encode_labels(args[2], context)
    var allc = encoded.classes
    var codes = encoded.codes
    if Int(py=codes.size) != n:
        raise Error("X and y have different numbers of rows")
    var skip = -1
    for i in range(len(allc)):  # driver-loop: over the k class labels (the unlabeled marker)
        var c = allc[i]
        if _semi_eq(c, PythonObject(-1)) and not family_is_str(c):
            skip = i
            break
    var classes = Python.list()
    for i in range(len(allc)):  # driver-loop: over the k class labels without the marker
        if i != skip:
            _ = py_call(classes.append, allc[i])
    var nc = len(classes)
    var ld = _semi_out(Python.tuple(n, nc), "<f4", context)
    var ys = _semi_out(Python.tuple(n, nc), "<f4", context)
    var unlabeled = _semi_out(Python.tuple(n), "<i4", context)
    var a = Float64(1.0)
    if spreading:
        a = _semi_f32_scalar(1.0 - _semi_float(family_field(fields, "alpha")))
    if n != 0 and nc != 0:
        var marker = skip if skip >= 0 else len(allc)
        _semi_op(binding, "p2m_lp_labels",
                 Python.list(_semi_addr(codes), _semi_addr(ld), _semi_addr(ys), _semi_addr(unlabeled)),
                 Python.list(PythonObject(n), PythonObject(nc), PythonObject(marker)), Python.list(PythonObject(a)),
                 context)
        _ = codes  # keep the Python object alive through the native call above
    var knn = _semi_eq(family_field(fields, "kernel"), PythonObject("knn"))
    var graph = PythonObject(None)
    var dense = PythonObject(None)
    # Lane a8b-device-x-neighbors-rest: X's finite test rides behind the
    # upload of the graph's first op (the public admission deferred it).
    var status = List[Int32](length=1, fill=Int32(0))
    if knn:
        var k = min(_semi_int(family_field(fields, "n_neighbors")), n)
        var idx = _semi_knn_idx(binding, x, x, k, context, Int(status.unsafe_ptr()))
        admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
        graph = _lp_compact(binding, idx, n, 1 if spreading else 0, context)
    else:
        dense = _lp_build_graph(model, fields, binding, x, spreading, context, Int(status.unsafe_ptr()))
    _ = status^
    var ystatic = ys
    if not spreading:
        # propagation's static rows: `lp_clamp` with ld as both inputs.
        ystatic = _semi_out(Python.tuple(n, nc), "<f4", context)
        _semi_op(binding, "lp_clamp",
                 Python.list(_semi_addr(ld), _semi_addr(ld), _semi_addr(unlabeled), _semi_addr(ystatic)),
                 Python.list(PythonObject(n), PythonObject(nc)), Python.list(), context)
    if knn:
        # The kNN-graph loop as one resident op (`lp_iterate_knn`), on every
        # binding (lane n4-mds-labelprop-nmf-loops, 2026-11-09).
        # TOMBSTONE: label-propagation compact-graph binding loop (REFACTORED: lp_iterate_knn on every vendor and tier) deleted 2026-11-09 by lane/n4-mds-labelprop-nmf-loops; code recoverable at f59d913b0b.
        # Restore: git apply -3 experiments/removed/N4_MDS_LABELPROP_NMF_HOST_LOOPS.patch; record in docs/TOMBSTONES.md.
        var cols = graph[0]
        var vals = graph[1]
        var info = _semi_out(Python.tuple(2), "<i4", context)
        var words = _semi_tol_words(fields)
        var max_iter = _semi_int(family_field(fields, "max_iter"))
        var alpha_f = _lp_alpha_f32(fields, spreading)
        _semi_op(binding, "lp_iterate_knn",
                 Python.list(_semi_addr(cols), _semi_addr(vals), _semi_addr(ld), _semi_addr(ystatic),
                             _semi_addr(unlabeled), _semi_addr(info)),
                 Python.list(PythonObject(n), PythonObject(Int(py=cols.shape[1])), PythonObject(nc), PythonObject(max_iter),
                             PythonObject(1 if spreading else 0), PythonObject(words[0]), PythonObject(words[1])),
                 Python.list(PythonObject(alpha_f)), context)
        _ = cols  # keep the Python object alive through the native call above
        _ = vals  # keep the Python object alive through the native call above
        _ = ystatic  # keep the Python object alive through the native call above
        _ = unlabeled  # keep the Python object alive through the native call above
        var resident_iter = Int(FAMILY_I32P(unsafe_from_address=Int(py=_semi_addr(info)))[0])
        _ = info  # keep the Python object alive through the read above
        _lp_finish(fields, binding, x, classes, ld, resident_iter, context)
        return
    # The dense loop as one resident op (`lp_iterate`).
    var info = _semi_out(Python.tuple(2), "<i4", context)
    var words = _semi_tol_words(fields)
    var max_iter = _semi_int(family_field(fields, "max_iter"))
    var mode = (1 if spreading else 0) + (2 if _semi_truth(py_get(fields, PythonObject("_classical_raw_graph"),
                                                                     PythonObject(False))) else 0)
    var alpha_f = _lp_alpha_f32(fields, spreading)
    _semi_op(binding, "lp_iterate",
             Python.list(_semi_addr(dense), _semi_addr(ld), _semi_addr(ystatic), _semi_addr(unlabeled),
                         _semi_addr(info)),
             Python.list(PythonObject(n), PythonObject(nc), PythonObject(max_iter), PythonObject(mode),
                         PythonObject(words[0]), PythonObject(words[1])),
             Python.list(PythonObject(alpha_f)), context)
    _ = dense  # keep the Python object alive through the native call above
    _ = ystatic  # keep the Python object alive through the native call above
    _ = unlabeled  # keep the Python object alive through the native call above
    var dense_iter = Int(FAMILY_I32P(unsafe_from_address=Int(py=_semi_addr(info)))[0])
    _ = info  # keep the Python object alive through the read above
    _lp_finish(fields, binding, x, classes, ld, dense_iter, context)


def _lp_dense_available(binding: PythonObject) raises -> Bool:
    """The binding carries `lp_dense_proba` (export on a loaded module,
    capability bit 6 on the native descriptor route)."""
    if xn_is_module(binding):
        return family_call(_semi_b().getattr, xn_module(binding), PythonObject("x_neighbors_lp_dense_proba"),
                           PythonObject(None)) is not None
    return (xn_capabilities(binding) & 64) != 0


def _lp_dense_fused(model: PythonObject, binding: PythonObject, source: PythonObject, want_codes: Bool,
                    context: PythonObject) raises -> PythonObject:
    """kernel='rbf' predict_proba (float32 (nq, C)) or predict codes (int32
    (nq,)) through `lp_dense_proba` (lane ne3-g-svc-gp-neighbors, N324): the
    kernel matrix, the product and the normalization stay on the device;
    the same kind, gamma and coef0 words `_semi_rbf` passes."""
    var fields = py_fields(model)
    var q = _semi_x(source, context)
    var xfit = family_field(fields, "X_")
    var dist = family_field(fields, "label_distributions_")
    var nq = Int(py=q.shape[0])
    var n = Int(py=xfit.shape[0])
    var c = Int(py=dist.shape[1])
    if nq == 0 or n == 0 or c == 0:
        return PythonObject(None)  # an empty side: the caller takes the op chain
    # The unused output is a one-word placeholder (the descriptor refuses a
    # null address).
    var proba = _semi_out(Python.tuple(1 if want_codes else nq, 1 if want_codes else c), "<f4", context)
    var codes = _semi_out(Python.tuple(nq if want_codes else 1), "<i4", context)
    var kind = context["family_svc_xn_kernels"]["rbf"]
    var g = _semi_f32_scalar(_semi_float(family_field(fields, "gamma")))
    # Lane a8b-device-x-neighbors-rest: the queries' finite test behind the
    # session's upload (its status slot).
    var status = List[Int32](length=1, fill=Int32(0))
    _semi_op(binding, "lp_dense_proba",
             Python.list(_semi_addr(q), _semi_addr(xfit), _semi_addr(dist), _semi_addr(proba), _semi_addr(codes)),
             Python.list(PythonObject(nq), PythonObject(n), PythonObject(Int(py=q.shape[1])), PythonObject(c),
                         PythonObject(_semi_int(kind)), PythonObject(1 if want_codes else 0),
                         PythonObject(Int(status.unsafe_ptr()))),
             Python.list(PythonObject(g), PythonObject(_semi_f32_scalar(0.0))), context)
    admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
    _ = status^
    _ = q  # keep the Python object alive through the native call above
    _ = xfit  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    return codes if want_codes else proba


def _lp_proba(model: PythonObject, binding: PythonObject, source: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """`_LabelPropagationBase.predict_proba(X)`."""
    var fields = py_fields(model)
    var knn = _semi_eq(family_field(fields, "kernel"), PythonObject("knn"))
    if not knn and _lp_dense_available(binding):
        var fused = _lp_dense_fused(model, binding, source, False, context)
        if fused is not None:
            return fused
    var q = _semi_x(source, context)
    var xfit = family_field(fields, "X_")
    var n = Int(py=xfit.shape[0])
    var p: PythonObject
    # Lane a8b-device-x-neighbors-rest: the queries' finite test behind the
    # first op's upload (its status slot).
    var status = List[Int32](length=1, fill=Int32(0))
    if knn:
        var k = min(_semi_int(family_field(fields, "n_neighbors")), n)
        var idx = _semi_knn_idx(binding, q, xfit, k, context, Int(status.unsafe_ptr()))
        admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
        var graph = _lp_compact(binding, idx, n, 2, context)
        p = _lp_knn_product(binding, graph, n, family_field(fields, "label_distributions_"), context)
    else:
        var w = _semi_rbf(binding, q, xfit, family_field(fields, "gamma"), context, Int(status.unsafe_ptr()))
        admitted_finite_refuse(Int(status[0]), String(py_type(model).__name__))
        p = _semi_matmul(binding, w, family_field(fields, "label_distributions_"), context)
    _ = status^
    return _semi_row_normalize(binding, p, context)


def _lp_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`_LabelPropagationBase.predict(X)`."""
    var fields = py_fields(model)
    if not _semi_eq(family_field(fields, "kernel"), PythonObject("knn")) and _lp_dense_available(binding):
        # Only the nq codes come back (N324); the decode is `_argmax_labels`'.
        var codes = _lp_dense_fused(model, binding, source, True, context)
        if codes is not None:
            return _semi_class_array(family_field(fields, "classes_"), codes, context)
    var proba = _lp_proba(model, binding, source, context)
    return _semi_argmax_labels(binding, family_field(py_fields(model), "classes_"), proba, context)


# ---------------------------------------------------------------- dispatch


def _neighbors_semi_operation_route(model: PythonObject, operation: String, var args: PythonObject,
                                    context: PythonObject) raises -> PythonObject:
    args = xn_admit_arguments(args, context)
    classical_connect_numpy(context)
    if operation == "family_xn_semi_achi2_fit":
        _achi2_fit(model, args, context)
        return model
    if operation == "family_xn_semi_achi2_transform":
        return _achi2_transform(model, args, context)
    if operation == "family_xn_semi_achi2_fit_transform":
        _achi2_fit(model, args, context)
        return _achi2_transform(model, args, context)
    if operation == "family_xn_semi_schi2_fit":
        _schi2_fit(model, args, context)
        return model
    if operation == "family_xn_semi_schi2_transform":
        return _schi2_transform(model, args, context)
    if operation == "family_xn_semi_schi2_fit_transform":
        _schi2_fit(model, args, context)
        return _schi2_transform(model, args, context)
    if operation == "family_xn_semi_lp_fit":
        _lp_fit(model, args, context)
        return model
    if operation == "family_xn_semi_lp_predict_proba":
        return _lp_proba(model, args[0], args[1], context)
    if operation == "family_xn_semi_lp_predict":
        return _lp_predict(model, args[0], args[1], context)
    if operation == "family_xn_semi_lp_score":
        var prediction = _lp_predict(model, args[0], args[1], context)
        return PythonObject(classical_accuracy(args[2], prediction, args[3], PythonObject(None), context))
    raise Error("RuntimeError: unknown neighbors operation " + operation)


def neighbors_semi_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """The public operation: the estimators that formerly took the `_mode`
    class-creation wrappers run their whole lifecycle around the owner
    (providers/python/family_public_lifecycle_pyapi.mojo); every other
    operation goes straight to the owner."""
    var lifecycle = family_lifecycle_spec(operation)
    if lifecycle.method.byte_length() == 0:
        return _neighbors_semi_operation_route(model, operation, args, context)
    var call = family_lifecycle_begin(model, operation, lifecycle, args, 1, context)
    var result = PythonObject(None)
    try:
        result = _neighbors_semi_operation_route(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return family_lifecycle_finish(model, lifecycle, result, call, context)
