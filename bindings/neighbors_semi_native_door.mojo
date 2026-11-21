# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): LabelPropagation / LabelSpreading and the AdditiveChi2Sampler /
# SkewedChi2Sampler public methods (python/mojolearn/_expansion_neighbors.py).
# Each `family_xn_semi_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python method's refusals, operations
# and order, so the bits do not move.
#
# The shell hands over the binding `_bind()` resolved (the x_neighbors
# binding of the model's numeric mode) and the method's arguments. Each
# operation is UNPACK (this file: Python reads, admission, the published
# arrays), RUN (neighbors/chi2_run.mojo, neighbors/label_propagation_run.mojo:
# typed requests on the image's native descriptor entry, no Python object,
# the GIL released once) and PACK (this file: refusals of deferred status
# words, publication). Import-time switches the Python module read once
# (MOJOLEARN_XN_OLD_ITEMS, MOJOLEARN_XN_UNFUSED_KNN) are read from the
# context, where the module put them at its import.
from providers.python.x_neighbors_descriptor_pyapi_door import xn_admit_arguments, xn_module, xn_is_module
from providers.python.x_neighbors_admission_pyapi_door import xn_capabilities
from providers.python.pcs_rng_pyapi_pack import pcs_seed, _pcs_connect
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.admitted_input_unpack import admitted_finite_refuse
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.py_cpython_table import cpython_table
from bindings.classical_arrays_door import classical_address, classical_wrap_array
from bindings.native_array_storage_door import native_array_unfilled
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_unpack import (
    family_repr,
    family_is_int,
    family_is_str,
    family_isinstance,
)
from bindings.family_common_pack import (
    family_call,
    family_call_kw,
    family_array,
    family_field,
)
from bindings.classical_labels_door import classical_encode_labels, classical_decode_labels, classical_accuracy
from providers.python.family_public_lifecycle_pyapi_door import family_lifecycle_begin, family_lifecycle_finish
from providers.python.family_public_lifecycle_run import family_lifecycle_spec
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_end, neighbors_public_refusal
from core.shared_slot import SharedSlot
from neighbors.neighbors_graph_run import xn_f32_word, xn_rows_match_check
from std.python._cpython import GILReleased
from bindings.lof_provider_unpack import LofConnection
from neighbors.chi2_run import (
    Achi2FitRequest, Achi2Request, Schi2FitRequest, Schi2Request, achi2_fit_run, achi2_interval_check,
    achi2_transform_run, achi2_csr_run, schi2_fit_run, schi2_check_run, schi2_transform_run,
)
from neighbors.neighbors_semi_run import (
    xn_achi2_width, xn_schi2_missing_fit, xn_semi_knn_k, xn_lp_alpha_check, xn_lp_static_weight,
)
from neighbors.label_propagation_run import LpFitRequest, LpFitOutcome, LpQueryRequest, lp_fit_run, lp_query_run


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


def _semi_out(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.empty(shape, typestr)` (and `_empty_out`): a zero-filled
    C-order Array, with `_check_shape`'s negative-dimension refusal."""
    for k in range(len(shape)):  # driver-loop: over (shape: array rank metadata): at most two dimensions
        if Int(py=shape[k]) < 0:
            raise Error("mojolearn: negative dimension in shape " + family_repr(shape))
    return family_array(shape, typestr, context)


def _semi_out_written(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A float32 C-order Array for an output its one writer stores in full
    before it is read or escapes (no zero-fill pass), with `_semi_out`'s
    negative-dimension refusal.

    Perf (lane fb12-rows, 2026-11-17; algos/additive-chi2 fit_transform and
    transform): `achi2_item` stores every one of its cell's 2 steps - 1
    outputs (zeros for a zero input), on the host route and on the device
    route's one full download (x_neighbors/device_ops.mojo op_achi2). Before:
    the zero-filled `_semi_out`, a host write over the whole map (100,000 x
    33 float32 = 13 MB on the board's taxi block; cost per output element on
    every shape) that the writer then overwrote."""
    var count = 1
    for k in range(len(shape)):  # driver-loop: over (shape: array rank metadata): at most two dimensions
        var extent = Int(py=shape[k])
        if extent < 0:
            raise Error("mojolearn: negative dimension in shape " + family_repr(shape))
        count *= extent
    return classical_wrap_array(native_array_unfilled("f", count), shape, "<f4", "C", context)


def _semi_x(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_f32(X)`: a scipy.sparse matrix densified, then
    `as_f32_c(ndim=2, name="X")`."""
    return admit_family(source, 2, "X", context, dense=True).array


def _semi_at(value: PythonObject) raises -> Int:
    return Int(py=classical_address(value))


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
#
# Door split (lane dw-neighbors-b, 2026-11-21): UNPACK reads the fields and
# admits X into a typed request, RUN is neighbors/chi2_run.mojo (no Python
# object; one GIL release around it), PACK publishes. The x_neighbors image
# is reached through its native descriptor entry (`LofConnection`: the
# family's operation-local image pin), never through a Python export.


def _achi2_interval_unpack(fields: PythonObject) raises -> Tuple[Float64, Bool]:
    """`AdditiveChi2Sampler._interval` as (interval, valid). `valid` False:
    no interval was given and sample_steps has no default one; RUN's
    `achi2_interval_check` refuses it where the public method did."""
    var given = family_field(fields, "sample_interval")
    if given is not None:
        return (_semi_float(given), True)
    var table = Python.dict()
    py_setitem(table, PythonObject(1), PythonObject(0.8))
    py_setitem(table, PythonObject(2), PythonObject(0.5))
    py_setitem(table, PythonObject(3), PythonObject(0.4))
    var steps = family_field(fields, "sample_steps")
    if not Bool(py=family_call(native_loaded_module("builtins").bool, family_call(native_loaded_module("operator").contains, table, steps))):
        return (Float64(0.0), False)
    return (Float64(py=table[steps]), True)


def _achi2_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`AdditiveChi2Sampler.fit(X)`."""
    # UNPACK
    var fields = py_fields(model)
    var x = _semi_x(args[1], context)
    var count = Int(py=x.size)
    # XN_FAST_ACHI2_DEVSCAN: the device scan; it answers -2 below its size gate.
    var scan = count != 0 and (_semi_kfeat_flags(args[0], context) & 1) != 0
    var request = Achi2FitRequest(_semi_at(x), count, scan)
    var connection = LofConnection(args[0])
    # RUN
    with GILReleased(Python()):
        achi2_fit_run(connection.driver, request)
    _ = x  # keep the Python object alive through the native call above
    # PACK
    achi2_interval_check(_achi2_interval_unpack(fields)[1])
    py_setitem(fields, "n_features_in_", PythonObject(Int(py=x.shape[1])))


def _achi2_transform(model: PythonObject, args: PythonObject, context: PythonObject,
                     checked: Bool = False) raises -> PythonObject:
    """`AdditiveChi2Sampler.transform(X)`: the dense map; a sparse input
    gives the map's nonzeros as a csr matrix, built by SciPy as before.
    `checked`: `_achi2_fit` has just refused a negative entry of this same X
    (fit_transform), so the transform's own scan is not repeated."""
    # UNPACK
    var b = _semi_b()
    var fields = py_fields(model)
    var source = args[1]
    var sparse = (Bool(py=py_call(b.hasattr, source, PythonObject("toarray")))
                  and Bool(py=py_call(b.hasattr, source, PythonObject("nnz"))))
    var x = _semi_x(source, context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var steps = _semi_int(family_field(fields, "sample_steps"))
    var width = xn_achi2_width(d, steps)
    # Perf (lane fb12-rows, 2026-11-17; algos/additive-chi2 fit_transform):
    # in fit_transform the fit's check (the device scan, or the same host
    # minimum) already refused a negative entry of this X, so the second full
    # pass over X is skipped there (`checked`). transform alone keeps it.
    var dst = _semi_out_written(Python.tuple(n, width), context)
    var interval = _achi2_interval_unpack(fields)
    var request = Achi2Request(_semi_at(x), _semi_at(dst), n, d, steps, interval[0], interval[1], checked, sparse)
    var connection = LofConnection(args[0])
    var nnz = 0
    # RUN
    with GILReleased(Python()):
        nnz = achi2_transform_run(connection.driver, request)
    _ = x  # keep the Python object alive through the native call above
    _ = dst  # keep the Python object alive through the native call above
    # PACK
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
        var data = _semi_out(Python.tuple(nnz), "<f4", context)
        var indices = _semi_out(Python.tuple(nnz), "<i4", context)
        var indptr = _semi_out(Python.tuple(n + 1), "<i4", context)
        var data_at = _semi_at(data)
        var indices_at = _semi_at(indices)
        var indptr_at = _semi_at(indptr)
        with GILReleased(Python()):
            achi2_csr_run(request.dst, n, width, data_at, indices_at, indptr_at)
        _ = dst  # keep the Python object alive through the native pass above
        var keywords = Python.dict()
        py_setitem(keywords, "shape", Python.tuple(n, width))
        return family_call_kw(family_call(b.getattr, sp, PythonObject("csr_matrix")), keywords,
                              Python.tuple(data, indices, indptr))
    return dst


# ---------------------------------------------------------------- SkewedChi2Sampler


# The pending-z paths are deleted (lane fx-leftovers item 10, 2026-11-21):
# `_schi2_z` (the FAST + Apple SCHI2_LAZYW fit's host-drawn z, expanded to
# the weights at the first read or transform) has had no producer since lane
# cpu3-python (5a2b5cecb9, 2026-10-04) made every fit take the counter-based
# draw (`schi2_fit`); the getter, the setter, the fit's pop, the transform's
# pending arm and `schi2_weights_run` existed only for it. The SCHI2_LAZYW
# one-call transform over the stored weights (kfeat bit 4) is unchanged.
# Plain deletion: no define; recoverable at this commit's parent.
def _schi2_set_weights(fields: PythonObject, w: PythonObject) raises:
    """The `random_weights_` setter: store the weights."""
    py_setitem(fields, "_random_weights", w)


def _schi2_weights(fields: PythonObject) raises -> PythonObject:
    """The `random_weights_` getter: the stored weights."""
    if py_contains(fields, PythonObject("_random_weights")):
        return fields["_random_weights"]
    if py_contains(fields, PythonObject("random_weights_")):
        return fields["random_weights_"]
    raise Error("AttributeError: 'SkewedChi2Sampler' object has no attribute 'random_weights_'")


def schi2_public_weights(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The public `random_weights_` getter as one native call."""
    classical_connect_numpy(context)
    return _schi2_weights(py_fields(model))


def schi2_public_set_weights(model: PythonObject, value: PythonObject) raises:
    """The public `random_weights_` setter as one native call."""
    _schi2_set_weights(py_fields(model), value)


def _schi2_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`SkewedChi2Sampler.fit(X)`: the weights and offsets from the
    counter-based draw of the bound image (the `schi2_fit` operation)."""
    # UNPACK
    var operator = native_loaded_module("operator")
    var fields = py_fields(model)
    var x = _semi_x(args[1], context)
    var d = Int(py=x.shape[1])
    var nc = _semi_int(family_field(fields, "n_components"))
    var seed = _semi_seed(family_field(fields, "random_state"), context)
    var usable = family_is_int(seed) and d > 0 and nc > 0
    if not usable:
        xn_schi2_missing_fit(d, nc)
        return
    var w = _semi_out(Python.tuple(d, nc), "<f4", context)
    var off = _semi_out(Python.tuple(nc), "<f4", context)
    var word = Int(py=family_call(operator.and_, seed, PythonObject(0xFFFFFFFF)))
    var request = Schi2FitRequest(word, d, nc, _semi_at(w), _semi_at(off))
    var connection = LofConnection(args[0])
    # RUN
    with GILReleased(Python()):
        schi2_fit_run(connection.driver, request)
    _ = w  # keep the Python object alive through the native call above
    _ = off  # keep the Python object alive through the native call above
    # PACK
    _schi2_set_weights(fields, w)
    py_setitem(fields, "random_offset_", off)
    py_setitem(fields, "n_features_in_", PythonObject(d))


def _schi2_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SkewedChi2Sampler.transform(X)`."""
    # UNPACK, with the entry refusal where the public method made it (before
    # n_components and the fitted arrays are read).
    var b = _semi_b()
    var fields = py_fields(model)
    var binding = args[0]
    var x = _semi_x(args[1], context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var count = Int(py=x.size)
    var x_at = _semi_at(x)
    var skew = Float64(0.0)
    if count != 0:
        skew = _semi_float(family_field(fields, "skewedness"))
        with GILReleased(Python()):
            schi2_check_run(x_at, count, skew)
    var nc = _semi_int(family_field(fields, "n_components"))
    var lazy = (_semi_kfeat_flags(binding, context) & 4) != 0
    # SCHI2_LAZYW (FAST + Apple): log and the map over the stored weights: one operation.
    lazy = lazy and _semi_eq(PythonObject(d), py_get(fields, PythonObject("n_features_in_"), PythonObject(None)))
    var w = PythonObject(None)
    var scratch = PythonObject(None)
    var scratch_at = 0
    var dst: PythonObject
    if lazy:
        w = _schi2_weights(fields)
        dst = _semi_out(Python.tuple(n, nc), "<f4", context)
    else:
        if count == 0:
            skew = _semi_float(family_field(fields, "skewedness"))
        # `_unary(X, _U_LOG, 1.0, skewedness)` into the scratch, then the map.
        scratch = _semi_out(py_call(b.tuple, x.shape), "<f4", context)
        scratch_at = _semi_at(scratch)
        dst = _semi_out(Python.tuple(n, nc), "<f4", context)
        w = _schi2_weights(fields)
    var offset = family_field(fields, "random_offset_")
    if lazy and count == 0:
        skew = _semi_float(family_field(fields, "skewedness"))
    var request = Schi2Request(x_at, _semi_at(w), 0, _semi_at(offset), scratch_at, _semi_at(dst), n, d, nc,
                               skew, lazy, False)
    var connection = LofConnection(binding)
    # RUN
    with GILReleased(Python()):
        schi2_transform_run(connection.driver, request)
    _ = x  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = scratch  # keep the Python object alive through the native call above
    _ = offset  # keep the Python object alive through the native call above
    _ = dst  # keep the Python object alive through the native call above
    return dst


# ---------------------------------------------------------------- label propagation
#
# Door split (lane dw-neighbors-b, 2026-11-21): UNPACK reads the fields,
# admits X and y and allocates the published arrays; RUN is
# neighbors/label_propagation_run.mojo (no Python object; one GIL release
# around it; every intermediate is its native scratch); PACK refuses a
# non-finite X and publishes.


def _semi_class_at(classes: PythonObject, is_list: Bool, i: Int) raises -> PythonObject:
    """`classes[i]`; a list (or list subclass) by the borrowed PyList_GetItem,
    so no subclass `__getitem__` runs."""
    if is_list:
        var table = cpython_table()
        return PythonObject(from_borrowed=table[].PyList_GetItem(classes._obj_ptr, i))
    # held in a local first: never return a subscript of a borrowed PythonObject directly (AGENTS.md, Mojo 1.2 tail call)
    var item = classes[i]
    return item^


def _semi_class_array(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_class_array`: int or bool classes decode as ints, real classes as
    floats, anything else as the labels themselves."""
    var b = _semi_b()
    var all_int = True
    var all_real = True
    # The class tuples are built once per call, not twice per class.
    var int_kinds = Python.tuple(b.bool, b.int)
    var real_kinds = Python.tuple(b.int, b.float)
    # A list `classes_` (the fit's ClassLabels, a list subclass) is read by
    # PyList_GetItem: `classes[i]` ran ClassLabels.__getitem__, a Python frame
    # per class twice per predict (census py-census-30, 2026-11-15). Its int
    # index is list.__getitem__ itself, so the values are the same objects.
    var is_list = family_isinstance(classes, b.list)
    for i in range(len(classes)):  # driver-loop: over the k class labels (type test)
        var v = _semi_class_at(classes, is_list, i)
        if all_int and not family_isinstance(v, int_kinds):
            all_int = False
        if all_real and not family_isinstance(v, real_kinds):
            all_real = False
    if all_int or all_real:
        var values = Python.list()
        for i in range(len(classes)):  # driver-loop: over the k class values
            _ = py_call(values.append, family_call(b.int if all_int else b.float, _semi_class_at(classes, is_list, i)))
        return classical_decode_labels(values, codes, context)
    return classical_decode_labels(classes, codes, context)


def _lp_alpha_f32(fields: PythonObject, spreading: Bool) raises -> Float64:
    """`_f32_scalar(self.alpha)` for spreading, 0.0 for propagation."""
    if not spreading:
        return 0.0
    return xn_f32_word(_semi_float(family_field(fields, "alpha")))


def _lp_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`_LabelPropagationBase.fit(X, y)` (and `LabelSpreading.fit`'s alpha
    refusal first)."""
    # UNPACK, with the refusals and the field reads in the public method's order
    var b = _semi_b()
    var binding = args[0]
    var fields = py_fields(model)
    var variant = py_call(b.getattr, py_type(model), PythonObject("_variant"))
    var spreading = not _semi_eq(variant, PythonObject("propagation"))
    if spreading:
        xn_lp_alpha_check(_semi_float(family_field(fields, "alpha")))
    var x = _semi_x(args[1], context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    # The label steps run on the estimator's device (labels_scope), read once.
    var label_context = labels_scope(model, context)
    var encoded = classical_encode_labels(args[2], label_context)
    var allc = encoded.classes
    var codes = encoded.codes
    xn_rows_match_check(Int(py=codes.size), n)
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
    var marker = skip if skip >= 0 else len(allc)
    var weight = Float64(1.0)
    if spreading:
        weight = xn_lp_static_weight(_semi_float(family_field(fields, "alpha")))
    var kernel = family_field(fields, "kernel")
    var knn = _semi_eq(kernel, PythonObject("knn"))
    var k = 0
    var kind = 0
    var gamma = Float64(0.0)
    var raw = False
    var unfused = False
    var old_items = False
    if knn:
        k = xn_semi_knn_k(_semi_int(family_field(fields, "n_neighbors")), n)
        unfused = _semi_truth(context["family_xn_semi_unfused_knn"])
        if not unfused:
            old_items = _semi_truth(context["family_svc_xn_old_items"])
    else:
        if not _semi_eq(kernel, PythonObject("rbf")):
            raise Error("NotImplementedError: " + String(py_type(model).__name__) + ": kernel=" + family_repr(kernel)
                        + " is not implemented ('rbf' or 'knn')")
        var kind_word = context["family_svc_xn_kernels"]["rbf"]
        gamma = _semi_float(family_field(fields, "gamma"))
        kind = _semi_int(kind_word)
        # LabelPropagation's row normalization or LabelSpreading's normalized
        # Laplacian; the raw affinity under the image's classical-normalization switch.
        var flag = family_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_classical_graph_normalization"), PythonObject(None))
        raw = flag is not None and _semi_truth(family_call(flag))
        if spreading and not raw:
            old_items = _semi_truth(context["family_svc_xn_old_items"])
    var tol = _semi_float(family_field(fields, "tol"))
    var max_iter = _semi_int(family_field(fields, "max_iter"))
    var alpha = _lp_alpha_f32(fields, spreading)
    var final = _semi_out(Python.tuple(n, nc), "<f4", context)
    var labels = _semi_out(Python.tuple(n), "<i4", context)
    var request = LpFitRequest(_semi_at(x), _semi_at(codes), _semi_at(final), _semi_at(labels), n, d, nc, marker,
                               spreading, knn, k, kind, gamma, raw, weight, alpha, tol, max_iter, unfused, old_items)
    var connection = LofConnection(binding)
    var outcome = LpFitOutcome()
    # RUN: the label step, the graph and the loop as one resident operation
    # (`lp_iterate_knn` / `lp_iterate`), on every binding (lane
    # n4-mds-labelprop-nmf-loops, 2026-11-09).
    # TOMBSTONE: label-propagation compact-graph binding loop (REFACTORED: lp_iterate_knn on every vendor and tier) deleted 2026-11-09 by lane/n4-mds-labelprop-nmf-loops; code recoverable at f59d913b0b.
    # Restore: git apply -3 experiments/removed/N4_MDS_LABELPROP_NMF_HOST_LOOPS.patch; record in docs/TOMBSTONES.md.
    with GILReleased(Python()):
        outcome = lp_fit_run(connection.driver, request)
    _ = x  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    _ = final  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    # PACK: X's finite status (read behind the upload of the graph's first
    # operation) is refused before anything is published; then the fitted
    # state in the Python method's order.
    if outcome.status != 0:
        admitted_finite_refuse(outcome.status, String(py_type(model).__name__))
    if not knn:
        py_setitem(fields, "_classical_raw_graph", PythonObject(raw))
    py_setitem(fields, "X_", x)
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "label_distributions_", final)
    py_setitem(fields, "n_iter_", PythonObject(outcome.iterations))
    py_setitem(fields, "transduction_", _semi_class_array(classes, labels, label_context))
    py_setitem(fields, "n_features_in_", PythonObject(d))


def _lp_dense_available(binding: PythonObject) raises -> Bool:
    """The binding carries `lp_dense_proba` (export on a loaded module,
    capability bit 6 on the native descriptor route)."""
    if xn_is_module(binding):
        return family_call(_semi_b().getattr, xn_module(binding), PythonObject("x_neighbors_lp_dense_proba"),
                           PythonObject(None)) is not None
    return (xn_capabilities(binding) & 64) != 0


def _lp_query(model: PythonObject, binding: PythonObject, source: PythonObject, want_labels: Bool,
              context: PythonObject) raises -> PythonObject:
    """predict_proba's float32 (nq, C) Array, or predict's int32 (nq,) class
    codes: kernel='rbf' through `lp_dense_proba` when the image carries it
    (lane ne3-g-svc-gp-neighbors, N324), else the operation chain."""
    # UNPACK
    var fields = py_fields(model)
    var knn = _semi_eq(family_field(fields, "kernel"), PythonObject("knn"))
    var fused = not knn and _lp_dense_available(binding)
    var q = _semi_x(source, context)
    var xfit = family_field(fields, "X_")
    var nq = Int(py=q.shape[0])
    var n = Int(py=xfit.shape[0])
    var k = 0
    var kind = 0
    var gamma = Float64(0.0)
    var unfused = False
    var old_items = False
    if knn:
        k = xn_semi_knn_k(_semi_int(family_field(fields, "n_neighbors")), n)
        unfused = _semi_truth(context["family_xn_semi_unfused_knn"])
        if not unfused:
            old_items = _semi_truth(context["family_svc_xn_old_items"])
    else:
        var kind_word = context["family_svc_xn_kernels"]["rbf"]
        gamma = _semi_float(family_field(fields, "gamma"))
        kind = _semi_int(kind_word)
    var dist = family_field(fields, "label_distributions_")
    var c = Int(py=dist.shape[1])
    var result: PythonObject
    var dst = 0
    var labels = 0
    if want_labels:
        result = _semi_out(Python.tuple(nq), "<i4", context)
        labels = _semi_at(result)
    else:
        result = _semi_out(Python.tuple(nq, c), "<f4", context)
        dst = _semi_at(result)
    var request = LpQueryRequest(_semi_at(q), _semi_at(xfit), _semi_at(dist), dst, labels, nq, n,
                                 Int(py=q.shape[1]), c, knn, k, kind, gamma, fused, want_labels, unfused, old_items)
    var connection = LofConnection(binding)
    var status = 0
    # RUN
    with GILReleased(Python()):
        status = lp_query_run(connection.driver, request)
    _ = q  # keep the Python object alive through the native call above
    _ = xfit  # keep the Python object alive through the native call above
    _ = dist  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    # PACK: the queries' finite status, read behind the first operation's upload
    if status != 0:
        admitted_finite_refuse(status, String(py_type(model).__name__))
    return result


def _lp_proba(model: PythonObject, binding: PythonObject, source: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """`_LabelPropagationBase.predict_proba(X)`."""
    return _lp_query(model, binding, source, False, context)


def _lp_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`_LabelPropagationBase.predict(X)`: only the class codes come back;
    the decode is `_argmax_labels`'."""
    var codes = _lp_query(model, binding, source, True, context)
    return _semi_class_array(family_field(py_fields(model), "classes_"), codes, labels_scope(model, context))


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
        return _achi2_transform(model, args, context, True)
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
        return PythonObject(classical_accuracy(args[2], prediction, args[3], PythonObject(None), labels_scope(model, context)))
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
