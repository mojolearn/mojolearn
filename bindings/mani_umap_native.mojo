# SPDX-License-Identifier: Apache-2.0
# UMAP's public methods, one native call each (python/mojolearn/_umap_impl.py:
# `fit`, `fit_transform`, `transform`). This owner selects exact retained
# compiled entries in the requested or fitted mode; it validates parameters with
# `_parameters` / `_extras` / `_target`'s refusals (same wording, same order),
# builds a 'random' or 'pca' start with the decomp kit's cells exactly as
# `_initial` did (bindings/mani_kit.mojo: the same x_decomp entries, operands
# and order; the identical-mode kit, `_svd_flip_v` on the default-mode kit),
# encodes a categorical target, runs `umap_fit_transform` /
# `umap_fit_transform_ex` / `umap_transform` with the argument lists they
# always took and stores the fitted attributes the saved-model schema reads.
# The same entries on the same operands: the bits do not move.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setitem, py_contains, py_type
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.pca_exact import pca_copied
from bindings.manifold_settings_native import ManifoldSettings
from bindings.umap_parameters_native import umap_parameters_fields, umap_scalar as _umap_scalar, umap_integer as _umap_integer
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_is_str, family_type_name, family_call, family_array,
    family_as_f32, family_field, family_shape_repr,
)
from bindings.spectral_umap_routes import spectral_umap_route, spectral_umap_entry as family_entry
from core.family_cast_impl import family_all_finite_f32
from core.mani_impl import mani_same_words
from bindings.mani_kit import (
    MKit, mk_from_input, mk_r, mk_c, mk_n, mk_rows, mk_t, mk_take_cols, mk_col_range, mk_out2, mk_store,
    mk_store_ptr, mk_at, OP_SUB, OP_SCALE, OP_ADDS, OP_DIV, OP_MUL, OP_ABS, OP_ADD,
)

comptime _F32_MAX = 3.4028234663852886e+38
comptime _I64_MAX = 9223372036854775807


def _umap_mode(settings: ManifoldSettings, context: PythonObject) raises -> String:
    """`(self.numeric_mode or _backend.default_mode()).strip().lower()`."""
    var b = Python.import_module("builtins")
    var want = settings.requested
    if want is not None and py_type(want) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    if not Bool(py=py_call(b.bool, want)):
        want = settings.default_mode
    return String(py=py_call(py_call(want.strip).lower))


def _umap_check_mode(binding: PythonObject, mode: String, message: String, context: PythonObject) raises:
    """`binding.umap_numeric_mode() != {"fast": 0, "identical": 1}[mode]`."""
    if mode != "fast" and mode != "identical":
        raise Error("KeyError: " + family_repr(PythonObject(mode)))
    var got = family_call(family_entry(binding, "umap_numeric_mode", context))
    var want = 0 if mode == "fast" else 1
    if Bool(py=py_call(Python.import_module("operator").ne, got, PythonObject(want))):
        raise Error("RuntimeError: " + message)


def _umap_legacy_tail(config: PythonObject) raises -> Bool:
    """`config[8:] == [1.0, 1.0, 5]`."""
    var op = Python.import_module("operator")
    return (Bool(py=py_call(op.eq, config[8], PythonObject(1.0))) and Bool(py=py_call(op.eq, config[9], PythonObject(1.0)))
            and Bool(py=py_call(op.eq, config[10], PythonObject(5))))


def _umap_default_extras(extras: PythonObject) raises -> Bool:
    """`extras == (-1, 2.0, 0.0, 0.0)`."""
    return Bool(py=py_call(Python.import_module("operator").eq, extras,
                           Python.tuple(PythonObject(-1), PythonObject(2.0), PythonObject(0.0), PythonObject(0.0))))


def _umap_head(config: PythonObject, count: Int) raises -> PythonObject:
    """`config[:count]` as a list."""
    var out = Python.list()
    for i in range(count):  # small-loop(config: the UMAP parameter words): 11
        _ = py_call(out.append, config[i])
    return out


def _umap_concat(a: PythonObject, b: PythonObject) raises -> PythonObject:
    """`[*a, *b]`."""
    var out = py_call(Python.import_module("builtins").list, a)
    _ = py_call(out.extend, b)
    return out


def _umap_copy_f32(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """A new owned C-order float32 Array holding a copy of `value`."""
    var out = family_array(py_call(Python.import_module("builtins").tuple, value.shape), "<f4", context)
    var count = Int(py=value.size)
    if count > 0:
        memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))),
               src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(value))), count=count)
        _ = out  # keep the Python object alive through the native call above
    return out


def _umap_same_bytes(a: PythonObject, b: PythonObject, count: Int) raises -> Bool:
    """Whether two C-order float32 arrays hold the same bytes."""
    var pa = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(a)))
    var pb = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(b)))
    var _keep_ret_pa_r214 = mani_same_words(pa, pb, count)
    _ = pa  # keep the Python object alive through the native call above
    var _keep_ret_pa = _keep_ret_pa_r214^
    _ = pa  # keep the Python object alive through the native call above
    return _keep_ret_pa^


def _umap_finite(value: PythonObject) raises -> Bool:
    """`all_finite(value)` of a C-order float32 array."""
    var count = Int(py=value.size)
    if count == 0:
        return True
    return family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(value))), count)


def _umap_initial(settings: ManifoldSettings, x: PythonObject, n: Int, nc: Int, seed: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`_initial(x, n, nc, seed)`: None for 'spectral'; 'random' uniform(-10,
    10) on the Philox stream; 'pca' umap-learn's PCA of the data scaled to
    max |coordinate| 10 plus N(0, 1e-4) noise; an array as given."""
    var init = family_field(settings.values, "init")
    if family_is_str(init) and String(py=init) == "spectral":
        return PythonObject(None)
    if family_is_str(init):
        # `_Kit("identical")`, and `_host_kit()` (the default mode) for the
        # host moves and `_svd_flip_v`
        var k = MKit("identical", spectral_umap_route(context, "decomp", PythonObject("identical")),
                     context)
        var hk = MKit(String(settings.default_mode),
                      spectral_umap_route(context, "decomp", settings.default_mode), context)
        if String(py=init) == "random":
            var u = k.rand(n, nc, seed, 90, 0)
            var scaled = k.ew1(OP_SCALE, u, 20.0)
            return mk_out2(k.ew1(OP_ADDS, scaled, -10.0), context)
        # 'pca'
        var m = mk_from_input(x, "X", context)
        if nc > min(mk_r(m), mk_c(m)):
            raise Error("UMAP init='pca' needs n_components <= min(n_samples, n_features)")
        # k.center(M, k.colmean(M))
        var sums = k.colsum(m)
        var mean = k.ew1(OP_SCALE, sums, 1.0 / Float64(mk_r(m)))
        var xc = k.ew2(OP_SUB, m, mean)
        var vt: PythonObject
        if mk_r(m) >= mk_c(m):
            var got = k.svd(xc, hk)
            vt = got[1]
        else:
            # wide data: the SVD of Xc^T gives Xc's left vectors U (its Vt
            # rows); Xc's right vectors are then U^T Xc / S
            var xct = mk_t(xc, hk, context)
            var got = k.svd(xct, hk)
            var s = got[0]
            var ut = got[1]
            var store = mk_store(s, context)
            var sp = mk_store_ptr(store)
            _ = store  # keep the Python object alive through the native call above
            for j in range(nc):  # small-loop(components: the zero singular value check): bounded by n_components
                if not (Float64(sp[j]) > 0):
                    raise Error("UMAP init='pca' found a zero singular value")
                _ = store  # keep the Python object alive through the native calls above
            _ = store^
            var head = mk_rows(ut, 0, nc, context)
            vt = k.mm(head, xc, False, False)
            var kept = mk_take_cols(s, mk_col_range(0, nc), context)
            vt = k.ew2(OP_DIV, vt, mk_t(kept, hk, context))
        # _svd_flip_v(Vt.rows(0, nc)) on the default-mode kit
        var top = mk_rows(vt, 0, nc, context)
        var signs = hk.absmax_signs(top, False)
        vt = hk.ew2(OP_MUL, top, signs)
        var coords = k.mm(xc, vt, False, True)
        # max |coords| in Mojo: the positions of the minimum of -|coords|
        var mag = k.ew1(OP_ABS, coords, 0.0)
        var at = k.argmin_all(k.ew1(OP_SCALE, mag, -1.0))
        var peak = 0.0
        if mk_r(at) > 0:
            var where = Int(mk_at(at, 0, context))
            peak = mk_at(mag, where, context)
        if not (peak > 0):
            raise Error("UMAP init='pca' found a zero spread")
        coords = k.ew1(OP_SCALE, coords, 10.0 / peak)
        var noise = k.ew1(OP_SCALE, k.rand(n, nc, seed, 91, 1), 1e-4)
        return mk_out2(k.ew2(OP_ADD, coords, noise), context)
    var a = family_as_f32(init, 2, "init", context)
    if Int(py=a.shape[0]) != n or Int(py=a.shape[1]) != nc:
        raise Error("UMAP init array has shape " + family_shape_repr(a) + "; (" + String(n) + ", " + String(nc)
                    + ") is needed")
    if not _umap_finite(a):
        raise Error("UMAP init array must be finite")
    return a


def _umap_target(settings: ManifoldSettings, y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_target(y, n)`: (kind, dims, codes, target_n_neighbors, weight)."""
    var b = Python.import_module("builtins")
    var op = Python.import_module("operator")
    var fields = settings.values
    var tm = family_field(fields, "target_metric")
    var w = _umap_scalar(family_field(fields, "target_weight"), "target_weight", context)
    if not (w >= 0 and w <= 1):
        raise Error("UMAP target_weight must be in [0, 1]")
    var tk = _umap_integer(family_field(fields, "target_n_neighbors"), "target_n_neighbors", -1, _I64_MAX, context)
    if Bool(py=py_call(op.contains, Python.tuple(PythonObject(0), PythonObject(1)), tk)):
        raise Error("UMAP target_n_neighbors must be -1 or >= 2")
    var tk_kept = py_call(b.max, tk, PythonObject(0))
    if Bool(py=py_call(b.bool, py_call(op.eq, tm, PythonObject("categorical")))):
        var vals: PythonObject
        if Bool(py=py_call(b.hasattr, y, PythonObject("tolist"))):
            vals = family_call(py_call(b.getattr, y, PythonObject("tolist")))
        else:
            vals = py_call(b.list, y)
        var count = Int(py=py_call(b.len, vals))
        var nested = False
        var seq_types = Python.tuple(b.list, b.tuple)
        for i in range(count):  # cpu-route: categorical y label objects ordered by (type, value)
            if Bool(py=py_call(b.isinstance, vals[i], seq_types)):
                nested = True
                break
        if count != n or nested:
            raise Error("UMAP categorical y must be one label per row")
        var unknown = PythonObject(-1)
        var labels = py_call(b.set)
        for i in range(count):  # cpu-route: categorical y label objects ordered by (type, value)
            if Bool(py=py_call(b.bool, py_call(op.ne, vals[i], unknown))):
                _ = py_call(labels.add, vals[i])
        var keyed = Python.list()
        for v in labels:  # cpu-route: categorical y label objects ordered by (type, value)
            _ = py_call(keyed.append, Python.tuple(py_call(b.str, py_type(v)), v))
        var order = py_call(b.sorted, keyed)
        var code = Python.dict()
        for i in range(Int(py=py_call(b.len, order))):  # cpu-route: categorical y label objects ordered by (type, value)
            py_setitem(code, order[i][1], PythonObject(Float64(i)))
        var t = family_array(Python.tuple(n), "<f4", context)
        var tp = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(t)))
        _ = t  # keep the Python object alive through the native call above
        for i in range(count):  # cpu-route: categorical y label objects ordered by (type, value)
            if Bool(py=py_call(b.bool, py_call(op.eq, vals[i], unknown))):
                tp[i] = Float32(-1.0)
                _ = t  # keep the Python object alive through the native call above
            else:
                tp[i] = Float32(Float64(py=code[vals[i]]))
                _ = t  # keep the Python object alive through the native call above
        return Python.tuple(PythonObject(1), PythonObject(1), t, tk_kept, PythonObject(w))
    if Bool(py=py_call(op.contains, Python.tuple(PythonObject("l2"), PythonObject("euclidean")), tm)):
        var t = family_as_f32(y, -1, "y", context)
        var nd = Int(py=t.ndim)
        var dims = 1
        var rows = -1
        if nd == 1:
            rows = Int(py=t.shape[0])
        elif nd == 2:
            rows = Int(py=t.shape[0])
            dims = Int(py=t.shape[1])
        if (nd != 1 and nd != 2) or rows != n or not _umap_finite(t):
            raise Error("UMAP continuous y must be finite with one row per sample")
        return Python.tuple(PythonObject(2), PythonObject(dims), t, tk_kept, PythonObject(w))
    raise Error("NotImplementedError: UMAP target_metric=" + family_repr(tm) + " is not implemented; "
                + "'categorical' and 'l2' / 'euclidean' are")


def _umap_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`UMAP.fit(X, y)`; returns the embedding."""
    classical_connect_numpy(context)
    var binding = spectral_umap_route(context, "metrics", settings.mode)
    var source = args[0]
    var y = args[1]
    var validated = umap_parameters_fields(settings.values, context)
    var config = validated[0]
    var extras = validated[1]
    var x = family_as_f32(source, 2, "X", context)
    var copied = pca_copied(source, context)
    if not _umap_finite(x):
        raise Error("UMAP input coordinates must be finite")
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var nc = Int(py=config[1])
    if Bool(py=py_call(Python.import_module("operator").gt, config[0], PythonObject(n))):
        raise Error("UMAP n_neighbors exceeds n_samples")
    var initial = _umap_initial(settings, x, n, nc, config[7], context)
    if initial is None and n < 2 * nc + 4:
        raise Error("UMAP has too few samples for spectral initialization")
    var target = PythonObject(None)
    if not (y is None):
        target = _umap_target(settings, y, n, context)
    var embedding = family_array(Python.tuple(n, nc), "<f4", context)
    var mode = _umap_mode(settings, context)
    _umap_check_mode(binding, mode, "UMAP binary numeric mode disagrees with requested mode", context)
    var legacy = initial is None and target is None and _umap_default_extras(extras) and (nc == 2 or nc == 3)
    var columns: PythonObject
    if legacy:
        # Preserve the legacy ABI for default controls and existing wheels.
        var native_config = _umap_head(config, 8) if _umap_legacy_tail(config) else config
        var params = _umap_concat(Python.list(PythonObject(n), PythonObject(d)), native_config)
        columns = family_call(family_entry(binding, "umap_fit_transform", context), classical_address(x),
                              classical_address(embedding), params)
        _ = x  # keep the Python object alive through the native call above
        _ = embedding  # keep the Python object alive through the native call above
    else:
        var tkind = PythonObject(0)
        var tdims = PythonObject(1)
        var tarr = PythonObject(None)
        var tk = PythonObject(0)
        var tw = PythonObject(0.5)
        if not (target is None):
            tkind = target[0]
            tdims = target[1]
            tarr = target[2]
            tk = target[3]
            tw = target[4]
        var addrs = Python.list(classical_address(x), classical_address(embedding),
                                PythonObject(0) if initial is None else classical_address(initial),
                                PythonObject(0) if tarr is None else classical_address(tarr))
        _ = embedding  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = initial  # keep the Python object alive through the native call above
        _ = tarr  # keep the Python object alive through the native call above
        var params = _umap_concat(_umap_concat(Python.list(PythonObject(n), PythonObject(d)), config), extras)
        _ = py_call(params.extend, Python.list(tkind, tdims, tk, tw))
        columns = family_call(family_entry(binding, "umap_fit_transform_ex", context), addrs, params)
        _ = tarr  # keep the Python object alive through the native call above
        _ = embedding  # keep the Python object alive through the native call above
        _ = initial  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    _ = initial^
    _ = target^
    if Bool(py=py_call(Python.import_module("operator").ne, columns, PythonObject(nc))) or not _umap_finite(embedding):
        raise Error("RuntimeError: UMAP returned an invalid embedding")
    # Prepare all retained state before publishing a successful fit. Copies
    # prevent caller edits of X or embedding_ from changing transform's model.
    var training = _umap_copy_f32(x, context)
    var frozen = _umap_copy_f32(embedding, context)
    _ = x^
    py_setitem(settings.fields, "embedding_", embedding)
    py_setitem(settings.fields, "n_features_in_", PythonObject(d))
    py_setitem(settings.fields, "input_copied_", PythonObject(copied))
    py_setitem(settings.fields, "_transform_training", training)
    py_setitem(settings.fields, "_transform_embedding", frozen)
    py_setitem(settings.fields, "_transform_config", _umap_concat_tuple(config, extras))
    py_setitem(settings.fields, "_transform_mode", PythonObject(mode))
    return embedding


def _umap_concat_tuple(config: PythonObject, extras: PythonObject) raises -> PythonObject:
    """`tuple(config) + tuple(extras)`."""
    var b = Python.import_module("builtins")
    return py_call(Python.import_module("operator").add, py_call(b.tuple, config), py_call(b.tuple, extras))


def _umap_transform(model: PythonObject, settings: ManifoldSettings, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`UMAP.transform(X)`."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var op = Python.import_module("operator")
    if not py_contains(settings.values, "_transform_training"):
        raise Error("UMAP transform requires a successful fit")
    var validated = umap_parameters_fields(settings.values, context)
    var config = validated[0]
    var extras = validated[1]
    var mode = _umap_mode(settings, context)
    var fitted_config = family_field(settings.values, "_transform_config")
    var fitted_mode = family_field(settings.values, "_transform_mode")
    if (Bool(py=py_call(b.bool, py_call(op.ne, _umap_concat_tuple(config, extras), fitted_config)))
            or Bool(py=py_call(b.bool, py_call(op.ne, PythonObject(mode), fitted_mode)))):
        raise Error("UMAP parameters or numeric mode changed after fit; refit before transform")
    # The fitted route is selected only after the live-mode/refit refusal,
    # so changing a fitted identical model to unavailable fast still says refit.
    var binding = spectral_umap_route(context, "metrics", fitted_mode)
    var x = family_as_f32(args[0], 2, "X", context)
    var training = family_field(settings.values, "_transform_training")
    var fitted_embedding = family_field(settings.values, "_transform_embedding")
    var nq = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var nt = Int(py=training.shape[0])
    if d != Int(py=training.shape[1]):
        raise Error("UMAP transform feature count differs from fitted data")
    if not _umap_finite(x):
        raise Error("UMAP transform input coordinates must be finite")
    # the C-order bytes compared (`x.tobytes() == training.tobytes()`)
    if nq == nt and nq * d > 0:
        if _umap_same_bytes(x, training, nq * d):
            return _umap_copy_f32(fitted_embedding, context)
    elif nq == nt:
        return _umap_copy_f32(fitted_embedding, context)
    _umap_check_mode(binding, mode, "UMAP binary numeric mode disagrees with fitted mode", context)
    var nc = Int(py=config[1])
    var output = family_array(Python.tuple(nq, nc), "<f4", context)
    var native_config = _umap_head(config, 8) if _umap_legacy_tail(config) else config
    if not _umap_default_extras(extras):
        native_config = _umap_concat(config, extras)
    var params = _umap_concat(Python.list(PythonObject(nt), PythonObject(nq), PythonObject(d)), native_config)
    var columns = family_call(family_entry(binding, "umap_transform", context),
                              Python.list(classical_address(training), classical_address(fitted_embedding),
                                          classical_address(x), classical_address(output)), params)
    _ = fitted_embedding  # keep the Python object alive through the native call above
    _ = training  # keep the Python object alive through the native call above
    _ = output  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    _ = training^
    _ = fitted_embedding^
    if Bool(py=py_call(op.ne, columns, PythonObject(nc))) or not _umap_finite(output):
        raise Error("RuntimeError: UMAP transform returned an invalid embedding")
    return output


def umap_operation(model: PythonObject, settings: ManifoldSettings, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """UMAP: fit / fit_transform args = (X, y); transform args = (X,)."""
    if operation == "family_mani_umap_fit":
        _ = _umap_fit(model, settings, args, context)
        return model
    if operation == "family_mani_umap_fit_transform":
        return _umap_fit(model, settings, args, context)
    if operation == "family_mani_umap_transform":
        return _umap_transform(model, settings, args, context)
    raise Error("RuntimeError: unknown UMAP operation " + operation)
