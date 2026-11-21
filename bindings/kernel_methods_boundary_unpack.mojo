# SPDX-License-Identifier: Apache-2.0
# UNPACK of the kernel methods door (bindings/kernel_methods_native_door.mojo):
# KernelRidge, Nystroem and RBFSampler. Everything here reads Python objects
# and answers typed values: the binding route of the call (the provider of
# an identical module, or the list entries of a module without one), every
# parameter refusal of the public methods in their wording and order
# (kernel, degree, gamma, coef0, alpha, the chi2 kernels' negative input, the
# precomputed square, n_components, the seed), the fitted-state checks, and
# the pins of every buffer whose address RUN reads
# (kernel_methods/native_call_run.mojo).
from core.mode_answer import MODE_FAMILY_KERNEL, answer_mode_word
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.admitted_family_door import admit_family_meta, admit_family_weights
from bindings.native_caller_frame_pack import native_frame_globals
from bindings.native_loaded_module_door import native_loaded_module
from bindings.trees_side_records_unpack import trees_side_get, trees_side_put
from bindings.classical_validate_unpack import classical_view
from bindings.classical_warnings_unpack import classical_warning
from bindings.metrics_native_support_door import metric_float, metric_is_real, metric_is_bool
from bindings.host_inference_scalars_door import host_inference_primitive_number
from core.cast_in import CAST_KIND_SIGNED
from core.host_finite_scan import host_all_finite_f32_strided
from bindings.admitted_input_unpack import admitted_finite_refuse
from providers.python.kernel_public_lifecycle_pyapi_unpack import kernel_public_finite, kernel_public_column_target
from providers.python.kernel_method_descriptor_pyapi_door import kernel_method_provider, kernel_method_metadata
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry, neural_descriptor_provider_id
from bindings.py_balanced import py_type, py_get, py_contains, py_fields, py_item
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_shape_once, array_metadata_once
from bindings.family_common_unpack import family_repr, family_snapshot_min
from bindings.family_common_pack import family_call, family_entry, family_field
from bindings.svm_keys_unpack import svm_keys
from kernel_methods.native_call_run import KmKernel

comptime _KM_MODE_SLOT = "p3_km_binding_mode_checked_v1"
comptime KM_KERNEL_PRECOMPUTED = 4
comptime KM_KERNEL_CHI2 = 7
comptime KM_KERNEL_ADDITIVE_CHI2 = 8
comptime _PY2MOJO_SCALE_GAMMA = 1


struct KmRoute(Movable):
    """The binding route of one operation: the module the estimator's device
    resolved, its provider (None for a module without one: `fast`), the
    provider's flag word, and the provider id word RUN executes on (0 for a
    module without the provider). `native` pins the provider behind the id."""
    var binding: PythonObject
    var native: PythonObject
    var flags: Int
    var provider_id: Int

    def __init__(out self, binding: PythonObject, native: PythonObject, flags: Int, provider_id: Int):
        self.binding = binding
        self.native = native
        self.flags = flags
        self.provider_id = provider_id

    def described(self) -> Bool:
        return self.provider_id != 0


@fieldwise_init
struct KmParams(ImplicitlyCopyable, Movable):
    """The fitted `_kernel_params` tuple, typed: the kernel words and the
    ridge alpha (KernelRidge) or the seed (Nystroem)."""
    var kernel: KmKernel
    var alpha: Float64
    var seed: Int


struct KmUnpacked(Movable):
    """One operation after UNPACK. Scalars are typed; `a` holds the addresses
    of the fitted or input buffers in the entry's order; the pins keep those
    buffers until the door has run and packed. Fields an operation does not
    use keep their zero value."""
    var route: KmRoute
    var fields: PythonObject
    # The list entry of a module without the provider, where UNPACK selects it.
    var entry: PythonObject
    var pointer: Bool    # `entry` is the pointer entry (it answers status words)
    var joint: Bool      # fit and transform are one entry
    var gather: Bool     # a device fold this route cannot take: gather host rows
    var scaled: Bool     # gamma='scale': RUN computes gamma before the fit
    var weighted: Bool
    var squeeze: Bool
    var rows: Int
    var cols: Int
    var count: Int
    var targets: Int
    var kernel: KmKernel
    var alpha: Float64
    var seed: Int
    var info: Int
    var sweeps: Int
    var sigma: Float64
    var scale: Float64
    var a: List[Int]
    var x_fit: PythonObject
    var dual: PythonObject
    var pin_a: ArrayPin
    var pin_b: ArrayPin
    var pin_c: ArrayPin
    var pin_d: ArrayPin
    var pin_e: ArrayPin

    def __init__(out self, var route: KmRoute, fields: PythonObject):
        self.route = route^
        self.fields = fields
        self.entry = PythonObject(None)
        self.pointer = False
        self.joint = False
        self.gather = False
        self.scaled = False
        self.weighted = False
        self.squeeze = True
        self.rows = 0
        self.cols = 0
        self.count = 0
        self.targets = 1
        self.kernel = KmKernel(0, 0, 0.0, 0.0)
        self.alpha = 0.0
        self.seed = 0
        self.info = 0
        self.sweeps = 0
        self.sigma = 0.0
        self.scale = 0.0
        self.a = List[Int]()
        self.x_fit = PythonObject(None)
        self.dual = PythonObject(None)
        self.pin_a = ArrayPin()
        self.pin_b = ArrayPin()
        self.pin_c = ArrayPin()
        self.pin_d = ArrayPin()
        self.pin_e = ArrayPin()


def _km_warning_location() raises -> PythonObject:
    # Match warnings.warn's current Python caller through the supported frame
    # API; sys._getframe/f_code would invoke user audit hooks during execution.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyEval_GetFrame")()
    if not raw:
        return PythonObject(None)
    var frame = PythonObject(from_borrowed=raw)
    raw = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetCode")(frame._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var code = PythonObject(from_owned=raw)
    var line = cpy.lib.get_function[Int32]("PyFrame_GetLineNumber")(frame._obj_ptr)
    var nk = svm_keys()
    var module = py_get(native_frame_globals(frame), nk[].k___name__, "")
    if py_type(module) is not py_type(PythonObject("")):
        module = PythonObject("")
    return Python.tuple(code.co_filename, Int(line), module)


def km_int(value: PythonObject, name: String) raises -> Int:
    # Admit the exact scalar before conversion: int subclasses can run __int__.
    if py_type(value) is not py_type(PythonObject(0)):
        raise Error("TypeError: mojolearn kernel methods: " + name + " must be a native int")
    return Int(py=value)


def km_number(value: PythonObject, name: String, context: PythonObject) raises -> Float64:
    if not metric_is_real(value, context) or metric_is_bool(value, context):
        raise Error("TypeError: mojolearn kernel methods: " + name + " must be a native real number")
    return metric_float(value, context)


def km_seed(value: PythonObject, context: PythonObject) raises -> Int:
    # Native NumPy scalar seeds remain supported without their number slots.
    var word = host_inference_primitive_number(value, context, True)
    return km_int(word, "random_state")


def km_bool(value: PythonObject, name: String) raises -> Bool:
    if py_type(value) is not py_type(PythonObject(True)):
        raise Error("TypeError: mojolearn kernel methods: " + name + " must be a native bool")
    return Bool(py=value)


def km_tuple(value: PythonObject, count: Int, name: String) raises -> PythonObject:
    if py_type(value) is not py_type(Python.tuple()):
        raise Error("TypeError: mojolearn kernel methods: " + name + " must be a native tuple")
    if len(value) != count:
        raise Error("ValueError: mojolearn kernel methods: " + name + " has an invalid size")
    return value


def _km_params(fields: PythonObject, ridge: Bool, context: PythonObject) raises -> KmParams:
    """The fitted `_kernel_params`, each word admitted by its exact type (the
    last one first, as always)."""
    var params = km_tuple(family_field(fields, "_kernel_params"), 5, "_kernel_params")
    var alpha = 0.0
    var seed = 0
    if ridge:
        alpha = km_number(params[4], "alpha", context)
    else:
        seed = km_int(params[4], "random_state")
    var kernel = km_int(params[0], "kernel")
    var degree = km_int(params[1], "degree")
    var gamma = km_number(params[2], "gamma", context)
    var coef0 = km_number(params[3], "coef0", context)
    return KmParams(KmKernel(kernel, degree, gamma, coef0), alpha, seed)


def _km_i32_snapshot(value: PythonObject, count: Int, context: PythonObject, mut pin: ArrayPin) raises -> ArrayMetadata:
    var view = classical_view(value, context, "component_indices_")
    var shape = array_shape_once(view, context)
    if len(shape) != 1 or shape[0] != count:
        raise Error("ValueError: component_indices_ must match the fitted component count")
    var result = array_metadata_once(view, shape^, "component_indices_", context, pin)
    if result.source.kind != CAST_KIND_SIGNED or result.source.itemsize != 4 or not result.c_order:
        raise Error("TypeError: component_indices_ must be a contiguous native int32 array")
    return result^


def _km_kernel_code(kernel: PythonObject, where: String, precomputed: Bool) raises -> Int:
    if py_type(kernel) is py_type(PythonObject("")):
        var name = String(kernel)
        if name == "precomputed":
            if precomputed:
                return KM_KERNEL_PRECOMPUTED
            raise Error("mojolearn " + where + ": kernel='precomputed' is refused by name; the "
                        + "lane forms the kernel matrix on the device and takes no matrix "
                        + "from the caller (kernel_methods/NOT_IMPLEMENTED.tsv)")
        if name == "linear":
            return 0
        if name == "poly" or name == "polynomial":
            return 1
        if name == "rbf":
            return 2
        if name == "sigmoid":
            return 3
        if name == "laplacian":
            return 5
        if name == "cosine":
            return 6
        if name == "chi2":
            return 7
        if name == "additive_chi2":
            return 8
        raise Error("mojolearn " + where + ": kernel must be a supported kernel name, got " + name)
    return km_int(kernel, "kernel")


def _km_real(value: PythonObject, name: String, where: String, context: PythonObject) raises -> Float64:
    return km_number(value, where + "." + name, context)


def _km_gamma(gamma: PythonObject, n_features: Int, where: String, kernel: Int, context: PythonObject) raises -> Float64:
    """`kernel_methods._gamma_for`: None is 1 / n_features (1.0 for chi2)."""
    if gamma is None and kernel == KM_KERNEL_CHI2:
        return 1.0
    if gamma is None:
        return 1.0 / Float64(n_features)
    return _km_real(gamma, "gamma", where, context)


def _km_chi2_input(x: ArrayMetadata, kernel: Int, where: String) raises:
    """scikit-learn's chi2 kernels refuse negative input, before any device work."""
    if (kernel == KM_KERNEL_CHI2 or kernel == KM_KERNEL_ADDITIVE_CHI2) and x.count > 0 and Float64(family_snapshot_min(x)) < 0.0:
        raise Error("mojolearn " + where + ": X contains negative values (the chi2 kernels need x >= 0)")


def _km_query_scan(xq_meta: ArrayMetadata, where: String) raises:
    """The finite test of an X whose entry returns no status word (the
    provider route, the list entry): one parallel host pass over the
    admitted float32 words, scikit-learn's wording (lane a7a)."""
    var src = xq_meta.source
    if src.rows > 0 and src.cols > 0 and not host_all_finite_f32_strided(src.addr, src.s0, src.s1, src.rows, src.cols):
        admitted_finite_refuse(1, where)


def _km_route(binding: PythonObject, native: PythonObject) raises -> KmRoute:
    if native is None:
        return KmRoute(binding, native, 0, 0)
    return KmRoute(binding, native, kernel_method_metadata(native, 14), neural_descriptor_provider_id(native))


def km_extension(fields: PythonObject, binding: PythonObject, where: String, context: PythonObject) raises -> KmRoute:
    """`_KernelMethodBase._extension`: the binary's own compile-time mode
    must be the one requested. Answers the route of this operation."""
    var b = native_loaded_module("builtins")
    var nk = svm_keys()
    var want = py_get(fields, nk[].k_numeric_mode, PythonObject(None))
    if want is None:
        want = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if py_type(want) is not py_type(PythonObject("")):
        raise Error("TypeError: numeric_mode must be a native string or None")
    # The spelling is normalized natively (lane p3-svm-gp-kernel, row 472):
    # no Python strip()/lower() call; only "identical" and "fast" pass.
    var selected = String(String(want).strip()).lower()
    if selected != "identical" and selected != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    answer_mode_word(selected, MODE_FAMILY_KERNEL)
    want = PythonObject(selected)
    var native = kernel_method_provider(fields, binding, selected == "identical")
    # The binary's compile-time mode never changes for one binding object:
    # once it agreed with a requested mode, the check is skipped for that
    # binding and mode (a side record weak-keyed by the binding).
    var checked = trees_side_get(_KM_MODE_SLOT, binding)
    if (checked is not None and py_type(checked) is b.tuple and len(checked) == 2 and py_contains(checked[0], want)
            and ((native is None and checked[1] is None) or (native is not None and checked[1] is native))):
        return _km_route(binding, native)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var getter = Optional[PythonObject]()
    if native is None:
        var found = neural_descriptor_entry(binding, "kernel_methods_numeric_mode", False)
        if found is not None:
            getter = Optional(found)
    if native is not None or Bool(getter):
        var got = kernel_method_metadata(native, 13) if native is not None else km_int(family_call(getter.value()), "numeric mode")
        var codes = py_item(context, nk[].k_family_km_mode_codes)
        var wanted = py_get(codes, want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            var owner_name = py_get(py_fields(binding), nk[].k___name__, PythonObject("kernel methods binding"))
            if py_type(owner_name) is not b.str:
                raise Error("TypeError: kernel methods binding name must be a native string")
            raise Error("RuntimeError: mojolearn " + where + ": numeric_mode=" + family_repr(want) + " was requested "
                        + "but " + String(owner_name) + " reports compile-time mode code "
                        + String(got) + "; the binary and the directory it sits in disagree, rebuild "
                        + "it with bash bindings/build_kernel_methods.sh")
        _ = trees_side_put(_KM_MODE_SLOT, binding, Python.tuple(Python.tuple(want), native))
    return _km_route(binding, native)


def km_door(binding: PythonObject, name: String) raises -> PythonObject:
    """The binding's compiled entry `name`, or None where it registers none."""
    return neural_descriptor_entry(binding, name, False)


def _km_pointer_entry(binding: PythonObject, pointer: String, name: String, context: PythonObject) raises -> PythonObject:
    """The pointer entry when the binding registers it, else the list entry
    (the same addresses and parameters)."""
    var entry = neural_descriptor_entry(binding, pointer, False)
    if entry is not None:
        return entry
    return family_entry(binding, name, context)


def km_listed_entry(route: KmRoute, name: String, context: PythonObject) raises -> PythonObject:
    """The list entry `name` of a module without the provider; None on the
    provider route (RUN calls the table)."""
    if route.described():
        return PythonObject(None)
    return family_entry(route.binding, name, context)


def _km_degree(fields: PythonObject, where: String) raises -> Int:
    return km_int(family_field(fields, "degree"), where + ".degree")


def km_scale_gamma_admit(route: KmRoute, cells: Int) raises:
    """gamma='scale' before any buffer: the module must compute the exact
    limbs, and X must hold at least one cell."""
    var flags = 0
    if route.described():
        flags = kernel_method_metadata(route.native, 15)
    else:
        var flags_entry = neural_descriptor_entry(route.binding, "py2mojo_linear_flags", False)
        if flags_entry is not None:
            flags = km_int(family_call(flags_entry), "linear flags")
    if (flags & _PY2MOJO_SCALE_GAMMA) == 0:
        raise Error("RuntimeError: mojolearn: this binding has no scale_gamma_limbs (an older binary or a "
                    + "-D MOJOLEARN_PY2MOJO_linear_OFF build); rebuild it")
    if cells == 0:
        raise Error("gamma='scale' needs at least one cell of X")


# ---------------------------------------------------------------- KernelRidge


def km_krr_fit_unpack(model: PythonObject, args: PythonObject, context: PythonObject, x_meta: ArrayMetadata,
                      fold_active: Bool) raises -> KmUnpacked:
    """`KernelRidge.fit(X, y, sample_weight)`: args (binding, X, y,
    sample_weight). `gather` answers a device fold this route cannot take (a
    chi2 kernel, whose sign test reads host words; kernel='precomputed'; the
    provider route; a binding without the pointer entry): nothing after the
    kernel name was read, the door gathers the fold's host rows and unpacks
    again. The pointer entry tests X on the device and answers its status
    words; the other routes are tested here (`_km_query_scan`)."""
    var where = String("KernelRidge")
    var fields = py_fields(model)
    var route = km_extension(fields, args[0], where, context)
    var ptr_entry = PythonObject(None)
    if not route.described():
        ptr_entry = neural_descriptor_entry(route.binding, "kernel_ridge_fit_ptr", False)
    var n = x_meta.source.rows
    var d = x_meta.source.cols
    if args[2] is None:
        raise Error("ValueError: KernelRidge requires y to be passed, but the target y is None.")
    var target_pin = ArrayPin()
    var target = admit_family_meta(args[2], -1, "y", context, target_pin)
    kernel_public_finite(target, "y")
    var rank = len(target.shape)
    var t = 1
    var squeeze = True
    if rank == 2:
        t = target.source.cols
        squeeze = kernel_public_column_target(args[2], target, context)
    elif rank != 1:
        raise Error("mojolearn " + where + ": y must be 1-D or 2-D, got " + String(rank) + "-D")
    if target.source.rows != n:
        raise Error("mojolearn " + where + ": y has " + String(target.source.rows) + " rows, X has " + String(n))
    # `_kp`: the kernel, degree, gamma and coef0, in that order.
    var kernel = _km_kernel_code(family_field(fields, "kernel"), where, True)
    var u = KmUnpacked(route^, fields)
    u.entry = ptr_entry
    u.pointer = ptr_entry is not None
    u.rows = n
    u.cols = d
    u.targets = t
    u.squeeze = squeeze
    if fold_active and (kernel == KM_KERNEL_CHI2 or kernel == KM_KERNEL_ADDITIVE_CHI2
                        or kernel == KM_KERNEL_PRECOMPUTED or not u.pointer):
        u.gather = True
        return u^
    var degree = _km_degree(fields, where)
    var gamma = _km_gamma(family_field(fields, "gamma"), d, where, kernel, context)
    var coef0 = _km_real(family_field(fields, "coef0"), "coef0", where, context)
    u.kernel = KmKernel(kernel, degree, gamma, coef0)
    if not fold_active:
        _km_chi2_input(x_meta, kernel, where)
    if kernel == KM_KERNEL_PRECOMPUTED and d != n:
        raise Error("mojolearn " + where + ": kernel='precomputed' needs the square "
                    + "n x n kernel matrix as X, got " + String(n) + " x " + String(d))
    if not u.pointer:
        # The list entry and the provider route return no status word.
        _km_query_scan(x_meta, where)
    u.alpha = _km_real(family_field(fields, "alpha"), "alpha", where, context)
    u.a.append(target.source.addr)
    u.pin_a = target_pin^
    var weights_address = 0
    if args[3] is not None:
        var w_pin = ArrayPin()
        var snapshot = admit_family_weights(args[3], n, False, context, w_pin, wide=True)
        weights_address = snapshot.source.addr
        u.weighted = True
        u.pin_b = w_pin^
    u.a.append(weights_address)
    if not u.route.described() and not u.pointer:
        u.entry = family_entry(u.route.binding, "kernel_ridge_fit", context)
    return u^


def km_krr_predict_unpack(model: PythonObject, binding: PythonObject, context: PythonObject,
                          xq_meta: ArrayMetadata, fold_active: Bool) raises -> KmUnpacked:
    """`KernelRidge.predict(X)` up to the route. For a device fold only the
    fitted dimensions and the route are read (`gather` as in the fit: a chi2
    kernel, the provider route, a binding without the pointer entry);
    `km_krr_fitted_unpack` reads the rest."""
    var where = String("KernelRidge")
    var fields = py_fields(model)
    var nk = svm_keys()
    if not py_contains(fields, nk[].k_dual_coef_):
        raise Error("mojolearn " + where + ": call fit before predict")
    var q = xq_meta.source.rows
    var d = xq_meta.source.cols
    var fitted = km_int(family_field(fields, "n_features_in_"), "n_features_in_")
    if d != fitted:
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(fitted))
    var x_fit_obj = family_field(fields, "X_fit_")
    var t = km_int(family_field(fields, "n_targets_"), "n_targets_")
    var kp = _km_params(fields, True, context)
    var route = km_extension(fields, binding, where, context)
    var ptr_entry = PythonObject(None)
    if not route.described():
        ptr_entry = neural_descriptor_entry(route.binding, "kernel_ridge_predict_ptr", False)
    var u = KmUnpacked(route^, fields)
    u.entry = ptr_entry
    u.pointer = ptr_entry is not None
    u.rows = q
    u.cols = d
    u.targets = t
    u.kernel = kp.kernel
    u.alpha = kp.alpha
    u.x_fit = x_fit_obj
    var chi2 = kp.kernel.kernel == KM_KERNEL_CHI2 or kp.kernel.kernel == KM_KERNEL_ADDITIVE_CHI2
    if fold_active:
        u.gather = chi2 or not u.pointer
        return u^
    _km_chi2_input(xq_meta, kp.kernel.kernel, where)
    u.dual = family_field(fields, "dual_coef_")
    u.squeeze = km_bool(family_field(fields, "_squeeze"), "_squeeze")
    u.info = km_int(family_field(fields, "info_"), "info_")
    if u.route.described() or not u.pointer:
        # The provider route and the list entry return no status word.
        _km_query_scan(xq_meta, where)
    return u^


def km_krr_fitted_unpack(mut u: KmUnpacked, fold: Bool, context: PythonObject) raises:
    """The fitted X_fit_ and dual_coef_ of a predict that uploads them (no
    resident hit): `u.a` is their two addresses, `u.count` the fitted rows.
    A device fold reads dual_coef_, `_squeeze` and info_ here."""
    var x_fit_meta = admit_family_meta(u.x_fit, 2, "X_fit_", context, u.pin_a)
    var n = x_fit_meta.source.rows
    if fold:
        u.dual = family_field(u.fields, "dual_coef_")
    var dual_meta = admit_family_meta(u.dual, -1, "dual_coef_", context, u.pin_b)
    if u.targets < 1 or x_fit_meta.source.cols != u.cols or dual_meta.count != n * u.targets:
        raise Error("ValueError: KernelRidge fitted arrays do not match the fitted dimensions")
    if fold:
        u.squeeze = km_bool(family_field(u.fields, "_squeeze"), "_squeeze")
        u.info = km_int(family_field(u.fields, "info_"), "info_")
    u.count = n
    u.a.append(x_fit_meta.source.addr)
    u.a.append(dual_meta.source.addr)


# ------------------------------------------------------------------- Nystroem


def km_nys_fit_unpack(model: PythonObject, binding: PythonObject, transform: Bool, context: PythonObject,
                      x_meta: ArrayMetadata) raises -> KmUnpacked:
    """`Nystroem.fit(X)` / `fit_transform(X)`. `joint`: the module runs the
    fit and the transform as one entry (X uploaded once)."""
    var where = String("Nystroem")
    var fields = py_fields(model)
    var n = x_meta.source.rows
    var d = x_meta.source.cols
    var kernel = _km_kernel_code(family_field(fields, "kernel"), where, False)
    var degree = _km_degree(fields, where)
    var q = km_int(family_field(fields, "n_components"), where + ".n_components")
    if q < 1:
        raise Error("mojolearn " + where + ": n_components must be positive, got " + String(q))
    if q > n:
        # scikit-learn's rule and warning (kernel_approximation.py,
        # Nystroem.fit): n_components becomes n_samples (the full kernel).
        classical_warning("n_components > n_samples. This is not possible.\n"
                          + "n_components was set to n_samples, which results"
                          + " in inefficient evaluation of the full kernel.",
                          native_loaded_module("builtins").UserWarning, context,
                          _km_warning_location())
        q = n
    var gamma = _km_gamma(family_field(fields, "gamma"), d, where, kernel, context)
    _km_chi2_input(x_meta, kernel, where)
    var coef0 = _km_real(family_field(fields, "coef0"), "coef0", where, context)
    var seed = km_seed(family_field(fields, "random_state"), context)
    var route = km_extension(fields, binding, where, context)
    var u = KmUnpacked(route^, fields)
    u.rows = n
    u.cols = d
    u.count = q
    u.kernel = KmKernel(kernel, degree, gamma, coef0)
    u.seed = seed
    if u.route.described():
        u.joint = transform and (u.route.flags & 6) == 6
        return u^
    if transform:
        var joint = km_door(u.route.binding, "nystroem_fit_transform_ptr")
        if joint is not None:
            u.joint = True
            u.entry = joint
            return u^
    u.entry = _km_pointer_entry(u.route.binding, "nystroem_fit_ptr", "nystroem_fit", context)
    return u^


def km_nys_transform_unpack(model: PythonObject, binding: PythonObject, context: PythonObject,
                            x_meta: ArrayMetadata) raises -> KmUnpacked:
    """`Nystroem.transform(X)`: `u.a` is components, indices, normalization,
    eigenvalues, eigenvectors."""
    var where = String("Nystroem")
    var fields = py_fields(model)
    var nk = svm_keys()
    if not py_contains(fields, nk[].k_components_):
        raise Error("mojolearn " + where + ": call fit before transform")
    var m = x_meta.source.rows
    var d = x_meta.source.cols
    var fitted = km_int(family_field(fields, "n_features_in_"), "n_features_in_")
    if d != fitted:
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(fitted))
    var components_pin = ArrayPin()
    var indices_pin = ArrayPin()
    var normalization_pin = ArrayPin()
    var eigenvalues_pin = ArrayPin()
    var eigenvectors_pin = ArrayPin()
    var components_meta = admit_family_meta(family_field(fields, "components_"), 2, "components_", context, components_pin)
    var q = components_meta.source.rows
    var kp = _km_params(fields, False, context)
    _km_chi2_input(x_meta, kp.kernel.kernel, where)
    var indices_meta = _km_i32_snapshot(family_field(fields, "component_indices_"), q, context, indices_pin)
    var normalization_meta = admit_family_meta(family_field(fields, "normalization_"), 2, "normalization_", context, normalization_pin)
    var eigenvalues_meta = admit_family_meta(family_field(fields, "eigenvalues_"), 1, "eigenvalues_", context, eigenvalues_pin)
    var eigenvectors_meta = admit_family_meta(family_field(fields, "eigenvectors_"), 2, "eigenvectors_", context, eigenvectors_pin)
    if (components_meta.source.cols != d or normalization_meta.source.rows != q or normalization_meta.source.cols != q
            or eigenvalues_meta.count != q or eigenvectors_meta.source.rows != q or eigenvectors_meta.source.cols != q):
        raise Error("ValueError: Nystroem fitted arrays do not match the fitted dimensions")
    var route = km_extension(fields, binding, where, context)
    var u = KmUnpacked(route^, fields)
    u.entry = km_listed_entry(u.route, "nystroem_transform", context)
    u.rows = m
    u.cols = d
    u.count = q
    u.kernel = kp.kernel
    u.seed = kp.seed
    u.sweeps = km_int(family_field(fields, "sweeps_"), "sweeps_")
    u.a.append(components_meta.source.addr)
    u.a.append(indices_meta.source.addr)
    u.a.append(normalization_meta.source.addr)
    u.a.append(eigenvalues_meta.source.addr)
    u.a.append(eigenvectors_meta.source.addr)
    u.pin_a = components_pin^
    u.pin_b = indices_pin^
    u.pin_c = normalization_pin^
    u.pin_d = eigenvalues_pin^
    u.pin_e = eigenvectors_pin^
    return u^


# ----------------------------------------------------------------- RBFSampler


def km_rbf_fit_unpack(model: PythonObject, binding: PythonObject, transform: Bool, context: PythonObject,
                      x_meta: ArrayMetadata) raises -> KmUnpacked:
    """`RBFSampler.fit(X)` / `fit_transform(X)`: `_fit_params` in their order
    (gamma, then n_components, then the seed). gamma='scale' is admitted here
    (`scaled`) and computed by RUN before the fit. `joint`: the module's one
    fit-and-transform entry (lane apple-fast-w2-kfeat
    MOJOLEARN_KM_FAST_RBF_RESIDENT, a module without the provider only)."""
    var where = String("RBFSampler")
    var fields = py_fields(model)
    var m = x_meta.source.rows
    var d = x_meta.source.cols
    var gamma_param = family_field(fields, "gamma")
    var gamma = 0.0
    var scaled = py_type(gamma_param) is py_type(PythonObject("")) and String(gamma_param) == "scale"
    var route = KmRoute(binding, PythonObject(None), 0, 0)
    if scaled:
        route = km_extension(fields, binding, where, context)
        km_scale_gamma_admit(route, x_meta.count)
    else:
        gamma = _km_real(gamma_param, "gamma", where, context)
    var q = km_int(family_field(fields, "n_components"), where + ".n_components")
    if q < 1:
        # BY NAME, BEFORE ANY BUFFER (the MI300X n_components=0 run, 2026-09-14).
        raise Error("mojolearn " + where + ": n_components must be positive, got " + String(q))
    var seed = km_seed(family_field(fields, "random_state"), context)
    if not scaled:
        route = km_extension(fields, binding, where, context)
    var u = KmUnpacked(route^, fields)
    u.scaled = scaled
    u.rows = m
    u.cols = d
    u.count = q
    u.kernel = KmKernel(0, 0, gamma, 0.0)
    u.seed = seed
    if u.route.described():
        return u^
    if transform:
        var joint = neural_descriptor_entry(u.route.binding, "rbf_sampler_fit_transform_resident", False)
        if joint is not None:
            u.joint = True
            u.entry = joint
            return u^
    u.entry = family_entry(u.route.binding, "rbf_sampler_fit", context)
    return u^


def km_rbf_transform_unpack(model: PythonObject, binding: PythonObject, context: PythonObject,
                            x_meta: ArrayMetadata) raises -> KmUnpacked:
    """`RBFSampler.transform(X)`: `u.a` is random_weights_, random_offset_."""
    var where = String("RBFSampler")
    var fields = py_fields(model)
    var nk = svm_keys()
    if not py_contains(fields, nk[].k_random_weights_):
        raise Error("mojolearn " + where + ": call fit before transform")
    var m = x_meta.source.rows
    var d = x_meta.source.cols
    var fitted = km_int(family_field(fields, "n_features_in_"), "n_features_in_")
    if d != fitted:
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(fitted))
    var weights_pin = ArrayPin()
    var offset_pin = ArrayPin()
    var weights_meta = admit_family_meta(family_field(fields, "random_weights_"), 2, "random_weights_", context, weights_pin)
    var q = weights_meta.source.cols
    var params_in = km_tuple(family_field(fields, "_params"), 2, "_params")
    var gamma = km_number(params_in[0], "gamma", context)
    var seed = km_int(params_in[1], "random_state")
    var offset_meta = admit_family_meta(family_field(fields, "random_offset_"), 1, "random_offset_", context, offset_pin)
    if weights_meta.source.rows != d or offset_meta.count != q:
        raise Error("ValueError: RBFSampler fitted arrays do not match the fitted dimensions")
    var route = km_extension(fields, binding, where, context)
    var u = KmUnpacked(route^, fields)
    u.rows = m
    u.cols = d
    u.count = q
    u.kernel = KmKernel(0, 0, gamma, 0.0)
    u.seed = seed
    u.sigma = km_number(family_field(fields, "sigma_"), "sigma_", context)
    u.scale = km_number(family_field(fields, "scale_"), "scale_", context)
    u.entry = km_listed_entry(u.route, "rbf_sampler_transform", context)
    u.a.append(weights_meta.source.addr)
    u.a.append(offset_meta.source.addr)
    u.pin_a = weights_pin^
    u.pin_b = offset_pin^
    return u^
