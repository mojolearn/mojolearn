# SPDX-License-Identifier: Apache-2.0
# Cholesky `fit`, one native call (python/mojolearn/_cholesky_impl.py). The
# shell hands over `_door_binding()`: the binding that serves this object's
# tier (the CPU binding on a CPU-only install, whose mode read-back is
# `linalg_numeric_mode`; the GPU binding's is `gp_numeric_mode`). This owner makes the
# Python method's refusals, cross-checks the mode (`_extension`), reads the
# profile's ridge for `jitter=None`, runs the binding's `cholesky_factor`
# with the argument lists bindings/_mojolearn_gp.mojo documents, and stores
# the factor. The same kernel on the same inputs: the bits do not move.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.py_balanced import py_call, py_get, py_fields, py_setitem
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.py_model_records import records_stamped_get, records_stamped_put, records_stampable, ContentRecord, RECORDS_PREPARED
from bindings.family_common_unpack import (
    FAMILY_F64P,
    family_repr,
    family_shape_repr,
)
from bindings.family_common_pack import (
    family_call,
    family_entry,
    family_array,
)
from bindings.py_balanced import py_contains, py_type
from bindings.parallel_plan_door import plan_devices, plan_enter, plan_exit, plan_require_identical
from core.device_plan import DEVICE_PLAN_WORKER, DEVICE_PLAN_CHOLESKY
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_pandas_door import classical_admit_value
from bindings.host_inference_scalars_door import host_inference_scalar
from providers.python.estimator_binding_pyapi_door import estimator_binding
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_fit_begin, estimator_fit_end
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.linalg_keys_unpack import linalg_keys, LinalgKeys
from bindings.py_balanced import py_item
from bindings.mode_answer_door import fast_answers_compiled_code
from providers.python.device_route_boundary_door import (
    device_binding_held, device_binding_device, device_admit, device_route_clear,
)
from core.device_route import DEVICE_CPU
from core.mode_answer import MODE_FAMILY_LINALG
from bindings.operation_names_chol import (
    CHOL_OPN_FAMILY_CHOL_INITIALIZE, CHOL_OPN_FAMILY_CHOL_LOGDET, CHOL_OPN_FAMILY_CHOL_BINDING,
    CHOL_OPN_FAMILY_CHOL_DOOR, CHOL_OPN_FAMILY_CHOL_VENDOR, CHOL_OPN_FAMILY_CHOL_PROFILE_JITTER,
    CHOL_OPN_FAMILY_CHOL_FIT, CHOL_OPN_FAMILY_CHOL_SOLVE, CHOL_OPN_FAMILY_CHOL_PAR_FIT,
    CHOL_OPN_FAMILY_CHOL_PAR_SOLVE)


@fieldwise_init
struct CholImage(Movable):
    """The compiled image of one Cholesky estimator: the GP device image
    (`gp`: it links every cholesky kernel; readers `gp_*`) or the linalg CPU
    image (readers `linalg_*`), from the estimator's binding route."""
    var module: PythonObject
    var gp: Bool


def _chol_image(model: PythonObject, context: PythonObject) raises -> CholImage:
    """The estimator's image, from its binding route
    (providers/python/device_route_boundary_door.mojo). The route holds the
    module under the name it was resolved by: `_mojolearn_gp` on a device
    route, `_mojolearn_linalg` on a CPU route (a CPU-only installation, or
    `device="cpu"` beside a device set: the CPU identical module connected at
    import, `_mojolearn_linalg_host`). A hit is one or two held-binding reads
    (record read, name and pointer compares); the process is not asked which
    set it is and the device is not read. A miss (first operation, a changed
    mode object, after `__init__` or `set_params`) reads the device once,
    picks the name from it and resolves through the shared estimator door."""
    var nk = linalg_keys()
    var requested = py_get(py_fields(model), nk[].k_numeric_mode)
    var backend = py_item(context, nk[].k_backend_fields)
    var default_mode = PythonObject(None)
    if requested is None:
        default_mode = py_item(backend, nk[].k__DEFAULT_MODE)
    var held = device_binding_held(model, "_mojolearn_gp", False, requested, default_mode, context)
    if held is not None:
        return CholImage(held, True)
    held = device_binding_held(model, "_mojolearn_linalg", False, requested, default_mode, context)
    if held is not None:
        return CholImage(held, False)
    var cpu_only = py_item(backend, nk[].k__CPU_ONLY) is not None
    var gp = not cpu_only
    if gp:
        gp = device_binding_device(model, py_fields(model), context, cpu_only,
                                   native_loaded_module("builtins")) != DEVICE_CPU
    var name = "_mojolearn_gp" if gp else "_mojolearn_linalg"
    return CholImage(estimator_binding(model, name, context, False), gp)


def _chol_field(fields: PythonObject, key: PythonObject, name: String) raises -> PythonObject:
    """`family_field` with the held key (bw2-linalg): `fields[name]`, refused
    as the attribute lookup refused it; `name` is the refusal text only (it
    built the key str up to twice per call)."""
    var value = py_get(fields, key, PythonObject(None))
    if value is None and not py_contains(fields, key):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return value


def _chol_meta_key(nk: UnsafePointer[LinalgKeys, MutUntrackedOrigin], i: Int) -> PythonObject:
    """The held key of fixed Cholesky metadata field `i` (jitter, numeric_mode,
    n_, info_, nb_, _logdet, jitter_): the loops pick their key by index and
    build no str (they built one per field and test)."""
    if i == 0:
        return nk[].k_jitter.copy()
    if i == 1:
        return nk[].k_numeric_mode.copy()
    if i == 2:
        return nk[].k_n_.copy()
    if i == 3:
        return nk[].k_info_.copy()
    if i == 4:
        return nk[].k_nb_.copy()
    if i == 5:
        return nk[].k__logdet.copy()
    return nk[].k_jitter_.copy()


def _chol_fit_key(nk: UnsafePointer[LinalgKeys, MutUntrackedOrigin], i: Int) -> PythonObject:
    """The held key of fit-owned field `i` (n_, L_, info_, nb_, _logdet,
    jitter_, _mojolearn_fitted)."""
    if i == 0:
        return nk[].k_n_.copy()
    if i == 1:
        return nk[].k_L_.copy()
    if i == 2:
        return nk[].k_info_.copy()
    if i == 3:
        return nk[].k_nb_.copy()
    if i == 4:
        return nk[].k__logdet.copy()
    if i == 5:
        return nk[].k_jitter_.copy()
    return nk[].k__mojolearn_fitted.copy()


def _chol_admit_state(fields: PythonObject, context: PythonObject) raises:
    var nk = linalg_keys()
    # Exact owner storage is admitted by the shared family frontdoor. Its
    # scalar fields still require admission before any native conversion slot.
    for i in range(7):  # driver-loop: fixed Cholesky metadata fields
        var name = _chol_meta_key(nk, i)
        if py_contains(fields, name):
            host_inference_scalar(py_item(fields, name), context)
    if py_contains(fields, nk[].k_L_):
        classical_admit_value(py_item(fields, nk[].k_L_), context)
    if py_contains(fields, nk[].k__mojolearn_after_fit):
        raise Error("TypeError: Cholesky cannot invoke Python fit callbacks")


comptime _CHOL_STAMP = "linalg_chol_scalars_stamp_v1"


def _chol_admit_state_once(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    """`_chol_admit_state` with its scalar half admitted once per generation
    (lane ne1-e-records, row N238): while the model's generation word
    (bindings/py_model_records.mojo) is unchanged, its stored names and
    values are the same objects, and exact builtin scalars admit the same
    way again. `L_` is admitted on every call (an ndarray can be reshaped or
    retyped in place)."""
    var nk = linalg_keys()
    var stamped = records_stamped_get(RECORDS_PREPARED, model, _CHOL_STAMP)
    if stamped is not None and stamped[0] is fields:
        if py_contains(fields, nk[].k_L_):
            classical_admit_value(py_item(fields, nk[].k_L_), context)
        return
    _chol_admit_state(fields, context)
    if not (py_fields(model) is fields) or not records_stampable(model):
        return
    var b = native_loaded_module("builtins")
    for i in range(7):  # small-loop(fixed Cholesky metadata fields)
        var name = _chol_meta_key(nk, i)
        if py_contains(fields, name):
            var value = py_item(fields, name)
            var cls = py_type(value)
            if not (value is None or cls is b.bool or cls is b.int or cls is b.float or cls is b.str):
                return
    _ = records_stamped_put(RECORDS_PREPARED, model, _CHOL_STAMP, ContentRecord(), Python.tuple(fields))


def _chol_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = linalg_keys()
    var b = native_loaded_module("builtins")
    var jitter = args[0]
    classical_admit_value(jitter, context)
    if jitter is not None:
        if py_type(jitter) is b.bool or not Bool(py=py_call(b.isinstance, jitter, Python.tuple(b.int, b.float))):
            raise Error("TypeError: mojolearn Cholesky: jitter must be a float, 0.0 or None")
    # The parameter is stored as given (scikit-learn's get_params/clone
    # contract is identity); fit converts it to a float.
    var fields = py_fields(model)
    var mode = args[1]
    # NumericModeMixin's former constructor kept an existing instance mode
    # when numeric_mode=None was supplied during reinitialization.
    if mode is None and py_contains(fields, nk[].k_numeric_mode):
        mode = py_item(fields, nk[].k_numeric_mode)
    if mode is not None and py_type(mode) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    # `device=None | "gpu" | "cpu"`: checked where it is given, in the shared
    # words, before any field is published. The native reconstruction of
    # `set_params` and the model-selection constructor pass it third.
    var device = PythonObject(None)
    if len(args) > 2:
        device = args[2]
        classical_admit_value(device, context)
        device_admit(device, 0 if py_item(py_item(context, nk[].k_backend_fields), nk[].k__CPU_ONLY) is not None else 1, b)
    # Reinitialization must admit every released owner before publishing any
    # replacement. A user can have assigned an object with a Python finalizer
    # since the previous constructor; retaining the dictionary until this pass
    # succeeds keeps refusal atomic and runs none of those lifecycle hooks.
    var names = Python.tuple(nk[].k_jitter, nk[].k_numeric_mode)
    var replacements = Python.tuple(jitter, mode)
    for i in range(len(names)):  # driver-loop: two constructor-owned fields, admitted before any replacement
        if py_contains(fields, names[i]) and fields[names[i]] is not replacements[i]:
            native_release_admit(fields[names[i]], context)
    py_setitem(fields, nk[].k_jitter, jitter)
    py_setitem(fields, nk[].k_numeric_mode, mode)
    py_setitem(fields, "device", device)
    # A constructor run again (`set_params`) resolves the image again.
    device_route_clear(model)
    return PythonObject(None)


def _chol_extension(model: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = linalg_keys()
    # The public route is resolved from the connected registry. Legacy internal
    # callers may still provide the exact module; read its dictionary directly,
    # never an optional Python module __getattr__ hook.
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var want = py_get(fields, nk[].k_numeric_mode)
    if want is None:
        want = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if py_type(want) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    # The module it was handed answers which image it is: the GP image
    # exports `gp_numeric_mode`, the linalg CPU image `linalg_numeric_mode`.
    var getter = neural_descriptor_entry(binding, "gp_numeric_mode", False)
    if getter is None:
        getter = neural_descriptor_entry(binding, "linalg_numeric_mode", False)
    if getter is not None:
        var got = Int(py=family_call(getter))
        var codes = py_item(context, nk[].k_family_chol_mode_codes)
        var wanted = py_get(codes, want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            # lane fx-fast-mask: `fast` answered by the identical set (LINALG)
            # passes on the mismatch branch only; a matching call pays nothing.
            if not fast_answers_compiled_code(want, codes, got, MODE_FAMILY_LINALG):
                raise Error("RuntimeError: mojolearn Cholesky: requested numeric mode differs from the compiled binding")
    return binding


def _chol_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`Cholesky.fit(A)`: factor A + jitter * I."""
    var nk = linalg_keys()
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    for i in range(7):  # driver-loop: admit replaced fit owners before numerical work or publication
        var name = _chol_fit_key(nk, i)
        if py_contains(fields, name):
            native_release_admit(py_item(fields, name), context)
    var a = admit_family(args[1], 2, "A", context).array
    var n = Int(py=a.shape[0])
    var m = Int(py=a.shape[1])
    if n != m:
        raise Error("mojolearn Cholesky: A must be square, got shape " + family_shape_repr(a))
    var binding = _chol_extension(model, args[0], context)
    var jitter = _chol_field(fields, nk[].k_jitter, "jitter")
    if jitter is None:
        var profile = neural_descriptor_entry(binding, "cholesky_profile_jitter")
        jitter = family_call(b.float, family_call(profile))
    else:
        jitter = PythonObject(Float64(py=jitter))
    var l_out = family_array(Python.tuple(n, n), "<f4", context)
    # info, nb, logdet, jitter -- in that order.
    var scalars = family_array(Python.tuple(4), "<f8", context)
    var factor = neural_descriptor_entry(binding, "cholesky_factor")
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::cholesky_factor_binding:
    # a, l_out, scalars_out; n, jitter.
    var info = family_call(factor, Python.list(classical_address(a), classical_address(l_out), classical_address(scalars)),
                           Python.list(PythonObject(n), jitter))
    _ = scalars  # keep the Python object alive through the native call above
    _ = l_out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    var words = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(scalars)))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_, PythonObject(n))
    py_setitem(fields, nk[].k_L_, l_out)
    py_setitem(fields, nk[].k_info_, family_call(b.int, info))
    _ = scalars  # keep the Python object alive through the native call above
    _ = l_out  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_nb_, PythonObject(Int(words[1])))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k__logdet, PythonObject(words[2]))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_jitter_, PythonObject(words[3]))
    _ = scalars  # keep the Python object alive through the native call above
    _ = scalars^


def _chol_solve(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Cholesky.solve(B)`: A X = B from the factor, in the method's refusal
    order and wording; args (binding, B)."""
    var nk = linalg_keys()
    classical_connect_numpy(context)
    var fields = py_fields(model)
    if not py_contains(fields, nk[].k_L_):
        raise Error("ValueError: mojolearn Cholesky: call fit before solve")
    var b = admit_family(args[1], -1, "B", context).array
    var rank = Int(py=b.ndim)
    var rows: Int
    var nrhs: Int
    if rank == 1:
        rows = Int(py=b.shape[0])
        nrhs = 1
    elif rank == 2:
        rows = Int(py=b.shape[0])
        nrhs = Int(py=b.shape[1])
    else:
        raise Error("ValueError: mojolearn Cholesky: B must be 1-D or 2-D, got " + String(rank) + "-D")
    var n = Int(py=_chol_field(fields, nk[].k_n_, "n_"))
    if rows != n:
        raise Error("ValueError: mojolearn Cholesky: B has " + String(rows) + " rows, the factor is "
                    + String(n) + " x " + String(n))
    if nrhs < 1:
        raise Error("ValueError: mojolearn Cholesky: B must have at least one column")
    var l = admit_family(_chol_field(fields, nk[].k_L_, "L_"), 2, "L_", context).array
    if Int(py=l.size) != n * n:
        raise Error("ValueError: mojolearn Cholesky: L_ size differs from its fitted matrix dimensions")
    var x_out = family_array(Python.tuple(n) if rank == 1 else Python.tuple(n, nrhs), "<f4", context)
    var solve = neural_descriptor_entry(_chol_extension(model, args[0], context), "cholesky_solve")
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::cholesky_solve_binding:
    # l, b, x_out; n, nrhs, info, nb, logdet, jitter.
    _ = family_call(solve, Python.list(classical_address(l), classical_address(b), classical_address(x_out)),
                    Python.list(PythonObject(n), PythonObject(nrhs), _chol_field(fields, nk[].k_info_, "info_"),
                                _chol_field(fields, nk[].k_nb_, "nb_"), _chol_field(fields, nk[].k__logdet, "_logdet"),
                                _chol_field(fields, nk[].k_jitter_, "jitter_")))
    _ = x_out  # keep the Python object alive through the native call above
    _ = l^
    _ = b^
    return x_out


# ---------------------------------------------------------------------------
# parallel_classical.fit_cholesky / solve_cholesky, one native call each
# (lane/native-parallel-driver-20261024). The old cooperative worker set
# MOJOLEARN_CHOLESKY_DEVICE_COUNT for these two operations only; the plan
# carries the same families (the worker's environment plus Cholesky).
# ---------------------------------------------------------------------------

comptime _CHOL_REBUILD = "rebuild GP binding for operation-level multi-GPU Cholesky"


def _chol_par_admit(model: PythonObject, context: PythonObject) raises:
    """`parallel_classical._admit_cholesky`."""
    var nk = linalg_keys()
    if py_type(model) is not py_item(context, nk[].k_family_chol_par_class):
        raise Error("TypeError: requires mojolearn.Cholesky")
    plan_require_identical(model, "parallel Cholesky requires IDENTICAL numeric mode", MODE_FAMILY_LINALG)


def _chol_par(model: PythonObject, args: PythonObject, solve: Bool, context: PythonObject) raises -> PythonObject:
    """args (binding, A or B, devices): the binding `Cholesky._door_binding`
    resolves."""
    _chol_par_admit(model, context)
    if solve:
        var nk = linalg_keys()
        if not py_contains(py_fields(model), nk[].k_L_):
            raise Error("ValueError: Cholesky is not fitted")
    var devices = plan_devices(args[2], String("cholesky_solve") if solve else String("cholesky_fit"), True, context)
    var binding = args[0]
    var out = PythonObject(None)
    plan_enter(binding, devices, _CHOL_REBUILD, DEVICE_PLAN_WORKER | DEVICE_PLAN_CHOLESKY)
    try:
        if solve:
            out = _chol_solve(model, Python.tuple(binding, args[1]), context)
        else:
            _chol_fit(model, Python.tuple(binding, args[1]), context)
            out = model
    finally:
        plan_exit(binding)
    return out


def linalg_unknown_operation(operation: String) raises:
    """The error of a `family_chol_*` name that is not an operation (it has no
    leaf code): the owner's unchanged text. Always raises."""
    raise Error("RuntimeError: unknown linalg operation " + operation)


def linalg_operation(model: PythonObject, operation: Int, var args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    # bw2-linalg: `operation` is the leaf code of bindings/operation_names_chol.mojo,
    # resolved once by the family door (it was up to 14 String compares here).
    var nk = linalg_keys()
    classical_connect_numpy(context)
    if operation == CHOL_OPN_FAMILY_CHOL_INITIALIZE:
        return _chol_initialize(model, args, context)
    var fields = py_fields(model)
    _chol_admit_state_once(model, fields, context)
    if operation == CHOL_OPN_FAMILY_CHOL_LOGDET:
        if not py_contains(fields, nk[].k_info_):
            raise Error("AttributeError: mojolearn Cholesky: logdet_ is set by fit")
        var info = Int(py=py_item(fields, nk[].k_info_))
        if info != 0:
            raise Error("ValueError: mojolearn Cholesky: the factorization failed (info=" + String(info)
                        + "), so there is no determinant to report (DEVIATION 1634)")
        return _chol_field(fields, nk[].k__logdet, "_logdet")
    if operation >= CHOL_OPN_FAMILY_CHOL_BINDING and operation <= CHOL_OPN_FAMILY_CHOL_PROFILE_JITTER:
        var image = _chol_image(model, context)
        var binding = _chol_extension(model, image.module, context)
        if operation == CHOL_OPN_FAMILY_CHOL_BINDING:
            return binding
        if operation == CHOL_OPN_FAMILY_CHOL_DOOR:
            return Python.tuple(binding, PythonObject("gp_numeric_mode" if image.gp else "linalg_numeric_mode"))
        if operation == CHOL_OPN_FAMILY_CHOL_VENDOR:
            var name = "gp_vendor" if image.gp else "linalg_host_vendor"
            var getter = neural_descriptor_entry(binding, name, False)
            return PythonObject(None) if getter is None else family_call(getter)
        return PythonObject(Float64(py=family_call(neural_descriptor_entry(binding, "cholesky_profile_jitter"))))
    if len(args) > 0 and args[0] is None:
        # The public form names no binding: the estimator's image takes the
        # first slot. Fixed public argument counts (fit and solve two, the
        # parallel pair three): the tuple is built directly.
        var module = _chol_image(model, context).module
        if len(args) == 2:
            args = Python.tuple(module, args[1])
        elif len(args) == 3:
            args = Python.tuple(module, args[1], args[2])
        elif len(args) == 1:
            args = Python.tuple(module)
        else:
            raise Error("TypeError: mojolearn Cholesky: an operation takes at most three arguments")
    if operation == CHOL_OPN_FAMILY_CHOL_FIT:
        var depth = estimator_fit_begin(fields)
        try:
            _chol_fit(model, args, context)
        finally:
            estimator_fit_end(model, depth)
        py_setitem(fields, nk[].k__mojolearn_fitted, PythonObject(True))
        return model
    if operation == CHOL_OPN_FAMILY_CHOL_SOLVE:
        return _chol_solve(model, args, context)
    if operation == CHOL_OPN_FAMILY_CHOL_PAR_FIT:
        return _chol_par(model, args, False, context)
    if operation == CHOL_OPN_FAMILY_CHOL_PAR_SOLVE:
        return _chol_par(model, args, True, context)
    # Unreachable from the family door (every leaf is handled above).
    raise Error("RuntimeError: unknown linalg operation " + String(operation))
