# SPDX-License-Identifier: Apache-2.0
# PACK of the kernel methods door (bindings/kernel_methods_native_door.mojo):
# the public arrays an operation writes (allocated before RUN, which receives
# their addresses), the fitted fields of KernelRidge, Nystroem and RBFSampler
# (published together after RUN), the fitted X the model owns, the pickle
# state, and the ownership record of KernelRidge's resident device copy
# (`_resident_holder`). It also packs a typed request back into the two
# Python lists the entries of a module without the provider take.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from providers.python.neural_descriptor_pyapi_door import (
    neural_descriptor_adopt, neural_descriptor_resource_close, neural_descriptor_entry,
)
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_setattr
from bindings.classical_arrays_door import classical_address
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_python_shape
from bindings.mode_metadata_native_door import _fields
from bindings.public_attributes_native_door import _string_keys
from bindings.family_common_pack import family_call, family_array
from bindings.svm_keys_unpack import svm_keys
from bindings.kernel_methods_boundary_unpack import KmRoute, KmUnpacked, km_int, km_tuple
from kernel_methods.native_call_run import KmCall, KM_OP_KRR_PREDICT_RESIDENT

comptime _HOLDER = "_resident_holder"
comptime _RELEASE = "kernel_ridge_resident_release"


@fieldwise_init
struct KmArray(Movable):
    """A public array and its address for RUN."""
    var value: PythonObject
    var address: Int


def km_vector(count: Int, typestr: String, context: PythonObject) raises -> KmArray:
    var value = family_array(Python.tuple(PythonObject(count)), typestr, context)
    return KmArray(value, Int(py=classical_address(value)))


def km_matrix(rows: Int, cols: Int, context: PythonObject) raises -> KmArray:
    """`rows` x `cols` zeroed float32 cells."""
    var value = family_array(Python.tuple(PythonObject(rows), PythonObject(cols)), "<f4", context)
    return KmArray(value, Int(py=classical_address(value)))


def km_krr_public(rows: Int, targets: Int, squeeze: Bool, context: PythonObject) raises -> KmArray:
    """dual_coef_ or a prediction: (rows,) for a 1-D target, else (rows, targets)."""
    if squeeze:
        return km_vector(rows, "<f4", context)
    return km_matrix(rows, targets, context)


def km_listed(entry: PythonObject, call: KmCall) raises -> PythonObject:
    """A module without the provider (`fast`): the request as its entry's
    addresses list and parameters list (a float for every slot of
    `call.listed`, an int otherwise), the resident handle first where the
    request leads with one. The entry's own answer."""
    var first = 1 if call.operation == KM_OP_KRR_PREDICT_RESIDENT else 0
    var addresses = Python.list()
    for i in range(first, len(call.a)):  # small-loop(at most nine fixed native buffer slots)
        _ = py_call(addresses.append, PythonObject(call.a[i]))
    var params = Python.list()
    for i in range(len(call.p)):  # small-loop(at most ten scalar parameter slots)
        if (call.listed & (1 << i)) != 0:
            _ = py_call(params.append, PythonObject(call.f[i]))
        else:
            _ = py_call(params.append, PythonObject(call.p[i]))
    if first == 1:
        return family_call(entry, PythonObject(call.a[0]), addresses, params)
    return family_call(entry, addresses, params)


# ---------------------------------------------------------------- fitted state


def km_fit_owned(x_meta: ArrayMetadata, x_pin: ArrayPin, context: PythonObject) raises -> PythonObject:
    """`x_meta`'s float32 C-order rows as an Array the model owns: the
    snapshot's own copy when the admission made one, else one memcpy (the
    caller's own Array is never kept, so freezing the fit never freezes it)."""
    if x_meta.copied and x_meta.owned_writable:
        return x_pin.keep
    var result = family_array(array_python_shape(x_meta), "<f4", context)
    unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))),
           src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=x_meta.source.addr), count=x_meta.count)
    return result


def km_krr_fit_pack(model: PythonObject, u: KmUnpacked, kept: PythonObject, dual: PythonObject,
                    info: Int) raises:
    """KernelRidge's fitted fields. `kept` (X_fit_) is an owned copy, never
    the caller's own X (hazard class A, caller-aliased inputs, lane
    p3-svm-gp-kernel): the resident predict freezes X_fit_ read-only, which
    must never reach the caller's array. The last fit's resident arrays are
    released; the next predict uploads the new ones once."""
    var nk = svm_keys()
    var fields = u.fields
    py_setitem(fields, nk[].k_X_fit_, kept)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(u.cols))
    py_setitem(fields, nk[].k_n_targets_, PythonObject(u.targets))
    py_setitem(fields, nk[].k__squeeze, PythonObject(u.squeeze))
    py_setitem(fields, nk[].k_dual_coef_, dual)
    py_setitem(fields, nk[].k_info_, PythonObject(info))
    py_setitem(fields, nk[].k__kernel_params, Python.tuple(
        PythonObject(u.kernel.kernel), PythonObject(u.kernel.degree), PythonObject(u.kernel.gamma),
        PythonObject(u.kernel.coef0), PythonObject(u.alpha)))
    km_resident_release(model, _RELEASE)
    py_setitem(fields, _HOLDER, Python.dict())


def km_nys_fit_pack(u: KmUnpacked, components: PythonObject, indices: PythonObject, normalization: PythonObject,
                    eigenvalues: PythonObject, eigenvectors: PythonObject, sweeps: Int) raises:
    var nk = svm_keys()
    var fields = u.fields
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(u.cols))
    py_setitem(fields, nk[].k_components_, components)
    py_setitem(fields, nk[].k_component_indices_, indices)
    py_setitem(fields, nk[].k_normalization_, normalization)
    py_setitem(fields, nk[].k_eigenvalues_, eigenvalues)
    py_setitem(fields, nk[].k_eigenvectors_, eigenvectors)
    py_setitem(fields, nk[].k_sweeps_, PythonObject(sweeps))
    py_setitem(fields, nk[].k__kernel_params, Python.tuple(
        PythonObject(u.kernel.kernel), PythonObject(u.kernel.degree), PythonObject(u.kernel.gamma),
        PythonObject(u.kernel.coef0), PythonObject(u.seed)))


def km_rbf_fit_pack(u: KmUnpacked, weights: PythonObject, offset: PythonObject, sigma: Float64,
                    scale: Float64) raises:
    var nk = svm_keys()
    var fields = u.fields
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(u.cols))
    py_setitem(fields, nk[].k_random_weights_, weights)
    py_setitem(fields, nk[].k_random_offset_, offset)
    py_setitem(fields, nk[].k_sigma_, PythonObject(sigma))
    py_setitem(fields, nk[].k_scale_, PythonObject(scale))
    py_setitem(fields, nk[].k__params, Python.tuple(PythonObject(u.kernel.gamma), PythonObject(u.seed)))


def km_getstate(model: PythonObject) raises -> PythonObject:
    var state = py_call(native_loaded_module("builtins").dict, _fields(model))
    _ = py_call(state.pop, PythonObject(_HOLDER), PythonObject(None))
    _ = py_call(state.pop, "_native_kernel_method_provider", PythonObject(None))
    return state


def km_setstate(model: PythonObject, state: PythonObject) raises:
    var fields = _fields(model)
    if py_type(state) is not native_loaded_module("builtins").dict:
        raise Error("TypeError: kernel state requires an ordinary native dictionary")
    _string_keys(state)
    _ = py_call(fields.update, state)
    py_setitem(fields, _HOLDER, Python.dict())


# ------------------------------------------- KernelRidge's resident device copy
# KernelRidge.predict reads X_fit_ and dual_coef_ from a device copy minted at
# the first predict after a fit (`kernel_ridge_resident_prepare`,
# kernel_methods/krr_resident.mojo) instead of uploading both per call. The
# holder records (key, handle, minting binding[, resource owner, provider]);
# the key retains X_fit_ and dual_coef_ by identity with the shape they were
# admitted with.


def km_resident_door(route: KmRoute) raises -> PythonObject:
    """The prepare entry of a module without the provider, or None where it
    registers none (and on the provider route, whose flag word answers)."""
    if route.described():
        return PythonObject(None)
    return neural_descriptor_entry(route.binding, "kernel_ridge_resident_prepare", False)


def km_resident_release(model: PythonObject, release: String) raises:
    """Drop the resident handle, if held, through the minting binding's
    `release` entry. Quiet on a binding that cannot be reached any more and
    on a handle already released (a copied instance)."""
    var b = native_loaded_module("builtins")
    var holder = py_get(py_fields(model), PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return
    var nk = svm_keys()
    var held = py_get(holder, nk[].k_index, PythonObject(None))
    if held is None:
        return
    if py_type(held) is not py_type(Python.tuple()):
        raise Error("TypeError: kernel resident ownership must be a native tuple")
    if len(held) != 3 and len(held) != 5:
        raise Error("ValueError: kernel resident ownership has an invalid size")
    py_setitem(holder, nk[].k_index, PythonObject(None))
    try:
        if len(held) == 5:
            neural_descriptor_resource_close(held[3], [])
            return
        _ = py_call(neural_descriptor_entry(held[2], release), PythonObject(km_int(held[1], "resident handle")))
    except:
        pass


def _km_freezable(value: PythonObject) raises -> Bool:
    """True for an exact mojolearn Array that owns its store (see
    `_km_freeze_fitted`)."""
    var module = py_get(native_loaded_module("sys").modules, "mojolearn._array")
    return module is not None and py_type(value) is module.Array and value._base is None


def _km_freeze_fitted(value: PythonObject) raises -> Bool:
    """Make a fitted array read-only at the buffer level before its device
    copy is taken (hazard class A, lane p3-svm-gp-kernel): an in-place write
    (`model.dual_coef_[...] = v`, `np.asarray(model.X_fit_)[:] = v`) then
    raises the normal read-only error instead of leaving the resident copy
    stale; replacing the attribute is noticed by identity and re-uploads.
    Only an exact mojolearn Array owning its store can be frozen (its views
    and the NumPy views of it inherit the flag); anything else returns False
    and the predict keeps its per-call upload."""
    if not _km_freezable(value):
        return False
    py_setattr(value, "_readonly", PythonObject(True))
    return True


def km_resident_hit(model: PythonObject, route: KmRoute, prepare: PythonObject,
                    x_fit_obj: PythonObject, dual_obj: PythonObject, d: Int, t: Int) raises -> Int:
    """The held resident upload's row count when it matches exactly what
    `km_resident_state` would accept for these fitted objects, else -1. Never
    releases or raises on a malformed holder: the full path does that.
    `prepare` is `km_resident_door(route)`."""
    if not route.described():
        if prepare is None:
            return -1
    elif (route.flags & 1) == 0:
        return -1
    var b = native_loaded_module("builtins")
    var holder = py_get(py_fields(model), PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return -1
    var nk = svm_keys()
    var held = py_get(holder, nk[].k_index, PythonObject(None))
    if held is None or py_type(held) is not b.tuple or (len(held) != 3 and len(held) != 5):
        return -1
    var key = held[0]
    if py_type(key) is not b.tuple or len(key) != 5 or py_type(held[1]) is not b.int:
        return -1
    if py_type(key[2]) is not b.int or py_type(key[3]) is not b.int or py_type(key[4]) is not b.int:
        return -1
    if (key[0] is x_fit_obj and key[1] is dual_obj and held[2] is route.binding
            and Int(py=key[3]) == d and Int(py=key[4]) == t and t >= 1
            and ((not route.described() and len(held) == 3)
                 or (route.described() and len(held) == 5 and held[4] is route.native))):
        return Int(py=key[2])
    return -1


def km_resident_handle(model: PythonObject) raises -> Int:
    """The handle of the upload `km_resident_hit` just matched."""
    var nk = svm_keys()
    return Int(py=py_get(py_get(py_fields(model), PythonObject(_HOLDER)), nk[].k_index)[1])


comptime KM_RESIDENT_NONE = 0     # per-call upload (no resident door, or arrays that cannot be frozen)
comptime KM_RESIDENT_HELD = 1     # the answered handle is the held upload
comptime KM_RESIDENT_PREPARE = 2  # fitted arrays frozen: RUN prepares, then `km_resident_keep`


def km_resident_state(model: PythonObject, route: KmRoute, prepare: PythonObject,
                      x_fit_obj: PythonObject, dual_obj: PythonObject,
                      n: Int, d: Int, t: Int) raises -> Tuple[Int, Int]:
    """The device copy of the fit's X and dual: (HELD, handle) while
    `X_fit_` and `dual_coef_` are the same objects (the held key keeps
    references to them) and the binding is the minting one; (PREPARE, 0)
    when it must be uploaded (a stale upload was released and the fitted
    arrays are now read-only); (NONE, 0) where the binding has no resident
    door. Only the holder dictionary changes."""
    if not route.described():
        if prepare is None:
            # The module this estimator was handed (its binding route) keeps
            # no resident fit: a host module registers fit / predict /
            # transform only, and the predict takes the per-call entry of
            # that same module. A device copy minted by ANOTHER module
            # (set_params(device=...) on a fitted estimator) is released
            # here through its minting binding instead of outliving the
            # route; `km_resident_hit` never matches it (held[2] is not
            # this binding).
            km_resident_release(model, _RELEASE)
            return (KM_RESIDENT_NONE, 0)
    elif (route.flags & 1) == 0:
        return (KM_RESIDENT_NONE, 0)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var holder = py_get(fields, PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        holder = Python.dict()
        py_setitem(fields, _HOLDER, holder)
    var nk = svm_keys()
    var held = py_get(holder, nk[].k_index, PythonObject(None))
    if held is not None:
        if py_type(held) is not py_type(Python.tuple()):
            raise Error("TypeError: kernel resident ownership must be a native tuple")
        if len(held) != 3 and len(held) != 5:
            raise Error("ValueError: kernel resident ownership has an invalid size")
        var key = km_tuple(held[0], 5, "resident key")
        var handle = km_int(held[1], "resident handle")
        if (key[0] is x_fit_obj and key[1] is dual_obj and held[2] is route.binding and km_int(key[2], "resident rows") == n
                and km_int(key[3], "resident features") == d and km_int(key[4], "resident targets") == t
                and ((not route.described() and len(held) == 3)
                     or (route.described() and len(held) == 5 and held[4] is route.native))):
            return (KM_RESIDENT_HELD, handle)
        km_resident_release(model, _RELEASE)
    # A device copy is taken only of fitted arrays frozen read-only; an
    # array that cannot be frozen keeps the per-call upload (correct first).
    if not _km_freezable(x_fit_obj) or not _km_freezable(dual_obj):
        return (KM_RESIDENT_NONE, 0)
    _ = _km_freeze_fitted(x_fit_obj)
    _ = _km_freeze_fitted(dual_obj)
    return (KM_RESIDENT_PREPARE, 0)


def km_resident_keep(model: PythonObject, route: KmRoute, x_fit_obj: PythonObject, dual_obj: PythonObject,
                     n: Int, d: Int, t: Int, handle: Int) raises:
    """Record the upload RUN just prepared. On the provider route the handle
    is adopted as a native resource the holder owns (closed by operation 5)."""
    var nk = svm_keys()
    var holder = py_get(py_fields(model), PythonObject(_HOLDER))
    var key = Python.tuple(x_fit_obj, dual_obj, PythonObject(n), PythonObject(d), PythonObject(t))
    if route.described():
        var owner = neural_descriptor_adopt(route.native, handle, 5, [])
        py_setitem(holder, nk[].k_index, Python.tuple(key, PythonObject(handle), route.binding, owner, route.native))
    else:
        py_setitem(holder, nk[].k_index, Python.tuple(key, PythonObject(handle), route.binding))
