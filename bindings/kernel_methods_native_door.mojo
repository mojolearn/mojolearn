# SPDX-License-Identifier: Apache-2.0
# KernelRidge `fit`/`predict`/`score`, Nystroem and RBFSampler
# `fit`/`transform`/`fit_transform` (python/mojolearn/kernel_methods.py), in
# three parts:
#   UNPACK  bindings/kernel_methods_boundary_unpack.mojo: the binding route
#           of the call, the public methods' refusals in their wording and
#           order (kernel, degree, gamma, coef0, alpha, the chi2 kernels'
#           negative input, the precomputed square), the cross-check of the
#           binary's compile-time mode, the fitted-state checks, typed
#           scalars and the pins of every buffer;
#   RUN     kernel_methods/native_call_run.mojo: one typed request per named
#           operation of the provider table (addresses, integer words,
#           Float64 words in the order bindings/_mojolearn_kernel_methods.mojo
#           documents), called with the GIL released; no Python object;
#   PACK    bindings/kernel_methods_boundary_pack.mojo: the public arrays,
#           the fitted fields, the resident-copy ownership record, the pickle
#           state.
# This file sequences them and holds the public lifecycle envelope, the
# dispatch and the two multi-GPU drivers. The same kernels on the same
# inputs: the bits do not move.
#
# A module without the provider (`fast`) takes the same typed request packed
# into its entries' two lists (`km_listed`). Lane migrate-classical-b2:
# KernelRidge.predict reads X_fit_ and dual_coef_ from a device copy minted
# at the first predict after a fit; Nystroem.fit_transform uploads X once
# (`nystroem_fit_transform_ptr`); RBFSampler.fit / fit_transform are one
# native call, gamma='scale' rounded from the binding's exact limbs in exact
# integer math (core/scale_gamma_exact.mojo).
from core.mode_answer import MODE_FAMILY_KERNEL
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family_door import admit_family, admit_family_meta, admit_family_fold_meta
from bindings.admitted_input_unpack import admit_x_unrecorded, admitted_finite_refuse, admit_rows_host_unchecked
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.device_rows_door_run import DoorRowsStore
from providers.python.estimator_binding_pyapi_door import estimator_arguments, estimator_binding
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_fit_begin, estimator_fit_end
from providers.python.kernel_public_lifecycle_pyapi_unpack import kernel_public_finite, kernel_public_ignored_target, kernel_public_refusal, kernel_public_feature_snapshot, kernel_public_fit_complete
from providers.python.kernel_public_lifecycle_pyapi_pack import kernel_public_admit, kernel_public_result
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from core.neural_descriptor_run import neural_descriptor_record_admit
from bindings.mcls_cluster_door import mcls_cluster_operation
from bindings.operation_names_mcls import MCLS_OPN_FAMILY_MCLS_R2
from bindings.operation_names_km import (
    KM_OPN_FAMILY_KM_GETSTATE, KM_OPN_FAMILY_KM_SETSTATE, KM_OPN_FAMILY_KM_PAR_FIT, KM_OPN_FAMILY_KM_PAR_APPLY,
    KM_OPN_FAMILY_KM_KRR_FIT, KM_OPN_FAMILY_KM_KRR_PREDICT, KM_OPN_FAMILY_KM_KRR_SCORE, KM_OPN_FAMILY_KM_NYS_FIT,
    KM_OPN_FAMILY_KM_NYS_FIT_TRANSFORM, KM_OPN_FAMILY_KM_NYS_TRANSFORM, KM_OPN_FAMILY_KM_RBF_FIT,
    KM_OPN_FAMILY_KM_RBF_FIT_TRANSFORM, KM_OPN_FAMILY_KM_RBF_TRANSFORM,
)
from bindings.parallel_plan_door import plan_devices, plan_enter, plan_exit, plan_field, plan_require_identical
from core.device_plan import DEVICE_PLAN_WORKER, DEVICE_PLAN_CHOLESKY
from bindings.py_balanced import py_type, py_get, py_fields, py_item
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_pack import family_call
from bindings.svm_keys_unpack import svm_keys
from bindings.kernel_methods_boundary_unpack import (
    KmRoute, KmUnpacked, km_int, km_listed_entry,
    km_krr_fit_unpack, km_krr_predict_unpack, km_krr_fitted_unpack,
    km_nys_fit_unpack, km_nys_transform_unpack, km_rbf_fit_unpack, km_rbf_transform_unpack,
)
from bindings.kernel_methods_boundary_pack import (
    km_vector, km_matrix, km_krr_public, km_listed, km_fit_owned,
    km_krr_fit_pack, km_nys_fit_pack, km_rbf_fit_pack, km_getstate, km_setstate,
    km_resident_door, km_resident_hit, km_resident_handle, km_resident_state, km_resident_keep,
    KM_RESIDENT_HELD, KM_RESIDENT_PREPARE,
)
from kernel_methods.native_call_run import (
    KmCall, km_call_admit, km_call_run,
    km_krr_fit_call, km_krr_predict_call, km_krr_prepare_call,
    km_nys_fit_call, km_nys_transform_call, km_rbf_fit_call, km_rbf_transform_call, km_rbf_joint_call,
    km_scale_gamma_words, km_scale_gamma_run, km_scale_gamma_value,
    KM_OP_KRR_PREDICT, KM_OP_KRR_PREDICT_PTR, KM_OP_KRR_PREDICT_RESIDENT,
    KM_OP_NYS_FIT_TRANSFORM_PTR, KM_OP_NYS_FIT, KM_OP_NYS_FIT_PTR, KM_OP_SCALE_GAMMA_LIMBS,
)


def _km_run(route: KmRoute, entry: PythonObject, call: KmCall) raises -> PythonObject:
    """RUN one request on the route: the provider table with the GIL
    released (no Python list, no Python scalar), or, for a module without
    the provider, its list `entry` (`km_listed`). The entry's answer."""
    if route.described():
        km_call_admit(route.provider_id, call)
        var result = 0
        with GILReleased(Python()):
            result = km_call_run(route.provider_id, call)
        return PythonObject(result)
    return km_listed(entry, call)


def _km_scale_gamma(route: KmRoute, x_meta: ArrayMetadata, d: Int) raises -> Float64:
    """`_scale_gamma.scale_gamma_x` (admitted by `km_scale_gamma_admit`): the
    module's exact limbs of the float32 cells into words RUN owns, the one
    rounding in exact integer math."""
    var words = km_scale_gamma_words()
    var cells = x_meta.count
    if route.described():
        neural_descriptor_record_admit(route.provider_id, KM_OP_SCALE_GAMMA_LIMBS)
        with GILReleased(Python()):
            km_scale_gamma_run(route.provider_id, x_meta.source.addr, cells, words)
    else:
        _ = family_call(neural_descriptor_entry(route.binding, "scale_gamma_limbs"), PythonObject(x_meta.source.addr),
                        PythonObject(cells), PythonObject(Int(words.unsafe_ptr())))
        _ = words[0]  # the entry wrote through the address above
    return km_scale_gamma_value(words, cells, d)


def _krr_fit(model: PythonObject, args: PythonObject, context: PythonObject, x_meta: ArrayMetadata,
             x_pin: ArrayPin, fold: DoorRowsStore) raises:
    """`KernelRidge.fit(X, y, sample_weight)`.

    X (lane a8a): `fold` active is a cross-validation fold (`x_meta` then
    holds no host words) that `kernel_ridge_fit_ptr` gathers on the device
    and whose device copy comes home as `X_fit_` (no host gather); a route
    that cannot take it (UNPACK's `gather`) gathers its host rows here
    instead. The pointer entry tests X on the device and returns its status
    words; the other routes are tested by UNPACK; one refusal wording either
    way."""
    classical_connect_numpy(context)
    var where = String("KernelRidge")
    var u = km_krr_fit_unpack(model, args, context, x_meta, fold.active())
    if u.gather:
        # This route reads host words: the fold's host rows, gathered in
        # parallel out of model selection's host base (unchecked: X is
        # tested by UNPACK like any host X).
        var rows_pin = ArrayPin()
        var rows = admit_family_meta(admit_rows_host_unchecked(args[1], context), 2, "X", context, rows_pin, True)
        _krr_fit(model, args, context, rows, rows_pin, DoorRowsStore.none())
        _ = u
        return
    var n = u.rows
    var d = u.cols
    var t = u.targets
    var described = u.route.described()
    var dual = km_krr_public(n, t, u.squeeze, context)
    var scalars = List[Float64](length=1, fill=0.0)
    # A fold's X_fit_ is the entry's device copy of the gathered rows.
    var fold_fit = PythonObject(None)
    var fold_fit_address = 0
    if fold.active():
        var gathered = km_matrix(n, d, context)
        fold_fit = gathered.value
        fold_fit_address = gathered.address
    var pointer_operation = u.pointer
    if described:
        pointer_operation = (u.route.flags & 1) != 0
    var call = km_krr_fit_call(pointer_operation, x_meta.source.addr, u.a[0], dual.address,
                               Int(scalars.unsafe_ptr()), n, d, t, u.kernel, u.alpha)
    if u.pointer:
        call.address(u.a[1])
        call.address(fold_fit_address)
        call.address(fold.base_id)
        call.address(fold.base_rows)
        call.address(fold.index_id)
    elif u.weighted:
        call.address(u.a[1])
    var answer = _km_run(u.route, u.entry, call)
    _ = scalars[0]  # the entry wrote through the address above
    var info = answer
    if u.pointer:
        # kernel_ridge_fit_ptr's [info, finite, range] words (lane a8a)
        if fold.active():
            door_rows_refuse(Int(py=answer[1]), Int(py=answer[2]), where, fold.base_rows)
        else:
            admitted_finite_refuse(Int(py=answer[1]), where)
        info = answer[0]
    var fitted_info = km_int(info, "info_")
    var kept = fold_fit if fold.active() else km_fit_owned(x_meta, x_pin, context)
    km_krr_fit_pack(model, u, kept, dual.value, fitted_info)
    # The owners of y, the weights and the public arrays outlive RUN and PACK.
    _ = dual
    _ = u
    _ = x_meta
    _ = x_pin


def _krr_resident_predict(u: KmUnpacked, fitted_rows: Int, handle: Int, xq_meta: ArrayMetadata,
                          context: PythonObject) raises -> PythonObject:
    """ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_resident_binding:
    handle; x_new, out; the predict's ten params."""
    var dst = km_krr_public(u.rows, u.targets, u.squeeze, context)
    var call = km_krr_predict_call(KM_OP_KRR_PREDICT_RESIDENT, fitted_rows, u.cols, u.targets, u.kernel, u.alpha,
                                   u.info, u.rows)
    call.address(handle)
    call.address(xq_meta.source.addr)
    call.address(dst.address)
    var words = _km_run(u.route, km_listed_entry(u.route, "kernel_ridge_predict_resident", context), call)
    if not u.route.described():
        admitted_finite_refuse(Int(py=words[0]), "KernelRidge")
    return dst.value


def _krr_predict(model: PythonObject, args: PythonObject, context: PythonObject,
                 xq_meta: ArrayMetadata, xq_pin: ArrayPin, resident: Bool = True,
                 fold: DoorRowsStore = DoorRowsStore.none()) raises -> PythonObject:
    """`KernelRidge.predict(X)`: K(X, X_fit_) @ dual_coef_. `resident=False`
    (the multi-GPU driver) skips the device copy of the fit: a copy minted on
    a device plan's root would outlive the call on that device. The resident
    and the per-call predict run the same kernels on the same bytes.

    The query (lane a7a): `fold` active is a cross-validation fold
    (`xq_meta` then holds no host words) that `kernel_ridge_predict_ptr`
    gathers on the device out of model selection's resident base (no host
    gather, no upload of X) with the fitted X_fit_ and dual_coef_ uploaded
    from their host words; a route that cannot take it (UNPACK's `gather`)
    gathers its host rows here instead. The GPU entries test the query and
    return its status words; the other routes are tested by UNPACK; one
    refusal wording either way."""
    classical_connect_numpy(context)
    var where = String("KernelRidge")
    var u = km_krr_predict_unpack(model, args[0], context, xq_meta, fold.active())
    var q = u.rows
    var d = u.cols
    var t = u.targets
    var described = u.route.described()
    if fold.active():
        if u.gather:
            # This route reads host words: the fold's host rows, gathered in
            # parallel out of model selection's host base (unchecked: the
            # query is tested by UNPACK like any host query).
            var rows_pin = ArrayPin()
            var rows = admit_family_meta(admit_rows_host_unchecked(args[1], context), 2, "X", context, rows_pin, True)
            var fold_out = _krr_predict(model, args, context, rows, rows_pin, resident)
            _ = rows_pin^
            return fold_out
        km_krr_fitted_unpack(u, True, context)
        var fold_dst = km_krr_public(q, t, u.squeeze, context)
        var fold_call = km_krr_predict_call(KM_OP_KRR_PREDICT_PTR, u.count, d, t, u.kernel, u.alpha, u.info, q)
        # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_ptr_binding:
        # x_fit, dual, x_new (0: the fold), out, then the fold's store ids.
        fold_call.address(u.a[0])
        fold_call.address(u.a[1])
        fold_call.address(0)
        fold_call.address(fold_dst.address)
        fold_call.address(fold.base_id)
        fold_call.address(fold.base_rows)
        fold_call.address(fold.index_id)
        var fold_words = km_listed(u.entry, fold_call)
        door_rows_refuse(Int(py=fold_words[0]), Int(py=fold_words[1]), where, fold.base_rows)
        _ = u
        return fold_dst.value
    var prepare = PythonObject(None)
    if resident:
        # Resident hit first: the held key retains X_fit_ and dual_coef_ by
        # identity with the shape they were admitted with at upload, so the
        # two fitted-array snapshots are needed only to upload again.
        prepare = km_resident_door(u.route)
        var held_rows = km_resident_hit(model, u.route, prepare, u.x_fit, u.dual, d, t)
        if held_rows >= 0:
            var hit = _krr_resident_predict(u, held_rows, km_resident_handle(model), xq_meta, context)
            _ = u
            _ = xq_meta
            _ = xq_pin
            return hit
    km_krr_fitted_unpack(u, False, context)
    var n = u.count
    if resident:
        var state = km_resident_state(model, u.route, prepare, u.x_fit, u.dual, n, d, t)
        var handle = state[1]
        if state[0] == KM_RESIDENT_PREPARE:
            var upload = km_krr_prepare_call(u.a[0], u.a[1], n, d, t)
            handle = Int(py=_km_run(u.route, prepare, upload))
            km_resident_keep(model, u.route, u.x_fit, u.dual, n, d, t, handle)
        if state[0] == KM_RESIDENT_HELD or state[0] == KM_RESIDENT_PREPARE:
            var held = _krr_resident_predict(u, n, handle, xq_meta, context)
            _ = u
            _ = xq_meta
            _ = xq_pin
            return held
    var dst = km_krr_public(q, t, u.squeeze, context)
    var pointer_operation = u.pointer
    var entry = u.entry
    if described:
        pointer_operation = (u.route.flags & 1) != 0
    elif not u.pointer:
        entry = km_listed_entry(u.route, "kernel_ridge_predict", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_binding:
    # x_fit, dual, x_new, out; n, d, t, kernel, degree, gamma, coef0, alpha, info, q.
    var call = km_krr_predict_call(KM_OP_KRR_PREDICT_PTR if pointer_operation else KM_OP_KRR_PREDICT,
                                   n, d, t, u.kernel, u.alpha, u.info, q)
    call.address(u.a[0])
    call.address(u.a[1])
    call.address(xq_meta.source.addr)
    call.address(dst.address)
    var words = _km_run(u.route, entry, call)
    if u.pointer:
        # kernel_ridge_predict_ptr's [finite, range] words (lane a7a)
        admitted_finite_refuse(Int(py=words[0]), where)
    _ = u
    _ = xq_meta
    _ = xq_pin
    return dst.value


def _nys_fit(model: PythonObject, args: PythonObject, context: PythonObject,
             x_meta: ArrayMetadata, x_pin: ArrayPin, transform: Bool = False) raises -> PythonObject:
    """`Nystroem.fit(X)`: the basis rows and the normalization. With
    `transform` (`fit_transform`), returns the transform of X: one
    `nystroem_fit_transform_ptr` call (X uploaded once) where the binding
    registers it, else the fit and then the transform of the same admitted X."""
    classical_connect_numpy(context)
    var u = km_nys_fit_unpack(model, args[0], transform, context, x_meta)
    var n = u.rows
    var d = u.cols
    var q = u.count
    var components = km_matrix(q, d, context)
    var indices = km_vector(q, "<i4", context)
    var normalization = km_matrix(q, q, context)
    var eigenvalues = km_vector(q, "<f4", context)
    var eigenvectors = km_matrix(q, q, context)
    var scalars = List[Float64](length=1, fill=0.0)
    var operation = KM_OP_NYS_FIT
    if u.joint:
        operation = KM_OP_NYS_FIT_TRANSFORM_PTR
    elif u.route.described() and (u.route.flags & 2) != 0:
        operation = KM_OP_NYS_FIT_PTR
    var call = km_nys_fit_call(operation, x_meta.source.addr, components.address, indices.address,
                               normalization.address, eigenvalues.address, eigenvectors.address,
                               Int(scalars.unsafe_ptr()), n, d, u.kernel, q, u.seed)
    var result = PythonObject(None)
    if u.joint:
        # ORDER MATCHES nystroem_fit_transform_ptr_binding: the seven fit
        # addresses, then the n x q transform.
        var joint_dst = km_matrix(n, q, context)
        call.address(joint_dst.address)
        result = joint_dst.value
    var sweeps = km_int(_km_run(u.route, u.entry, call), "sweeps_")
    _ = scalars[0]  # the entry wrote through the address above
    km_nys_fit_pack(u, components.value, indices.value, normalization.value, eigenvalues.value,
                    eigenvectors.value, sweeps)
    if transform and not u.joint:
        var dst = km_matrix(n, q, context)
        var second = km_nys_transform_call(components.address, indices.address, normalization.address,
                                           eigenvalues.address, eigenvectors.address, x_meta.source.addr,
                                           dst.address, q, d, u.kernel, u.seed, sweeps, n)
        _ = _km_run(u.route, km_listed_entry(u.route, "nystroem_transform", context), second)
        result = dst.value
    # The public arrays and X outlive RUN.
    _ = components
    _ = indices
    _ = normalization
    _ = eigenvalues
    _ = eigenvectors
    _ = u
    _ = x_meta
    _ = x_pin
    return result


def _nys_transform(model: PythonObject, args: PythonObject, context: PythonObject, x_meta: ArrayMetadata,
                   x_pin: ArrayPin) raises -> PythonObject:
    """`Nystroem.transform(X)`: K(X, components_) @ normalization_.T."""
    classical_connect_numpy(context)
    var u = km_nys_transform_unpack(model, args[0], context, x_meta)
    var dst = km_matrix(u.rows, u.count, context)
    var call = km_nys_transform_call(u.a[0], u.a[1], u.a[2], u.a[3], u.a[4], x_meta.source.addr, dst.address,
                                     u.count, u.cols, u.kernel, u.seed, u.sweeps, u.rows)
    _ = _km_run(u.route, u.entry, call)
    # The pins of the fitted arrays and X outlive RUN.
    _ = u
    _ = x_meta
    _ = x_pin
    return dst.value


def _rbf_transform(model: PythonObject, args: PythonObject, context: PythonObject, x_meta: ArrayMetadata,
                   x_pin: ArrayPin) raises -> PythonObject:
    """`RBFSampler.transform(X)`: scale_ * cos(X @ random_weights_ + random_offset_)."""
    classical_connect_numpy(context)
    var u = km_rbf_transform_unpack(model, args[0], context, x_meta)
    var dst = km_matrix(u.rows, u.count, context)
    var call = km_rbf_transform_call(u.a[0], u.a[1], x_meta.source.addr, dst.address, u.cols, u.count,
                                     u.kernel.gamma, u.seed, u.sigma, u.scale, u.rows)
    _ = _km_run(u.route, u.entry, call)
    # The pins of the fitted arrays and X outlive RUN.
    _ = u
    _ = x_meta
    _ = x_pin
    return dst.value


def _rbf_fit(model: PythonObject, args: PythonObject, transform: Bool, context: PythonObject, x_meta: ArrayMetadata,
             x_pin: ArrayPin) raises -> PythonObject:
    """`RBFSampler.fit(X)` (returns the model) or, with `transform`,
    `fit_transform(X)`: one `rbf_sampler_fit_transform_resident` call where
    the binding registers it, else the fit and then the transform."""
    classical_connect_numpy(context)
    var u = km_rbf_fit_unpack(model, args[0], transform, context, x_meta)
    var m = u.rows
    var d = u.cols
    var q = u.count
    if u.scaled:
        u.kernel.gamma = _km_scale_gamma(u.route, x_meta, d)
    var gamma = u.kernel.gamma
    var weights = km_matrix(d, q, context)
    var offset = km_vector(q, "<f4", context)
    var scalars = List[Float64](length=2, fill=0.0)
    var result = PythonObject(None)
    if u.joint:
        # lane apple-fast-w2-kfeat MOJOLEARN_KM_FAST_RBF_RESIDENT
        # (kernel_methods/rbf_resident.mojo): the draws, the upload, the
        # projection and every download in one binding call and one wait.
        var joint_dst = km_matrix(m, q, context)
        var joint_call = km_rbf_joint_call(weights.address, offset.address, Int(scalars.unsafe_ptr()),
                                           x_meta.source.addr, joint_dst.address, d, q, gamma, u.seed, m)
        _ = km_listed(u.entry, joint_call)
        result = joint_dst.value
    else:
        var call = km_rbf_fit_call(weights.address, offset.address, Int(scalars.unsafe_ptr()), d, q, gamma, u.seed)
        _ = _km_run(u.route, u.entry, call)
    var sigma = scalars[0]
    var scale = scalars[1]
    km_rbf_fit_pack(u, weights.value, offset.value, sigma, scale)
    if not transform:
        _ = u
        _ = x_meta
        _ = x_pin
        return model
    if not u.joint:
        var dst = km_matrix(m, q, context)
        var second = km_rbf_transform_call(weights.address, offset.address, x_meta.source.addr, dst.address,
                                           d, q, gamma, u.seed, sigma, scale, m)
        _ = _km_run(u.route, km_listed_entry(u.route, "rbf_sampler_transform", context), second)
        result = dst.value
    # The public arrays and X outlive RUN.
    _ = weights
    _ = offset
    _ = u
    _ = x_meta
    _ = x_pin
    return result


# ---------------------------------------------------------------------------
# The multi-GPU drivers parallel_classical.fit_kernel_method and
# apply_kernel_method, one native call each (lane/native-parallel-driver-
# 20261024, plan NATIVE_PARALLEL_DRIVER_20261024 (git history)). They used to send
# the estimator by pickle to ONE cooperative worker process whose visibility
# mask held the selected devices and whose MOJOLEARN_SVM_DEVICE_COUNT and
# MOJOLEARN_CHOLESKY_DEVICE_COUNT held their number. Here the caller's thread
# binds a device plan in the binding (core/device_plan.mojo) around the plain
# owners above: the kernel rows, the factor's trailing rows and the solve's
# columns go to the same owners with the same partition and join order, and
# the root to devices[0]. No bit moves: par-kernel-ridge and par-nystroem
# hash the same arrays (tools/identity_break.py).
# ---------------------------------------------------------------------------


def _km_par_admit(model: PythonObject, context: PythonObject) raises -> Bool:
    """`parallel_classical._admit_kernel_method`, in its order and wording;
    True for a KernelRidge, False for a Nystroem."""
    var nk = svm_keys()
    var classes = py_item(context, nk[].k_family_km_par_classes)
    var cls = py_type(model)
    var ridge = cls is classes[0]
    if not ridge and cls is not classes[1]:
        raise Error("TypeError: requires mojolearn.KernelRidge or Nystroem")
    plan_require_identical(model, "parallel kernel methods require IDENTICAL numeric mode", MODE_FAMILY_KERNEL)
    var kernel = plan_field(model, "kernel")
    if py_type(kernel) is py_type(PythonObject("")) and String(kernel) == "laplacian":
        raise Error("ValueError: parallel kernel methods refuse kernel='laplacian' by name: its L1 "
                    + "distance matrix has no distributed row seam in this lane")
    return ridge


comptime _KM_REBUILD = "rebuild kernel methods binding for distributed kernel rows"


def _km_par_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`fit_kernel_method(estimator, X, y, devices=...)`: args (binding, X, y,
    devices), the binding the plain methods use (`_KernelMethodBase._bind`)."""
    var ridge = _km_par_admit(model, context)
    if ridge and args[2] is None:
        raise Error("ValueError: KernelRidge fit requires y")
    var devices = plan_devices(args[3], "km_fit", True, context)
    var binding = args[0]
    var x_pin = ArrayPin()
    var x = admit_family_meta(args[1], 2, "X", context, x_pin, True)
    plan_enter(binding, devices, _KM_REBUILD, DEVICE_PLAN_WORKER | DEVICE_PLAN_CHOLESKY)
    try:
        if ridge:
            _krr_fit(model, Python.tuple(binding, args[1], args[2], PythonObject(None)), context, x, x_pin,
                     DoorRowsStore.none())
        else:
            _ = _nys_fit(model, Python.tuple(binding, args[1]), context, x, x_pin)
    finally:
        plan_exit(binding)
    _ = x^
    _ = x_pin
    return model


def _km_par_apply(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`apply_kernel_method(estimator, X, devices=...)`: args (binding, X,
    devices). The native result (float32 for KernelRidge.predict); the driver
    applies the plain method's public conversion for its own caller."""
    var ridge = _km_par_admit(model, context)
    var devices = plan_devices(args[2], "km_apply", True, context)
    var binding = args[0]
    var x_pin = ArrayPin()
    var x = admit_family_meta(args[1], 2, "X", context, x_pin, True)
    var out = PythonObject(None)
    plan_enter(binding, devices, _KM_REBUILD, DEVICE_PLAN_WORKER | DEVICE_PLAN_CHOLESKY)
    try:
        if ridge:
            out = _krr_predict(model, Python.tuple(binding, args[1]), context, x, x_pin, False)
        else:
            out = _nys_transform(model, Python.tuple(binding, args[1]), context, x, x_pin)
    finally:
        plan_exit(binding)
    _ = x^
    _ = x_pin
    return out

def kernel_methods_unknown_operation(operation: String) raises:
    """A `family_km_*` name the operation table does not hold (the shells
    pass table names only): the text the name chain raised."""
    raise Error("RuntimeError: unknown kernel methods operation " + operation)


def _km_method(operation: Int) -> String:
    """The public method an operation serves, for the not-fitted text (a
    static literal; the old door built it per call from four suffix tests)."""
    if operation == KM_OPN_FAMILY_KM_NYS_FIT_TRANSFORM or operation == KM_OPN_FAMILY_KM_RBF_FIT_TRANSFORM:
        return "fit_transform"
    if (operation == KM_OPN_FAMILY_KM_KRR_FIT or operation == KM_OPN_FAMILY_KM_NYS_FIT
            or operation == KM_OPN_FAMILY_KM_RBF_FIT):
        return "fit"
    if operation == KM_OPN_FAMILY_KM_KRR_PREDICT:
        return "predict"
    if operation == KM_OPN_FAMILY_KM_KRR_SCORE:
        return "score"
    return "transform"


def kernel_methods_operation(model: PythonObject, operation: Int, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """`operation` is the leaf code of bindings/operation_names_km.mojo
    (lane bw2-svm: one range test or integer compare per step; the door
    resolved the caller's str once)."""
    if operation == KM_OPN_FAMILY_KM_PAR_FIT or operation == KM_OPN_FAMILY_KM_PAR_APPLY:
        var admitted = estimator_arguments(model, "_mojolearn_kernel_methods", args, context)
        if operation == KM_OPN_FAMILY_KM_PAR_FIT:
            return _km_par_fit(model, admitted, context)
        return _km_par_apply(model, admitted, context)
    if operation == KM_OPN_FAMILY_KM_GETSTATE:
        return km_getstate(model)
    if operation == KM_OPN_FAMILY_KM_SETSTATE:
        km_setstate(model, args[0])
        return PythonObject(None)
    # The fits: the old suffix test `endswith("fit") or endswith("fit_transform")`.
    var fitting = (operation == KM_OPN_FAMILY_KM_KRR_FIT or operation == KM_OPN_FAMILY_KM_NYS_FIT
                   or operation == KM_OPN_FAMILY_KM_RBF_FIT or operation == KM_OPN_FAMILY_KM_NYS_FIT_TRANSFORM
                   or operation == KM_OPN_FAMILY_KM_RBF_FIT_TRANSFORM)
    var fields = kernel_public_admit(model, _km_method(operation), fitting, args[1], context)
    # Lane a7a: a KernelRidge predict / score of a cross-validation fold
    # keeps the fold on the device (admit_x `device_x`: its store ids, no
    # host gather); `_krr_predict` gathers it there or, on a route that
    # reads host words, gathers its host rows.
    var fold = DoorRowsStore.none()
    var krr_query = operation == KM_OPN_FAMILY_KM_KRR_PREDICT or operation == KM_OPN_FAMILY_KM_KRR_SCORE
    # Lane a8a: the fit takes a fold on the device too (`_krr_fit`).
    var krr_device_x = krr_query or operation == KM_OPN_FAMILY_KM_KRR_FIT
    var x: ArrayMetadata
    var x_pin = ArrayPin()  # the door's pin of X, held through the operation
    if krr_device_x and door_is_device_rows(args[1], context):
        var fx = admit_x_unrecorded(args[1], "KernelRidge", context, device_x=True, finite=False)
        fold = fx.store
        if fold.active():
            x = admit_family_fold_meta(fx.rows, fx.cols)
        else:
            x = admit_family_meta(fx.array, 2, "X", context, x_pin, True)
    else:
        x = admit_family_meta(args[1], 2, "X", context, x_pin, True)
    if operation >= KM_OPN_FAMILY_KM_RBF_FIT and operation <= KM_OPN_FAMILY_KM_RBF_TRANSFORM:
        kernel_public_finite(x, "X")
    if fitting and not (operation >= KM_OPN_FAMILY_KM_KRR_FIT and operation <= KM_OPN_FAMILY_KM_KRR_SCORE):
        kernel_public_ignored_target(args[2] if len(args) > 2 else PythonObject(None), x.source.rows, context)
    var feature_names = kernel_public_feature_snapshot(args[1], context) if fitting else PythonObject(None)
    var admitted = estimator_arguments(model, "_mojolearn_kernel_methods", args, context)
    var outside = estimator_outside()
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        result = _kernel_methods_operation(model, operation, admitted, context, x, x_pin, fold)
    except error:
        var failure = kernel_public_refusal(error, fitting, depth > 0, x.source.rows, x.source.cols, model)
        raise failure^
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        kernel_public_fit_complete(model, feature_names)
    _ = feature_names^
    _ = x^
    _ = x_pin
    return kernel_public_result(result, operation == KM_OPN_FAMILY_KM_KRR_PREDICT, outside, context)


def _kernel_methods_operation(model: PythonObject, operation: Int, args: PythonObject,
                              context: PythonObject, x: ArrayMetadata, x_pin: ArrayPin,
                              fold: DoorRowsStore = DoorRowsStore.none()) raises -> PythonObject:
    if operation == KM_OPN_FAMILY_KM_KRR_FIT:
        _krr_fit(model, args, context, x, x_pin, fold)
        return model
    if operation == KM_OPN_FAMILY_KM_KRR_PREDICT:
        return _krr_predict(model, args, context, x, x_pin, True, fold)
    if operation == KM_OPN_FAMILY_KM_KRR_SCORE:
        var truth = admit_family(args[2], -1, "y", context, finite=True).array
        var prediction = _krr_predict(model, args, context, x, x_pin, True, fold)
        var nk = svm_keys()
        var mode = py_get(py_fields(model), nk[].k_numeric_mode, PythonObject(None))
        var metrics = estimator_binding(model, "_mojolearn_metrics", context)
        return mcls_cluster_operation(MCLS_OPN_FAMILY_MCLS_R2, Python.tuple(metrics, truth, prediction, args[3],
                                      PythonObject("uniform_average"), PythonObject(True), mode), context)
    if operation == KM_OPN_FAMILY_KM_NYS_FIT:
        _ = _nys_fit(model, args, context, x, x_pin)
        return model
    if operation == KM_OPN_FAMILY_KM_NYS_TRANSFORM:
        return _nys_transform(model, args, context, x, x_pin)
    if operation == KM_OPN_FAMILY_KM_RBF_TRANSFORM:
        return _rbf_transform(model, args, context, x, x_pin)
    if operation == KM_OPN_FAMILY_KM_NYS_FIT_TRANSFORM:
        return _nys_fit(model, args, context, x, x_pin, True)
    if operation == KM_OPN_FAMILY_KM_RBF_FIT:
        return _rbf_fit(model, args, False, context, x, x_pin)
    if operation == KM_OPN_FAMILY_KM_RBF_FIT_TRANSFORM:
        return _rbf_fit(model, args, True, context, x, x_pin)
    raise Error("RuntimeError: unknown kernel methods operation " + String(operation))
