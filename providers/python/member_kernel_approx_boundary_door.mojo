# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the KernelApprox family (Nystroem, RBFSampler):
its place in `bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md;
worked form providers/python/member_linear_boundary_door.mojo). Kind 0 is
Nystroem, kind 1 RBFSampler
(`_mojolearn_member = (MEMBER_FAMILY_KERNEL_APPROX, kind)` on each class).

  * `kernel_approx_member_admit` (UNPACK, once per member): the estimator's
    own parameter checks (bindings/kernel_methods_boundary_unpack.mojo
    `km_nys_fit_unpack` / `km_rbf_fit_unpack`: the kernel name, degree,
    n_components, gamma, coef0, random_state), its binding and mode check
    (`km_extension`), gamma='scale''s capability check, the provider table
    in the route's entry slot, its capability word in `route.flags`, and
    the 9 parameter words of core/kernel_approx_member.mojo. `fast` (no
    provider) is not published: admit returns False.
  * `kernel_approx_member_pack` (PACK): the fitted attributes the
    estimator's own fit writes (`km_nys_fit_pack` / `km_rbf_fit_pack`,
    `kernel_public_fit_complete`), from the member state's blocks.
  * `kernel_approx_member_state` (UNPACK): a member state from a fitted
    Nystroem or RBFSampler.

SERIAL BY NATURE: one member's admission or publication (record reads and
one copy per fitted block); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family_meta
from bindings.family_common_pack import family_field
from bindings.kernel_methods_boundary_pack import km_matrix, km_vector, km_nys_fit_pack, km_rbf_fit_pack
from bindings.kernel_methods_boundary_unpack import (
    KmRoute, KmUnpacked, km_extension, km_int, km_number, km_seed, km_tuple, km_scale_gamma_admit,
    _km_kernel_code, _km_degree, _km_real, _km_params, _km_i32_snapshot,
)
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_type
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    MEMBER_F64, MEMBER_I32,
)
from kernel_methods.descriptor_schema import KERNEL_METHOD_OPERATIONS
from kernel_methods.native_call_run import KmKernel
from providers.python.array_metadata_pyapi_door import ArrayPin
from providers.python.host_estimator_binding_pyapi_door import estimator_member_binding
from providers.python.kernel_method_descriptor_pyapi_door import kernel_method_metadata
from providers.python.kernel_public_lifecycle_pyapi_unpack import kernel_public_fit_complete
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_entry
from x_decomp.descriptor_schema import xd_vendor_code

comptime _B_FIRST = MEMBER_BLOCK_FAMILY


def kernel_approx_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this configuration has no member
    entry (`fast`)."""
    if kind != 0 and kind != 1:
        raise Error("ValueError: mojolearn: unknown kernel approximation member kind " + String(kind))
    var nystroem = kind == 0
    var where = String("Nystroem") if nystroem else String("RBFSampler")
    # An explicit Host* class (its own host-only word, lane mf-iterimp3b): the host image its own methods
    # bind (`estimator_member_binding`); any other class: the estimator's binding route, as before.
    var binding = estimator_member_binding(model, "_mojolearn_kernel_methods", context)
    var words = List[Float64](length=9, fill=0.0)
    words[0] = Float64(kind)
    words[1] = 2.0
    words[2] = 3.0
    var seed = 0
    var q = 0
    var route = KmRoute(binding, PythonObject(None), 0, 0)
    if nystroem:
        # `km_nys_fit_unpack`'s order (the rows-dependent n_components rule
        # runs at fit, in RUN; its UserWarning is raised at PACK, below).
        var kernel = _km_kernel_code(family_field(fields, "kernel"), where, False)
        words[1] = Float64(kernel)
        words[2] = Float64(_km_degree(fields, where))
        q = km_int(family_field(fields, "n_components"), where + ".n_components")
        if q < 1:
            raise Error("mojolearn " + where + ": n_components must be positive, got " + String(q))
        var gamma_source = family_field(fields, "gamma")
        if gamma_source is None:
            words[3] = 1.0
        else:
            words[4] = _km_real(gamma_source, "gamma", where, context)
        words[5] = _km_real(family_field(fields, "coef0"), "coef0", where, context)
        seed = km_seed(family_field(fields, "random_state"), context)
        route = km_extension(fields, binding, where, context)
    else:
        # `km_rbf_fit_unpack`'s order: gamma, n_components, the seed.
        var gamma_source = family_field(fields, "gamma")
        var scaled = py_type(gamma_source) is py_type(PythonObject("")) and String(gamma_source) == "scale"
        if scaled:
            route = km_extension(fields, binding, where, context)
            # The capability half of `km_scale_gamma_admit` (the cell count
            # is tested at fit, in RUN).
            km_scale_gamma_admit(route, 1)
            words[3] = 2.0
        else:
            words[4] = _km_real(gamma_source, "gamma", where, context)
        q = km_int(family_field(fields, "n_components"), where + ".n_components")
        if q < 1:
            raise Error("mojolearn " + where + ": n_components must be positive, got " + String(q))
        seed = km_seed(family_field(fields, "random_state"), context)
        if not scaled:
            route = km_extension(fields, binding, where, context)
    if not route.described():
        return False
    words[6] = Float64(q)
    words[7] = Float64(seed & 0xFFFFFFFF)
    words[8] = Float64(seed >> 32)
    var mode = kernel_method_metadata(route.native, 13)
    var vendor = xd_vendor_code(String(py_call(neural_descriptor_entry(binding, "kernel_methods_vendor"))))
    var entry = route.native.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    var device_route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    device_route.mode = mode
    device_route.vendor = vendor
    device_route.entry = EntryRef(entry, entry_ref_word(mode, vendor, KERNEL_METHOD_OPERATIONS))
    device_route.flags = route.flags
    owner.plan.route = device_route
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, route.native)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _float_array(plan: MemberPlan, mut call: MemberCall, block: Int, rows: Int, cols: Int, vector: Bool,
                 context: PythonObject) raises -> PythonObject:
    """A new public float32 array holding one exported block."""
    var view = _export(plan, call, block)
    var array = km_vector(cols, "<f4", context) if vector else km_matrix(rows, cols, context)
    if rows * cols > 0:
        unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=array.address),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data), count=rows * cols)
    return array.value


def kernel_approx_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names and `_kernel_params` /
    `_params` its transform reads)."""
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var q = Int(s[5])
    var d = Int(s[6])
    var u = KmUnpacked(KmRoute(PythonObject(None), PythonObject(None), 0, 0), fields)
    u.cols = d
    u.count = q
    u.kernel = KmKernel(Int(s[1]), Int(s[2]), s[3], s[4])
    u.seed = (Int(s[11]) << 32) | (Int(s[10]) & 0xFFFFFFFF)
    var sweeps = Int(s[7])
    var sigma = s[8]
    var scale = s[9]
    if Int(s[0]) == 0:
        # The estimator's own rule (`km_nys_fit_unpack`; sklearn Nystroem.fit):
        # admit holds no X, so RUN clamps n_components to the rows; its
        # UserWarning is the state's fit notice, raised by the caller
        # (core/member_notices.mojo), never here.
        var components = _float_array(plan, call, _B_FIRST, q, d, False, context)
        var indices_view = _export(plan, call, _B_FIRST + 1)
        var indices = km_vector(q, "<i4", context)
        if q > 0:
            unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=indices.address),
                src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=indices_view.data), count=q)
        var normalization = _float_array(plan, call, _B_FIRST + 2, q, q, False, context)
        var eigenvalues = _float_array(plan, call, _B_FIRST + 3, 1, q, True, context)
        var eigenvectors = _float_array(plan, call, _B_FIRST + 4, q, q, False, context)
        km_nys_fit_pack(u, components, indices.value, normalization, eigenvalues, eigenvectors, sweeps)
        _ = indices^
    else:
        var weights = _float_array(plan, call, _B_FIRST, d, q, False, context)
        var offset = _float_array(plan, call, _B_FIRST + 1, 1, q, True, context)
        km_rbf_fit_pack(u, weights, offset, sigma, scale)
    kernel_public_fit_complete(model, PythonObject(None))


def kernel_approx_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var nystroem = plan.kind == 0
    var d = km_int(family_field(fields, "n_features_in_"), "n_features_in_")
    var scalars = List[Float64](length=12, fill=0.0)
    scalars[0] = Float64(plan.kind)
    var pins = List[ArrayPin]()
    var views = List[MemberMatrix]()
    var q = 0
    var seed = 0
    if nystroem:
        var kp = _km_params(fields, False, context)
        var components_pin = ArrayPin()
        var components = admit_family_meta(family_field(fields, "components_"), 2, "components_", context,
            components_pin)
        q = components.source.rows
        var indices_pin = ArrayPin()
        var indices = _km_i32_snapshot(family_field(fields, "component_indices_"), q, context, indices_pin)
        var normalization_pin = ArrayPin()
        var normalization = admit_family_meta(family_field(fields, "normalization_"), 2, "normalization_", context,
            normalization_pin)
        var eigenvalues_pin = ArrayPin()
        var eigenvalues = admit_family_meta(family_field(fields, "eigenvalues_"), 1, "eigenvalues_", context,
            eigenvalues_pin)
        var eigenvectors_pin = ArrayPin()
        var eigenvectors = admit_family_meta(family_field(fields, "eigenvectors_"), 2, "eigenvectors_", context,
            eigenvectors_pin)
        scalars[1] = Float64(kp.kernel.kernel)
        scalars[2] = Float64(kp.kernel.degree)
        scalars[3] = kp.kernel.gamma
        scalars[4] = kp.kernel.coef0
        scalars[7] = Float64(km_int(family_field(fields, "sweeps_"), "sweeps_"))
        seed = kp.seed
        views.append(MemberMatrix(components.source.addr, q, d))
        views.append(MemberMatrix(indices.source.addr, 1, q, MEMBER_I32))
        views.append(MemberMatrix(normalization.source.addr, q, q))
        views.append(MemberMatrix(eigenvalues.source.addr, 1, q))
        views.append(MemberMatrix(eigenvectors.source.addr, q, q))
        pins.append(components_pin^)
        pins.append(indices_pin^)
        pins.append(normalization_pin^)
        pins.append(eigenvalues_pin^)
        pins.append(eigenvectors_pin^)
    else:
        var params_in = km_tuple(family_field(fields, "_params"), 2, "_params")
        var weights_pin = ArrayPin()
        var weights = admit_family_meta(family_field(fields, "random_weights_"), 2, "random_weights_", context,
            weights_pin)
        q = weights.source.cols
        var offset_pin = ArrayPin()
        var offset = admit_family_meta(family_field(fields, "random_offset_"), 1, "random_offset_", context,
            offset_pin)
        if weights.source.rows != d or offset.count != q:
            raise Error("ValueError: RBFSampler fitted arrays do not match the fitted dimensions")
        scalars[1] = 2.0
        scalars[3] = km_number(params_in[0], "gamma", context)
        scalars[8] = km_number(family_field(fields, "sigma_"), "sigma_", context)
        scalars[9] = km_number(family_field(fields, "scale_"), "scale_", context)
        seed = km_int(params_in[1], "random_state")
        views.append(MemberMatrix(weights.source.addr, d, q))
        views.append(MemberMatrix(offset.source.addr, 1, q))
        pins.append(weights_pin^)
        pins.append(offset_pin^)
    scalars[5] = Float64(q)
    scalars[6] = Float64(d)
    scalars[10] = Float64(seed & 0xFFFFFFFF)
    scalars[11] = Float64(seed >> 32)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 12, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        for i in range(len(views)):  # driver-loop: over the kind's fitted blocks (two or five)
            call.block = _B_FIRST + i
            call.x = views[i]
            if i == len(views) - 1:
                call.flags = MEMBER_IMPORT_DONE
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = pins^
    _ = len(scalars)
    return call.state
