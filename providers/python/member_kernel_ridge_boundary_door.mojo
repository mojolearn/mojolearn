# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the KernelRidge family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; worked form
providers/python/member_linear_boundary_door.mojo). Kind 0 is KernelRidge
(`_mojolearn_member = (MEMBER_FAMILY_KERNEL_RIDGE, 0)` on the class).

  * `kernel_ridge_member_admit` (UNPACK, once per member): the estimator's
    own parameter checks (bindings/kernel_methods_boundary_unpack.mojo: the
    kernel name, degree, gamma, coef0, alpha), its binding and its mode
    check (`km_extension`: the binary's compile-time mode must be the one
    requested), the provider table in the route's entry slot, its
    capability word in `route.flags`, and the 6 parameter words of
    core/kernel_ridge_member.mojo (kernel='precomputed' among them since
    lane mf-adm3). A `fast` binding's door has no provider (it keeps its
    list and pointer entries); its member takes the image's own table
    (`kernel_method_member_provider`, lane mf-img3: the same typed bodies),
    so every configuration the door fits is admitted.
  * `kernel_ridge_member_pack` (PACK): the fitted attributes the estimator's
    own fit writes (`km_krr_fit_pack`, `kernel_public_fit_complete`), from
    the member state's blocks. A one-column y is packed 1-D (`_squeeze`).
  * `kernel_ridge_member_state` (UNPACK): a member state from a fitted
    KernelRidge.

SERIAL BY NATURE: one member's admission or publication (record reads and
one copy per fitted block); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family_meta
from bindings.family_common_pack import family_field
from bindings.kernel_methods_boundary_pack import km_krr_public, km_matrix, km_krr_fit_pack
from bindings.kernel_methods_boundary_unpack import (
    KmRoute, KmUnpacked, km_extension, km_int, _km_kernel_code, _km_degree, _km_real, _km_params,
)
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from kernel_methods.descriptor_schema import KERNEL_METHOD_OPERATIONS
from kernel_methods.native_call_run import KmKernel
from providers.python.array_metadata_pyapi_door import ArrayPin
from providers.python.host_estimator_binding_pyapi_door import estimator_member_binding
from providers.python.kernel_method_descriptor_pyapi_door import kernel_method_metadata, kernel_method_member_provider
from providers.python.kernel_public_lifecycle_pyapi_unpack import kernel_public_fit_complete
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_provider_id,
)
from x_decomp.descriptor_schema import xd_vendor_code

comptime _WHERE = "KernelRidge"
# core/kernel_ridge_member.mojo's block names (docstring order).
comptime _B_X_FIT = MEMBER_BLOCK_FAMILY
comptime _B_DUAL = MEMBER_BLOCK_FAMILY + 1


def kernel_ridge_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. Always True (every configuration the
    estimator's door fits has the member entry)."""
    if kind != 0:
        raise Error("ValueError: mojolearn: unknown KernelRidge member kind " + String(kind))
    # The estimator's own checks, in its fit's order (`km_krr_fit_unpack`).
    var kernel = _km_kernel_code(family_field(fields, "kernel"), _WHERE, True)
    # An explicit Host* class (its own host-only word, lane mf-iterimp3b): the host image its own methods
    # bind (`estimator_member_binding`); any other class: the estimator's binding route, as before.
    var binding = estimator_member_binding(model, "_mojolearn_kernel_methods", context)
    var route = km_extension(fields, binding, _WHERE, context)
    if not route.described():
        # A `fast` binding (lane mf-img3): its door keeps its list and pointer entries, the member runs the
        # same bodies through the image's own provider table (`kernel_method_member_provider`).
        var table = kernel_method_member_provider(fields, binding)
        route = KmRoute(binding, table, kernel_method_metadata(table, 14), neural_descriptor_provider_id(table))
    var degree = _km_degree(fields, _WHERE)
    var gamma_source = family_field(fields, "gamma")
    var gamma_kind = 1.0 if gamma_source is None else 0.0
    var gamma = 0.0 if gamma_source is None else _km_real(gamma_source, "gamma", _WHERE, context)
    var coef0 = _km_real(family_field(fields, "coef0"), "coef0", _WHERE, context)
    var alpha = _km_real(family_field(fields, "alpha"), "alpha", _WHERE, context)
    var mode = kernel_method_metadata(route.native, 13)
    # The binary's own compiled vendor word ('cpu' for the host image, else
    # 'metal' / 'cuda' / 'hip').
    var vendor = xd_vendor_code(String(py_call(neural_descriptor_entry(binding, "kernel_methods_vendor"))))
    var entry = route.native.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    var device_route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    device_route.mode = mode
    device_route.vendor = vendor
    device_route.entry = EntryRef(entry, entry_ref_word(mode, vendor, KERNEL_METHOD_OPERATIONS))
    device_route.flags = route.flags
    var words = List[Float64](length=6, fill=0.0)
    words[0] = Float64(kernel)
    words[1] = Float64(degree)
    words[2] = gamma_kind
    words[3] = gamma
    words[4] = coef0
    words[5] = alpha
    owner.plan.route = device_route
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, route.native)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def kernel_ridge_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names and the same resident
    ownership record its predict reads)."""
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var n = Int(s[5])
    var d = Int(s[6])
    var t = Int(s[7])
    var info = Int(s[8])
    var u = KmUnpacked(KmRoute(PythonObject(None), PythonObject(None), 0, 0), fields)
    u.cols = d
    u.targets = t
    u.squeeze = t == 1
    u.kernel = KmKernel(Int(s[0]), Int(s[1]), s[2], s[3])
    u.alpha = s[4]
    var x_view = _export(plan, call, _B_X_FIT)
    var kept = km_matrix(n, d, context)
    if n * d > 0:
        unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=kept.address),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=x_view.data), count=n * d)
    var dual_view = _export(plan, call, _B_DUAL)
    var dual = km_krr_public(n, t, u.squeeze, context)
    if n * t > 0:
        unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dual.address),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dual_view.data), count=n * t)
    km_krr_fit_pack(model, u, kept.value, dual.value, info)
    kernel_public_fit_complete(model, PythonObject(None))
    _ = kept^
    _ = dual^


def kernel_ridge_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var x_source = family_field(fields, "X_fit_")
    var dual_source = family_field(fields, "dual_coef_")
    var t = km_int(family_field(fields, "n_targets_"), "n_targets_")
    var info = km_int(family_field(fields, "info_"), "info_")
    var kp = _km_params(fields, True, context)
    var x_pin = ArrayPin()
    var dual_pin = ArrayPin()
    var x_meta = admit_family_meta(x_source, 2, "X_fit_", context, x_pin)
    var dual_meta = admit_family_meta(dual_source, -1, "dual_coef_", context, dual_pin)
    var n = x_meta.source.rows
    var d = x_meta.source.cols
    if t < 1 or dual_meta.count != n * t:
        raise Error("ValueError: KernelRidge fitted arrays do not match the fitted dimensions")
    var scalars = List[Float64](length=9, fill=0.0)
    scalars[0] = Float64(kp.kernel.kernel)
    scalars[1] = Float64(kp.kernel.degree)
    scalars[2] = kp.kernel.gamma
    scalars[3] = kp.kernel.coef0
    scalars[4] = kp.alpha
    scalars[5] = Float64(n)
    scalars[6] = Float64(d)
    scalars[7] = Float64(t)
    scalars[8] = Float64(info)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 9, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_X_FIT
        call.x = MemberMatrix(x_meta.source.addr, n, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_DUAL
        call.x = MemberMatrix(dual_meta.source.addr, n, t)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = x_pin^
    _ = dual_pin^
    _ = len(scalars)
    return call.state
