# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XDecomp family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; RUN:
core/xdecomp_member.mojo). Published: IncrementalPCA (kind 0,
`_mojolearn_member = (32, 0)`), NMF (kind 3, `(32, 3)`),
GaussianRandomProjection (kind 5, `(32, 5)`), SparseRandomProjection (kind 6,
`(32, 6)`). The family's other classes (FastICA FactorAnalysis
DictionaryLearning MiniBatchDictionaryLearning SparsePCA MiniBatchSparsePCA
SparseCoder LatentDirichletAllocation Isomap MDS ClassicalMDS
LocallyLinearEmbedding MinCovDet EllipticEnvelope) carry no mark yet: a
caller keeps their own doors, and admit refuses their kinds in words.

  * `xdecomp_member_admit` (UNPACK, once per member): the estimator's own
    route through its resolver (the IncrementalPCA door's `_ipca_route`: a
    hit is one record read), turned into the member's `DeviceRoute`, and the
    parameter words of the member file, read with the door's own checks.
  * `xdecomp_member_pack` (PACK): the fitted attributes of the estimator's
    own fit, from the member state's blocks.
  * `xdecomp_member_state` (UNPACK): a member state from a fitted estimator.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and seven block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_f32
from bindings.classical_arrays_door import classical_address
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_get, py_setitem, py_call, py_contains
from bindings.decomp_device_route_door import decomp_route_hit
from bindings.decomp_truth_boundary_unpack import decomp_truth
from bindings.ipca_descriptor_boundary_door import _ipca_route as ipca_route, ipca_provider
from bindings.ipca_native_door import _int as ipca_int, _f32 as ipca_f32
from bindings.ipca_owner_boundary_door import ipca_owner_drop
from x_decomp.ipca_descriptor_boundary_run import ipca_driver, ipca_mode, ipca_vendor, ipca_resident
from bindings.decomp_scalar_admission_unpack import decomp_scalar_admit, decomp_scalar_repr
from bindings.decomp_scalar_compare_unpack import decomp_name_admitted as decomp_name
from bindings.decomp_owner_common_unpack import decomp_set_mode
from bindings.classical_arrays_door import classical_typed_native
from bindings.nmf_descriptor_boundary_door import _nmf_route as nmf_route, NMF_PROVIDER_PINS
from bindings.nmf_native_door import (
    _int as nmf_int, _float as nmf_float, _truth as nmf_truth, _seed as nmf_seed, _validate as nmf_validate,
)
from x_decomp.nmf_descriptor_boundary_run import nmf_driver, nmf_mode, nmf_vendor, nmf_capabilities
from x_decomp.nmf_owner import NMF_INIT_RANDOM, NMF_INIT_NNDSVD, NMF_INIT_NNDSVDA, NMF_INIT_NNDSVDAR
from core.nmf_native_api import NMF_NATIVE_COUNT, NMF_CAP_FIT, NMF_CAP_INVERSE
from core.device_route import DeviceRoute
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.ipca_native_api import IPCA_NATIVE_COUNT
from core.xdecomp_member import (
    XDECOMP_KIND_IPCA, XDECOMP_KIND_NMF, XDECOMP_ROUTE_RESIDENT, XDECOMP_ROUTE_NMF_INVERSE, XDECOMP_BLOCK_COMPONENTS, XDECOMP_BLOCK_SINGULAR,
    XDECOMP_BLOCK_MEAN, XDECOMP_BLOCK_VAR, XDECOMP_BLOCK_VARIANCE, XDECOMP_BLOCK_RATIO,
    XDECOMP_KIND_GRP, XDECOMP_KIND_SRP, XDECOMP_ROUTE_POOL, XDECOMP_BLOCK_RP_INVERSE,
)
from bindings.decomp_kit_native_door import (
    decomp_entries_routed, decomp_entries_ref, decomp_resident_binding, decomp_resident_drop,
)
from bindings.decomp_owner_common_unpack import (
    decomp_eq, decomp_int, decomp_float, decomp_seed, decomp_repr, decomp_f32,
)
from bindings.py_cpython_table import cpython_table
from providers.python.jl_scalar_pyapi_unpack import jl_scalar_word, jl_type_word, jl_type_words
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)


def _ipca_mode_object(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The door's `_set_mode`: numeric_mode, else the process default object.
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    return mode


def _ipca_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    var mode = _ipca_mode_object(fields, context)
    var connection = ipca_route(model, mode, fields, context)
    var hit = decomp_route_hit(model, mode, context)
    var words = List[Float64](length=4, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_IPCA)
    var requested = fields["n_components"]
    words[1] = -1.0 if requested is None else Float64(ipca_int(requested, context))
    words[2] = 1.0 if decomp_truth(fields["whiten"], context) else 0.0
    var batch = fields["batch_size"]
    words[3] = -1.0 if batch is None else Float64(ipca_int(batch, context))
    var route = DeviceRoute(hit.route.device if hit.found else 0)
    route.mode = ipca_mode(connection)
    # The connection's PCA-style vendor code (1 cpu, 2 metal, 3 cuda, 4 hip).
    route.vendor = ipca_vendor(connection) - 1
    route.entry = EntryRef(ipca_driver(connection).entry, entry_ref_word(route.mode, route.vendor, IPCA_NATIVE_COUNT))
    route.flags = XDECOMP_ROUTE_RESIDENT if ipca_resident(connection) else 0
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = ipca_provider(connection)


def _nmf_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    # The door's data-independent checks, in its order (bindings/nmf_native_door.mojo
    # `nmf_fit_transform` / `_run`); the data-dependent ones the member raises at fit.
    var mode = _ipca_mode_object(fields, context)
    var connection = nmf_route(model, fields, mode, context, NMF_CAP_FIT, "x_decomp_nmf_fit")
    var hit = decomp_route_hit(model, mode, context)
    var options = nmf_validate(fields, context)
    var words = List[Float64](length=12, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_NMF)
    var init_value = fields["init"]
    decomp_scalar_admit(init_value, context)
    if decomp_name(init_value, "custom"):
        raise Error("ValueError: mojolearn NMF: init='custom' takes W and H at fit; it does not run as a member")
    var requested = fields["n_components"]
    decomp_scalar_admit(requested, context)
    words[1] = -1.0 if (requested is None or decomp_name(requested, "auto")) else Float64(nmf_int(requested, context))
    var code = -1
    if nmf_truth(init_value, context):
        if decomp_name(init_value, "random"):
            code = NMF_INIT_RANDOM
        elif decomp_name(init_value, "nndsvd"):
            code = NMF_INIT_NNDSVD
        elif decomp_name(init_value, "nndsvda"):
            code = NMF_INIT_NNDSVDA
        elif decomp_name(init_value, "nndsvdar"):
            code = NMF_INIT_NNDSVDAR
        else:
            raise Error("init=" + decomp_scalar_repr(init_value, context)
                        + " is not supported here (random, nndsvd, nndsvda, nndsvdar)")
    words[2] = Float64(code)
    words[3] = Float64(options.solver)
    words[4] = options.beta
    # The door's `_regs` words (scaled by rows at each call in RUN).
    var alpha_h = fields["alpha_H"]
    var a_w = nmf_float(fields["alpha_W"], context)
    decomp_scalar_admit(alpha_h, context)
    var a_h = a_w if decomp_name(alpha_h, "same") else nmf_float(alpha_h, context)
    var l1 = nmf_float(fields["l1_ratio"], context)
    words[5] = Float64(nmf_int(fields["max_iter"], context))
    words[10] = 1.0 if nmf_truth(fields["shuffle"], context) else 0.0
    words[6] = nmf_float(fields["tol"], context)
    words[7] = a_w
    words[8] = a_h
    words[9] = l1
    words[11] = Float64(nmf_seed(fields["random_state"]))
    var route = DeviceRoute(hit.route.device if hit.found else 0)
    route.mode = nmf_mode(connection)
    route.vendor = nmf_vendor(connection) - 1
    route.entry = EntryRef(nmf_driver(connection).entry, entry_ref_word(route.mode, route.vendor, NMF_NATIVE_COUNT))
    route.flags = XDECOMP_ROUTE_NMF_INVERSE if (nmf_capabilities(connection) & NMF_CAP_INVERSE) != 0 else 0
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = NMF_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]


def _rp_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    mut owner: MemberOwner) raises:
    # bindings/rp_native_door.mojo `rp_fit`: the estimator's entries through
    # its resolver (`decomp_entries_routed`: a hit is one record read), the
    # data-independent checks in the door's words; the target dimension of
    # 'auto' and the density of 'auto' need X's shape: RUN computes them.
    var mode = _ipca_mode_object(fields, context)
    var entries = decomp_entries_routed(model, fields, mode, context)
    var hit = decomp_route_hit(model, mode, context)
    var words = List[Float64](length=6, fill=0.0)
    words[0] = Float64(kind)
    var requested = fields["n_components"]
    if decomp_eq(requested, PythonObject("auto")):
        words[1] = -1.0
        # johnson_lindenstrauss_min_dim's own admission of eps (its refusals).
        var eps = fields["eps"]
        words[2] = jl_scalar_word(eps._obj_ptr, jl_type_word(eps._obj_ptr), True, context._obj_ptr,
            cpython_table(), jl_type_words())
        _ = eps
    else:
        var count = decomp_int(requested)
        if count <= 0:
            raise Error("n_components must be greater than 0, got " + String(count))
        words[1] = Float64(count)
    if kind == XDECOMP_KIND_SRP:
        var given = fields["density"]
        if decomp_eq(given, PythonObject("auto")):
            words[3] = -1.0
        else:
            var density = decomp_float(given)
            if not (density > 0 and density <= 1):
                raise Error("Expected density in range ]0, 1], got: " + decomp_repr(PythonObject(density)))
            words[3] = density
    words[4] = 1.0 if Bool(py=fields["compute_inverse_components"]) else 0.0
    words[5] = Float64(decomp_seed(fields["random_state"]))
    var entry = decomp_entries_ref(entries)
    var key = entries["x_decomp_native_key"]
    var route = DeviceRoute(hit.route.device if hit.found else 0)
    route.mode = Int(py=key[0])
    route.vendor = Int(py=key[1])
    route.entry = entry
    route.flags = XDECOMP_ROUTE_POOL if decomp_resident_binding(entries) else 0
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = entries


def xdecomp_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind == XDECOMP_KIND_IPCA:
        _ipca_admit(model, fields, context, owner)
        return True
    if kind == XDECOMP_KIND_NMF:
        _nmf_admit(model, fields, context, owner)
        return True
    if kind == XDECOMP_KIND_GRP or kind == XDECOMP_KIND_SRP:
        _rp_admit(model, fields, kind, context, owner)
        return True
    raise Error("ValueError: mojolearn: XDecomp member kind " + String(kind) + " is not published")


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _array(view: MemberMatrix, matrix: Bool, context: PythonObject) raises -> PythonObject:
    # The door's `_f32`: the same native array type its transform and save read.
    var array = ipca_f32(view.rows if matrix else view.cols, view.cols if matrix else 0, context, not matrix)
    var count = view.rows * view.cols
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(array))), src=f32_ptr(view.data), count=count)
    return array


def xdecomp_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (bindings/ipca_native_door.mojo
    `ipca_fit` / `_finish_fit` / `_publish`: the same names and array types)."""
    if plan.kind == XDECOMP_KIND_NMF:
        _nmf_pack(model, fields, context, plan, state)
        return
    if plan.kind == XDECOMP_KIND_GRP or plan.kind == XDECOMP_KIND_SRP:
        _rp_pack(model, fields, context, plan, state)
        return
    if plan.kind != XDECOMP_KIND_IPCA:
        raise Error("ValueError: mojolearn: XDecomp member kind " + String(plan.kind) + " is not published")
    # New components replace the old: the model's resident owners go now.
    ipca_owner_drop(model)
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var count = Int(scalars[0])
    var features = Int(scalars[1])
    var seen = Int(scalars[2])
    var noise = scalars[3]
    var batch = Int(scalars[4])
    var components = _array(_export(plan, call, XDECOMP_BLOCK_COMPONENTS), True, context)
    var singular = _array(_export(plan, call, XDECOMP_BLOCK_SINGULAR), False, context)
    var mean = _array(_export(plan, call, XDECOMP_BLOCK_MEAN), False, context)
    var variance = _array(_export(plan, call, XDECOMP_BLOCK_VAR), False, context)
    var explained = _array(_export(plan, call, XDECOMP_BLOCK_VARIANCE), False, context)
    var ratio = _array(_export(plan, call, XDECOMP_BLOCK_RATIO), False, context)
    py_setitem(fields, "numeric_mode_", _ipca_mode_object(fields, context))
    py_setitem(fields, "batch_size_", PythonObject(batch))
    py_setitem(fields, "n_samples_seen_", PythonObject(seen))
    py_setitem(fields, "n_components_", PythonObject(count))
    py_setitem(fields, "n_features_in_", PythonObject(features))
    py_setitem(fields, "components_", components)
    py_setitem(fields, "singular_values_", singular)
    py_setitem(fields, "mean_", mean)
    py_setitem(fields, "var_", variance)
    py_setitem(fields, "explained_variance_", explained)
    py_setitem(fields, "explained_variance_ratio_", ratio)
    py_setitem(fields, "noise_variance_", PythonObject(noise))


def _nmf_pack(model: PythonObject, fields: PythonObject, context: PythonObject, plan: MemberPlan,
    state: Int) raises:
    # bindings/nmf_native_door.mojo `nmf_fit_transform`: numeric_mode_, then
    # components_, n_components_, n_iter_, n_features_in_, reconstruction_err_.
    decomp_set_mode(fields, context)
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var count = Int(scalars[0])
    var features = Int(scalars[1])
    var iterations = Int(scalars[2])
    var residual = scalars[3]
    var view = _export(plan, call, XDECOMP_BLOCK_COMPONENTS)
    var shape = List[Int](length=1, fill=count)
    shape.append(features)
    var h = classical_typed_native(shape, "<f4", context["output_context"])
    if count * features > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(h))), src=f32_ptr(view.data), count=count * features)
    py_setitem(fields, "components_", h)
    py_setitem(fields, "n_components_", PythonObject(count))
    py_setitem(fields, "n_iter_", PythonObject(iterations))
    py_setitem(fields, "n_features_in_", PythonObject(features))
    py_setitem(fields, "reconstruction_err_", PythonObject(residual))


def _import(plan: MemberPlan, mut call: MemberCall, fields: PythonObject, name: String, ndim: Int, block: Int,
    context: PythonObject) raises:
    var source = py_get(fields, name)
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var words = admit_f32(source, ndim, name, context)
    call.block = block
    call.x = MemberMatrix(words.address, 1, words.rows * (words.cols if ndim == 2 else 1))
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = words


def xdecomp_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    if plan.kind == XDECOMP_KIND_NMF:
        return _nmf_state(fields, context, plan)
    if plan.kind == XDECOMP_KIND_GRP or plan.kind == XDECOMP_KIND_SRP:
        return _rp_state(fields, context, plan)
    if plan.kind != XDECOMP_KIND_IPCA:
        raise Error("ValueError: mojolearn: XDecomp member kind " + String(plan.kind) + " is not published")
    var count = py_get(fields, "n_components_")
    var features = py_get(fields, "n_features_in_")
    var seen = py_get(fields, "n_samples_seen_")
    var noise = py_get(fields, "noise_variance_")
    var batch = py_get(fields, "batch_size_")
    if count is None or features is None or py_get(fields, "components_") is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var scalars: List[Float64] = [Float64(ipca_int(count, context)), Float64(ipca_int(features, context)),
        Float64(ipca_int(seen, context)) if seen is not None else 0.0,
        Float64(py=noise) if noise is not None else 0.0,
        Float64(ipca_int(batch, context)) if batch is not None else 0.0]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "components_", 2, XDECOMP_BLOCK_COMPONENTS, context)
        _import(plan, call, fields, "singular_values_", 1, XDECOMP_BLOCK_SINGULAR, context)
        _import(plan, call, fields, "mean_", 1, XDECOMP_BLOCK_MEAN, context)
        _import(plan, call, fields, "var_", 1, XDECOMP_BLOCK_VAR, context)
        _import(plan, call, fields, "explained_variance_", 1, XDECOMP_BLOCK_VARIANCE, context)
        _import(plan, call, fields, "explained_variance_ratio_", 1, XDECOMP_BLOCK_RATIO, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


def _nmf_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    var count = py_get(fields, "n_components_")
    var features = py_get(fields, "n_features_in_")
    var iterations = py_get(fields, "n_iter_")
    var residual = py_get(fields, "reconstruction_err_")
    if count is None or features is None or py_get(fields, "components_") is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var scalars: List[Float64] = [Float64(nmf_int(count, context)), Float64(nmf_int(features, context)),
        Float64(nmf_int(iterations, context)) if iterations is not None else 0.0,
        Float64(py=residual) if residual is not None else 0.0]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "components_", 2, XDECOMP_BLOCK_COMPONENTS, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


def _rp_matrix(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    # The door's `decomp_f32` array (the type its transform and save read).
    var array = decomp_f32(Python.tuple(view.rows, view.cols), context)
    var count = view.rows * view.cols
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(array))), src=f32_ptr(view.data), count=count)
    return array


def _rp_pack(model: PythonObject, fields: PythonObject, context: PythonObject, plan: MemberPlan,
    state: Int) raises:
    # bindings/rp_native_door.mojo `rp_fit`, in its order: the resident
    # copies of the old matrix dropped, numeric_mode_, density_ (Sparse),
    # n_components_, n_features_in_, the `components_` setter's fields,
    # inverse_components_ (with compute_inverse_components).
    decomp_resident_drop(model)
    decomp_set_mode(fields, context)
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var count = Int(scalars[0])
    var features = Int(scalars[1])
    var density = scalars[2]
    var inverse = scalars[3] != 0.0
    var components = _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS), context)
    if plan.kind == XDECOMP_KIND_SRP:
        py_setitem(fields, "density_", PythonObject(density))
    py_setitem(fields, "n_components_", PythonObject(count))
    py_setitem(fields, "n_features_in_", PythonObject(features))
    if py_contains(fields, "_rp_lazy"):
        _ = py_call(fields.pop, "_rp_lazy")
    py_setitem(fields, "_rp_components", components)
    if inverse:
        py_setitem(fields, "inverse_components_", _rp_matrix(_export(plan, call, XDECOMP_BLOCK_RP_INVERSE), context))


def _rp_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # The fitted fields the door's transform reads: `_rp_components` (what
    # the `components_` property holds), n_features_in_, inverse_components_.
    var count = py_get(fields, "n_components_")
    var features = py_get(fields, "n_features_in_")
    if count is None or features is None or py_get(fields, "_rp_components") is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var density = py_get(fields, "density_")
    var inverse = py_get(fields, "inverse_components_") is not None
    var scalars: List[Float64] = [Float64(decomp_int(count)), Float64(decomp_int(features)),
        Float64(py=density) if density is not None else 0.0, 1.0 if inverse else 0.0]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "_rp_components", 2, XDECOMP_BLOCK_COMPONENTS, context)
        if inverse:
            _import(plan, call, fields, "inverse_components_", 2, XDECOMP_BLOCK_RP_INVERSE, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state
