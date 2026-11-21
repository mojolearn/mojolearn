# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XDecomp family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; RUN:
core/xdecomp_member.mojo). Published: IncrementalPCA (kind 0,
`_mojolearn_member = (32, 0)`). The family's other classes
(GaussianRandomProjection SparseRandomProjection NMF FastICA FactorAnalysis
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
from bindings.py_balanced import py_get, py_setitem
from bindings.decomp_device_route_door import decomp_route_hit
from bindings.decomp_truth_boundary_unpack import decomp_truth
from bindings.ipca_descriptor_boundary_door import _ipca_route as ipca_route, ipca_provider
from bindings.ipca_native_door import _int as ipca_int, _f32 as ipca_f32
from bindings.ipca_owner_boundary_door import ipca_owner_drop
from x_decomp.ipca_descriptor_boundary_run import ipca_driver, ipca_mode, ipca_vendor, ipca_resident
from core.device_route import DeviceRoute
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.ipca_native_api import IPCA_NATIVE_COUNT
from core.xdecomp_member import (
    XDECOMP_KIND_IPCA, XDECOMP_ROUTE_RESIDENT, XDECOMP_BLOCK_COMPONENTS, XDECOMP_BLOCK_SINGULAR,
    XDECOMP_BLOCK_MEAN, XDECOMP_BLOCK_VAR, XDECOMP_BLOCK_VARIANCE, XDECOMP_BLOCK_RATIO,
)
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


def xdecomp_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind == XDECOMP_KIND_IPCA:
        _ipca_admit(model, fields, context, owner)
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
