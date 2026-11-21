# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XDecomp family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; RUN:
core/xdecomp_member.mojo). Published: IncrementalPCA (kind 0,
`_mojolearn_member = (32, 0)`), NMF (kind 3, `(32, 3)`),
GaussianRandomProjection (kind 5, `(32, 5)`), SparseRandomProjection (kind 6,
`(32, 6)`), FastICA (kind 7, `(32, 7)`), FactorAnalysis (kind 8, `(32, 8)`),
DictionaryLearning (9), MiniBatchDictionaryLearning (10), SparseCoder (11),
SparsePCA (12), MiniBatchSparsePCA (13), EllipticEnvelope (14). Falls back at admit (False): a
callback, code_init or dict_init (the dictionary learners), V_init or U_init
(SparsePCA), a w_init (FastICA). The family's other classes carry no mark
yet: a caller keeps their own doors, and admit refuses their kinds in words.

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
    XDECOMP_KIND_GRP, XDECOMP_KIND_SRP, XDECOMP_ROUTE_POOL, XDECOMP_BLOCK_RP_INVERSE, XDECOMP_KIND_ICA,
    XDECOMP_KIND_FA, XDECOMP_KIND_DL, XDECOMP_KIND_MBDL, XDECOMP_KIND_CODER, XDECOMP_KIND_SPCA, XDECOMP_KIND_MBSPCA,
    XDECOMP_DL_PARAM_COUNT, XDECOMP_MBDL_PARAM_COUNT, XDECOMP_CODER_PARAM_COUNT, XDECOMP_SPCA_PARAM_COUNT,
    XDECOMP_MBSPCA_PARAM_COUNT, XDECOMP_ROUTE_KIT_RESIDENT, XDECOMP_ROUTE_KIT_SHIFT, XDECOMP_KIND_ENVELOPE,
    XDECOMP_ENVELOPE_PARAM_COUNT,
)
from bindings.mcd_native_door import _typed as mcd_typed
from core.decomp_native_impl import decomp_f32_to_u8
from bindings.dict_learning_native_unpack import (
    _check_algos as dl_check_algos, _algo_index as dl_algo_index, _fit_index as dl_fit_index,
    _dict_words as dl_dict_words, _coder_dictionary as dl_coder_dictionary, _is_str as dl_is_str,
)
from bindings.py_lists import py_list_f64
from bindings.ica_native_unpack import _whiten_code as ica_whiten_code
from bindings.native_loaded_module_door import native_loaded_module
from bindings.decomp_kit_native_door import (
    decomp_entries_routed, decomp_entries_ref, decomp_resident_binding, decomp_resident_drop,
    decomp_kit_unpack, decomp_seed as kit_seed, decomp_seed_word, XD_SPARSE_ALGOS,
)
from bindings.decomp_owner_common_unpack import (
    decomp_eq, decomp_int, decomp_float, decomp_seed, decomp_repr, decomp_f32, decomp_le,
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
    _ = _table_route(model, fields, context, owner)
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
    owner.plan.params = words^


def _table_route(model: PythonObject, fields: PythonObject, context: PythonObject,
    mut owner: MemberOwner) raises -> PythonObject:
    """The x_decomp operations table of the estimator's image, through its
    resolver (`decomp_entries_routed`: a hit is one record read), as the
    random projection, FastICA and FactorAnalysis doors resolve it:
    `owner.plan.route` (mode and vendor from the entries' native key, bit
    XDECOMP_ROUTE_POOL on a GPU binding) and `owner.pins` = the entries."""
    var mode = _ipca_mode_object(fields, context)
    var entries = decomp_entries_routed(model, fields, mode, context)
    var hit = decomp_route_hit(model, mode, context)
    var entry = decomp_entries_ref(entries)
    var key = entries["x_decomp_native_key"]
    var route = DeviceRoute(hit.route.device if hit.found else 0)
    route.mode = Int(py=key[0])
    route.vendor = Int(py=key[1])
    route.entry = entry
    route.flags = XDECOMP_ROUTE_POOL if decomp_resident_binding(entries) else 0
    owner.plan.route = route
    owner.pins = entries
    return entries


def _ica_admit(model: PythonObject, fields: PythonObject, context: PythonObject,
    mut owner: MemberOwner) raises -> Bool:
    # bindings/ica_native_unpack.mojo `ica_fit`: its data-independent checks
    # in its words; nc = min(n_components, rows, features) in RUN. A given
    # w_init is an array of the fit's (nc, nc): not a member (False).
    if py_get(fields, "w_init") is not None:
        return False
    _ = _table_route(model, fields, context, owner)
    var operator = native_loaded_module("operator")
    var algorithm = fields["algorithm"]
    if not Bool(py=py_call(operator.contains, Python.tuple("parallel", "deflation"), algorithm)):
        raise Error("algorithm must be 'parallel' or 'deflation'")
    var code = ica_whiten_code(fields["whiten"])
    var words = List[Float64](length=9, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_ICA)
    var requested = fields["n_components"]
    words[1] = Float64(decomp_int(requested)) if (code != 0 and requested is not None) else -1.0
    words[2] = Float64(code)
    words[3] = 0.0 if decomp_eq(algorithm, PythonObject("parallel")) else 1.0
    var alpha = PythonObject(1.0)
    var fun = fields["fun"]
    if decomp_eq(fun, PythonObject("logcosh")):
        var fun_args = fields["fun_args"]
        var options = fun_args if Bool(py=fun_args) else Python.dict()
        alpha = py_call(options.get, "alpha", 1.0)
        if not (decomp_le(PythonObject(1), alpha) and decomp_le(alpha, PythonObject(2))):
            raise Error("alpha must be in [1,2]")
        words[4] = 0.0
    elif decomp_eq(fun, PythonObject("exp")):
        words[4] = 1.0
    elif decomp_eq(fun, PythonObject("cube")):
        words[4] = 2.0
    else:
        raise Error("fun must be 'logcosh', 'exp' or 'cube' (a callable is not carried)")
    words[5] = decomp_float(alpha)
    words[6] = Float64(decomp_int(fields["max_iter"]))
    words[7] = decomp_float(fields["tol"])
    words[8] = Float64(decomp_seed(fields["random_state"]))
    owner.plan.params = words^
    return True


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
    if kind == XDECOMP_KIND_ICA:
        return _ica_admit(model, fields, context, owner)
    if kind == XDECOMP_KIND_FA:
        _fa_admit(model, fields, context, owner)
        return True
    if kind == XDECOMP_KIND_DL or kind == XDECOMP_KIND_MBDL:
        return _dl_admit(model, fields, kind, context, owner)
    if kind == XDECOMP_KIND_CODER:
        _coder_admit(model, fields, context, owner)
        return True
    if kind == XDECOMP_KIND_SPCA or kind == XDECOMP_KIND_MBSPCA:
        return _spca_admit(model, fields, kind, context, owner)
    if kind == XDECOMP_KIND_ENVELOPE:
        _mcd_admit(model, fields, context, owner)
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
    if plan.kind == XDECOMP_KIND_ICA:
        _ica_pack(fields, context, plan, state)
        return
    if plan.kind == XDECOMP_KIND_FA:
        _fa_pack(fields, context, plan, state)
        return
    if plan.kind >= XDECOMP_KIND_DL and plan.kind <= XDECOMP_KIND_MBSPCA:
        _kit_pack(fields, context, plan, state)
        return
    if plan.kind == XDECOMP_KIND_ENVELOPE:
        _mcd_pack(fields, context, plan, state)
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
    if plan.kind == XDECOMP_KIND_ICA:
        return _ica_state(fields, context, plan)
    if plan.kind == XDECOMP_KIND_FA:
        return _fa_state(fields, context, plan)
    if plan.kind >= XDECOMP_KIND_DL and plan.kind <= XDECOMP_KIND_MBSPCA:
        return _kit_state(fields, context, plan)
    if plan.kind == XDECOMP_KIND_ENVELOPE:
        return _mcd_state(fields, context, plan)
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


def _ica_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    # bindings/ica_native_unpack.mojo `ica_fit`, in its order: numeric_mode_,
    # n_iter_, whitening_ and mean_ (whitening), components_, mixing_,
    # `_whiten` (the parameter as given), n_features_in_.
    decomp_set_mode(fields, context)
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var features = Int(scalars[1])
    var iterations = Int(scalars[2])
    var whiten = scalars[3] != 0.0
    py_setitem(fields, "n_iter_", PythonObject(iterations))
    if whiten:
        py_setitem(fields, "whitening_", _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 2), context))
        var mean_view = _export(plan, call, XDECOMP_BLOCK_COMPONENTS + 3)
        var mean = decomp_f32(Python.tuple(features), context)
        if features > 0:
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(mean))), src=f32_ptr(mean_view.data), count=features)
        py_setitem(fields, "mean_", mean)
    py_setitem(fields, "components_", _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS), context))
    py_setitem(fields, "mixing_", _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 1), context))
    py_setitem(fields, "_whiten", fields["whiten"])
    py_setitem(fields, "n_features_in_", PythonObject(features))


def _ica_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # What the door's transform reads: components_, mixing_, `_whiten`,
    # mean_ (whitening); whitening_ and n_iter_ when present.
    var features = py_get(fields, "n_features_in_")
    var components = py_get(fields, "components_")
    var whiten_value = py_get(fields, "_whiten")
    if features is None or components is None or py_get(fields, "mixing_") is None or whiten_value is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var whiten = Bool(py=whiten_value)
    var iterations = py_get(fields, "n_iter_")
    var scalars: List[Float64] = [Float64(Int(py=components.shape[0])), Float64(decomp_int(features)),
        Float64(decomp_int(iterations)) if iterations is not None else 0.0, 1.0 if whiten else 0.0,
        Float64(Int(py=components.shape[1]))]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "components_", 2, XDECOMP_BLOCK_COMPONENTS, context)
        _import(plan, call, fields, "mixing_", 2, XDECOMP_BLOCK_COMPONENTS + 1, context)
        if whiten:
            if py_get(fields, "whitening_") is not None:
                _import(plan, call, fields, "whitening_", 2, XDECOMP_BLOCK_COMPONENTS + 2, context)
            _import(plan, call, fields, "mean_", 1, XDECOMP_BLOCK_COMPONENTS + 3, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


def _fa_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    # bindings/fa_native_unpack.mojo `fa_fit`: its two option checks first,
    # in its words; noise_variance_init read as `_psi_init` reads it (its
    # length checked against n_features in RUN, with the door's words).
    var operator = native_loaded_module("operator")
    if not Bool(py=py_call(operator.contains, Python.tuple("lapack", "randomized"), fields["svd_method"])):
        raise Error("svd_method must be 'lapack' or 'randomized'")
    var rotation = fields["rotation"]
    if not Bool(py=py_call(operator.contains, Python.tuple(None, "varimax", "quartimax"), rotation)):
        raise Error("rotation must be None, 'varimax' or 'quartimax'")
    _ = _table_route(model, fields, context, owner)
    var words = List[Float64](length=6, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_FA)
    var requested = fields["n_components"]
    words[1] = Float64(decomp_int(requested)) if Bool(py=requested) else -1.0
    words[2] = Float64(decomp_int(fields["max_iter"]))
    var code = 0
    if rotation is not None:
        code = 1 if decomp_eq(rotation, PythonObject("varimax")) else 2
    words[3] = Float64(code)
    words[4] = decomp_float(fields["tol"])
    words[5] = -1.0
    var given = fields["noise_variance_init"]
    if given is not None:
        var values = py_call(native_loaded_module("builtins").list, given)
        var count = len(values)
        words[5] = Float64(count)
        for j in range(count):  # driver-loop: over the caller's noise_variance_init entries (model structure)
            words.append(decomp_float(values[j]))
    owner.plan.params = words^


def _fa_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    # bindings/fa_native_unpack.mojo `fa_fit`, in its order: numeric_mode_,
    # components_, noise_variance_, mean_, loglike_, n_iter_, n_features_in_.
    decomp_set_mode(fields, context)
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var words = scalars_view.rows * scalars_view.cols
    var features = Int(scalars[1])
    var iterations = Int(scalars[2])
    var loglike = Python.list()
    for k in range(3, words):  # driver-loop: over the EM iterations' log-likelihoods
        _ = py_call(loglike.append, PythonObject(scalars[k]))
    py_setitem(fields, "components_", _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS), context))
    for block in range(2):  # driver-loop: noise_variance_ and mean_
        var view = _export(plan, call, XDECOMP_BLOCK_COMPONENTS + 1 + block)
        var vector = decomp_f32(Python.tuple(features), context)
        if features > 0:
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(vector))), src=f32_ptr(view.data), count=features)
        py_setitem(fields, PythonObject("noise_variance_") if block == 0 else PythonObject("mean_"), vector)
    py_setitem(fields, "loglike_", loglike)
    py_setitem(fields, "n_iter_", PythonObject(iterations))
    py_setitem(fields, "n_features_in_", PythonObject(features))


def _fa_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # What the door's queries read: components_, noise_variance_, mean_.
    var components = py_get(fields, "components_")
    if (components is None or py_get(fields, "noise_variance_") is None or py_get(fields, "mean_") is None):
        raise Error("NotFittedError: this member is not fitted yet")
    var iterations = py_get(fields, "n_iter_")
    var scalars: List[Float64] = [Float64(Int(py=components.shape[0])), Float64(Int(py=components.shape[1])),
        Float64(decomp_int(iterations)) if iterations is not None else 0.0]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "components_", 2, XDECOMP_BLOCK_COMPONENTS, context)
        _import(plan, call, fields, "noise_variance_", 1, XDECOMP_BLOCK_COMPONENTS + 1, context)
        _import(plan, call, fields, "mean_", 1, XDECOMP_BLOCK_COMPONENTS + 2, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


# ---------------------------------------------------------------- the dictionary-learning kit


def _kit_route(model: PythonObject, fields: PythonObject, context: PythonObject,
    mut owner: MemberOwner) raises -> PythonObject:
    """`_table_route` plus what `decomp_kit_unpack(entries)` reads: the
    image's residency and its optional entries, as route flags."""
    var entries = _table_route(model, fields, context, owner)
    var kit = decomp_kit_unpack(entries)
    var flags = owner.plan.route.flags | (kit.present << XDECOMP_ROUTE_KIT_SHIFT)
    if kit.resident:
        flags |= XDECOMP_ROUTE_KIT_RESIDENT
    owner.plan.route.flags = flags
    return entries


def _flag(value: PythonObject) raises -> Float64:
    return 1.0 if Bool(py=value) else 0.0


def _coding_admit(fields: PythonObject, mut words: List[Float64]) raises:
    # bindings/dict_learning_native_unpack.mojo `_coding_words` /
    # `_encode_words`: each option read only where its route reads it; the
    # lars / omp default budget needs the width and the atoms (RUN).
    var index = dl_algo_index(fields["transform_algorithm"])
    if index < 0:
        raise Error("transform_algorithm must be one of " + String(XD_SPARSE_ALGOS))
    words[1] = Float64(index)
    words[2] = -1.0
    words[3] = 1.0
    if index == 2 or index == 3:
        var nnz = fields["transform_n_nonzero_coefs"]
        if nnz is not None:
            words[2] = Float64(decomp_int(nnz))
    else:
        var alpha = py_get(fields, "transform_alpha")
        if alpha is None:
            alpha = py_get(fields, "alpha", PythonObject(1.0))
        if alpha is not None:
            words[3] = decomp_float(alpha)
    var positive = fields["positive_code"]
    if index == 4:
        words[6] = _flag(positive)
    elif index == 0 or index == 1:
        words[4] = Float64(decomp_int(fields["transform_max_iter"]))
        words[5] = Float64(decomp_int(positive))
    words[7] = _flag(fields["split_sign"])


def _dl_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    mut owner: MemberOwner) raises -> Bool:
    # `dictionary_learning_fit` / `minibatch_dictionary_fit`: the algorithm
    # check first, in its words. A callback (refused by the door), a
    # code_init or a dict_init (arrays of the fit's shapes) keep the door.
    dl_check_algos(fields["fit_algorithm"], fields["transform_algorithm"])
    if py_get(fields, "callback") is not None or py_get(fields, "dict_init") is not None:
        return False
    if kind == XDECOMP_KIND_DL and py_get(fields, "code_init") is not None:
        return False
    _ = _kit_route(model, fields, context, owner)
    var count = XDECOMP_DL_PARAM_COUNT if kind == XDECOMP_KIND_DL else XDECOMP_MBDL_PARAM_COUNT
    var words = List[Float64](length=count, fill=0.0)
    words[0] = Float64(kind)
    var requested = fields["n_components"]
    words[8] = -1.0 if requested is None else Float64(decomp_int(requested))
    var method = dl_fit_index(fields["fit_algorithm"])
    if kind == XDECOMP_KIND_DL:
        var dw = dl_dict_words(-1, fields["alpha"], fields["max_iter"], fields["tol"], method, fields["random_state"],
                               Bool(py=fields["positive_dict"]), Bool(py=fields["positive_code"]),
                               fields["transform_max_iter"])
        words[9] = dw.alpha
        words[10] = Float64(dw.max_iter)
        words[11] = dw.tol
        words[12] = Float64(dw.method)
        words[13] = Float64(dw.seed)
        words[14] = 1.0 if dw.positive_dict else 0.0
        words[15] = 1.0 if dw.positive_code else 0.0
        words[16] = Float64(dw.method_max_iter)
    else:
        # `_minibatch_words`' reads in its order; min(batch_size, n) and the
        # zero-batch refusal need n (RUN).
        words[9] = Float64(decomp_int(fields["batch_size"]))
        words[10] = Float64(decomp_seed_word(kit_seed(fields["random_state"])))
        words[11] = _flag(fields["shuffle"])
        words[12] = decomp_float(fields["max_iter"])
        var patience = fields["max_no_improvement"]
        words[13] = -1.0 if patience is None else Float64(decomp_int(patience))
        words[14] = Float64(method)
        words[15] = Float64(decomp_int(fields["transform_max_iter"]))
        words[16] = _flag(fields["positive_dict"])
        words[17] = _flag(fields["positive_code"])
        words[18] = decomp_float(fields["alpha"])
        words[19] = decomp_float(fields["tol"])
    _coding_admit(fields, words)
    owner.plan.params = words^
    return True


def _coder_admit(model: PythonObject, fields: PythonObject, context: PythonObject,
    mut owner: MemberOwner) raises:
    # `sparse_coder`: the algorithm check and the dictionary admitted as
    # float32 once (`_coder_dictionary`); the plan pins it with the entries
    # and its words go to the fit by address (no copy into the plan).
    var dictionary = dl_coder_dictionary(fields, context)
    var entries = _kit_route(model, fields, context, owner)
    owner.pins = Python.tuple(entries, dictionary)
    var words = List[Float64](length=XDECOMP_CODER_PARAM_COUNT, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_CODER)
    _coding_admit(fields, words)
    words[8] = Float64(Int(py=classical_address(dictionary)))
    words[9] = Float64(Int(py=dictionary.shape[0]))
    words[10] = Float64(Int(py=dictionary.shape[1]))
    owner.plan.params = words^


def _spca_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    mut owner: MemberOwner) raises -> Bool:
    # `sparse_pca_fit`: the method check first, in its words; a V_init or a
    # U_init (arrays of the fit's shapes) keeps the door.
    var method = fields["method"]
    if kind == XDECOMP_KIND_SPCA:
        if not dl_is_str(method, "lars") and not dl_is_str(method, "cd"):
            raise Error("method must be 'lars' or 'cd'")
        if py_get(fields, "V_init") is not None or py_get(fields, "U_init") is not None:
            return False
    else:
        dl_check_algos(method, PythonObject("lasso_lars"))
    _ = _kit_route(model, fields, context, owner)
    var count = XDECOMP_SPCA_PARAM_COUNT if kind == XDECOMP_KIND_SPCA else XDECOMP_MBSPCA_PARAM_COUNT
    var words = List[Float64](length=count, fill=0.0)
    words[0] = Float64(kind)
    var requested = fields["n_components"]
    words[1] = -1.0 if requested is None else Float64(decomp_int(requested))
    if kind == XDECOMP_KIND_SPCA:
        var dw = dl_dict_words(-1, fields["alpha"], fields["max_iter"], fields["tol"], dl_fit_index(method),
                               fields["random_state"], False, False, PythonObject(1000))
        words[2] = dw.alpha
        words[3] = Float64(dw.max_iter)
        words[4] = dw.tol
        words[5] = Float64(dw.method)
        words[6] = Float64(dw.seed)
        words[7] = decomp_float(fields["ridge_alpha"])
    else:
        words[2] = decomp_float(fields["alpha"])
        words[3] = Float64(decomp_int(fields["batch_size"]))
        words[4] = Float64(decomp_seed_word(kit_seed(fields["random_state"])))
        words[5] = _flag(fields["shuffle"])
        words[6] = decomp_float(fields["max_iter"])
        var patience = fields["max_no_improvement"]
        words[7] = -1.0 if patience is None else Float64(decomp_int(patience))
        words[8] = Float64(dl_fit_index(method))
        words[9] = decomp_float(fields["tol"])
        words[10] = decomp_float(fields["ridge_alpha"])
    owner.plan.params = words^
    return True


def _kit_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    # The doors' fitted attributes, in their order (dictionary_learning_fit,
    # minibatch_dictionary_fit, sparse_coder fit, sparse_pca_fit).
    decomp_set_mode(fields, context)
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var words = scalars_view.rows * scalars_view.cols
    var count = Int(scalars[0])
    var features = Int(scalars[1])
    var iterations = Int(scalars[2])
    var steps = Int(scalars[3])
    var errors = List[Float64](capacity=max(words - 4, 0))
    for k in range(4, words):  # driver-loop: over the fit's recorded errors
        errors.append(scalars[k])
    var kind = plan.kind
    var spca = kind == XDECOMP_KIND_SPCA or kind == XDECOMP_KIND_MBSPCA
    if kind == XDECOMP_KIND_MBDL:
        py_setitem(fields, "n_steps_", PythonObject(steps))
        py_setitem(fields, "n_iter_", PythonObject(iterations))
    py_setitem(fields, "components_", _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS), context))
    if kind == XDECOMP_KIND_CODER or spca:
        py_setitem(fields, "n_components_", PythonObject(count))
    if kind == XDECOMP_KIND_DL or kind == XDECOMP_KIND_SPCA:
        py_setitem(fields, "error_", py_list_f64(errors))
        py_setitem(fields, "n_iter_", PythonObject(iterations))
    if kind == XDECOMP_KIND_MBSPCA:
        py_setitem(fields, "n_iter_", PythonObject(iterations))
    if spca:
        var view = _export(plan, call, XDECOMP_BLOCK_COMPONENTS + 1)
        var mean = decomp_f32(Python.tuple(features), context)
        if features > 0:
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(mean))), src=f32_ptr(view.data), count=features)
        py_setitem(fields, "mean_", mean)
    py_setitem(fields, "n_features_in_", PythonObject(features))


def _kit_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # What the doors' queries read: components_ (and mean_ for the
    # SparsePCA kinds); SparseCoder reads its dictionary parameter, as its
    # transform door does.
    var kind = plan.kind
    var spca = kind == XDECOMP_KIND_SPCA or kind == XDECOMP_KIND_MBSPCA
    var source = PythonObject(None)
    if kind == XDECOMP_KIND_CODER:
        source = dl_coder_dictionary(fields, context)
    else:
        source = py_get(fields, "components_")
        if source is None or (spca and py_get(fields, "mean_") is None):
            raise Error("NotFittedError: this member is not fitted yet")
    var iterations = py_get(fields, "n_iter_")
    var steps = py_get(fields, "n_steps_")
    var scalars: List[Float64] = [Float64(Int(py=source.shape[0])), Float64(Int(py=source.shape[1])),
        Float64(decomp_int(iterations)) if iterations is not None else 0.0,
        Float64(decomp_int(steps)) if steps is not None else 0.0]
    var call = member_new(plan)
    try:
        if kind == XDECOMP_KIND_CODER:
            call.block = XDECOMP_BLOCK_COMPONENTS
            call.x = MemberMatrix(Int(py=classical_address(source)), 1, Int(scalars[0]) * Int(scalars[1]))
            call.flags = 0
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        else:
            _import(plan, call, fields, "components_", 2, XDECOMP_BLOCK_COMPONENTS, context)
        if spca:
            _import(plan, call, fields, "mean_", 1, XDECOMP_BLOCK_COMPONENTS + 1, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    _ = source
    return call.state


# ---------------------------------------------------------------- EllipticEnvelope


def _mcd_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    # bindings/mcd_native_door.mojo `mcd_operation` fit: the contamination
    # check first, in its words; then `mcd_fit`'s reads (h needs n: RUN).
    var c = fields["contamination"]
    var op = native_loaded_module("operator")
    if not (Bool(py=py_call(op.lt, PythonObject(0), c)) and Bool(py=py_call(op.le, c, PythonObject(0.5)))):
        raise Error("contamination must be in (0, 0.5]")
    _ = _kit_route(model, fields, context, owner)
    var words = List[Float64](length=XDECOMP_ENVELOPE_PARAM_COUNT, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_ENVELOPE)
    var fraction = fields["support_fraction"]
    words[1] = -1.0 if fraction is None else decomp_float(fraction)
    words[2] = Float64(decomp_seed_word(kit_seed(fields["random_state"])))
    words[3] = _flag(fields["assume_centered"])
    words[4] = decomp_float(c)
    words[5] = _flag(fields["store_precision"])
    owner.plan.params = words^


def _mcd_vector(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    # A fitted (n,) float32 vector, the door's `decomp_kit_out_shape` type.
    var count = view.rows * view.cols
    var vector = decomp_f32(Python.tuple(count), context)
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(vector))), src=f32_ptr(view.data), count=count)
    return vector


def _mcd_support(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    # The door's `_support_pack`: the 0 / 1 words as the public '<u1' (n,) Array.
    var n = view.rows * view.cols
    var out = mcd_typed(n, "<u1", "B", 1, context)
    if n > 0:
        decomp_f32_to_u8(f32_ptr(view.data), MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr)), n)
    return out


def _mcd_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    # `mcd_fit`'s fitted fields, in its order.
    decomp_set_mode(fields, context)
    py_setitem(fields, "_seed", kit_seed(fields["random_state"]))
    py_setitem(fields, "_draws", PythonObject(0))
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var features = Int(scalars[0])
    var offset = scalars[2]
    py_setitem(fields, "raw_location_", _mcd_vector(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 2), context))
    py_setitem(fields, "raw_covariance_", _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 3), context))
    py_setitem(fields, "raw_support_", _mcd_support(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 6), context))
    py_setitem(fields, "location_", _mcd_vector(_export(plan, call, XDECOMP_BLOCK_COMPONENTS), context))
    py_setitem(fields, "covariance_", _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 4), context))
    var stored = _rp_matrix(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 1), context)
    py_setitem(fields, "_precision", stored)
    py_setitem(fields, "precision_", stored if Bool(py=fields["store_precision"]) else PythonObject(None))
    py_setitem(fields, "support_", _mcd_support(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 7), context))
    py_setitem(fields, "dist_", _mcd_vector(_export(plan, call, XDECOMP_BLOCK_COMPONENTS + 5), context))
    py_setitem(fields, "n_features_in_", PythonObject(features))
    py_setitem(fields, "offset_", PythonObject(offset))


def _mcd_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    # What the door's queries read: location_, _precision, offset_.
    var precision = py_get(fields, "_precision")
    var offset = py_get(fields, "offset_")
    if precision is None or py_get(fields, "location_") is None or offset is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var dist = py_get(fields, "dist_")
    var scalars: List[Float64] = [Float64(Int(py=precision.shape[0])),
        Float64(Int(py=dist.shape[0])) if dist is not None else 0.0, Float64(py=offset)]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "location_", 1, XDECOMP_BLOCK_COMPONENTS, context)
        _import(plan, call, fields, "_precision", 2, XDECOMP_BLOCK_COMPONENTS + 1, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state
