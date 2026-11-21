# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XNeighbors family (UNPACK / PACK of
core/xneighbors_member.mojo). Published: NearestCentroid (kind 0),
LocalOutlierFactor (kind 1), OneClassSVM (kind 2). Not yet published (admit
refuses with the kind): KernelPCA PolynomialCountSketch AdditiveChi2Sampler SkewedChi2Sampler
LabelPropagation LabelSpreading KNNImputer SVGP.

  * `xneighbors_member_admit` (UNPACK): the estimator's own parameter checks
    (bindings/neighbors_local_native_door.mojo `_nc_fit_unpack`), its route
    (`estimator_binding(model, "_mojolearn_x_neighbors")`, the image its own
    fit and predict call) -> `route.entry` = that image's x_neighbors native
    operations entry, `route.mode` / `route.vendor` = the image's own words
    (`x_neighbors_numeric_mode`, `x_neighbors_vendor`), the image's fused and
    wide capabilities as parameter words (the connection's cached probe).
    A device image built with no GPU vendor publishes nothing: admit answers
    False and the caller keeps the estimator's doors.
  * `xneighbors_member_pack` (PACK): the attributes the estimator's fit
    writes (`centroids_`, `deviations_`, `within_class_std_dev_`,
    `class_prior_`, `n_features_in_`); `classes_` and `_codes_classes` are
    the caller's labels (`member_pack_labeled`, taken with
    `member_labels_take`; predict maps codes through `classes_`), or the
    class codes 0 .. classes-1 when it gave none.
    LocalOutlierFactor: `_fit_X` (a new Array: the member's copy of the rows
    it borrowed), `_fit_dist`, `_lrd`, `negative_outlier_factor_`, `offset_`,
    `n_neighbors_`, `n_features_in_`, `n_samples_fit_`, as `_lof_fit_pack`.
    OneClassSVM: admit = `oc_fit_unpack`'s checks and words with the door's
    own helpers on the shared family context (a precomputed kernel answers
    False: the caller keeps the doors), the solve the door picks for the
    image (the resident solve's entry in `route.second`), the image's fused
    labels and `family_svc_xn_old_items`; pack = `oc_fit_pack`'s fields.
  * `xneighbors_member_state` (UNPACK): a member state from a fitted
    estimator's arrays (NearestCentroid: the class count is `class_prior_`'s
    length).
  The LocalOutlierFactor A/B arms (`unfused_knn`, `old_items`) are read
  where the estimator's own door reads them: `family_xn_loc_flags` of the
  shared family context (`_native_family_bridge._CONTEXT`), never defaulted.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and four block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x, admit_c_view, admitted_dtype, admitted_storage_native
from bindings.classical_arrays_door import classical_address
from bindings.classical_labels_door import labels_cast
from bindings.family_common_pack import family_field
from bindings.member_owner_unpack import MemberOwner, member_labels_take
from bindings.nearest_centroid_provider_unpack import NcConnection
from bindings.ocsvm_provider_unpack import OcConnection
from bindings.ocsvm_scalar_boundary_door import oc_kernel_admit, oc_parameter_repr, oc_precomputed, oc_name
from bindings.svc_ocsvm_boundary_unpack import (
    _oc_float, _oc_int, _oc_kernel_code, _fast_tier_native, _ocsvm_labels_available,
)
from bindings.svc_common_pack import svc_scalar_array
from bindings.svc_common_unpack import svc_truthy
from core.ocsvm_resident_api import OCSVM_RESIDENT_COUNT
from svm.svc_ocsvm_native_run import OC_SOLVE_GRAM, OC_SOLVE_RESIDENT, OC_SOLVE_DESCRIPTOR
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from providers.python.x_neighbors_admission_pyapi_door import xn_is_module, xn_capabilities
from bindings.neighbors_local_native_door import _lof_check
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_contains, py_fields, py_get, py_setitem, py_type
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU, device_vendor_code
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_IMPORT_DONE, MEMBER_F64, MEMBER_I32,
)
from core.xneighbors_member import (
    XNEIGHBORS_KIND_NEAREST_CENTROID, XNEIGHBORS_KIND_LOF, XNEIGHBORS_KIND_OCSVM,
    OC_BLOCK_VECTORS, OC_BLOCK_DUAL, OC_BLOCK_SUPPORT, OC_BLOCK_SCALARS, OC_GAMMA_VALUE, OC_GAMMA_SCALE, OC_GAMMA_AUTO,
    NC_BLOCK_CENTROIDS, NC_BLOCK_STD, NC_BLOCK_DEVIATIONS, NC_BLOCK_PRIOR,
    LOF_BLOCK_ROWS, LOF_BLOCK_DISTANCES, LOF_BLOCK_DENSITIES, LOF_BLOCK_SCORES, LOF_BLOCK_OFFSET,
)
from neighbors.neighbors_local_run import nc_refuse_metric, nc_check_priors_ndim
from neighbors.nearest_centroid_run import NC_PRIORS_EMPIRICAL, NC_PRIORS_UNIFORM, NC_PRIORS_GIVEN
from providers.python.estimator_binding_pyapi_door import estimator_binding
from providers.python.x_neighbors_descriptor_pyapi_door import xn_admit, xn_module
from x_neighbors.descriptor_schema import XN_OPERATION_COUNT

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def _image_word(module: PythonObject, name: String, b: PythonObject) raises -> PythonObject:
    """One compiled export of the x_neighbors image, called once at UNPACK."""
    var reader = py_call(b.getattr, module, PythonObject(name), PythonObject(None))
    if reader is None:
        raise Error("ImportError: mojolearn: the x_neighbors binding has no " + name + "; rebuild it")
    return py_call(reader)


def _route(model: PythonObject, context: PythonObject, b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """The route of the image the estimator's own doors call: its x_neighbors
    native operations entry and the image's (mode, vendor) words. False: a
    device image built with no GPU vendor (it publishes no member entry)."""
    var binding = estimator_binding(model, "_mojolearn_x_neighbors", context)
    var module = xn_module(binding)
    var mode = Int(py=_image_word(module, "x_neighbors_numeric_mode", b))
    var vendor = device_vendor_code(String(_image_word(module, "x_neighbors_vendor", b)))
    if vendor < 0:
        return False
    var admitted = xn_admit(binding)
    var connection = NcConnection(admitted)
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(connection.driver.entry, entry_ref_word(mode, vendor, XN_OPERATION_COUNT))
    owner.plan.route = route
    owner.pins = Python.tuple(binding, admitted, connection.provider)
    # The image's capability words (the connection's cached probe).
    owner.plan.params = [1.0 if connection.fused else 0.0, 1.0 if connection.wide else 0.0]
    _ = connection^
    return True


def _family_context() raises -> PythonObject:
    """The shared family context every x_neighbors door is called with
    (`_native_family_bridge._CONTEXT`, the estimator modules' `_NATIVE_CONTEXT`):
    the tables and A/B arms the doors read come from here, never from a
    caller's context."""
    var bridge = py_fields(native_loaded_module("mojolearn._native_family_bridge"))
    var family = py_get(bridge, PythonObject("_CONTEXT"), PythonObject(None))
    if family is None:
        raise Error("ImportError: mojolearn: the family bridge is not connected")
    return family


def _loc_flags() raises -> PythonObject:
    """The LocalOutlierFactor A/B arms exactly where the estimator's own door
    reads them (bindings/neighbors_local_native_door.mojo `_flag`): the
    `family_xn_loc_flags` table of the shared family context every
    x_neighbors door is called with (`_native_family_bridge._CONTEXT`, filled
    at `_expansion_neighbors` import). Never a default: a missing table is
    the same KeyError the door raises, so the caller falls back."""
    var family = _family_context()
    var flags = py_get(family, PythonObject("family_xn_loc_flags"), PythonObject(None))
    if flags is None:
        raise Error("KeyError: 'family_xn_loc_flags'")
    return flags


def _flag_word(flags: PythonObject, name: String) raises -> Float64:
    return 1.0 if Bool(py=flags[PythonObject(name)]) else 0.0


def _lof_admit(fields: PythonObject, context: PythonObject, b: PythonObject, mut owner: MemberOwner) raises:
    # `LocalOutlierFactor._check_params`, the estimator's own checks.
    _lof_check(fields, context)
    var contamination = family_field(fields, "contamination")
    var auto = py_type(contamination) is b.str and String(contamination) == "auto"
    var flags = _loc_flags()
    var words: List[Float64] = [Float64(XNEIGHBORS_KIND_LOF),
        Float64(Int(py=py_call(b.int, family_field(fields, "n_neighbors")))),
        1.0 if Bool(py=py_call(b.bool, family_field(fields, "novelty"))) else 0.0,
        1.0 if auto else 0.0, 0.0 if auto else Float64(py=py_call(b.float, contamination)),
        _flag_word(flags, "unfused_knn"), _flag_word(flags, "old_items")]
    owner.plan.params = words^


def _ocsvm_admit(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    mut owner: MemberOwner) raises -> Bool:
    """`oc_fit_unpack`'s parameter checks and words, in its order, read with
    the door's own helpers on the shared family context; the solve and the
    query path the door would pick for this image. False (not a member, the
    caller keeps the estimator's doors): a precomputed kernel (its queries
    index the Gram columns), or an image that publishes no member entry."""
    var family = _family_context()
    var kernel = family_field(fields, "kernel")
    if Bool(py=py_call(b.callable, kernel)):
        raise Error("NotImplementedError: OneClassSVM callable kernels require a native kernel name")
    if not oc_kernel_admit(kernel, family):
        raise Error("NotImplementedError: OneClassSVM: kernel=" + oc_parameter_repr(kernel, family)
            + " is not implemented")
    var nu = _oc_float(family_field(fields, "nu"), family)
    if not (0.0 < nu and nu <= 1.0):
        raise Error("nu must be in (0, 1]")
    if oc_precomputed(kernel, family):
        return False
    var degree = _oc_int(family_field(fields, "degree"), family)
    var coef0 = _oc_float(family_field(fields, "coef0"), family)
    var gamma = family_field(fields, "gamma")
    var gamma_mode = OC_GAMMA_VALUE
    var gamma_value = 0.0
    if gamma is None or oc_name(gamma, "auto", family):
        gamma_mode = OC_GAMMA_AUTO
    elif oc_name(gamma, "scale", family):
        gamma_mode = OC_GAMMA_SCALE
    else:
        gamma_value = _oc_float(gamma, family)
    var max_iter = _oc_int(family_field(fields, "max_iter"), family)
    var tol = _oc_float(family_field(fields, "tol"), family)
    var code = _oc_kernel_code(kernel, family)
    if not _route(model, context, b, owner):
        return False
    var admitted = owner.pins[1]
    var connection = OcConnection(admitted)
    # The door's resident probe, once per member (the fit asks it per call).
    var fast = False
    var resident_available = False
    if xn_is_module(admitted):
        fast = _fast_tier_native(connection, admitted, family)
        var entry = "x_neighbors_ocsvm_resident" if fast else "x_neighbors_ocsvm_resident_idn"
        resident_available = neural_descriptor_entry(xn_module(admitted), entry, False) is not None
    else:
        resident_available = (xn_capabilities(admitted) & 1) != 0
    var solve = OC_SOLVE_GRAM
    if resident_available:
        if xn_is_module(admitted):
            solve = OC_SOLVE_RESIDENT
            var resident = connection.resident_driver()
            owner.plan.route.second = EntryRef(resident.entry,
                entry_ref_word(owner.plan.route.mode, owner.plan.route.vendor, OCSVM_RESIDENT_COUNT))
            owner.plan.route.has_second = True
        else:
            solve = OC_SOLVE_DESCRIPTOR
    var fused_labels = _ocsvm_labels_available(admitted)
    var old_items = svc_truthy(family["family_svc_xn_old_items"])
    var words: List[Float64] = [Float64(XNEIGHBORS_KIND_OCSVM), Float64(code), nu, Float64(degree), coef0,
        Float64(gamma_mode), gamma_value, Float64(10000000 if max_iter < 0 else max_iter), tol, Float64(solve),
        1.0 if fast else 0.0, 1.0 if fused_labels else 0.0, 1.0 if old_items else 0.0]
    owner.plan.params = words^
    owner.pins = Python.tuple(owner.pins, connection.provider, connection.resident_provider)
    _ = connection^
    return True


def xneighbors_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: its image publishes no member entry."""
    if kind == XNEIGHBORS_KIND_OCSVM:
        return _ocsvm_admit(model, fields, context, b, owner)
    if kind != XNEIGHBORS_KIND_NEAREST_CENTROID and kind != XNEIGHBORS_KIND_LOF:
        raise Error("ValueError: mojolearn: XNeighbors member kind " + String(kind) + " is not published")
    if kind == XNEIGHBORS_KIND_LOF:
        _lof_admit(fields, context, b, owner)
        var lof_words = owner.plan.params.copy()
        if not _route(model, context, b, owner):
            return False
        owner.plan.params = lof_words^
        return True
    # The estimator's own parameter checks, in its fit's order.
    var metric = family_field(fields, "metric")
    if not py_contains(Python.tuple(PythonObject("euclidean"), PythonObject("manhattan")), metric):
        nc_refuse_metric()
    var euclidean = String(metric) == "euclidean"
    var priors = family_field(fields, "priors")
    var priors_kind = NC_PRIORS_GIVEN
    var given = PythonObject(None)
    if py_type(priors) is b.str:
        var word = String(priors)
        if word == "empirical":
            priors_kind = NC_PRIORS_EMPIRICAL
        elif word == "uniform":
            priors_kind = NC_PRIORS_UNIFORM
    if priors_kind == NC_PRIORS_GIVEN:
        given = admit_c_view(priors, "priors", context)
        nc_check_priors_ndim(Int(py=given.ndim))
    var threshold = family_field(fields, "shrink_threshold")
    # The door's `float(threshold) if threshold else 0.0`: a zero threshold
    # and None both mean no shrinking.
    var shrink = 0.0
    if threshold is not None:
        shrink = Float64(py=py_call(b.float, threshold))
    # The image the estimator's own fit and predict call, and its words.
    if not _route(model, context, b, owner):
        return False
    var words: List[Float64] = [Float64(kind), 0.0 if euclidean else 1.0, Float64(priors_kind), shrink,
        owner.plan.params[0], owner.plan.params[1]]
    if priors_kind == NC_PRIORS_GIVEN:
        var count = Int(py=given.size)
        var start = len(words)
        words.resize(start + count, 0.0)
        labels_cast(Int(py=classical_address(given)), admitted_dtype(given, context),
            Int(words.unsafe_ptr()) + start * 8, "<f8", count, context)
    _ = given
    owner.plan.params = words^
    return True


def _publish_block(call: MemberCall, typestr: String, context: PythonObject, width: Int,
    one_dimensional: Bool) raises -> PythonObject:
    """A new Array holding the exported block (the estimator's own fit
    allocates the same attribute with `admitted_storage_native`)."""
    var rows = call.dst.rows
    var cols = call.dst.cols
    var shape: List[Int] = [rows, cols]
    if one_dimensional:
        shape = [cols]
    var array = admitted_storage_native(shape, typestr, context, as_array=True)
    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(array))),
        src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=rows * cols * width)
    return array


def _nc_pack(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan, state: Int) raises:
    var call = MemberCall(plan.route, state)
    call.block = NC_BLOCK_PRIOR
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var prior = _publish_block(call, "<f8", context, 8, True)
    var classes = call.dst.cols
    call.block = NC_BLOCK_CENTROIDS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var centroids = _publish_block(call, "<f4", context, 4, False)
    var features = call.dst.cols
    call.block = NC_BLOCK_DEVIATIONS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var deviations = _publish_block(call, "<f4", context, 4, False)
    call.block = NC_BLOCK_STD
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var std = _publish_block(call, "<f4", context, 4, True)
    # The caller's labels (member_pack_labeled), taken once: code c is
    # labels[c]; the estimator's own fit writes its classes to both fields.
    var codes = member_labels_take(fields, b)
    if codes is None:
        codes = Python.list(py_call(b.range, PythonObject(classes)))
    py_setitem(fields, "class_prior_", prior)
    py_setitem(fields, "centroids_", centroids)
    py_setitem(fields, "deviations_", deviations)
    py_setitem(fields, "within_class_std_dev_", std)
    py_setitem(fields, "classes_", codes)
    py_setitem(fields, "_codes_classes", codes)
    py_setitem(fields, "n_features_in_", PythonObject(features))


def _f32_block(fields: PythonObject, name: String, ndim: Int, context: PythonObject) raises -> PythonObject:
    var source = py_get(fields, PythonObject(name))
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    return admit_x(source, ndim, name, context)


def _nc_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    var centroids = _f32_block(fields, "centroids_", 2, context)
    var deviations = _f32_block(fields, "deviations_", 2, context)
    var std = _f32_block(fields, "within_class_std_dev_", 1, context)
    var prior_source = py_get(fields, "class_prior_")
    if prior_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var prior_view = admit_c_view(prior_source, "class_prior_", context)
    var classes = Int(py=prior_view.size)
    var prior = admitted_storage_native([classes], "<f8", context, as_array=True)
    labels_cast(Int(py=classical_address(prior_view)), admitted_dtype(prior_view, context),
        Int(py=classical_address(prior)), "<f8", classes, context)
    var features = Int(py=std.size)
    var call = member_new(plan)
    try:
        call.block = NC_BLOCK_CENTROIDS
        call.x = MemberMatrix(Int(py=classical_address(centroids)), Int(py=centroids.shape[0]),
            Int(py=centroids.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = NC_BLOCK_DEVIATIONS
        call.x = MemberMatrix(Int(py=classical_address(deviations)), Int(py=deviations.shape[0]),
            Int(py=deviations.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = NC_BLOCK_STD
        call.x = MemberMatrix(Int(py=classical_address(std)), 1, features)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = NC_BLOCK_PRIOR
        call.x = MemberMatrix(Int(py=classical_address(prior)), 1, classes, MEMBER_F64)
        call.classes = classes
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = centroids
    _ = deviations
    _ = std
    _ = prior_view
    _ = prior
    return call.state


def _lof_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    """`_lof_fit_pack`'s fields, in its order."""
    var call = MemberCall(plan.route, state)
    call.block = LOF_BLOCK_ROWS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var fit_x = _publish_block(call, "<f4", context, 4, False)
    var rows = call.dst.rows
    var features = call.dst.cols
    call.block = LOF_BLOCK_DISTANCES
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var distances = _publish_block(call, "<f4", context, 4, False)
    var k = call.dst.cols
    call.block = LOF_BLOCK_DENSITIES
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var densities = _publish_block(call, "<f4", context, 4, True)
    call.block = LOF_BLOCK_SCORES
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scores = _publish_block(call, "<f4", context, 4, True)
    call.block = LOF_BLOCK_OFFSET
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var offset = _F64P(unsafe_from_address=call.dst.data)[0]
    py_setitem(fields, "_fit_X", fit_x)
    py_setitem(fields, "_fit_dist", distances)
    py_setitem(fields, "_lrd", densities)
    py_setitem(fields, "n_neighbors_", PythonObject(k))
    py_setitem(fields, "n_features_in_", PythonObject(features))
    py_setitem(fields, "n_samples_fit_", PythonObject(rows))
    py_setitem(fields, "negative_outlier_factor_", scores)
    py_setitem(fields, "offset_", PythonObject(offset))


def _lof_state(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan) raises -> Int:
    var fit_x = _f32_block(fields, "_fit_X", 2, context)
    var distances = _f32_block(fields, "_fit_dist", 2, context)
    var densities = _f32_block(fields, "_lrd", 1, context)
    var scores = _f32_block(fields, "negative_outlier_factor_", 1, context)
    var offset_source = py_get(fields, PythonObject("offset_"))
    if offset_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var offset: List[Float64] = [Float64(py=py_call(b.float, offset_source))]
    var call = member_new(plan)
    try:
        call.block = LOF_BLOCK_ROWS
        call.x = MemberMatrix(Int(py=classical_address(fit_x)), Int(py=fit_x.shape[0]), Int(py=fit_x.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = LOF_BLOCK_DISTANCES
        call.x = MemberMatrix(Int(py=classical_address(distances)), Int(py=distances.shape[0]),
            Int(py=distances.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = LOF_BLOCK_DENSITIES
        call.x = MemberMatrix(Int(py=classical_address(densities)), 1, Int(py=densities.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = LOF_BLOCK_SCORES
        call.x = MemberMatrix(Int(py=classical_address(scores)), 1, Int(py=scores.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = LOF_BLOCK_OFFSET
        call.x = MemberMatrix(Int(offset.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = fit_x
    _ = distances
    _ = densities
    _ = scores
    _ = offset^
    return call.state


def _ocsvm_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    """`oc_fit_pack`'s fields (an unweighted, kernel fit), in its order."""
    var call = MemberCall(plan.route, state)
    call.block = OC_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = _F64P(unsafe_from_address=call.dst.data)
    var gamma = scalars[0]
    var rho = scalars[1]
    var iterations = Int(scalars[2])
    var features = Int(scalars[3])
    call.block = OC_BLOCK_SUPPORT
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var support = _publish_block(call, "<i4", context, 4, True)
    call.block = OC_BLOCK_VECTORS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var vectors = _publish_block(call, "<f4", context, 4, False)
    call.block = OC_BLOCK_DUAL
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var dual = _publish_block(call, "<f4", context, 4, False)
    py_setitem(fields, "_gamma", PythonObject(gamma))
    py_setitem(fields, "support_", support)
    py_setitem(fields, "support_vectors_", vectors)
    py_setitem(fields, "dual_coef_", dual)
    py_setitem(fields, "intercept_", svc_scalar_array(-rho, "<f4", context))
    py_setitem(fields, "offset_", PythonObject(rho))
    py_setitem(fields, "n_iter_", PythonObject(iterations))
    py_setitem(fields, "n_features_in_", PythonObject(features))
    py_setitem(fields, "fit_status_", PythonObject(0))


def _ocsvm_state(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan) raises -> Int:
    var vectors = _f32_block(fields, "support_vectors_", 2, context)
    var dual = _f32_block(fields, "dual_coef_", 2, context)
    var support_source = py_get(fields, PythonObject("support_"))
    var gamma_source = py_get(fields, PythonObject("_gamma"))
    var offset_source = py_get(fields, PythonObject("offset_"))
    var iterations_source = py_get(fields, PythonObject("n_iter_"))
    if support_source is None or gamma_source is None or offset_source is None or iterations_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var support_view = admit_c_view(support_source, "support_", context)
    var supports = Int(py=support_view.size)
    var support = admitted_storage_native([supports], "<i4", context, as_array=True)
    labels_cast(Int(py=classical_address(support_view)), admitted_dtype(support_view, context),
        Int(py=classical_address(support)), "<i4", supports, context)
    var scalars: List[Float64] = [Float64(py=py_call(b.float, gamma_source)),
        Float64(py=py_call(b.float, offset_source)), Float64(Int(py=py_call(b.int, iterations_source))),
        Float64(Int(py=vectors.shape[1]))]
    var call = member_new(plan)
    try:
        call.block = OC_BLOCK_VECTORS
        call.x = MemberMatrix(Int(py=classical_address(vectors)), Int(py=vectors.shape[0]), Int(py=vectors.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = OC_BLOCK_DUAL
        call.x = MemberMatrix(Int(py=classical_address(dual)), 1, Int(py=dual.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = OC_BLOCK_SUPPORT
        call.x = MemberMatrix(Int(py=classical_address(support)), 1, supports, MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = OC_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 4, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = vectors
    _ = dual
    _ = support_view
    _ = support
    _ = scalars^
    return call.state


def xneighbors_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does."""
    if plan.kind == XNEIGHBORS_KIND_LOF:
        _lof_pack(fields, context, plan, state)
    elif plan.kind == XNEIGHBORS_KIND_OCSVM:
        _ocsvm_pack(fields, context, plan, state)
    else:
        _nc_pack(fields, context, b, plan, state)


def xneighbors_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    if plan.kind == XNEIGHBORS_KIND_LOF:
        return _lof_state(fields, context, b, plan)
    if plan.kind == XNEIGHBORS_KIND_OCSVM:
        return _ocsvm_state(fields, context, b, plan)
    return _nc_state(fields, context, plan)
