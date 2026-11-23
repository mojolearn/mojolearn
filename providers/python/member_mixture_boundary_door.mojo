# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Mixture family (GaussianMixture, kind 0;
`_mojolearn_member = (MEMBER_FAMILY_MIXTURE, 0)` on the class). Recipe:
MEMBER-ABI-V2.md; RUN: core/mixture_member.mojo.

  * `mixture_member_admit` (UNPACK, once per member): the default fit only.
    An instance whose options need the option-parity fit
    (`_gmm_needs_ext`: covariance_type other than 'full', init_params
    'k-means++' or 'random_from_data', n_init != 1, warm_start, explicit
    starts) answers False and the caller keeps the class's own doors, as
    does an init_params name the entry would refuse (the class's door
    refuses it by name) and the inference-only CPU binding (no fit). Else
    the estimator's own reads in its fit's order
    (bindings/mixture_public_native_unpack.mojo `gmm_default_refuse_options`,
    `gmm_fit_settings`, the mode check, `gmm_fit_random_state`), its binding
    through the estimator's own route (`estimator_binding(model,
    "_mojolearn_mixture")`) and that binding's `mixture_native_api_v1`
    entry as the route's `entry`, its capability word in `route.flags`.
  * `mixture_member_pack` (PACK): the attributes `_gmm_default_fit` writes
    (bindings/mixture_public_native_pack.mojo: `_ext` None, n_features_in_,
    weights_, means_, covariances_, precisions_cholesky_, log_det_chol_,
    n_iter_, converged_, lower_bound_), after dropping the model's admission
    and operand records as the class's own fit does.
  * `mixture_member_state` (UNPACK): a member state from a fitted model.

SERIAL BY NATURE: one member's admission or publication (a few attribute
reads and block copies); a caller runs its members in parallel.
"""
from std.memory import bitcast, unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.boundary_read_door import read_bool
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cluster_side_records_unpack import cluster_record_drop
from bindings.family_common_pack import family_array
from bindings.member_owner_unpack import MemberOwner
from bindings.mixture_public_native_unpack import (
    _GMM_ADMIT_SLOT, _GMM_OPERANDS_SLOT, _gmm_needs_ext, _gmm_connection, _check_mode, _get,
    gmm_default_refuse_options, gmm_fit_settings, gmm_fit_random_state,
)
from bindings.py_balanced import py_call, py_get, py_setattr
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU, device_vendor_code
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from core.mixture_native_api import MIXTURE_NATIVE_MODE, MIXTURE_NATIVE_CAPABILITIES, MIXTURE_NATIVE_COUNT
from providers.python.host_estimator_binding_pyapi_door import estimator_member_binding


def mixture_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False where the class's own doors run the
    instance (see the module notes)."""
    if kind != 0 or _gmm_needs_ext(model):
        return False
    classical_connect_numpy(context)
    gmm_default_refuse_options(model)
    var settings = gmm_fit_settings(model)
    if settings.init_name != "kmeans" and settings.init_name != "random":
        return False
    # An explicit Host* class (its own host-only word, lane mf-iterimp3b): the host image its own methods
    # bind (`estimator_member_binding`); any other class: the estimator's binding route, as before.
    var binding = estimator_member_binding(model, "_mojolearn_mixture", context, False)
    # The inference-only CPU binding registers the scoring transport but no fit.
    if String(py=binding.__name__).endswith("_infer_host"):
        return False
    _check_mode(binding, model, context)
    var random_state = gmm_fit_random_state(model)
    var connection = _gmm_connection(binding)
    var mode = connection.driver.metadata(MIXTURE_NATIVE_MODE)
    var vendor = device_vendor_code(String(py_call(binding.mixture_vendor)))
    if vendor < 0:
        raise Error("ImportError: mojolearn: the mixture binding names no vendor")
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(connection.driver.entry, entry_ref_word(mode, vendor, MIXTURE_NATIVE_COUNT))
    route.flags = connection.driver.metadata(MIXTURE_NATIVE_CAPABILITIES)
    owner.plan.route = route
    var words: List[Float64] = [Float64(settings.k), Float64(settings.max_iter), settings.tol, settings.reg_covar,
        bitcast[DType.float64](Int64(random_state)), 1.0 if settings.init_name == "random" else 0.0]
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, connection.provider)
    return True


def _block(plan: MemberPlan, mut call: MemberCall, block: Int, shape: PythonObject,
    context: PythonObject) raises -> PythonObject:
    """EXPORT one float32 block into a new Array of `shape` (one copy)."""
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var array = family_array(shape, "<f4", context)
    var count = call.dst.rows * call.dst.cols
    if count > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(array))), src=f32_ptr(call.dst.data), count=count)
    return array


def mixture_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    `_gmm_default_fit` does (same names, dtypes and shapes)."""
    classical_connect_numpy(context)
    # A fit replaces the fitted state: the class's own fit drops these first.
    cluster_record_drop(_GMM_ADMIT_SLOT, model)
    cluster_record_drop(_GMM_OPERANDS_SLOT, model)
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_FAMILY + 1
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var k = call.dst.rows
    var d = call.dst.cols
    var weights = _block(plan, call, MEMBER_BLOCK_FAMILY, Python.tuple(k), context)
    var means = _block(plan, call, MEMBER_BLOCK_FAMILY + 1, Python.tuple(k, d), context)
    var covariances = _block(plan, call, MEMBER_BLOCK_FAMILY + 2, Python.tuple(k, d, d), context)
    var precisions = _block(plan, call, MEMBER_BLOCK_FAMILY + 3, Python.tuple(k, d, d), context)
    var log_det = _block(plan, call, MEMBER_BLOCK_FAMILY + 4, Python.tuple(k), context)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var n_iter = Int(scalars[0])
    var converged = scalars[1] != 0.0
    var lower = scalars[2]
    py_setattr(model, "_ext", PythonObject(None))
    py_setattr(model, "n_features_in_", PythonObject(d))
    py_setattr(model, "weights_", weights)
    py_setattr(model, "means_", means)
    py_setattr(model, "covariances_", covariances)
    py_setattr(model, "precisions_cholesky_", precisions)
    py_setattr(model, "log_det_chol_", log_det)
    py_setattr(model, "n_iter_", PythonObject(n_iter))
    py_setattr(model, "converged_", PythonObject(converged))
    py_setattr(model, "lower_bound_", PythonObject(lower))


def mixture_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted model of `model` (a
    default-fit GaussianMixture: `_ext` None). The caller frees it."""
    classical_connect_numpy(context)
    if py_get(fields, PythonObject("_ext"), PythonObject(None)) is not None:
        raise Error("TypeError: mojolearn: this GaussianMixture was fitted by the option-parity fit;"
            + " it is not a member state")
    var source = py_get(fields, PythonObject("weights_"), PythonObject(None))
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var names: List[String] = ["weights_", "means_", "covariances_", "precisions_cholesky_", "log_det_chol_"]
    var arrays = List[PythonObject]()
    for i in range(5):  # driver-loop: five fitted arrays
        arrays.append(admit_family(_get(model, names[i]), -1, names[i], context).array)
    var k = Int(py=arrays[0].size)
    var d = Int(py=arrays[1].size) // k if k > 0 else 0
    var scalars: List[Float64] = [Float64(py=_get(model, "n_iter_")), 1.0 if read_bool(_get(model, "converged_")) else 0.0,
        Float64(py=_get(model, "lower_bound_"))]
    var call = member_new(plan)
    try:
        # The means first: they set the feature count.
        var order: List[Int] = [1, 0, 2, 3, 4]
        for j in range(5):  # driver-loop: five fitted blocks
            var i = order[j]
            var rows = 1 if (i == 0 or i == 4) else k
            call.block = MEMBER_BLOCK_FAMILY + i
            call.x = MemberMatrix(Int(py=classical_address(arrays[i])), rows, Int(py=arrays[i].size) // rows)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 3, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(arrays)
    _ = len(scalars)
    _ = source
    _ = d
    return call.state
