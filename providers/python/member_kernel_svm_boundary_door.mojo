# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the KernelSvm family (SVC, SVR): its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; worked form
providers/python/member_linear_boundary_door.mojo). Kind 0 is SVC, kind 1
SVR (`_mojolearn_member = (MEMBER_FAMILY_KERNEL_SVM, kind)` on each class).

  * `kernel_svm_member_admit` (UNPACK, once per member): the estimator's
    own parameter checks (`svm_settings_unpack`: its refusals and the
    `_hp_*` working values), its binding through the estimator's binding
    route (device='cpu' beside a GPU set answers the host image), that
    binding's SVC tables (and the SVR table for kind 1 or a GPU binding) in the route's
    slots, and the 17 parameter words of core/kernel_svm_member.mojo.
    kernel='precomputed' and an SVC class_weight are not published: admit
    returns False and the caller keeps the estimator's own doors.
  * `kernel_svm_member_pack` (PACK): the fitted attributes the estimator's
    own fit writes (bindings/svc_kernel_fit_pack.mojo,
    bindings/svc_kernel_svr_pack.mojo), from the member state's blocks.
    `classes_` is the caller's.
  * `kernel_svm_member_state` (UNPACK): a member state from a fitted SVC or
    SVR.

SERIAL BY NATURE: one member's admission or publication (record reads and
one copy per fitted block); a caller runs its members in parallel.
"""
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_address
from bindings.family_common_pack import family_call, family_array
from bindings.member_owner_unpack import MemberOwner, member_host_only
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_contains, py_get, py_item, py_setitem
from bindings.svc_common_pack import svc_prefix, svc_scalar_array
from bindings.svc_common_unpack import svc_field, svc_has, svc_float, svc_int, svc_read_f32
from bindings.svc_kernel_params_unpack import svm_settings_unpack, SVM_GAMMA_SCALE
from bindings.svc_kernel_provider_unpack import svc_connection
from bindings.svc_kernel_rows_unpack import _i32_array
from bindings.svm_keys_unpack import svm_keys
from bindings.svr_native_boundary_unpack import svr_connection
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_FAMILY,
    MEMBER_IMPORT_DONE, MEMBER_F64, MEMBER_I32,
)
from core.svc_native_api import SVC_TABLE_COUNT, SVC_DEVICE_COUNT
from core.svr_native_api import SVR_NATIVE_COUNT
from providers.python.estimator_binding_pyapi_door import estimator_binding, estimator_binding_host
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from x_decomp.descriptor_schema import xd_vendor_code

# The scale_gamma bit of the SVC table's flags (bindings/svc_kernel_params_unpack.mojo).
comptime _PY2MOJO_SCALE_GAMMA = 1
# core/kernel_svm_member.mojo's block names (docstring order).
comptime _B_SUPPORT = MEMBER_BLOCK_FAMILY
comptime _B_VECTORS = MEMBER_BLOCK_FAMILY + 1
comptime _B_DUAL = MEMBER_BLOCK_FAMILY + 2
comptime _B_N_SUPPORT = MEMBER_BLOCK_FAMILY + 3
comptime _B_N_ITER = MEMBER_BLOCK_FAMILY + 4
comptime _B_MACHINES = MEMBER_BLOCK_FAMILY + 5
comptime _B_PROB = MEMBER_BLOCK_FAMILY + 6
comptime _B_MACHINE0 = MEMBER_BLOCK_FAMILY + 16


def kernel_svm_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this configuration has no member
    entry (kernel='precomputed', an SVC class_weight)."""
    var classifier = kind == 0
    if kind != 0 and kind != 1:
        raise Error("ValueError: mojolearn: unknown kernel SVM member kind " + String(kind))
    # The estimator's own checks, on a SCRATCH copy of its fields: the private
    # settings they write (`_hp_kernel`, `_hp_C`, ...) never reach the caller's
    # template estimator; PACK writes them into the clone it packs.
    var scratch = py_call(native_loaded_module("builtins").dict, fields)
    var hp = svm_settings_unpack(model, scratch, classifier, context)
    var nk = svm_keys()
    if hp.precomputed:
        return False
    if classifier and py_get(scratch, nk[].k_class_weight, PythonObject(None)) is not None:
        return False
    # An explicit HostSVC / HostSVR (its own host-only word, lane mf-cal3b): the host image its own fit
    # binds; any other class: the estimator's binding route.
    var binding = (estimator_binding_host(model, "_mojolearn_svm", context) if member_host_only(model)
                   else estimator_binding(model, "_mojolearn_svm", context))
    var connection = svc_connection(binding)
    var drivers = connection.drivers.copy()
    if hp.gamma_kind == SVM_GAMMA_SCALE and (drivers.flags() & _PY2MOJO_SCALE_GAMMA) == 0:
        raise Error("RuntimeError: mojolearn: this binding has no scale_gamma_limbs (an older binary or a "
                    + "-D MOJOLEARN_PY2MOJO_linear_OFF build); rebuild it")
    var mode = drivers.mode()
    # The binary's own compiled vendor word (read from its exact module
    # dictionary: 'cpu' for the host image, else 'metal' / 'cuda' / 'hip').
    var vendor = xd_vendor_code(String(py_call(neural_descriptor_entry(binding, "svm_vendor"))))
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(drivers.table, entry_ref_word(mode, vendor, SVC_TABLE_COUNT))
    if drivers.has_device:
        route.second = EntryRef(drivers.device, entry_ref_word(mode, vendor, SVC_DEVICE_COUNT))
        route.has_second = True
    var svr_owner = PythonObject(None)
    # The SVR table: SVR's own entries, and on a GPU binding the lease of the
    # state's device copies for SVC too (core/kernel_svm_member.mojo `_lease`).
    if not classifier or drivers.has_device:
        var svr = svr_connection(binding)
        route.third = EntryRef(svr.driver.entry, entry_ref_word(mode, vendor, SVR_NATIVE_COUNT))
        route.has_third = True
        svr_owner = svr.provider
    if connection.candidate:
        route.flags = 1
    # random_state (validated: None or an int in [0, 2**64)) as two 32-bit halves.
    var low = 0
    var high = 0
    if classifier and hp.probability:
        var rs = svc_field(scratch, nk[].k__hp_random_state, "_hp_random_state")
        if rs is not None:
            var operator = native_loaded_module("operator")
            low = Int(py=family_call(operator.and_, rs, PythonObject(0xFFFFFFFF)))
            high = Int(py=family_call(operator.rshift, rs, PythonObject(32)))
    var shape_ovo = 0.0
    var break_ties = 0.0
    if classifier:
        if String(svc_field(scratch, nk[].k_decision_function_shape, "decision_function_shape")) == "ovo":
            shape_ovo = 1.0
        if Bool(py=svc_field(scratch, nk[].k__hp_break_ties, "_hp_break_ties")):
            break_ties = 1.0
    var words = List[Float64](length=17, fill=0.0)
    words[0] = Float64(kind)
    words[1] = Float64(hp.kernel)
    words[2] = Float64(hp.degree)
    words[3] = hp.coef0
    words[4] = Float64(hp.gamma_kind)
    words[5] = hp.gamma
    words[6] = hp.c
    words[7] = hp.tol
    words[8] = hp.epsilon
    words[9] = Float64(hp.max_iter)
    words[10] = Float64(hp.nochange_steps)
    words[11] = hp.cache_mib
    words[12] = 1.0 if hp.probability else 0.0
    words[13] = shape_ovo
    words[14] = break_ties
    words[15] = Float64(low)
    words[16] = Float64(high)
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, connection.table, connection.device, svr_owner)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _array(view: MemberMatrix, typestr: String, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    return svc_prefix(view.data, view.rows * view.cols, shape, typestr, context)


def kernel_svm_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names and layout its predict,
    decision_function, predict_proba and score read). Every public array
    is built before the first field is written."""
    var classifier = plan.kind == 0
    # The clone's `_hp_*` working values, through the estimator's own checks.
    _ = svm_settings_unpack(model, fields, classifier, context)
    var nk = svm_keys()
    var call = MemberCall(plan.route, state)
    var scalars = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)
    var k = Int(words[1])
    var gamma = words[2]
    var machines = Int(words[3])
    var total = Int(words[4])
    var cols = Int(words[5])
    var label0 = words[6]
    var label1 = words[7]
    var probability = words[8] != 0.0
    var table_view = _export(plan, call, _B_MACHINES)
    var table = List[Float64](length=4 * machines, fill=0.0)
    var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=table_view.data)
    for i in range(4 * machines):  # small-loop(the machine table: four words per machine)
        table[i] = source[i]
    var support = _array(_export(plan, call, _B_SUPPORT), "<i4", Python.tuple(PythonObject(total)), context)
    var vectors = _array(_export(plan, call, _B_VECTORS), "<f4", Python.tuple(PythonObject(total), PythonObject(cols)),
        context)
    var dual_rows = k - 1 if k > 2 else 1
    var dual = _array(_export(plan, call, _B_DUAL), "<f4", Python.tuple(PythonObject(dual_rows), PythonObject(total)),
        context)
    var intercept_view = _export(plan, call, MEMBER_BLOCK_INTERCEPT)
    var n_support_view = _export(plan, call, _B_N_SUPPORT)
    var n_iter_view = _export(plan, call, _B_N_ITER)
    var prob_ab = Python.list()
    var prob_a = PythonObject(None)
    var prob_b = PythonObject(None)
    if probability:
        var pairs = k * (k - 1) // 2
        var prob = _export(plan, call, _B_PROB)
        var ab = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=prob.data)
        prob_a = family_array(Python.tuple(PythonObject(pairs)), "<f8", context)
        prob_b = family_array(Python.tuple(PythonObject(pairs)), "<f8", context)
        var pa = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(prob_a)))
        var pb = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(prob_b)))
        for q in range(pairs):  # driver-loop: over the class pairs (their sigmoid parameters)
            _ = py_call(prob_ab.append, Python.tuple(PythonObject(ab[2 * q]), PythonObject(ab[2 * q + 1])))
            pa[q] = ab[2 * q]
            pb[q] = ab[2 * q + 1]
    var first = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=intercept_view.data)[0]
    var iterations = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=n_iter_view.data)[0]
    if classifier and k > 2:
        var pairs_list = Python.list()
        for q in range(machines):  # driver-loop: over the class pairs (one fitted machine each)
            var n_sv = Int(table[4 * q + 3])
            var pair = Python.dict()
            py_setitem(pair, nk[].k_i, PythonObject(Int(table[4 * q])))
            py_setitem(pair, nk[].k_j, PythonObject(Int(table[4 * q + 1])))
            py_setitem(pair, nk[].k_dual, _array(_export(plan, call, _B_MACHINE0 + 3 * q), "<f4",
                Python.tuple(PythonObject(1), PythonObject(n_sv)), context))
            py_setitem(pair, nk[].k_sv, _array(_export(plan, call, _B_MACHINE0 + 3 * q + 1), "<f4",
                Python.tuple(PythonObject(n_sv), PythonObject(cols)), context))
            py_setitem(pair, nk[].k_support, _array(_export(plan, call, _B_MACHINE0 + 3 * q + 2), "<i4",
                Python.tuple(PythonObject(n_sv)), context))
            py_setitem(pair, nk[].k_b, PythonObject(table[4 * q + 2]))
            py_setitem(pair, nk[].k_n_iter, PythonObject(Int(MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=n_iter_view.data)[q])))
            _ = py_call(pairs_list.append, pair)
        var intercept = _array(intercept_view, "<f4", Python.tuple(PythonObject(machines)), context)
        var n_support = _array(n_support_view, "<i4", Python.tuple(PythonObject(k)), context)
        var n_iter = _array(n_iter_view, "<i4", Python.tuple(PythonObject(machines)), context)
        py_setitem(fields, nk[].k__pairs, pairs_list)
        py_setitem(fields, nk[].k_intercept_, intercept)
        py_setitem(fields, nk[].k_n_support_, n_support)
        py_setitem(fields, nk[].k_n_iter_, n_iter)
    else:
        if py_contains(fields, nk[].k__pairs):
            _ = family_call(fields.pop, PythonObject("_pairs"), PythonObject(None))
        py_setitem(fields, nk[].k_intercept_, svc_scalar_array(Float64(first), "<f4", context))
        py_setitem(fields, nk[].k_n_support_, PythonObject(total))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(iterations)))
    # The member was fitted on the caller's own rows, never on the user's array.
    py_setitem(fields, nk[].k_input_copied_, PythonObject(True))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
    py_setitem(fields, nk[].k_support_, support)
    py_setitem(fields, nk[].k_support_vectors_, vectors)
    py_setitem(fields, nk[].k_dual_coef_, dual)
    py_setitem(fields, nk[].k__gamma, PythonObject(gamma))
    if classifier:
        py_setitem(fields, nk[].k__label0, PythonObject(label0))
        py_setitem(fields, nk[].k__label1, PythonObject(label1))
        if probability:
            py_setitem(fields, nk[].k__prob_ab, prob_ab)
            py_setitem(fields, nk[].k_probA_, prob_a)
            py_setitem(fields, nk[].k_probB_, prob_b)
        elif py_contains(fields, nk[].k__prob_ab):
            _ = family_call(fields.pop, PythonObject("_prob_ab"), PythonObject(None))
    py_setitem(fields, "_mojolearn_fitted", True)


def _import(plan: MemberPlan, mut call: MemberCall, block: Int, view: MemberMatrix) raises:
    call.block = block
    call.x = view
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)


def _f32_view(value: PythonObject, ndim: Int, name: String, rows: Int, cols: Int, context: PythonObject,
              pins: PythonObject) raises -> MemberMatrix:
    var held = admit_x(value, ndim, name, context)
    if Int(py=held.size) != rows * cols:
        raise Error("ValueError: mojolearn: the fitted " + name + " has the wrong size")
    _ = py_call(pins.append, held)
    return MemberMatrix(Int(py=classical_address(held)), rows, cols)


def _i32_view(value: PythonObject, name: String, count: Int, context: PythonObject,
              pins: PythonObject) raises -> MemberMatrix:
    var held = _i32_array(value, context)
    if Int(py=held.size) != count:
        raise Error("ValueError: mojolearn: the fitted " + name + " has the wrong size")
    _ = py_call(pins.append, held)
    return MemberMatrix(Int(py=classical_address(held)), 1, count, MEMBER_I32)


def kernel_svm_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var nk = svm_keys()
    if not svc_has(fields, nk[].k_dual_coef_):
        raise Error("NotFittedError: this member is not fitted yet")
    var classifier = plan.kind == 0
    var cols = svc_int(svc_field(fields, nk[].k_n_features_in_, "n_features_in_"))
    var pairs = py_get(fields, nk[].k__pairs, PythonObject(None)) if classifier else PythonObject(None)
    var k = len(svc_field(fields, nk[].k_classes_, "classes_")) if classifier else 0
    var ab = py_get(fields, nk[].k__prob_ab, PythonObject(None)) if classifier else PythonObject(None)
    var probability = ab is not None
    var pins = Python.list()
    var machines = 1 if pairs is None else len(pairs)
    var table = List[Float64](length=4 * machines, fill=0.0)
    var total = 0
    if pairs is None:
        total = svc_int(svc_field(fields, nk[].k_n_support_, "n_support_"))
        table[1] = 1.0 if classifier else 0.0
        table[2] = svc_read_f32(svc_field(fields, nk[].k_intercept_, "intercept_"), 0)
        table[3] = Float64(total)
    else:
        total = Int(py=svc_field(fields, nk[].k_support_, "support_").size)
        for q in range(machines):  # driver-loop: over the class pairs (their table rows)
            var p = pairs[q]
            table[4 * q] = Float64(Int(py=py_item(p, nk[].k_i)))
            table[4 * q + 1] = Float64(Int(py=py_item(p, nk[].k_j)))
            table[4 * q + 2] = svc_float(py_item(p, nk[].k_b))
            table[4 * q + 3] = Float64(Int(py=py_item(p, nk[].k_dual).shape[1]))
    var scalars = List[Float64](length=9, fill=0.0)
    scalars[0] = Float64(plan.kind)
    scalars[1] = Float64(k)
    scalars[2] = svc_float(svc_field(fields, nk[].k__gamma, "_gamma"))
    scalars[3] = Float64(machines)
    scalars[4] = Float64(total)
    scalars[5] = Float64(cols)
    if classifier:
        scalars[6] = svc_float(svc_field(fields, nk[].k__label0, "_label0"))
        scalars[7] = svc_float(svc_field(fields, nk[].k__label1, "_label1"))
    scalars[8] = 1.0 if probability else 0.0
    var prob = List[Float64](length=2 * len(ab) if probability else 0, fill=0.0)
    if probability:
        for q in range(len(ab)):  # driver-loop: over the class pairs (their sigmoid parameters)
            prob[2 * q] = svc_float(ab[q][0])
            prob[2 * q + 1] = svc_float(ab[q][1])
    var call = member_new(plan)
    try:
        _import(plan, call, MEMBER_BLOCK_SCALARS, MemberMatrix(Int(scalars.unsafe_ptr()), 1, 9, MEMBER_F64))
        _import(plan, call, _B_MACHINES, MemberMatrix(Int(table.unsafe_ptr()), machines, 4, MEMBER_F64))
        if probability and len(prob) > 0:
            _import(plan, call, _B_PROB, MemberMatrix(Int(prob.unsafe_ptr()), 1, len(prob), MEMBER_F64))
        if pairs is None:
            if total > 0:
                _import(plan, call, _B_MACHINE0, _f32_view(svc_field(fields, nk[].k_dual_coef_, "dual_coef_"), 2,
                    "dual_coef_", 1, total, context, pins))
                _import(plan, call, _B_MACHINE0 + 1, _f32_view(svc_field(fields, nk[].k_support_vectors_,
                    "support_vectors_"), 2, "support_vectors_", total, cols, context, pins))
                _import(plan, call, _B_MACHINE0 + 2, _i32_view(svc_field(fields, nk[].k_support_, "support_"),
                    "support_", total, context, pins))
        else:
            for q in range(machines):  # driver-loop: over the class pairs (their fitted machines)
                var p = pairs[q]
                var n = Int(table[4 * q + 3])
                if n == 0:
                    continue
                _import(plan, call, _B_MACHINE0 + 3 * q, _f32_view(py_item(p, nk[].k_dual), 2, "a pair's duals", 1, n,
                    context, pins))
                _import(plan, call, _B_MACHINE0 + 3 * q + 1, _f32_view(py_item(p, nk[].k_sv), 2,
                    "a pair's support rows", n, cols, context, pins))
                _import(plan, call, _B_MACHINE0 + 3 * q + 2, _i32_view(py_item(p, nk[].k_support), "a pair's support",
                    n, context, pins))
            if total > 0:
                _import(plan, call, _B_SUPPORT, _i32_view(svc_field(fields, nk[].k_support_, "support_"), "support_",
                    total, context, pins))
                _import(plan, call, _B_VECTORS, _f32_view(svc_field(fields, nk[].k_support_vectors_,
                    "support_vectors_"), 2, "support_vectors_", total, cols, context, pins))
                _import(plan, call, _B_DUAL, _f32_view(svc_field(fields, nk[].k_dual_coef_, "dual_coef_"), 2,
                    "dual_coef_", k - 1, total, context, pins))
            _import(plan, call, MEMBER_BLOCK_INTERCEPT, _f32_view(svc_field(fields, nk[].k_intercept_, "intercept_"),
                1, "intercept_", 1, machines, context, pins))
            _import(plan, call, _B_N_SUPPORT, _i32_view(svc_field(fields, nk[].k_n_support_, "n_support_"),
                "n_support_", k, context, pins))
            _import(plan, call, _B_N_ITER, _i32_view(svc_field(fields, nk[].k_n_iter_, "n_iter_"), "n_iter_",
                machines, context, pins))
        call.block = 0
        call.x = MemberMatrix()
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = pins
    _ = len(scalars)
    _ = len(table)
    _ = len(prob)
    return call.state
