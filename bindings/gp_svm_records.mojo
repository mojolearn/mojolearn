# SPDX-License-Identifier: Apache-2.0
# Per-model admitted-state records of the Gaussian process, kernel SVM,
# OneClassSVM and kernel-method public doors (lane perf2-gp-svm, 2026-11-06).
#
# Each predict-side public call of these estimators re-admitted the whole
# fitted state: a class-ancestry scan for Python fit callbacks, about sixteen
# scalar fields, seven array fields, two kernel-tree snapshots, the class list
# and, for the one-vs-rest classifier, every member's arrays; the decoders
# rebuilt their class tables; the GP owners repacked the kernel. Those results
# depend only on objects the model already holds, so they are kept here:
#
# - Class-state checks (the fit-callback scan, the binary storage class) use
#   the per-class records of bindings/native_class_admission.mojo (CPython
#   type watchers; Python 3.10/3.11 run the full scan every call, as before).
# - Model records live in the prepared-owner registry
#   (bindings/native_prepared_owner.mojo, through trees_side_records), keyed
#   by a weak reference to the model, outside its `__dict__`: predict leaves
#   `__dict__` unchanged, pickles and copies never carry a record, and a copy
#   or an unpickled model rebuilds its record on first use. A refit drops it.
# - A record is valid while its key matches: the identities of every object
#   the skipped admission read (exact containers item by item, each non-scalar
#   with its class). Values are compared by identity (exact ints by value), so
#   no user `__eq__`/`__hash__` runs. A value whose admission can depend on
#   more than its identity and class (a 0-d ndarray scalar field, a torch or
#   container-subclass array field) is admitted again on every call.
#
# Not noticed (as before: the full admissions never read them either): in-place
# writes into a fitted ndarray's elements. Nothing here computes on data or
# changes an arithmetic or fold order.
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules
from bindings.native_class_admission import class_record, class_admit
from bindings.trees_side_records import trees_side_get, trees_side_put, trees_side_drop, trees_key_same, trees_key_shallow
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.classical_attributes import _type_mro
from bindings.classical_pandas import classical_admit_value
from bindings.host_inference_scalars import host_inference_scalar
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.gp_kernel_public_native import gp_kernel_snapshot, gp_admit_kernel
from bindings.gp_common_native import gp_kernel_arrays


comptime _CALLBACK_SLOT = "gpsvm_no_fit_callback_class_v1"
comptime _KERNEL_CLASS_SLOT = "gp_kernel_class_admitted_v1"
comptime _GP_STATE_SLOT = "gp_admitted_state_v1"
comptime _GP_PACK_SLOT = "gp_kernel_pack_v1"


# ------------------------------------------------------------ class state


def gpsvm_no_fit_callbacks(model: PythonObject, fields: PythonObject, message: String) raises:
    """Refuse a model whose class ancestry or instance defines the Python fit
    callback `_mojolearn_after_fit` (`message` is the refusal). The ancestry
    scan runs once per class state; the instance field is read every call."""
    var mods = NativeModules()
    var cls = py_type(model)
    if class_record(mods, _CALLBACK_SLOT, cls) is None:
        var ancestry = _type_mro(cls)
        for i in range(len(ancestry)):  # driver-loop: no completion callback may execute from a native operation
            if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
                raise Error(message)
        # The payload never retains the class (a weak-keyed record).
        class_admit(mods, _CALLBACK_SLOT, cls, PythonObject(True))
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error(message)


# ------------------------------------------------------------ helpers


def _builtin_scalar(mut mods: NativeModules, value: PythonObject) raises -> Bool:
    var b = mods.module("builtins")
    var cls = py_type(value)
    return (cls is b.int or cls is b.float or cls is b.bool or cls is b.str or cls is b.bytes
            or value is None)


def _key_fields(mut mods: NativeModules, key: PythonObject, fields: PythonObject, skip: List[String]) raises:
    """Append an exact field dictionary's entries (names, values, exact
    containers item by item, non-scalars with their class), except the
    call-written `skip` names, then the count of entries appended."""
    var b = mods.module("builtins")
    var items = py_call(b.list, py_call(fields.items))
    var count = 0
    for i in range(len(items)):  # driver-loop: over a fitted model's stored fields (identity only)
        var name = items[i][0]
        if py_type(name) is b.str:
            var text = String(name)
            var skipped = False
            for j in range(len(skip)):  # small-loop(skip: call-written field names)
                if skip[j] == text:
                    skipped = True
                    break
            if skipped:
                continue
        _ = py_call(key.append, name)
        trees_key_shallow(mods, key, items[i][1], 0)
        count += 1
    _ = py_call(key.append, PythonObject(count))


def _context_key(key: PythonObject, context: PythonObject, names: List[String]) raises:
    """The admission context and the connected entries the skipped admission
    read, by identity."""
    _ = py_call(key.append, context)
    for i in range(len(names)):  # small-loop(names: fixed connected context entries)
        _ = py_call(key.append, py_get(context, PythonObject(names[i])))


# ------------------------------------------------------------ GP kernels


def _gp_kernel_nodes(kernel: PythonObject, context: PythonObject, nodes: PythonObject,
                     classes: PythonObject) raises -> Bool:
    """Append every kernel object of `kernel`'s tree to `nodes` and its class
    to `classes` (each once). False when the strict snapshot refuses the
    kernel: the caller then keeps no record."""
    if kernel is None:
        return True
    try:
        var snapshot = gp_kernel_snapshot(kernel, context)
        for i in range(len(snapshot.owners)):  # driver-loop: over the kernel tree's distinct nodes
            var node = snapshot.owners[i]
            var seen = False
            for j in range(len(nodes)):  # driver-loop: over the nodes already recorded (identity)
                if nodes[j] is node:
                    seen = True
                    break
            if not seen:
                _ = py_call(nodes.append, node)
            var cls = py_type(node)
            var known = False
            for j in range(len(classes)):  # driver-loop: over the kernel classes already recorded (identity)
                if classes[j] is cls:
                    known = True
                    break
            if not known:
                _ = py_call(classes.append, cls)
        _ = snapshot^
    except:
        return False
    return True


def _gp_kernel_key(mut mods: NativeModules, key: PythonObject, nodes: PythonObject) raises:
    """Every recorded kernel node, its class and its instance fields. A node
    replaced anywhere in the tree changes its parent's fields, so the
    recorded node list stays complete while the key matches."""
    var none = List[String]()
    _ = py_call(key.append, PythonObject(len(nodes)))
    for i in range(len(nodes)):  # driver-loop: over the recorded kernel nodes (identity only)
        var node = nodes[i]
        _ = py_call(key.append, node)
        _ = py_call(key.append, py_type(node))
        _key_fields(mods, key, _fields(mods, node), none)


def _gp_kernel_classes_admitted(mut mods: NativeModules, classes: PythonObject) raises -> Bool:
    for i in range(len(classes)):  # driver-loop: over the recorded kernel classes
        if class_record(mods, _KERNEL_CLASS_SLOT, classes[i]) is None:
            return False
    return True


def _gp_kernel_classes_admit(mut mods: NativeModules, classes: PythonObject) raises:
    # Called after the strict kernel snapshot passed for every class listed.
    for i in range(len(classes)):  # driver-loop: over the admitted kernel classes
        class_admit(mods, _KERNEL_CLASS_SLOT, classes[i], PythonObject(True))


def gp_kernel_arrays_reused(owner: PythonObject, kernel: PythonObject, context: PythonObject) raises -> PythonObject:
    """`gp_kernel_arrays(kernel)` (the read-only packed kernel the GP bindings
    take), packed once per kernel state of `owner` and reused while every node
    of the tree and its fields are the same objects."""
    var mods = NativeModules()
    var held = trees_side_get(_GP_PACK_SLOT, owner)
    if held is not None and held[0] is kernel:
        var key = Python.list()
        _context_key(key, context, _gp_pack_context())
        _gp_kernel_key(mods, key, held[1])
        if trees_key_same(held[2], key):
            return held[3]
    var arrays = gp_kernel_arrays(kernel, context)
    var nodes = Python.list()
    var classes = Python.list()
    if _gp_kernel_nodes(kernel, context, nodes, classes):
        var key = Python.list()
        _context_key(key, context, _gp_pack_context())
        _gp_kernel_key(mods, key, nodes)
        _ = trees_side_put(_GP_PACK_SLOT, owner, Python.tuple(kernel, nodes, key, arrays))
    return arrays


# ------------------------------------------------------------ GP fitted state

# The GP fitted-state schema `_gp_admit_state` admits on a non-fit call
# (providers/python/gp_public_lifecycle_pyapi.mojo); the residual admissions
# below must name the same fields.
def _gp_state_context() -> List[String]:
    var names = List[String]()
    for name in ["family_gp_binary_laplace", "family_gp_binary_namespace", "family_gp_kernel_nodes",
                 "family_gp_kernel_bounds", "scalar_types", "native_value_types", "numeric_dtypes",
                 "ndarray", "array_class", "mapping_types", "buffer_owner_types"]:  # small-loop(fixed connected context entry names)
        names.append(String(name))
    return names^


def _gp_pack_context() -> List[String]:
    var names = List[String]()
    names.append("family_gp_kernel_nodes")
    return names^


def _gp_skip() -> List[String]:
    # Written by predict(return_std=True) itself; never admitted.
    var skip = List[String]()
    skip.append("clamped_")
    skip.append("n_clamped_")
    return skip^


def _gp_state_key(mut mods: NativeModules, fields: PythonObject, nodes: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    var key = Python.list()
    _context_key(key, context, _gp_state_context())
    _key_fields(mods, key, fields, _gp_skip())
    var fits = py_get(fields, "estimators_")
    if fits is not None:
        var b = mods.module("builtins")
        if py_type(fits) is b.list or py_type(fits) is b.tuple:
            var none = List[String]()
            for i in range(len(fits)):  # driver-loop: retained one-vs-rest binary fit owners (identity only)
                var member = fits[i]
                _ = py_call(key.append, member)
                _ = py_call(key.append, py_type(member))
                if py_type(member) is context["family_gp_binary_laplace"]:
                    _key_fields(mods, key, _fields(mods, member), none)
    _gp_kernel_key(mods, key, nodes)
    return key


def _gp_residual_scalars(fields: PythonObject, names: PythonObject, context: PythonObject) raises:
    # host_inference_scalar admits builtin and NumPy scalars by type; a 0-d
    # ndarray also by its size and rank, which an in-place reshape can change.
    var array_type = py_get(context, "ndarray")
    if array_type is None:
        return
    for i in range(len(names)):  # driver-loop: fixed scalar-state field schema
        var value = py_get(fields, names[i])
        if value is not None and py_type(value) is array_type:
            host_inference_scalar(value, context, True)


def _gp_residual_arrays(fields: PythonObject, names: PythonObject, context: PythonObject) raises:
    # An exact ndarray or Array is admitted by its type alone; anything else
    # (torch, memoryview, containers) is admitted again.
    var array_type = py_get(context, "ndarray")
    var array_class = context["array_class"]
    for i in range(len(names)):  # driver-loop: fixed retained-buffer schema
        var value = py_get(fields, names[i])
        if value is None:
            continue
        var cls = py_type(value)
        if cls is array_class or (array_type is not None and cls is array_type):
            continue
        classical_admit_value(value, context)


def _gp_classes_settled(mut mods: NativeModules, classes: PythonObject, context: PythonObject) raises -> Bool:
    """`classes_` is an exact list or tuple of exact builtin scalars: its
    admission depends on nothing the key does not hold."""
    var b = mods.module("builtins")
    if py_type(classes) is not b.list and py_type(classes) is not b.tuple:
        return False
    for i in range(len(classes)):  # driver-loop: over the fitted classes
        if not _builtin_scalar(mods, classes[i]):
            return False
    return True


def gp_state_forget(model: PythonObject) raises:
    """A fit replaces the fitted state: release the record now."""
    trees_side_drop(_GP_STATE_SLOT, model)
    trees_side_drop(_GP_PACK_SLOT, model)


def gp_state_admitted(model: PythonObject, fields: PythonObject, scalar_names: PythonObject,
                      array_names: PythonObject, member_scalars: PythonObject, member_arrays: PythonObject,
                      context: PythonObject) raises -> Bool:
    """True when `model`'s fitted state is the state a full non-fit admission
    accepted, after the residual admissions passed again; the caller then
    skips the full admission. False: run it (and `gp_state_remember`)."""
    var mods = NativeModules()
    var held = trees_side_get(_GP_STATE_SLOT, model)
    if held is None:
        return False
    var key = _gp_state_key(mods, fields, held[1], context)
    if not trees_key_same(held[0], key):
        return False
    if not _gp_kernel_classes_admitted(mods, held[2]):
        # A kernel class changed (or no type watchers): its snapshot again.
        for name in ["_kernel", "kernel_"]:  # small-loop(fixed kernel state fields)
            var value = py_get(fields, name)
            if value is not None:
                gp_admit_kernel(value, context)
    _gp_residual_scalars(fields, scalar_names, context)
    _gp_residual_arrays(fields, array_names, context)
    var classes = py_get(fields, "classes_")
    if classes is not None and not _gp_classes_settled(mods, classes, context):
        classical_admit_value(classifier_label_list(classes, context), context)
    var fits = py_get(fields, "estimators_")
    if fits is not None:
        for i in range(len(fits)):  # driver-loop: retained one-vs-rest binary fit owners
            var own = _fields(mods, fits[i])
            _gp_residual_arrays(own, member_arrays, context)
            _gp_residual_scalars(own, member_scalars, context)
    return True


def gp_state_remember(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    """Record the state a full non-fit admission just accepted."""
    var mods = NativeModules()
    var nodes = Python.list()
    var classes = Python.list()
    for name in ["_kernel", "kernel_"]:  # small-loop(fixed kernel state fields)
        if not _gp_kernel_nodes(py_get(fields, name), context, nodes, classes):
            return
    _gp_kernel_classes_admit(mods, classes)
    var key = _gp_state_key(mods, fields, nodes, context)
    _ = trees_side_put(_GP_STATE_SLOT, model, Python.tuple(key, nodes, classes))


# ------------------------------------------------------------ class tables


def gpsvm_class_table(owner: PythonObject, slot: String, labels: PythonObject) raises -> PythonObject:
    """The class table recorded for `owner` under `slot` when `labels` (the
    native label list of `classes_`) holds the same label objects; else
    None."""
    var held = trees_side_get(slot, owner)
    if held is None:
        return PythonObject(None)
    var key = Python.list()
    _ = py_call(key.append, PythonObject(len(labels)))
    for i in range(len(labels)):  # driver-loop: over the fitted classes (identity only)
        _ = py_call(key.append, labels[i])
    if not trees_key_same(held[0], key):
        return PythonObject(None)
    return held[1]


def gpsvm_class_table_put(owner: PythonObject, slot: String, labels: PythonObject, table: PythonObject) raises:
    var key = Python.list()
    _ = py_call(key.append, PythonObject(len(labels)))
    for i in range(len(labels)):  # driver-loop: over the fitted classes (identity only)
        _ = py_call(key.append, labels[i])
    _ = trees_side_put(slot, owner, Python.tuple(key, table))


# ------------------------------------------------------------ OneClassSVM


comptime _OCSVM_COEF_SLOT = "ocsvm_coef_snapshot_v1"


def ocsvm_coef_reused(owner: PythonObject, dual: PythonObject) raises -> Bool:
    """`dual` is the coefficient object whose float32 snapshot was the object
    itself last call (no conversion): the snapshot can be skipped."""
    var held = trees_side_get(_OCSVM_COEF_SLOT, owner)
    return held is not None and held[0] is dual and held[1] is py_type(dual)


def ocsvm_coef_remember(owner: PythonObject, dual: PythonObject, coef: PythonObject) raises:
    if coef is dual:
        _ = trees_side_put(_OCSVM_COEF_SLOT, owner, Python.tuple(dual, py_type(dual)))
    else:
        trees_side_drop(_OCSVM_COEF_SLOT, owner)
