# SPDX-License-Identifier: Apache-2.0
# Gaussian process admitted-state records (lane perf2-gp-svm, 2026-11-06):
# the fitted-state record of the GP public door, the per-model packed kernel
# and the admitted-X registry. The validity rules are those of
# bindings/gp_svm_records.mojo (identity keys in the weak-keyed prepared-owner
# registry, class-state records through CPython type watchers).
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules, native_loaded_module
from bindings.native_class_admission import class_record, class_admit
from bindings.trees_side_records import trees_key_same
from bindings.gp_svm_records import gpsvm_side_get, gpsvm_side_put, gpsvm_side_drop
from bindings.native_model_records import (
    records_key_append, records_stamped_get, records_stamped_put, records_stampable, records_drop,
    ContentRecord, content_fields, content_object, RECORDS_GP_SVM,
)
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.mode_metadata_native import _fields
from bindings.classical_pandas import classical_admit_value
from bindings.host_inference_scalars import host_inference_scalar
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.gp_kernel_public_native import gp_kernel_snapshot, gp_admit_kernel
from bindings.gp_kernel_cache_boundary import gp_packed_kernel_reused
from bindings.gp_svm_records import gpsvm_builtin_scalar, gpsvm_key_fields, gpsvm_context_key
from core.shared_slot import SharedSlot


comptime _KERNEL_CLASS_SLOT = "gp_kernel_class_admitted_v1"
comptime _GP_STATE_SLOT = "gp_admitted_state_v1"
comptime _GP_PACK_SLOT = "gp_kernel_pack_v1"


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
    records_key_append(key, PythonObject(len(nodes)))
    for i in range(len(nodes)):  # driver-loop: over the recorded kernel nodes (identity only)
        var node = nodes[i]
        records_key_append(key, node)
        records_key_append(key, py_type(node))
        gpsvm_key_fields(mods, key, _fields(mods, node), none)


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
    return gp_packed_kernel_reused(owner, kernel, context)


comptime _GP_STATE_STAMP = "gp_fitted_state_stamp_v1"


def _gp_state_stamp(model: PythonObject, fields: PythonObject, held: PythonObject, context: PythonObject) raises:
    """Stamp the fitted-state record `held` with the model's generation and
    the native content words of what `_gp_state_key` reads: the model's field
    contents, the Laplace binary members and every kernel node. Only when
    `fields` is the model's own instance dictionary (the record holds it)."""
    if not (py_fields(model) is fields) or not records_stampable(model):
        return
    var content = ContentRecord()
    content_fields(content, fields, 4)
    var fits = py_get(fields, "estimators_")
    if fits is not None:
        var b = native_loaded_module("builtins")
        if py_type(fits) is b.list or py_type(fits) is b.tuple:
            for i in range(len(fits)):  # driver-loop: retained one-vs-rest binary fit owners (content words)
                if py_type(fits[i]) is context["family_gp_binary_laplace"]:
                    content_object(content, fits[i], 4)
    var nodes = held[1]
    for i in range(len(nodes)):  # driver-loop: over the recorded kernel nodes (content words, once per record)
        content_object(content, nodes[i], 4)
    var context_key = Python.list()
    gpsvm_context_key(context_key, context, _gp_state_context())
    _ = records_stamped_put(RECORDS_GP_SVM, model, _GP_STATE_STAMP, content^, Python.tuple(fields, held, context_key))


# ------------------------------------------------------------ GP fitted state

# The GP fitted-state schema `_gp_admit_state` admits on a non-fit call
# (providers/python/gp_public_lifecycle_pyapi.mojo); the residual admissions
# below must name the same fields.
def _gp_state_context() -> List[String]:
    var names = List[String]()
    names.append("family_gp_binary_laplace")
    names.append("family_gp_binary_namespace")
    names.append("family_gp_kernel_nodes")
    names.append("family_gp_kernel_bounds")
    names.append("scalar_types")
    names.append("native_value_types")
    names.append("numeric_dtypes")
    names.append("ndarray")
    names.append("array_class")
    names.append("mapping_types")
    names.append("buffer_owner_types")
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
    gpsvm_context_key(key, context, _gp_state_context())
    gpsvm_key_fields(mods, key, fields, _gp_skip())
    var fits = py_get(fields, "estimators_")
    if fits is not None:
        var b = mods.module("builtins")
        if py_type(fits) is b.list or py_type(fits) is b.tuple:
            var none = List[String]()
            for i in range(len(fits)):  # driver-loop: retained one-vs-rest binary fit owners (identity only)
                var member = fits[i]
                records_key_append(key, member)
                records_key_append(key, py_type(member))
                if py_type(member) is context["family_gp_binary_laplace"]:
                    gpsvm_key_fields(mods, key, _fields(mods, member), none)
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
        if not gpsvm_builtin_scalar(mods, classes[i]):
            return False
    return True


def gp_state_forget(model: PythonObject) raises:
    """A fit replaces the fitted state: release the record now."""
    gpsvm_side_drop(_GP_STATE_SLOT, model)
    gpsvm_side_drop(_GP_PACK_SLOT, model)
    records_drop(RECORDS_GP_SVM, model, _GP_STATE_STAMP)


def gp_state_admitted(model: PythonObject, fields: PythonObject, scalar_names: PythonObject,
                      array_names: PythonObject, member_scalars: PythonObject, member_arrays: PythonObject,
                      context: PythonObject) raises -> Bool:
    """True when `model`'s fitted state is the state a full non-fit admission
    accepted, after the residual admissions passed again; the caller then
    skips the full admission. False: run it (and `gp_state_remember`)."""
    var mods = NativeModules()
    var held = gpsvm_side_get(_GP_STATE_SLOT, model)
    if held is None:
        return False
    # Generation fast path (lane ne1-e-records, row N320): one word for the
    # model plus native member and node words instead of the key list.
    var stamped = records_stamped_get(RECORDS_GP_SVM, model, _GP_STATE_STAMP)
    var same = False
    if stamped is not None and stamped[0] is fields and stamped[1] is held:
        var context_key = Python.list()
        gpsvm_context_key(context_key, context, _gp_state_context())
        same = trees_key_same(stamped[2], context_key)
    if not same:
        var key = _gp_state_key(mods, fields, held[1], context)
        if not trees_key_same(held[0], key):
            return False
        _gp_state_stamp(model, fields, held, context)
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
    var record = Python.tuple(key, nodes, classes)
    if gpsvm_side_put(_GP_STATE_SLOT, model, record):
        _gp_state_stamp(model, fields, record, context)


# ------------------------------------------------------------ admitted X
#
# The GP public door admits X with classical_check_x (C-order float32, the
# scikit-learn refusals) and the numerical owner converted the same X again
# (`admit_family`, a second full metadata snapshot). The door registers its
# admitted X here for the duration of its operation; the owner skips its
# conversion only for a registered exact ndarray or Array. Any other caller's
# X is converted as before. Entries are strong references (no address reuse
# while registered); the door releases its own entry when the operation ends,
# and a concurrent release only causes the conversion to run.


struct GpAdmittedInputs(Defaultable, Movable):
    var items: List[PythonObject]
    # Lane a7a: True where the door left the finite test to the kernel
    # entry (admit_x `defer_finite`: a borrowed float32 C-order X the door
    # did not scan); the owner refuses through the entry's status word, or
    # scans it itself on a route whose entry returns none.
    var deferred: List[Bool]

    def __init__(out self):
        self.items = List[PythonObject]()
        self.deferred = List[Bool]()


comptime GP_ADMITTED_X = SharedSlot[GpAdmittedInputs, "MojolearnGpAdmittedInputsV2"]


def gp_x_mark(value: PythonObject, deferred: Bool = False) raises:
    var g = GP_ADMITTED_X.get_or_create_ptr()
    g[].items.append(value)
    g[].deferred.append(deferred)


def gp_x_deferred(value: PythonObject) raises -> Bool:
    """`value` is a registered X whose finite test the door deferred to the
    kernel entry (lane a7a)."""
    var g = GP_ADMITTED_X.get_or_create_ptr()
    for i in range(len(g[].items)):  # small-loop(items: the running doors' admitted inputs, usually one)
        if g[].items[i] is value:
            return g[].deferred[i]
    return False


def gp_x_known(value: PythonObject, context: PythonObject) raises -> Bool:
    """`value` is the C-order float32 X the running public door admitted."""
    var cls = py_type(value)
    if cls is not context["array_class"] and cls is not py_get(context, "ndarray"):
        return False
    var g = GP_ADMITTED_X.get_or_create_ptr()
    for i in range(len(g[].items)):  # small-loop(items: the running doors' admitted inputs, usually one)
        if g[].items[i] is value:
            return True
    return False


def gp_x_release(value: PythonObject) raises:
    var g = GP_ADMITTED_X.get_or_create_ptr()
    var found = -1
    for i in range(len(g[].items)):  # small-loop(items: the running doors' admitted inputs, usually one)
        if g[].items[i] is value:
            found = i
    if found < 0:
        return
    # Hold a reference while the registry changes, so no destructor runs
    # inside the update.
    var held = g[].items.pop(found)
    _ = g[].deferred.pop(found)
    _ = held^
