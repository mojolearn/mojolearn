# SPDX-License-Identifier: Apache-2.0
# Whole TreeExplainer operations. Existing xtrees SHAP/spread kernels own all
# numerical work; this owner admits inputs, captures their metadata/storage,
# prepares the flat forest and owns cache release without Python finalizers.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_attr import py_attr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields, _mode
from bindings.native_instance_item import native_instance_item
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_attributes import classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.trees_public_state import _release_tree_values
from bindings.trees_public_inputs import trees_table, trees_call, trees_view, trees_check_rank, trees_cast, trees_empty, trees_vector, trees_full_f32, trees_address, trees_reshape
from providers.python.array_metadata_pyapi import array_owner_once
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry

comptime _I32 = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _U8 = MutPointer[UInt8, MutUntrackedOrigin]


def _get(fields: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    return native_instance_item(fields, PythonObject(name)) if py_contains(fields, PythonObject(name)) else default


def _get(fields: PythonObject, name: String) raises -> PythonObject:
    return _get(fields, name, PythonObject(None))


def _int(value: PythonObject, name: String, positive: Bool = False) raises -> Int:
    if py_type(value) is not native_loaded_module("builtins").int:
        raise Error("TypeError: TreeExplainer " + name + " must be an exact integer")
    var result = Int(py=value)
    if positive and result < 1:
        raise Error("ValueError: TreeExplainer " + name + " must be positive")
    return result


def _entry(table: PythonObject, name: String, required: Bool = True) raises -> PythonObject:
    var entry = py_get(table, name)
    if entry is None and not required:
        return entry
    if entry is None or py_type(entry) is not py_type(native_loaded_module("builtins").len):
        raise Error("TypeError: TreeExplainer requires compiled entry " + name)
    if neural_descriptor_entry(entry.__self__, name) is not entry:
        raise Error("TypeError: TreeExplainer compiled entry changed after connection")
    return entry


def _kind(model: PythonObject, context: PythonObject) raises -> String:
    var cls = py_type(model)
    # The CPU host forest registers itself when its module loads (CPU installs).
    var host = py_get(context, "trees_explainer_host_forest")
    if host is not None and cls is host:
        return "rf_regressor"  # the archive keeps its actual output count
    # Compare identities before dictionary lookup; custom metaclasses must not
    # gain a hash/equality callback while selecting the native model layout.
    for group_index in range(2):  # driver-loop: two connected class tables
        var group = "trees_classes" if group_index == 0 else "trees_boost_classes"
        var table = context[group]
        var keys = py_call(native_loaded_module("builtins").list, table)
        for i in range(len(keys)):  # driver-loop: connected exact tree classes
            if cls is keys[i]:
                var kind = String(table[keys[i]])
                if kind == "rf_classifier" or kind == "rf_regressor" or kind == "et_classifier" or kind == "et_regressor" or kind == "dt_classifier" or kind == "dt_regressor" or kind == "dart_classifier" or kind == "dart_regressor":
                    return kind
    raise Error("NotImplementedError: TreeExplainer requires a fitted built-in flat forest, decision tree or DART model")


def _capture(model: PythonObject, context: PythonObject, captures: PythonObject) raises -> PythonObject:
    var identity = PythonObject(Int(model._obj_ptr))
    if not py_contains(captures, identity):
        _ = _kind(model, context)
        var fields = _fields(model)
        py_setitem(captures, identity, Python.tuple(model, py_call(py_attr(fields, "copy"))))
    return captures[identity][1]


def _snapshot(source: PythonObject, dtype: String, rank: Int, name: String,
              context: PythonObject, snapshots: PythonObject) raises -> PythonObject:
    var identity = PythonObject(Int(source._obj_ptr))
    var raw = PythonObject(None)
    if py_contains(snapshots, identity):
        raw = snapshots[identity][1]
        if Int(py=raw.ndim) != rank:
            raise Error("ValueError: TreeExplainer buffer has the wrong rank")
    else:
        classical_admit_value(source, context)
        var owners = array_owner_once(source, context)
        var view = trees_view(source, name, context)
        trees_check_rank(view, rank, name)
        var shape = Python.tuple(view.shape[0]) if rank == 1 else Python.tuple(view.shape[0], view.shape[1])
        # Copy captured strides before any extension call releases the GIL.
        # Each distinct source owner is admitted/captured once per operation.
        raw = trees_empty(shape, view.typestr, context)
        var dst = _U8(unsafe_from_address=trees_address(raw))
        var cols = view.shape[1] if rank == 2 else 1
        for row in range(view.shape[0]):  # driver-loop: admitted buffer layout, no numerical algorithm
            for col in range(cols):  # driver-loop: physical input strides
                var offset = row * view.strides[0] + (col * view.strides[1] if rank == 2 else 0)
                memcpy(dest=dst + (row * cols + col) * view.item,
                       src=_U8(unsafe_from_address=view.addr + offset), count=view.item)
        py_setitem(snapshots, identity, Python.tuple(source, raw, owners))
        _ = owners
        _ = view
    var source_type = String(raw.dtype)
    if source_type == dtype:
        return raw
    var out = trees_empty(raw.shape, dtype, context)
    _ = _entry(context["trees_helpers"], "cast_elements")
    var count = Int(py=raw.shape[0]) * (Int(py=raw.shape[1]) if rank == 2 else 1)
    trees_cast(trees_address(raw), source_type, trees_address(out), dtype, count, context)
    _ = raw
    return out


def _join(parts: PythonObject, count: Int, dtype: String, context: PythonObject) raises -> PythonObject:
    var out = trees_vector(count, dtype, context)
    var dst = _U8(unsafe_from_address=trees_address(out))
    var at = 0
    for i in range(len(parts)):  # driver-loop: flatten captured tree buffers in boosting order
        var part = parts[i]
        var n = Int(py=part.shape[0])
        memcpy(dest=dst + at * 4, src=_U8(unsafe_from_address=trees_address(part)), count=n * 4)
        at += n
        _ = part
    if at != count:
        raise Error("ValueError: inconsistent DART tree storage")
    return out


def _flat(model: PythonObject, fields: PythonObject, context: PythonObject,
          captures: PythonObject, table: PythonObject, snapshots: PythonObject) raises -> PythonObject:
    var kind = _kind(model, context)
    var arrays = Python.list()
    var k = 1
    var scales = PythonObject(None)
    var initial = PythonObject(None)
    if not kind.startswith("dart_"):
        k = _int(_get(fields, "_num_outputs", PythonObject(1)), "output count", True)
        var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
        for i in range(5):  # driver-loop: flat forest representation fields
            var value = _get(fields, String(names[i]))
            if value is None:
                raise Error("RuntimeError: the model is not fitted yet")
            _ = py_call(arrays.append, _snapshot(value, "<i4" if i == 0 or i == 1 or i == 3 else "<f4", 1, "forest", context, snapshots))
        var count = _int(_get(fields, "_n_trees"), "tree count", True)
        if Int(py=arrays[0].shape[0]) != count + 1:
            raise Error("ValueError: inconsistent forest tree count")
        scales = trees_full_f32(count, 1.0 / Float64(count), context)
        initial = trees_full_f32(k, 0.0, context)
    else:
        var trees = _get(fields, "trees_")
        var values = _get(fields, "tree_values_")
        var b = native_loaded_module("builtins")
        if trees is None:
            raise Error("RuntimeError: the model is not fitted yet")
        if (py_type(trees) is not b.list and py_type(trees) is not b.tuple) or (py_type(values) is not b.list and py_type(values) is not b.tuple):
            raise Error("TypeError: DART fitted trees/values require ordinary list or tuple storage")
        # Freeze these sequences before native calls permit application mutation.
        trees = py_call(b.tuple, trees)
        values = py_call(b.tuple, values)
        var count = len(trees)
        if count < 1 or len(values) != count:
            raise Error("ValueError: inconsistent DART tree values")
        k = _int(_get(fields, "n_classes_", PythonObject(1)), "output count", True)
        var init = _get(fields, "init_score_")
        if k == 1:
            classical_admit_value(init, context)
            initial = _snapshot(Python.list(init), "<f4", 1, "init_score_", context, snapshots)
        else:
            initial = _snapshot(init, "<f4", 1, "init_score_", context, snapshots)
        if Int(py=initial.shape[0]) != k:
            raise Error("ValueError: DART initial score width differs from output count")
        scales = _snapshot(_get(fields, "tree_coefs_"), "<f4", 1, "tree coefficients", context, snapshots)
        if Int(py=scales.shape[0]) != count:
            raise Error("ValueError: DART tree coefficient count differs from tree count")
        var offsets = trees_vector(count + 1, "<i4", context)
        var off = _I32(unsafe_from_address=trees_address(offsets))
        var columns = Python.list()
        var questions = Python.list()
        var children = Python.list()
        var leaves = Python.list()
        var total = 0
        for i in range(count):  # driver-loop: capture each distinct fitted tree once
            var state = _capture(trees[i], context, captures)
            var tree_offsets = _snapshot(_get(state, "_offsets"), "<i4", 1, "tree offsets", context, snapshots)
            if Int(py=tree_offsets.shape[0]) != 2:
                raise Error("ValueError: a DART tree must hold one tree")
            var tp = _I32(unsafe_from_address=trees_address(tree_offsets))
            var size = Int(tp[1]) - Int(tp[0])
            if tp[0] != 0 or size < 1 or total > 2147483647 - size:
                raise Error("ValueError: invalid DART tree offsets")
            var col = _snapshot(_get(state, "_colid"), "<i4", 1, "tree columns", context, snapshots)
            var q = _snapshot(_get(state, "_quesval"), "<f4", 1, "tree thresholds", context, snapshots)
            var lc = _snapshot(_get(state, "_left_child"), "<i4", 1, "tree children", context, snapshots)
            var val = _snapshot(values[i], "<f4", 1, "tree values", context, snapshots)
            if Int(py=col.shape[0]) != size or Int(py=q.shape[0]) != size or Int(py=lc.shape[0]) != size or Int(py=val.shape[0]) != size:
                raise Error("ValueError: DART node buffers differ from tree offsets")
            _ = py_call(columns.append, col)
            _ = py_call(questions.append, q)
            _ = py_call(children.append, lc)
            _ = py_call(leaves.append, val)
            total += size
            off[i + 1] = Int32(total)
            _ = tree_offsets
        var vals = _join(leaves, total, "<f4", context)
        var expanded = vals
        if k > 1:
            expanded = trees_vector(total * k, "<f4", context)
            _ = py_call(_entry(table, "x_trees_spread_leaves"), trees_address(vals), trees_address(offsets), trees_address(expanded), Python.list(total, count, k))
        arrays = Python.list(offsets, _join(columns, total, "<i4", context), _join(questions, total, "<f4", context), _join(children, total, "<i4", context), expanded)
    return Python.tuple(py_call(native_loaded_module("builtins").tuple, arrays), scales, initial)


struct TreeShapOwner(Movable, Writable):
    var state: PythonObject
    var release: PythonObject
    var handle: Int
    var closed: Bool

    def __init__(out self, state: PythonObject, release: PythonObject, handle: Int):
        self.state = state
        self.release = release
        self.handle = handle
        self.closed = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_TreeShapOwner")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def close(mut self) raises:
        var handle = self.handle
        self.handle = 0
        self.closed = True
        if handle > 0:
            _ = py_call(self.release, handle)

    def __deinit__(deinit self):
        if self.handle > 0:
            try:
                _ = py_call(self.release, self.handle)
            except:
                pass


def _admit_owner(owner: PythonObject) raises:
    var prototype = PythonObject(alloc=TreeShapOwner(PythonObject(None), PythonObject(None), 0))
    if py_type(owner) is not py_type(prototype):
        raise Error("TypeError: TreeExplainer requires its native cache owner")
    if py_contains(_class_fields(py_type(owner)), "__del__"):
        raise Error("TypeError: TreeExplainer refuses injected Python finalizers")


def _replace(fields: PythonObject, changes: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var names = py_call(b.list, changes)
    var todo = Python.list()
    for i in range(len(names)):  # driver-loop: admit displaced owners before any state mutation
        var old = _get(fields, String(names[i]))
        if old is not changes[names[i]]:
            _ = py_call(todo.append, old)
    var seen = Python.dict()
    var at = 0
    while at < len(todo):  # driver-loop: cycle-safe owner admission without releasing Python callbacks
        var value = todo[at]
        at += 1
        var identity = PythonObject(Int(value._obj_ptr))
        if py_contains(seen, identity):
            continue
        py_setitem(seen, identity, value)
        var kept = False
        for i in range(len(names)):  # driver-loop: unchanged identities survive dictionary replacement
            kept = kept or value is changes[names[i]]
        if kept:
            continue
        var prototype = PythonObject(alloc=TreeShapOwner(PythonObject(None), PythonObject(None), 0))
        if py_type(value) is py_type(prototype):
            native_release_weakrefs(value, context)
            _admit_owner(value)
            var previous_owner = value.downcast_value_ptr[TreeShapOwner]()
            if previous_owner[].state is not None:
                native_release_admit(previous_owner[].state[5], context)
        elif py_type(value) is b.list or py_type(value) is b.tuple:
            native_release_weakrefs(value, context)
            for i in range(len(value)):
                _ = py_call(todo.append, value[i])
        else:
            var host = py_get(context, "trees_explainer_host_forest")
            var tree = host is not None and py_type(value) is host
            for group_index in range(2):  # driver-loop: connected class tables
                var group = "trees_classes" if group_index == 0 else "trees_boost_classes"
                var keys = py_call(b.list, context[group])
                for i in range(len(keys)):
                    tree = tree or py_type(value) is keys[i]
            if tree:
                native_release_weakrefs(value, context)
                var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(value))
                for i in range(len(mro)):
                    if py_contains(_class_fields(mro[i]), "__del__"):
                        raise Error("TypeError: TreeExplainer reinitialization cannot release Python model finalizers")
                var previous = _fields(value)
                var values = py_call(b.list, previous.values())
                for i in range(len(values)):
                    _ = py_call(todo.append, values[i])
            else:
                _release_tree_values(Python.tuple(value), py_call(b.list, changes.values()), context)
    _ = py_call(fields.update, changes)


def _construct(fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var model = args[0]
    var data = args[1]
    var perturb = args[2]
    var output = args[3]
    if py_type(perturb) is not b.str or py_type(output) is not b.str:
        raise Error("TypeError: TreeExplainer settings must be exact strings")
    if String(perturb) != "tree_path_dependent" and String(perturb) != "auto":
        raise Error("NotImplementedError: TreeExplainer supports the path-dependent algorithm only")
    if String(output) != "raw":
        raise Error("NotImplementedError: TreeExplainer supports raw model output only")
    if data is None:
        raise Error("NotImplementedError: TreeExplainer requires background data for node covers")
    var captures = Python.dict()
    var snapshots = Python.dict()
    var state = _capture(model, context, captures)
    var requested = args[4] if args[4] is not None else _get(state, "numeric_mode")
    var mode = _mode(requested if requested is not None else context["backend_fields"]["_DEFAULT_MODE"])
    var captured_mode = _get(state, "_fit_numeric_mode")
    if captured_mode is not None and _mode(captured_mode) != mode:
        raise Error("ValueError: TreeExplainer numeric_mode differs from the fitted model")
    var table = trees_table(context, mode, "_mojolearn_x_trees")
    var bg = _snapshot(data, "<f4", 2, "data", context, snapshots)
    var nb = Int(py=bg.shape[0])
    var d = Int(py=bg.shape[1])
    var flat = _flat(model, state, context, captures, table, snapshots)
    var forest = flat[0]
    var scales = flat[1]
    var ev = flat[2]
    var k = Int(py=ev.shape[0])
    var nt = Int(py=forest[0].shape[0]) - 1
    var offsets = _I32(unsafe_from_address=trees_address(forest[0]))
    if nt < 1 or offsets[0] != 0:
        raise Error("ValueError: TreeExplainer: the model holds no trees")
    var nn = Int(offsets[nt])
    for i in range(nt):  # driver-loop: forest storage extent admission before kernels dereference it
        if offsets[i + 1] <= offsets[i]:
            raise Error("ValueError: TreeExplainer requires increasing tree offsets")
    if nn < 1 or Int(py=forest[1].shape[0]) != nn or Int(py=forest[2].shape[0]) != nn or Int(py=forest[3].shape[0]) != nn or Int(py=forest[4].shape[0]) != nn * k:
        raise Error("ValueError: TreeExplainer node buffers differ from forest offsets")
    var cover = trees_vector(nn, "<i4", context)
    var meta = trees_vector(3, "<i4", context)
    var addresses = Python.list()
    # Hold backing storage independently; public Array slots can change.
    # The private owner must not retain mutable exposed Array metadata.
    var keep = Python.list()
    for i in range(5):
        _ = py_call(addresses.append, trees_address(forest[i]))
        _ = py_call(keep.append, array_owner_once(forest[i], context))
    _ = py_call(keep.append, array_owner_once(scales, context))
    _ = py_call(keep.append, array_owner_once(cover, context))
    var prep = Python.list(addresses, trees_address(scales), trees_address(bg), trees_address(cover), trees_address(ev), trees_address(meta), Python.list(nb, d, nt, k, nn))
    var retain = _entry(table, "x_trees_tree_shap_prepare_retain", False)
    var release = _entry(table, "x_trees_tree_shap_cache_release", False)
    var cached = _entry(table, "x_trees_tree_shap_cache_values", False)
    var plain = _entry(table, "x_trees_tree_shap")
    var owner = PythonObject(alloc=TreeShapOwner(PythonObject(None), release, 0))
    var held = owner.downcast_value_ptr[TreeShapOwner]()
    if retain is not None:
        if release is None or cached is None:
            raise Error("RuntimeError: TreeExplainer retained cache entries are incomplete")
        held[].handle = _int(trees_call(retain, prep), "cache handle", True)
    else:
        _ = trees_call(_entry(table, "x_trees_tree_shap_prepare"), prep)
    var mp = _I32(unsafe_from_address=trees_address(meta))
    var slots = Int(mp[0])
    var depth = Int(mp[1])
    var width = 8
    while width < min(slots, depth) + 1:
        width *= 2
    if width > 256:
        raise Error("NotImplementedError: TreeExplainer supports paths with at most 255 distinct split features")
    var enabled = _entry(table, "x_trees_tree_shap_cache_enabled", False)
    if held[].handle == 0 and enabled is not None:
        var flag = py_call(enabled)
        if py_type(flag) is not b.bool and py_type(flag) is not b.int:
            raise Error("TypeError: native SHAP cache setting must be a boolean or integer")
        if Bool(py=flag):
            if release is None or cached is None:
                raise Error("RuntimeError: TreeExplainer cache entries are incomplete")
            held[].handle = _int(py_call(_entry(table, "x_trees_tree_shap_cache_create"), addresses, trees_address(scales), trees_address(cover), Python.list(d, nt, k, nn)), "cache handle", True)
    var shape = Python.tuple(nt, k, nn, slots, width)
    held[].state = Python.tuple(addresses, trees_address(scales), trees_address(cover), shape, d, keep, plain, cached, mode)
    var changes = Python.dict()
    var names = Python.tuple("model", "data", "feature_perturbation", "model_output", "numeric_mode", "_bg", "n_outputs_", "_forest", "_tscale", "_cover", "_shape", "expected_value", "n_features_in_", "_shap_native", "_shap_cache", "_shap_closed")
    var expected = PythonObject(Float64(_F32(unsafe_from_address=trees_address(ev))[0])) if k == 1 else ev
    var values = Python.tuple(model, data, perturb, output, requested, bg, k, forest, scales, cover, shape, expected, d, owner, PythonObject(held[].handle) if held[].handle > 0 else PythonObject(None), False)
    for i in range(len(names)):
        py_setitem(changes, names[i], values[i])
    _replace(fields, changes, context)
    _ = captures
    _ = ev
    _ = meta
    return PythonObject(None)


def trees_explainer_operation(model: PythonObject, operation: String, args: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    classical_check_layout(model, context)
    var fields = _fields(model)
    if operation == "trees_explainer_init":
        return _construct(fields, args, context)
    var owner = _get(fields, "_shap_native")
    if owner is None:
        raise Error("RuntimeError: TreeExplainer has not been initialized by the native owner")
    _admit_owner(owner)
    var held = owner.downcast_value_ptr[TreeShapOwner]()
    if operation == "trees_explainer_close":
        var changes = Python.dict()
        py_setitem(changes, "_shap_cache", PythonObject(None))
        py_setitem(changes, "_shap_closed", PythonObject(True))
        _replace(fields, changes, context)
        held[].close()
        return PythonObject(None)
    if operation != "trees_explainer_values":
        raise Error("unknown TreeExplainer operation")
    if held[].closed:
        raise Error("RuntimeError: this TreeExplainer has been closed")
    var state = held[].state
    var requested = _get(fields, "numeric_mode")
    if requested is not None and _mode(requested) != String(state[8]):
        raise Error("ValueError: TreeExplainer numeric_mode changed after preparation")
    var snapshots = Python.dict()
    var x = _snapshot(args[0], "<f4", 2, "X", context, snapshots)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if d != Int(py=state[4]):
        raise Error("ValueError: X feature count differs from TreeExplainer background data")
    var shape = state[3]
    var nt = Int(py=shape[0])
    var k = Int(py=shape[1])
    var nn = Int(py=shape[2])
    var slots = Int(py=shape[3])
    var width = Int(py=shape[4])
    var phi = trees_vector(n * d * k, "<f4", context)
    if held[].handle > 0:
        _ = py_call(state[7], held[].handle, trees_address(x), trees_address(phi), Python.list(n, slots, width))
    else:
        _ = py_call(state[6], state[0], state[1], state[2], trees_address(x), trees_address(phi), Python.list(n, d, nt, k, nn, slots, width))
    _ = x
    _ = owner
    return trees_reshape(phi, Python.tuple(n, d) if k == 1 else Python.tuple(n, d, k), context)
