# SPDX-License-Identifier: Apache-2.0
# Forest auxiliary buffer planning only. The existing x_trees entry owns every
# walk, leaf ID and strict increasing-tree prefix reduction on every target.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native import _fields
from bindings.native_instance_item import native_instance_item
from bindings.classical_pandas import classical_admit_value
from bindings.classical_attributes import classical_check_layout
from bindings.trees_public_forest import forest_kind, forest_effective_mode, _check_mode
from bindings.trees_public_inputs import (
    trees_builtins, trees_f32, trees_view, trees_empty, trees_address, trees_table, trees_need,
)


def _field(fields: PythonObject, name: String) raises -> PythonObject:
    if not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: forest model has no attribute '" + name + "'")
    return native_instance_item(fields, PythonObject(name))


def tree_export_state(model: PythonObject, operation: String) raises -> PythonObject:
    # Shallow copies keep arbitrary application metadata by identity. Reading
    # actual dictionary storage avoids __dict__/copy hooks, and the original
    # dictionary retains removed resident owners throughout this operation.
    var fields = _fields(model)
    var state = py_call(trees_builtins().dict, fields)
    if operation == "trees_forest_getstate":
        _ = py_call(state.pop, "_resident_holder", PythonObject(None))
        _ = py_call(state.pop, "_resident_forest", PythonObject(None))
    elif operation == "trees_iforest_getstate":
        _ = py_call(state.pop, "_if_token", PythonObject(None))
        _ = py_call(state.pop, "_if_lifetime", PythonObject(None))
    else:
        _ = py_call(state.pop, "_resident", PythonObject(None))
    _ = fields
    return state


def _dimension(fields: PythonObject, name: String, context: PythonObject) raises -> Int:
    var value = _field(fields, name)
    classical_admit_value(value, context)
    return Int(py=py_call_typed(trees_builtins().int, value))


def _model_buffers(fields: PythonObject, trees: Int, outputs: Int,
                   context: PythonObject) raises -> PythonObject:
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var types = Python.tuple("<i4", "<i4", "<f4", "<i4", "<f4")
    var addresses = Python.list()
    var owners = Python.list()
    var sizes = List[Int]()
    for i in range(5):  # driver-loop: five immutable forest buffers, no numerical work
        var value = _field(fields, String(names[i]))
        classical_admit_value(value, context)
        var view = trees_view(value, String(names[i]), context)
        if view.ndim() != 1 or view.typestr != String(types[i]) or not view.c_order():
            raise Error("TypeError: forest auxiliary requires flat native " + String(types[i]) + " model storage")
        _ = py_call(addresses.append, PythonObject(view.addr))
        _ = py_call(owners.append, view.keep)
        sizes.append(view.size())
    # The kernel checks each node/child, but an address-only ABI cannot know
    # buffer lengths. Bound offsets before handing it pointers; spare trailing
    # storage remains allowed, as in a retained prefix/slice of a forest.
    if sizes[0] < trees + 1:
        raise Error("forest auxiliary model offsets do not cover all trees")
    var offsets = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=addresses[0]))
    var previous = Int(offsets[0])
    if previous < 0:
        raise Error("forest auxiliary model offsets must be nonnegative")
    for i in range(trees):  # driver-loop: integer storage bounds for each retained tree
        var end = Int(offsets[i + 1])
        if end <= previous or end > sizes[1] or end > sizes[2] or end > sizes[3] or end > sizes[4] // outputs:
            raise Error("forest auxiliary model buffers do not match tree offsets")
        previous = end
    return Python.tuple(addresses, owners)


def forest_shape(model: PythonObject, operation: String,
                 context: PythonObject) raises -> PythonObject:
    """Finished scalar depth/leaf count from the existing native tree walk."""
    var fields = _fields(model)
    if not py_contains(fields, "_offsets"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    classical_admit_value(py_get(fields, "numeric_mode"), context)
    # Shape metadata historically selects the requested/default tier, without
    # imposing prediction's fitted-tier mismatch check on this integer walk.
    var requested = _check_mode(py_get(fields, "numeric_mode"))
    if requested is None:
        requested = context["backend_fields"]["_DEFAULT_MODE"]
        classical_admit_value(requested, context)
        requested = _check_mode(requested)
    var mode = String(requested)
    var entry = PythonObject(None)
    try:
        entry = py_get(trees_table(context, mode, "_mojolearn_x_trees"), "x_trees_tree_shape")
    except error:
        # Preserve the absent host-export refusal of the old public helper.
        if not String(error).startswith("ImportError:"):
            raise error^
    if entry is None:
        raise Error("NotImplementedError: mojolearn: get_depth/get_n_leaves need the x_trees binding's x_trees_tree_shape (rebuild without -D MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF)")
    var offsets_object = _field(fields, "_offsets")
    var left_object = _field(fields, "_left_child")
    classical_admit_value(offsets_object, context)
    classical_admit_value(left_object, context)
    var offsets = trees_view(offsets_object, "offsets", context)
    var left = trees_view(left_object, "left", context)
    if offsets.ndim() != 1 or offsets.typestr != "<i4" or not offsets.c_order() or offsets.size() < 2:
        raise Error("TypeError: tree shape requires flat int32 offsets covering the first tree")
    if left.ndim() != 1 or left.typestr != "<i4" or not left.c_order():
        raise Error("TypeError: tree shape requires flat int32 child storage")
    var offset_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=offsets.addr)
    var lo = Int(offset_ptr[0])
    var hi = Int(offset_ptr[1])
    if lo < 0 or hi <= lo or hi > left.size():
        raise Error("tree shape offsets exceed child storage")
    var result = trees_empty(Python.tuple(2), "<i4", context)
    var result_address = trees_address(result)
    _ = py_call_typed(entry, PythonObject(left.addr + 4 * lo), PythonObject(result_address), Python.list(hi - lo))
    var result_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=result_address)
    var value = Int(result_ptr[0 if operation == "trees_public_forest_depth" else 1])
    _ = result
    _ = offsets
    _ = left
    return PythonObject(value)


def embedding_apply(model: PythonObject, args: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """RandomTreesEmbedding.apply admission, dispatch, allocation and output."""
    var b = trees_builtins()
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: tree embedding requires ordinary class metadata")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var kind = PythonObject(None)
    for i in range(len(ancestry)):  # driver-loop: passive class ancestry preserves inherited embedding interfaces
        kind = py_get(context["trees_boost_classes"], ancestry[i])
        if kind is not None:
            break
    if kind is None or String(kind) != "rte":
        raise Error("TypeError: tree embedding requires its connected class")
    var fields = _fields(model)
    if not py_contains(fields, "forest_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    classical_admit_value(args, context)
    var x = trees_f32(args[0], 2, "C", "X", context)
    var features = _dimension(fields, "n_features_in_", context)
    if x.cols != features:
        raise Error("X has " + String(x.cols) + " features, fit saw " + String(features))
    var trees = _dimension(fields, "n_trees_", context)
    if features < 1 or trees < 1 or x.rows > 2147483647 or features > 2147483647 or trees > 2147483647:
        raise Error("tree embedding requires positive Int32 dimensions")
    if x.rows > 0 and trees > 2305843009213693951 // x.rows:
        raise Error("tree embedding output exceeds addressable bytes")
    var forest = _field(fields, "forest_")
    classical_check_layout(forest, context)
    var forest_fields = _fields(forest)
    var buffers = _model_buffers(forest_fields, trees, 1, context)
    classical_admit_value(py_get(fields, "numeric_mode"), context)
    # The embedding's former _bind selected its requested/default mode.
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    classical_admit_value(mode, context)
    var native = trees_table(context, String(_check_mode(mode)), "_mojolearn_x_trees")
    var entry = trees_need(native, "x_trees_apply", "mojolearn: the x_trees binding has no x_trees_apply", "ImportError")
    var out = trees_empty(Python.tuple(x.rows, trees), "<i4", context)
    _ = py_call_typed(entry, buffers[0][0], buffers[0][1], buffers[0][2], buffers[0][3],
                     PythonObject(x.addr), PythonObject(trees_address(out)),
                     Python.list(x.rows, features, 0, trees))
    _ = x
    _ = buffers^
    return out


def forest_auxiliary(model: PythonObject, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    # The old auxiliary APIs were not predict decorators: retain their own
    # fitted refusal and as_f32_c rank/empty/width validation contract.
    if not py_contains(fields, "_offsets"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    classical_admit_value(args, context)
    var x = trees_f32(args[0], 2, "C", "X", context)
    var features = _dimension(fields, "n_features_in_", context)
    if x.cols != features:
        raise Error("X has " + String(x.cols) + " features, fit saw " + String(features))
    var trees = _dimension(fields, "_n_trees", context)
    var outputs = _dimension(fields, "_num_outputs", context)
    var rows = x.rows
    if features < 1 or trees < 1 or outputs < 1:
        raise Error("forest auxiliary requires positive dimensions")
    # Existing kernel ABI limits: Int32 dimensions and signed byte offsets,
    # checked before allocating requested outputs, never workload thresholds.
    if rows > 2147483647 or features > 2147483647 or trees > 2147483647 or outputs > 2147483647:
        raise Error("forest auxiliary dimensions exceed Int32")
    if rows > 0 and (features > 2305843009213693951 // rows or trees > 2305843009213693951 // rows // outputs):
        raise Error("forest auxiliary requested buffers exceed addressable bytes")
    var leaf_requested = operation == "apply" or operation == "predict_with_leaves"
    var prefix_requested = operation == "staged_predict"
    var prediction_requested = operation == "predict_with_leaves"
    if operation == "auxiliary":
        leaf_requested = Bool(py=py_call_typed(trees_builtins().bool, args[1]))
        prefix_requested = Bool(py=py_call_typed(trees_builtins().bool, args[2]))
        prediction_requested = Bool(py=py_call_typed(trees_builtins().bool, args[3]))
    var leaf = trees_empty(Python.tuple(rows, trees), "<i4", context) if leaf_requested else PythonObject(None)
    var prefix = trees_empty(Python.tuple(rows, trees, outputs), "<f4", context) if prefix_requested else PythonObject(None)
    var prediction = trees_empty(Python.tuple(rows, outputs), "<f4", context) if prediction_requested else PythonObject(None)
    classical_admit_value(py_get(fields, "numeric_mode"), context)
    classical_admit_value(py_get(fields, "_fit_numeric_mode"), context)
    var mode = forest_effective_mode(fields, context)
    var native = trees_table(context, mode, "_mojolearn_x_trees")
    var entry = trees_need(native, "x_trees_forest_auxiliary",
                           "this tree binding does not expose combined auxiliary outputs", "NotImplementedError")
    var buffers = _model_buffers(fields, trees, outputs, context)
    # DecisionTree retains its RF input convention, including a random tree
    # whose flat buffers came from the native inner ExtraTrees fit.
    var rf_input = not forest_kind(model, context).startswith("et_")
    var wrote = py_call_typed(entry, buffers[0], PythonObject(x.addr),
                             PythonObject(trees_address(leaf) if leaf_requested else 0),
                             PythonObject(trees_address(prefix) if prefix_requested else 0),
                             PythonObject(trees_address(prediction) if prediction_requested else 0),
                             Python.list(PythonObject(rows), PythonObject(features), PythonObject(trees),
                                         PythonObject(outputs), PythonObject(1 if rf_input else 0)))
    _ = x
    _ = buffers^
    if Int(py=wrote) != rows:
        raise Error("RuntimeError: forest auxiliary wrote " + String(Int(py=wrote)) + " of " + String(rows) + " rows")
    if operation == "apply":
        return leaf
    if operation == "staged_predict":
        return prefix
    if operation == "predict_with_leaves":
        return Python.tuple(prediction, leaf)
    return Python.tuple(prediction, leaf, prefix)
