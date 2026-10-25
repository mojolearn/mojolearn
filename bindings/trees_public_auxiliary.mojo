# SPDX-License-Identifier: Apache-2.0
# Forest auxiliary buffer planning only. The existing x_trees entry owns every
# walk, leaf ID and strict increasing-tree prefix reduction on every target.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.mode_metadata_native import _fields
from bindings.native_instance_item import native_instance_item
from bindings.classical_pandas import classical_admit_value
from bindings.trees_public_forest import forest_kind, forest_effective_mode
from bindings.trees_public_inputs import (
    trees_builtins, trees_f32, trees_view, trees_empty, trees_address, trees_table, trees_need,
)


def _field(fields: PythonObject, name: String) raises -> PythonObject:
    if not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: forest model has no attribute '" + name + "'")
    return native_instance_item(fields, PythonObject(name))


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
