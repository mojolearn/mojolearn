# SPDX-License-Identifier: Apache-2.0
# The tree-ensemble owner's operations (`classical_native_call` operations
# `trees_*`, connected by python/mojolearn/_native_trees_bridge.py). Each
# public method of a connected tree class is one call here.
from std.python import Python, PythonObject
from bindings.classical_attributes import classical_connect_numpy
from bindings.trees_public_forest import forest_fit, forest_predict, forest_predict_proba
from bindings.trees_public_score import forest_score


def trees_public_operation(model: PythonObject, operation: String, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    if py_unavailable(context):
        raise Error("ImportError: mojolearn: the tree ensembles are not connected")
    # Connect a loaded NumPy's native types (ndarray inputs, pandas
    # storage); outputs stay mojolearn Arrays.
    classical_connect_numpy(context)
    if operation == "trees_forest_fit":
        return forest_fit(model, args, context)
    if operation == "trees_forest_predict":
        return forest_predict(model, args, context)
    if operation == "trees_forest_predict_proba":
        return forest_predict_proba(model, args, context)
    if operation == "trees_score_forest":
        return forest_score(model, args, context)
    raise Error("unknown tree ensemble operation " + operation)


def py_unavailable(context: PythonObject) raises -> Bool:
    from bindings.py_balanced import py_contains
    return py_contains(context, "trees_unavailable")
