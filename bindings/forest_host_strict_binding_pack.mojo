# SPDX-License-Identifier: Apache-2.0
"""The host (CPU column) strict snapshot of a fitted forest (N044, lane
ne1-a-trees, 2026-11-06).

`rf_predict_proba`, `rf_predict_reg` and `et_predict` of the host bindings
rebuild every tree from the five flat model arrays (`rf_host_trees`,
`et_host_trees`) on every call before the walk. These entries rebuild them
once per fitted model and keep them under a handle the tree door's resident
table holds (bindings/trees_resident_table_run.mojo), so a predict walks the kept
trees. The walk is the same function over the same trees and the same rows
(`rf_host_predict` / `et_host_predict`, the sequential vote), so no result
changes.

Entries (host bindings only; the door looks for `forest_strict_prepare`):
- `forest_strict_prepare(offsets, colid, quesval, left_child, leaves, [features, trees, outputs])` -> handle
- `forest_strict_predict(handle, x, out, [rows, features, outputs])` -> rows (ROW-major x)
- `forest_strict_release(handle)`

The caller (the door) pins the handle across a predict, so a release from
another thread never frees trees a walk reads; the walk itself runs with the
GIL released, as the per-call entries do. The query X is still copied into a
List before the walk: `DecisionTree.predict_one` reads a `List[Float32]` row
store (a borrowed-pointer walk changes ensemble/decisiontree, not this file;
the copy is one memcpy, the rebuild it replaces was per node).
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.forest_host_strict_run import (
    forest_strict_prepare_host_native,
    forest_strict_predict_host_check,
    forest_strict_predict_host_walk,
    forest_strict_release_host_native,
)


def forest_strict_prepare_host_binding[RF_INPUT: Bool](
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`forest_strict_prepare` (UNPACK of `forest_strict_prepare_host_native`)."""
    if len(params) != 3:
        raise Error("forest_strict_prepare requires features, trees, outputs")
    var model: List[Int] = [Int(py=offsets_addr), Int(py=colid_addr), Int(py=quesval_addr),
                            Int(py=left_child_addr), Int(py=leaves_addr)]
    return PythonObject(forest_strict_prepare_host_native[RF_INPUT](
        model, Int(py=params[0]), Int(py=params[1]), Int(py=params[2])))


def forest_strict_predict_host_native[RF_INPUT: Bool](
    handle: Int, x_address: Int, out_addr: Int, rows: Int, features: Int, outputs: Int,
) raises -> Int:
    """The sequential vote over the kept trees (XF_STRICT_PREDICT):
    `rows * outputs` float32 values into `out`, exactly what `rf_predict_*` /
    `et_predict` write. The caller holds the GIL; the walk releases it.
    The door of the two RUN halves in core/forest_host_strict_run.mojo: the
    checks run holding the GIL (it serializes the registry), the walk runs
    with it released, as before the split."""
    var trees = forest_strict_predict_host_check[RF_INPUT](handle, rows, features, outputs)
    if rows == 0:
        return 0
    with GILReleased(Python()):
        forest_strict_predict_host_walk[RF_INPUT](handle, x_address, out_addr, rows, features, trees, outputs)
    return rows


def forest_strict_predict_host_binding[RF_INPUT: Bool](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`forest_strict_predict` (UNPACK)."""
    if len(params) != 3:
        raise Error("strict prediction requires rows, features, outputs")
    return PythonObject(forest_strict_predict_host_native[RF_INPUT](
        Int(py=handle), Int(py=x_addr), Int(py=out_addr), Int(py=params[0]), Int(py=params[1]), Int(py=params[2])))


def forest_strict_release_host_binding[RF_INPUT: Bool](handle: PythonObject) raises -> PythonObject:
    forest_strict_release_host_native[RF_INPUT](Int(py=handle))
    return PythonObject(None)
