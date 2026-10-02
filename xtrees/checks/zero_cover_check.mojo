# SPDX-License-Identifier: Apache-2.0
"""CPU-only TreeSHAP regression: zero-cover repeated-feature path.

mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . xtrees/checks/zero_cover_check.mojo
"""
from xtrees.shap import shap_path_width
from xtrees.shap_host import shap_prepare, tree_shap_values
from std.memory import bitcast


def _a[dt: DType](mut x: List[Scalar[dt]]) -> Int:
    return Int(x.unsafe_ptr())


def main() raises:
    # Background row 0 reaches the left leaf only. The right subtree splits
    # the same feature again; its cold path has zero one/zero fractions.
    var offsets: List[Int32] = [0, 5]
    var columns: List[Int32] = [0, -1, 0, -1, -1]
    var thresholds: List[Float32] = [0, 0, 0.5, 0, 0]
    var left: List[Int32] = [1, -1, 3, -1, -1]
    var leaves: List[Float32] = [0, 1, 0, 2, 3]
    var tscale: List[Float32] = [1]
    var bg: List[Float32] = [-1]
    var rows: List[Float32] = [-1, 1]
    var cover = List[Int32](length=5, fill=0)
    var ev: List[Float32] = [0]
    var meta = List[Int32](length=3, fill=0)
    var phi = List[Float32](length=2, fill=0)
    var forest: List[Int] = [_a(offsets), _a(columns), _a(thresholds), _a(left), _a(leaves)]
    shap_prepare(forest, _a(tscale), _a(bg), _a(cover), _a(ev), _a(meta), 1, 1, 1, 1, 5)
    if cover[0] != 1 or cover[1] != 1 or cover[2] != 0:
        raise Error("zero-cover TreeSHAP: background counts wrong")
    var w = shap_path_width(Int(meta[1]), Int(meta[0]))
    tree_shap_values(forest, _a(tscale), _a(cover), _a(rows), _a(phi), 2, 1, 1, 1, 5, Int(meta[0]), w)
    if phi[0] != 0 or phi[1] != 2 or ev[0] != 1:
        raise Error("zero-cover TreeSHAP must be finite with contributions [0, 2] and expected value 1")
    if phi[0] + ev[0] != 1 or phi[1] + ev[0] != 3:
        raise Error("zero-cover TreeSHAP additivity failed")
    print("PASS zero-cover TreeSHAP", bitcast[DType.uint32](phi[0]), bitcast[DType.uint32](phi[1]))
    _ = len(offsets) + len(columns) + len(thresholds) + len(left) + len(leaves) + len(tscale) + len(bg)
    _ = len(rows) + len(cover) + len(ev) + len(meta) + len(phi)
