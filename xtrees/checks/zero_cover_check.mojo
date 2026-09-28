# SPDX-License-Identifier: Apache-2.0
"""CPU-only TreeSHAP regression: zero-cover repeated-feature path.

mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . xtrees/checks/zero_cover_check.mojo
"""
from xtrees.shap import tree_shap, expected_value
from std.memory import bitcast


def _pi(mut x: List[Int32]) -> MutPointer[Int32, MutUntrackedOrigin]:
    return x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _pf(mut x: List[Float32]) -> MutPointer[Float32, MutUntrackedOrigin]:
    return x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _pd(mut x: List[Float64]) -> MutPointer[Float64, MutUntrackedOrigin]:
    return x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def main() raises:
    # Background visits only the left leaf. The right subtree splits the
    # same feature again; its cold path has zero one/zero fractions.
    var offsets: List[Int32] = [0, 5]
    var columns: List[Int32] = [0, -1, 0, -1, -1]
    var thresholds: List[Float32] = [0, 0, 0.5, 0, 0]
    var left: List[Int32] = [1, -1, 3, -1, -1]
    var leaves: List[Float32] = [0, 1, 0, 2, 3]
    var cover: List[Float64] = [1, 1, 0, 0, 0]
    var rows: List[Float32] = [-1, 1]
    var phi: List[Float64] = [0, 0]
    var ev: List[Float64] = [0]
    tree_shap(_pi(offsets), _pi(columns), _pf(thresholds), _pi(left),
              _pf(leaves), _pd(cover), _pf(rows), 2, 1, 1, 1, 1.0, _pd(phi))
    expected_value(_pi(offsets), _pi(left), _pf(leaves), _pd(cover), 1, 1, 1.0, _pd(ev))
    if phi[0] != 0 or phi[1] != 2 or ev[0] != 1:
        raise Error("zero-cover TreeSHAP must be finite with contributions [0, 2] and expected value 1")
    if phi[0] + ev[0] != 1 or phi[1] + ev[0] != 3:
        raise Error("zero-cover TreeSHAP additivity failed")
    print("PASS zero-cover TreeSHAP", bitcast[DType.uint64](phi[0]), bitcast[DType.uint64](phi[1]))
    _ = len(offsets) + len(columns) + len(thresholds) + len(left) + len(leaves) + len(cover) + len(rows) + len(phi) + len(ev)
