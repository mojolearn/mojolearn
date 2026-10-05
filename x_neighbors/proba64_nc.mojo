# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""NearestCentroid.predict_proba as float64 words (lane apple-fast-q-clf,
2026-10-04, QUALITY-FIX, FAST default; -D MOJOLEARN_PROBA64_QOLD keeps the
float32 `softmax_item`). The x_prep/proba64.mojo fix for this binding: a
float32 probability is exactly 1.0 past a ~16.6-nat decision gap, so
proba[:, 1] loses the winner's confidence. Audit (board-quality-audit-
2026-10-04): nearest-centroid istella log loss FAST 4.299 vs scikit-learn
4.118, accuracy the same. Row t of the decision matrix: the winner's
complement c kept apart and 1 - c written exactly from c's bits, every other
class the exact widening of its float32 probability (checks/f64_words.mojo);
res holds 2 * n * c float32 words, (low, high) per value.
OUTCOME (M3 afc_ab_def, 1 run per arm, 2026-10-04, tag rab5-proba64):
nearest-centroid istella log loss 4.299 -> 4.118, accuracy identical, 143.96
-> 142.92 ms. KEEP."""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_exp, identical_div
from checks.f64_words import widen_bits, one_minus_bits, put64
from x_neighbors.items import FP

#: The switch: the bindings register `xn_softmax64` only under it (Python's probe).
comptime NC_PROBA64 = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and not is_defined["MOJOLEARN_PROBA64_QOLD"]()


def softmax64_item(t: Int, x: FP, res: FP, n: Int, c: Int):
    """Row t of x (n x c) -> res words 2 * (t * c + j), j < c."""
    var mx = x.unsafe_load(t * c)
    var km = 0
    for j in range(1, c):
        var v = x.unsafe_load(t * c + j)
        if v > mx:
            mx = v
            km = j
    var tail = Float32(0)
    for j in range(c):
        if j != km:
            tail = ftz(tail + ftz(identical_exp(ftz(x.unsafe_load(t * c + j) - mx))))
    var den = ftz(Float32(1) + tail)
    for j in range(c):
        if j != km:
            var e = ftz(identical_exp(ftz(x.unsafe_load(t * c + j) - mx)))
            put64(res, 2 * (t * c + j), widen_bits(ftz(identical_div(e, den))))
    var cc = ftz(identical_div(tail, den))
    if cc <= Float32(0.5):
        put64(res, 2 * (t * c + km), one_minus_bits(cc))
    else:
        put64(res, 2 * (t * c + km), widen_bits(ftz(identical_div(Float32(1), den))))
