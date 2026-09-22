# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host RandomForest builder on non-finite X: NaN refused, +-inf fitted.

    pixi run check-rf-nonfinite

Found by the pip smoke of 0.8.13 (2026-09-22): a fit on an X with NaN never
returned. A NaN bins to 0 in the histogram (`host_lower_bound`'s
`values[mid] < NaN` is False) but partitions RIGHT (`NaN <= quesval` is
False), so a split whose histogram-left side is only NaN rows hands the
right child every row of its parent, and at the default unlimited depth
(`max_depth=None` -> INT32_MAX) that node splits forever. Measured before
the fix: this file's NaN case ran past 200 s without returning.

+-inf is NOT refused: +inf lands past the last quantile (bin nb-1, right of
every split whose quesval is finite) and -inf at bin 0 (left of every
split), in the histogram and the partition alike, so the fit terminates
and is deterministic. The inf cases here fit twice and compare every model
array bit for bit. (The Python estimators refuse NaN and inf alike before
reaching the builder; this file guards the builder itself.)
"""
from std.math import inf, nan

from ensemble.host.rf_oracle import (
    RF_CRITERION_END,
    RfHostForest,
    RfHostParams,
    rf_host_fit,
)
from ensemble.host_layout import RF_NAN_REFUSAL

comptime N = 350
comptime D = 10


def _data(mode: Int) -> Tuple[List[Float32], List[Int32]]:
    var x = List[Float32](length=N * D, fill=Float32(0))
    var y = List[Int32](length=N, fill=Int32(0))
    var s = UInt64(12345)
    for i in range(N * D):
        s = s * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int((s >> 33) % 1000)) / 100.0
    for r in range(N):
        y[r] = Int32(1) if x[r] + x[N + r] > 10.0 else Int32(0)
    if mode == 1:  # one +inf cell
        x[N + 3] = inf[DType.float32]()
    elif mode == 2:  # +inf and -inf scattered over two columns
        for r in range(0, N, 7):
            x[2 * N + r] = inf[DType.float32]()
        for r in range(0, N, 11):
            x[4 * N + r] = -inf[DType.float32]()
    elif mode == 3:  # a whole +inf column
        for r in range(N):
            x[N + r] = inf[DType.float32]()
    elif mode == 4:  # the hang: NaN in about 30% of the cells
        for i in range(N * D):
            if x[i] > 7.0:
                x[i] = nan[DType.float32]()
    elif mode == 5:  # one NaN cell
        x[5 * N + 17] = nan[DType.float32]()
    return (x^, y^)


def _fit(mode: Int) raises -> RfHostForest:
    var d = _data(mode)
    # max_depth INT32_MAX is `max_depth=None`, the Python default.
    var p = RfHostParams(
        3, 2147483647, -1, Float32(1.0), 128, 1, 2, Float32(0), True,
        Float32(1.0), UInt64(0), 4, 4096, RF_CRITERION_END,
    )
    return rf_host_fit(d[0].copy(), d[1], List[Float32](), N, D, 2, True, p, Float32(1.0))


def _same(a: RfHostForest, b: RfHostForest) -> Bool:
    if a.n_nodes() != b.n_nodes() or len(a.leaves) != len(b.leaves):
        return False
    for i in range(a.n_nodes()):
        if a.colid[i] != b.colid[i] or a.left_child[i] != b.left_child[i]:
            return False
        if a.quesval[i] != b.quesval[i] and not (a.quesval[i] != a.quesval[i]):
            return False
    for i in range(len(a.leaves)):
        if a.leaves[i] != b.leaves[i]:
            return False
    return True


def main() raises:
    var failed = 0
    for mode in range(4):
        var a = _fit(mode)
        var b = _fit(mode)
        if _same(a, b):
            print("  mode", mode, ": fitted,", a.n_nodes(), "nodes, refit identical")
        else:
            print("  mode", mode, ": FAIL refit differs")
            failed += 1
    for mode in range(4, 6):
        try:
            _ = _fit(mode)
            print("  mode", mode, ": FAIL NaN accepted")
            failed += 1
        except e:
            if String(e) == RF_NAN_REFUSAL:
                print("  mode", mode, ": NaN refused by name")
            else:
                print("  mode", mode, ": FAIL wrong error:", e)
                failed += 1
    if failed != 0:
        raise Error("rf non-finite check: " + String(failed) + " case(s) failed")
    print("  all 6 cases pass")
