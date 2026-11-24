# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The two-level FeatureFreq fit's data checks (gbdt/trees_gbdt_fit_run.mojo `gbdt_feature_freq_check`,
`gbdt_weights_check`, lane mf-meta5): THE ONE BODY the fit door (`feature_freq_unpack`) and the Gbdt member's
kind 3 (core/gbdt_member.mojo `_fit_feature_freq`) both run. Each case is a column-major X the door's words
name; the words are the door's.

    test_feature_freq_checks   a valid X passes; then, one refusal each in the door's order: no columns,
                               a source out of range, a NaN, a negative code, a code gap, a constant numeric
                               column; the weights: a negative weight, all zero, a NaN

FAIL-PROOF: in `gbdt_feature_freq_check` skip the presence pass (`seen`) -> the code-gap case passes and the
test fails; flip `lo == hi` to `lo != hi` -> the valid X is refused.
"""
from std.testing import TestSuite
from std.math import nan
from checks.native_tests.testsuite_cases import require
from gbdt.trees_gbdt_fit_run import gbdt_feature_freq_check, gbdt_weights_check

comptime N = 8


def _x() -> List[Float32]:
    """Three columns, column-major: two sources (codes 0..1 and 0..3, every code present), one numeric."""
    var x = List[Float32](length=N * 3, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        x[i] = Float32(i % 2)
        x[N + i] = Float32(i % 4)
        x[2 * N + i] = Float32(i) * 0.5
    return x^


def _refused(x: List[Float32], n: Int, d: Int, sources: List[Int], words: String) raises -> Bool:
    try:
        gbdt_feature_freq_check(Int(x.unsafe_ptr()), n, d, sources)
    except error:
        return String(error).find(words) >= 0
    return False


def _weights_refused(w: List[Float32], words: String) raises -> Bool:
    try:
        gbdt_weights_check(Int(w.unsafe_ptr()), len(w))
    except error:
        return String(error).find(words) >= 0
    return False


def test_feature_freq_checks() raises:
    var sources: List[Int] = [0, 1]
    var x = _x()
    gbdt_feature_freq_check(Int(x.unsafe_ptr()), N, 3, sources)
    require(_refused(x, N, 1, sources, "needs rows and columns"), "one column is refused")
    var outside: List[Int] = [0, 3]
    require(_refused(x, N, 3, outside, "sources [0, 3] outside 3 features"), "a source out of range is refused")
    var bad = _x()
    bad[2 * N + 1] = nan[DType.float32]()
    require(_refused(bad, N, 3, sources, "X must be finite"), "a NaN is refused")
    bad = _x()
    bad[3] = -1.0
    require(_refused(bad, N, 3, sources, "non-negative integer category codes"), "a negative code is refused")
    bad = _x()
    for i in range(N):  # small-loop(fixture: codes 0 and 2, code 1 missing)
        bad[N + i] = Float32(2 * (i % 2))
    require(_refused(bad, N, 3, sources, "column 1 has codes up to 2"), "a code gap is refused")
    bad = _x()
    for i in range(N):  # small-loop(fixture: a constant numeric column)
        bad[2 * N + i] = 1.0
    require(_refused(bad, N, 3, sources, "column 2 is constant"), "a constant numeric column is refused")
    var w: List[Float32] = [1.0, 0.5, 0.0, 2.0]
    gbdt_weights_check(Int(w.unsafe_ptr()), len(w))
    var negative: List[Float32] = [1.0, -0.5]
    require(_weights_refused(negative, "finite and non-negative"), "a negative weight is refused")
    var zero: List[Float32] = [0.0, 0.0]
    require(_weights_refused(zero, "positive sum"), "all-zero weights are refused")
    var missing: List[Float32] = [1.0, nan[DType.float32]()]
    require(_weights_refused(missing, "finite and non-negative"), "a NaN weight is refused")
    _ = x^
    _ = bad^
    _ = w^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
