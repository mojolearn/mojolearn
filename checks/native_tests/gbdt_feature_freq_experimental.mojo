# SPDX-License-Identifier: Apache-2.0
"""The two-level FeatureFreq fit entry on the literal inputs of
test_feature_freq_experimental_surface.py (lane tests-to-mojo-trees, 2026-11-16).

Moved from the Python tests (their numeric arms; the public door's admission,
refusals, returned estimator, `n_features_in_` and option/weight forwarding
stay there):

  unit 0  `test_experimental_feature_freq_validates_and_forwards`: X 4x3
          [[0, 0, -2.5], [0, 1, 0.25], [1, 0, 3.5], [1, 1, 8]], y [-5, -3, 3, 5],
          sources [0, 1], l2_leaf_reg 4, random_state 7: the model text is not
          empty, two fits at learning_rate 0.2 return the same text, and a fit at
          0.5 returns another.
  unit 1  `test_experimental_feature_freq_forwards_weights` (device column only;
          the host entry refuses weights by name, which `gbdt_public_fits`
          asserts): X 4x2 [[0, 0], [0, 1], [1, 0], [1, 1]], y [-2, -1, 1, 2],
          the constructor defaults (learning_rate 0.03, l2_leaf_reg 3,
          random_state 0): unit weights give the unweighted text, weights
          [1, 2, 3, 4] another.

Both entries take X column-major (the door admits it with order="F") and the
door passes y's address with zero weights when no sample_weight is given;
the check does the same. Host column: `gbdt/mojolearn_gbdt_host_run.mojo`
`gbdt_fit_two_level_feature_freq_run`; device column:
`gbdt/mojolearn_gbdt_run.mojo` `gbdt_fit_two_level_feature_freq_run` under the
family's device plan, as `bindings/_mojolearn_gbdt.mojo` calls it.
"""
from gbdt.mojolearn_gbdt_host_run import gbdt_fit_two_level_feature_freq_run as ff_fit_host
from gbdt.mojolearn_gbdt_run import _gbdt_device_plan, gbdt_fit_two_level_feature_freq_run as ff_fit_device
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require

comptime FF_CPU_UNITS = 1
comptime FF_GPU_UNITS = 2

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]


def _column_major(rows: List[List[Float32]]) -> List[Float32]:
    """X[i][j] at j * n + i, as the door's order="F" admission lays it out."""
    var n = len(rows)
    var d = len(rows[0])
    var out = List[Float32](length=n * d, fill=Float32(0))
    for i in range(n):  # small-loop(four literal rows)
        for j in range(d):  # small-loop(two or three literal columns)
            out[j * n + i] = rows[i][j]
    return out^


def _fit(
    x: List[Float32], n: Int, d: Int, y: List[Float32], weights: List[Float32],
    learning_rate: Float32, l2_leaf_reg: Float32, random_state: Int,
) raises -> String:
    """One fit through the column's entry; empty `weights` is no sample_weight."""
    var sources: List[UInt32] = [0, 1]
    var n_weights = len(weights)
    var x_address = Int(x.unsafe_ptr())
    var y_address = Int(y.unsafe_ptr())
    var w_address = Int(weights.unsafe_ptr()) if n_weights > 0 else y_address
    var sp = _U32P(unsafe_from_address=Int(sources.unsafe_ptr()))
    var text: String
    comptime if CPU_COLUMN_BUILD:
        text = ff_fit_host(x_address, y_address, sp, n, d, n_weights, 2, learning_rate, l2_leaf_reg)
    else:
        text = ff_fit_device(
            _F32P(unsafe_from_address=x_address), n, d, _F32P(unsafe_from_address=y_address),
            _F32P(unsafe_from_address=w_address), n_weights, sp, 2, learning_rate, l2_leaf_reg,
            UInt64(random_state), _gbdt_device_plan())
    _ = sources^  # read by address by the fit above
    return text^


def validates_and_forwards() raises:
    var rows: List[List[Float32]] = [[0.0, 0.0, -2.5], [0.0, 1.0, 0.25], [1.0, 0.0, 3.5], [1.0, 1.0, 8.0]]
    var x = _column_major(rows)
    var y: List[Float32] = [-5.0, -3.0, 3.0, 5.0]
    var none = List[Float32]()
    var first = _fit(x, 4, 3, y, none, 0.2, 4.0, 7)
    require(first.byte_length() > 0, "FeatureFreq fit returned an empty model")
    var second = _fit(x, 4, 3, y, none, 0.2, 4.0, 7)
    require(first == second, "two FeatureFreq fits returned different model text")
    var faster = _fit(x, 4, 3, y, none, 0.5, 4.0, 7)
    require(faster != first, "learning_rate did not reach the FeatureFreq fit")
    _ = x^
    _ = y^


def forwards_weights() raises:
    var rows: List[List[Float32]] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    var x = _column_major(rows)
    var y: List[Float32] = [-2.0, -1.0, 1.0, 2.0]
    var none = List[Float32]()
    var unit = List[Float32](length=4, fill=Float32(1))
    var ramp: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var plain = _fit(x, 4, 2, y, none, 0.03, 3.0, 0)
    require(_fit(x, 4, 2, y, unit, 0.03, 3.0, 0) == plain, "unit weights changed the FeatureFreq model")
    require(_fit(x, 4, 2, y, ramp, 0.03, 3.0, 0) != plain, "weights did not reach the FeatureFreq fit")
    _ = x^
    _ = y^
    _ = unit^
    _ = ramp^


def run_unit(unit: Int) raises:
    if unit == 0:
        validates_and_forwards()
    else:
        forwards_weights()


def _run_cpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(FF_CPU_UNITS)
    def task(unit: Int) {imm cases}:
        try:
            run_unit(unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, FF_CPU_UNITS)


# Each device fit owns the process device context for its whole graph, so the
# device units run in order.
def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(FF_GPU_UNITS)
    for unit in range(FF_GPU_UNITS):  # driver-loop: over the ordered shared-context units
        try:
            run_unit(unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("gbdt_feature_freq_experimental", cpu=True, gpu=True, serial="", run=run_cases)
