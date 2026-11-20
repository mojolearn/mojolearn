# SPDX-License-Identifier: Apache-2.0
"""IsolationForest fit then export: one fit, the fitted forest read out.

Lane fix-iforest-double-fit (2026-11-13). The bindings' `fit` sequence
(`bindings/_mojolearn_svm.mojo`: `iforest_run` with seventeen params and
token 0, then `iforest_forest_export`'s sizes and copy calls with the token
it returned) runs here natively, without Python, on every route: a
refit-per-call binary (`-D MOJOLEARN_IDN_IF_RESIDENT_OFF`) takes the same
seventeen-param fit form since this lane. Checked per fixture:

- one fit: the export's take finds the forest `iforest_run_resident` kept
  under its token, and `if_resident_fit_if_empty` leaves that entry (same
  token, no new entry), which is the only way the export can fit;
- the forest read out from that one fit equals, bit for bit, the forest a
  separate fresh fit on the same key and seed exports (the resident route's
  export), and its offsets span its nodes.

Fixtures run in order on one GPU (ordered GPU work on one process context);
the compares inside a fixture run one task per word across the host cores.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.device_plan import device_plan_current
from core.host_parallel import host_parallelize
from isolation_forest.checks.if_fixture import blob_fixture
from isolation_forest.impl.isolation_forest import IDN_IF_QUERY_DEVICE
from isolation_forest.estimator import (
    IF_WANT_SCORE_SAMPLES,
    IFResidentEntry,
    IFResidentKey,
    if_resident_fit_if_empty,
    if_resident_new_token,
    if_resident_put,
    if_resident_release,
    if_resident_take,
    iforest_run_resident,
)
from isolation_forest.forest_io import IFForestWords, if_held_forest_export, if_held_forest_sizes
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The device column (was iforest_forest_export_gpu_suite.mojo) ----
comptime IF_EXPORT_FIXTURES = 3


struct _Forest(Movable):
    """One exported compact forest on the host."""

    var offsets: List[Int32]
    var feature: List[Int32]
    var threshold: List[Float32]
    var left: List[Int32]
    var right: List[Int32]

    def __init__(out self, n_trees: Int, total: Int):
        self.offsets = List[Int32](length=n_trees + 1, fill=Int32(-1))
        self.feature = List[Int32](length=total, fill=Int32(-1))
        self.threshold = List[Float32](length=total, fill=Float32(-1))
        self.left = List[Int32](length=total, fill=Int32(-1))
        self.right = List[Int32](length=total, fill=Int32(-1))


def _export(held: List[IFResidentEntry]) raises -> _Forest:
    """`iforest_forest_export`'s sizes call then its copy call on `held`."""
    var sizes = if_held_forest_sizes(held)
    var forest = _Forest(sizes[0], sizes[1])
    var words = IFForestWords(
        Int(forest.offsets.unsafe_ptr()), Int(forest.feature.unsafe_ptr()), Int(forest.threshold.unsafe_ptr()),
        Int(forest.left.unsafe_ptr()), Int(forest.right.unsafe_ptr()), sizes[0], sizes[1],
    )
    if_held_forest_export(held, words)
    return forest^


def _same_words(a_addr: Int, b_addr: Int, count: Int) raises -> Bool:
    """Bit-for-bit equality of `count` 32-bit words, one task per word."""
    if count <= 0:
        return True
    var failures = List[Int](length=count, fill=0)
    var fp = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(failures.unsafe_ptr()))
    var ap = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=a_addr)
    var bp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=b_addr)

    def cell(index: Int) {imm fp, imm ap, imm bp}:
        fp.unsafe_store(index, Int(ap[index] != bp[index]))

    host_parallelize(cell, count)
    var bad = 0
    for failure in failures:
        bad += failure
    _ = failures^
    return bad == 0


def _same(a: _Forest, b: _Forest) raises -> Bool:
    if len(a.offsets) != len(b.offsets) or len(a.feature) != len(b.feature):
        return False
    return (
        _same_words(Int(a.offsets.unsafe_ptr()), Int(b.offsets.unsafe_ptr()), len(a.offsets))
        and _same_words(Int(a.feature.unsafe_ptr()), Int(b.feature.unsafe_ptr()), len(a.feature))
        and _same_words(Int(a.threshold.unsafe_ptr()), Int(b.threshold.unsafe_ptr()), len(a.threshold))
        and _same_words(Int(a.left.unsafe_ptr()), Int(b.left.unsafe_ptr()), len(a.left))
        and _same_words(Int(a.right.unsafe_ptr()), Int(b.right.unsafe_ptr()), len(a.right))
    )


def fixture(unit: Int) raises:
    var n = 300 + 97 * unit
    var d = 5 + unit
    var train = blob_fixture(n, d, 41 + unit, 6)
    var key = IFResidentKey(
        Int(train.unsafe_ptr()), n, d, 12 + 4 * unit, 0, 0, 1.0, -1 if unit != 1 else 6, 0, 0, 1.0,
        unit == 2, 7 + unit, unit != 1, 0.05,
    )
    var plan = device_plan_current("MOJOLEARN_IFOREST_DEVICE_COUNT")
    var fitted_token = 0
    var fresh_token = 0
    try:
        # `fit`: iforest_run, seventeen params, token 0; it scores the first
        # training row and keeps the forest under a new token.
        var held = if_resident_take(0, key)
        require(len(held) == 0, "iforest fit: token 0 takes no forest")
        fitted_token = if_resident_new_token()
        var score = List[Float32](length=1, fill=Float32(0))
        var query = List[Float32]()
        var q_addr = 0
        comptime if IDN_IF_QUERY_DEVICE:
            q_addr = Int(train.unsafe_ptr())
        else:
            for k in range(d):  # driver-loop: over the one query row's features
                query.append(train[k])
        _ = iforest_run_resident(plan, held, key, fitted_token, query, q_addr, 1, IF_WANT_SCORE_SAMPLES,
                                 Int(score.unsafe_ptr()), 0, False)
        require(len(held) == 1 and held[0].token == fitted_token, "iforest fit keeps one forest under its token")
        if_resident_put(held)

        # The export's sizes call and copy call, with the token fit returned.
        var taken = if_resident_take(fitted_token, key)
        require(len(taken) == 1, "iforest export finds the forest fit kept (no second fit)")
        if_resident_fit_if_empty(plan, taken, key, if_resident_new_token())
        require(len(taken) == 1 and taken[0].token == fitted_token,
                "iforest export does not fit when the fitted forest is held")
        var exported = _export(taken)
        var n_trees = len(exported.offsets) - 1
        require(n_trees == key.n_estimators, "iforest export holds every tree")
        require(Int(exported.offsets[0]) == 0 and Int(exported.offsets[n_trees]) == len(exported.feature),
                "iforest export offsets span the nodes")
        # A refit-per-call binary drops the forest after the copy call.
        _ = taken^

        # The resident route's export on the same key and seed: a separate fit.
        var fresh = List[IFResidentEntry]()
        fresh_token = if_resident_new_token()
        if_resident_fit_if_empty(plan, fresh, key, fresh_token)
        require(len(fresh) == 1 and fresh[0].token == fresh_token, "iforest fresh fit keeps its own forest")
        var reference = _export(fresh)
        _ = fresh^
        require(_same(exported, reference), "iforest export of the one fit equals a fresh fit's export bit for bit")
        _ = score
        _ = query
    finally:
        if_resident_release(fitted_token)
        if_resident_release(fresh_token)
        _ = train


def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var cases = report.cases(IF_EXPORT_FIXTURES)
    for unit in range(IF_EXPORT_FIXTURES):  # driver-loop: over fixtures, ordered GPU work on one context
        try:
            fixture(unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("iforest_forest_export", cpu=False, gpu=True, serial="", run=run_cases)
