# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The trees DART table's host entry, reached natively (lane
calltable-gbdt-forest-dart, 2026-11-11).

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the publication `_mojolearn_x_trees_host`
makes at module init (`publish_x_trees_dart_host(0)`), a key and a schema
mismatch and an unknown operation refuse. Then one case per published DART
operation, each through `xtrees/dart_calls.mojo` and the published pointer,
with literal expected values or the operation's own refusal text. No Python
participates.

The cases are independent units (each opens its own session or reads its
own forest) and run across the native host worker pool.
Then (lane bm-rest-rte, 2026-11-23, from test_meta_members_native.py
`test_dart_and_adaboost_fits_repeat_bit_for_bit`, its DART half) two
DARTRegressor(n_estimators=8, random_state=0) fits of the original's X =
default_rng(1).random((120, 4), float32), y = X[:, 0] + 0.5 * X[:, 1]
(`fixtures/trees_rte/rep-*`) through the product's typed RUN and the host
images (`bm_rest_boost_fits.mojo`): their predictions are equal bit for bit,
and the members' seeds (`_sub_seed(random_state, j)`, what each member's
`get_params()['random_state']` holds) are distinct. The two fits run as two
units across the host pool. The AdaBoost half stays in Python (its rounds
compose members in the door).
Run in a fresh process (the first check needs an unpublished table).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, assert_same_words
from checks.native_tests.bm_rest_boost_fits import publish_host_images, DartSpec, dart_fit, dart_raw
from xtrees.trees_boost_run import _sub_seed
from core.entry_table import entry_ref_same
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.trees_entry_table import TreesEntryKey
from core.trees_entry_table_dart import trees_dart_table_entry
from xtrees.dart_abi import XD_DART_ABI, XD_OP_COUNT, XD_CAP_ROUND, XD_CAP_RESIDENT, XD_STEP, XD_ADD
from xtrees.dart_host import DART_HOST
from xtrees.dart_runs_host import publish_x_trees_dart_host
from xtrees.dart_calls import (
    xd_ref, xd_dart, xd_caps, xd_open, xd_close, xd_predict, xd_forest_prepare, xd_forest_predict, xd_forest_release,
)

comptime TREES_DART_CASES = 6


def _key() -> TreesEntryKey:
    return TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def precondition_unpublished() raises:
    try:
        _ = xd_caps(xd_ref(_key()))
    except problem:
        _refuses(problem, "provider is not published")
        return
    raise Error("an unpublished DART table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = xd_ref(_key())
    var second = xd_ref(_key())
    require(entry_ref_same(first, second), "trees DART: a second resolution returned another entry")
    try:
        _ = trees_dart_table_entry(_key(), 0, String(XD_DART_ABI), XD_OP_COUNT + 1)
        raise Error("trees DART: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "trees DART: unexpected refusal: " + String(problem))


def preconditions_published() raises:
    ref_resolution()
    try:
        _ = xd_caps(xd_ref(TreesEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0)))
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = trees_dart_table_entry(_key(), 0, String("mojolearn.trees.other.v1"), XD_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        var none = List[Int]()
        _ = xd_dart(xd_ref(_key()), XD_OP_COUNT, none, none, List[Float64]())
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")


def run_case(unit: Int) raises:
    var key = _key()
    if unit == 0:
        # XD_CAPS: the host image keeps no resident forest; the round bit follows DART_HOST.
        var caps = xd_caps(xd_ref(key))
        require((caps & XD_CAP_RESIDENT) == 0, "host caps claim a resident forest")
        require(((caps & XD_CAP_ROUND) != 0) == DART_HOST, "host caps round bit differs from DART_HOST")
        return
    if unit == 1:
        # XD_PREDICT: one split tree (x <= 0.5 left: 1.0, else 3.0), coef 1, start 0.
        var x: List[Float32] = [0.0, 1.0]
        var colid: List[Int32] = [0, 0, 0]
        var ques: List[Float32] = [0.5, 0.0, 0.0]
        var left: List[Int32] = [1, -1, -1]
        var vals: List[Float32] = [0.0, 1.0, 3.0]
        var cols: List[Int] = [Int(colid.unsafe_ptr())]
        var qs: List[Int] = [Int(ques.unsafe_ptr())]
        var ls: List[Int] = [Int(left.unsafe_ptr())]
        var vs: List[Int] = [Int(vals.unsafe_ptr())]
        var sizes: List[Int] = [3]
        var coefs: List[Float64] = [1.0]
        var inits: List[Float64] = [0.0]
        var dst: List[Float64] = [-1.0, -1.0]
        var wrote = xd_predict(xd_ref(key), Int(x.unsafe_ptr()), Int(dst.unsafe_ptr()), Int(cols.unsafe_ptr()),
                               Int(qs.unsafe_ptr()), Int(ls.unsafe_ptr()), Int(vs.unsafe_ptr()),
                               Int(sizes.unsafe_ptr()), Int(coefs.unsafe_ptr()), Int(inits.unsafe_ptr()), 2, 1, 1, 1)
        require(wrote == 2 and dst[0] == 1.0 and dst[1] == 3.0,
                "dart predict: got " + String(dst[0]) + ", " + String(dst[1]))
        _ = x
        _ = colid
        _ = ques
        _ = left
        _ = vals
        _ = cols
        _ = qs
        _ = ls
        _ = vs
        _ = sizes
        _ = coefs
        _ = inits
        return
    if unit == 2:
        # XD_OPEN + XD_CLOSE: a one-round session opens and closes with a clean bad word.
        var x: List[Float32] = [0.0, 1.0]
        var y: List[Float32] = [1.0, 2.0]
        var inits: List[Float32] = [1.5]
        var bad: List[Int32] = [7]
        try:
            var handle = xd_open(xd_ref(key), Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(inits.unsafe_ptr()), 2, 1, 1, 0,
                                 1, 3)
            _ = xd_close(xd_ref(key), handle, Int(bad.unsafe_ptr()))
            require(DART_HOST, "a build without the host round opened a session")
            require(bad[0] == 0, "dart close: bad word " + String(bad[0]))
        except problem:
            if DART_HOST:
                raise problem^
            _refuses(problem, "no x_trees_dart_* round")
        _ = x
        _ = y
        _ = inits
        _ = bad
        return
    if unit == 3:
        # XD_STEP: the former entry's arity refusal (or the build's round refusal).
        var addrs: List[Int] = [0, 0, 0, 0]
        var ints: List[Int] = [0, 0]
        var floats: List[Float64] = [0.0, 0.0]
        try:
            _ = xd_dart(xd_ref(key), XD_STEP, addrs, ints, floats)
            raise Error("a short dart step did not refuse")
        except problem:
            if DART_HOST:
                _refuses(problem, "x_trees_dart_step: params must hold 5 values")
            else:
                _refuses(problem, "no x_trees_dart_* round")
        return
    if unit == 4:
        # XD_ADD: the former entry's arity refusal (or the build's round refusal).
        var addrs: List[Int] = [0, 0, 0, 0, 0]
        var ints: List[Int] = [0]
        var floats: List[Float64] = [0.0]
        try:
            _ = xd_dart(xd_ref(key), XD_ADD, addrs, ints, floats)
            raise Error("a short dart add did not refuse")
        except problem:
            if DART_HOST:
                _refuses(problem, "x_trees_dart_add: params must hold 11 values")
            else:
                _refuses(problem, "no x_trees_dart_* round")
        return
    # unit 5: the kept-forest operations refuse on the host image.
    var out2: List[Int] = [0, 0]
    try:
        _ = xd_forest_prepare(xd_ref(key), 0, 0, 0, 0, 0, 0, 0, Int(out2.unsafe_ptr()), 1, 1, 0)
        raise Error("a host forest prepare did not refuse")
    except problem:
        _refuses(problem, "keeps no resident forest")
    try:
        _ = xd_forest_predict(xd_ref(key), 1, 0, 0, 0, 1, 1)
        raise Error("a host forest predict did not refuse")
    except problem:
        _refuses(problem, "keeps no resident forest")
    try:
        _ = xd_forest_release(xd_ref(key), 1)
        raise Error("a host forest release did not refuse")
    except problem:
        _refuses(problem, "keeps no resident forest")
    _ = out2


comptime _REP_DIR = "checks/native_tests/fixtures/trees_rte/"
comptime _REP_N = 120
comptime _REP_D = 4
comptime _REP_TREES = 8


def dart_repeat() raises:
    """Two DART fits of the same data and seed predict the same bits; the members' seeds are distinct."""
    publish_host_images()
    var preds = List[Float64](length=2 * _REP_N, fill=0.0)
    var pp = preds.unsafe_ptr()

    def fit(unit: Int) raises {imm pp}:
        var x = fixture_f32(_REP_DIR + "rep-x.f32", _REP_N * _REP_D)
        var y = fixture_f32(_REP_DIR + "rep-y.f32", _REP_N)
        var m = dart_fit(DartSpec(_REP_TREES, 0), x, _REP_N, _REP_D, y, 1, 0)
        require(m.trees() == _REP_TREES, "DART trees " + String(m.trees()))
        var p = dart_raw(m, x, _REP_N)
        for i in range(_REP_N):  # small-loop(one fit's predictions into its half of the table)
            pp[unsafe_offset=unit * _REP_N + i] = p[i]

    check_cases("dart_repeat_fits", 2, fit)
    var a = List[Float64](length=_REP_N, fill=0.0)
    var b = List[Float64](length=_REP_N, fill=0.0)
    for i in range(_REP_N):  # small-loop(oracle: split the two fits' predictions)
        a[i] = preds[i]
        b[i] = preds[_REP_N + i]
    assert_same_words[DType.float64](b, a, "DARTRegressor(n_estimators=8, random_state=0) refit predict")
    for j in range(_REP_TREES):  # small-loop(oracle: every pair of member seeds)
        for i in range(j):  # small-loop(the earlier members)
            require(_sub_seed(0, i) != _sub_seed(0, j), "members " + String(i) + " and " + String(j)
                    + " share the seed " + String(_sub_seed(0, j)))
    _ = preds^


def test_unpublished_then_published_operations() raises:
    """One test, in this order: the first call needs the process table unpublished (TestSuite may run any
    test alone)."""
    precondition_unpublished()
    # The publication _mojolearn_x_trees_host makes at module init.
    publish_x_trees_dart_host(0)
    preconditions_published()
    check_cases("case", TREES_DART_CASES, run_case)
    # The other host images publish after the DART table's own preconditions.
    dart_repeat()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
