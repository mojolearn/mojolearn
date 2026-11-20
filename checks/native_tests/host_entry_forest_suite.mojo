# SPDX-License-Identifier: Apache-2.0
"""The host entry table's forest host entry, reached natively (lane
calltable-host-misc, 2026-11-11).

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the same publication `_mojolearn_forest_host`
makes at module init (`publish_forest_host_entries`, twice: a repeated init
is accepted), a tier, a vendor and a schema mismatch refuse, and an unknown
operation refuses through the entry's sink. The two grove operations share
one handle registry, so their cases run in order too. Then one case per
remaining published operation, each through `core/host_entry_table_calls.mojo`
and the published pointer, with literal expected values or the run's own
refusal text (the text its former Python export raised). No Python
participates. Production build only: a sabotage build changes the divisor.

The fixture forest is one tree of three nodes: node 0 tests column 0 at 0.5
(x <= 0.5 goes to its left child, node 1), nodes 1 and 2 are leaves. Rows
x = 0.0 and x = 1.0 reach nodes 1 and 2.

The independent cases run across the native host worker pool.
Run in a fresh process (the first check needs an unpublished table).
"""
from core.entry_table import entry_ref_same
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.host_entry_table import (
    HostEntryKey, host_table_entry, HOST_TABLE_FOREST_HOST, FOREST_HOST_ABI, FH_OP_COUNT,
    FH_RF_PREDICT_PROBA, FH_RF_PREDICT_REG, FH_ET_PREDICT,
)
from core.host_entry_table_forest import publish_forest_host_entries
from core.host_entry_table_calls import (
    fh_ref, fh_call, fh_vote, fh_groves_prepare, fh_groves_predict, fh_gbdt_expand_ctr, fh_gbdt_predict,
    fh_gbdt_sigmoid_pair,
)

comptime HOST_FOREST_CASES = 7


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> HostEntryKey:
    return HostEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


struct Fixture(Movable):
    """The one-tree forest of the module note, `outputs` values per node."""

    var offsets: List[Int32]
    var colid: List[Int32]
    var quesval: List[Float32]
    var left: List[Int32]
    var leaves: List[Float32]
    var x: List[Float32]

    def __init__(out self, outputs: Int):
        self.offsets = [0, 3]
        self.colid = [0, 0, 0]
        self.quesval = [0.5, 0.0, 0.0]
        self.left = [1, -1, -1]
        if outputs == 1:
            self.leaves = [0.0, 3.0, 5.0]
        else:
            self.leaves = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        self.x = [0.0, 1.0]

    def model(mut self) -> List[Int]:
        var addrs: List[Int] = [
            Int(self.offsets.unsafe_ptr()), Int(self.colid.unsafe_ptr()), Int(self.quesval.unsafe_ptr()),
            Int(self.left.unsafe_ptr()), Int(self.leaves.unsafe_ptr()),
        ]
        return addrs^

    def x_addr(mut self) -> Int:
        return Int(self.x.unsafe_ptr())


def _is_one_hot_pair(values: List[Float32]) -> Bool:
    return values[0] == 1.0 and values[1] == 0.0 and values[2] == 0.0 and values[3] == 1.0


def precondition_unpublished() raises:
    var raw: List[Float64] = [0.0]
    var out: List[Float64] = [0.0, 0.0]
    try:
        _ = fh_gbdt_sigmoid_pair(fh_ref(_key()), Int(raw.unsafe_ptr()), Int(out.unsafe_ptr()), 1)
    except problem:
        _refuses(problem, "provider is not published")
        _ = raw
        _ = out
        return
    raise Error("an unpublished host table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = fh_ref(_key())
    var second = fh_ref(_key())
    require(entry_ref_same(first, second), "forest host: a second resolution returned another entry")
    try:
        _ = host_table_entry[HOST_TABLE_FOREST_HOST](_key(), 0, String(FOREST_HOST_ABI), FH_OP_COUNT + 1)
        raise Error("forest host: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "forest host: unexpected refusal: " + String(problem))


def preconditions_published() raises:
    ref_resolution()
    var raw: List[Float64] = [0.0]
    var out: List[Float64] = [0.0, 0.0]
    try:
        _ = fh_gbdt_sigmoid_pair(fh_ref(HostEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0)), Int(raw.unsafe_ptr()),
                                 Int(out.unsafe_ptr()), 1)
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = fh_gbdt_sigmoid_pair(fh_ref(HostEntryKey(Int(GLOBAL_NUMERIC_MODE), 2)), Int(raw.unsafe_ptr()),
                                 Int(out.unsafe_ptr()), 1)
        raise Error("a GPU vendor key did not refuse")
    except problem:
        _refuses(problem, "invalid tier/vendor key")
    try:
        _ = host_table_entry[HOST_TABLE_FOREST_HOST](_key(), 0, String("mojolearn.host.other.v1"), FH_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        var none = List[Int]()
        _ = fh_call(fh_ref(_key()), FH_OP_COUNT, none, none, List[Float64]())
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")
    _ = raw
    _ = out


def grove_cases() raises:
    """FH_GROVES_PREPARE and FH_GROVES_PREDICT, in order (one registry)."""
    var f = Fixture(2)
    var handle = fh_groves_prepare(fh_ref(_key()), f.model(), 1, 1, 2, 3, 0)
    require(handle > 0, "groves prepare handle")
    var out = List[Float32](length=4, fill=Float32(-1.0))
    var wrote = fh_groves_predict(fh_ref(_key()), handle, f.x_addr(), Int(out.unsafe_ptr()), 2, 1, 2, 0)
    require(wrote == 2 and _is_one_hot_pair(out), "groves predict")
    try:
        _ = fh_groves_predict(fh_ref(_key()), handle, f.x_addr(), Int(out.unsafe_ptr()), 2, 1, 2, 1)
        raise Error("a handle of the other family did not refuse")
    except problem:
        _refuses(problem, "unknown or released host grove forest handle")
    try:
        _ = fh_groves_prepare(fh_ref(_key()), f.model(), 1, 1, 2, 3, 2)
        raise Error("an unknown family did not refuse")
    except problem:
        _refuses(problem, "family must be 0 (RandomForest) or 1 (ExtraTrees)")
    _ = out^
    _ = f^


def run_case(unit: Int) raises:
    var k = fh_ref(_key())
    if unit == 0:
        var f = Fixture(2)
        var out = List[Float32](length=4, fill=Float32(-1.0))
        var addrs = f.model()
        addrs.append(f.x_addr())
        addrs.append(Int(out.unsafe_ptr()))
        require(fh_vote(k, FH_RF_PREDICT_PROBA, addrs, 2, 1, 1, 2, 3) == 2 and _is_one_hot_pair(out), "rf proba")
        _ = out^
        _ = f^
    elif unit == 1:
        var f = Fixture(1)
        var out = List[Float32](length=2, fill=Float32(-1.0))
        var addrs = f.model()
        addrs.append(f.x_addr())
        addrs.append(Int(out.unsafe_ptr()))
        require(fh_vote(k, FH_RF_PREDICT_REG, addrs, 2, 1, 1, 1, 3) == 2 and out[0] == 3.0 and out[1] == 5.0,
                "rf reg")
        try:
            _ = fh_vote(k, FH_RF_PREDICT_PROBA, addrs, 2, 1, 1, 1, 3)
            raise Error("a one-output probability vote did not refuse")
        except problem:
            _refuses(problem, "forest_host_rf_predict_proba: num_outputs must be >= 2")
        _ = out^
        _ = f^
    elif unit == 2:
        var f = Fixture(2)
        var out = List[Float32](length=4, fill=Float32(-1.0))
        var addrs = f.model()
        addrs.append(f.x_addr())
        addrs.append(Int(out.unsafe_ptr()))
        require(fh_vote(k, FH_ET_PREDICT, addrs, 2, 1, 1, 2, 3) == 2 and _is_one_hot_pair(out), "et predict")
        try:
            _ = fh_vote(k, FH_ET_PREDICT, addrs, 2, 1, 1, 2, 2)
            raise Error("tree offsets past n_nodes did not refuse")
        except problem:
            _refuses(problem, "forest_host_et_predict: tree_offsets must start at 0 and end at n_nodes")
        _ = out^
        _ = f^
    elif unit == 3:
        # The expansion refuses an empty raw width before it reads a buffer.
        var addrs = List[Int](length=10, fill=0)
        var params: List[Int] = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        try:
            _ = fh_gbdt_expand_ctr(k, addrs, params)
            raise Error("an empty raw width did not refuse")
        except problem:
            _refuses(problem, "forest_host_gbdt_expand_ctr: n_raw must be positive and n_cols in [n_raw, 65536]")
        var short: List[Int] = [1, 1, 1]
        try:
            _ = fh_gbdt_expand_ctr(k, addrs, short)
            raise Error("a short parameter list did not refuse")
        except problem:
            _refuses(problem, "forest_host_gbdt_expand_ctr: expected 10 addresses and 11 params")
    elif unit == 4:
        # The prediction refuses dim 0 before it reads a buffer.
        var addrs = List[Int](length=15, fill=0)
        var params: List[Int] = [1, 1, 0, 0, 0, 0, 0, 0, 1]
        try:
            _ = fh_gbdt_predict(k, addrs, params, 0.25)
            raise Error("dim 0 did not refuse")
        except problem:
            _refuses(problem, "forest_host_gbdt_predict: dim must be in [1, 65536]")
        var short = List[Int](length=7, fill=0)
        try:
            _ = fh_gbdt_predict(k, short, params, 0.25)
            raise Error("a short address list did not refuse")
        except problem:
            _refuses(problem, "forest_host_gbdt_predict: expected 15 addresses and 8 or 9 params")
    elif unit == 5:
        var raw: List[Float64] = [0.0, 0.0]
        var out: List[Float64] = [-1.0, -1.0, -1.0, -1.0]
        require(fh_gbdt_sigmoid_pair(k, Int(raw.unsafe_ptr()), Int(out.unsafe_ptr()), 2) == 2
                and out[0] == 0.5 and out[1] == 0.5 and out[2] == 0.5 and out[3] == 0.5, "sigmoid pair")
        try:
            _ = fh_gbdt_sigmoid_pair(k, Int(raw.unsafe_ptr()), Int(out.unsafe_ptr()), -1)
            raise Error("a negative row count did not refuse")
        except problem:
            _refuses(problem, "forest_host_gbdt_sigmoid_pair: n must be non-negative")
        _ = raw
        _ = out
    else:
        # A vote over a null x refuses through the run's pointer check.
        var f = Fixture(2)
        var out = List[Float32](length=4, fill=Float32(-1.0))
        var addrs = f.model()
        addrs.append(0)
        addrs.append(Int(out.unsafe_ptr()))
        try:
            _ = fh_vote(k, FH_RF_PREDICT_PROBA, addrs, 2, 1, 1, 2, 3)
            raise Error("a null input did not refuse")
        except problem:
            _refuses(problem, "null float32 buffer address")
        _ = out^
        _ = f^


def run_checks() raises:
    precondition_unpublished()
    # The publication _mojolearn_forest_host makes at module init; a repeated
    # init publishes the same pointer and is accepted.
    publish_forest_host_entries()
    publish_forest_host_entries()
    preconditions_published()
    grove_cases()
    var failures = List[Int](length=HOST_FOREST_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL host forest entry case", unit, String(problem))
    host_parallelize(task, HOST_FOREST_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "host forest entry cases failed: " + String(failed))
    print("PASS host forest entry", HOST_FOREST_CASES + 1)
