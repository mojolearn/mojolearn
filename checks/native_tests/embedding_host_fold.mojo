# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The host Embedding's gather and fold on the CPU column (lane py2mojo-D, 2026-11-22; ported from
python/mojolearn/tests/test_cpu_training_embedding_ivf.py `test_embedding_runs_on_the_host_when_built`, its numeric
half).

The calls are the typed bodies the `_mojolearn_embedding_host` doors run with the GIL released
(embedding/host/embedding_host_words_run.mojo `embedding_host_forward_released`, `embedding_host_backward_released`,
with `embedding_host_config`). Inputs: the original's NumPy draws (seed 4: W 11 x 5, ids 37, dY 37 x 5), stored in
`fixtures/emb_host/` by `tools/py2mojo_native_fixtures_b7.py`. The reference is the original's: the forward is
`W[ids]`; dW is seeded +0.0, then `dW[ids[t]] += dY[t]` in ascending t in float32, and row padding_idx = 2 is stored
+0.0. Every comparison is bit for bit.

    unit 0  forward equals the gather W[ids]
    unit 1  PLAN_SCAN: the full backward is the reference; the backward of ids[13:] carried onto the backward of
            ids[:13] is the full one; no cell of the padding row has its sign bit set
    unit 2  PLAN_SORT: the same three claims, and its bytes equal PLAN_SCAN's

The out-of-range id refusal, the sabotage and column words of the module, and the public object stay in the
Python test. Each unit is independent; the case table runs them across host cores.
"""
from std.memory import bitcast
from embedding.checks.embedding_oracle import EMB_NO_PADDING_IDX
from embedding.host.embedding_host import HOST_PLAN_SCAN, HOST_PLAN_SORT
from embedding.host.embedding_host_words_run import (
    embedding_host_backward_released,
    embedding_host_config,
    embedding_host_forward_released,
)
from core.hostptr import f32_ptr, i32_ptr
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32, fixture_i32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime EMB_DIR = "checks/native_tests/fixtures/emb_host/"
comptime V = 11
comptime D = 5
comptime T = 37
comptime PAD = 2
comptime SPLIT = 13
comptime EMB_HOST_UNITS = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths " + String(len(a)) + " and " + String(len(b)))
    for i in range(len(a)):  # driver-loop: over one fixture's compared cells
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(label + ": cell " + String(i) + " is " + String(a[i]) + ", expected " + String(b[i]))


def forward(w: List[Float32], ids: List[Int32]) raises -> List[Float32]:
    var cfg = embedding_host_config(V, D, EMB_NO_PADDING_IDX, False)
    var y = List[Float32](length=len(ids) * D, fill=Float32(-7))
    embedding_host_forward_released(f32_ptr(Int(w.unsafe_ptr())), i32_ptr(Int(ids.unsafe_ptr())), len(ids), cfg,
                                    Int(y.unsafe_ptr()))
    _ = w  # read through its address by the synchronous call above
    _ = ids
    return y^


def backward(ids: List[Int32], dy: List[Float32], plan: Int, var carried: List[Float32]) raises -> List[Float32]:
    """The door's backward: `carried` empty is a fresh dW (accumulate 0), else it is folded onto (accumulate 1)."""
    var accumulate = len(carried) > 0
    var dw: List[Float32]
    if accumulate:
        dw = carried^
    else:
        dw = List[Float32](length=V * D, fill=Float32(-7))  # any contents: the seed stores +0.0
    var cfg = embedding_host_config(V, D, PAD, accumulate)
    embedding_host_backward_released(f32_ptr(Int(dw.unsafe_ptr())), f32_ptr(Int(dy.unsafe_ptr())), ids, len(ids),
                                     cfg, plan, V * D)
    _ = dy
    return dw^


def head(values: List[Int32], count: Int) -> List[Int32]:
    var out = List[Int32](capacity=count)
    for i in range(count):  # driver-loop: over the fixture's positions
        out.append(values[i])
    return out^


def tail(values: List[Int32], first: Int) -> List[Int32]:
    var out = List[Int32](capacity=len(values) - first)
    for i in range(first, len(values)):  # driver-loop: over the fixture's positions
        out.append(values[i])
    return out^


def rows(values: List[Float32], first: Int, last: Int) -> List[Float32]:
    var out = List[Float32](capacity=(last - first) * D)
    for i in range(first * D, last * D):  # driver-loop: over the fixture's dY cells
        out.append(values[i])
    return out^


def reference_fold(ids: List[Int32], dy: List[Float32]) -> List[Float32]:
    """Test oracle: the original's `want` (float32 adds in ascending t, then row PAD stored +0.0)."""
    var want = List[Float32](length=V * D, fill=Float32(0))
    for t in range(len(ids)):  # driver-loop: over the fixture's positions, ascending (the reference order)
        for j in range(D):
            want[Int(ids[t]) * D + j] += dy[t * D + j]
    for j in range(D):
        want[PAD * D + j] = Float32(0)
    return want^


def plan_claims(plan: Int, name: String) raises -> List[Float32]:
    var ids = fixture_i32(EMB_DIR + "ids.i32", T)
    var dy = fixture_f32(EMB_DIR + "dy.f32", T * D)
    var full = backward(ids, dy, plan, List[Float32]())
    var first = backward(head(ids, SPLIT), rows(dy, 0, SPLIT), plan, List[Float32]())
    var carried = backward(tail(ids, SPLIT), rows(dy, SPLIT, T), plan, first^)
    same(carried, full, name + " carried microbatch against the unsplit gradient")
    same(full, reference_fold(ids, dy), name + " backward against the ascending fold")
    for j in range(D):
        require(bitcast[DType.uint32](full[PAD * D + j]) == UInt32(0),
                name + ": padding row cell " + String(j) + " is not +0.0")
    return full^


def run_unit(unit: Int) raises:
    if unit == 0:
        var w = fixture_f32(EMB_DIR + "w.f32", V * D)
        var ids = fixture_i32(EMB_DIR + "ids.i32", T)
        var want = List[Float32](capacity=T * D)
        for t in range(T):  # driver-loop: over the fixture's positions (the gather oracle)
            for j in range(D):
                want.append(w[Int(ids[t]) * D + j])
        same(forward(w, ids), want, "forward against W[ids]")
    elif unit == 1:
        _ = plan_claims(HOST_PLAN_SCAN, "scan")
    elif unit == 2:
        var sort = plan_claims(HOST_PLAN_SORT, "sort")
        same(sort, plan_claims(HOST_PLAN_SCAN, "scan"), "sort bytes against scan bytes")
    else:
        raise Error("unknown embedding host unit")


def test_embedding_host_fold() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    check_cases("embedding_host_fold", EMB_HOST_UNITS, run_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
