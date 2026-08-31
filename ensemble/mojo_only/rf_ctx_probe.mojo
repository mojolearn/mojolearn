# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Two random-forest fits in one process, the shape of the RTX 4090 hang.

Written 2026-08-29 after `tools/diag/rtx4090_hang.sh` leg 1
(`bench/results/e1/2026-08-29_163200-runpod-nvidia/diag/d1a_rf_steps.log`):
through the Python binding, the FIRST `RandomForestClassifier.fit` returned
in 1.5 s, `predict` and `predict_proba` in 0.03 s, and the SECOND fit in the
same process never returned. The binding creates one `DeviceContext` per
call; this file asks the same question without Python or the GIL:

    -D RF_SAME_CTX=1   two fits on ONE DeviceContext
    -D RF_TWO_CTX=1    a fit on ctx1, ctx1 released, a fit on a new ctx2
                       (the binding's shape)

Every phase prints a flushed line, so a hang names its phase.

    mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D RF_TWO_CTX=1 \\
        ensemble/mojo_only/rf_ctx_probe.mojo -o /tmp/rf_two && /tmp/rf_two
"""

from std.sys import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI
from ensemble.randomforest import RF_params, fit_forest
from isolation_forest.mojo_only.if_fixture import mix64

comptime DT = DType.float32
comptime LT = DType.int32
comptime ClsObj = ClassificationObjectiveFunction[DT, LT, ClassificationBin]
comptime RF_SAME_CTX = is_defined["RF_SAME_CTX"]()
comptime RF_TWO_CTX = is_defined["RF_TWO_CTX"]()

comptime N_ROWS = 20000
comptime N_COLS = 16


def _say(msg: String, t0: Int):
    print("rfprobe +" + String((perf_counter_ns() - t0) // 1_000_000) + "ms: " + msg, flush=True)


def _params() -> RF_params:
    """`RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7)`
    as the Python layer spells it (its other defaults)."""
    return RF_params(
        n_trees=Int32(16),
        bootstrap=True,
        max_samples=Float32(1.0),
        seed=UInt64(7),
        n_streams=Int32(4),
        tree_params=DecisionTreeParams(
            max_depth=Int32(8),
            max_leaves=Int32(-1),
            max_features=Float32(1.0),
            max_n_bins=Int32(128),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            split_criterion=GINI,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(4096),
        ),
    )


def one_fit(ctx: DeviceContext, tag: String, t0: Int) raises -> Int:
    """The binding's body: pinned host buffers, upload, `fit_forest`,
    synchronize, buffers released in the binding's order."""
    var hx = ctx.enqueue_create_host_buffer[DT](N_ROWS * N_COLS)
    var hy = ctx.enqueue_create_host_buffer[LT](N_ROWS)
    ctx.synchronize()
    # column-major hashed data, a signed rule on columns 3 and 4
    for c in range(N_COLS):
        for r in range(N_ROWS):
            var v = Float32(Int(mix64(r, c, 3) & 0xFFFF)) / Float32(65536.0) * 4.0 - 2.0
            hx.unsafe_ptr().unsafe_store(c * N_ROWS + r, v)
    for r in range(N_ROWS):
        var a = hx.unsafe_ptr().unsafe_load(3 * N_ROWS + r)
        var b = hx.unsafe_ptr().unsafe_load(4 * N_ROWS + r)
        hy.unsafe_ptr().unsafe_store(r, Int32(1) if a + 0.5 * b > 0 else Int32(0))
    var dx = ctx.enqueue_create_buffer[DT](N_ROWS * N_COLS)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var dy = ctx.enqueue_create_buffer[LT](N_ROWS)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()
    var p = _params()
    _say(tag + ": fit_forest begin", t0)
    var forest = fit_forest[ClsObj](ctx, dx, dy, dsw, N_ROWS, N_COLS, 2, p)
    ctx.synchronize()
    var nodes = len(forest.trees[0].sparsetree)
    _say(tag + ": fit_forest returned, tree0 nodes " + String(nodes), t0)
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    _ = forest^
    return nodes


def main() raises:
    var t0 = perf_counter_ns()
    comptime if RF_SAME_CTX:
        _say("SAME_CTX: ctx", t0)
        var ctx = DeviceContext()
        var a = one_fit(ctx, "fit1", t0)
        var b = one_fit(ctx, "fit2", t0)
        _say("SAME_CTX: tree0 nodes equal " + String(a == b), t0)
    comptime if RF_TWO_CTX:
        _say("TWO_CTX: ctx1", t0)
        var ctx1 = DeviceContext()
        var a = one_fit(ctx1, "fit1", t0)
        _ = ctx1^
        _say("TWO_CTX: ctx1 released, creating ctx2", t0)
        var ctx2 = DeviceContext()
        _say("TWO_CTX: ctx2 created", t0)
        var b = one_fit(ctx2, "fit2", t0)
        _say("TWO_CTX: tree0 nodes equal " + String(a == b), t0)
    print("rfprobe DONE", flush=True)
