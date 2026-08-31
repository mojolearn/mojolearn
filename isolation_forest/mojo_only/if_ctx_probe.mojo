# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DeviceContext lifetimes around one isolation-forest fit: the 4090 hang.

Written 2026-08-29 after `tools/diag/rtx4090_hang.sh` leg 1: `if_hang_probe`
(ONE DeviceContext, fit and score) passed on the RTX 4090 with the M4's
checksums, while the Python binding's `IsolationForest.fit` hung with the
GPU idle and every host thread in futex wait. The binding's path,
`isolation_forest/estimator.mojo::iforest_run_host`, differed from the probe
in exactly one thing: it created a SECOND DeviceContext
(`IsolationForestEstimator.__init__` built its empty model on
`DeviceContext()` while `iforest_run_host`'s own `ctx` was alive, and `fit`
then replaced that model's buffers with ones on the other context). This
file takes the lifetimes apart:

    -D T1=1   ctx1 and ctx2 alive; fit on ctx2
    -D T2=1   fit on ctx1; ctx1 released; fit on a new ctx2
    -D T3=1   ctx1 and ctx2 alive; fit on ctx1, ctx2 never used
    -D T4=1   the binding's shape before the fix: model created on ctx B,
              fitted on ctx A, so B's buffers are freed while A works

    mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_IF_DIAG_TRACE=1 -D T4=1 \\
        isolation_forest/mojo_only/if_ctx_probe.mojo -o /tmp/t4 && /tmp/t4
"""

from std.sys import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from isolation_forest.mojo_only.if_fixture import mix64, to_column_major
from isolation_forest.ported.isolation_forest.isolation_forest import (
    IF_params,
    IFLaunchKnobs,
    IsolationForestModel,
    fit,
    score_samples,
)

comptime MODE_T1 = is_defined["T1"]()
comptime MODE_T2 = is_defined["T2"]()
comptime MODE_T3 = is_defined["T3"]()
comptime MODE_T4 = is_defined["T4"]()


def _say(msg: String, t0: Int):
    print("ctxprobe +" + String((perf_counter_ns() - t0) // 1_000_000) + "ms: " + msg, flush=True)


def one_fit(
    ctx: DeviceContext,
    mut model: IsolationForestModel,
    x: List[Float32],
    x_col: List[Float32],
    n: Int,
    d: Int,
    t0: Int,
) raises:
    var params = IF_params(4, 32, -1, -1, False, 5)
    var trace = IdentityTrace.to_path("")
    _say("fit begin", t0)
    fit(ctx, model, x_col, n, d, params, trace, IFLaunchKnobs.default())
    _say("fit returned, n_nodes0=" + String(model.tree_n_nodes_host[0]), t0)
    var s = score_samples(ctx, model, x, n, d, trace, IFLaunchKnobs.default())
    _say("score returned, len " + String(len(s)), t0)


def main() raises:
    var t0 = perf_counter_ns()
    var n = 64
    var d = 4
    var x = List[Float32]()
    for i in range(n):
        for j in range(d):
            x.append(Float32(Int(mix64(i, j, 17) & 0xFFFF)) / Float32(65536.0) * 4.0 - 2.0)
    var x_col = to_column_major(x, n, d)
    comptime if MODE_T1:
        _say("T1: ctx1", t0)
        var ctx1 = DeviceContext()
        _say("T1: ctx2", t0)
        var ctx2 = DeviceContext()
        var model = IsolationForestModel(ctx2)
        _say("T1: fit on ctx2 with ctx1 alive", t0)
        one_fit(ctx2, model, x, x_col, n, d, t0)
        _ = ctx1^
    comptime if MODE_T2:
        _say("T2: ctx1 fit", t0)
        var ctx1 = DeviceContext()
        var model1 = IsolationForestModel(ctx1)
        one_fit(ctx1, model1, x, x_col, n, d, t0)
        _ = model1^
        _ = ctx1^
        _say("T2: ctx1 released, ctx2 fit", t0)
        var ctx2 = DeviceContext()
        var model2 = IsolationForestModel(ctx2)
        one_fit(ctx2, model2, x, x_col, n, d, t0)
    comptime if MODE_T3:
        _say("T3: ctx1", t0)
        var ctx1 = DeviceContext()
        _say("T3: ctx2 (never used)", t0)
        var ctx2 = DeviceContext()
        var model = IsolationForestModel(ctx1)
        _say("T3: fit on ctx1 with ctx2 alive", t0)
        one_fit(ctx1, model, x, x_col, n, d, t0)
        _ = ctx2^
    comptime if MODE_T4:
        _say("T4: ctx A", t0)
        var ctx_a = DeviceContext()
        _say("T4: model on a fresh DeviceContext B (the estimator's old __init__)", t0)
        var model = IsolationForestModel(DeviceContext())
        _say("T4: fit on ctx A, B's eight buffers replaced during fit", t0)
        one_fit(ctx_a, model, x, x_col, n, d, t0)
    print("ctxprobe DONE", flush=True)
