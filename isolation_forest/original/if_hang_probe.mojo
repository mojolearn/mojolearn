# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The smallest program that reaches `build_isolation_trees_global_kernel`.

Written 2026-08-29 for the RTX 4090 hang (`IsolationForest.fit` never
returned on four RunPod 4090 hosts while the same source passed on an H100,
an M4 and an MI325X). `if_check.mojo` reaches the same kernel only after
two xorwow gates, the refusal gate and the host oracle, and it compiles in
about six minutes on a rented box; this file compiles the fit path alone
so `tools/diag/rtx4090_hang.sh` can build it once per bisect guard
(isolation_tree_builder.mojo, `MOJOLEARN_IF_DIAG_*`) inside one lease.

Shape: 4 trees, n = 64, d = 4, max_samples = 32, the `if_check` first-tree
shape that already stalls, so size is not the variable. Every phase prints
a flushed line before and after, so a hang names its phase; under
`-D MOJOLEARN_IF_DIAG_TRACE=1` the launch itself is split into
"enqueue" (where MAX compiles the kernel) and "synchronize" (where a
resident kernel would sit).

    mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_IF_DIAG_TRACE=1 \\
        isolation_forest/original/if_hang_probe.mojo
"""

from std.memory import bitcast
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from isolation_forest.original.if_fixture import mix64, to_column_major
from isolation_forest.derived.isolation_forest.isolation_forest import (
    IF_params,
    IFLaunchKnobs,
    IsolationForestModel,
    fit,
    read_i32,
    score_samples,
)
from original.numerics import numeric_mode_name


def _say(msg: String, t0: Int):
    var ms = (perf_counter_ns() - t0) // 1_000_000
    print("probe +" + String(ms) + "ms: " + msg, flush=True)


def main() raises:
    var t0 = perf_counter_ns()
    _say("start [" + numeric_mode_name() + "]", t0)
    var n = 64
    var d = 4
    var x = List[Float32]()
    for i in range(n):
        for j in range(d):
            var h = mix64(i, j, 17)
            x.append(Float32(Int(h & 0xFFFF)) / Float32(65536.0) * 4.0 - 2.0)
    var x_col = to_column_major(x, n, d)
    var params = IF_params(4, 32, -1, -1, False, 5)
    _say("data built, creating DeviceContext", t0)
    var ctx = DeviceContext()
    _say("DeviceContext created, fit begin", t0)
    var trace = IdentityTrace.to_path("")
    var model = IsolationForestModel(ctx)
    fit(ctx, model, x_col, n, d, params, trace, IFLaunchKnobs.default())
    _say("fit returned", t0)
    var nodes = String("")
    var built = True
    for t in range(params.n_estimators):
        nodes += String(model.tree_n_nodes_host[t]) + " "
        if model.tree_n_nodes_host[t] < 1:
            built = False
    print("probe n_nodes per tree: " + nodes, flush=True)
    if not built:
        # The poison fill (0) is still in the node arrays, so the kernel
        # wrote nothing (a truncated bisect build, or a launch that never
        # ran). Scoring such a forest self-loops in `traverse_global_tree`
        # on node 0 (measured on the M4 with MOJOLEARN_IF_DIAG_ENTRY_RETURN),
        # which would put a hang in the wrong phase. Stop here, by name.
        print("probe FOREST NOT BUILT (n_nodes poison): scoring skipped", flush=True)
        print("probe DONE", flush=True)
        return
    var feat = read_i32(ctx, model.node_feature, params.n_estimators * model.max_nodes_per_tree)
    var acc: UInt32 = 0
    for i in range(len(feat)):
        acc = acc * 31 + bitcast[DType.uint32](feat[i])
    print("probe feat checksum: " + String(acc), flush=True)
    _say("score_samples begin", t0)
    var scores = score_samples(ctx, model, x, n, d, trace, IFLaunchKnobs.default())
    var sacc: UInt32 = 0
    for i in range(len(scores)):
        sacc = sacc * 31 + bitcast[DType.uint32](scores[i])
    _say("score_samples returned, checksum " + String(sacc), t0)
    print("probe DONE", flush=True)
