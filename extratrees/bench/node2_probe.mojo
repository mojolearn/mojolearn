# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Why did (seed 2, tree 0, node 2) on year refuse to split? Ask the cells.

    pixi run mojo run -I . extratrees/bench/node2_probe.mojo

Reconstructs the exact node the depth probe found leafed at depth 1 --
148,196 rows, 90 non-constant columns, sums in-slot -- and runs ONE
`search_batch_regression` on it with the exact fitted keys (seed 2, tree 0,
node id 2, k = 90). Prints the winner's fields; if the winner is the
invalid sentinel, re-runs cellwise through the host oracle
`node_feature_score_host` (device_draw=True) on a handful of columns to
name the per-cell status. Diagnostic only; not part of any suite.
"""

from max.gpu.host import DeviceContext

from extratrees.bench.bench_data import read_column_prefix, read_f32
from extratrees.estimator import quantize_labels
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_MSE,
    DecisionTreeParams,
)
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    make_level_workspace,
    search_batch_regression,
    upload_dataset,
    DEVICE_TPB,
    n_sampled_cols_for,
    PhaseClock,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    FeatureRange,
    node_feature_min_max,
    node_feature_score_host,
)
from extratrees.checks.pcg_rng import key_for


def main() raises:
    var data_dir = String("/Users/andrewhendel/.cache/mojolearn")
    var n_rows = 515345
    var n_all = 91
    var n_feat = 90
    var yfull = read_f32(data_dir + "/year_y.f32")
    _ = yfull
    var x = read_column_prefix(
        data_dir + "/year_Xcol.f32", n_rows, n_rows, n_all
    )
    # target = column 0; features = columns 1..90 (column-major slices)
    var target = List[Float32](length=n_rows, fill=Float32(0.0))
    for r in range(n_rows):
        target[r] = x[r]
    var xcols = List[Float32](length=n_rows * n_feat, fill=Float32(0.0))
    for c in range(n_feat):
        for r in range(n_rows):
            xcols[c * n_rows + r] = x[(c + 1) * n_rows + r]

    # The fitted root: feature 54, threshold from its exact bits.
    var q = Float32(from_bits=UInt32(1097851860))
    var root_col = 54
    var left = List[Int32]()
    var right = List[Int32]()
    for r in range(n_rows):
        var v = xcols[root_col * n_rows + r]
        if v <= q:
            left.append(Int32(r))
        else:
            right.append(Int32(r))
    print("root split col", root_col, "q", q, ": left", len(left),
          "right", len(right))

    var ql = quantize_labels(target, Int32(n_rows))
    print("scale", ql[1])
    var ctx = DeviceContext()
    var dev = upload_dataset(
        ctx, xcols, ql[0], Int32(n_rows), Int32(n_feat), 1
    )

    # Row layout: left block then right block; node 2 = the right block.
    var row_ids = List[Int32]()
    for i in range(len(left)):
        row_ids.append(left[i])
    var base = len(row_ids)
    for i in range(len(right)):
        row_ids.append(right[i])
    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var h_row_ids = ctx.enqueue_create_host_buffer[DType.int32](n_rows)
    ctx.synchronize()
    for i in range(n_rows):
        h_row_ids.unsafe_ptr().unsafe_store(i, row_ids[i])
    ctx.enqueue_copy(dst_buf=d_row_ids, src_ptr=h_row_ids.unsafe_ptr())
    ctx.synchronize()

    var params = DecisionTreeParams()
    params.max_depth = 8
    params.split_criterion = CRITERION_MSE
    params.max_features = 1.0
    var k = n_sampled_cols_for(params, Int32(n_feat))
    print("k =", k)
    var ws = make_level_workspace(
        ctx, Int(params.max_batch_size), Int32(n_rows), Int32(n_feat), 1,
        Int(k), DEVICE_TPB,
    )
    var items = List[NodeWorkItem]()
    items.append(
        NodeWorkItem(Int32(2), Int32(1), InstanceRange(Int32(base),
        Int32(len(right))))
    )
    var trees = List[Int32]()
    trees.append(Int32(0))
    var clock = PhaseClock(False)
    var found = search_batch_regression(
        ctx, ws, dev, d_row_ids, items, Int(k), params, Int32(n_rows),
        Int32(n_feat), trees, UInt64(2), True, List[Int32](), False, clock,
    )
    var s = found[0][0]
    print(
        "winner: colid", s.colid, "quesval", s.quesval, "metric",
        s.best_metric_val, "n_left", s.n_left, "any_nonconst",
        found[1][0],
    )

    # Host-oracle cellwise on the first 8 columns, to name statuses.
    for c in range(8):
        var extent = FeatureRange(Float32(0), Float32(0), Int32(0))
        # host min/max over the node's rows for column c
        var mn = Float32(3.0e38)
        var mx = Float32(-3.0e38)
        for i in range(len(right)):
            var v = xcols[c * n_rows + Int(right[i])]
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        extent = FeatureRange(mn, mx, Int32(0))
        var cand = node_feature_score_host(
            xcols.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            row_ids.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            ql[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            n_rows,
            base,
            len(right),
            c,
            extent,
            key_for(UInt64(2), UInt32(0), UInt32(2), UInt32(c)),
            1,
            False,
            1,
        )
        print(
            "col", c, "status", cand.status, "thresh", cand.threshold,
            "n_left", cand.n_left, "n_total", cand.n_total, "num",
            cand.gini_num, "den", cand.gini_den,
        )
    _ = ws^
    _ = d_row_ids^
    _ = h_row_ids^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
