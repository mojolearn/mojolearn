# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The RF early-exit device loop grows the host queue's tree, bit for bit.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
        -D MOJOLEARN_IDN_RF_DEVICE_LOOP_EARLY_EXIT=1 -I . \
        checks/native_tests/rf_device_loop_early_exit_gpu.mojo

`-D MOJOLEARN_IDN_RF_DEVICE_LOOP_EARLY_EXIT` (or the plain device loop,
`-D MOJOLEARN_IDN_RF_DEVICE_LOOP`) must be set: without a device-loop form both
fits take the host queue and the comparison proves nothing, so the suite
refuses to run.

The fixture makes the sampling rounds matter: 16 columns, `max_features`
0.25 (k = 4, four rounds), and only columns 0-3 carry signal; columns 4-15
are constant. A node whose round-0 draw is all constant columns finds no
split and is retried in round 1 (and so on), while a batch whose nodes all
split in round 0 has nothing to retry, which is where the early exit stops.
Two builders with the same parameters, seed and tree id grow one tree each:
one on the host queue (`force_host_queue`), one on the device loop. Every
node word (column, threshold bits, metric bits, left child, instance count)
and every leaf value bit must agree.
"""

from max.gpu.host import DeviceContext
from std.memory import bitcast

from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
)
from ensemble.decisiontree.batched_levelalgo.builder import (
    Builder,
    IDN_RF_DEVICE_LOOP_EARLY_EXIT,
    RF_DEVICE_QUEUE,
)
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.quantiles import (
    compute_quantiles,
    Quantiles,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI

comptime DT = DType.float32
comptime LT = DType.int32

comptime N_ROWS = 3000
comptime N_COLS = 16
comptime N_INFORMATIVE = 4
comptime N_CLASSES = 4
comptime MAX_FEATURES = Float32(0.25)


def _hash01(i: Int, j: Int) -> Float32:
    """A hashed value in [0, 1), distinct per (row, column)."""
    var h = UInt32(i * 2654435761 + j * 40503 + 17)
    h ^= h >> 15
    h = h * UInt32(2246822519)
    h ^= h >> 13
    return Float32(Int(h % UInt32(1000))) / Float32(1000.0)


struct TreeWords(Movable):
    var colid: List[Int32]
    var ques: List[UInt32]
    var metric: List[UInt32]
    var left: List[Int64]
    var count: List[Int32]
    var leaf: List[UInt32]
    var loop_round: Int

    def __init__(out self):
        self.colid = List[Int32]()
        self.ques = List[UInt32]()
        self.metric = List[UInt32]()
        self.left = List[Int64]()
        self.count = List[Int32]()
        self.leaf = List[UInt32]()
        self.loop_round = 0


def _grow(ctx: DeviceContext, force_host_queue: Bool) raises -> TreeWords:
    var params = DecisionTreeParams(
        max_depth=Int32(8),
        max_leaves=Int32(-1),
        max_features=MAX_FEATURES,
        max_n_bins=Int32(16),
        min_samples_leaf=Int32(1),
        min_samples_split=Int32(2),
        split_criterion=GINI,
        min_impurity_decrease=Float32(0.0),
        max_batch_size=Int32(16),
    )

    # Columns 0-3: hashed values; the class is two thresholds on them plus
    # a hashed 15% flip, so the tree keeps splitting to depth 8. Columns
    # 4-15: constant, so no split exists on them and a node that drew only
    # those is retried in the next sampling round.
    var hx = ctx.enqueue_create_host_buffer[DT](N_ROWS * N_COLS)
    var hy = ctx.enqueue_create_host_buffer[LT](N_ROWS)
    for i in range(N_ROWS):  # fixture setup on the host, not product code
        for j in range(N_COLS):
            var v = Float32(1.0)
            if j < N_INFORMATIVE:
                v = _hash01(i, j)
            hx.unsafe_ptr().unsafe_store(j * N_ROWS + i, v)
        var cls = 0
        if _hash01(i, 0) + Float32(0.3) * _hash01(i, 2) > Float32(0.6):
            cls += 1
        if _hash01(i, 1) - Float32(0.2) * _hash01(i, 3) > Float32(0.4):
            cls += 2
        if _hash01(i, 99) < Float32(0.15):
            cls = (cls + 1) % N_CLASSES
        hy.unsafe_ptr().unsafe_store(i, Int32(cls))
    var dx = ctx.enqueue_create_buffer[DT](N_ROWS * N_COLS)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var dy = ctx.enqueue_create_buffer[LT](N_ROWS)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())

    var hr = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    for i in range(N_ROWS):
        hr.unsafe_ptr().unsafe_store(i, Int32(i))
    var dr = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    ctx.enqueue_copy(dst_buf=dr, src_ptr=hr.unsafe_ptr())

    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()

    var qr = compute_quantiles(
        ctx, dx, Int(params.max_n_bins), N_ROWS, N_COLS, seed=UInt64(7)
    )
    var quantiles = Quantiles[DT](
        qr.quantiles_array.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        qr.n_bins_array.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
    )

    var dataset = DatasetView[DT, LT](
        dx.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dy.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dsw.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(N_ROWS),
        Int64(N_COLS),
        Int64(1),
        Int64(N_ROWS),
        Int32(N_ROWS),
        Int32(N_COLS),
        dr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int32(N_CLASSES),
        False,
        dx.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8](),
        False,
        False,
    )

    var builder = Builder[
        ClassificationObjectiveFunction[DT, LT, ClassificationBin]
    ](
        ctx, params, Int32(0), UInt64(42), N_ROWS, N_COLS,
        Int32(N_CLASSES),
    )
    builder.force_host_queue = force_host_queue
    var tree = builder.train(ctx, dataset, quantiles)

    var words = TreeWords()
    words.loop_round = builder.loop_round
    for i in range(len(tree.sparsetree)):
        ref n = tree.sparsetree[i]
        words.colid.append(n.ColumnId())
        words.ques.append(bitcast[DType.uint32](n.QueryValue()))
        words.metric.append(bitcast[DType.uint32](n.BestMetric()))
        words.left.append(n.LeftChildId())
        words.count.append(n.InstanceCount())
    for i in range(len(tree.vector_leaf)):
        words.leaf.append(bitcast[DType.uint32](tree.vector_leaf[i]))

    _ = qr^
    _ = dx^
    _ = dy^
    _ = dr^
    _ = dsw^
    _ = hx^
    _ = hy^
    _ = hr^
    _ = builder^
    return words^


def _diff_u32(name: String, a: List[UInt32], b: List[UInt32]) raises:
    if len(a) != len(b):
        raise Error(name + ": length " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error(name + " differs at " + String(i))


def _diff_i32(name: String, a: List[Int32], b: List[Int32]) raises:
    if len(a) != len(b):
        raise Error(name + ": length " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error(name + " differs at " + String(i))


def _diff_i64(name: String, a: List[Int64], b: List[Int64]) raises:
    if len(a) != len(b):
        raise Error(name + ": length " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error(name + " differs at " + String(i))


def run_checks() raises:
    comptime if not RF_DEVICE_QUEUE:
        raise Error(
            "rf_device_loop_early_exit: build with"
            " -D MOJOLEARN_IDN_RF_DEVICE_LOOP_EARLY_EXIT=1 (or"
            " -D MOJOLEARN_IDN_RF_DEVICE_LOOP=1); without a device-loop form"
            " both fits take the host queue and nothing is compared"
        )
    var ctx = DeviceContext()
    var host = _grow(ctx, True)
    var dev = _grow(ctx, False)
    if host.loop_round != 0:
        raise Error("the host-queue fit touched the device loop")
    comptime if IDN_RF_DEVICE_LOOP_EARLY_EXIT:
        if dev.loop_round < 1:
            raise Error(
                "the device fit did not take the early-exit loop (the"
                " fixture must have more than one sampling round)"
            )
    var splits = 0
    var deep_cols = 0
    for i in range(len(host.colid)):
        if host.colid[i] >= Int32(0):
            splits += 1
            if host.colid[i] >= Int32(N_INFORMATIVE):
                deep_cols += 1
    if splits < 8:
        raise Error(
            "fixture too small: the host tree has only "
            + String(splits)
            + " splits"
        )
    if deep_cols != 0:
        raise Error("a split landed on a constant column: fixture is wrong")
    _diff_i32("colid", host.colid, dev.colid)
    _diff_u32("quesval bits", host.ques, dev.ques)
    _diff_u32("best_metric bits", host.metric, dev.metric)
    _diff_i64("left child", host.left, dev.left)
    _diff_i32("instance count", host.count, dev.count)
    _diff_u32("leaf bits", host.leaf, dev.leaf)
    print(
        "rf_device_loop_early_exit: OK,",
        len(host.colid),
        "nodes,",
        splits,
        "splits, identical bits (early exit:",
        IDN_RF_DEVICE_LOOP_EARLY_EXIT,
        ")",
    )
