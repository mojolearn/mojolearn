# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Compile root of the executor's tree-builder ops (lanes t1-tree-builder-ops
# and t1b-tree-leaf-and-node-ops; the forest rows 11-12 compile through the
# RandomForest binding,
# docs/plans/EXECUTOR_T1_TREE_OPS.md) until lane T2 moves the GBDT round onto
# them: every op except TREE_OP_SEARCH (reached through the GBDT binding's
# fit) instantiated once on a tiny problem, so `mojo build
# checks/executor_t1_tree_ops_compile.mojo` type-checks every kernel for the
# target. Running it enqueues an 8-row problem and prints a few words; it is
# not a verifier.
from max.gpu.host import DeviceContext

from autograd.program_loop_device import LoopRegisters
from core.device_zero import enqueue_fill
from core.step_gate import step_fail_joins_loop, step_gate_open_word
from gbdt.gpu_util.kernel.bootstrap import (
    BOOTSTRAP_KERNEL_BAYESIAN,
    bootstrap_grid_blocks,
    create_bootstrap_seeds,
    tree_op_bootstrap,
    tree_op_seed_snapshot,
)
from gbdt.methods.doc_parallel_boosting import (
    LeafEstimateOpState,
    make_test_arm,
    tree_op_heldout_loss,
    tree_op_leaf_estimate,
)
from gbdt.options.catboost_options import LEAF_ESTIMATION_NEWTON
from gbdt.methods.kernel.tree_gate_walk import (
    TreeRecords,
    tree_op_apply,
    tree_op_gate_walk,
    tree_op_leaf_bins,
    tree_op_tree_records,
    tree_structure_words,
)
from gbdt.methods.leaves_estimation.doc_parallel_leaves_estimator import (
    DeviceLeafPartitioner,
    tree_op_partition,
)
from gbdt.overfitting_detector.overfitting_detector import OD_ITER, make_overfitting_detector
from gbdt.overfitting_detector.od_device import OdDeviceState, od_op_init, od_op_step
from gbdt.targets.kernel.pointwise_targets import OBJECTIVE_LOGLOSS, OBJECTIVE_RMSE
from gbdt.tree_ops import N_TREE_OPS, TREE_OP_FIRST, tree_op_name


def main() raises:
    var ctx = DeviceContext()
    var n = 8
    var max_depth = 2
    var n_features = 1
    var gate = step_gate_open_word(ctx)
    var regs = LoopRegisters(ctx)
    regs.reset(ctx)

    # bootstrap
    var seeds = create_bootstrap_seeds(ctx, UInt64(7))
    var stats = ctx.enqueue_create_buffer[DType.float32](2 * n)
    enqueue_fill(ctx, stats, Float32(1.0))
    var mags = ctx.enqueue_create_buffer[DType.float32](2 * bootstrap_grid_blocks(n))
    var seed_snap = ctx.enqueue_create_buffer[DType.uint64](len(seeds))
    tree_op_seed_snapshot(ctx, regs.words, seeds, seed_snap)
    tree_op_bootstrap(ctx, regs.words, BOOTSTRAP_KERNEL_BAYESIAN, seeds, stats, n, Float32(1.0), mags, True, 2)

    # gate walk, records, bins, partition, apply
    var winners = ctx.enqueue_create_buffer[DType.uint32](2 * max_depth)
    enqueue_fill(ctx, winners, UInt32(0))
    var table = ctx.enqueue_create_buffer[DType.uint32](4 * n_features)
    enqueue_fill(ctx, table, UInt32(0))
    var structure = ctx.enqueue_create_buffer[DType.int32](tree_structure_words(max_depth))
    tree_op_gate_walk(ctx, gate, winners, table, max_depth, structure)
    step_fail_joins_loop(ctx, regs.words, structure, 1)
    var records = TreeRecords(ctx, max_depth)
    tree_op_tree_records(ctx, regs.words, structure, table, records)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n * n_features)
    enqueue_fill(ctx, cindex, UInt32(0))
    var parts = DeviceLeafPartitioner(ctx, n, 1 << max_depth)
    tree_op_leaf_bins(ctx, regs.words, structure, records, cindex, n, parts.bins)
    var p_status = ctx.enqueue_create_buffer[DType.int32](1)
    tree_op_partition(ctx, regs.words, parts, n, 1 << max_depth, p_status)
    step_fail_joins_loop(ctx, regs.words, p_status, 1)
    var leaves = ctx.enqueue_create_buffer[DType.float32](1 << max_depth)
    enqueue_fill(ctx, leaves, Float32(0.5))
    var cursor = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, cursor, Float32(0.0))

    # leaf estimation (lane t1b-tree-leaf-and-node-ops): a Logloss Newton
    # walk with a two-evaluation line search, weighted
    var est_state = LeafEstimateOpState(ctx, n, 1 << max_depth)
    var l_targets = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, l_targets, Float32(1.0))
    var l_weights = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, l_weights, Float32(1.0))
    var leaf_out = ctx.enqueue_create_buffer[DType.float32](1 << max_depth)
    var leaf_scaled = ctx.enqueue_create_buffer[DType.float32](1 << max_depth)
    var l_status = ctx.enqueue_create_buffer[DType.int32](1)
    tree_op_leaf_estimate(
        ctx, regs.words, est_state, parts, l_targets, l_weights, True, cursor,
        OBJECTIVE_LOGLOSS, Float32(0.5), Float32(0.5), Float32(0.5), Float32(3.0),
        LEAF_ESTIMATION_NEWTON, 2, 2, Float32(0.1), leaf_out, leaf_scaled, l_status,
    )
    step_fail_joins_loop(ctx, regs.words, l_status, 1)
    tree_op_apply(ctx, regs.words, structure, records, cindex, leaf_scaled, n, cursor, 1, 0)
    _ = leaves^

    # held-out loss and the detector
    var t_cindex = ctx.enqueue_create_buffer[DType.uint32](n * n_features)
    enqueue_fill(ctx, t_cindex, UInt32(0))
    var t_targets = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, t_targets, Float32(1.0))
    var test = make_test_arm(ctx, n, t_cindex^, t_targets^, 1, 2, max_depth)
    var losses = ctx.enqueue_create_buffer[DType.uint64](4)
    tree_op_heldout_loss(
        ctx, regs.words, test, OBJECTIVE_RMSE, 1, Float32(0.5), Float32(0.5), 1, losses, 1,
    )
    var det = make_overfitting_detector(OD_ITER, False, 0.0, 2, True)
    var od = OdDeviceState(ctx)
    od_op_init(ctx, od, det)
    od_op_step(ctx, regs.words, od, losses, 1)

    regs.read(ctx)
    print("tree ops:", N_TREE_OPS, "first", tree_op_name(TREE_OP_FIRST), "done", regs.word(0), "fail", regs.failure())
