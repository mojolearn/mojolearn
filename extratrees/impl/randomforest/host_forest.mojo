# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ExtraTrees HOST forest fits: the reference loops over sklearn's host
splitter (`fit_classification`, `fit_regression`) and the CPU column's
restatement of the device trainer (`fit_forest_exact`, trees across the host
thread pool).

Moved out of `randomforest.mojo` (cpu-gpu-cleanup t-forest) because that
module is the GPU trainer's and every GPU binding reaches it. Nothing here
changed. The callers are the CPU-only install (`bindings/_mojolearn_trees_host
.mojo`, through `extratrees/host_estimator.mojo`) and the checks.
"""

from extratrees.impl.decisiontree.decisiontree import (
    DecisionTreeParams,
    validity_check,
)
from extratrees.impl.decisiontree.flatnode import TreeMetaDataNode
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    DEVICE_MAX_ACC,
    et_identical_bins_wanted,
    n_sampled_cols_for,
    train_tree_exact,
)
from extratrees.impl.decisiontree.batched_levelalgo.host_binned import (
    HostBinTables,
    host_bin_tables,
)
from extratrees.impl.decisiontree.batched_levelalgo.host_builder import (
    train_classification,
    train_regression,
)
from extratrees.impl.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.impl.randomforest.randomforest import (
    BOOTSTRAP_DEFAULT,
    Forest,
    error_checking,
    resolve_n_sampled_rows,
    row_sample_for,
)
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count


def fit_classification(
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    params: DecisionTreeParams,
    n_trees: Int32,
    seed: UInt64,
    bootstrap: Bool = BOOTSTRAP_DEFAULT,
    n_sampled_rows: Int32 = 0,
) raises -> Forest:
    """`randomforest.cuh:155-195`, the tree loop.

    Theirs runs the loop under OpenMP across `n_streams` CUDA streams
    (`:161-167`). Ours is serial, and that is not a deviation to record but a
    consequence of a fact already in the traps register: **Metal has no
    streams**, so their overlap has nothing to implement onto. The trees are
    independent either way — tree `i` reads `x` and writes its own `row_ids`
    and its own tree — so the answer does not depend on the order they run in,
    which is the property that makes the serial form a faithful stand-in
    rather than a different algorithm.
    """
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)
    if n_classes < 1:
        raise Error("n_classes must be >= 1; got " + String(n_classes))

    var n_sampled = resolve_n_sampled_rows(n_rows, bootstrap, n_sampled_rows)
    var forest = Forest(n_classes)
    for tree_id in range(Int(n_trees)):
        # `:169` -- each tree gets its OWN row list, because `train_*`
        # partitions it in place. `:59-67` -- the bootstrap arm is keyed by
        # `(seed, tree_id)` (DEVIATION 460).
        var row_ids = row_sample_for(
            n_rows, bootstrap, n_sampled, seed, Int32(tree_id)
        )
        var dataset = Dataset(
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                x_col_major.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                labels.unsafe_ptr()
            ),
            n_rows,
            n_cols,
            n_sampled,
            n_cols,
            rebind[MutPointer[Int32, MutUntrackedOrigin]](
                row_ids.unsafe_ptr()
            ),
            n_classes,
        )
        # `:180-191` -- `i` is passed as the tree id, which is what makes the
        # trees differ (`:59-62` hashes it into the seed).
        forest.trees.append(
            train_classification(
                dataset, params, Int32(tree_id), seed, n_classes
            )
        )
        _ = row_ids.unsafe_ptr()
    forest.n_trees = n_trees
    return forest^


def fit_regression(
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    params: DecisionTreeParams,
    n_trees: Int32,
    seed: UInt64,
    bootstrap: Bool = BOOTSTRAP_DEFAULT,
    n_sampled_rows: Int32 = 0,
) raises -> Forest:
    """The regression arm of the same loop."""
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)

    var n_sampled = resolve_n_sampled_rows(n_rows, bootstrap, n_sampled_rows)
    var forest = Forest(1)
    for tree_id in range(Int(n_trees)):
        var row_ids = row_sample_for(
            n_rows, bootstrap, n_sampled, seed, Int32(tree_id)
        )
        var dataset = Dataset(
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                x_col_major.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                labels.unsafe_ptr()
            ),
            n_rows,
            n_cols,
            n_sampled,
            n_cols,
            rebind[MutPointer[Int32, MutUntrackedOrigin]](
                row_ids.unsafe_ptr()
            ),
            1,
        )
        forest.trees.append(
            train_regression(dataset, params, Int32(tree_id), seed)
        )
        _ = row_ids.unsafe_ptr()
    forest.n_trees = n_trees
    return forest^


def fit_forest_exact(
    x_col_major: List[Float32],
    labels: List[Float32],
    labels_q: List[Int32],
    n_rows: Int32,
    n_cols: Int32,
    num_outputs: Int32,
    params: DecisionTreeParams,
    n_trees: Int32,
    seed: UInt64,
    is_classification: Bool,
    inv_scale: Float32,
    bootstrap: Bool = BOOTSTRAP_DEFAULT,
    n_sampled_rows: Int32 = 0,
    tree_start: Int = 0,
) raises -> Forest:
    """The forest loop of `fit_classification` / `fit_regression` over
    `train_tree_exact`, the HOST RESTATEMENT OF THE DEVICE TRAINER (the CPU
    training lane, 2026-09-14; the block comment above `train_tree_exact`
    in `builder.mojo`). `labels_q` is the device's label plane: the class
    ids `class_ids_for` derives for a classifier (`num_outputs = n_classes`,
    `inv_scale = 1`), `quantize_labels_host`'s fixed point for a regressor
    (`num_outputs = 1`, `inv_scale = Float32(1 / scale)`). `labels` is the
    float plane the `Dataset` carries beside it; the exact search never
    reads it. The device's own refusal on the class count
    (`train_forest_classification_device`, DEVIATION 172) is restated so a
    fit the device refuses is refused here in the same words.

    `tree_start` is the device arms' global tree ID offset
    (`fit_classification_device` / `fit_regression_device`, the
    `tree_ids` list): tree `i` of this call is tree `tree_start + i` of the
    whole forest in both its row sample and its split key, which is what a
    `parallel_ensemble.fit_forest` shard asks for (lane/cpu-training-par-wave2,
    2026-09-15). At 0 the loop is the one it was."""
    if tree_start < 0 or tree_start + Int(n_trees) > 2147483647:
        raise Error("invalid global tree range")
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)
    if num_outputs < 1:
        raise Error("num_outputs must be >= 1; got " + String(num_outputs))
    if is_classification and Int(num_outputs) > DEVICE_MAX_ACC:
        raise Error(
            "the device score kernel is built for at most "
            + String(DEVICE_MAX_ACC)
            + " classes; got "
            + String(num_outputs)
            + " (DEVIATION 172: shared sizing is comptime here)"
        )
    if len(labels_q) != Int(n_rows) or len(labels) != Int(n_rows):
        raise Error(
            "fit_forest_exact: labels and labels_q must both be n_rows long"
        )
    var n_sampled = resolve_n_sampled_rows(n_rows, bootstrap, n_sampled_rows)
    var forest = Forest(num_outputs)
    var labels_q_p = labels_q.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var x_p = rebind[MutPointer[Float32, MutUntrackedOrigin]](x_col_major.unsafe_ptr())
    # `IDN_ET_BINNED` (fam2-forests, a candidate arm, default OFF): the
    # device's regression forest loop bins X when
    # `et_identical_bins_wanted(n_cols, k)`; the same gate builds the same
    # borders and codes here and the exact search reads them. Off (always,
    # without the define) the tables are one element and unread.
    var bin_tables = HostBinTables()
    if not is_classification and et_identical_bins_wanted(
        Int(n_cols), Int(n_sampled_cols_for(params, n_cols))
    ):
        bin_tables = host_bin_tables(x_col_major, Int(n_rows), Int(n_cols))
    var bins = bin_tables.view()
    var labels_p = rebind[MutPointer[Float32, MutUntrackedOrigin]](labels.unsafe_ptr())
    # THE TREES, ONE TASK PER CONTIGUOUS TREE RANGE (lane/trees-cpu,
    # 2026-09-28). A tree reads X, the label planes and its own seed and
    # writes only its own `TreeMetaDataNode`, so trees may run on different
    # threads with every statement of a tree in its serial order; each task
    # runs in the calling thread's environment (`core/host_parallel.mojo::
    # host_parallelize`; the pool's FTZ+DAZ is DEVIATION 5900), and the trees are appended in
    # tree order after the join: the forest is the serial walk's bytes at
    # every MOJOLEARN_CPU_THREADS.
    var n = Int(n_trees)
    var slots = List[Optional[TreeMetaDataNode[DType.float32]]](capacity=n)
    for _ in range(n):
        slots.append(None)
    var tasks = host_predict_task_count(n)
    var chunk = host_predict_chunk(n, tasks)
    var failed = List[Bool](length=tasks, fill=False)
    var messages = List[String](length=tasks, fill=String(""))

    def _tree_task(task: Int) {mut slots, mut failed, mut messages, imm x_p, imm labels_p, imm labels_q_p, imm n_rows, imm n_cols, imm n_sampled, imm bootstrap, imm seed, imm tree_start, imm num_outputs, imm params, imm is_classification, imm inv_scale, imm chunk, imm n, imm tasks, imm bins}:
        var lo = task * chunk
        var hi = min(lo + chunk, n)
        try:
            for tree_id in range(lo, hi):
                var row_ids = row_sample_for(
                    n_rows, bootstrap, n_sampled, seed, Int32(tree_start + tree_id)
                )
                var dataset = Dataset(
                    x_p,
                    labels_p,
                    n_rows,
                    n_cols,
                    n_sampled,
                    n_cols,
                    rebind[MutPointer[Int32, MutUntrackedOrigin]](
                        row_ids.unsafe_ptr()
                    ),
                    num_outputs,
                )
                slots[tree_id] = train_tree_exact(
                    dataset, labels_q_p, params, Int32(tree_start + tree_id), seed,
                    is_classification, Int(num_outputs), inv_scale, bins,
                )
                _ = row_ids.unsafe_ptr()
        except e:
            failed[task] = True
            messages[task] = String(e)

    if tasks <= 1:
        _tree_task(0)
    else:
        host_parallelize(_tree_task, tasks)
    _ = x_col_major.unsafe_ptr()
    _ = labels.unsafe_ptr()
    _ = labels_q.unsafe_ptr()
    _ = bin_tables.codes.unsafe_ptr()
    _ = bin_tables.q.unsafe_ptr()
    _ = bin_tables.nb.unsafe_ptr()
    # the serial walk raised its first failing tree's error; the lowest
    # failing task holds the lowest trees and stopped at its first
    for k in range(tasks):
        if failed[k]:
            raise Error(messages[k])
    for t in range(n):
        forest.trees.append(slots[t].take())
    forest.n_trees = n_trees
    return forest^
