# SPDX-License-Identifier: Apache-2.0
"""Typed prediction and output-copy loops moved word for word from
bindings/_mojolearn_forest_host.mojo. The shell unpacks the Python arguments,
releases the GIL and calls these."""
from checks.numerics import identical_exp64
from core.forest_host_predict import (
    FOREST_HOST_SABOTAGE,
    et_host_predict,
    et_host_trees,
    rf_host_predict,
    rf_host_trees,
)
from core.gbdt_host_ctr import gbdt_host_expand_ctr
from core.gbdt_host_predict import gbdt_host_predict


def forest_host_vote_into(
    is_rf: Bool,
    rows: List[Float32],
    offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    colid_p: MutPointer[Int32, MutUntrackedOrigin],
    quesval_p: MutPointer[Float32, MutUntrackedOrigin],
    left_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    op: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int, n_cols: Int, n_trees: Int, num_outputs: Int, n_nodes: Int,
) raises:
    """The RF or ET host vote, written to the caller's `op`."""
    var out = List[Float32](length=n_rows * num_outputs, fill=Float32(0.0))
    if is_rf:
        var trees = rf_host_trees(
            offsets_p, colid_p, quesval_p, left_p, leaves_p,
            n_trees, n_nodes, n_cols, num_outputs,
        )
        rf_host_predict(trees, rows, n_rows, n_cols, n_trees, num_outputs, out)
    else:
        var trees = et_host_trees(
            offsets_p, colid_p, quesval_p, left_p, leaves_p,
            n_trees, n_nodes, n_cols, num_outputs,
        )
        et_host_predict(trees, rows, n_rows, n_cols, n_trees, num_outputs, out)
    for i in range(n_rows * num_outputs):
        op[i] = out[i]


def forest_host_gbdt_raw_into(
    rows: List[Float32], n_rows: Int, n_cols: Int,
    border_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    borders_p: MutPointer[Float32, MutUntrackedOrigin],
    fold_counts_p: MutPointer[Int32, MutUntrackedOrigin],
    one_hot_p: MutPointer[Int32, MutUntrackedOrigin],
    nan_p: MutPointer[Int32, MutUntrackedOrigin],
    tree_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    split_feature_p: MutPointer[Int32, MutUntrackedOrigin],
    split_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    split_take_bin_p: MutPointer[Int32, MutUntrackedOrigin],
    node_left_p: MutPointer[Int32, MutUntrackedOrigin],
    node_right_p: MutPointer[Int32, MutUntrackedOrigin],
    leaf_offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    n_trees: Int, dim: Int, non_symmetric: Bool, n_splits: Int,
    n_leaf_values: Int, bias_value: Float64,
    op: MutPointer[Float32, MutUntrackedOrigin],
    row_major: Bool,
) raises:
    """The GBDT raw approxes, `n_rows * dim` values, written to `op`."""
    var out = List[Float32](length=n_rows * dim, fill=Float32(0.0))
    gbdt_host_predict(
        rows, n_rows, n_cols,
        border_offsets_p, borders_p, fold_counts_p, one_hot_p, nan_p,
        tree_offsets_p, split_feature_p, split_bin_p, split_take_bin_p,
        node_left_p, node_right_p, leaf_offsets_p, leaves_p,
        n_trees, dim, non_symmetric, n_splits, n_leaf_values, bias_value,
        out, row_major=row_major,
    )
    for i in range(n_rows * dim):
        op[i] = out[i]


def forest_host_gbdt_expand_ctr_into(
    entry: String,
    x: List[Float32], n_rows: Int, n_cols: Int,
    ctr_ints: List[Int32], ctr_floats: List[Float32], n_ctr: Int,
    tensor_ints: List[Int32], tensor_floats: List[Float32], n_tensor: Int,
    counts: List[Int32], border_offsets: List[Int32], borders: List[Float32],
    one_hot: List[Int32],
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """The CTR expansion of the raw columns, written to `op`."""
    var out = gbdt_host_expand_ctr(
        x, n_rows, n_cols, ctr_ints, ctr_floats, n_ctr, tensor_ints,
        tensor_floats, n_tensor, counts, border_offsets, borders, one_hot,
    )
    if len(out) != n_rows * n_cols:
        raise Error(entry + ": the expansion wrote " + String(len(out)) + " values")
    for i in range(n_rows * n_cols):
        op[i] = out[i]


def forest_host_gbdt_sigmoid_into(
    rp: MutPointer[Float64, MutUntrackedOrigin],
    op: MutPointer[Float64, MutUntrackedOrigin],
    count: Int,
):
    """`out[i] = 1 / (1 + exp(-raw[i]))` in double through `identical_exp64`."""
    for i in range(count):
        var r = rp.unsafe_load(i)
        op.unsafe_store(i, 1.0 / (1.0 + identical_exp64(-r)))


def forest_host_gbdt_sigmoid_pair_into(
    rp: MutPointer[Float64, MutUntrackedOrigin],
    op: MutPointer[Float64, MutUntrackedOrigin],
    count: Int,
):
    """The two probability columns `1 - p`, `p` (swapped under sabotage)."""
    for i in range(count):
        var r = rp.unsafe_load(i)
        var p = 1.0 / (1.0 + identical_exp64(-r))
        comptime if FOREST_HOST_SABOTAGE:
            op.unsafe_store(2 * i, p)
            op.unsafe_store(2 * i + 1, 1.0 - p)
        else:
            op.unsafe_store(2 * i, 1.0 - p)
            op.unsafe_store(2 * i + 1, p)
