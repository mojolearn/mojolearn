# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU saved-forest operations as typed native runs, and their one native
entry in the host table (lane calltable-host-misc, 2026-11-11).

Each `forest_host_<name>_run` is the body the Python export
`forest_host_<name>` of `bindings/_mojolearn_forest_host.mojo` ran, moved
here verbatim after its arguments were unpacked into plain words; the Python
export now only unpacks and calls it (it stays for the Python callers: the
host surface, the legacy reference tests and the identity tools). Native
callers in other extension modules (`bindings/host_tree_native.mojo`) reach
the same runs through `forest_host_entry_v1`, published at module init in
`core/host_entry_table.mojo` family `HOST_TABLE_FOREST_HOST`, with no Python
in between. HOST ONLY: nothing here imports the device side. No kernel, fold
order or route changed.

The runs are synchronous calls over caller-owned buffers; the vote, grove and
GBDT bodies parallelize inside. The dispatch is a fixed comparison chain on
the operation code, no data work.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.forest_host_groves import HostGroveForest, HostGroveRegistry
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from core.host_entry_table import (
    HostEntryKey, host_table_publish, HOST_TABLE_FOREST_HOST, FOREST_HOST_ABI, FH_OP_COUNT,
    FH_RF_PREDICT_PROBA, FH_RF_PREDICT_REG, FH_ET_PREDICT, FH_GROVES_PREPARE, FH_GROVES_PREDICT,
    FH_GBDT_EXPAND_CTR,
)
from core.mojolearn_forest_host_impl import (
    forest_host_gbdt_expand_ctr_into,
    forest_host_gbdt_raw_into,
    forest_host_vote_into,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_words
from core.shared_slot import SharedSlot


comptime FAMILY_RF = 0
comptime FAMILY_ET = 1

# The host grove registries of the forest host binding, named apart from the
# rf and trees host families' ("MojolearnRFResidentForestHostV1",
# "MojolearnETResidentForestHostV1" in bindings/forest_host_groves_binding.mojo),
# so two host bindings loaded into one process never share a handle space.
# (Moved unchanged from bindings/_mojolearn_forest_host.mojo: same names,
# same layout, so the handle space is the one the Python exports used.)
comptime RF_GROVES = SharedSlot[HostGroveRegistry, "MojolearnForestHostRFGrovesV1"]
comptime ET_GROVES = SharedSlot[HostGroveRegistry, "MojolearnForestHostETGrovesV1"]
#: Rows per call, so `n_rows * n_cols` stays far from any Int edge.
comptime FOREST_HOST_MAX_ROWS = 1073741824


def forest_host_family(family: Int, entry: String) raises -> Int:
    if family != FAMILY_RF and family != FAMILY_ET:
        raise Error(entry + ": family must be 0 (RandomForest) or 1 (ExtraTrees)")
    return family


def forest_host_groves_prepare_run(addresses: List[Int], params: List[Int], family_word: Int) raises -> Int:
    """The host grove snapshot of a saved forest: the five model arrays
    copied and validated once, a handle back."""
    var entry = String("forest_host_groves_prepare")
    var family = forest_host_family(family_word, entry)
    if len(addresses) != 5 or len(params) != 4:
        raise Error(entry + ": expected 5 addresses and 4 params")
    var n_trees = params[0]
    var n_cols = params[1]
    var num_outputs = params[2]
    var n_nodes = params[3]
    if n_cols <= 0 or n_cols > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_cols must be in [1, 2^30]")
    if n_trees <= 0 or n_trees > 2147483646:
        raise Error(entry + ": n_trees must be in [1, 2^31 - 2]")
    if num_outputs <= 0 or n_nodes <= 0 or n_nodes > 2147483647 // num_outputs:
        raise Error(entry + ": num_outputs and n_nodes must be positive and n_nodes * num_outputs must fit int32")
    if n_nodes < n_trees:
        raise Error(entry + ": fewer nodes than trees")
    var offsets_p = i32_ptr(addresses[0])
    var colid_p = i32_ptr(addresses[1])
    var quesval_p = f32_ptr(addresses[2])
    var left_p = i32_ptr(addresses[3])
    var leaves_p = f32_ptr(addresses[4])
    if Int(offsets_p[0]) != 0 or Int(offsets_p[n_trees]) != n_nodes:
        raise Error(entry + ": tree_offsets must start at 0 and end at n_nodes")
    var offsets = read_i32(addresses[0], n_trees + 1)
    var columns = read_i32(addresses[1], n_nodes)
    var thresholds = read_f32(addresses[2], n_nodes)
    var left = read_i32(addresses[3], n_nodes)
    var leaves = read_f32(addresses[4], n_nodes * num_outputs)
    _ = colid_p
    _ = quesval_p
    _ = left_p
    _ = leaves_p
    var model = HostGroveForest(
        offsets^, columns^, thresholds^, left^, leaves^, n_cols, num_outputs
    )
    if family == FAMILY_RF:
        return RF_GROVES.get_or_create_ptr()[].prepare(model^)
    return ET_GROVES.get_or_create_ptr()[].prepare(model^)


def forest_host_groves_predict_run(handle: Int, x_addr: Int, out_addr: Int,
                                   params: List[Int], family_word: Int) raises -> Int:
    """The grove engine's answer for `n_rows` ROW-major rows: the divided
    fold, `n_rows * num_outputs` float32 into `out`. Returns the rows."""
    var entry = String("forest_host_groves_predict")
    var family = forest_host_family(family_word, entry)
    if len(params) != 3:
        raise Error(entry + ": expected 3 params [n_rows, n_cols, num_outputs]")
    var n_rows = params[0]
    var n_cols = params[1]
    var num_outputs = params[2]
    if n_rows <= 0 or n_rows > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_rows must be in [1, 2^30]")
    if n_cols <= 0 or n_cols > FOREST_HOST_MAX_ROWS or num_outputs <= 0:
        raise Error(entry + ": n_cols and num_outputs must be positive")
    if n_rows > 2147483647 // n_cols or n_rows > 2147483647 // num_outputs:
        raise Error(entry + ": n_rows * n_cols and n_rows * num_outputs must fit int32")
    var xp = f32_ptr(x_addr)
    var op = f32_ptr(out_addr)
    var id = handle
    var state = RF_GROVES.get_or_create_ptr()
    if family == FAMILY_ET:
        state = ET_GROVES.get_or_create_ptr()
    if id not in state[].entries:
        raise Error(entry + ": unknown or released host grove forest handle")
    if family == FAMILY_RF:
        state[].entries[id].predict_into[True](xp, op, n_rows, n_cols, num_outputs)
    else:
        state[].entries[id].predict_into[False](xp, op, n_rows, n_cols, num_outputs)
    return n_rows


def forest_host_groves_release_run(handle: Int, family_word: Int) raises:
    var family = forest_host_family(family_word, String("forest_host_groves_release"))
    if family == FAMILY_RF:
        RF_GROVES.get_or_create_ptr()[].release(handle)
    else:
        ET_GROVES.get_or_create_ptr()[].release(handle)


def forest_host_predict_run(addresses: List[Int], params: List[Int], family: Int, entry: String) raises -> Int:
    """The sequential vote of the three predict exports (no GIL needed)."""
    if len(addresses) != 7 or len(params) != 5:
        raise Error(entry + ": expected 7 addresses and 5 params")
    var n_rows = params[0]
    var n_cols = params[1]
    var n_trees = params[2]
    var num_outputs = params[3]
    var n_nodes = params[4]
    if n_rows <= 0 or n_rows > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_rows must be in [1, 2^30]")
    if n_cols <= 0 or n_cols > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_cols must be in [1, 2^30]")
    if n_trees <= 0 or n_trees > 2147483646:
        raise Error(entry + ": n_trees must be in [1, 2^31 - 2]")
    if num_outputs <= 0 or n_nodes <= 0 or n_nodes > 2147483647 // num_outputs:
        raise Error(entry + ": num_outputs and n_nodes must be positive and n_nodes * num_outputs must fit int32")
    if n_nodes < n_trees:
        raise Error(entry + ": fewer nodes than trees")
    var offsets_p = i32_ptr(addresses[0])
    var colid_p = i32_ptr(addresses[1])
    var quesval_p = f32_ptr(addresses[2])
    var left_p = i32_ptr(addresses[3])
    var leaves_p = f32_ptr(addresses[4])
    var x_addr = addresses[5]
    var op = f32_ptr(addresses[6])
    if Int(offsets_p[0]) != 0 or Int(offsets_p[n_trees]) != n_nodes:
        raise Error(entry + ": tree_offsets must start at 0 and end at n_nodes")
    var rows = read_f32(x_addr, n_rows * n_cols)
    forest_host_vote_into(
        family == FAMILY_RF, rows, offsets_p, colid_p, quesval_p, left_p,
        leaves_p, op, n_rows, n_cols, n_trees, num_outputs, n_nodes,
    )
    return n_rows


def forest_host_rf_predict_proba_run(addresses: List[Int], params: List[Int]) raises -> Int:
    """`rf_predict_proba`'s answer on the host: the divided vote, `num_outputs >= 2`."""
    if len(params) < 4:
        raise Error("IndexError: list index out of range")
    if params[3] < 2:
        raise Error("forest_host_rf_predict_proba: num_outputs must be >= 2")
    return forest_host_predict_run(addresses, params, FAMILY_RF, "forest_host_rf_predict_proba")


def forest_host_rf_predict_reg_run(addresses: List[Int], params: List[Int]) raises -> Int:
    """`rf_predict_reg`'s answer on the host: the one-output vote, `num_outputs == 1`."""
    if len(params) < 4:
        raise Error("IndexError: list index out of range")
    if params[3] != 1:
        raise Error("forest_host_rf_predict_reg: num_outputs must be 1")
    return forest_host_predict_run(addresses, params, FAMILY_RF, "forest_host_rf_predict_reg")


def forest_host_et_predict_run(addresses: List[Int], params: List[Int]) raises -> Int:
    """`et_predict`'s answer on the host: the divided vote for classifier and
    regressor alike, `num_outputs >= 1`."""
    return forest_host_predict_run(addresses, params, FAMILY_ET, "forest_host_et_predict")


def forest_host_gbdt_predict_run(addresses: List[Int], params: List[Int], bias_value: Float64) raises -> Int:
    """`gbdt_predict` / `gbdt_predict_multi` at `PREDICT_RAW` on the host:
    the raw approxes, `n_rows * dim` float32, ROW-major."""
    var entry = String("forest_host_gbdt_predict")
    if len(addresses) != 15 or (len(params) != 8 and len(params) != 9):
        raise Error(entry + ": expected 15 addresses and 8 or 9 params")
    var n_rows = params[0]
    var n_cols = params[1]
    var n_trees = params[2]
    var dim = params[3]
    var non_symmetric = params[4] != 0
    var n_splits = params[5]
    var n_leaf_values = params[6]
    var n_borders = params[7]
    var row_major = len(params) == 9 and params[8] != 0
    if n_rows <= 0 or n_rows > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_rows must be in [1, 2^30]")
    if n_cols <= 0 or n_cols > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_cols must be in [1, 2^30]")
    if n_trees < 0 or n_trees > 2147483646:
        raise Error(entry + ": n_trees must be in [0, 2^31 - 2]")
    if dim <= 0 or dim > 65536:
        raise Error(entry + ": dim must be in [1, 65536]")
    if n_splits < 0 or n_leaf_values < 0 or n_borders < 0:
        raise Error(entry + ": counts must be non-negative")
    if n_splits > 2147483647 or n_leaf_values > 2147483647 or n_borders > 2147483647:
        raise Error(entry + ": counts must fit int32")
    var border_offsets_p = i32_ptr(addresses[0])
    var borders_p = f32_ptr(addresses[1])
    var fold_counts_p = i32_ptr(addresses[2])
    var one_hot_p = i32_ptr(addresses[3])
    var nan_p = i32_ptr(addresses[4])
    var tree_offsets_p = i32_ptr(addresses[5])
    var split_feature_p = i32_ptr(addresses[6])
    var split_bin_p = i32_ptr(addresses[7])
    var split_take_bin_p = i32_ptr(addresses[8])
    var node_left_p = i32_ptr(addresses[9])
    var node_right_p = i32_ptr(addresses[10])
    var leaf_offsets_p = i32_ptr(addresses[11])
    var leaves_p = f32_ptr(addresses[12])
    var x_addr = addresses[13]
    var op = f32_ptr(addresses[14])
    if Int(border_offsets_p[0]) != 0 or Int(border_offsets_p[n_cols]) != n_borders:
        raise Error(entry + ": border_offsets must start at 0 and end at n_borders")
    var rows = read_f32(x_addr, n_rows * n_cols)
    forest_host_gbdt_raw_into(
        rows, n_rows, n_cols,
        border_offsets_p, borders_p, fold_counts_p, one_hot_p, nan_p,
        tree_offsets_p, split_feature_p, split_bin_p, split_take_bin_p,
        node_left_p, node_right_p, leaf_offsets_p, leaves_p,
        n_trees, dim, non_symmetric, n_splits, n_leaf_values, bias_value,
        op, row_major,
    )
    return n_rows


def forest_host_gbdt_expand_ctr_run(addresses: List[Int], params: List[Int]) raises -> Int:
    """`predict_floats`'s CTR step on the host: raw columns to model
    columns. Returns the columns written."""
    var entry = String("forest_host_gbdt_expand_ctr")
    if len(addresses) != 10 or len(params) != 11:
        raise Error(entry + ": expected 10 addresses and 11 params")
    var n_rows = params[0]
    var n_raw = params[1]
    var n_cols = params[2]
    var n_ctr = params[3]
    var n_ctr_ints = params[4]
    var n_ctr_floats = params[5]
    var n_tensor = params[6]
    var n_tensor_ints = params[7]
    var n_tensor_floats = params[8]
    var n_counts = params[9]
    var n_borders = params[10]
    if n_rows <= 0 or n_rows > FOREST_HOST_MAX_ROWS:
        raise Error(entry + ": n_rows must be in [1, 2^30]")
    if n_raw <= 0 or n_cols < n_raw or n_cols > 65536:
        raise Error(entry + ": n_raw must be positive and n_cols in [n_raw, 65536]")
    if n_rows > 2147483647 // n_cols:
        raise Error(entry + ": n_rows * n_cols must fit int32")
    for i in range(3, 11):  # small-loop(the eight count words of params)
        if params[i] < 0 or params[i] > 2147483647:
            raise Error(entry + ": counts must be non-negative and fit int32")
    var op = f32_ptr(addresses[9])
    var x = read_f32(addresses[0], n_rows * n_raw)
    var ctr_ints = read_i32(addresses[1], n_ctr_ints)
    var ctr_floats = read_f32(addresses[2], n_ctr_floats)
    var tensor_ints = read_i32(addresses[3], n_tensor_ints)
    var tensor_floats = read_f32(addresses[4], n_tensor_floats)
    var counts = read_i32(addresses[5], n_counts)
    var border_offsets = read_i32(addresses[6], n_cols + 1)
    var borders = read_f32(addresses[7], n_borders)
    var one_hot = read_i32(addresses[8], n_cols)
    forest_host_gbdt_expand_ctr_into(
        entry, x, n_rows, n_cols, ctr_ints, ctr_floats, n_ctr, tensor_ints,
        tensor_floats, n_tensor, counts, border_offsets, borders, one_hot, op,
    )
    return n_cols


def forest_host_dispatch(operation: Int, addresses: List[Int], words: List[Int], bias: Float64) raises -> Int:
    """One host table operation over unpacked words (layouts in
    core/host_entry_table.mojo)."""
    if operation == FH_RF_PREDICT_PROBA:
        return forest_host_rf_predict_proba_run(addresses, words)
    if operation == FH_RF_PREDICT_REG:
        return forest_host_rf_predict_reg_run(addresses, words)
    if operation == FH_ET_PREDICT:
        return forest_host_et_predict_run(addresses, words)
    if operation == FH_GROVES_PREPARE:
        if len(words) != 5:
            raise Error("forest_host_groves_prepare: expected 4 params and the family word")
        var params: List[Int] = [words[0], words[1], words[2], words[3]]
        return forest_host_groves_prepare_run(addresses, params, words[4])
    if operation == FH_GROVES_PREDICT:
        if len(addresses) != 2 or len(words) != 5:
            raise Error("forest_host_groves_predict: expected 2 addresses, the handle, 3 params and the family word")
        var params: List[Int] = [words[1], words[2], words[3]]
        return forest_host_groves_predict_run(words[0], addresses[0], addresses[1], params, words[4])
    if operation == FH_GBDT_EXPAND_CTR:
        return forest_host_gbdt_expand_ctr_run(addresses, words)
    return forest_host_gbdt_predict_run(addresses, words, bias)


def forest_host_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The forest host entry of the host table (`FOREST_HOST_ABI`): addresses
    span aa[0:na], Int words pp[0:np], Float64 words ff[0:nf] (ff[0] is the
    GBDT bias). Writes the run's count or handle to result; a refusal goes
    to sink."""
    try:
        if operation < 0 or operation >= FH_OP_COUNT:
            raise Error("forest host entry: unknown operation " + String(operation))
        if na < 0 or np < 0 or nf < 0 or not result:
            raise Error("forest host entry: invalid request spans")
        var addresses = native_descriptor_words(aa, na)
        var words = native_descriptor_words(pp, np)
        var bias = Float64(0.0)
        if nf > 0:
            if not ff:
                raise Error("forest host entry: null request span")
            bias = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)[0]
        var value = forest_host_dispatch(operation, addresses, words, bias)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ForestHostEntryOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the forest host table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _ENTRY_OWNER_NAME = "MojolearnForestHostEntryOwnerV1"
comptime FOREST_HOST_ENTRY_OWNER = SharedSlot[ForestHostEntryOwner, _ENTRY_OWNER_NAME]


def publish_forest_host_entries() raises:
    """Publish this image's forest host entry at module init
    (`_mojolearn_forest_host`). A repeated init publishes the same pointer;
    a second image with the same tier and another pointer refuses
    (core/entry_table.mojo)."""
    var owner = FOREST_HOST_ENTRY_OWNER.get_or_create_ptr()
    var key = HostEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    host_table_publish[HOST_TABLE_FOREST_HOST](key, Int(owner),
        [String(FOREST_HOST_ABI)], [FH_OP_COUNT], [forest_host_entry_v1])
