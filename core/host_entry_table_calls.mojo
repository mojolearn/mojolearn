# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the forest host operations through the host entry
table (lane calltable-host-misc, 2026-11-11). No Python: a caller passes
buffer addresses and plain scalars and keeps the buffers alive until the call
returns. Layouts: `core/host_entry_table.mojo`; runs:
`core/host_entry_table_forest.mojo`.

`fh_call` reads the published table once per call (a lock-free acquire load
and a schema compare) and calls the pointer; a missing or mismatched
publication refuses with the table's error. The refusal text of a run is the
text its former Python export raised. The caller decides the GIL: the grove
operations keep it held, the others may release it.
"""
from core.native_descriptor_call import native_descriptor_call
from core.host_entry_table import (
    HostEntryKey, host_table_entry, HOST_TABLE_FOREST_HOST, FOREST_HOST_ABI, FH_OP_COUNT,
    FH_RF_PREDICT_PROBA, FH_RF_PREDICT_REG, FH_ET_PREDICT, FH_GROVES_PREPARE, FH_GROVES_PREDICT,
    FH_GBDT_EXPAND_CTR, FH_GBDT_PREDICT,
)


def fh_call(key: HostEntryKey, op: Int, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """One forest host operation in the forest host image the key names."""
    var entry = host_table_entry[HOST_TABLE_FOREST_HOST](key, 0, String(FOREST_HOST_ABI), FH_OP_COUNT)
    return native_descriptor_call(entry, op, addrs, ints, floats)


def fh_vote(key: HostEntryKey, op: Int, addrs: List[Int], n_rows: Int, n_cols: Int, n_trees: Int,
            num_outputs: Int, n_nodes: Int) raises -> Int:
    """FH_RF_PREDICT_PROBA, FH_RF_PREDICT_REG or FH_ET_PREDICT over the seven
    addresses [offsets, colid, quesval, left, leaves, x, out]. Returns rows."""
    if op != FH_RF_PREDICT_PROBA and op != FH_RF_PREDICT_REG and op != FH_ET_PREDICT:
        raise Error("forest host entry: not a vote operation")
    var ints: List[Int] = [n_rows, n_cols, n_trees, num_outputs, n_nodes]
    var floats = List[Float64](length=len(ints), fill=0.0)
    return fh_call(key, op, addrs, ints, floats)


def fh_groves_prepare(key: HostEntryKey, addrs: List[Int], n_trees: Int, n_cols: Int, num_outputs: Int,
                      n_nodes: Int, family: Int) raises -> Int:
    """The host grove snapshot of the five model addresses; the handle back."""
    var ints: List[Int] = [n_trees, n_cols, num_outputs, n_nodes, family]
    var floats = List[Float64](length=len(ints), fill=0.0)
    return fh_call(key, FH_GROVES_PREPARE, addrs, ints, floats)


def fh_groves_predict(key: HostEntryKey, handle: Int, x: Int, dst: Int, n_rows: Int, n_cols: Int,
                      num_outputs: Int, family: Int) raises -> Int:
    """The grove engine over `n_rows` ROW-major rows of x into dst. Returns rows."""
    var addrs: List[Int] = [x, dst]
    var ints: List[Int] = [handle, n_rows, n_cols, num_outputs, family]
    var floats = List[Float64](length=len(ints), fill=0.0)
    return fh_call(key, FH_GROVES_PREDICT, addrs, ints, floats)


def fh_gbdt_expand_ctr(key: HostEntryKey, addrs: List[Int], params: List[Int]) raises -> Int:
    """The CTR expansion: ten addresses, eleven words. Returns columns."""
    var floats = List[Float64](length=len(params), fill=0.0)
    return fh_call(key, FH_GBDT_EXPAND_CTR, addrs, params, floats)


def fh_gbdt_predict(key: HostEntryKey, addrs: List[Int], params: List[Int], bias: Float64) raises -> Int:
    """The raw GBDT approxes: fifteen addresses, eight or nine words and the
    float64 bias. Returns rows."""
    var floats = List[Float64](length=len(params), fill=0.0)
    if len(floats) == 0:
        floats.append(bias)
    else:
        floats[0] = bias
    return fh_call(key, FH_GBDT_PREDICT, addrs, params, floats)
