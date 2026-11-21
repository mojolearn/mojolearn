# SPDX-License-Identifier: Apache-2.0
"""A fitted forest as plain values, and the predict that reads it (RUN of
bindings/trees_public_forest_door.mojo; no Python import).

`ForestState` is the fitted state a predict needs, read ONCE per predict by
the door's UNPACK: the image's entry and capability bits, the numeric mode
and engine as words, the route they select, the dimensions and the validity
words of the route's snapshot. The model bytes are the five flat arrays the
fit exported; the door admits them (their addresses and sizes) only where a
route reads them: on a snapshot miss, or for the sequential walk.

Routes (the door's former string tests on the engine and the capabilities):
  FOREST_ROUTE_SEQUENTIAL  the image's sequential walk over the five arrays
                           (XF_PREDICT), nothing kept.
  FOREST_ROUTE_STRICT      a host image's strict snapshot (XF_STRICT_*),
                           kept in the tree resident table.
  FOREST_ROUTE_RESIDENT    the device snapshot (XF_RESIDENT_*), kept in the
                           tree resident table; `ordered` names its
                           aggregation (-1 the image's default, 1 strict,
                           0 the 32-grove graph).

The resident table (bindings/trees_resident_table_run.mojo) is typed; the
door keeps the Python pins of a kept snapshot (the minting binding and the
five arrays) beside it, under the same handle.

One synchronous launch per call: the parallelism is inside the image's
predict. Nothing here loops over rows.
"""
from core.entry_table import EntryRef
from core.forest_calls import (
    xf_predict, xf_resident_prepare, xf_resident_predict, xf_resident_labels, xf_strict_prepare, xf_strict_predict,
    xf_snapshot_release,
)
from core.forest_fit_request import (
    XF_CAP_ORDERED_RESIDENT, XF_CAP_FUSED_LABELS, XF_CAP_RESIDENT_LABELS, XF_CAP_STRICT,
)
from bindings.admitted_input_run import AdmittedXView
from bindings.trees_resident_table_run import trees_run_put, trees_run_pin
from xtrees.trees_forest_run import ForestHandle

comptime FOREST_ROUTE_SEQUENTIAL = 0
comptime FOREST_ROUTE_STRICT = 1
comptime FOREST_ROUTE_RESIDENT = 2


def forest_route(groves: Bool, caps: Int) -> Int:
    """The predict route of an engine on an image: the sequential engine of
    an image without the ordered resident snapshot walks the arrays (through
    its strict snapshot where it has one); every other pair predicts on the
    resident snapshot."""
    if not groves and (caps & XF_CAP_ORDERED_RESIDENT) == 0:
        if (caps & XF_CAP_STRICT) != 0:
            return FOREST_ROUTE_STRICT
        return FOREST_ROUTE_SEQUENTIAL
    return FOREST_ROUTE_RESIDENT


struct ForestState(Movable):
    """One fitted forest as a predict reads it (see the module docstring)."""
    var key: EntryRef
    #: the identity word of the minting binding (pinned by the door)
    var native: Int
    var caps: Int
    var fast: Bool
    var groves: Bool
    var route: Int
    var ordered: Int
    var features: Int
    var trees: Int
    var outputs: Int
    #: the validity words the route's snapshot is kept under
    var words: List[Int]

    def __init__(out self, key: EntryRef, native: Int, caps: Int, fast: Bool, groves: Bool, features: Int,
                 trees: Int, outputs: Int):
        self.key = key.copy()
        self.native = native
        self.caps = caps
        self.fast = fast
        self.groves = groves
        self.route = forest_route(groves, caps)
        # The aggregation (`_resident_ordered_flag`): strict for the
        # sequential engine, the 32-grove graph named explicitly under
        # IDENTICAL, the binary's default otherwise.
        self.ordered = -1
        if (caps & XF_CAP_ORDERED_RESIDENT) != 0:
            if not groves:
                self.ordered = 1
            elif not fast:
                self.ordered = 0
        self.features = features
        self.trees = trees
        self.outputs = outputs
        self.words = List[Int](capacity=5)
        if self.route != FOREST_ROUTE_SEQUENTIAL:
            self.words.append(features)
            self.words.append(trees)
            self.words.append(outputs)
        if self.route == FOREST_ROUTE_RESIDENT:
            self.words.append(1 if fast else 0)
            self.words.append(self.ordered)


def forest_labels_route(state: ForestState) -> Bool:
    """`ForestProtocol._predict_forest_labels`: whether the resident
    snapshot answers a classifier's codes directly (the compiled fused
    labels under IDENTICAL, the grove labels under FAST)."""
    if not state.fast:
        if (state.caps & XF_CAP_FUSED_LABELS) == 0:
            return False
        if state.route != FOREST_ROUTE_RESIDENT:
            return False
    elif not state.groves:
        return False
    return (state.caps & XF_CAP_RESIDENT_LABELS) != 0


def forest_shape_check(state: ForestState, sizes: List[Int], offsets: Int) raises:
    """The five arrays' element counts against the fitted dimensions, and
    the last tree offset against the node count."""
    var nodes = sizes[1]
    var last = -1
    if sizes[0] > 0:
        last = Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=offsets)[sizes[0] - 1])
    if (sizes[0] != state.trees + 1 or nodes < 1 or sizes[2] != nodes or sizes[3] != nodes
            or sizes[4] != nodes * state.outputs or last != nodes):
        raise Error("forest model array shapes do not match metadata")


def forest_snapshot_prepare(state: ForestState, kind: Int, release_op: Int, model_key: Int, generation: Int,
                            addrs: List[Int], nodes: Int, mirrors: List[Int],
                            watchable: Bool) raises -> ForestHandle:
    """A new snapshot of the five arrays at `addrs` for the state's route,
    kept in the tree resident table under (`model_key`, `generation`,
    `kind`), the state's words and the `mirrors` identity words when the
    model can be watched (`watchable`) and the table has room. The returned
    handle is pinned when kept; otherwise the call owns it and releases it
    after its launch."""
    var handle = 0
    if state.route == FOREST_ROUTE_STRICT:
        handle = xf_strict_prepare(state.key, addrs, state.features, state.trees, state.outputs)
    else:
        handle = xf_resident_prepare(state.key, addrs, state.features, state.trees, state.outputs, state.ordered)
    # The snapshot's bytes: the five flat arrays, 4 bytes an item.
    var bytes = 4 * ((state.trees + 1) + 3 * nodes + nodes * state.outputs)
    var kept = False
    try:
        if watchable:
            kept = trees_run_put(model_key, generation, kind, state.native, state.key.call, release_op, handle,
                                 state.words.copy(), mirrors.copy(), bytes)
        if kept:
            trees_run_pin(handle)
    except error:
        xf_snapshot_release(state.key, release_op, handle)
        raise error^
    return ForestHandle(handle, kept, state.key, release_op)


def forest_vote(state: ForestState, handle: Int, addrs: List[Int], x: AdmittedXView, dst: Int,
                regression: Bool) raises -> Int:
    """The rows' votes or values (flat float32 at `dst`) through the state's
    route: the sequential walk over the arrays at `addrs`, or the snapshot
    `handle`. Returns the rows written."""
    if state.route == FOREST_ROUTE_SEQUENTIAL:
        return xf_predict(state.key, addrs, x.address, dst, x.rows, x.cols, state.trees, state.outputs, regression)
    if state.route == FOREST_ROUTE_STRICT:
        return xf_strict_predict(state.key, handle, x.address, dst, x.rows, x.cols, state.outputs)
    return xf_resident_predict(state.key, handle, x.address, dst, x.rows, x.cols, state.outputs)


def forest_labels(state: ForestState, handle: Int, x: AdmittedXView, dst: Int) raises -> Int:
    """A classifier's int32 class codes at `dst` from the resident snapshot
    `handle`. Returns the rows written."""
    return xf_resident_labels(state.key, handle, x.address, dst, x.rows, x.cols, state.outputs)


def forest_wrote_check(wrote: Int, rows: Int, what: String) raises:
    if wrote != rows:
        raise Error("RuntimeError: " + what + " wrote " + String(wrote) + " of " + String(rows) + " rows")
