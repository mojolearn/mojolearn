# SPDX-License-Identifier: Apache-2.0
"""The dispatch and publication of a forest image's native entry (lane
calltable-forest, 2026-11-11; ABI in core/forest_fit_request.mojo, table in
core/forest_entry_table.mojo).

Each forest binding (`_mojolearn_rf`, `_mojolearn_rf_host`,
`_mojolearn_trees`, `_mojolearn_trees_host`) defines a `ForestImage` (its
fits, export registry, sequential and resident predict and host strict
snapshot, which differ between the four images) and an `abi("C")` entry that
calls `forest_entry_dispatch` with it, then publishes that entry at module
init with `forest_entry_publish`. No Python object crosses. An operation an
image does not serve refuses in the words the door used for the missing
Python entry. Publication is a small metadata operation with no
independent numerical work to split.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.entry_table import entry_op_plain, entry_op_gil_free
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.trees_entry_table import TreesEntryKey
from core.gbdt_entry_table import gbdt_vendor_word
from core.forest_entry_table import ForestEntryKey, forest_table_publish
from core.forest_fit_request import (
    ForestFitArgs, FOREST_ENTRY_ABI, XF_OP_COUNT, XF_CAPS, XF_NUMERIC_MODE, XF_DT_DEFAULT_BINS, XF_FIT, XF_FIT_DART,
    XF_EXPORT_SIZES, XF_EXPORT_INTO, XF_EXPORT_VERIFY, XF_EXPORT_RELEASE, XF_PREDICT, XF_RESIDENT_PREPARE,
    XF_RESIDENT_PREDICT, XF_RESIDENT_LABELS, XF_RESIDENT_RELEASE, XF_STRICT_PREPARE, XF_STRICT_PREDICT,
    XF_STRICT_RELEASE, XF_SESSION_OPEN, XF_SESSION_CLOSE, XF_FIT_SESSION, XF_ADA_OPEN, XF_FIT_ADA_ROUND,
    XF_STAGE_INPUT, XF_UNSTAGE_INPUT, XF_FIT_ROWS, XF_W_SHAPE, XF_SHAPE_CLASSIFIER, XF_SHAPE_ROWMAJOR, forest_fit_args_at,
)


trait ForestImage:
    @staticmethod
    def caps() -> Int:
        """The image's XF_CAP_* bits."""
        ...

    @staticmethod
    def dt_default_bins() -> Int:
        ...

    @staticmethod
    def fit(classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
        """One fit, kept in the image's export registry; its handle."""
        ...

    @staticmethod
    def fit_dart(args: ForestFitArgs) raises -> Int:
        ...

    @staticmethod
    def export_sizes(handle: Int, out_addr: Int) raises:
        ...

    @staticmethod
    def export_into(handle: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
        ...

    @staticmethod
    def export_verify(handle: Int, dst: List[Int]) raises -> Int:
        ...

    @staticmethod
    def export_release(handle: Int) raises:
        ...

    @staticmethod
    def predict(model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int, outputs: Int,
                regression: Bool, free: Bool) raises -> Int:
        """`free`: the caller holds no GIL (see `forest_entry_dispatch`); an
        image that releases around its walk runs it bare."""
        ...

    @staticmethod
    def resident_prepare(model: List[Int], features: Int, trees: Int, outputs: Int, ordered: Int) raises -> Int:
        ...

    @staticmethod
    def resident_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        ...

    @staticmethod
    def resident_labels(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        ...

    @staticmethod
    def resident_release(handle: Int) raises:
        ...

    @staticmethod
    def strict_prepare(model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
        ...

    @staticmethod
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int,
                       free: Bool) raises -> Int:
        ...

    @staticmethod
    def strict_release(handle: Int) raises:
        ...

    @staticmethod
    def session_open(x: Int, rows: Int, cols: Int, row_major: Bool, share_tables: Bool) raises -> Int:
        """X kept for the member fits of one ensemble; the session handle."""
        ...

    @staticmethod
    def session_close(handle: Int) raises:
        ...

    @staticmethod
    def fit_session(classifier: Bool, args: ForestFitArgs) raises -> Int:
        """A member fit on `args.session`'s X; its export handle."""
        ...

    @staticmethod
    def ada_open(handle: Int, w: Int, y: Int, cw: Int, n: Int, k: Int) raises:
        ...

    @staticmethod
    def fit_ada_round(classifier: Bool, args: ForestFitArgs) raises -> Int:
        """One AdaBoost round on `args.session`; the member's export handle."""
        ...

    @staticmethod
    def stage_input(x: Int, rows: Int, cols: Int) raises -> Int:
        """XF_STAGE_INPUT: 1 when the rows at `x` are staged as the image's
        shared resident input, 0 when not (a host image stages nothing)."""
        ...

    @staticmethod
    def unstage_input() raises:
        """XF_UNSTAGE_INPUT: end the staged input (a host image: nothing)."""
        ...

    @staticmethod
    def fit_rows(classifier: Bool, args: ForestFitArgs) raises -> Int:
        """XF_FIT_ROWS (lane vdf-scope-trees): a fit on rows of the image's
        device store slot `args.session`, taken on the device; its export
        handle. An image without XF_CAP_DEV_ROWS refuses in words."""
        ...


def _words(address: Int, count: Int) -> List[Int]:
    var out = List[Int](capacity=count)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    for k in range(count):  # small-loop(request words): at most seven addresses
        out.append(p[k])
    return out^


def _sub(a: List[Int], lo: Int, hi: Int) -> List[Int]:
    var out = List[Int](capacity=hi - lo)
    for k in range(lo, hi):  # small-loop(request words): the five model array addresses
        out.append(a[k])
    return out^


def _need(na: Int, np: Int, want_a: Int, want_p: Int, what: String) raises:
    if na != want_a or np != want_p:
        raise Error(what + ": needs " + String(want_a) + " addresses and " + String(want_p) + " integers")


def _run[I: ForestImage](op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, free: Bool) raises -> Int:
    # TOMBSTONE: the refusal of a data session operation from a caller with
    # no GIL ("needs the caller to hold the GIL") deleted 2026-11-23 by lane
    # mf-ada2: the session registries have their own lock words
    # (core/mojolearn_rf_run.mojo, core/mojolearn_rf_host_run.mojo) and the
    # session bodies run bare (`forest_session_entry_releases`). Not an
    # experiment arm.
    var p = _words(pp, np)
    if op == XF_CAPS:
        return I.caps()
    if op == XF_NUMERIC_MODE:
        return Int(GLOBAL_NUMERIC_MODE)
    if op == XF_DT_DEFAULT_BINS:
        return I.dt_default_bins()
    if (op == XF_FIT or op == XF_FIT_DART or op == XF_FIT_SESSION or op == XF_FIT_ADA_ROUND
            or op == XF_FIT_ROWS):
        var args = forest_fit_args_at(aa, na, pp, np, ff)
        args.gil_free = free
        if op == XF_FIT_DART:
            return I.fit_dart(args)
        var shape = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)[XF_W_SHAPE]
        if op == XF_FIT_SESSION:
            return I.fit_session((shape & XF_SHAPE_CLASSIFIER) != 0, args)
        if op == XF_FIT_ADA_ROUND:
            return I.fit_ada_round((shape & XF_SHAPE_CLASSIFIER) != 0, args)
        if op == XF_FIT_ROWS:
            return I.fit_rows((shape & XF_SHAPE_CLASSIFIER) != 0, args)
        return I.fit((shape & XF_SHAPE_CLASSIFIER) != 0, (shape & XF_SHAPE_ROWMAJOR) != 0, args)
    var a = _words(aa, na)
    if op == XF_SESSION_OPEN:
        _need(na, np, 1, 4, "forest data session open")
        return I.session_open(a[0], p[0], p[1], p[2] != 0, p[3] != 0)
    if op == XF_SESSION_CLOSE:
        _need(na, np, 0, 1, "forest data session close")
        I.session_close(p[0])
        return 0
    if op == XF_ADA_OPEN:
        _need(na, np, 3, 3, "forest AdaBoost open")
        I.ada_open(p[0], a[0], a[1], a[2], p[1], p[2])
        return 0
    if op == XF_EXPORT_SIZES:
        _need(na, np, 2, 0, "forest export sizes")
        I.export_sizes(a[0], a[1])
        return 0
    if op == XF_EXPORT_INTO:
        _need(na, np, 6, 3, "forest_export")
        I.export_into(a[0], _sub(a, 1, 6), p[0], p[1], p[2])
        return 0
    if op == XF_EXPORT_VERIFY:
        _need(na, np, 6, 0, "forest export verify")
        return I.export_verify(a[0], _sub(a, 1, 6))
    if op == XF_EXPORT_RELEASE:
        _need(na, np, 1, 0, "forest_export_release")
        I.export_release(a[0])
        return 0
    if op == XF_PREDICT:
        _need(na, np, 7, 5, "forest predict")
        return I.predict(_sub(a, 0, 5), a[5], a[6], p[0], p[1], p[2], p[3], p[4] != 0, free)
    if op == XF_RESIDENT_PREPARE:
        _need(na, np, 5, 4, "forest_prepare_gpu")
        return I.resident_prepare(a, p[0], p[1], p[2], p[3])
    if op == XF_RESIDENT_PREDICT or op == XF_RESIDENT_LABELS or op == XF_STRICT_PREDICT:
        _need(na, np, 3, 3, "resident prediction")
        if op == XF_RESIDENT_PREDICT:
            return I.resident_predict(a[0], a[1], a[2], p[0], p[1], p[2])
        if op == XF_RESIDENT_LABELS:
            return I.resident_labels(a[0], a[1], a[2], p[0], p[1], p[2])
        return I.strict_predict(a[0], a[1], a[2], p[0], p[1], p[2], free)
    if op == XF_RESIDENT_RELEASE or op == XF_STRICT_RELEASE:
        # ints [handle]: the shape bindings/trees_resident_table_run.mojo's
        # native release calls with
        _need(na, np, 0, 1, "forest snapshot release")
        if op == XF_RESIDENT_RELEASE:
            I.resident_release(p[0])
        else:
            I.strict_release(p[0])
        return 0
    if op == XF_STRICT_PREPARE:
        _need(na, np, 5, 3, "forest_strict_prepare")
        return I.strict_prepare(a, p[0], p[1], p[2])
    if op == XF_STAGE_INPUT:
        _need(na, np, 1, 2, "forest input staging")
        if p[0] < 1 or p[1] < 1 or p[0] > 2147483647 // p[1]:
            return 0
        return I.stage_input(a[0], p[0], p[1])
    if op == XF_UNSTAGE_INPUT:
        _need(na, np, 0, 0, "forest input unstaging")
        I.unstage_input()
        return 0
    raise Error("forest entry: unknown operation " + String(op))


def forest_entry_dispatch[I: ForestImage](operation_word: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    """The body of a forest image's `abi("C")` entry: span admission, the
    operation, the count to result, a refusal to sink.

    THE GIL (lane fx-gil, core/entry_table.mojo): an operation word carrying
    `ENTRY_OP_GIL_FREE` comes from a caller that holds no GIL (a member's
    worker). The image's fits, host walk and strict walk then run their one
    body bare (`ForestFitArgs.gil_free`, the `free` word of `predict` /
    `strict_predict`); without the bit they release around that same body,
    as their doors need. Nothing else in an image touches the GIL, and no
    registry an operation reaches relies on it (each has its own lock word:
    core/forest_export_run.mojo, core/forest_inference_model.mojo,
    core/forest_host_groves.mojo, core/forest_host_strict_run.mojo, and the
    data session registries core/mojolearn_rf_run.mojo `RF_SESSIONS` and
    core/mojolearn_rf_host_run.mojo `RF_HOST_SESSIONS`, lane mf-ada2)."""
    try:
        var free = entry_op_gil_free(operation_word)
        var operation = entry_op_plain(operation_word)
        if operation < 0 or operation >= XF_OP_COUNT:
            raise Error("forest entry: unknown operation " + String(operation))
        if na < 0 or np < 0 or np != nf or not result:
            raise Error("forest entry: invalid request spans")
        if (na > 0 and not aa) or (np > 0 and (not pp or not ff)):
            raise Error("forest entry: null request span")
        var value = _run[I](operation, aa, na, pp, np, ff, free)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def forest_host_entry_releases(operation: Int) -> Bool:
    """True for the operations a host image's binding entry releases the GIL
    around when its caller holds it (lane bm-trees, 2026-11-22; the gbdt
    host image's `gbdt_entry_releases`): the fits (XF_FIT, XF_FIT_DART), the
    sequential predict walk and the strict predict, the bodies the images
    released around before their bodies moved to core. The binding then runs
    the dispatch with `ENTRY_OP_GIL_FREE` added, so the image's bodies run
    bare. False for an operation already carrying `ENTRY_OP_GIL_FREE` (that
    caller holds no GIL) and for every other operation (a few words, or a
    data session operation: lane mf-ada2 moved those to
    `forest_session_entry_releases`, which this includes)."""
    if entry_op_gil_free(operation):
        return False
    return (operation == XF_FIT or operation == XF_FIT_DART or operation == XF_PREDICT
            or operation == XF_STRICT_PREDICT or forest_session_entry_releases(operation))


def forest_session_entry_releases(operation: Int) -> Bool:
    """True for a data session operation (XF_SESSION_OPEN, XF_SESSION_CLOSE,
    XF_FIT_SESSION, XF_ADA_OPEN, XF_FIT_ADA_ROUND) from a caller that holds
    the GIL (lane mf-ada2, 2026-11-23). Their registries have their own lock
    words and their bodies never touch the GIL, so an image's binding entry
    releases around the WHOLE dispatch of such an operation and adds
    `ENTRY_OP_GIL_FREE` (its upload, fit and launches then run bare), as the
    host image does for its fits; a typed RUN that holds no GIL
    (xtrees/trees_ada_fit_run.mojo) sends the bit itself and the entry runs
    the same dispatch with no release."""
    if entry_op_gil_free(operation):
        return False
    return (operation == XF_SESSION_OPEN or operation == XF_SESSION_CLOSE or operation == XF_FIT_SESSION
            or operation == XF_ADA_OPEN or operation == XF_FIT_ADA_ROUND)


struct ForestEntryOwner(Defaultable, Movable):
    """An image-lifetime publisher word."""

    var word: Int

    def __init__(out self):
        self.word = 1


def forest_entry_publish[owner_name: StaticString](family: Int, vendor: StaticString,
                                                   entry: NativeDescriptorCall) raises:
    """Publish an image's forest entry in its family's table under (its
    tier, the vendor word of the string it exports as `rf_vendor` /
    `trees_vendor`). A build with no vendor word ("none") publishes nothing;
    its callers refuse with the table's "provider is not published"."""
    var word = gbdt_vendor_word(vendor)
    if word < 0:
        return
    comptime OWNER = SharedSlot[ForestEntryOwner, owner_name]
    var owner = OWNER.get_or_create_ptr()
    var key = ForestEntryKey(family, TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), word))
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    forest_table_publish(key, Int(owner), [String(FOREST_ENTRY_ABI)], [XF_OP_COUNT], entries)
