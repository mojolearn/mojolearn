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
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.trees_entry_table import TreesEntryKey
from core.gbdt_entry_table import gbdt_vendor_word
from core.forest_entry_table import ForestEntryKey, forest_table_publish
from core.forest_fit_request import (
    ForestFitArgs, FOREST_ENTRY_ABI, XF_OP_COUNT, XF_CAPS, XF_NUMERIC_MODE, XF_DT_DEFAULT_BINS, XF_FIT, XF_FIT_DART,
    XF_EXPORT_SIZES, XF_EXPORT_INTO, XF_EXPORT_VERIFY, XF_EXPORT_RELEASE, XF_PREDICT, XF_RESIDENT_PREPARE,
    XF_RESIDENT_PREDICT, XF_RESIDENT_LABELS, XF_RESIDENT_RELEASE, XF_STRICT_PREPARE, XF_STRICT_PREDICT,
    XF_STRICT_RELEASE, XF_W_SHAPE, XF_SHAPE_CLASSIFIER, XF_SHAPE_ROWMAJOR, forest_fit_args_at,
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
                regression: Bool) raises -> Int:
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
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        ...

    @staticmethod
    def strict_release(handle: Int) raises:
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


def _run[I: ForestImage](op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int) raises -> Int:
    var p = _words(pp, np)
    if op == XF_CAPS:
        return I.caps()
    if op == XF_NUMERIC_MODE:
        return Int(GLOBAL_NUMERIC_MODE)
    if op == XF_DT_DEFAULT_BINS:
        return I.dt_default_bins()
    if op == XF_FIT or op == XF_FIT_DART:
        var args = forest_fit_args_at(aa, na, pp, np, ff)
        if op == XF_FIT_DART:
            return I.fit_dart(args)
        var shape = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)[XF_W_SHAPE]
        return I.fit((shape & XF_SHAPE_CLASSIFIER) != 0, (shape & XF_SHAPE_ROWMAJOR) != 0, args)
    var a = _words(aa, na)
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
        return I.predict(_sub(a, 0, 5), a[5], a[6], p[0], p[1], p[2], p[3], p[4] != 0)
    if op == XF_RESIDENT_PREPARE:
        _need(na, np, 5, 4, "forest_prepare_gpu")
        return I.resident_prepare(a, p[0], p[1], p[2], p[3])
    if op == XF_RESIDENT_PREDICT or op == XF_RESIDENT_LABELS or op == XF_STRICT_PREDICT:
        _need(na, np, 3, 3, "resident prediction")
        if op == XF_RESIDENT_PREDICT:
            return I.resident_predict(a[0], a[1], a[2], p[0], p[1], p[2])
        if op == XF_RESIDENT_LABELS:
            return I.resident_labels(a[0], a[1], a[2], p[0], p[1], p[2])
        return I.strict_predict(a[0], a[1], a[2], p[0], p[1], p[2])
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
    raise Error("forest entry: unknown operation " + String(op))


def forest_entry_dispatch[I: ForestImage](operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    """The body of a forest image's `abi("C")` entry: span admission, the
    operation, the count to result, a refusal to sink."""
    try:
        if operation < 0 or operation >= XF_OP_COUNT:
            raise Error("forest entry: unknown operation " + String(operation))
        if na < 0 or np < 0 or np != nf or not result:
            raise Error("forest entry: invalid request spans")
        if (na > 0 and not aa) or (np > 0 and (not pp or not ff)):
            raise Error("forest entry: null request span")
        var value = _run[I](operation, aa, na, pp, np, ff)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


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
