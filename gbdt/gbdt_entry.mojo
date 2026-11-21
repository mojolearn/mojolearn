# SPDX-License-Identifier: Apache-2.0
"""The dispatch and publication of a gbdt image's native entry (lane
calltable-gbdt-forest-dart, 2026-11-11; ABI in gbdt/fit_request.mojo, table
in core/gbdt_entry_table.mojo).

Both gbdt bindings define a `GbdtImage` (their fit, row-major capability and
model dimension, which differ between the GPU and the host binding) and an
`abi("C")` entry that calls `gbdt_entry_dispatch` with it, then publish that
entry at module init with `gbdt_entry_publish`. No Python object crosses.
Publication is a small metadata operation with no independent numerical
work to split.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.trees_entry_table import TreesEntryKey
from core.gbdt_entry_table import gbdt_table_publish, gbdt_vendor_word
from gbdt.fit_request import (
    GbdtFitArgs, GbdtFitOutcome, GBDT_ENTRY_ABI, XG_OP_COUNT, XG_CAPS, XG_NUMERIC_MODE, XG_FIT, XG_FIT_ROWMAJOR,
    XG_OUTCOME_SIZES, XG_OUTCOME_COPY, XG_OUTCOME_FREE, XG_MODEL_DIM, XG_CAP_ROW_MAJOR, gbdt_outcome_box,
    gbdt_outcome_sizes, gbdt_outcome_copy, gbdt_outcome_free, gbdt_string_at,
    XG_PREDICT, XG_PREDICT_MULTI, XG_BINARY, XG_SIGMOID_PAIR, XG_RESIDENT_PREPARE, XG_RESIDENT_PREDICT,
    XG_CAP_RESIDENT,
)


trait GbdtImage:
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        ...

    @staticmethod
    def row_major() -> Bool:
        ...

    @staticmethod
    def model_dim(text: String) raises -> Int:
        ...

    # The predict operations (lane dw-trees-boost, 2026-11-21). Addresses are
    # the caller's buffers as plain words; each method releases the GIL
    # around its own work.
    @staticmethod
    def predict(text: String, x: Int, n_rows: Int, dst: Int) raises -> Int:
        ...

    @staticmethod
    def predict_multi(text: String, x: Int, n_rows: Int, dst: Int, mode: Int) raises -> Int:
        ...

    @staticmethod
    def binary(probabilities: Bool, raw: Int, dst: Int, n: Int) raises -> Int:
        ...

    @staticmethod
    def sigmoid_pair(raw: Int, dst: Int, n: Int) raises -> Int:
        ...

    @staticmethod
    def resident() -> Bool:
        ...

    @staticmethod
    def resident_prepare(text: String) raises -> Int:
        ...

    @staticmethod
    def resident_predict(handle: Int, x: Int, n_rows: Int, dst: Int, mode: Int, row_major: Bool) raises -> Int:
        ...


def _addr(aa: Int, k: Int) -> Int:
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)[k]


def _fit_args(aa: Int, na: Int, pp: Int, np: Int, ff: Int) raises -> GbdtFitArgs:
    """The request spans as `GbdtFitArgs`: six addresses, the flag bytes
    (one per parameter, or 0 for none), then (address, bytes) per string;
    the int/float views."""
    if na < 7 or (na - 7) % 2 != 0:
        raise Error("gbdt_fit: the request needs six addresses, the flags and (address, bytes) per string")
    var strs = List[String]()
    var k = 7
    while k < na:  # driver-loop: over the four to eight option strings
        strs.append(gbdt_string_at(_addr(aa, k), _addr(aa, k + 1)))
        k += 2
    var ints = List[Int](capacity=np)
    var floats = List[Float64](capacity=np)
    var bad = List[UInt8](length=np, fill=0)
    var ip = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var flags = _addr(aa, 6)
    for i in range(np):  # small-loop(the fit's parameter slots): request words, not row data
        ints.append(ip[i])
        floats.append(fp[i])
        if flags:
            bad[i] = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=flags)[i]
    return GbdtFitArgs(_addr(aa, 0), _addr(aa, 1), _addr(aa, 2), _addr(aa, 3), _addr(aa, 4), _addr(aa, 5),
                       ints^, floats^, bad^, strs^)


def _int(pp: Int, k: Int) -> Int:
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)[k]


def _spans(what: StaticString, na: Int, need_a: Int, np: Int, need_p: Int) raises:
    if na != need_a or np != need_p:
        raise Error(String(what) + ": needs " + String(need_a) + " addresses and " + String(need_p)
                    + " parameters, got " + String(na) + " and " + String(np))


def _run_predict[I: GbdtImage](op: Int, aa: Int, na: Int, pp: Int, np: Int) raises -> Int:
    """The predict operations: the request words read, the image's method."""
    if op == XG_PREDICT:
        _spans("gbdt_predict", na, 3, np, 2)
        return I.predict(gbdt_string_at(_addr(aa, 0), _int(pp, 0)), _addr(aa, 1), _int(pp, 1), _addr(aa, 2))
    if op == XG_PREDICT_MULTI:
        _spans("gbdt_predict_multi", na, 3, np, 3)
        return I.predict_multi(gbdt_string_at(_addr(aa, 0), _int(pp, 0)), _addr(aa, 1), _int(pp, 1), _addr(aa, 2),
                               _int(pp, 2))
    if op == XG_BINARY:
        _spans("binary prediction", na, 2, np, 2)
        return I.binary(_int(pp, 1) != 0, _addr(aa, 0), _addr(aa, 1), _int(pp, 0))
    if op == XG_SIGMOID_PAIR:
        _spans("gbdt_sigmoid_pair", na, 2, np, 1)
        return I.sigmoid_pair(_addr(aa, 0), _addr(aa, 1), _int(pp, 0))
    if op == XG_RESIDENT_PREPARE:
        _spans("gbdt_resident_prepare", na, 1, np, 1)
        return I.resident_prepare(gbdt_string_at(_addr(aa, 0), _int(pp, 0)))
    _spans("gbdt_resident_predict", na, 2, np, 4)
    return I.resident_predict(_int(pp, 0), _addr(aa, 0), _int(pp, 1), _addr(aa, 1), _int(pp, 2), _int(pp, 3) != 0)


def _run[I: GbdtImage](op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int) raises -> Int:
    if op == XG_CAPS:
        return (XG_CAP_ROW_MAJOR if I.row_major() else 0) | (XG_CAP_RESIDENT if I.resident() else 0)
    if op >= XG_PREDICT and op <= XG_RESIDENT_PREDICT:
        return _run_predict[I](op, aa, na, pp, np)
    if op == XG_NUMERIC_MODE:
        return Int(GLOBAL_NUMERIC_MODE)
    if op == XG_FIT or op == XG_FIT_ROWMAJOR:
        var args = _fit_args(aa, na, pp, np, ff)
        return gbdt_outcome_box(I.fit(op == XG_FIT_ROWMAJOR, args))
    if op == XG_OUTCOME_SIZES:
        if na != 2:
            raise Error("gbdt outcome sizes: needs [outcome, out6]")
        gbdt_outcome_sizes(_addr(aa, 0), _addr(aa, 1))
        return 0
    if op == XG_OUTCOME_COPY:
        if na != 4:
            raise Error("gbdt outcome copy: needs [outcome, text, learn, test]")
        gbdt_outcome_copy(_addr(aa, 0), _addr(aa, 1), _addr(aa, 2), _addr(aa, 3))
        return 0
    if op == XG_OUTCOME_FREE:
        if na != 1:
            raise Error("gbdt outcome free: needs [outcome]")
        gbdt_outcome_free(_addr(aa, 0))
        return 0
    if op == XG_MODEL_DIM:
        if na != 1 or np != 1:
            raise Error("gbdt_model_dim: needs [text] and [text bytes]")
        return I.model_dim(gbdt_string_at(_addr(aa, 0), MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)[0]))
    raise Error("gbdt entry: unknown operation " + String(op))


def gbdt_entry_dispatch[I: GbdtImage](operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    """The body of a gbdt image's `abi("C")` entry: span admission, the
    operation, the count to result, a refusal to sink."""
    try:
        if operation < 0 or operation >= XG_OP_COUNT:
            raise Error("gbdt entry: unknown operation " + String(operation))
        if na < 0 or np < 0 or np != nf or not result:
            raise Error("gbdt entry: invalid request spans")
        if (na > 0 and not aa) or (np > 0 and (not pp or not ff)):
            raise Error("gbdt entry: null request span")
        var value = _run[I](operation, aa, na, pp, np, ff)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct GbdtEntryOwner(Defaultable, Movable):
    """An image-lifetime publisher word."""

    var word: Int

    def __init__(out self):
        self.word = 1


def gbdt_entry_publish[owner_name: StaticString](vendor: StaticString, entry: NativeDescriptorCall) raises:
    """Publish an image's gbdt entry under (its tier, the vendor word of the
    string it exports as `gbdt_vendor`). A build with no vendor word
    ("none") publishes nothing; its callers refuse with the table's
    "provider is not published"."""
    var word = gbdt_vendor_word(vendor)
    if word < 0:
        return
    comptime OWNER = SharedSlot[GbdtEntryOwner, owner_name]
    var owner = OWNER.get_or_create_ptr()
    var key = TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), word)
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    gbdt_table_publish(key, Int(owner), [String(GBDT_ENTRY_ABI)], [XG_OP_COUNT], entries)
