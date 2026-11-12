# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of a gbdt image's entry (lane calltable-gbdt-forest-dart,
2026-11-11; ABI gbdt/fit_request.mojo, table core/gbdt_entry_table.mojo).
No Python: the caller passes plain values and keeps its buffers alive until
the call returns. The caller resolves the table once per fit (`xg_ref`; a
missing or mismatched publication refuses there with the table's error)
and every call is one indirect call through the `EntryRef` (lane
entry-table-zero-alloc); a refusal of the fit comes back in the words the
fit raised.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word
from core.trees_entry_table import TreesEntryKey
from core.gbdt_entry_table import gbdt_table_entry
from gbdt.fit_request import (
    GbdtFitArgs, GbdtFitOutcome, GBDT_ENTRY_ABI, XG_OP_COUNT, XG_CAPS, XG_NUMERIC_MODE, XG_FIT, XG_FIT_ROWMAJOR,
    XG_OUTCOME_SIZES, XG_OUTCOME_COPY, XG_OUTCOME_FREE, XG_MODEL_DIM, XG_CAP_ROW_MAJOR, gbdt_string_at,
)


def xg_ref(key: TreesEntryKey) raises -> EntryRef:
    """The gbdt entry of the image the key names, resolved once."""
    return EntryRef(gbdt_table_entry(key, 0, String(GBDT_ENTRY_ABI), XG_OP_COUNT),
                    entry_ref_word(key.mode, key.vendor, XG_OP_COUNT))


def _none_f() -> List[Float64]:
    return List[Float64]()


def xg_row_major(entry: EntryRef) raises -> Bool:
    """The former `gbdt_fit_row_major_available` probe (absent = False)."""
    var none = List[Int]()
    return (entry_ref_call(entry, XG_CAPS, none, none, _none_f()) & XG_CAP_ROW_MAJOR) != 0


def xg_numeric_mode(entry: EntryRef) raises -> Int:
    """The image's compiled tier (the former `gbdt_numeric_mode`)."""
    var none = List[Int]()
    return entry_ref_call(entry, XG_NUMERIC_MODE, none, none, _none_f())


def xg_model_dim(entry: EntryRef, text: String) raises -> Int:
    """The model's approx dimension (the former `gbdt_model_dim`)."""
    var addrs: List[Int] = [Int(text.unsafe_ptr())]
    var ints: List[Int] = [text.byte_length()]
    var floats: List[Float64] = [Float64(text.byte_length())]
    var dim = entry_ref_call(entry, XG_MODEL_DIM, addrs, ints, floats)
    _ = text
    return dim


def xg_fit(entry: EntryRef, row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
    """The fit in the image the key names, its outcome copied out and freed
    in that image (the former `gbdt_fit` / `gbdt_fit_rowmajor` entries)."""
    var addrs: List[Int] = [args.x, args.y, args.weights, args.cat_flags, args.eval_x, args.eval_y,
                            Int(args.bad.unsafe_ptr()) if len(args.bad) else 0]
    for k in range(len(args.strs)):  # small-loop(strs: four to eight option names)
        addrs.append(Int(args.strs[k].unsafe_ptr()))
        addrs.append(args.strs[k].byte_length())
    var handle = entry_ref_call(entry, XG_FIT_ROWMAJOR if row_major else XG_FIT, addrs, args.ints,
                                        args.floats)
    _ = args  # the strings, views and flags stay alive through the call above
    var sizes: List[Int] = [0, 0, 0, 0, 0, 0]
    var outcome_addrs: List[Int] = [handle, Int(sizes.unsafe_ptr())]
    var none = List[Int]()
    try:
        _ = entry_ref_call(entry, XG_OUTCOME_SIZES, outcome_addrs, none, _none_f())
        # The model text is copied ONCE, straight from the image's outcome
        # String (sizes[5], valid until XG_OUTCOME_FREE below) into ours; the
        # former path copied it into a byte List here, then into a second
        # List and into the String in `gbdt_string_at` (round2-fixes,
        # 2026-11-11; perf audit 2 A pattern 7).
        var text = gbdt_string_at(sizes[5], sizes[0])
        var learn = List[Float64](length=sizes[3], fill=0.0)
        var test = List[Float64](length=sizes[4], fill=0.0)
        var copy_addrs: List[Int] = [handle, 0, Int(learn.unsafe_ptr()), Int(test.unsafe_ptr())]
        _ = entry_ref_call(entry, XG_OUTCOME_COPY, copy_addrs, none, _none_f())
        return GbdtFitOutcome(text^, sizes[1], sizes[2] != 0, learn^, test^)
    finally:
        var free_addrs: List[Int] = [handle]
        _ = entry_ref_call(entry, XG_OUTCOME_FREE, free_addrs, none, _none_f())
        _ = sizes
