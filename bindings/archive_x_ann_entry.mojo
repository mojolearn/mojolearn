# SPDX-License-Identifier: Apache-2.0
"""The GPU x_ann image's entry of the archive export table
(core/archive_entry_table.mojo, family ARCHIVE_TABLE_X_ANN; lane
calltable-parallel-archive, 2026-11-11).

The body of the `x_ann_index_export_codes` door (which stays for its Python
caller in `_expansion_ann.py`), reached natively by the archive writer
(bindings/archive_schema_native.mojo) when it saves a resident IVF-PQ
index's codes. Published only where the door is registered
(`IVF_PQ_RESIDENT_FIT`).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.archive_entry_table import ARCHIVE_TABLE_X_ANN, archive_check_spans, archive_table_publish
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.parallel_entry_table import ParallelEntryKey, parallel_entry_vendor
from core.shared_slot import SharedSlot
from x_ann.resident import x_ann_index_export_codes


def archive_x_ann_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        archive_check_spans(operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        x_ann_index_export_codes(p[0], a[0], p[1])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ArchiveXAnnOwner(Defaultable, Movable):
    """The image-lifetime publisher word of this image's archive export."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime ARCHIVE_X_ANN_OWNER = SharedSlot[ArchiveXAnnOwner,
    "MojolearnArchiveXAnnOwnerIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnArchiveXAnnOwnerFastV1"]


def publish_archive_x_ann() raises:
    """Publish at module init of `_mojolearn_x_ann` under its compiled vendor
    (none for a build with no GPU vendor)."""
    var word = parallel_entry_vendor(COMPILED_VENDOR)
    if word < 1:
        return
    var owner = ARCHIVE_X_ANN_OWNER.get_or_create_ptr()
    archive_table_publish(ARCHIVE_TABLE_X_ANN, ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), word), Int(owner),
                          archive_x_ann_v1)
