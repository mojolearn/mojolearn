# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Counts of the host-to-device copies of FITTED operands, per family, for the
check "a second identical call uploads nothing"
(checks/native_tests/resident_second_call.mojo; lane fx-missing-checks,
2026-11-19).

EVERYTHING HERE IS COMPILED ONLY UNDER `-D MOJOLEARN_RESIDENT_UPLOAD_COUNT=1`,
the define of the native checks' GPU build (pixi `check-native-gpu`,
tools/ops/native_checks_gpu_job.sh). In every other build, the wheels
included, `resident_upload_note` is an empty inline function and the readers
return 0: no counter, no slot, no lock, no instruction at the call site.

A family calls `resident_upload_note(RUC_<FAMILY>, cells)` at the one place
where a fitted operand (training rows, centers, an index, knots, a weight
block, a registry, optimizer state) is copied from host memory to the device:
the miss path of its resident store, or its handle's constructor. A query
upload is never noted. The check reads the totals around a call.

Each note is two atomic integer adds on the calling thread. The families'
upload sites are driver code that runs once per public call; the slot is
shared by every module in the process (core/shared_slot.mojo), and two calls
noting the same family from two threads both count (lane arena-session,
2026-11-23: the adds were plain `+=`, which could lose one).
"""
from std.atomic import Atomic, Ordering
from std.sys.compile import is_defined
from core.shared_slot import SharedSlot

comptime RESIDENT_UPLOAD_COUNT = is_defined["MOJOLEARN_RESIDENT_UPLOAD_COUNT"]()

#: The families (an index into the two lists below).
comptime RUC_KDE = 0
comptime RUC_KNN_INDEX = 1
comptime RUC_KMEANS_CENTERS = 2
comptime RUC_OPTIMIZER_STATE = 3
comptime RUC_X_LINEAR = 4
comptime RUC_ISOTONIC = 5
comptime RUC_DECOMP_KIT = 6
comptime RUC_FAMILIES = 7


struct _ResidentUploadCounts(Defaultable, Movable):
    #: one Int64 word per family each, changed only by atomic adds
    var copies: List[Int64]
    var cells: List[Int64]

    def __init__(out self):
        self.copies = List[Int64](length=RUC_FAMILIES, fill=Int64(0))
        self.cells = List[Int64](length=RUC_FAMILIES, fill=Int64(0))


@always_inline
def _ruc_word(mut words: List[Int64], family: Int) -> MutPointer[Int64, MutUntrackedOrigin]:
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr())) + family


# Referenced only inside the comptime branches below, so a build without the
# define never creates the slot.
comptime _RUC_SLOT = SharedSlot[_ResidentUploadCounts, "MojolearnResidentUploadCountsV2"]


@always_inline
def resident_upload_note(family: Int, cells: Int) raises:
    """One host-to-device copy of `cells` fitted words of `family`."""
    comptime if RESIDENT_UPLOAD_COUNT:
        var slot = _RUC_SLOT.get_or_create_ptr()
        _ = Atomic[Int64].fetch_add(_ruc_word(slot[].copies, family), Int64(1))
        _ = Atomic[Int64].fetch_add(_ruc_word(slot[].cells, family), Int64(cells))
    return


def resident_upload_copies(family: Int) raises -> Int:
    """The family's fitted copies since process start (0 without the define)."""
    comptime if RESIDENT_UPLOAD_COUNT:
        var slot = _RUC_SLOT.get_or_create_ptr()
        return Int(Atomic.load[ordering=Ordering.ACQUIRE](_ruc_word(slot[].copies, family)))
    return 0

