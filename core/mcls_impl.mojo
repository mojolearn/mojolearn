# SPDX-License-Identifier: Apache-2.0
"""Typed input marshaling of the classification metrics owner
(bindings/metrics_cls_native.mojo, lane mcb2-mcls): the label range `lab.min()` / `lab.max()` the entropy entry takes,
and the copies of a kernel's leading outputs (`precision[:size + 1].copy()`).
No arithmetic on metric values happens here: the kernels compute."""


# TOMBSTONE: mcls_read_i64 and mcls_narrow_i32 (serial byte-walk label reads, merged into admit_ids) deleted 2026-11-08 by lane a3-admit-decomp-prep-metrics; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A3_ADMISSION_HELPERS.patch. They are
# `admitted_ids_i64` and `admitted_narrow_i32` (core/admitted_ids.mojo).


def mcls_min_i32(src: MutPointer[Int32, MutUntrackedOrigin], count: Int) -> Int:
    """`lab.min()` of `count` >= 1 int32 labels."""
    var lo = src[0]
    for i in range(1, count):  # native input marshaling: one reduction over the checked labels
        if src[i] < lo:
            lo = src[i]
    return Int(lo)


def mcls_max_i32(src: MutPointer[Int32, MutUntrackedOrigin], count: Int) -> Int:
    """`lab.max()` of `count` >= 1 int32 labels."""
    var hi = src[0]
    for i in range(1, count):  # native input marshaling: one reduction over the checked labels
        if src[i] > hi:
            hi = src[i]
    return Int(hi)


def mcls_copy_words(src: Int, dst: Int, words: Int):
    """Copy `words` 4-byte words (a kernel output's kept prefix)."""
    var s = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=src)
    var d = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(words):  # native output marshaling: the kept prefix of a kernel output
        d[i] = s[i]
