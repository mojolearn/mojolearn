# SPDX-License-Identifier: Apache-2.0
"""Small reductions of the classical family owners
(bindings/family_common.mojo): the minimum a chi2 kernel's negative-input
refusal reads, the maximum, and a constant label fill. The casts and the
finiteness scan that lived here are the one admission path
(core/admitted_input.mojo, core/host_finite_scan.mojo)."""


# TOMBSTONE: family_raw_word (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_raw_word`.


# TOMBSTONE: family_cast_rows (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f32`.


# TOMBSTONE: family_cast_rows_finite (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f32`.


def family_min_f32(src: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Float32:
    """The minimum of `count` float32 words (count >= 1); a NaN never wins,
    as `Array.min` after a NaN-free cast."""
    var lo = src[0]
    for i in range(1, count):  # native input marshaling: one reduction over the checked input
        if src[i] < lo:
            lo = src[i]
    return lo


# TOMBSTONE: family_cast_rows_f64 (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f64`.


# TOMBSTONE: family_all_finite_f32 (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/host_finite_scan.mojo `host_all_finite_f32`.


def family_max_f32(src: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Float32:
    """The maximum of `count` float32 words (count >= 1)."""
    var hi = src[0]
    for i in range(1, count):  # native input marshaling: one reduction over the checked input
        if src[i] > hi:
            hi = src[i]
    return hi


def family_fill_i32(out_buf: MutPointer[Int32, MutUntrackedOrigin], n: Int, value: Int32):
    for i in range(n):  # native output marshaling: one constant label per row
        out_buf[i] = value
