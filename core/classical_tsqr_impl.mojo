# SPDX-License-Identifier: Apache-2.0
"""Typed host fills and copies moved word for word from bindings/classical_tsqr.mojo."""


def tsqr_fill_constant(data: MutPointer[Float32, MutUntrackedOrigin], count: Int, value: Float32):
    # Native host constant construction, matching the original input array.
    for i in range(count):
        data[i] = value


def tsqr_fill_indices(data: MutPointer[Float32, MutUntrackedOrigin], start: Int, stop: Int):
    # Native index metadata, bounded by the TSQR width (512).
    for i in range(stop - start):
        data[i] = Float32(start + i)


def tsqr_copy_columns(
    src: MutPointer[Float32, MutUntrackedOrigin], dst: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int, start: Int, width: Int,
):
    # Host ABI copy of existing host words; no arithmetic route changes.
    for r in range(rows):
        for c in range(width):
            dst[r * width + c] = src[r * cols + start + c]


def tsqr_fill_eye(data: MutPointer[Float32, MutUntrackedOrigin], m: Int, n: Int):
    for i in range(min(m, n)):
        data[i * (n + 1)] = 1.0
