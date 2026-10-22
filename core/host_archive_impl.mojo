# SPDX-License-Identifier: Apache-2.0
# Saved-state layout copies preserve every stored float32 bit, including
# NaN payloads. They perform no numerical normalization or validation.


def host_archive_copy_f32(source: Int, stride0: Int, stride1: Int, rows: Int, cols: Int,
                          destination: Int):
    var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=destination)
    for row in range(rows):
        for col in range(cols):
            var src = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source + row * stride0 + col * stride1)
            dst[row * cols + col] = src[0]
