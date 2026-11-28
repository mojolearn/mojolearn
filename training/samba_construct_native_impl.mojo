# SPDX-License-Identifier: Apache-2.0
"""Typed constant fill moved word for word from bindings/samba_construct_native.mojo."""


def samba_fill_ones(dst: MutPointer[Float32, MutUntrackedOrigin], count: Int):
    for j in range(count):
        dst[j] = 1.0
