# SPDX-License-Identifier: Apache-2.0
"""Typed per-row length copies moved word for word from bindings/neural_block_public.mojo."""
from std.memory import memcpy
from core.hostptr import f32_ptr


def neural_copy_row_prefixes(dst_addr: Int, src_addr: Int, sizes: List[Int], b: Int, l: Int, d: Int) raises:
    for row in range(b):
        memcpy(dest=f32_ptr(dst_addr) + row * l * d, src=f32_ptr(src_addr) + row * l * d, count=sizes[row] * d)
