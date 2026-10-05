# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A stable LSD radix sort of (u32 key, u32 value) pairs over the LOW
`key_bits` bits of the key, for any tier (lane nr-small D5, 2026-10-04).

It runs `core/fast_radix_sort`'s own kernels (digit counts per tile, the
multi-block exclusive scan, the stable scatter; unchanged, imported) for
only as many 8-bit passes as the keys need, rounded up to an even count so
the answer ends where it started. A pass over a digit that is zero in every
key is the identity permutation (stable, one bin), so skipping it changes
nothing: the result is the unique stable order by key, the same pairs in
the same order as the four-pass sort. Integer work only, no atomics, no
floats: every vendor and the host twin agree on every word, which is why
an IDENTICAL caller may use it (the graph CSR build in `x_cnn/device.mojo`
already sorts with the four-pass form under IDENTICAL).

Every key MUST be below `2^key_bits` (the caller's guarantee).
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.fast_radix_sort import (
    FRS_BINS,
    FRS_TILE,
    _frs_count_kernel,
    _frs_scatter_kernel,
    frs_counts_len,
    frs_exclusive_scan,
    frs_scan_blocks,
)


def stable_radix_passes(key_bits: Int) -> Int:
    """8-bit passes for keys below `2^key_bits`: ceil(key_bits / 8),
    rounded up to an even count (2 or 4)."""
    if key_bits <= 16:
        return 2
    return 4


def key_bits_for(max_key: Int) -> Int:
    """The bit length of `max_key` (0 for 0): every key `<= max_key` is
    below `2^key_bits_for(max_key)`."""
    var b = 0
    var v = max_key
    while v > 0:
        b += 1
        v = v >> 1
    return b


def stable_radix_counts_len(size: Int) -> Int:
    """int32 slots `counts` must hold for `size` elements."""
    return frs_counts_len(size)


def stable_radix_sort_pairs_u32(
    ctx: DeviceContext,
    size: Int,
    key_bits: Int,
    mut keys: DeviceBuffer[DType.uint32],
    mut values: DeviceBuffer[DType.uint32],
    mut temp_keys: DeviceBuffer[DType.uint32],
    mut temp_values: DeviceBuffer[DType.uint32],
    mut counts: DeviceBuffer[DType.int32],
    mut bsum: DeviceBuffer[DType.int32],
) raises:
    """Sort `keys[0:size]` ascending carrying `values`, stably, in place,
    over the low `key_bits` bits (`stable_radix_passes(key_bits)` passes of
    five launches). `counts` holds at least `stable_radix_counts_len(size)`
    slots and `bsum` at least `stable_radix_bsum_len(size)`; both are the
    caller's, so nothing is allocated or freed here and the caller decides
    when to wait. Enqueued only; nothing waits."""
    if size <= 0:
        return
    var n_tiles = (size + FRS_TILE - 1) // FRS_TILE
    var m = FRS_BINS * n_tiles
    var passes = stable_radix_passes(key_bits)
    for p in range(passes):
        var shift = Int32(8 * p)
        if p % 2 == 0:
            ctx.enqueue_function[_frs_count_kernel](
                keys.unsafe_ptr(), counts.unsafe_ptr(), Int32(size), shift,
                Int32(n_tiles), grid_dim=n_tiles, block_dim=FRS_TILE,
            )
        else:
            ctx.enqueue_function[_frs_count_kernel](
                temp_keys.unsafe_ptr(), counts.unsafe_ptr(), Int32(size), shift,
                Int32(n_tiles), grid_dim=n_tiles, block_dim=FRS_TILE,
            )
        frs_exclusive_scan(ctx, counts, m, bsum)
        if p % 2 == 0:
            ctx.enqueue_function[_frs_scatter_kernel](
                keys.unsafe_ptr(), values.unsafe_ptr(), temp_keys.unsafe_ptr(),
                temp_values.unsafe_ptr(), counts.unsafe_ptr(), Int32(size),
                shift, Int32(n_tiles), grid_dim=n_tiles, block_dim=FRS_TILE,
            )
        else:
            ctx.enqueue_function[_frs_scatter_kernel](
                temp_keys.unsafe_ptr(), temp_values.unsafe_ptr(), keys.unsafe_ptr(),
                values.unsafe_ptr(), counts.unsafe_ptr(), Int32(size),
                shift, Int32(n_tiles), grid_dim=n_tiles, block_dim=FRS_TILE,
            )


def stable_radix_bsum_len(size: Int) -> Int:
    """int32 slots `bsum` must hold for `size` elements (at least one)."""
    var n_tiles = (size + FRS_TILE - 1) // FRS_TILE
    var nb = frs_scan_blocks(FRS_BINS * n_tiles)
    return nb if nb > 0 else 1
