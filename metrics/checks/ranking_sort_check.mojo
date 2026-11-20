# SPDX-License-Identifier: Apache-2.0
"""Device gate for the ranking metric's sort: the eight-bit-digit stable sort
(`core/stable_radix_sort.mojo`) and the one-bit sort
(`gbdt/gpu_util/kernel/radix_sort.mojo`) give the same pairs in the same
order over the full 32-bit key, on keys with heavy ties, high bits set and
lengths that are not a multiple of either sort's tile.

    pixi run mojo run -I . metrics/checks/ranking_sort_check.mojo
"""
from max.gpu.host import DeviceContext

from core.stable_radix_sort import (
    stable_radix_bsum_len, stable_radix_counts_len, stable_radix_sort_pairs_u32,
)
from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins


def _check(ctx: DeviceContext, n: Int, distinct: Int) raises:
    # Heavy ties: `distinct` key values over n rows, spread over all four
    # digits; the value is the input position, so a stable sort is unique.
    var keys = List[UInt32]()
    var values = List[UInt32]()
    for i in range(n):
        var g = (i * 40503) % distinct
        keys.append(UInt32(g) * UInt32(0x01010101) + (UInt32(0x80000000) if g % 3 == 0 else UInt32(0)))
        values.append(UInt32(i))
    var base_k = List[UInt32]()
    var base_v = List[UInt32]()
    # driver-loop: over the two sorts
    for arm in range(2):
        var dk = ctx.enqueue_create_buffer[DType.uint32](n)
        var dv = ctx.enqueue_create_buffer[DType.uint32](n)
        var tk = ctx.enqueue_create_buffer[DType.uint32](n)
        var tv = ctx.enqueue_create_buffer[DType.uint32](n)
        var counts = ctx.enqueue_create_buffer[DType.int32](stable_radix_counts_len(n))
        # one slot per 512-element block for the one-bit sort, the scan
        # sums for the eight-bit sort: the larger of the two.
        var one_bit_blocks = (n + 511) // 512
        var sort_bsum = stable_radix_bsum_len(n)
        var bsum = ctx.enqueue_create_buffer[DType.int32](
            one_bit_blocks if one_bit_blocks > sort_bsum else sort_bsum
        )
        ctx.enqueue_copy(dst_buf=dk, src_ptr=keys.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dv, src_ptr=values.unsafe_ptr())
        if arm == 0:
            launch_radix_sort_bins(ctx, n, 0, 32, dk, dv, tk, tv, counts, bsum)
        else:
            stable_radix_sort_pairs_u32(ctx, n, 32, dk, dv, tk, tv, counts, bsum)
        var hk = ctx.enqueue_create_host_buffer[DType.uint32](n)
        var hv = ctx.enqueue_create_host_buffer[DType.uint32](n)
        ctx.enqueue_copy(dst_ptr=hk.unsafe_ptr(), src_buf=dk)
        ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=dv)
        ctx.synchronize()
        for i in range(n):
            if keys[Int(hv[i])] != hk[i]:
                raise Error("ranking sort: a key lost its value")
            if i > 0 and (hk[i] < hk[i - 1] or (hk[i] == hk[i - 1] and hv[i] <= hv[i - 1])):
                raise Error("ranking sort: order is not the stable order by key")
            if arm == 0:
                base_k.append(hk[i])
                base_v.append(hv[i])
            elif base_k[i] != hk[i] or base_v[i] != hv[i]:
                raise Error("ranking sort: eight-bit and one-bit sorts disagree")
        _ = dk^
        _ = dv^
        _ = tk^
        _ = tv^
        _ = counts^
        _ = bsum^
        _ = hk^
        _ = hv^
    _ = keys^
    _ = values^


def main() raises:
    var ctx = DeviceContext()
    var sizes: List[Int] = [1, 2, 255, 257, 513, 4099, 70001]
    for n in sizes:
        _check(ctx, n, 1)
        _check(ctx, n, 2)
        _check(ctx, n, 7)
        _check(ctx, n, 251)
    print("PASS ranking sort: eight-bit and one-bit stable sorts give the same pairs (28 cases, heavy ties)")
