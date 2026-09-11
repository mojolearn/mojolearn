# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2512: `core/device_zero.mojo` zeroes EXACTLY the span it is
given. Every byte inside is 0 afterwards, every byte outside keeps its
0xA5 fill, over offsets and lengths that exercise the aligned body, the
byte tail, the unaligned-start arm and the empty span. Run:

    pixi run mojo run -I . core/device_zero_check.mojo
"""
from max.gpu.host import DeviceContext
from core.device_zero import enqueue_zero_bytes


def check_span(ctx: DeviceContext, total: Int, offset: Int, nbytes: Int) raises -> Bool:
    var d = ctx.enqueue_create_buffer[DType.uint8](total)
    var h = ctx.enqueue_create_host_buffer[DType.uint8](total)
    for i in range(total):
        h[i] = UInt8(0xA5)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    var p = d.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    enqueue_zero_bytes(ctx, p + offset, nbytes)
    ctx.enqueue_copy(dst_buf=h, src_buf=d)
    ctx.synchronize()
    var bad = 0
    for i in range(total):
        var inside = i >= offset and i < offset + nbytes
        var want = UInt8(0) if inside else UInt8(0xA5)
        if h[i] != want:
            bad += 1
    if bad != 0:
        print("FAIL total", total, "offset", offset, "nbytes", nbytes, "bad bytes", bad)
    _ = d^
    _ = h^
    return bad == 0


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    var offsets = [0, 4, 16, 1, 3, 32]
    var lengths = [0, 1, 3, 4, 15, 16, 17, 31, 33, 4096, 4101, 1 << 20, (1 << 20) + 7]
    var n_cases = 0
    for o in range(len(offsets)):
        for l in range(len(lengths)):
            var total = offsets[o] + lengths[l] + 64
            ok = check_span(ctx, total, offsets[o], lengths[l]) and ok
            n_cases += 1
    # a large aligned span past the block cap (grid-stride path)
    ok = check_span(ctx, 64 << 20, 16, (64 << 20) - 48) and ok
    n_cases += 1
    if not ok:
        raise Error("device_zero_check FAILED")
    print("device_zero_check PASS,", n_cases, "spans")
