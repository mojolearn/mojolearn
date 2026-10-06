# SPDX-License-Identifier: Apache-2.0
"""NI14 bounded cooperative col2im pages, source-only/unqualified.

Each dcols value contributes to one input pixel, so caching it for reuse by
multiple output pixels would not remove useful loads. This arm instead loads
adjacent tap words cooperatively, then serves each pixel's canonical gather
from shared memory. It trades halo traffic/barriers for coalesced reads.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from x_cnn.ops import (
    FP, IP, CP_C, CP_H, CP_W, CP_KH, CP_KW, CP_OH, CP_OW,
    CP_SH, CP_SW, CP_PH, CP_PW, CP_DH, CP_DW, CP_REV,
    col2im_bounded_at,
)

comptime NI14_TILED_COL2IM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI14_TILED_COL2IM"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
# 32 output pixels per block; 8 contiguous tap words per staged source
# position. At most 64 positions * 8 words = 2 KiB shared memory. The
# launch uses 256 threads to load a page in at most two passes. These are
# memory-transaction/shared-storage choices, independent of dataset sizes.
comptime PIXELS = 32
comptime TAPS = 8
comptime SOURCE_POSITIONS = 64
comptime THREADS = 256


def _col2im_pages(dcols: FP, dx: FP, p: IP, count_in: Int32):
    var page = stack_allocation[SOURCE_POSITIONS * TAPS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var begin = Int(block_idx.x) * PIXELS
    var count = Int(count_in)
    var C = Int(p.unsafe_load(CP_C)); var H = Int(p.unsafe_load(CP_H)); var W = Int(p.unsafe_load(CP_W))
    var KH = Int(p.unsafe_load(CP_KH)); var KW = Int(p.unsafe_load(CP_KW))
    var OH = Int(p.unsafe_load(CP_OH)); var OW = Int(p.unsafe_load(CP_OW))
    var SH = Int(p.unsafe_load(CP_SH)); var SW = Int(p.unsafe_load(CP_SW))
    var PH = Int(p.unsafe_load(CP_PH)); var PW = Int(p.unsafe_load(CP_PW))
    var DH = Int(p.unsafe_load(CP_DH)); var DW = Int(p.unsafe_load(CP_DW))
    var w0 = begin % W
    # A page covers the source positions needed by this spatial strip and
    # its active tap window. This algebraic bound follows dilation/stride;
    # it is not a crossover tuned to a benchmark. Boundary strips that
    # cross rows use the existing address-bounded kernel per pixel.
    var max_positions = (PIXELS - 1 + (min(TAPS, KW) - 1) * DW + SW - 1) // SW + 1
    if w0 + PIXELS > W or max_positions > SOURCE_POSITIONS:
        if tid < PIXELS and begin + tid < count:
            col2im_bounded_at(begin + tid, dcols, dx, dx, dx, p, p)
        return  # block-uniform before any barrier
    var plane_row = begin // W
    var h = plane_row % H
    var nc = plane_row // H
    var c = nc % C; var n = nc // C
    var ckk = C * KH * KW
    var acc = Float32(0)
    for a in range(KH):
        var kh = KH - 1 - a if Int(p.unsafe_load(CP_REV)) != 0 else a
        var th = h + PH - kh * DH
        if th < 0 or th % SH != 0:
            continue
        var oh = th // SH
        if oh >= OH:
            continue
        var tap_begin = 0
        while tap_begin < KW:
            var tap_end = min(tap_begin + TAPS, KW)
            # For this row strip and tap window, source ow ranges from
            # ceil((w0+PW-last_tap*DW)/SW) to floor((last_w+PW-first_tap*DW)/SW).
            var lo_num = w0 + PW - (tap_end - 1) * DW
            var ow_lo = (max(lo_num, 0) + SW - 1) // SW
            var hi_num = w0 + PIXELS - 1 + PW - tap_begin * DW
            var ow_hi = min(OW, hi_num // SW + 1) if hi_num >= 0 else 0
            var positions = max(0, ow_hi - ow_lo)
            var word = tid
            while word < positions * TAPS:
                var ow = ow_lo + word // TAPS
                var kw = tap_begin + word % TAPS
                var value = Float32(0)
                if kw < tap_end:
                    var row = (n * OH + oh) * OW + ow
                    value = dcols.unsafe_load(row * ckk + (c * KH + kh) * KW + kw)
                page[word] = value
                word += THREADS
            barrier()
            if tid < PIXELS and begin + tid < count:
                var w = w0 + tid
                for kw in range(tap_begin, tap_end):
                    var tw = w + PW - kw * DW
                    if tw < 0 or tw % SW != 0:
                        continue
                    var ow = tw // SW
                    if ow >= OW:
                        continue
                    # Each pixel adds kh/kw in exactly the reference order,
                    # including CP_REV. Tiling never creates partial sums.
                    var value = page[(ow - ow_lo) * TAPS + kw - tap_begin]
                    acc = ftz(acc + ftz(value))
            barrier()
            tap_begin = tap_end
    if tid < PIXELS and begin + tid < count:
        dx.unsafe_store(begin + tid, acc)


def neural_col2im_tiled(
    ctx: DeviceContext, mut dcols: DeviceBuffer[DType.float32],
    mut dx: DeviceBuffer[DType.float32], mut parameters: DeviceBuffer[DType.int32], count: Int,
) raises:
    if count <= 0:
        return
    ctx.enqueue_function[_col2im_pages](
        dcols.unsafe_ptr(), dx.unsafe_ptr(), parameters.unsafe_ptr(), Int32(count),
        grid_dim=((count + PIXELS - 1) // PIXELS, 1, 1), block_dim=(THREADS, 1, 1),
    )
