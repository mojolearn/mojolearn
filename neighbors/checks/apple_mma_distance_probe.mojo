# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-identical-neural (2026-09-27): `apple_mma_distance_tile_kernel`
against `pinned_distance_register_tile_kernel` (the shipped register tile),
bit for bit, on vectors built to reach the one window where Apple's
flush-before-round differs from the contract's round-then-flush.

Planted pairs: a query (2 - 2^-23) 2^-64 and an index word 2^-63 in the same
feature, the rest zero, so the dot is exactly 2^-126 - 2^-150 (rtf: the
smallest normal; flush-before-round: zero), the query norm flushes to zero
and the index norm is 2^-126: the correct distance clamps to 0 and the
flush-before-round one does not. Ordinary rows fill the rest; ragged
shapes and several feature counts cover the padding. A kernel that took the
matrix value on a refused cell fails here (checked by a source sabotage).

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \\
        neighbors/checks/apple_mma_distance_probe.mojo
"""

from std.memory import bitcast
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from checks.numerics import ftz, identical_mul_add
from neighbors.checks.apple_mma_distance import (
    AMD_BM, AMD_BN, AMD_NT, apple_mma_distance_tile_kernel,
)
from neighbors.checks.pinned_distance_tile import (
    RT_ROWS, RT_TILE_COLS, RT_TPB, pinned_distance_register_tile_kernel,
)


def _mix(x_in: UInt32) -> UInt32:
    var x = x_in
    x ^= x >> 16
    x *= UInt32(0x7FEB352D)
    x ^= x >> 15
    x *= UInt32(0x846CA68B)
    x ^= x >> 16
    return x


def _ord(i: Int, f: Int, seed: Int) -> Float32:
    var h = _mix(UInt32(i * 7919 + f * 104729 + seed * 13 + 1))
    var e = Int(_mix(h) % 12) + 120
    return bitcast[DType.float32]((h & UInt32(0x80000000)) | (UInt32(e) << 23) | (h & UInt32(0x7FFFFF)))


def main() raises:
    comptime assert has_apple_gpu_accelerator(), "Apple GPU probe"
    var ctx = DeviceContext()
    var total = 0
    var bad = 0
    var planted_cells = 0
    var ds: List[Int] = [16, 8, 13, 32, 3, 24]
    var nrs: List[Int] = [64, 130, 77, 128, 40, 96]
    var ncs: List[Int] = [128, 200, 131, 256, 70, 192]
    for cs in range(6):
        var d = ds[cs]
        var nr = nrs[cs]
        var nc = ncs[cs]
        var hq = List[Float32](length=nr * d, fill=0)
        var hy = List[Float32](length=d * nc, fill=0)   # feature-major, stride nc
        for i in range(nr):
            if i % 5 == 1:
                hq[i * d + (i % d)] = bitcast[DType.float32](UInt32(0x1FFFFFFF))
            else:
                for f in range(d):
                    hq[i * d + f] = _ord(i, f, cs)
        for j in range(nc):
            if j % 3 == 2:
                for f in range(d):
                    hy[f * nc + j] = bitcast[DType.float32](UInt32(0x20000000)) if f == (j % d) else Float32(0.0)
            else:
                for f in range(d):
                    hy[f * nc + j] = _ord(j + 5000, f, cs)
        # norms as the estimator computes them: the pinned ascending chain
        var qn = List[Float32](length=nr, fill=0)
        var yn = List[Float32](length=nc, fill=0)
        for i in range(nr):
            var t = Float32(0.0)
            for f in range(d):
                var v = ftz(hq[i * d + f])
                t = ftz(identical_mul_add(v, v, t))
            qn[i] = t
        for j in range(nc):
            var t = Float32(0.0)
            for f in range(d):
                var v = ftz(hy[f * nc + j])
                t = ftz(identical_mul_add(v, v, t))
            yn[j] = t
        for i in range(nr):
            for j in range(nc):
                if i % 5 == 1 and j % 3 == 2 and (i % d) == (j % d):
                    planted_cells += 1
        var dq = ctx.enqueue_create_buffer[DType.float32](nr * d)
        var dy = ctx.enqueue_create_buffer[DType.float32](d * nc)
        var dqn = ctx.enqueue_create_buffer[DType.float32](nr)
        var dyn = ctx.enqueue_create_buffer[DType.float32](nc)
        var z1 = ctx.enqueue_create_buffer[DType.float32](nr * nc)
        var z2 = ctx.enqueue_create_buffer[DType.float32](nr * nc)
        var dm = ctx.enqueue_create_buffer[DType.float32](max(nr, nc))
        var dm2 = ctx.enqueue_create_buffer[DType.float32](max(nr, nc))
        ctx.enqueue_copy(dq, hq.unsafe_ptr())
        ctx.enqueue_copy(dy, hy.unsafe_ptr())
        ctx.enqueue_copy(dqn, qn.unsafe_ptr())
        ctx.enqueue_copy(dyn, yn.unsafe_ptr())
        for is_sqrt in range(2):
            ctx.enqueue_function[apple_mma_distance_tile_kernel](
                z1.unsafe_ptr(), dq.unsafe_ptr(), dy.unsafe_ptr(), dqn.unsafe_ptr(), dyn.unsafe_ptr(),
                Int32(nr), Int32(nc), Int32(nc), Int32(d), Int32(is_sqrt),
                grid_dim=((nc + AMD_BN - 1) // AMD_BN, (nr + AMD_BM - 1) // AMD_BM, 1),
                block_dim=(AMD_NT, 1, 1),
            )
            ctx.enqueue_function[pinned_distance_register_tile_kernel[False]](
                z2.unsafe_ptr(), dq.unsafe_ptr(), dy.unsafe_ptr(), dqn.unsafe_ptr(), dyn.unsafe_ptr(),
                dm.unsafe_ptr(), dm2.unsafe_ptr(),
                Int32(nr), Int32(nc), Int32(nc), Int32(d), Int32(is_sqrt),
                grid_dim=((nc + RT_TILE_COLS - 1) // RT_TILE_COLS, (nr + RT_ROWS - 1) // RT_ROWS, 1),
                block_dim=(RT_TPB, 1, 1),
            )
            ctx.synchronize()
            var h1 = List[Float32](length=nr * nc, fill=0)
            var h2 = List[Float32](length=nr * nc, fill=0)
            ctx.enqueue_copy(h1.unsafe_ptr(), z1)
            ctx.enqueue_copy(h2.unsafe_ptr(), z2)
            ctx.synchronize()
            for c in range(nr * nc):
                total += 1
                if bitcast[DType.uint32](h1[c]) != bitcast[DType.uint32](h2[c]):
                    if bad < 5:
                        print("MISMATCH cs", cs, "cell", c // nc, c % nc, "mma", h1[c], "register", h2[c])
                    bad += 1
    print("APPLE_MMA_DIST_PROBE cells", total, "planted_pairs", planted_cells, "mismatches", bad)
    if planted_cells == 0:
        raise Error("vacuous: no planted window pair")
    if bad != 0:
        raise Error("apple_mma_distance_tile_kernel differs from the register tile")
    print("== neighbors/checks/apple_mma_distance_probe.mojo PASSED ==")
