# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-identical-neural (2026-09-26): `chol_left_update_amma_kernel`
against its per-cell definition, bit for bit, on panels built to reach the
one window where Apple's flush-before-round FMA differs from the contract's
round-then-flush step.

Every cell (i, j) of the column block [j0, j0 + 32), i >= j, takes for each
panel p in order `c = ftz(ftz(c) - ftz(ftz(G_p)))`, with `G_p` the k = 32
chain `G = ftz(fma_rn(L[i][k], L[j][k], G))` from +0.0 over the panel's
columns, operands flushed. The host computes that with its correctly
rounded `fma` (round, then flush). The panels mix rows of ordinary words
with rows whose exponents sit near 2^-64, so products and partial sums
straddle 2^-126, and zero accumulators meet tiny products. A device that
ran the bare matrix chain on those cells would differ from the host; the
probe also reports how many cells the window actually moved (host rtf vs a
host flush-before-round chain), so a pass cannot be vacuous.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \\
        cholesky/checks/left_update_window_probe.mojo
"""

from std.math import fma
from std.memory import bitcast
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from checks.numerics import ftz
from cholesky.checks.potrf import CHOL_APPLE_LEFT, _chol_left_update


def _mix(x_in: UInt32) -> UInt32:
    var x = x_in
    x ^= x >> 16
    x *= UInt32(0x7FEB352D)
    x ^= x >> 15
    x *= UInt32(0x846CA68B)
    x ^= x >> 16
    return x


def _word(i: Int, k: Int, seed: Int) -> Float32:
    # Planted rows: one word per panel (column 5 of each 32), zeros
    # elsewhere. An even planted row holds (2 - 2^-23) 2^-64, an odd one
    # 2^-63, so an even-odd pair's product is exactly 2^-126 - 2^-150: rtf
    # rounds it (a tie, to even) to the smallest normal, flush-before-round
    # returns zero.
    if i % 7 == 3 + seed % 2:
        if k % 32 != 5:
            return Float32(0.0)
        if i % 2 == 0:
            return bitcast[DType.float32](UInt32(0x1FFFFFFF))
        return bitcast[DType.float32](UInt32(0x20000000))
    var h = _mix(UInt32(i * 7919 + k * 104729 + seed * 1299709 + 1))
    var h2 = _mix(h ^ UInt32(0x5BD1E995))
    var sign = h & UInt32(0x80000000)
    var mant = h & UInt32(0x007FFFFF)
    var kind = Int(_mix(UInt32(i * 31 + seed)) % 4)
    var e: Int
    if kind == 0:
        e = Int(h2 % 16) - 8 + 127          # ordinary
    elif kind == 1:
        e = Int(h2 % 3) - 65 + 127          # near 2^-64: products near 2^-128
    elif kind == 2:
        e = Int(h2 % 40) - 100 + 127        # tiny: underflowing products
    else:
        e = Int(h2 % 70) - 64 + 127         # wide
    if (h2 >> 20) % 16 == 0:
        return bitcast[DType.float32](sign)  # signed zeros
    return bitcast[DType.float32](sign | (UInt32(e) << 23) | mant)


def _fbr_step(a: Float32, b: Float32, c: Float32) -> Float32:
    """Flush-before-round, for the non-vacuity count only: the exact result
    below 2^-126 flushes to its signed zero before rounding."""
    var r = fma(a, b, c)
    var exact_small = Float64(a) * Float64(b) + Float64(c)
    if abs(exact_small) < 1.1754943508222875e-38 and abs(r) >= Float32(1.1754943508222875e-38):
        return bitcast[DType.float32](bitcast[DType.uint32](r) & UInt32(0x80000000))
    return ftz(r)


def main() raises:
    comptime assert has_apple_gpu_accelerator(), "Apple GPU probe"
    comptime assert CHOL_APPLE_LEFT, "the left-looking update is compiled out"
    var ctx = DeviceContext()
    var n = 320
    var nb = 32
    var total_bad = 0
    var total_moved = 0
    var total_cells = 0
    for seed in range(6):
        for jblock in [2, 5, 8]:
            var j0 = jblock * nb
            var np = jblock
            var h = List[Float32](length=n * n, fill=0)
            for i in range(n):
                for k in range(n):
                    h[i * n + k] = _word(i, k, seed)
            var d = ctx.enqueue_create_buffer[DType.float32](n * n)
            ctx.enqueue_copy(d, h.unsafe_ptr())
            ctx.synchronize()
            _chol_left_update(ctx, d, n, j0, nb, np)
            ctx.synchronize()
            var o = List[Float32](length=n * n, fill=0)
            ctx.enqueue_copy(o.unsafe_ptr(), d)
            ctx.synchronize()
            for i in range(j0, n):
                for j in range(j0, j0 + nb):
                    if j > i:
                        continue
                    var c = h[i * n + j]
                    var cf = c
                    for p in range(np):
                        var g = Float32(0.0)
                        var gf = Float32(0.0)
                        for k in range(p * nb, p * nb + nb):
                            var a = ftz(h[i * n + k])
                            var b = ftz(h[j * n + k])
                            g = ftz(fma(a, b, g))
                            gf = _fbr_step(a, b, gf)
                        c = ftz(ftz(c) - ftz(ftz(g)))
                        cf = ftz(ftz(cf) - ftz(ftz(gf)))
                    total_cells += 1
                    if bitcast[DType.uint32](c) != bitcast[DType.uint32](cf):
                        total_moved += 1
                    if bitcast[DType.uint32](o[i * n + j]) != bitcast[DType.uint32](c):
                        if total_bad < 5:
                            print("MISMATCH seed", seed, "j0", j0, "cell", i, j,
                                  "device", o[i * n + j], "host", c)
                        total_bad += 1
            _ = d^
    print("LEFT_WINDOW_PROBE cells", total_cells, "window_moved", total_moved, "mismatches", total_bad)
    if total_moved == 0:
        raise Error("vacuous: no cell's rtf chain differed from flush-before-round")
    if total_bad != 0:
        raise Error("chol_left_update_amma_kernel differs from the per-cell contract")
    print("== cholesky/checks/left_update_window_probe.mojo PASSED ==")
