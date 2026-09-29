# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/linfit-speed: what ONE device thread's fold step costs (ns per step)
for the spellings the linear fits use, over n rows of a row-major n x d
matrix (column 0) and an n vector. One thread, one launch each.

    MOJOLEARN_NUMERIC_MODE=identical pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
        bench/results/linfit_speed/chain_bench.mojo
"""
from std.gpu import thread_idx
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from x_linear.ops import FP, fz, xmad, ld, st, fmad
from x_linear.tops import chain_fmad, chain_fmad_scaled, fold_fa


def k_plain(x: FP, v: FP, res: FP, n: Int, d: Int, mode: Int):
    if Int(thread_idx.x) != 0:
        return
    var acc = Float32(0)
    if mode == 0:
        # fma chain, operands from memory (row stride d), no flush
        for i in range(n):
            acc = xmad(ld(v, i), ld(x, i * d), acc)
    elif mode == 1:
        # the flushed chain step, plain loop (fmad)
        for i in range(n):
            acc = fmad(ld(v, i), ld(x, i * d), acc)
    elif mode == 2:
        acc = chain_fmad(v, 0, 1, x, 0, d, n)
    elif mode == 3:
        acc = chain_fmad_scaled(v, x, 0, 1, d, n)
    elif mode == 4:
        acc = fold_fa(v, 0, 1, n)
    elif mode == 5:
        # register-only fma chain (no loads): the arithmetic latency floor
        var a = ld(v, 0)
        var b = ld(v, 1)
        for i in range(n):
            acc = xmad(a, b, acc)
    elif mode == 6:
        # register-only flushed chain
        var a = ld(v, 0)
        var b = ld(v, 1)
        for i in range(n):
            acc = fz(xmad(a, b, acc))
    st(res, 0, acc)


def main() raises:
    var ctx = DeviceContext()
    comptime n = 1 << 20
    comptime d = 11
    var dx = ctx.enqueue_create_buffer[DType.float32](n * d)
    var dv = ctx.enqueue_create_buffer[DType.float32](n)
    var do = ctx.enqueue_create_buffer[DType.float32](1)
    dx.enqueue_fill(Float32(0.5))
    dv.enqueue_fill(Float32(0.25))
    ctx.synchronize()
    var names = ["fma+loads", "fmad+loads", "chain_fmad", "chain_fmad_scaled", "fold_fa", "fma regs", "fz(fma) regs"]
    for mode in range(7):
        for rep in range(2):
            var t0 = perf_counter_ns()
            ctx.enqueue_function[k_plain](
                dx.unsafe_ptr(), dv.unsafe_ptr(), do.unsafe_ptr(), n, d, mode, grid_dim=1, block_dim=32,
            )
            ctx.synchronize()
            var t1 = perf_counter_ns()
            if rep == 1:
                print("CHAIN", names[mode], Float64(t1 - t0) / Float64(n), "ns/step")
    _ = dx^
    _ = dv^
    _ = do^
