# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Standalone legacy-byte qualification and synchronized layout experiment.

Main lane only; this file has not been built or run by its author.
  tools/with_build_lock.sh tools/with_identical_mode.sh pixi run mojo run \
      -I . bench/gemv_serial_layout_main.mojo

Small gates always run. MOJOLEARN_GEMV_LAYOUT_LARGE=1 adds the timed case
(default M=K=2048; override MOJOLEARN_GEMV_LAYOUT_M/K). SAMPLES defaults
to 9 and refuses fewer than 7. Timings start only after ALL bit gates.
Each sample rotates arm order. Full includes transpose + product;
prepared assumes an unchanged transposed matrix. Allocation, initialization,
and validation are excluded, and host clock includes dispatch + synchronize.
Output CELL lines carry every baseline uint32 cell for cross-device comparison.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.gemm import PINNED_GEMM_TPB, pinned_gemv_n_kernel
from core.gemv_serial_layout_candidate import (
    serial_layout_gemv_into,
    serial_layout_gemv_prepared,
)


def _env_int(name: String, fallback: Int) raises -> Int:
    var value = String(getenv(name))
    if value == "":
        return fallback
    return Int(atol(value))


def _upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var dev = ctx.enqueue_create_buffer[DType.float32](len(values))
    var host = ctx.enqueue_create_host_buffer[DType.float32](len(values))
    ctx.synchronize()
    for i in range(len(values)):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _read(ctx: DeviceContext, mut dev: DeviceBuffer[DType.float32]) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](len(dev))
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=dev)
    ctx.synchronize()
    var result = List[Float32]()
    for i in range(len(dev)):
        result.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return result^


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    if len(a) != len(b):
        raise Error(label + " size mismatch")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(label + " bit mismatch at " + String(i))


def _launch(
    ctx: DeviceContext, mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32], mut y: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32], m: Int, k: Int, arm: Int,
) raises:
    if arm == 0:
        ctx.enqueue_function[pinned_gemv_n_kernel](
            z.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(), Int32(m), Int32(k),
            grid_dim=((m + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
    elif arm == 1:
        serial_layout_gemv_into(ctx, z, x, y, xt, m, k)
    else:
        serial_layout_gemv_prepared(ctx, z, xt, y, m, k)


def _case(m: Int, k: Int, profile: Int, timing: Bool, samples: Int) raises:
    var hx = List[Float32]()
    var hy = List[Float32]()
    for p in range(k):
        var value = Float32((p * 13 + 7) % 29 - 14) / Float32(16)
        if profile == 1:
            value = Float32(1)
        if profile == 2:
            value = Float32(0.5)
        hy.append(value)
    for i in range(m):
        for p in range(k):
            var value = Float32((i * 17 + p * 11 + 3) % 61 - 30) / Float32(32)
            if profile == 1:
                # Large cancellation separated by a small addend.
                if p % 4 == 0:
                    value = Float32(16777216)
                elif p % 4 == 1:
                    value = Float32(1)
                elif p % 4 == 2:
                    value = Float32(-16777216)
                else:
                    value = Float32(-1)
            if profile == 2:
                # Raw bits preserve signed zero, subnormal inputs and normal
                # values whose multiplication by 0.5 underflows.
                var bits = UInt32(0)
                if (p + i) % 5 == 1:
                    bits = UInt32(2147483648)
                elif (p + i) % 5 == 2:
                    bits = UInt32(1)
                elif (p + i) % 5 == 3:
                    bits = UInt32(8388608)
                elif (p + i) % 5 == 4:
                    bits = UInt32(2155872256)
                value = bitcast[DType.float32](bits)
            hx.append(value)
    with DeviceContext() as ctx:
        var x = _upload(ctx, hx)
        var y = _upload(ctx, hy)
        var xt = ctx.enqueue_create_buffer[DType.float32](m * k)
        var z = ctx.enqueue_create_buffer[DType.float32](m)
        var expected = List[Float32]()
        # Full arm prepares XT before prepared is reached. Every arm is
        # poisoned and repeated; no correctness claim rests on timing output.
        for repeat in range(2):
            for arm in range(3):
                z.enqueue_fill(Float32(-987654))
                ctx.synchronize()
                _launch(ctx, z, x, y, xt, m, k, arm)
                ctx.synchronize()
                var got = _read(ctx, z)
                for i in range(m):
                    if got[i] == Float32(-987654):
                        raise Error("poison survived at " + String(i))
                if repeat == 0 and arm == 0:
                    expected = got.copy()
                else:
                    _same(expected, got, "output")
        _same(hx, _read(ctx, x), "X mutated")
        _same(hy, _read(ctx, y), "Y mutated")
        var ht = _read(ctx, xt)
        for i in range(m):
            for p in range(k):
                if bitcast[DType.uint32](hx[i * k + p]) != bitcast[DType.uint32](ht[p * m + i]):
                    raise Error("transpose changed operand bits")
        for i in range(m):
            print("CELL", m, k, profile, i, bitcast[DType.uint32](expected[i]))
        print("BITGATE PASS", m, k, profile)
        if timing:
            print("TIMING shape", m, k, "scratch_bytes", 4 * m * k,
                  "allocation_excluded host_dispatch_and_sync_included")
            for warm in range(2):
                for arm in range(3):
                    _launch(ctx, z, x, y, xt, m, k, arm)
                    ctx.synchronize()
            for sample in range(samples):
                for position in range(3):
                    var arm = (sample + position) % 3
                    ctx.synchronize()
                    var start = perf_counter_ns()
                    _launch(ctx, z, x, y, xt, m, k, arm)
                    ctx.synchronize()
                    var ms = Float64(perf_counter_ns() - start) / Float64(1000000)
                    var label = String("legacy")
                    if arm == 1:
                        label = "full_transpose_plus_product"
                    elif arm == 2:
                        label = "prepared_product_only"
                    print("SAMPLE", sample, label, ms)
            _same(expected, _read(ctx, z), "post-timing output")
        _ = x^
        _ = y^
        _ = xt^
        _ = z^


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("GEMV layout gate requires IDENTICAL mode")
    var samples = _env_int("MOJOLEARN_GEMV_LAYOUT_SAMPLES", 9)
    if samples < 7:
        raise Error("at least seven interleaved samples required")
    for profile in range(3):
        _case(1, 1, profile, False, samples)
        _case(31, 33, profile, False, samples)
        _case(63, 127, profile, False, samples)
        _case(64, 128, profile, False, samples)
        _case(65, 129, profile, False, samples)
        _case(257, 513, profile, False, samples)
    if String(getenv("MOJOLEARN_GEMV_LAYOUT_LARGE")) == "1":
        var m = _env_int("MOJOLEARN_GEMV_LAYOUT_M", 2048)
        var k = _env_int("MOJOLEARN_GEMV_LAYOUT_K", 2048)
        if m <= 0 or k <= 0 or m > 16384 or k > 16384 or m * k > 16777216:
            raise Error("large fixture requires positive dimensions <=16384 and at most16M cells")
        _case(m, k, 0, True, samples)
    print("GEMV SERIAL LAYOUT QUALIFICATION PASS")
