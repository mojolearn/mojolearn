# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The launcher of the estimated Holt-Winters fit (`initialization_method=
"estimated"`). Moved out of `hw_estimate.mojo` (lane cpu3-seq, 2026-10-04) so
that module holds only the kernels and the per-thread arithmetic they share
with the host column; the launch sequence and its arguments are unchanged."""

from experiments.classical_identical_ideas.stats_controls import C58_SERIES4
from max.gpu.host import DeviceBuffer, DeviceContext

from holtwinters.impl.internal.hw_estimate import (
    HW_EST_BLOCK,
    HW_EST_STARTS,
    holtwinters_estimate_block_kernel,
    holtwinters_estimate_finish_kernel,
    holtwinters_estimate_gpu_kernel,
    hw_est_block_scratch_len,
    hw_est_parallel,
    hw_est_scratch_len,
)
from holtwinters.impl.internal.hw_utils import HW_EST_FAST_BLOCK


def holtwinters_estimate_gpu(
    ctx: DeviceContext,
    mut ts: DeviceBuffer[DType.float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    additive: Bool,
    mut start_level: DeviceBuffer[DType.float32],
    mut start_trend: DeviceBuffer[DType.float32],
    mut start_season: DeviceBuffer[DType.float32],
    mut level: DeviceBuffer[DType.float32],
    mut trend: DeviceBuffer[DType.float32],
    mut season: DeviceBuffer[DType.float32],
    mut alpha: DeviceBuffer[DType.float32],
    mut beta: DeviceBuffer[DType.float32],
    mut gamma: DeviceBuffer[DType.float32],
    mut error: DeviceBuffer[DType.float32],
    mut criterion: DeviceBuffer[DType.int32],
    mut niter: DeviceBuffer[DType.int32],
    mut theta_out: DeviceBuffer[DType.float32],
    tpb: Int,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = Float32(0.0),
    force_serial: Bool = False,
) raises:
    """Launch the estimated fit. `ts` is time-major; `theta_out` holds
    `(frequency + 5) * batch_size`. `tpb` is the finish/serial kernels'
    block width (scheduling only). `force_serial` takes the serial arm at
    any `frequency` (the gate uses it to hold the two arms to each other)."""
    if tpb <= 0:
        raise Error("holtwinters_estimate_gpu: tpb must be positive")
    var total_blocks = (batch_size + tpb - 1) // tpb
    if hw_est_parallel(frequency) and not force_serial:
        var d = frequency + 5
        var blocks = batch_size * HW_EST_STARTS
        var scratch = ctx.enqueue_create_buffer[DType.float32](blocks * hw_est_block_scratch_len(frequency) + scratch_pad)
        var cand_theta = ctx.enqueue_create_buffer[DType.float32](blocks * d + scratch_pad)
        var cand_sse = ctx.enqueue_create_buffer[DType.float32](blocks + scratch_pad)
        var cand_ints = ctx.enqueue_create_buffer[DType.int32](2 * blocks)
        var sw_all = ctx.enqueue_create_buffer[DType.float32](batch_size * frequency + scratch_pad)
        scratch.enqueue_fill(scratch_poison)
        cand_theta.enqueue_fill(scratch_poison)
        cand_sse.enqueue_fill(scratch_poison)
        sw_all.enqueue_fill(scratch_poison)
        ctx.synchronize()
        # Tried 2026-10-08 (MOJOLEARN_C58_SHARED_PREP, run ge123e6f9): per-series scale computed once by its own kernel, not per start block;
        # NV/AMD ets synthetic 0.996/0.980; rmse same -> noise, deleted. Code recoverable at main ad7ed2370; row in docs/apple-fast/EXPERIMENTS.md.
        if d <= HW_EST_FAST_BLOCK:
            ctx.enqueue_function[holtwinters_estimate_block_kernel[HW_EST_FAST_BLOCK]](
                ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
                Int32(1 if additive else 0),
                start_level.unsafe_ptr(), start_trend.unsafe_ptr(), start_season.unsafe_ptr(),
                scratch.unsafe_ptr(), cand_theta.unsafe_ptr(), cand_sse.unsafe_ptr(), cand_ints.unsafe_ptr(),
                level.unsafe_ptr(),
                grid_dim=(blocks, 1, 1),
                block_dim=(HW_EST_FAST_BLOCK, 1, 1),
            )
        else:
            ctx.enqueue_function[holtwinters_estimate_block_kernel[HW_EST_BLOCK]](
                ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
                Int32(1 if additive else 0),
                start_level.unsafe_ptr(), start_trend.unsafe_ptr(), start_season.unsafe_ptr(),
                scratch.unsafe_ptr(), cand_theta.unsafe_ptr(), cand_sse.unsafe_ptr(), cand_ints.unsafe_ptr(),
                level.unsafe_ptr(),
                grid_dim=(blocks, 1, 1),
                block_dim=(HW_EST_BLOCK, 1, 1),
            )
        ctx.synchronize()
        ctx.enqueue_function[holtwinters_estimate_finish_kernel](
            ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
            Int32(1 if additive else 0),
            cand_theta.unsafe_ptr(), cand_sse.unsafe_ptr(), cand_ints.unsafe_ptr(), sw_all.unsafe_ptr(),
            level.unsafe_ptr(), trend.unsafe_ptr(), season.unsafe_ptr(),
            alpha.unsafe_ptr(), beta.unsafe_ptr(), gamma.unsafe_ptr(), error.unsafe_ptr(),
            criterion.unsafe_ptr(), niter.unsafe_ptr(), theta_out.unsafe_ptr(),
            grid_dim=(total_blocks, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        _ = scratch^
        _ = cand_theta^
        _ = cand_sse^
        _ = cand_ints^
        _ = sw_all^
        return
    var scratch = ctx.enqueue_create_buffer[DType.float32](
        batch_size * hw_est_scratch_len(frequency) + scratch_pad
    )
    scratch.enqueue_fill(scratch_poison)
    ctx.synchronize()
    ctx.enqueue_function[holtwinters_estimate_gpu_kernel](
        ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
        Int32(1 if additive else 0),
        start_level.unsafe_ptr(), start_trend.unsafe_ptr(), start_season.unsafe_ptr(),
        scratch.unsafe_ptr(),
        level.unsafe_ptr(), trend.unsafe_ptr(), season.unsafe_ptr(),
        alpha.unsafe_ptr(), beta.unsafe_ptr(), gamma.unsafe_ptr(), error.unsafe_ptr(),
        criterion.unsafe_ptr(), niter.unsafe_ptr(), theta_out.unsafe_ptr(),
        grid_dim=(((batch_size + (4 if C58_SERIES4 else 1)*tpb - 1) // ((4 if C58_SERIES4 else 1)*tpb)), 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    _ = scratch^
