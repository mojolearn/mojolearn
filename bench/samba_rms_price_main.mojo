# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What DEVIATION 2649's row geometry costs the samba RMSNorm path.

`docs/lanes/BRIEF_step_glue_2026-09-11.md` section 11, WHAT IS NOT CLAIMED:

    The rows arms move the launch geometry of `llama_rms_norm` and
    `bwd_rms_norm` for EVERY caller on an NVIDIA build, including
    `training/samba_ops.mojo`, which the leg never timed. Bits are safe
    there by section 4.1 ... but the PRICE on the samba path at 16 threads
    per block is unmeasured.

This main is that measurement and nothing else. The flip was decided on the
byte LM step; samba calls the same two launchers through
`samba_rms_norm_forward_host` and `samba_rms_norm_backward_host`, and those
calls were never timed under either geometry.

THE GEOMETRY. The shipped launch is `LLAMA_TPB` / `BWD_TPB` = 128 threads per
block, one token row per thread, so `m` rows are `ceil(m / 128)` blocks. The
winner `optskip_noshadow_rows16` launches the same kernel at 16 threads per
block, so the same rows are `ceil(m / 16)` blocks: eight times as many blocks,
each an eighth the size. Nothing else about the kernel changes, and no bit
moves (section 4.1: the row fold never leaves the thread).

TWO STAGES PER SHAPE, and the pair is the point:

    kernel   `llama_rms_norm` and `bwd_rms_norm` on device buffers that are
             already resident. THIS IS THE GEOMETRY PRICE, undiluted.
    host     `samba_rms_norm_forward_host` / `..._backward_host` end to end,
             which is what a samba caller actually pays: upload, the same
             kernels, download, and the `_refuse_nonfinite` host scan over
             every input float. The scan is O(m * dm) on the CPU and is
             identical under both arms, so this stage says whether the
             geometry is even visible from where the caller stands.

ONE BINARY, TWO ARMS. Build once with `-D MOJOLEARN_STEP_GLUE_TRIAL=1` and
run twice under `MOJOLEARN_STEP_GLUE_ARM=shipped` and
`MOJOLEARN_STEP_GLUE_ARM=optskip_noshadow_rows16`. That is the step glue
leg's own reference pattern: the reference and the arm are the SAME binary,
so a compiler difference cannot be read as a geometry difference. The driver
alternates whole processes (`[[mojolearn-box-drifts]]`), which is why this
main prints one line per repeat rather than a summary.

THE SHAPES. `tools/samba_train_run.py` defaults are batch 8, seq 64,
d_model 64, so the documented samba shape is m = 512, dm = 64 -- four blocks
at 128 threads, which does not fill one H100 SM per block on a 132-SM board.
Two larger shapes follow it so the reading is a curve and not one point.

Every line is `PRICE <arm> <rows> <m>x<dm> <stage> <ms>`; `ARM` names the arm
this process resolved, which is the reach line (a shipped build resolves its
column default and never reads the environment at all).
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from checks.numerics import numeric_mode_name
from core.step_glue import (
    step_glue_arm_from_env,
    step_glue_arm_name,
    step_glue_rows_of,
)
from training.samba_ops import (
    samba_rms_norm_backward_host,
    samba_rms_norm_forward_host,
)
from transformer.checks.transformer_backward import bwd_rms_norm
from transformer.impl.llama.modeling_llama import llama_rms_norm


comptime KERNEL_REPEATS = 7
"""Repeats of the resident-buffer stage. Cheap, so take enough for a median."""

comptime HOST_REPEATS = 3
"""Repeats of the end-to-end stage. Each one re-scans every input float on the
CPU, so this stage is slow by construction and three is the budget."""

comptime EPS = Float32(1.0e-5)


def _u01(row: Int, k: Int, salt: Int) -> Float32:
    """A fixture value with an exact binary representation, so the numbers
    below are the same on every box and no backend can round the input
    differently. The same mixer `bench/identity_price_main.mojo` uses."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(Int((z >> 40) & UInt64(0xFFFF))) / Float32(65536.0)


def _report(arm: String, rows: Int, m: Int, dm: Int, stage: String, ms: Float64):
    print(
        "PRICE",
        arm,
        rows,
        String(m) + "x" + String(dm),
        stage,
        ms,
    )


def _kernel_stage(
    ctx: DeviceContext, arm: String, rows: Int, m: Int, dm: Int
) raises:
    """The two launchers on resident buffers. No upload, no download, no host
    scan: the difference between the arms here is the launch geometry and the
    launch geometry alone."""
    var cells = m * dm

    var hx = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](dm)
    var hdy = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var hones = ctx.enqueue_create_host_buffer[DType.float32](m)
    ctx.synchronize()
    for i in range(m):
        hones.unsafe_ptr().unsafe_store(i, Float32(1.0))
        for f in range(dm):
            hx.unsafe_ptr().unsafe_store(i * dm + f, _u01(i, f, 11))
            hdy.unsafe_ptr().unsafe_store(i * dm + f, _u01(i, f, 13))
    for f in range(dm):
        hw.unsafe_ptr().unsafe_store(f, Float32(1.0) + _u01(0, f, 17))

    var x = ctx.enqueue_create_buffer[DType.float32](cells)
    var w = ctx.enqueue_create_buffer[DType.float32](dm)
    var dy = ctx.enqueue_create_buffer[DType.float32](cells)
    var ones = ctx.enqueue_create_buffer[DType.float32](m)
    var y = ctx.enqueue_create_buffer[DType.float32](cells)
    var sumsq = ctx.enqueue_create_buffer[DType.float32](m)
    var dot_out = ctx.enqueue_create_buffer[DType.float32](m)
    var dx = ctx.enqueue_create_buffer[DType.float32](cells)
    var dw = ctx.enqueue_create_buffer[DType.float32](dm)
    var dh = ctx.enqueue_create_buffer[DType.float32](cells)
    var dprod = ctx.enqueue_create_buffer[DType.float32](cells)
    var rstd = ctx.enqueue_create_buffer[DType.float32](m)
    var dvcoef = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hdy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ones, src_ptr=hones.unsafe_ptr())
    ctx.synchronize()

    # One untimed pass first: the first launch of a kernel pays for its own
    # module load, and that cost belongs to neither arm.
    llama_rms_norm(ctx, sumsq, y, x, w, m, dm, EPS)
    ctx.synchronize()

    for _r in range(KERNEL_REPEATS):
        var t0 = perf_counter_ns()
        llama_rms_norm(ctx, sumsq, y, x, w, m, dm, EPS)
        ctx.synchronize()
        _report(
            arm, rows, m, dm, String("kernel.forward"),
            Float64(perf_counter_ns() - t0) / 1.0e6,
        )

    bwd_rms_norm(
        ctx, dot_out, dx, dw, dh, dprod, rstd, dvcoef, ones, dy, x, w, sumsq,
        m, dm, EPS,
    )
    ctx.synchronize()

    for _r in range(KERNEL_REPEATS):
        var t1 = perf_counter_ns()
        bwd_rms_norm(
            ctx, dot_out, dx, dw, dh, dprod, rstd, dvcoef, ones, dy, x, w,
            sumsq, m, dm, EPS,
        )
        ctx.synchronize()
        _report(
            arm, rows, m, dm, String("kernel.backward"),
            Float64(perf_counter_ns() - t1) / 1.0e6,
        )

    _ = hx^
    _ = hw^
    _ = hdy^
    _ = hones^
    _ = x^
    _ = w^
    _ = dy^
    _ = ones^
    _ = y^
    _ = sumsq^
    _ = dot_out^
    _ = dx^
    _ = dw^
    _ = dh^
    _ = dprod^
    _ = rstd^
    _ = dvcoef^


def _host_stage(
    ctx: DeviceContext, arm: String, rows: Int, m: Int, dm: Int
) raises:
    """The samba entry points themselves, called exactly as
    `bindings/_mojolearn_training.mojo` calls them. A host buffer's
    `unsafe_ptr()` is already the `MutPointer[Float32, MutUntrackedOrigin]`
    those signatures take."""
    var cells = m * dm

    var hx = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](dm)
    var hdy = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var hdx = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var hdw = ctx.enqueue_create_host_buffer[DType.float32](dm)
    ctx.synchronize()
    for i in range(m):
        for f in range(dm):
            hx.unsafe_ptr().unsafe_store(i * dm + f, _u01(i, f, 11))
            hdy.unsafe_ptr().unsafe_store(i * dm + f, _u01(i, f, 13))
    for f in range(dm):
        hw.unsafe_ptr().unsafe_store(f, Float32(1.0) + _u01(0, f, 17))

    for _r in range(HOST_REPEATS):
        var t0 = perf_counter_ns()
        _ = samba_rms_norm_forward_host(
            ctx, hy.unsafe_ptr(), hx.unsafe_ptr(), hw.unsafe_ptr(), m, dm, EPS
        )
        _report(
            arm, rows, m, dm, String("host.forward"),
            Float64(perf_counter_ns() - t0) / 1.0e6,
        )

    for _r in range(HOST_REPEATS):
        var t1 = perf_counter_ns()
        _ = samba_rms_norm_backward_host(
            ctx, hdx.unsafe_ptr(), hdw.unsafe_ptr(), hdy.unsafe_ptr(),
            hx.unsafe_ptr(), hw.unsafe_ptr(), m, dm, EPS,
        )
        _report(
            arm, rows, m, dm, String("host.backward"),
            Float64(perf_counter_ns() - t1) / 1.0e6,
        )

    _ = hx^
    _ = hw^
    _ = hdy^
    _ = hy^
    _ = hdx^
    _ = hdw^


def _shape(ctx: DeviceContext, arm: String, rows: Int, m: Int, dm: Int) raises:
    _kernel_stage(ctx, arm, rows, m, dm)
    _host_stage(ctx, arm, rows, m, dm)


def main() raises:
    var word = step_glue_arm_from_env()
    var arm = step_glue_arm_name(word)
    var rows = step_glue_rows_of(word)
    # The reach line. `rows` 0 is the shipped 128-thread geometry; 16 is
    # DEVIATION 2649's winner. A shipped build prints its column default here
    # without ever reading the environment.
    print("ARM", arm, "rows", rows, "mode", numeric_mode_name())

    var ctx = DeviceContext()
    # The documented samba shape first (batch 8 x seq 64, d_model 64), then
    # two larger ones so the reading is a curve.
    _shape(ctx, arm, rows, 512, 64)
    _shape(ctx, arm, rows, 2048, 512)
    _shape(ctx, arm, rows, 8192, 768)
