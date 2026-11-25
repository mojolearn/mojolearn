# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RMSNorm's math, written ONCE (Andrew, 2026-11-22: one body per operation; plan
docs/plans/CLEANUP_LATER_2026-11-25.md section 8, first kernel).

    y = weight * (x * rstd),  rstd = rsqrt(sum(x^2) / d + eps)

The seams (the IDENTICAL contract's S1-S3 of the Llama and Mamba blocks): the sum of squares is one fused
multiply-add per feature onto a flushed accumulator from +0.0 (the caller owns the ORDER of the terms:
ascending, a lane split, a tree); the mean is a rounded division, eps is added in its own rounding and the
reciprocal square root is the pinned one; the scale is two separately rounded products, x * rstd first.

Callers: autograd/transformer_primitives_ops.mojo `_rms` (the public rms_norm layer, host and device),
transformer/impl/llama/modeling_llama.mojo `llama_rms_norm_kernel` and `residual_rms_norm_kernel`,
mamba/impl/modeling/modeling_mamba.mojo `mamba_rms_norm_kernel`, `mamba_rms_lane_kernel` and
`mamba_rms_out_kernel`. Each keeps its own loop, fold order, launch and buffers.
"""
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add, identical_rsqrt


@always_inline
def rms_sq_step(xv: Float32, acc: Float32) -> Float32:
    """One term of the sum of squares: acc + x * x, one rounding (x already flushed)."""
    return ftz(identical_mul_add(xv, xv, acc))


@always_inline
def rms_rstd(sumsq: Float32, d: Int, eps: Float32) -> Float32:
    """rsqrt(sumsq / d + eps): the mean rounded, eps added in its own rounding."""
    return ftz(identical_rsqrt(ftz(ftz(identical_div(sumsq, Float32(d))) + eps)))


@always_inline
def rms_scale(xv: Float32, rstd: Float32, wv: Float32) -> Float32:
    """weight * (x * rstd), two separately rounded products."""
    return ftz(identical_mul(ftz(wv), ftz(identical_mul(ftz(xv), rstd))))
