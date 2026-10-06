# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MaxAbsScaler.fit from the caller's own buffer (lane/apple-fast-prep3,
2026-10-02). FAST on Apple by default (-D MOJOLEARN_PREP3_MAXABS_OFF reverts): the binding
registers `x_prep_maxabs_fit_direct` under PREP3_MAXABS alone, and
`maxabs_fit_direct` refuses in every other build, so no other build compiles
a launch here.

The program route (python/mojolearn/_expansion_prep.py MaxAbsScaler.fit) puts X
through `_Prog`: at the board's 1M x 220 the direct input is a device store
slot (one host-to-device copy of 880 MB) copied device to device into a
second 880 MB arena, then `colb_part` / `maxabs_fold`, then the arena comes
back. Here X goes up ONCE into the only buffer the kernels read, the
partials live in a small device buffer, and 2 d words come back.

Words: `maxabs_part_kernel` takes each (chunk, column) max |x| over the
non-NaN flushed entries exactly as `_cs_take` does (a maximum has one answer
in any order), `maxabs_fold_kernel` the largest partial and sklearn's
`_handle_zeros_in_scale` of it (`zero_to_one`): the program route's
max_abs_ and scale_ word for word.
"""
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz
from x_prep.common import FP, is_nan
from x_prep.prims import zero_to_one
from x_prep.device import x_prep_ctx
from x_prep.prep3 import PREP3_MAXABS, PREP3_MAXABS_POOL
from core.device_pool import pool_give, pool_take
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# AFCL-P02: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified.
# Halve work per column lane to expose more independent blocks for every
# width. This doubles partial storage; the existing width policy is unchanged.
comptime AFCL_P02 = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_AFCL_P02"]()

#: threads per block: one column each, consecutive columns of one row
comptime MA_TPB = 256
#: rows per chunk (wide X) and for a narrow X (more chunks, more threads)
comptime MA_ROWS = 512 if AFCL_P02 else 1024
comptime MA_ROWS_NARROW = 64 if AFCL_P02 else 128
comptime MA_NARROW = 64


def maxabs_part_kernel(x: FP, part: FP, n: Int32, d: Int32, rows: Int32, cg: Int32):
    """block = chunk * cg + column group; thread = column g*MA_TPB + tid:
    part[chunk*d + c] = max |x| over the chunk's non-NaN rows of column c
    (0 when none). Consecutive threads read consecutive words of a row."""
    var dd = Int(d)
    var b = Int(block_idx.x)
    var cgn = Int(cg)
    var chunk = b // cgn
    var c = (b - chunk * cgn) * MA_TPB + Int(thread_idx.x)
    if c >= dd:
        return
    var lo = chunk * Int(rows)
    var hi = min(lo + Int(rows), Int(n))
    var ma = Float32(0)
    for i in range(lo, hi):
        var v = ftz(x[i * dd + c])
        if is_nan(v):
            continue
        if abs(v) > ma:
            ma = abs(v)
    part[chunk * dd + c] = ma


def maxabs_fold_kernel(part: FP, dst: FP, d: Int32, nch: Int32):
    """thread = column c: dst[c] = the largest partial (max_abs_), dst[d + c]
    = `zero_to_one` of it (scale_)."""
    var dd = Int(d)
    var c = Int(block_idx.x) * MA_TPB + Int(thread_idx.x)
    if c >= dd:
        return
    var ma = Float32(0)
    for ch in range(Int(nch)):
        var v = part[ch * dd + c]
        if v > ma:
            ma = v
    dst[c] = ma
    dst[dd + c] = zero_to_one(ma)


def maxabs_fit_direct(x_addr: Int, n: Int, d: Int, out_addr: Int) raises:
    """max_abs_ (d words) then scale_ (d words) at out_addr from the n x d
    float32 C-order X at x_addr: one upload, two launches, 2 d words back."""
    comptime if not PREP3_MAXABS:
        raise Error("x_prep: maxabs_fit_direct is compiled under FAST on Apple (PREP3_MAXABS) only")
    if n <= 0 or d <= 0 or x_addr == 0 or out_addr == 0:
        raise Error("x_prep: maxabs_fit_direct needs a nonempty X and output")
    var rows = MA_ROWS if d >= MA_NARROW else MA_ROWS_NARROW
    var nch = (n + rows - 1) // rows
    var cg = (d + MA_TPB - 1) // MA_TPB
    var ctx = x_prep_ctx()
    comptime if PREP3_MAXABS_POOL:
        # lane/apple-fast-w4-small: pooled buffers, the same launches
        var px = pool_take["MojoXPrepMaxAbsX"](ctx, n * d)
        var pp = pool_take["MojoXPrepMaxAbsPart"](ctx, nch * d)
        var po = pool_take["MojoXPrepMaxAbsOut"](ctx, 2 * d)
        ctx.enqueue_copy(dst_buf=px, src_ptr=FP(unsafe_from_address=x_addr))
        ctx.enqueue_function[maxabs_part_kernel](
            FP(unsafe_from_address=Int(px.unsafe_ptr())), FP(unsafe_from_address=Int(pp.unsafe_ptr())),
            Int32(n), Int32(d), Int32(rows), Int32(cg),
            grid_dim=nch * cg, block_dim=MA_TPB,
        )
        ctx.enqueue_function[maxabs_fold_kernel](
            FP(unsafe_from_address=Int(pp.unsafe_ptr())), FP(unsafe_from_address=Int(po.unsafe_ptr())),
            Int32(d), Int32(nch), grid_dim=cg, block_dim=MA_TPB,
        )
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=out_addr), src_buf=po)
        ctx.synchronize()
        pool_give["MojoXPrepMaxAbsX"](px^)
        pool_give["MojoXPrepMaxAbsPart"](pp^)
        pool_give["MojoXPrepMaxAbsOut"](po^)
        _ = ctx^
        return
    var dx = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=FP(unsafe_from_address=x_addr))
    var dp = ctx.enqueue_create_buffer[DType.float32](nch * d)
    var dout = ctx.enqueue_create_buffer[DType.float32](2 * d)
    ctx.enqueue_function[maxabs_part_kernel](
        dx.unsafe_ptr(), dp.unsafe_ptr(), Int32(n), Int32(d), Int32(rows), Int32(cg),
        grid_dim=nch * cg, block_dim=MA_TPB,
    )
    ctx.enqueue_function[maxabs_fold_kernel](
        dp.unsafe_ptr(), dout.unsafe_ptr(), Int32(d), Int32(nch), grid_dim=cg, block_dim=MA_TPB,
    )
    ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=out_addr), src_buf=dout)
    ctx.synchronize()
    _ = dx^
    _ = dp^
    _ = dout^
    _ = ctx^
