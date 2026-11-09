# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The dictionary update's unused-atom resample (sklearn `_update_dict` with
`_resample_atom`) as ONE kernel body on the device (lane
n10-dictl-arima-loops, 2026-11-09): atom j's row is replaced by a noisy copy
of a random row of Y when A[j, j] <= 1e-6, and code column j is zeroed. The
decision, the draw and the counter are read and written on the device, so
the update loop (`x_decomp/dictl_dev.mojo`) reads nothing home per atom
(it read A[j, j] and two draws per atom before).

The words are the host column's (`x_decomp/dictl.mojo::resample_atom`):

* the test: `A[j, j] > 1e-6` in Float64 of the float32 word equals
  `A[j, j] > Float32(1e-6)` (Float32(1e-6) is the float32 just below 1e-6,
  so no float32 lies between them); NaN resamples in both;
* the draw: u = `rand_cell(0, seed, 40 + 2 c, 0)` (u = k / 2^24 exactly)
  and row index min(floor(u * rows), rows - 1); `floor(k * rows / 2^24)` in
  integers is the host's Float64 product floored for every rows < 2^29;
* the row's std: `ystd[idx]`, the caller's per-row `rowsum` / `OP_SQDIFF` /
  `OP_SQRT` cells over Y, the same per-row folds as the host's
  `total(row)` chain;
* the noise: `rand_cell(f, seed, 41 + 2 c, 1)` scaled by
  `mul(0.01f, std or 1)` (one float32 product: the host column takes the
  same float32 scale), added with `ew_cell(OP_ADD)`;
* c, the number of resamples so far, is a float word on the device (exact
  below 2^24).

`part`/`nblk`: when nblk > 0 (the FAST fused update, x_decomp/dict_fast.mojo)
a resampled row also writes its sum of squares into part[0] and zeros
part[1:nblk], so that update's norm kernel folds the resampled row.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from x_decomp.cells import F32Ptr, OP_ADD, OP_SCALE, add, ew_cell, mul, rand_cell

#: threads of the one block
comptime RS_TPB = 256
#: `A[j, j] > 1e-6` on float32 words (module docstring)
comptime RS_USED = Float32(1e-6)
#: 2^24: a kind-0 draw times this is its integer k exactly
comptime RS_GRID = Float32(16777216.0)


def dict_zero_code_kernel(code: F32Ptr, ajj: F32Ptr, code_rows: Int32, nc: Int32, j: Int32):
    """Code column j zeroed, a thread per row, when atom j is resampled (the
    same test as `dict_resample_kernel`)."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r < Int(code_rows) and not (ajj[0] > RS_USED):
        code[r * Int(nc) + Int(j)] = Float32(0)


def dict_resample_kernel(
    row: F32Ptr, ajj: F32Ptr, y: F32Ptr, ystd: F32Ptr, counter: F32Ptr,
    ny: Int32, m: Int32, seed: UInt32, part: F32Ptr, nblk: Int32,
):
    """One block of RS_TPB threads (module docstring). Every thread reads the
    test and the counter before the barrier; thread 0 alone writes the
    counter after it."""
    var tid = Int(thread_idx.x)
    var red = stack_allocation[RS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var hit = not (ajj[0] > RS_USED)
    var c = Int(counter[0])
    barrier()
    var M = Int(m)
    var sq = Float32(0)
    if hit:
        var u = rand_cell(0, seed, UInt32((40 + 2 * c) & 0xFFFFFFFF), 0)
        var rows = Int(ny)
        var idx = (Int(mul(u, RS_GRID)) * rows) >> 24
        if idx > rows - 1:
            idx = rows - 1
        var sd = ystd[idx]
        var sel = sd if sd != Float32(0) else Float32(1)
        var s = mul(Float32(0.01), sel)
        var sn = UInt32((41 + 2 * c) & 0xFFFFFFFF)
        var f = tid
        while f < M:
            var noise = ew_cell(OP_SCALE, rand_cell(f, seed, sn, 1), Float32(0), Float32(0), s)
            var v = ew_cell(OP_ADD, y[idx * M + f], noise, Float32(0), Float32(0))
            row[f] = v
            sq = add(sq, mul(v, v))
            f += RS_TPB
    red[tid] = sq
    barrier()
    var off = RS_TPB // 2
    while off > 0:
        if tid < off:
            red[tid] = add(red[tid], red[tid + off])
        barrier()
        off //= 2
    if hit and tid == 0:
        counter[0] = Float32(c + 1)
        if Int(nblk) > 0:
            part[0] = red[0]
            for q in range(1, Int(nblk)):
                part[q] = Float32(0)


def enqueue_dict_resample(
    ctx: DeviceContext, row: F32Ptr, ajj: F32Ptr, y: F32Ptr, ystd: F32Ptr, code: F32Ptr, counter: F32Ptr,
    ny: Int, m: Int, code_rows: Int, nc: Int, j: Int, seed: Int, part: F32Ptr, nblk: Int,
) raises:
    """Enqueue `dict_resample_kernel` for atom j (no read, no wait). `row`
    (1 x m) is the atom's updated row, overwritten when A[j, j] (`ajj`) marks
    the atom unused; `y` (ny x m) and `ystd` (ny) the rows it draws from;
    `code` (code_rows x nc) gets column j zeroed then; `counter` is the
    resample count word."""
    if m <= 0 or ny <= 0:
        return
    ctx.enqueue_function[dict_resample_kernel](  # small-launch(m: features of one dictionary atom): one atom row per launch, RS_TPB threads stride its m features
        row, ajj, y, ystd, counter, Int32(ny), Int32(m), UInt32(seed & 0xFFFFFFFF), part, Int32(nblk),
        grid_dim=1, block_dim=RS_TPB,
    )
    if code_rows > 0:
        ctx.enqueue_function[dict_zero_code_kernel](
            code, ajj, Int32(code_rows), Int32(nc), Int32(j),
            grid_dim=(code_rows + RS_TPB - 1) // RS_TPB, block_dim=RS_TPB,
        )
