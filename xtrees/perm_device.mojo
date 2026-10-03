# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Permutation SHAP's synthetic dataset on the device (lane/gap-nv-classical2).

`PermutationExplainer.shap_values` built every coalition mask as a Python
list (npermutations x (2d + 1) masks of d ints per explained row), flattened
them into an Array and expanded the masks serially on the host
(`xtrees/shap.mojo::mask_expand`, 97M words per row at istella's d = 220).
Here one device thread writes each word of the same synthetic matrix from
the permutations' inverses: coalition o of permutation p turns feature f on
when its position pos = inv[p, f] satisfies pos < o (the forward walk,
o <= d) or pos >= o - d (the backward walk, o > d) -- exactly the masks the
Python loop appended. The word is x[f] or bg[r, f]: a copy, no arithmetic,
so every bit equals `mask_expand`'s."""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx

comptime PERM_TPB = 256


def perm_synth_kernel(
    res: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    bg: MutPointer[Float32, MutAnyOrigin],
    inv: MutPointer[Int32, MutAnyOrigin],
    nb_in: Int32,
    d_in: Int32,
    total_in: Int64,
):
    var nb = Int(nb_in)
    var d = Int(d_in)
    var span = 2 * d + 1
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while e < Int(total_in):
        var f = e % d
        var q = e // d
        var r = q % nb
        var s = q // nb
        var p = s // span
        var o = s - p * span
        var pos = Int(inv[p * d + f])
        var on = pos < o if o <= d else pos >= o - d
        res[e] = x[f] if on else bg[r * d + f]
        e += stride


def perm_synthetic(
    x: MutPointer[Float32, MutUntrackedOrigin],
    bg: MutPointer[Float32, MutUntrackedOrigin],
    inv: MutPointer[Int32, MutUntrackedOrigin],
    res: MutPointer[Float32, MutUntrackedOrigin],
    nb: Int,
    d: Int,
    n_perm: Int,
) raises:
    """res[((p * (2d + 1) + o) * nb + r) * d + f] for every permutation p,
    coalition o, background row r and feature f."""
    var m = n_perm * (2 * d + 1)
    var total = m * nb * d
    if total <= 0:
        return
    var ctx = process_ctx["MojoXTreesPermContext"]()
    var dx = ctx.enqueue_create_buffer[DType.float32](d)
    var dbg = ctx.enqueue_create_buffer[DType.float32](nb * d)
    var dinv = ctx.enqueue_create_buffer[DType.int32](n_perm * d)
    var dres = ctx.enqueue_create_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dbg, src_ptr=bg)
    ctx.enqueue_copy(dst_buf=dinv, src_ptr=inv)
    var blocks = min((total + PERM_TPB - 1) // PERM_TPB, 65535 * 16)
    ctx.enqueue_function[perm_synth_kernel](
        dres.unsafe_ptr(), dx.unsafe_ptr(), dbg.unsafe_ptr(), dinv.unsafe_ptr(),
        Int32(nb), Int32(d), Int64(total),
        grid_dim=(blocks, 1, 1), block_dim=(PERM_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=res, src_buf=dres)
    ctx.synchronize()
    _ = dx^
    _ = dbg^
    _ = dinv^
    _ = dres^
