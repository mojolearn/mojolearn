# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MoE expert products as grouped identical GEMMs (lane gap-gemm-layers,
2026-10-08; plan docs/plans/gaps-2026-10-08.md section 8): THE DEVICE SIDE.
The switch and the host twin are `sequence/moe_grouped_fold.mojo`.

MOJOLEARN_IDN_MOE_GROUPED_GEMM, in `DeviceExec.launch` after the device
grouping of the (token, pick) pairs by expert (MOE_DEVGROUP's `order` and
`poff`, sequence/moe_reg.mojo):

  hidden:  xg[slot, :] = x[token(order[slot]), :]          (gather, one launch)
           for each expert e with c_e pairs:
               gu_e[c_e x 2F] = xg_e[c_e x D] . W_gu[e][2F x D]^T   identical_gemm_into OP_NT
           h[slot, f] = silu(gu[slot, f]) * gu[slot, F + f]  (epilogue, one launch)
  out:     for each expert e:
               s_e[c_e x D]  = h_e[c_e x F] . W_down[e][D x F]^T     identical_gemm_into OP_NT
           y[tok, d] = sum over picks j of w_j * s[slot_of(tok, j), d]  (combine, pick order)

The expert's rows are one contiguous slab of the slot-ordered buffers, so the
GEMM entry's contiguous operands serve without a strided variant; h and s
stay in slot order (no scatter), the combine reads them through
`slot_of[pair]` (one launch) in the same pick order with the same `fma3`
chain as `moe_combine_kernel`, so the combine's words are unchanged. The
grids of the per-expert products need the pair counts: ONE download of the
E + 1 offsets (a `ctx.synchronize()` on the executor's in-order stream, after
the grouping kernels; launch sizing only, no arithmetic on the host). The
per-expert GEMMs are the plan's "one launch per expert group": E launches
(plus the dispatcher's own fold launches) instead of `moe_reg`'s one, each on
the GEMM's tuned plans (kpack on NVIDIA, the MFMA plan on AMD).

BITS: each cell is the contract's cell (leaves of `contract_leaf_size(k)`,
the balanced tree) instead of `moe_reg`'s single chain; the host twin
spells the same cell (`moe_contract_cell`). Expected (plan 8d): the MoE
forward 29-34 ms -> 8-12 ms on the 4090 class; A/B on nv and amd decides.
Scratch: xg (T k D), gu (T k 2F), slot_of (T k) and the GEMM workspace are
the executor's own buffers (they live to its sync).
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_silu
from gemm.contract import OP_NT
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats
from sequence.ops import FP, fma3, ld, mul, st
from sequence.moe_grouped_fold import MOE_GROUPED_GEMM

comptime MG_TPB = 256


def moe_gather_rows_kernel(x: FP, order: FP, xg: FP, d_model: Int32, top_k: Int32, n_cells: Int32):
    """xg[slot, d] = x[order[slot] // k, d]: a copy, one thread per word."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_cells):
        return
    var D = Int(d_model)
    var slot = i // D
    var d = i - slot * D
    var tok = Int(order.unsafe_load(slot)) // Int(top_k)
    xg.unsafe_store(i, x.unsafe_load(tok * D + d))


def moe_hidden_epilogue_kernel(gu: FP, h: FP, n_ff: Int32, n_cells: Int32):
    """h[slot, f] = mul(ftz(silu(g)), u) on the stored (flushed) GEMM words
    g = gu[slot, f], u = gu[slot, F + f]: `moe_hidden_reg_kernel`'s store."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_cells):
        return
    var F = Int(n_ff)
    var slot = i // F
    var f = i - slot * F
    var g = ld(gu, slot * 2 * F + f)
    var u = ld(gu, slot * 2 * F + F + f)
    st(h, i, mul(ftz(identical_silu(g)), u))


def moe_slot_of_kernel(order: FP, slot_of: FP, n_pairs: Int32):
    """slot_of[order[slot]] = slot (the inverse of the grouping's order)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_pairs):
        slot_of.unsafe_store(Int(order.unsafe_load(i)), Float32(i))


def moe_combine_slot_kernel(w: FP, s: FP, slot_of: FP, y: FP, d_model: Int32, top_k: Int32, n_cells: Int32):
    """`moe_combine_kernel` reading s in slot order: y[tok, d] = fma3(w_j,
    s[slot_of[tok, j], d], y) over the picks in pick order from +0.0."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(n_cells):
        return
    var D = Int(d_model)
    var k = Int(top_k)
    var tok = t // D
    var d = t - tok * D
    var yv = Float32(0.0)
    for j in range(k):
        var p = tok * k + j
        var slot = Int(slot_of.unsafe_load(p))
        yv = fma3(ld(w, p), ld(s, slot * D + d), yv)
    st(y, t, yv)


@always_inline
def _view(ctx: DeviceContext, p: FP, off: Int, n: Int) -> DeviceBuffer[DType.float32]:
    """A non-owning view of `n` floats at `p + off` (the executor's buffer)."""
    return DeviceBuffer[DType.float32](ctx, (p + off).unsafe_origin_cast[MutAnyOrigin](), n if n > 0 else 1, owning=False)


def moe_grouped_offsets(ctx: DeviceContext, poff: FP, n_experts: Int) raises -> List[Int]:
    """The E + 1 pair offsets the grouping wrote, on the host (one wait on
    the executor's stream): the per-expert GEMM sizes."""
    var n = n_experts + 1
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    var view = _view(ctx, poff, 0, n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Int]()
    for e in range(n):  # small-loop(n: the E + 1 expert pair offsets): per-expert launch sizes, never data
        out.append(Int(host.unsafe_ptr()[e]))
    _ = view^
    _ = host^
    return out^


def moe_grouped_workspace(poff: List[Int], n_experts: Int, D: Int, F: Int) -> Int:
    """The GEMM workspace for every expert's two products (the maximum)."""
    var need = 1
    for e in range(n_experts):  # small-loop(n_experts: the experts): one workspace size per expert, no data
        var c = poff[e + 1] - poff[e]
        if c > 0:
            need = max(need, identical_gemm_workspace_max_floats(c, 2 * F, D))
            need = max(need, identical_gemm_workspace_max_floats(c, D, F))
    return need


def moe_grouped_hidden(
    ctx: DeviceContext, x: FP, gu_w: FP, order: FP, xg: FP, gu: FP, h: FP, mut wsb: DeviceBuffer[DType.float32],
    poff: List[Int], D: Int, F: Int, k: Int, n_experts: Int, n_pairs: Int,
) raises:
    """The hidden product: the gather, E grouped GEMMs, the silu * up
    epilogue. `wsb` holds `moe_grouped_workspace` floats."""
    if n_pairs <= 0:
        return
    ctx.enqueue_function[moe_gather_rows_kernel](
        x, order, xg, Int32(D), Int32(k), Int32(n_pairs * D),
        grid_dim=((n_pairs * D + MG_TPB - 1) // MG_TPB, 1, 1), block_dim=(MG_TPB, 1, 1),
    )
    for e in range(n_experts):  # small-loop(n_experts: the experts): one GEMM launch per expert, no data
        var c = poff[e + 1] - poff[e]
        if c <= 0:
            continue
        var a = _view(ctx, xg, poff[e] * D, c * D)
        var b = _view(ctx, gu_w, e * 2 * F * D, 2 * F * D)
        var cbuf = _view(ctx, gu, poff[e] * 2 * F, c * 2 * F)
        identical_gemm_into[False](ctx, cbuf, a, b, wsb, c, 2 * F, D, OP_NT)
        _ = a^
        _ = b^
        _ = cbuf^
    ctx.enqueue_function[moe_hidden_epilogue_kernel](
        gu, h, Int32(F), Int32(n_pairs * F),
        grid_dim=((n_pairs * F + MG_TPB - 1) // MG_TPB, 1, 1), block_dim=(MG_TPB, 1, 1),
    )


def moe_grouped_out(
    ctx: DeviceContext, h: FP, dn_w: FP, w: FP, order: FP, slot_of: FP, s: FP, y: FP,
    mut wsb: DeviceBuffer[DType.float32], poff: List[Int], D: Int, F: Int, k: Int, n_experts: Int, n_pairs: Int,
    n_tok: Int,
) raises:
    """The out product: E grouped GEMMs into the slot-ordered s, then the
    combine through slot_of."""
    if n_pairs <= 0:
        return
    for e in range(n_experts):  # small-loop(n_experts: the experts): one GEMM launch per expert, no data
        var c = poff[e + 1] - poff[e]
        if c <= 0:
            continue
        var a = _view(ctx, h, poff[e] * F, c * F)
        var b = _view(ctx, dn_w, e * D * F, D * F)
        var cbuf = _view(ctx, s, poff[e] * D, c * D)
        identical_gemm_into[False](ctx, cbuf, a, b, wsb, c, D, F, OP_NT)
        _ = a^
        _ = b^
        _ = cbuf^
    ctx.enqueue_function[moe_slot_of_kernel](
        order, slot_of, Int32(n_pairs),
        grid_dim=((n_pairs + MG_TPB - 1) // MG_TPB, 1, 1), block_dim=(MG_TPB, 1, 1),
    )
    ctx.enqueue_function[moe_combine_slot_kernel](
        w, s, slot_of, y, Int32(D), Int32(k), Int32(n_tok * D),
        grid_dim=((n_tok * D + MG_TPB - 1) // MG_TPB, 1, 1), block_dim=(MG_TPB, 1, 1),
    )
