# SPDX-License-Identifier: Apache-2.0
"""Element bodies of the classic attention layers (lane nn-attention, 2026-11-23), read by both executors
through `sequence/dispatch.mojo::apply` (HostExec: a host loop over the cells; DeviceExec: one GPU thread per
cell), so the host and the device column run the same function on the same words.

  op                        cells          body
  OP_NN_ATTN_SOFTMAX        B H T rows     scale, the causal and key-padding masks from the indices and the
                                           row's length, then the row softmax (core/row_cells.mojo
                                           softmax_row_from_max_f32, the tape softmax's body), in place
  OP_NN_ATTN_SOFTMAX_VJP    B H T rows     the softmax VJP (autograd/ops.mojo op_ag_softmax_vjp, the tape's
                                           body) times the scale; a masked word has y = 0, so its gradient is 0
  OP_NN_PE_ADD              B T d words    x + PE[t, j], PE from the rotary table's expressions (angle =
                                           t * 10000^(-2 (j // 2) / d), sin for even j, cos for odd j)

No mask tensor and no positional table exist: the masks are index comparisons inside the softmax row and the
encoding is computed where it is added, so neither costs a launch, a buffer or an upload per step.
"""
from std.sys.compile import is_defined
from sequence.ops import Args, FP
from checks.numerics import ftz, identical_mul, identical_div, identical_sin, identical_cos, portable_powf
from autograd.ops import op_ag_softmax_vjp
from core.row_cells import softmax_row_from_max_f32

comptime OP_NN_FIRST = 6700
comptime OP_NN_ATTN_SOFTMAX = 6700
comptime OP_NN_ATTN_SOFTMAX_VJP = 6701
comptime OP_NN_PE_ADD = 6702
comptime OP_NN_LAST = 6709

#: the softmax flags (a.i3): multihead_attention's own (autograd/nn_attention.mojo NN_MHA_*)
comptime NN_OP_CAUSAL = 2
comptime NN_OP_KEY_PADDING = 4
#: the score of a masked position (autograd/model_blocks.mojo ATTENTION_MASKED's value): exp(MASKED - max)
#: is +0.0 for every finite row max
comptime NN_OP_MASKED = Float32(-1.0e30)
comptime NN_OP_PE_BASE = Float32(10000.0)

# Fault injection (test builds only, autograd/nn_attention.mojo documents both arms)
comptime NN_SABOTAGE_PE = is_defined["MOJOLEARN_NN_ATTENTION_SABOTAGE_PE_SWAP"]()


@always_inline
def nn_attn_softmax_row(r: Int, a: Args):
    """p0 scores (rows, S), p1 probabilities, p2 the B row lengths (flags & 4); i0 S, i1 T, i2 heads,
    i3 flags; f0 the scale. Row r is (b, h, i) = query i of head h of batch row b."""
    var S = a.i0
    var T = a.i1
    var i = r % T
    var limit = S
    if (a.i3 & NN_OP_KEY_PADDING) != 0:
        limit = Int(a.p2.unsafe_load(r // (a.i2 * T)))
    var causal = (a.i3 & NN_OP_CAUSAL) != 0
    var base = r * S
    var m = NN_OP_MASKED
    for j in range(S):
        var v = NN_OP_MASKED
        if j < limit and not (causal and j > i):
            v = ftz(identical_mul(ftz(a.p0.unsafe_load(base + j)), a.f0))
        a.p1.unsafe_store(base + j, v)
        m = max(m, v)
    softmax_row_from_max_f32(a.p1, a.p1, base, S, m)


@always_inline
def nn_attn_softmax_vjp_row(r: Int, a: Args):
    """p0 y, p1 upstream, p2 the scores' gradient; i0 0 (softmax), i1 S; f0 the scale."""
    op_ag_softmax_vjp(r, a)
    var base = r * a.i1
    for j in range(a.i1):
        a.p2.unsafe_store(base + j, ftz(identical_mul(ftz(a.p2.unsafe_load(base + j)), a.f0)))


@always_inline
def nn_pe_value(t: Int, j: Int, d: Int) -> Float32:
    """PE[t, j]: graph_rotary's frequency of pair j // 2 (head_dim d, theta 10000) and the rope table's
    angle, cosine and sine (autograd/transformer_primitives.mojo, transformer_primitives_ops _rope_table)."""
    var exponent = ftz(identical_div(Float32(2 * (j // 2)), Float32(d)))
    var frequency = ftz(identical_div(Float32(1), portable_powf(NN_OP_PE_BASE, exponent)))
    var angle = ftz(identical_mul(Float32(t), frequency))
    comptime if NN_SABOTAGE_PE:
        return ftz(identical_cos(angle)) if j % 2 == 0 else ftz(identical_sin(angle))
    return ftz(identical_sin(angle)) if j % 2 == 0 else ftz(identical_cos(angle))


@always_inline
def apply_nn_attention[OP: Int](t: Int, a: Args):
    comptime if OP == OP_NN_ATTN_SOFTMAX:
        nn_attn_softmax_row(t, a)
    elif OP == OP_NN_ATTN_SOFTMAX_VJP:
        nn_attn_softmax_vjp_row(t, a)
    elif OP == OP_NN_PE_ADD:
        # p0 x (B, T, d), p1 y; i0 d, i1 T
        var d = a.i0
        var j = t % d
        var pos = (t // d) % a.i1
        a.p1.unsafe_store(t, ftz(ftz(a.p0.unsafe_load(t)) + nn_pe_value(pos, j, d)))
