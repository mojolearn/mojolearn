# SPDX-License-Identifier: Apache-2.0
# THE RANK-N LAYOUT KERNEL (lane ag-rankn-layout, docs/plans/AUTOGRAD_RANKN.md
# sections 2.5 and 2.6): one strided slab copy that serves concat (forward:
# each operand into its slab of the output; backward: each slab of the
# upstream gradient into its operand's gradient) and narrow / split
# (forward: the slab of the source into a dense buffer; backward: the
# upstream gradient into the slab of the operand's gradient).
#
# Order (C) of the plan: every destination word is written by exactly one
# thread and no two writes race, so there is no fold and no order to
# declare. Each word goes through ld / st (ftz), as the tape's transpose
# copy does (autograd/ops.mojo op_ag_transpose), so a copy of a word and a
# strided read of it give the same bits on the host, Apple, NVIDIA and AMD.
#
# Shared by HostExec and DeviceExec through sequence/dispatch.mojo (codes
# OP_RNL_FIRST..OP_RNL_LAST, 4400..4409; 4401 is reserved for the phase-2
# in-place slab add OP_RN_SLAB_ADD, its own A/B).
from sequence.ops import FP, Args, ld, st
from autograd.rankn import RN_MAX_RANK, rn_word

comptime OP_RN_SLAB_COPY = 4400
# 4401: OP_RN_SLAB_ADD (phase 2, not implemented; reserved)
comptime OP_RNL_FIRST = OP_RN_SLAB_COPY
comptime OP_RNL_LAST = 4409

# Custom tags of this lane (tags 160..175, AUTOGRAD_RANKN.md section 7).
comptime RN_CONCAT = 160
comptime RN_NARROW = 161
comptime RN_LAYOUT_TAG_FIRST = 160
comptime RN_LAYOUT_TAG_LAST = 175

# One slab-copy record in a descriptor buffer (int32 bit patterns, read
# with rn_word, never ld): word
#   0                 source base offset (words)
#   1                 destination base offset (words)
#   2 .. 10           RN_DESC_SHAPE of the iteration shape: [ndim, d0 .. d7]
#   11 .. 18          source strides s0 .. s7 (0 on an unused axis)
#   19 .. 26          destination strides s0 .. s7 (0 on an unused axis)
# Thread t = word t of the iteration shape, row-major: it decodes its
# coordinates (c0 .. c_{ndim-1}) and copies
#   dst[dst_base + sum c_k * dst_s_k] = src[src_base + sum c_k * src_s_k].
comptime RNL_REC_SRC_BASE = 0
comptime RNL_REC_DST_BASE = 1
comptime RNL_REC_SHAPE = 2
comptime RNL_REC_SRC_STRIDE = RNL_REC_SHAPE + 1 + RN_MAX_RANK
comptime RNL_REC_DST_STRIDE = RNL_REC_SRC_STRIDE + RN_MAX_RANK
comptime RNL_REC_WORDS = RNL_REC_DST_STRIDE + RN_MAX_RANK


# ---------------------------------------------------------------- bodies
@always_inline
def op_rn_slab_copy(t: Int, a: Args):
    """p0 source, p1 destination, p11 the descriptor; i0 the record's first
    word in the descriptor (RNL_REC_* layout above). Thread t = iteration
    word t: one ld from the source, one st to the destination."""
    var r = a.i0
    var src_at = rn_word(a.p11, r + RNL_REC_SRC_BASE)
    var dst_at = rn_word(a.p11, r + RNL_REC_DST_BASE)
    var nd = rn_word(a.p11, r + RNL_REC_SHAPE)
    var rem = t
    for k in range(nd):  # small-loop(ndim: tensor rank, at most 8): coordinate decode, no data
        var ax = nd - 1 - k
        var dk = rn_word(a.p11, r + RNL_REC_SHAPE + 1 + ax)
        var c = rem % dk
        rem = rem // dk
        src_at += c * rn_word(a.p11, r + RNL_REC_SRC_STRIDE + ax)
        dst_at += c * rn_word(a.p11, r + RNL_REC_DST_STRIDE + ax)
    st(a.p1, dst_at, ld(a.p0, src_at))


@always_inline
def apply_rankn_layout[OP: Int](t: Int, a: Args):
    comptime if OP == OP_RN_SLAB_COPY:
        op_rn_slab_copy(t, a)
