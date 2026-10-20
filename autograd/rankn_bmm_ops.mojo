# SPDX-License-Identifier: Apache-2.0
# THE BATCHED MATRIX PRODUCT (lane ag-rankn-bmm, docs/plans/AUTOGRAD_RANKN.md
# section 2.3). One body, OP_RN_BMM, serves the forward and both VJPs of
# autograd/rankn_bmm.mojo's rankn_matmul: one thread per output word, the
# operands read in place through strides (a permuted or transposed operand
# costs no copy), and the word is one chain of fused multiply-adds from
# +0.0:
#
#   C[t] = sum over (outer contraction coordinates, row-major), then
#          k ascending, of A[a_base + ...] * B[b_base + ...]
#
# folded through sequence/ops.mojo gemm_dot (k ascending, one fma a term),
# the chain op_gemm folds. With no outer contraction axis (the forward, and
# a gradient whose operand has every batch axis of the output) the word is
# therefore op_gemm's word for the same slice, hence the tiled and thin
# routes' word (sequence/gemm_tiled.mojo states that contract); the split-K
# order (sequence/gemm_split.mojo) is never taken here. The outer axes are
# the batch axes a broadcast operand's gradient sums over: the chain runs
# over (those coordinates row-major, then k) as ONE chain, gemm_dot called
# once per outer coordinate with the accumulator carried (the same sequence
# of fmas as one long loop). No fold kernel, no scratch, one declared order
# on the host and every vendor (the same body through sequence/dispatch.mojo).
#
# Descriptor (Args.p11, int32 bit patterns in executor words, read with
# rn_word, never ld):
#   [0] nf    free axes (the output's axes, row-major), 1..8
#   [1] nc    outer contraction axes, 0..8
#   [2] K     the inner contraction length (>= 1)
#   [3] sak   A's stride along the inner contraction
#   [4] sbk   B's stride along the inner contraction
#   [5 + 3 j .. 7 + 3 j]            free axis j: extent, A stride, B stride
#   [5 + 3 nf + 3 c .. 7 + 3 nf + 3 c]  outer axis c: extent, A stride, B stride
# A stride 0 (or B stride 0) on an axis means that operand does not move
# along it (a broadcast, or the other operand's matrix axis).
#
# Args: p0 A, p1 B (unbound when i10 = 1: B is A's buffer, the x @ x^T
# case; the body branches between two gemm_dot calls and never selects a
# pointer, autograd/ops.mojo _ld_right), p2 C (dense, row-major over the
# free axes), p11 the descriptor; i0 A's base offset, i1 B's base offset,
# i10 the one-buffer flag. Launch count = product of the free extents.
#
# Codes OP_RNB_FIRST..OP_RNB_LAST (4300..4309) are this family;
# sequence/dispatch.mojo routes and fences them. 4301 is reserved for the
# phase-2 tiled dispatch on NVIDIA and AMD (its own A/B, the same words).
from sequence.ops import Args, FP, st, gemm_dot
from autograd.rankn import rn_word

comptime OP_RN_BMM = 4300
# reserved: the phase-2 tiled batched dispatch (seq_gemm_route per slice)
comptime OP_RN_BMM_TILED = 4301
comptime OP_RNB_FIRST = OP_RN_BMM
comptime OP_RNB_LAST = 4309

# descriptor header words and record width
comptime RNB_HEAD = 5
comptime RNB_REC = 3


# ---------------------------------------------------------------- bodies
@always_inline
def _rnb_chain(a: Args, abase: Int, sak: Int, bbase: Int, sbk: Int, K: Int, acc: Float32) -> Float32:
    """gemm_dot over K terms, B from p0 when i10 = 1 (a branch, never a
    pointer select)."""
    if a.i10 == 1:
        return gemm_dot(a.p0, abase, sak, a.p0, bbase, sbk, K, acc)
    return gemm_dot(a.p0, abase, sak, a.p1, bbase, sbk, K, acc)


@always_inline
def op_rn_bmm(t: Int, a: Args):
    """C[t] = the declared chain above for output word t (see the header
    for the descriptor and the slots)."""
    var d = a.p11
    var nf = rn_word(d, 0)
    var nc = rn_word(d, 1)
    var K = rn_word(d, 2)
    var sak = rn_word(d, 3)
    var sbk = rn_word(d, 4)
    # decode t over the free axes, last axis fastest
    var rem = t
    var abase = a.i0
    var bbase = a.i1
    var j = nf - 1
    while j >= 0:  # small-loop(nf: tensor rank, at most 8): index decode
        var w = RNB_HEAD + RNB_REC * j
        var e = rn_word(d, w)
        var c = rem % e
        rem = rem // e
        abase += c * rn_word(d, w + 1)
        bbase += c * rn_word(d, w + 2)
        j -= 1
    # the outer contraction coordinates, row-major
    var outer = 1
    var cbase = RNB_HEAD + RNB_REC * nf
    for q in range(nc):  # small-loop(nc: tensor rank, at most 8): extent product
        outer *= rn_word(d, cbase + RNB_REC * q)
    var acc = Float32(0.0)
    for o in range(outer):  # the declared chain's outer coordinates, ascending
        var r = o
        var oa = 0
        var ob = 0
        var q = nc - 1
        while q >= 0:  # small-loop(nc: tensor rank, at most 8): index decode
            var w = cbase + RNB_REC * q
            var e = rn_word(d, w)
            var c = r % e
            r = r // e
            oa += c * rn_word(d, w + 1)
            ob += c * rn_word(d, w + 2)
            q -= 1
        acc = _rnb_chain(a, abase + oa, sak, bbase + ob, sbk, K, acc)
    st(a.p2, t, acc)


@always_inline
def apply_rn_bmm[OP: Int](t: Int, a: Args):
    comptime if OP == OP_RN_BMM:
        op_rn_bmm(t, a)
