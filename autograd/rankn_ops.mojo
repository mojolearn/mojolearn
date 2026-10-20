# SPDX-License-Identifier: Apache-2.0
# THE RANK-N VIEW COPY (lane ag-rankn-core, 2026-10-20;
# docs/plans/AUTOGRAD_RANKN.md sections 1.3 and 2.4).
#
# OP_RN_COPY_STRIDED writes a dense row-major output from a strided read of
# a source: a permute (or a narrow) materialised for a dense reader, and a
# permute's VJP (the upstream read through the inverse permutation into the
# operand's gradient). One thread per destination word; every destination
# word is written exactly once and no two threads write one word, so there
# is no order to declare (the plan's order (C)). Forward materialization
# copies the source words exactly (i2=1), matching OP_AG_TRANSPOSE's forward
# route. VJPs retain their established ld/st FTZ rule (i2=0).
# Correctness repair 2026-10-20: forward ld/st previously flushed subnormals
# while identity views did not. This intentionally changes those forward
# words within the new version; CPU/Apple/NVIDIA/AMD must qualify together.
# Arithmetic and reduction FTZ rules are unchanged.
#
# Shared by HostExec and DeviceExec through sequence/dispatch.mojo, so the
# host column and every GPU run the same body. Codes OP_RNC_FIRST..OP_RNC_LAST
# (4100..4109) are this family; sequence/dispatch.mojo routes and fences
# them. 4101..4109 are reserved for the core lane.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, ld, st
from autograd.rankn import RN_DESC_SHAPE_WORDS, rn_word

comptime OP_RN_COPY_STRIDED = 4100
comptime OP_RNC_FIRST = OP_RN_COPY_STRIDED
comptime OP_RNC_LAST = 4109


# ---------------------------------------------------------------- bodies
@always_inline
def op_rn_copy_strided(t: Int, a: Args):
    """p1[t] = p0[i0 + sum_k c_k * s_k], (c_0 .. c_{n-1}) the row-major
    coordinates of destination word t. p0 source, p1 destination, p11 the
    descriptor; i0 the source's base offset (elements), i1 the offset of a
    RN_DESC_VIEW record in p11 ([ndim, d0 .. d7] the destination shape,
    then [s0 .. s7] the source strides along those axes). Thread t =
    destination word t. i2=1 is an exact forward word copy; otherwise retain
    the established VJP normalization."""
    var at = a.i1
    var nd = rn_word(a.p11, at)
    var rem = t
    var src = a.i0
    for j in range(nd):  # small-loop(ndim: tensor rank, at most 8): the coordinate decode
        var k = nd - 1 - j
        var e = rn_word(a.p11, at + 1 + k)
        var c = rem % e
        rem = rem // e
        src += c * rn_word(a.p11, at + RN_DESC_SHAPE_WORDS + k)
    if a.i2 == 1:
        a.p1.unsafe_store(t, a.p0.unsafe_load(src))
    else:
        st(a.p1, t, ld(a.p0, src))


@always_inline
def apply_rankn_core[OP: Int](t: Int, a: Args):
    comptime if OP == OP_RN_COPY_STRIDED:
        op_rn_copy_strided(t, a)


# ---------------------------------------------------------------- drivers
def rn_copy_strided[E: Exec](
    mut ex: E, dst: FP, src: FP, base: Int, desc: FP, desc_at: Int, n: Int,
    preserve_words: Bool = False,
) raises:
    """dst (n words, dense) = src read at base + the strides of the
    RN_DESC_VIEW record at desc[desc_at]: one OP_RN_COPY_STRIDED launch."""
    var a = Args()
    a.p0 = src
    a.p1 = dst
    a.p11 = desc
    a.i0 = base
    a.i1 = desc_at
    a.i2 = Int(preserve_words)
    ex.launch[OP_RN_COPY_STRIDED](a, n)
