# SPDX-License-Identifier: Apache-2.0
# THE BATCHED MATRIX PRODUCT ON THE TAPE (lane ag-rankn-bmm,
# docs/plans/AUTOGRAD_RANKN.md section 2.3).
#
# rankn_matmul(tape, ex, x, y): x [..., M, K] @ y [..., K, N], the leading
# (batch) axes broadcast NumPy's way (equal, or 1; a missing axis is 1),
# output broadcast(batch) + [M, N]. A one-dimensional operand is its
# matrix view (one row of K for x, one row for y).
#
# Routes:
# - y of rank <= 2: Tape.matmul on x's matrix view, the existing GEMM, its
#   route, its words and its backward; the core lane's shape inheritance
#   gives the output x's leading axes + [N]. This covers [B, T, F] @ W[F, O].
# - otherwise: one OP_RN_BMM launch (autograd/rankn_bmm_ops.mojo), thread =
#   output word, operands read in place through Tape.operand_nd's strides
#   (a permute or transpose view costs no copy), recorded as CUSTOM tag
#   RN_BMM. Its VJPs are one OP_RN_BMM launch each: dX = G @ Y^T and
#   dY = X^T @ G with swapped strides; a batch axis the operand broadcasts
#   over becomes an outer contraction axis, so a shared operand's gradient
#   word is ONE chain over (its broadcast batch coordinates row-major, then
#   the contraction) from +0.0: no fold kernel, no scratch.
#
# Bits: with every batch axis of the operand present, each word is the
# gemm_dot chain op_gemm folds for the same slice, so a slice equals
# Tape.matmul on that slice wherever the 2-D product keeps its chain (outside
# gemm_split_shape). The broadcast chain is new and declared here; it is the
# same on the host and every vendor. No existing word changes: rank-2 and
# rank-N @ rank-2 callers reach Tape.matmul.
#
# Saved slots of an RN_BMM node: p0 x's buffer, p1 y's buffer (operand_nd's,
# the source of a pending view), p9 the dX descriptor, p10 the dY
# descriptor, p11 the forward descriptor (dummy where unused); i0 x's base
# offset, i1 y's base offset, i2 / i3 = 1 when x / y wants a gradient, i4 /
# i5 the x / y word counts, i6 the output word count, i10 = 1 when x and y
# read one buffer. The rule never reads `inputs` (a pending view's value
# buffer is unfilled): it reads the saved buffers through the saved strides.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_bmm_ops import OP_RN_BMM, RNB_HEAD, RNB_REC

# custom tags of this lane: 152..159 (plan section 7)
comptime RN_BMM = 152
comptime RNB_TAG_FIRST = 152
comptime RNB_TAG_LAST = 159

comptime _I32_MAX = 2147483647


@fieldwise_init
struct _Axis(ImplicitlyCopyable, Movable):
    """One axis of an OP_RN_BMM descriptor: extent, A stride, B stride."""
    var e: Int
    var sa: Int
    var sb: Int


@fieldwise_init
struct _Mat(Movable):
    """An operand's matrix part and batch axes, from Tape.operand_nd."""
    var buf: FP
    var base: Int
    var rows: Int
    var cols: Int
    var rs: Int     # row stride
    var cs: Int     # column stride
    var nb: Int     # batch axes (ndim - 2, or 0 for ndim <= 2)
    var be: InlineArray[Int, RN_MAX_RANK]   # batch extents
    var bs: InlineArray[Int, RN_MAX_RANK]   # batch strides

    def copy(self) -> Self:
        return _Mat(self.buf, self.base, self.rows, self.cols, self.rs, self.cs, self.nb, self.be.copy(), self.bs.copy())


def _mat[E: Exec](mut tape: Tape, mut ex: E, x: Tensor) raises -> _Mat:
    var o = tape.operand_nd(ex, x)
    var s = o[2].copy()
    var st = o[3].copy()
    var be = InlineArray[Int, RN_MAX_RANK](fill=1)
    var bs = InlineArray[Int, RN_MAX_RANK](fill=0)
    if s.ndim < 1:
        raise Error("matmul operands must have at least one dimension")
    if s.ndim == 1:
        # its matrix view: one row of d0
        return _Mat(o[0], o[1], 1, s.d[0], s.d[0] * st[0], st[0], 0, be^, bs^)
    var nb = s.ndim - 2
    for k in range(nb):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        be[k] = s.d[k]
        bs[k] = st[k]
    return _Mat(o[0], o[1], s.d[s.ndim - 2], s.d[s.ndim - 1], st[s.ndim - 2], st[s.ndim - 1], nb, be^, bs^)


@always_inline
def _ext(m: _Mat, j: Int, nb: Int) -> Int:
    """m's extent on output batch axis j of nb (right-aligned; 1 when absent)."""
    var k = j - (nb - m.nb)
    if k < 0:
        return 1
    return m.be[k]


@always_inline
def _stride(m: _Mat, j: Int, nb: Int) -> Int:
    """m's stride on output batch axis j of nb: 0 when absent or extent 1
    (a broadcast operand does not move along it)."""
    var k = j - (nb - m.nb)
    if k < 0 or m.be[k] == 1:
        return 0
    return m.bs[k]


def _words(free: List[_Axis], outer: List[_Axis], K: Int, sak: Int, sbk: Int) raises -> List[Int]:
    """The OP_RN_BMM descriptor (rankn_bmm_ops.mojo header)."""
    var w = List[Int]()
    w.append(len(free))
    w.append(len(outer))
    w.append(K)
    w.append(sak)
    w.append(sbk)
    for a in free:  # small-loop(len(free): tensor rank, at most 8): shape metadata, no data
        w.append(a.e)
        w.append(a.sa)
        w.append(a.sb)
    for a in outer:  # small-loop(len(outer): tensor rank, at most 8): shape metadata, no data
        w.append(a.e)
        w.append(a.sa)
        w.append(a.sb)
    for v in w:  # small-loop(len(w): at most 53 descriptor words): bounds, no data
        if v > _I32_MAX or v < -_I32_MAX:
            raise Error("matmul: a dimension or stride does not fit the launch's 32-bit words")
    return w^


def _launch[E: Exec](
    mut ex: E, dst: FP, pa: FP, abase: Int, pb: FP, bbase: Int, desc: FP, count: Int, one_buffer: Bool,
) raises:
    var a = Args()
    a.p0 = pa
    if not one_buffer:
        a.p1 = pb
    a.p2 = dst
    a.p11 = desc
    a.i0 = abase
    a.i1 = bbase
    a.i10 = 1 if one_buffer else 0
    ex.launch[OP_RN_BMM](a, count)


def rankn_matmul[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
    """The matrix product of the last two axes of x and y; leading axes are
    batch axes and broadcast."""
    var sy = tape.shape_of(y)
    if tape.shape_of(x).ndim < 1 or sy.ndim < 1:
        raise Error("matmul operands must have at least one dimension")
    if sy.ndim <= 2:
        # the existing GEMM on x's matrix view, its route and its words
        return tape.matmul(ex, x, y)
    var a = _mat(tape, ex, x)
    var b = _mat(tape, ex, y)
    var M = a.rows
    var K = a.cols
    var N = b.cols
    if b.rows != K:
        raise Error("matmul inner dimensions differ")
    var nb = max(a.nb, b.nb)
    if nb + 2 > RN_MAX_RANK:
        raise Error("matmul output has more than 8 dimensions")
    # the output's shape and row-major strides
    var d = InlineArray[Int, RN_MAX_RANK](fill=1)
    for j in range(nb):  # small-loop(nb: tensor rank, at most 8): shape metadata, no data
        var ea = _ext(a, j, nb)
        var eb = _ext(b, j, nb)
        if ea != eb and ea != 1 and eb != 1:
            raise Error("matmul batch dimensions do not broadcast")
        d[j] = max(ea, eb)
    d[nb] = M
    d[nb + 1] = N
    var shape = ShapeN(nb + 2, d)
    var gs = InlineArray[Int, RN_MAX_RANK](fill=0)
    var run = 1
    var j = nb + 1
    while j >= 0:  # small-loop(nb + 2: tensor rank, at most 8): shape metadata, no data
        gs[j] = run
        run *= d[j]
        j -= 1
    var count = run
    var one = Int(a.buf) == Int(b.buf)

    # forward: free = (batch..., m, n), inner contraction k
    var free = List[_Axis]()
    for q in range(nb):  # small-loop(nb: tensor rank, at most 8): shape metadata, no data
        free.append(_Axis(d[q], _stride(a, q, nb), _stride(b, q, nb)))
    free.append(_Axis(M, a.rs, 0))
    free.append(_Axis(N, 0, b.cs))
    var desc = tape.descriptor(ex, _words(free, List[_Axis](), K, a.cs, b.rs))
    var output = tape.reserve(ex, count)
    _launch(ex, output, a.buf, a.base, b.buf, b.base, desc, count, one)

    var saved = Args()
    saved.p0 = a.buf
    saved.p1 = b.buf
    saved.p11 = desc
    saved.i0 = a.base
    saved.i1 = b.base
    saved.i4 = 0
    saved.i5 = 0
    saved.i6 = count
    saved.i10 = 1 if one else 0
    var want_x = tape.is_recording() and tape.requires_grad(x)
    var want_y = tape.is_recording() and tape.requires_grad(y)
    if want_x:
        # dX = G @ Y^T over x's own axes: A = G (dense, strides gs), B = y.
        # An output batch axis x lacks or broadcasts is an outer axis.
        var fx = List[_Axis]()
        var ox = List[_Axis]()
        var cx = 1
        for q in range(nb):  # small-loop(nb: tensor rank, at most 8): shape metadata, no data
            var k = q - (nb - a.nb)
            if k >= 0:
                fx.append(_Axis(a.be[k], gs[q] if a.be[k] > 1 else 0, _stride(b, q, nb) if a.be[k] > 1 else 0))
                cx *= a.be[k]
            if (k < 0 or a.be[k] == 1) and d[q] > 1:
                ox.append(_Axis(d[q], gs[q], _stride(b, q, nb)))
        if tape.shape_of(x).ndim >= 2:
            fx.append(_Axis(M, gs[nb], 0))
        # Y^T(n, k) = Y(k, n): x's column k is y's row k
        fx.append(_Axis(K, 0, b.rs))
        saved.p9 = tape.descriptor(ex, _words(fx, ox, N, gs[nb + 1], b.cs))
        saved.i2 = 1
        saved.i4 = cx * M * K
    if want_y:
        # dY = X^T @ G over y's own axes: A = x, B = G (dense, strides gs)
        var fy = List[_Axis]()
        var oy = List[_Axis]()
        var cy = 1
        for q in range(nb):  # small-loop(nb: tensor rank, at most 8): shape metadata, no data
            var k = q - (nb - b.nb)
            if k >= 0:
                fy.append(_Axis(b.be[k], _stride(a, q, nb) if b.be[k] > 1 else 0, gs[q] if b.be[k] > 1 else 0))
                cy *= b.be[k]
            if (k < 0 or b.be[k] == 1) and d[q] > 1:
                oy.append(_Axis(d[q], _stride(a, q, nb), gs[q]))
        # X^T(k, m) = X(m, k): y's row k is x's column k
        fy.append(_Axis(K, a.cs, 0))
        fy.append(_Axis(N, 0, gs[nb + 1]))
        saved.p10 = tape.descriptor(ex, _words(fy, oy, M, a.rs, gs[nb]))
        saved.i3 = 1
        saved.i5 = cy * K * N
    return tape.custom_nd(ex, [x, y], output, shape, RN_BMM, saved)


@fieldwise_init
struct RankNBmmRules(BackwardRules):
    """The VJPs of rankn_matmul's RN_BMM node: one OP_RN_BMM launch per
    operand that wants a gradient, every destination word written once."""

    def supports(self, tag: Int) -> Bool:
        return tag == RN_BMM

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # every word of a wanted gradient is written; an unwanted one is untouched
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag != RN_BMM:
            raise Error("rank-N bmm rule: unknown tag")
        if saved.i2 == 1:
            # dX = G @ Y^T: A = the upstream (base 0), B = y's buffer
            _launch(ex, gradients[0], upstream, 0, saved.p1, saved.i1, saved.p9, saved.i4, False)
        if saved.i3 == 1:
            # dY = X^T @ G: A = x's buffer, B = the upstream (base 0)
            _launch(ex, gradients[1], saved.p0, saved.i0, upstream, 0, saved.p10, saved.i5, False)
