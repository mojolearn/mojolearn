# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`_expansion_decomp._Kit` in Mojo (lane/py-decomp-nbrs, 2026-09-28): the
same executor calls Python's `_Kit` makes, with the same operands,
broadcast modes (`mode_of`) and float32 scalars, for drivers that used to
loop in Python (x_decomp/mcd.mojo, x_decomp/lda_online.mojo).

ONE EXECUTOR. `Kit[E]` sends every call to `E`. The CPU binding runs it
on its executor; the GPU binding runs the same drivers on resident device
matrices instead (x_decomp/kit_device.mojo), the same cells to the same
bits (every x-decomp lane's GPU == CPU claim)."""
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.moves import MOVE_TRANSPOSE, move_src
from x_decomp.select_ops import SEL_ORDER_MAX, order_rank, sel_fold

# x_decomp/cells.mojo op codes (`_OP` in _expansion_decomp.py)
comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_EXP = 9
comptime OP_LOGS = 10
comptime OP_ABS = 13
comptime OP_SCALE = 14
comptime OP_RECIP = 16
comptime OP_ADDS = 22
comptime OP_DIGAMMA = 24
comptime OP_AXPY = 4
comptime OP_MAXS = 5
comptime OP_SQRT = 7
comptime OP_SQ = 8
comptime OP_MINS = 19
comptime OP_SQDIFF = 21
comptime OP_GTS = 23
comptime OP_SELECT = 35
comptime OP_MUZ = 36
comptime OP_TANH = 11
comptime OP_ONEMSQ = 12
comptime OP_EXPG = 25
comptime OP_EXPGP = 26
comptime OP_CUBE = 27
comptime OP_CUBEP = 28


struct Mat(Copyable, Movable):
    """A row-major float32 matrix (`_M`'s store). Never empty in memory: a
    0-element matrix keeps one slot so its address is valid."""
    var d: List[Float32]
    var r: Int
    var c: Int

    def __init__(out self, r: Int, c: Int):
        self.d = List[Float32](length=max(r * c, 1), fill=Float32(0))
        self.r = r
        self.c = c

    def p(self) -> F32Ptr:
        return F32Ptr(unsafe_from_address=Int(self.d.unsafe_ptr()))

    def n(self) -> Int:
        return self.r * self.c


def mat_from(ptr: F32Ptr, r: Int, c: Int) -> Mat:
    var m = Mat(r, c)
    for i in range(r * c):
        m.d[i] = ptr.unsafe_load(i)
    return m^


def mat_const(v: Float64, r: Int, c: Int) -> Mat:
    """`_M.of([v] * (r * c), r, c)`: v rounded once to float32."""
    var m = Mat(r, c)
    var f = Float32(v)
    for i in range(r * c):
        m.d[i] = f
    return m^


def mat_rows(X: Mat, a: Int, b: Int) -> Mat:
    """`_M.rows(a, b)`: an exact copy of rows a..b-1."""
    var out = Mat(b - a, X.c)
    var off = a * X.c
    for i in range((b - a) * X.c):
        out.d[i] = X.d[off + i]
    return out^


def mat_vec_t(var X: Mat) -> Mat:
    """`_M.T` of a vector (one row or one column): the same words."""
    var r = X.r
    X.r = X.c
    X.c = r
    return X^


def mat_t(X: Mat) -> Mat:
    """`_M.T`: the TRANSPOSE move's index function (exact copies)."""
    if X.r == 1 or X.c == 1:
        return mat_vec_t(X.copy())
    var out = Mat(X.c, X.r)
    var none = X.p()
    for t in range(X.n()):
        out.d[t] = X.d[move_src(MOVE_TRANSPOSE, t, none, X.r, X.c, 0, 0)]
    return out^


def mat_eye(n: Int) -> Mat:
    """`_eye(n)` (and the resident `diag_mask`): 1 on the diagonal."""
    var out = Mat(n, n)
    for i in range(n):
        out.d[i * n + i] = Float32(1)
    return out^


def take_rows(X: Mat, sel: List[Int]) -> Mat:
    """`_M.take_rows`: the rows of X in the order of `sel` (exact copies)."""
    var out = Mat(len(sel), X.c)
    var c = X.c
    for a in range(len(sel)):
        var src = sel[a] * c
        for j in range(c):
            out.d[a * c + j] = X.d[src + j]
    return out^


struct Kit[E: Exec](Movable):
    var one: Mat

    def __init__(out self):
        self.one = Mat(1, 1)

    @staticmethod
    def mode(X: Mat, A: Mat) raises -> Int:
        """`_Kit.ew`'s `mode_of`, in its order."""
        if X.r == A.r and X.c == A.c:
            return 0
        if X.r == 1 and X.c == A.c:
            return 1
        if X.c == 1 and X.r == A.r:
            return 2
        if X.r == 1 and X.c == 1:
            return 3
        raise Error("x_decomp: cannot broadcast")

    def ew1(self, op: Int, A: Mat, s: Float64) raises -> Mat:
        """`ew(op, A, s=s)`: B and C the unused 0 broadcast operand."""
        var out = Mat(A.r, A.c)
        if A.n() == 0:
            return out^
        Self.E.ew(op, A.p(), self.one.p(), 1, 3, self.one.p(), 1, 3, out.p(), A.n(), A.c, Float32(s))
        return out^

    def ew2(self, op: Int, A: Mat, B: Mat) raises -> Mat:
        """`ew(op, A, B)` (s = 0)."""
        var bm = Self.mode(B, A)
        var out = Mat(A.r, A.c)
        if A.n() == 0:
            return out^
        Self.E.ew(op, A.p(), B.p(), B.n(), bm, self.one.p(), 1, 3, out.p(), A.n(), A.c, Float32(0))
        return out^

    def ew2s(self, op: Int, A: Mat, B: Mat, s: Float64) raises -> Mat:
        """`ew(op, A, B, s=s)`."""
        var bm = Self.mode(B, A)
        var out = Mat(A.r, A.c)
        if A.n() == 0:
            return out^
        Self.E.ew(op, A.p(), B.p(), B.n(), bm, self.one.p(), 1, 3, out.p(), A.n(), A.c, Float32(s))
        return out^

    def ew3(self, op: Int, A: Mat, B: Mat, C: Mat, s: Float64) raises -> Mat:
        """`ew(op, A, B, C, s=s)`."""
        var bm = Self.mode(B, A)
        var cm = Self.mode(C, A)
        var out = Mat(A.r, A.c)
        if A.n() == 0:
            return out^
        Self.E.ew(op, A.p(), B.p(), B.n(), bm, C.p(), C.n(), cm, out.p(), A.n(), A.c, Float32(s))
        return out^

    def reduce(self, A: Mat, op: Int) raises -> Mat:
        """`_Kit.reduce`: max |A| (0), max (1) or min (2) as 1 x 1."""
        if A.n() == 0:
            raise Error("x_decomp: reduce of an empty matrix")
        var out = Mat(1, 1)
        out.d[0] = sel_fold(op, A.p(), 0, A.n())
        return out^

    @staticmethod
    def place_row(mut D: Mat, V: Mat, row: Int):
        """`_Kit.place_rows(D, V, row)`: V's rows copied into D from `row`."""
        var off = row * D.c
        for i in range(V.n()):
            D.d[off + i] = V.d[i]

    def word(self, A: Mat) -> Float64:
        """`A.s[0]` as Python reads it (the float32 word, exactly)."""
        return Float64(A.d[0])

    def order_small(self, A: Mat) raises -> List[Int32]:
        """`_Kit.order_small` read as ints: the stable ascending order of A's
        values (x_decomp/select_ops.mojo `order_rank`)."""
        var n = A.n()
        if n > SEL_ORDER_MAX:
            raise Error("x_decomp: order_small exceeds its bound")
        var out = List[Int32](length=max(n, 1), fill=Int32(0))
        for i in range(n):
            out[order_rank(A.p(), n, i)] = Int32(i)
        return out^

    def cd_rows(self, mut W: Mat, HHt: Mat, XHt: Mat, perm: List[Int32]) raises -> Float64:
        """`_Kit.cd_rows`: one sweep over W's rows in place; the folded
        violation `total(viol).s[0]`."""
        var viol = Mat(W.r, 1)
        var pp = I32Ptr(unsafe_from_address=Int(perm.unsafe_ptr()))
        Self.E.cd_rows(W.p(), HHt.p(), XHt.p(), pp, viol.p(), W.r, W.c)
        return self.word(self.total(viol))

    def mm(self, A: Mat, B: Mat, ta: Bool, tb: Bool) raises -> Mat:
        var m = A.c if ta else A.r
        var k = A.r if ta else A.c
        var k2 = B.c if tb else B.r
        var n = B.r if tb else B.c
        if k != k2:
            raise Error("x_decomp: gemm inner dimensions differ")
        var out = Mat(m, n)
        if m * n == 0:
            return out^
        Self.E.gemm(A.p(), B.p(), out.p(), m, k, n, ta, tb)
        return out^

    def colsum(self, A: Mat) raises -> Mat:
        var out = Mat(1, A.c)
        Self.E.colsum(A.p(), out.p(), A.r, A.c)
        return out^

    def rowsum(self, A: Mat) raises -> Mat:
        var out = Mat(A.r, 1)
        Self.E.rowsum(A.p(), out.p(), A.r, A.c)
        return out^

    def total(self, A: Mat) raises -> Mat:
        """`_Kit.total`: colsum(rowsum(A))."""
        return self.colsum(self.rowsum(A))

    def colmean(self, A: Mat) raises -> Mat:
        return self.ew1(OP_SCALE, self.colsum(A), 1.0 / Float64(A.r))

    def eigh(self, A: Mat, mut w: Mat, mut v: Mat) raises:
        Self.E.eigh(A.p(), w.p(), v.p(), A.r, 0)

    def lu(self, mut lu: Mat, mut piv: List[Int32], mut info: Mat) raises:
        """`_Kit.lu` on a copy the caller made (in place)."""
        var pp = I32Ptr(unsafe_from_address=Int(piv.unsafe_ptr()))
        Self.E.lu(lu.p(), pp, info.p(), lu.r)

    def rand(self, r: Int, c: Int, seed: Int, stream: Int, kind: Int) raises -> Mat:
        var out = Mat(r, c)
        if r * c == 0:
            return out^
        var sd = UInt32(seed & 0xFFFFFFFF)
        var st = UInt32(stream & 0xFFFFFFFF)
        Self.E.rand(out.p(), r * c, sd, st, kind)
        return out^

    def rand_gamma(self, r: Int, c: Int, seed: Int, stream: Int, shape: Float64) raises -> Mat:
        var out = Mat(r, c)
        if r * c == 0:
            return out^
        var a = Float32(shape)
        if not (a >= Float32(1)):
            raise Error("x_decomp: the gamma sampler takes shape >= 1")
        var sd = UInt32(seed & 0xFFFFFFFF)
        var st = UInt32(stream & 0xFFFFFFFF)
        Self.E.rand_gamma(out.p(), r * c, sd, st, a)
        return out^

    def lda_rows(
        self, X: Mat, EW: Mat, mut Dt: Mat, mut Et: Mat, prior: Float64, max_iter: Int, tol: Float64
    ) raises:
        """`_Kit.lda_rows`: Dt and Et (n x k) updated in place."""
        var n = X.r
        var k = EW.r
        var v = X.c
        var its = Mat(n, 1)
        var s = List[Float32](length=n * (v + k) if n > 0 else 1, fill=Float32(0))
        var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
        Self.E.lda_rows(X.p(), EW.p(), Dt.p(), Et.p(), ps, its.p(), n, k, v, Float32(prior), max_iter, Float32(tol))
        _ = s^
