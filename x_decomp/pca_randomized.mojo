# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`PCA(svd_solver='randomized').fit` as ONE native operation (lane
activate-pca-owner, 2026-10-21): scikit-learn's `_pca.py::_fit_truncated`
through `randomized_svd` (Halko et al.), the steps `_expansion_decomp.
_randomized_decompose` / `_rsvd_core` sequenced from Python with the decomp
kit (`_Kit`) as one x_decomp call each.

SAME CALLS, SAME ORDER, SAME OPERANDS. Every step below is the executor or
resident entry the Python kit called for it (`x_decomp_ew`,
`x_decomp_dev_gemm`, `x_decomp_dev_orth`, ...), reached through its typed
body (`*_run`) instead of the Python door, with the same broadcast modes,
float32 scalars, Philox seed and stream, power iterations, rank guard,
zero padding and `svd_flip` sign rule. A matrix lives where the Python kit
kept it: on a GPU binding every elementwise, product, fold, order and move
step runs on the resident device matrices (`RES`), and the small SVD and
the paddings read host words, as before. Host matrices are this kit's own
buffers; the caller's X is read in place and never written.

`RDev` is the resident half (pool, upload/download and the device entries);
`NoRDev` stands in on the host binding, where nothing is resident.
"""
from std.memory import memcpy

from x_decomp.api import absmax_sign_run, ew_run, gemm_run, move_run, order_f_run, rand_run, svd_run
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.select_ops import order_small_run

# x_decomp/cells.mojo op codes (`_OP` in _expansion_decomp.py)
comptime _OP_SUB = 1
comptime _OP_MUL = 2
comptime _OP_DIV = 3
comptime _OP_SQ = 8
comptime _OP_ABS = 13
comptime _OP_SCALE = 14
comptime _OP_ADDS = 22
comptime _OP_GTS = 23
comptime _OP_SELECT = 35
# x_decomp/moves.mojo move ops (`_MV_*` in _expansion_decomp.py)
comptime _MV_TAKE_ROWS = 0
comptime _MV_TRANSPOSE = 1
comptime _MV_TAKE_COLS = 4
#: `_Kit._SEL_MAX`: the device reduce's max (a NaN anywhere gives NaN)
comptime _SEL_MAX = 1
#: `_Kit.rand`: device draws from this many values up, host draws below
comptime _RAND_RESIDENT_MIN = 1024
#: the Philox stream and kind `_rsvd_core` draws its test matrix from
comptime _RSVD_STREAM = 30
comptime _RSVD_KIND = 1


trait RDev:
    """The resident (device) entries the Python kit reached on a GPU binding."""

    @staticmethod
    def alloc(n: Int) raises -> Int:
        ...

    @staticmethod
    def free(id: Int) raises:
        ...

    @staticmethod
    def upload(id: Int, src: F32Ptr, n: Int) raises:
        ...

    @staticmethod
    def download(id: Int, dst: F32Ptr, n: Int) raises:
        ...

    @staticmethod
    def ew(
        a: Int, b: Int, c: Int, dst: Int, op: Int, count: Int, d: Int, lb: Int, bm: Int, lc: Int, cm: Int,
        s: Float64,
    ) raises:
        ...

    @staticmethod
    def gemm(a: Int, b: Int, c: Int, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises:
        ...

    @staticmethod
    def colsum(a: Int, dst: Int, n: Int, d: Int) raises:
        ...

    @staticmethod
    def rowsum(a: Int, dst: Int, n: Int, d: Int) raises:
        ...

    @staticmethod
    def move(
        src: Int, idx: Int, dst: Int, op: Int, count: Int, a1: Int, a2: Int, a3: Int, ist: Int, ioff: Int,
        nsrc: Int, ndst: Int, nidx: Int,
    ) raises:
        ...

    @staticmethod
    def rand(dst: Int, count: Int, seed: Int, stream: Int, kind: Int) raises:
        ...

    @staticmethod
    def orth(a: Int, dst: Int, m: Int, l: Int) raises:
        ...

    @staticmethod
    def absmax(a: Int, dst: Int, n: Int, d: Int, by_col: Bool) raises:
        ...

    @staticmethod
    def has_svd() -> Bool:
        ...

    @staticmethod
    def svd(a: Int, s: F32Ptr, v: F32Ptr, m: Int, n: Int) raises:
        ...

    @staticmethod
    def reduce(a: Int, dst: Int, count: Int, op: Int) raises:
        ...

    @staticmethod
    def order_f(a: Int, skip: Int, dst: Int, cnt: Int, n: Int, neg: Int, has: Int) raises:
        ...

    @staticmethod
    def order_small(a: Int, dst: Int, n: Int) raises:
        ...


struct NoRDev(RDev):
    """The host binding: no resident entries (a kit with RES False never
    calls these)."""

    @staticmethod
    def alloc(n: Int) raises -> Int:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def free(id: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def upload(id: Int, src: F32Ptr, n: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def download(id: Int, dst: F32Ptr, n: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def ew(
        a: Int, b: Int, c: Int, dst: Int, op: Int, count: Int, d: Int, lb: Int, bm: Int, lc: Int, cm: Int,
        s: Float64,
    ) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def gemm(a: Int, b: Int, c: Int, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def colsum(a: Int, dst: Int, n: Int, d: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def rowsum(a: Int, dst: Int, n: Int, d: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def move(
        src: Int, idx: Int, dst: Int, op: Int, count: Int, a1: Int, a2: Int, a3: Int, ist: Int, ioff: Int,
        nsrc: Int, ndst: Int, nidx: Int,
    ) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def rand(dst: Int, count: Int, seed: Int, stream: Int, kind: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def orth(a: Int, dst: Int, m: Int, l: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def absmax(a: Int, dst: Int, n: Int, d: Int, by_col: Bool) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def has_svd() -> Bool:
        return False

    @staticmethod
    def svd(a: Int, s: F32Ptr, v: F32Ptr, m: Int, n: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def reduce(a: Int, dst: Int, count: Int, op: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def order_f(a: Int, skip: Int, dst: Int, cnt: Int, n: Int, neg: Int, has: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def order_small(a: Int, dst: Int, n: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")


struct RKMat(Copyable, Movable):
    """One kit matrix: its host words (`host` an address, 0 when it lives on
    the device) or its device matrix (`dev` a pool id, -1 when on the host)."""

    var host: Int
    var dev: Int
    var rows: Int
    var cols: Int

    def __init__(out self, host: Int, dev: Int, rows: Int, cols: Int):
        self.host = host
        self.dev = dev
        self.rows = rows
        self.cols = cols


struct RKit[E: Exec, D: RDev, RES: Bool](Movable):
    """`_Kit` for randomized PCA: matrices by index; `RES` a GPU binding
    (resident entries), else every step on the executor `E`."""

    var mats: List[RKMat]
    var hosts: List[List[Float32]]
    var allocations: List[Int]

    def __init__(out self):
        self.mats = List[RKMat]()
        self.hosts = List[List[Float32]]()
        self.allocations = List[Int]()

    def close(mut self) raises:
        # Views may share a device id, so only the allocation ledger frees.
        for i in range(len(self.allocations) - 1, -1, -1):  # driver-loop: over this fit's device allocations
            Self.D.free(self.allocations[i])
        self.allocations.clear()

    def r(self, a: Int) -> Int:
        return self.mats[a].rows

    def c(self, a: Int) -> Int:
        return self.mats[a].cols

    def add(mut self, host: Int, dev: Int, rows: Int, cols: Int) -> Int:
        self.mats.append(RKMat(host, dev, rows, cols))
        return len(self.mats) - 1

    def _host_words(mut self, n: Int) -> Int:
        # One spare word gives empty matrices a valid address, as _M._one.
        self.hosts.append(List[Float32](length=max(n, 1), fill=Float32(0)))
        return Int(self.hosts[len(self.hosts) - 1].unsafe_ptr())

    def zeros(mut self, rows: Int, cols: Int) -> Int:
        return self.add(self._host_words(rows * cols), -1, rows, cols)

    def device_output(mut self, rows: Int, cols: Int) raises -> Int:
        var handle = Self.D.alloc(max(rows * cols, 1))
        self.allocations.append(handle)
        return self.add(0, handle, rows, cols)

    def host(mut self, a: Int) raises -> Int:
        """`_M.s`: the host words, downloaded once (the matrix moves)."""
        if self.mats[a].host == 0:
            var n = self.mats[a].rows * self.mats[a].cols
            var words = self._host_words(n)
            if n:
                Self.D.download(self.mats[a].dev, F32Ptr(unsafe_from_address=words), n)
            self.mats[a].host = words
            self.mats[a].dev = -1
        return self.mats[a].host

    def ptr(mut self, a: Int) raises -> F32Ptr:
        return F32Ptr(unsafe_from_address=self.host(a))

    def device(mut self, a: Int) raises -> Int:
        """`_Kit._did`: the device id, uploading host words once (it moves)."""
        if self.mats[a].dev < 0:
            var n = self.mats[a].rows * self.mats[a].cols
            var words = self.host(a)
            var handle = Self.D.alloc(max(n, 1))
            self.allocations.append(handle)
            if n:
                Self.D.upload(handle, F32Ptr(unsafe_from_address=words), n)
            self.mats[a].dev = handle
            self.mats[a].host = 0
        return self.mats[a].dev

    def copy_out(mut self, a: Int, dst: F32Ptr, count: Int) raises:
        """The first `count` words of `a` into caller memory (`_M.out`)."""
        if count <= 0:
            return
        if self.mats[a].host == 0:
            Self.D.download(self.mats[a].dev, dst, count)
        else:
            memcpy(dest=dst, src=F32Ptr(unsafe_from_address=self.mats[a].host), count=count)

    def constant(mut self, value: Float32) -> Int:
        var out = self.zeros(1, 1)
        F32Ptr(unsafe_from_address=self.mats[out].host).unsafe_store(0, value)
        return out

    def indices(mut self, start: Int, stop: Int) -> Int:
        var out = self.zeros(stop - start, 1)
        var p = F32Ptr(unsafe_from_address=self.mats[out].host)
        for i in range(stop - start):  # small-loop(component indices): the move's index operand
            p.unsafe_store(i, Float32(start + i))
        return out

    def broadcast(self, a: Int, b: Int) raises -> Int:
        if self.mats[a].rows == self.mats[b].rows and self.mats[a].cols == self.mats[b].cols:
            return 0
        if self.mats[b].rows == 1 and self.mats[a].cols == self.mats[b].cols:
            return 1
        if self.mats[b].cols == 1 and self.mats[a].rows == self.mats[b].rows:
            return 2
        if self.mats[b].rows == 1 and self.mats[b].cols == 1:
            return 3
        raise Error("x_decomp: randomized PCA has an invalid internal broadcast")

    def ew(mut self, op: Int, a: Int, b: Int = -1, c: Int = -1, scalar: Float64 = 0.0) raises -> Int:
        var zero = self.zeros(1, 1)
        var ib = zero if b < 0 else b
        var ic = zero if c < 0 else c
        var bm = 3 if b < 0 else self.broadcast(a, ib)
        var cm = 3 if c < 0 else self.broadcast(a, ic)
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        var lb = self.mats[ib].rows * self.mats[ib].cols
        var lc = self.mats[ic].rows * self.mats[ic].cols
        comptime if Self.RES:
            if rows * cols:
                var out = self.device_output(rows, cols)
                var da = self.device(a)
                var db = self.device(ib)
                var dc = self.device(ic)
                Self.D.ew(da, db, dc, self.mats[out].dev, op, rows * cols, cols, lb, bm, lc, cm, scalar)
                return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var pa = self.ptr(a)
            var pb = self.ptr(ib)
            var pc = self.ptr(ic)
            ew_run[Self.E](op, pa, pb, lb, bm, pc, lc, cm, self.ptr(out), rows * cols, cols, Float32(scalar))
        return out

    def mm(mut self, a: Int, b: Int, ta: Bool = False, tb: Bool = False) raises -> Int:
        var m = self.mats[a].cols if ta else self.mats[a].rows
        var k = self.mats[a].rows if ta else self.mats[a].cols
        var k2 = self.mats[b].cols if tb else self.mats[b].rows
        var n = self.mats[b].rows if tb else self.mats[b].cols
        if k != k2:
            raise Error("x_decomp: randomized PCA has an invalid internal matrix product")
        comptime if Self.RES:
            if m * n and m * k and k * n:
                var out = self.device_output(m, n)
                var da = self.device(a)
                var db = self.device(b)
                Self.D.gemm(da, db, self.mats[out].dev, m, k, n, ta, tb)
                return out
        var out = self.zeros(m, n)
        if m * n:
            var pa = self.ptr(a)
            var pb = self.ptr(b)
            _ = gemm_run[Self.E](pa, pb, self.ptr(out), m, k, n, ta, tb)
        return out

    def sum(mut self, a: Int, columns: Bool) raises -> Int:
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        var nr = 1 if columns else rows
        var nc = cols if columns else 1
        comptime if Self.RES:
            if rows * cols:
                var out = self.device_output(nr, nc)
                var da = self.device(a)
                if columns:
                    Self.D.colsum(da, self.mats[out].dev, rows, cols)
                else:
                    Self.D.rowsum(da, self.mats[out].dev, rows, cols)
                return out
        var out = self.zeros(nr, nc)
        var pa = self.ptr(a)
        if columns:
            Self.E.colsum(pa, self.ptr(out), rows, cols)
        else:
            Self.E.rowsum(pa, self.ptr(out), rows, cols)
        return out

    def word(mut self, a: Int) raises -> Float64:
        """`_Kit.word(A)`: the first value, one word down."""
        var index = a
        if self.mats[a].dev >= 0 and self.mats[a].rows * self.mats[a].cols > 1:
            index = self.strided_first(a)
        return Float64(self.ptr(index).unsafe_load(0))

    def total(mut self, a: Int) raises -> Float64:
        """`_Kit.total`: rows ascending, then the row sums ascending."""
        var rows = self.sum(a, False)
        var total = self.sum(rows, True)
        return self.word(total)

    def move(
        mut self, a: Int, idx: Int, rows: Int, cols: Int, op: Int, count: Int, a1: Int, a2: Int, a3: Int,
        ist: Int, ioff: Int, nsrc: Int, ndst: Int, nidx: Int, resident: Bool,
    ) raises -> Int:
        comptime if Self.RES:
            if rows * cols and resident:
                var out = self.device_output(rows, cols)
                var da = self.device(a)
                var di = self.device(idx)
                Self.D.move(da, di, self.mats[out].dev, op, count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx)
                return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var pa = self.ptr(a)
            var pi = self.ptr(idx)
            move_run(op, pa, pi, self.ptr(out), count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx)
        return out

    def take_rows(mut self, a: Int, idx: Int) raises -> Int:
        var rows = self.mats[idx].rows * self.mats[idx].cols
        var cols = self.mats[a].cols
        return self.move(a, idx, rows, cols, _MV_TAKE_ROWS, rows * cols, cols, 0, 0, 1, 0,
                         self.mats[a].rows * cols, rows * cols, rows, Self.RES)

    def columns(mut self, a: Int, start: Int, stop: Int) raises -> Int:
        """`_M.cols(start, stop)`: the device gather for a resident matrix,
        else the host copy."""
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        var width = stop - start
        if start < 0 or stop > cols or width < 0:
            raise Error("x_decomp: randomized PCA column range out of bounds")
        if self.mats[a].dev >= 0 and width:
            var idx = self.indices(start, stop)
            return self.move(a, idx, rows, width, _MV_TAKE_COLS, rows * width, width, cols, 0, 0, 0,
                             rows * cols, rows * width, width, True)
        var out = self.zeros(rows, width)
        var src = self.ptr(a)
        var dst = self.ptr(out)
        for r in range(rows):  # small-loop(rows of a host-resident sketch factor): one row copy each
            memcpy(dest=dst.unsafe_offset(r * width), src=src.unsafe_offset(r * cols + start), count=width)
        return out

    def transpose(mut self, a: Int) raises -> Int:
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        if rows == 1 or cols == 1:
            # a vector's transpose is the same words (a view)
            return self.add(self.mats[a].host, self.mats[a].dev, cols, rows)
        var one = self.zeros(1, 1)
        return self.move(a, one, cols, rows, _MV_TRANSPOSE, rows * cols, rows, cols, 0, 0, 0,
                         rows * cols, rows * cols, 1, Self.RES)

    def strided_first(mut self, a: Int) raises -> Int:
        """`_Kit.strided(A, 1, 1)`: the first value as a 1 x 1 matrix."""
        var idx = self.constant(Float32(0))
        var n = self.mats[a].rows * self.mats[a].cols
        return self.move(a, idx, 1, 1, _MV_TAKE_COLS, 1, 1, 1, 0, 0, 0, n, 1, 1, Self.RES)

    def order(mut self, a: Int) raises -> Int:
        """`_Kit.order`: the stable ascending order as n x 1 exact floats;
        NaN refused."""
        var n = self.mats[a].rows * self.mats[a].cols
        comptime if Self.RES:
            if n:
                var maximum = self.device_output(1, 1)
                var da = self.device(a)
                Self.D.reduce(da, self.mats[maximum].dev, n, _SEL_MAX)
                var v = self.word(maximum)
                if v != v:
                    raise Error("x_decomp: a NaN has no order (refused)")
                var out = self.device_output(n, 1)
                var count = self.device_output(1, 1)
                Self.D.order_f(da, da, self.mats[out].dev, self.mats[count].dev, n, 0, 0)
                return out
        var out = self.zeros(n, 1)
        if n:
            order_f_run(self.host(a), n, self.host(out))
        return out

    def signs(mut self, a: Int) raises -> Int:
        """`_Kit.neg_signs`: -1 where a < 0, else +1."""
        var negated = self.ew(_OP_SCALE, a, scalar=-1.0)
        var negative = self.constant(Float32(-1.0))
        var positive = self.constant(Float32(1.0))
        return self.ew(_OP_SELECT, negated, negative, positive)

    def copy_host(mut self, a: Int) raises -> Int:
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        var out = self.zeros(rows, cols)
        var src = self.ptr(a)
        memcpy(dest=self.ptr(out), src=src, count=rows * cols)
        return out

    def rand(mut self, rows: Int, cols: Int, seed: Int) raises -> Int:
        """`_Kit.rand(rows, cols, seed, 30, 1)`."""
        var count = rows * cols
        comptime if Self.RES:
            if count >= _RAND_RESIDENT_MIN:
                var out = self.device_output(rows, cols)
                Self.D.rand(self.mats[out].dev, count, seed, _RSVD_STREAM, _RSVD_KIND)
                return out
        var out = self.zeros(rows, cols)
        if count:
            rand_run[Self.E](self.ptr(out), count, seed, _RSVD_STREAM, _RSVD_KIND)
        return out

    def orth(mut self, a: Int) raises -> Int:
        """`_Kit.orth`: a copy of a with its columns orthonormalized."""
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        comptime if Self.RES:
            if rows * cols:
                var out = self.device_output(rows, cols)
                var da = self.device(a)
                Self.D.orth(da, self.mats[out].dev, rows, cols)
                return out
        var out = self.copy_host(a)
        Self.E.orth(self.ptr(out), rows, cols)
        return out

    def absmax_signs(mut self, a: Int, columns: Bool) raises -> Int:
        """`_Kit.absmax_signs`: per column (1 x c) or row (r x 1), -1 where
        the largest-|.| entry is negative, else +1."""
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        var count = cols if columns else rows
        var out: Int
        comptime if Self.RES:
            out = self.device_output(1, count)
        else:
            out = self.zeros(1, count)
        if count and rows * cols:
            comptime if Self.RES:
                var da = self.device(a)
                Self.D.absmax(da, self.mats[out].dev, rows, cols, columns)
            else:
                var pa = self.ptr(a)
                _ = absmax_sign_run[Self.E](pa, self.ptr(out), rows, cols, columns)
        if not columns:
            out = self.transpose(out)
        return self.signs(out)

    def order_small(mut self, a: Int) raises -> Int:
        """`_Kit.order_small`: the stable ascending order (n <= 65536)."""
        var count = self.mats[a].rows * self.mats[a].cols
        var out: Int
        comptime if Self.RES:
            out = self.device_output(count, 1)
        else:
            out = self.zeros(count, 1)
        if count:
            comptime if Self.RES:
                var da = self.device(a)
                Self.D.order_small(da, self.mats[out].dev, count)
            else:
                var pa = self.ptr(a)
                order_small_run(pa, self.ptr(out), count)
        return out

    def tall_svd(mut self, a: Int) raises -> Tuple[Int, Int]:
        """`_Kit.svd` of a tall a: (S 1 x n descending, Vt n x n)."""
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        var s = self.zeros(1, cols)
        var v = self.zeros(cols, cols)
        var ps = self.ptr(s)
        var pv = self.ptr(v)
        var resident_svd = False
        comptime if Self.RES:
            resident_svd = Self.D.has_svd() and rows >= cols and cols >= 1
        if resident_svd:
            var da = self.device(a)
            Self.D.svd(da, ps, pv, rows, cols)
        else:
            svd_run[Self.E](self.ptr(a), ps, pv, rows, cols)
        var negated = self.ew(_OP_SCALE, s, scalar=-1.0)
        var order = self.order(negated)
        var st = self.transpose(s)
        var sorted_s = self.take_rows(st, order)
        s = self.transpose(sorted_s)
        var vt = self.transpose(v)
        vt = self.take_rows(vt, order)
        return (s, vt)

    def thin_svd(mut self, a: Int) raises -> Tuple[Int, Int, Int]:
        """`_thin_svd(k, a, min(a.shape), u_based=True)`."""
        var rows = self.mats[a].rows
        var cols = self.mats[a].cols
        var u: Int
        var s: Int
        var vt: Int
        if cols <= rows:
            var raw = self.tall_svd(a)
            s = raw[0]
            vt = raw[1]
            var av = self.mm(a, vt, tb=True)
            u = self.ew(_OP_DIV, av, s)
        else:
            var at = self.transpose(a)
            var raw = self.tall_svd(at)
            s = raw[0]
            u = self.transpose(raw[1])
            var ua = self.mm(u, a, ta=True)
            var st = self.transpose(s)
            vt = self.ew(_OP_DIV, ua, st)
        var signs = self.absmax_signs(u, True)
        u = self.ew(_OP_MUL, u, signs)
        var signst = self.transpose(signs)
        vt = self.ew(_OP_MUL, vt, signst)
        return (u, s, vt)

    def pad_rows(mut self, a: Int, rows: Int) raises -> Int:
        """`_rsvd_core`'s zero padding: a's host words then zero rows."""
        var old_rows = self.mats[a].rows
        var cols = self.mats[a].cols
        if rows < old_rows:
            raise Error("x_decomp: randomized PCA padding cannot remove rows")
        var out = self.zeros(rows, cols)
        var src = self.ptr(a)
        memcpy(dest=self.ptr(out), src=src, count=old_rows * cols)
        return out


def pca_randomized_core[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], original: Int, components: Int, oversamples: Int, iterations: Int, seed: Int
) raises -> Tuple[Int, Int, Int]:
    """`_rsvd_core(k, A, components, oversamples, iterations, 'auto',
    'auto', flip_sign=False, seed)`: (U, S, Vt)."""
    var transposed = kit.r(original) < kit.c(original)
    var a = kit.transpose(original) if transposed else original
    var rows = kit.r(a)
    var cols = kit.c(a)
    var width = min(components + oversamples, min(rows, cols))
    if width < components:
        raise Error("randomized PCA: the sketch is narrower than n_components")
    var q = kit.rand(cols, width, seed)
    for _ in range(max(iterations, 0)):  # driver-loop: over the requested randomized-SVD power iterations
        var aq = kit.mm(a, q)
        q = kit.orth(aq)
        var atq = kit.mm(a, q, ta=True)
        q = kit.orth(atq)
    var aq = kit.mm(a, q)
    q = kit.orth(aq)
    # The orth rank guard leaves a numerically dependent column of the
    # range exactly 0; the small SVD runs on the live columns only and the
    # components past the numerical rank are zero (DEVIATION 5318).
    var absolute = kit.ew(_OP_ABS, q)
    var sums = kit.sum(absolute, True)
    var absolute_sums = kit.ew(_OP_ABS, sums)
    var live = kit.ew(_OP_GTS, absolute_sums)
    var nlive = Int(kit.total(live))
    if nlive == 0:
        raise Error("randomized_svd: M is numerically zero")
    var full = kit.c(q)
    if nlive < full:
        var negated = kit.ew(_OP_SCALE, live, scalar=-1.0)
        var dead = kit.ew(_OP_ADDS, negated, scalar=1.0)
        var order = kit.order_small(dead)
        q = kit.move(q, order, rows, nlive, _MV_TAKE_COLS, rows * nlive, nlive, full, 0, 0, 0,
                     rows * full, rows * nlive, nlive, RES)
    var small = kit.mm(q, a, ta=True)
    var decomposed = kit.thin_svd(small)
    var uh = decomposed[0]
    var s = decomposed[1]
    var vt = decomposed[2]
    if kit.c(s) < full:
        var st = kit.transpose(s)
        st = kit.pad_rows(st, full)
        s = kit.transpose(st)
        vt = kit.pad_rows(vt, full)
        var uht = kit.transpose(uh)
        uht = kit.pad_rows(uht, full)
        uh = kit.transpose(uht)
    var u = kit.mm(q, uh)
    var component_rows = kit.indices(0, components)
    var out_s = kit.columns(s, 0, components)
    if transposed:
        var out_u = kit.take_rows(vt, component_rows)
        out_u = kit.transpose(out_u)
        var out_v = kit.columns(u, 0, components)
        out_v = kit.transpose(out_v)
        return (out_u, out_s, out_v)
    var out_u = kit.columns(u, 0, components)
    var out_v = kit.take_rows(vt, component_rows)
    return (out_u, out_s, out_v)


def pca_randomized_fit_run[E: Exec, D: RDev, RES: Bool](
    x: F32Ptr, components_out: F32Ptr, mean_out: F32Ptr, variance_out: F32Ptr, ratio_out: F32Ptr,
    singular_out: F32Ptr, rows: Int, cols: Int, components: Int, oversamples: Int, iterations: Int,
    auto_iterations: Bool, seed: Int,
) raises -> Float64:
    """`_randomized_decompose(X, components, center=True, ...)` then
    `svd_flip(u_based_decision=False)`: the five public arrays written
    (components x cols, cols, and components each), the noise variance
    returned. `auto_iterations`: scikit-learn's `iterated_power='auto'`."""
    if rows < 2 or cols < 2:
        raise Error("randomized PCA needs at least two rows and two features")
    if components < 1 or components >= min(rows, cols):
        raise Error("svd_solver='randomized' needs n_components < min(n_samples, n_features)")
    if oversamples < 0:
        raise Error("n_oversamples must be nonnegative")
    var power = iterations
    if auto_iterations:
        # scikit-learn's `randomized_svd` default, by the problem's size
        # ratio (not a dimension threshold): more power iterations when few
        # components are asked of a large spectrum.
        power = 7 if Float64(components) < 0.1 * Float64(min(rows, cols)) else 4
    var kit = RKit[E, D, RES]()
    try:
        var original = kit.add(Int(x), -1, rows, cols)
        var sums = kit.sum(original, True)
        var mean = kit.ew(_OP_SCALE, sums, scalar=1.0 / Float64(rows))
        var centered = kit.ew(_OP_SUB, original, mean)
        var decomposed = pca_randomized_core(kit, centered, components, oversamples, power, seed)
        var u = decomposed[0]
        var s = decomposed[1]
        var vt = decomposed[2]
        var signs = kit.absmax_signs(vt, False)
        vt = kit.ew(_OP_MUL, vt, signs)
        var signst = kit.transpose(signs)
        u = kit.ew(_OP_MUL, u, signst)
        var squared = kit.ew(_OP_SQ, s)
        var variance = kit.ew(_OP_SCALE, squared, scalar=1.0 / Float64(rows - 1))
        var data_squared = kit.ew(_OP_SQ, centered)
        var total_value = kit.total(data_squared)
        var total = kit.constant(Float32(total_value))
        total = kit.ew(_OP_SCALE, total, scalar=1.0 / Float64(rows - 1))
        var ratio = kit.ew(_OP_DIV, variance, total)
        var explained_value = kit.total(variance)
        var explained = kit.constant(Float32(explained_value))
        var noise = kit.ew(_OP_SUB, total, explained)
        noise = kit.ew(_OP_SCALE, noise, scalar=1.0 / Float64(min(rows, cols) - components))
        kit.copy_out(vt, components_out, components * cols)
        kit.copy_out(mean, mean_out, cols)
        kit.copy_out(s, singular_out, components)
        kit.copy_out(variance, variance_out, components)
        kit.copy_out(ratio, ratio_out, components)
        var noise_value = kit.word(noise)
        kit.close()
        return noise_value
    except error:
        kit.close()
        raise error^
