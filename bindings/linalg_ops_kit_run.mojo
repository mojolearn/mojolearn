# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the linalg decompositions (qr, eigh, svdvals, svd): the typed
x_decomp kit, python/mojolearn/_expansion_decomp.py's `_M` / `_Kit` for the
operations `_linalg_impl` used, entry for entry (lane entangled-kits,
2026-11-12; was the Python-holding `LinKit` of bindings/linalg_ops_kit.mojo).

A matrix lives on the host (a kit-owned float32 store, or a caller buffer the
door admitted and keeps alive: `borrow`) or on the device (a pool id of the
x_decomp image). A host matrix handed to a resident operation is uploaded once
and MOVES to the device; a device matrix read on the host is downloaded once
and moves back, exactly as `_M.s` / `_Kit._did` moved them. Every operation is
one call through the x_decomp image's dispatch (`EntryRef`, resolved by the
door's UNPACK); no Python object is held or called. The same operations on the
same words in the same order: the bits do not move.

The door (bindings/linalg_ops_decomp.mojo) releases the GIL around each RUN
sequence and keeps every admitted owner alive across it; PACK copies a
matrix's words into the public Array (`lin_kit_out`).
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call
from core.classical_tsqr_impl import tsqr_fill_constant, tsqr_copy_columns, tsqr_fill_eye
from x_decomp.descriptor_schema import (
    XD_CAP_RESIDENT, XD_CAP_DEV_SVD,
    XD_EW, XD_GEMM, XD_COLSUM, XD_ROWSUM, XD_MOVE,
    XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD, XD_DEV_DOWNLOAD, XD_DEV_EW, XD_DEV_GEMM,
    XD_DEV_COLSUM, XD_DEV_ROWSUM, XD_DEV_MOVE,
    XD_ORDER, XD_DEV_REDUCE, XD_DEV_ORDER, XD_ORDER_SMALL, XD_DEV_ORDER_SMALL,
    XD_SVD, XD_DEV_SVD, XD_RANK, XD_ORTH_DIAG, XD_DEV_ORTH_DIAG, XD_GEQRF, XD_ORGQR, XD_EIGH,
)

comptime KIT_F32P = MutPointer[Float32, MutUntrackedOrigin]

#: x_decomp/moves.mojo `move` ops
comptime MV_TAKE_ROWS = 0
comptime MV_TRANSPOSE = 1
comptime MV_PLACE_COLS = 2
comptime MV_TAKE_COLS = 4
comptime F32_INDEX_MAX = 1 << 24
comptime I32_MAX = (1 << 31) - 1

#: `_expansion_decomp._OP` codes the decompositions use
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_ABS = 13
comptime OP_SCALE = 14
comptime OP_ADDS = 22
comptime OP_LE = 34
comptime OP_SELECT = 35


struct KitStore(Movable):
    """A kit-owned host store (at least one word, so an empty matrix keeps a
    valid address, as `_M._one` did) and the matrices that view it."""
    var words: List[Float32]
    var references: Int

    def __init__(out self, cells: Int):
        self.words = List[Float32](length=max(cells, 1), fill=0.0)
        self.references = 0


struct KitMatrix(Copyable, Movable):
    # `store`: the kit store holding the host words, -1 when none.
    # `borrowed` (row N293): the address of an admitted caller buffer the
    # matrix reads in place (no host copy), kept alive by the door; 0 when the
    # matrix does not borrow. A borrowed matrix is a host matrix for every
    # route decision; it is never a write target (every operation writes a
    # fresh output).
    var store: Int
    var device: Int
    var rows: Int
    var cols: Int
    var borrowed: Int

    def __init__(out self, store: Int, device: Int, rows: Int, cols: Int):
        self.store = store
        self.device = device
        self.rows = rows
        self.cols = cols
        self.borrowed = 0


struct LinKit(Movable):
    """`_Kit(mode)` over the x_decomp image of the process's default mode."""
    var entry: EntryRef
    var matrices: List[KitMatrix]
    var stores: List[KitStore]
    var allocations: List[Int]
    var resident: Bool  # `_Kit._res`: the image has the resident operations (a GPU image)
    var dev_svd: Bool   # the image registers the resident SVD (IDN_SVD_RESIDENT)
    var zero: Int  # the kit's one 1 x 1 zero operand (row N294), -1 until made
    var minus_one: Int  # neg_signs' constants, made once per kit (-1 until made)
    var plus_one: Int

    def __init__(out self, entry: EntryRef, capabilities: Int):
        """`entry` and `capabilities` come from the door's UNPACK (the image's
        connected dispatch and its capability word, read once per binding)."""
        self.entry = entry
        self.matrices = List[KitMatrix]()
        self.stores = List[KitStore]()
        self.allocations = List[Int]()
        self.resident = (capabilities & XD_CAP_RESIDENT) != 0
        self.dev_svd = self.resident and (capabilities & XD_CAP_DEV_SVD) != 0
        self.zero = -1
        self.minus_one = -1
        self.plus_one = -1

    # ---- the x_decomp dispatch (x_decomp/descriptor_schema.mojo)
    def dcall(mut self, op: Int, ints: List[Int], s: Float64 = 0.0) raises -> Int:
        """A device-resident operation on pool ids."""
        var floats = List[Float64]()
        if op == XD_DEV_EW:
            floats.append(s)
        return entry_ref_call(self.entry, op, List[Int](), ints, floats)

    def hcall(mut self, op: Int, addresses: List[Int], ints: List[Int], s: Float64 = 0.0) raises -> Int:
        """A host-address operation; the caller keeps every address alive."""
        var floats = List[Float64]()
        if op == XD_EW:
            floats.append(s)
        return entry_ref_call(self.entry, op, addresses, ints, floats)

    def call(mut self, op: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
        return entry_ref_call(self.entry, op, addresses, ints, floats)

    def shared_zero(mut self) raises -> Int:
        """The kit's 1 x 1 zero (row N294): the absent-operand and unused-index
        constant. Read-only, so one per kit serves all."""
        if self.zero < 0:
            self.zero = self.zeros(1, 1)
        return self.zero

    def close(mut self):
        """Return every device buffer this call allocated to the pool."""
        for i in range(len(self.allocations) - 1, -1, -1):  # driver-loop: over native matrix allocation handles
            try:
                _ = self.dcall(XD_DEV_FREE, [self.allocations[i]])
            except:
                pass
        self.allocations.clear()

    def add(mut self, store: Int, rows: Int, cols: Int, device: Int = -1) -> Int:
        if store >= 0:
            self.stores[store].references += 1
        var result = len(self.matrices)
        self.matrices.append(KitMatrix(store, device, rows, cols))
        return result

    def _release_store(mut self, a: Int):
        # A matrix that moved to the device drops its host store; the words go
        # when no view (reshape) still reads them, as the Python `_M` dropped
        # its array.
        var store = self.matrices[a].store
        if store >= 0:
            self.stores[store].references -= 1
            if self.stores[store].references == 0:
                self.stores[store].words = List[Float32]()
            self.matrices[a].store = -1

    def n(self, a: Int) -> Int:
        return self.matrices[a].rows * self.matrices[a].cols

    def r(self, a: Int) -> Int:
        return self.matrices[a].rows

    def c(self, a: Int) -> Int:
        return self.matrices[a].cols

    def on_device(self, a: Int) -> Bool:
        return self.matrices[a].device >= 0

    def zeros(mut self, rows: Int, cols: Int) -> Int:
        """`_M.zeros(rows, cols)`."""
        var store = len(self.stores)
        self.stores.append(KitStore(rows * cols))
        return self.add(store, rows, cols)

    def borrow(mut self, addr: Int, rows: Int, cols: Int) -> Int:
        """`_xd_matrix` without the copy (row N293): a host matrix that reads
        the admitted C-order float32 buffer at `addr` in place (the door keeps
        its owner alive). A resident operation uploads straight from `addr`."""
        if rows * cols == 0 or addr == 0:
            # an empty operand keeps an owned spare word
            return self.zeros(rows, cols)
        var out = self.add(-1, rows, cols)
        self.matrices[out].borrowed = addr
        return out

    def device_output(mut self, rows: Int, cols: Int) raises -> Int:
        """`_Kit._dout`."""
        var handle = self.dcall(XD_DEV_ALLOC, [max(rows * cols, 1)])
        self.allocations.append(handle)
        return self.add(-1, rows, cols, handle)

    def addr(mut self, a: Int) raises -> Int:
        """`_M.s`: the host address of matrix a's words, downloading a device
        matrix once (it moves)."""
        if self.matrices[a].borrowed != 0 and self.matrices[a].device < 0:
            return self.matrices[a].borrowed
        if self.matrices[a].store < 0:
            var count = self.n(a)
            var store = len(self.stores)
            self.stores.append(KitStore(count))
            if count:
                _ = self.hcall(XD_DEV_DOWNLOAD, [Int(self.stores[store].words.unsafe_ptr())],
                               [self.matrices[a].device, count])
            self.stores[store].references = 1
            self.matrices[a].store = store
            self.matrices[a].device = -1
        return Int(self.stores[self.matrices[a].store].words.unsafe_ptr())

    def ptr(mut self, a: Int) raises -> KIT_F32P:
        return KIT_F32P(unsafe_from_address=self.addr(a))

    def device(mut self, a: Int) raises -> Int:
        """`_Kit._did`: the device id, uploading a host matrix (it moves)."""
        if self.matrices[a].device < 0:
            var count = self.n(a)
            var handle = self.dcall(XD_DEV_ALLOC, [max(count, 1)])
            self.allocations.append(handle)
            if count:
                # straight from the admitted buffer (row N293) or the store
                var source = self.addr(a)
                _ = self.hcall(XD_DEV_UPLOAD, [source], [handle, count])
            self.matrices[a].borrowed = 0
            self._release_store(a)
            self.matrices[a].device = handle
        return self.matrices[a].device

    def constant(mut self, value: Float32, rows: Int = 1, cols: Int = 1) raises -> Int:
        """`_M.of([value] * (rows * cols), rows, cols)` / `_Kit.const`."""
        var out = self.zeros(rows, cols)
        tsqr_fill_constant(self.ptr(out), rows * cols, value)
        return out

    def reshape(mut self, a: Int, rows: Int, cols: Int) -> Int:
        """`_M.reshape`: the same words (a view of the same store or buffer)."""
        var out = self.add(self.matrices[a].store, rows, cols, self.matrices[a].device)
        self.matrices[out].borrowed = self.matrices[a].borrowed
        return out

    def _mode(self, a: Int, x: Int) -> Int:
        if self.r(x) == self.r(a) and self.c(x) == self.c(a):
            return 0
        if self.r(x) == 1 and self.c(x) == self.c(a):
            return 1
        if self.c(x) == 1 and self.r(x) == self.r(a):
            return 2
        return 3

    def ew(mut self, op: Int, a: Int, b: Int = -1, c: Int = -1, scalar: Float64 = 0.0) raises -> Int:
        """`_Kit.ew(op, A, B, C, s)`."""
        # one shared zero per kit for the absent operands (row N294)
        var zero = self.shared_zero() if (b < 0 or c < 0) else -1
        var ib = zero if b < 0 else b
        var ic = zero if c < 0 else c
        var bm = 3 if b < 0 else self._mode(a, ib)
        var cm = 3 if c < 0 else self._mode(a, ic)
        if b >= 0 and bm == 3 and not (self.r(ib) == 1 and self.c(ib) == 1):
            raise Error("x_decomp: cannot broadcast " + String(self.r(ib)) + "x" + String(self.c(ib)) + " against "
                        + String(self.r(a)) + "x" + String(self.c(a)))
        if c >= 0 and cm == 3 and not (self.r(ic) == 1 and self.c(ic) == 1):
            raise Error("x_decomp: cannot broadcast " + String(self.r(ic)) + "x" + String(self.c(ic)) + " against "
                        + String(self.r(a)) + "x" + String(self.c(a)))
        var rows = self.r(a)
        var cols = self.c(a)
        var words: List[Int] = [op, rows * cols, cols, self.n(ib), bm, self.n(ic), cm]
        if rows * cols and self.resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var db = self.device(ib)
            var dc = self.device(ic)
            _ = self.dcall(XD_DEV_EW, [da, db, dc, self.matrices[out].device, words[0], words[1], words[2], words[3],
                                       words[4], words[5], words[6]], scalar)
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var pa = self.addr(a)
            var pb = self.addr(ib)
            var pc = self.addr(ic)
            var po = self.addr(out)
            # every store sits in self.stores, alive through the call
            _ = self.hcall(XD_EW, [pa, pb, pc, po], words, scalar)
        return out

    def mm(mut self, a: Int, b: Int, ta: Bool = False, tb: Bool = False) raises -> Int:
        """`_Kit.mm(A, B, ta, tb)`."""
        var m = self.c(a) if ta else self.r(a)
        var k = self.r(a) if ta else self.c(a)
        var k2 = self.c(b) if tb else self.r(b)
        var n = self.r(b) if tb else self.c(b)
        if k != k2:
            raise Error("x_decomp: gemm inner dimensions " + String(k) + " and " + String(k2) + " differ")
        var words: List[Int] = [m, k, n, Int(ta), Int(tb)]
        if m * n and m * k and k * n and self.resident:
            var out = self.device_output(m, n)
            var da = self.device(a)
            var db = self.device(b)
            _ = self.dcall(XD_DEV_GEMM, [da, db, self.matrices[out].device, m, k, n, Int(ta), Int(tb)])
            return out
        var out = self.zeros(m, n)
        if m * n:
            var pa = self.addr(a)
            var pb = self.addr(b)
            var po = self.addr(out)
            _ = self.hcall(XD_GEMM, [pa, pb, po], words)
        return out

    def _sum(mut self, a: Int, columns: Bool) raises -> Int:
        """`_Kit.colsum` / `rowsum`."""
        var rows = self.r(a)
        var cols = self.c(a)
        var nr = 1 if columns else rows
        var nc = cols if columns else 1
        if rows * cols and self.resident:
            var out = self.device_output(nr, nc)
            var da = self.device(a)
            _ = self.dcall(XD_DEV_COLSUM if columns else XD_DEV_ROWSUM, [da, self.matrices[out].device, rows, cols])
            return out
        var out = self.zeros(nr, nc)
        var pa = self.addr(a)
        var po = self.addr(out)
        _ = self.hcall(XD_COLSUM if columns else XD_ROWSUM, [pa, po], [rows, cols])
        return out

    def total_word(mut self, a: Int) raises -> Float32:
        """`_Kit.total(A).s[0]`: rows ascending, then the row sums ascending."""
        var rows = self._sum(a, False)
        var t = self._sum(rows, True)
        return self.ptr(t)[0]

    def move(mut self, a: Int, idx: Int, rows: Int, cols: Int, words: List[Int], resident: Bool) raises -> Int:
        """A move into a fresh rows x cols output; words = [op, count, a1,
        a2, a3, ist, ioff, nsrc, ndst, nidx]."""
        if rows * cols and resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var di = self.device(idx)
            var ints: List[Int] = [da, di, self.matrices[out].device]
            for i in range(len(words)):  # driver-loop: native Mojo values, no CPython call: (params: the ten move words): 10
                ints.append(words[i])
            _ = self.dcall(XD_DEV_MOVE, ints)
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var pa = self.addr(a)
            var pi = self.addr(idx)
            var po = self.addr(out)
            _ = self.hcall(XD_MOVE, [pa, pi, po], words)
        return out

    def take_rows(mut self, a: Int, idx: Int) raises -> Int:
        """`_M.take_rows(idx)` (the kit's TAKE_ROWS move; idx exact floats)."""
        if self.r(a) >= F32_INDEX_MAX:
            raise Error("x_decomp: take_rows exceeds the float32 row-index bound (2^24 rows)")
        var count = self.n(idx)
        var cols = self.c(a)
        return self.move(a, idx, count, cols, [MV_TAKE_ROWS, count * cols, cols, 0, 0, 1, 0, self.n(a), count * cols,
                                               count], self.resident)

    def take_cols(mut self, a: Int, start: Int, stop: Int) raises -> Int:
        """`_M.take_cols(range(start, stop))`: gathered on the device when
        the matrix is resident there, else one host copy."""
        var rows = self.r(a)
        var cols = self.c(a)
        var width = stop - start
        if self.on_device(a) and width and rows * width > 0 and rows < F32_INDEX_MAX and cols < F32_INDEX_MAX:
            var idx = self.zeros(1, width)
            var p = self.ptr(idx)
            for t in range(width):  # driver-loop: over the selected column indices (matrix columns)
                p[t] = Float32(start + t)
            return self.move(a, idx, rows, width, [MV_TAKE_COLS, rows * width, width, cols, 0, 0, 0, rows * cols,
                                                   rows * width, width], True)
        var out = self.zeros(rows, width)
        if rows * width:
            tsqr_copy_columns(self.ptr(a), self.ptr(out), rows, cols, start, width)
        return out

    def transpose(mut self, a: Int) raises -> Int:
        """`_M.T`."""
        var rows = self.r(a)
        var cols = self.c(a)
        var count = rows * cols
        if rows == 1 or cols == 1:
            return self.reshape(a, cols, rows)
        var words: List[Int] = [MV_TRANSPOSE, count, rows, cols, 0, 0, 0, count, count, 1]
        if count and count < I32_MAX and self.resident:
            # the index operand is unused: the source's own id stands in
            var da = self.device(a)
            var out = self.device_output(cols, rows)
            var ints: List[Int] = [da, da, self.matrices[out].device]
            for i in range(len(words)):  # driver-loop: native Mojo values, no CPython call: (params: the ten move words): 10
                ints.append(words[i])
            _ = self.dcall(XD_DEV_MOVE, ints)
            return out
        var out = self.zeros(cols, rows)
        if count:
            var one = self.shared_zero()
            var pa = self.addr(a)
            var pi = self.addr(one)
            var po = self.addr(out)
            _ = self.hcall(XD_MOVE, [pa, pi, po], words)
        return out

    def strided(mut self, a: Int, count: Int, stride: Int, offset: Int = 0) raises -> Int:
        """`_Kit.strided(A, count, stride, off)`."""
        var actual = offset + 1 if count == 1 else stride
        var idx = self.constant(Float32(offset))
        return self.move(a, idx, 1, count, [MV_TAKE_COLS, count, 1, actual, 0, 0, 0, self.n(a), count, 1],
                         self.resident)

    def word(mut self, a: Int) raises -> Float32:
        """`_Kit.word(A)`: the first value, one word down."""
        var index = a
        if self.on_device(a) and self.n(a) > 1:
            index = self.strided(a, 1, 1)
        return self.ptr(index)[0]

    def order(mut self, a: Int) raises -> Int:
        """`_Kit.order(A)`: the stable ascending order (ties to the lower
        index) as an n x 1 matrix of exact floats; NaN refused."""
        var n = self.n(a)
        if n and self.resident:
            if n >= F32_INDEX_MAX:
                raise Error("x_decomp: order_f exceeds the float32 index bound")
            # `_refuse_nan`: the max is NaN when any value is
            var maximum = self.device_output(1, 1)
            var da = self.device(a)
            _ = self.dcall(XD_DEV_REDUCE, [da, self.matrices[maximum].device, n, 1])
            var v = self.word(maximum)
            if v != v:
                raise Error("x_decomp: a NaN has no order (refused)")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            _ = self.dcall(XD_DEV_ORDER, [da, da, self.matrices[out].device, self.matrices[count].device, n, 0, 0])
            return out
        var out = self.zeros(n, 1)
        if n:
            var pa = self.addr(a)
            var po = self.addr(out)
            _ = self.hcall(XD_ORDER, [pa, po], [n])
        return out

    def order_small(mut self, a: Int) raises -> Int:
        """`_Kit.order_small(A)` (ties to the lower index, NaN last)."""
        var n = self.n(a)
        if n and self.resident:
            var out = self.device_output(n, 1)
            var da = self.device(a)
            _ = self.dcall(XD_DEV_ORDER_SMALL, [da, self.matrices[out].device, n])
            return out
        var out = self.zeros(n, 1)
        if n:
            var pa = self.addr(a)
            var po = self.addr(out)
            _ = self.hcall(XD_ORDER_SMALL, [pa, po], [n])
        return out

    def take_cols_m(mut self, a: Int, idx: Int, w: Int) raises -> Int:
        """`_Kit.take_cols_m(A, idx, w)`."""
        var rows = self.r(a)
        var cols = self.c(a)
        return self.move(a, idx, rows, w, [MV_TAKE_COLS, rows * w, w, cols, 0, 0, 0, rows * cols, rows * w, w],
                         self.resident)

    def hstack2(mut self, a: Int, b: Int) raises -> Int:
        """`_Kit.hstack([a, b])` (PLACE_COLS moves; `_hstack` on a host kit)."""
        var rows = self.r(a)
        var w = self.c(a) + self.c(b)
        var resident = self.resident and rows * w > 0
        var out = self.device_output(rows, w) if resident else self.zeros(rows, w)
        var offset = 0
        for part in range(2):  # driver-loop: over the two stacked matrices
            var source = a if part == 0 else b
            var count = self.n(source)
            if count:
                var words: List[Int] = [MV_PLACE_COLS, count, self.c(source), w, offset, 0, 0, count, rows * w, 1]
                var one = self.shared_zero()
                if resident:
                    var ds = self.device(source)
                    var di = self.device(one)
                    var ints: List[Int] = [ds, di, self.matrices[out].device]
                    for i in range(len(words)):  # driver-loop: native Mojo values, no CPython call: (params: the ten move words): 10
                        ints.append(words[i])
                    _ = self.dcall(XD_DEV_MOVE, ints)
                else:
                    var ps = self.addr(source)
                    var pi = self.addr(one)
                    var po = self.addr(out)
                    _ = self.hcall(XD_MOVE, [ps, pi, po], words)
            offset += self.c(source)
        return out

    def neg_signs(mut self, v: Int) raises -> Int:
        """`_Kit.neg_signs(V)`: -1 where V < 0, else +1 (NaN: +1)."""
        var negated = self.ew(OP_SCALE, v, scalar=-1.0)
        # the two constants, made once per kit (row N294); read-only operands
        if self.minus_one < 0:
            self.minus_one = self.constant(-1.0)
            self.plus_one = self.constant(1.0)
        return self.ew(OP_SELECT, negated, self.minus_one, self.plus_one, 0.0)

    def copy(mut self, a: Int) raises -> Int:
        """`_M.copy()`: a host copy (a device matrix is downloaded first)."""
        var rows = self.r(a)
        var cols = self.c(a)
        var out = self.zeros(rows, cols)
        if rows * cols:
            unsafe_memcpy(dest=self.ptr(out), src=self.ptr(a), count=rows * cols)
        return out

    def geqrf(mut self, a: Int, mut tau: Int) raises -> Int:
        """`_Kit.geqrf(A)`: (h, tau), LAPACK geqrf's factored form."""
        var h = self.copy(a)
        var kk = min(self.r(a), self.c(a))
        tau = self.zeros(1, kk)
        var ph = self.addr(h)
        var pt = self.addr(tau)
        _ = self.hcall(XD_GEQRF, [ph, pt], [self.r(a), self.c(a)])
        return h

    def orgqr(mut self, h: Int, tau: Int, qc: Int) raises -> Int:
        """`_Kit.orgqr(h, tau, qc)`."""
        var q = self.zeros(self.r(h), qc)
        var ph = self.addr(h)
        var pt = self.addr(tau)
        var pq = self.addr(q)
        _ = self.hcall(XD_ORGQR, [ph, pt, pq], [self.r(h), self.c(h), self.c(tau), qc])
        return q

    def eigh(mut self, a: Int, uplo: Int, mut v: Int) raises -> Int:
        """`_Kit.eigh(A, uplo)`: (w 1 x n ascending, v n x n columns)."""
        var n = self.r(a)
        var w = self.zeros(1, n)
        v = self.zeros(n, n)
        # TOMBSTONE: the device-resident x_decomp_dev_eigh branch (dead code: both callers,
        # linalg_ops_decomp `_lin_eigh` and `svd(hermitian=True)`, pass a freshly borrowed host
        # matrix, so `on_device(a)` was never true) deleted 2026-11-09 by f2b6b13bd3; code recoverable
        # at f2b6b13bd3^.
        # Restore: git apply experiments/removed/dead_code_p5_20261109.patch
        var pa = self.addr(a)
        var pw = self.addr(w)
        var pv = self.addr(v)
        _ = self.hcall(XD_EIGH, [pa, pw, pv], [n, uplo])
        return w

    def svd(mut self, a: Int, mut vt: Int) raises -> Int:
        """`_Kit.svd(A)`: (S 1 x n descending, Vt n x n) of a tall A."""
        var m = self.r(a)
        var n = self.c(a)
        var s = self.zeros(1, n)
        var v = self.zeros(n, n)
        if m >= n and n >= 1 and self.dev_svd:
            var da = self.device(a)
            var ps = self.addr(s)
            var pv = self.addr(v)
            _ = self.call(XD_DEV_SVD, [ps, pv], [da, m, n], List[Float64]())
        else:
            var pa = self.addr(a)
            var ps = self.addr(s)
            var pv = self.addr(v)
            _ = self.hcall(XD_SVD, [pa, ps, pv], [m, n])
        # descending, ties to the lower index: the stable ascending order of -s
        var negated = self.ew(OP_SCALE, s, scalar=-1.0)
        var o = self.order(negated)
        var st = self.transpose(s)
        var taken = self.take_rows(st, o)
        var sorted_s = self.transpose(taken)
        var vtt = self.transpose(v)
        vt = self.take_rows(vtt, o)
        return sorted_s

    def orth_diag(mut self, a: Int, mut diag: Int) raises -> Int:
        """`_Kit.orth_diag(A)`: (Q, the passes' R-diagonal products)."""
        var rows = self.r(a)
        var cols = self.c(a)
        diag = self.zeros(1, cols)
        if rows * cols and self.resident:
            var q = self.device_output(rows, cols)
            var da = self.device(a)
            var pd = self.addr(diag)
            _ = self.call(XD_DEV_ORTH_DIAG, [pd], [da, self.matrices[q].device, rows, cols], List[Float64]())
            return q
        var q = self.copy(a)
        var pq = self.addr(q)
        var pd = self.addr(diag)
        _ = self.hcall(XD_ORTH_DIAG, [pq, pd], [rows, cols])
        return q

    def rank_above(mut self, s: Int, n: Int, rtol: Float64) raises -> Int:
        """How many of the n singular values at s are > 0 and > s[0] * rtol
        (x_decomp/api.mojo `rank_above_run`)."""
        var ps = self.addr(s)
        return self.call(XD_RANK, [ps], [n], [rtol])

    def eye(mut self, rows: Int, cols: Int) raises -> Int:
        var out = self.zeros(rows, cols)
        tsqr_fill_eye(self.ptr(out), rows, cols)
        return out


def lin_svd_tall_run(mut kit: LinKit, a: Int, full: Bool, rtol: Float64, mut s_out: Int, mut vt_out: Int) raises -> Int:
    """`_svd_tall(k, A, full)`: (U, S, Vt) of a tall A; returns U. `rtol` is
    the module's `_SVD_NULL_RTOL_MODULE`, read by the door's UNPACK."""
    var m = kit.r(a)
    var n = kit.c(a)
    var vt = 0
    var s = kit.svd(a, vt)
    s_out = s
    vt_out = vt
    var r = 0
    if n:
        # the numerical rank, counted in Mojo (x_decomp/api.mojo rank_above_run)
        r = kit.rank_above(s, n, rtol)
    var av = kit.mm(a, vt, tb=True)
    var num = kit.take_cols(av, 0, r) if r < n else av
    var den = kit.take_cols(s, 0, r) if r < n else s
    var ug = kit.ew(OP_DIV, num, den)
    var width = m if full else n
    if r == n and width == n:
        var d = 0
        var qo = kit.orth_diag(ug, d)
        var zero = kit.zeros(1, 1)
        var magnitude = kit.ew(OP_ABS, d)
        var guarded = kit.ew(OP_LE, magnitude, zero)
        if kit.total_word(guarded) == 0:
            var column_signs = kit.neg_signs(d)
            return kit.ew(OP_MUL, qo, column_signs)
    if r == 0:
        return kit.eye(m, width)
    var tau = 0
    var h = kit.geqrf(ug, tau)
    var qc = kit.orgqr(h, tau, width)
    var diagonal = kit.strided(h, r, r + 1, 0)
    var sg = kit.neg_signs(diagonal)
    if width > r:
        var trailing = kit.constant(1.0, 1, width - r)
        sg = kit.hstack2(sg, trailing)
    return kit.ew(OP_MUL, qc, sg)


def lin_svd_hermitian_run(mut kit: LinKit, m: Int, rows: Int, mut s_out: Int, mut vt_out: Int) raises -> Int:
    """`svd(a, hermitian=True)` of the square matrix m: returns U; S and Vt
    through the out words."""
    var v = 0
    var w = kit.eigh(m, 1, v)
    # numpy: argsort(|w|) (ascending) reversed, so equal magnitudes come
    # higher index first: the stable ascending order of -|w| read back
    # to front, on the kit
    var rev = kit.zeros(1, rows)
    var rp = kit.ptr(rev)
    for t in range(rows):  # driver-loop: over the index table n-1 .. 0 (matrix columns)
        rp[t] = Float32(rows - 1 - t)
    var wr = kit.take_cols_m(w, rev, rows)
    var wr_abs = kit.ew(OP_ABS, wr)
    var keys = kit.ew(OP_SCALE, wr_abs, scalar=-1.0)
    var o = kit.order_small(keys)
    var o_row = kit.reshape(o, 1, rows)
    var o_neg = kit.ew(OP_SCALE, o_row, scalar=-1.0)
    var order = kit.ew(OP_ADDS, o_neg, scalar=Float64(rows - 1))
    var w_abs = kit.ew(OP_ABS, w)
    s_out = kit.take_cols_m(w_abs, order, rows)
    var u = kit.take_cols_m(v, order, rows)
    var ut = kit.transpose(u)
    var w_ordered = kit.take_cols_m(w, order, rows)
    var w_signs = kit.neg_signs(w_ordered)
    var signs = kit.reshape(w_signs, rows, 1)
    vt_out = kit.ew(OP_MUL, ut, signs)
    return u
