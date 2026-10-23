# SPDX-License-Identifier: Apache-2.0
# The x_decomp cells the linalg decompositions (qr, eigh, svdvals, svd) call,
# driven natively: python/mojolearn/_expansion_decomp.py's `_M` / `_Kit`
# semantics for the operations `_linalg_impl` used, entry for entry. A matrix
# lives on the host (a float32 Array) or on the device (a resident buffer of a
# GPU binding, `x_decomp/resident.mojo`); a host matrix handed to a resident
# entry is uploaded once and MOVES to the device, a device matrix read on the
# host is downloaded once and moves back, exactly as `_M.s` / `_Kit._did` moved
# them. The same entries on the same words in the same order: the bits do not
# move. Every entry is called through `family_call`, so a cell's own refusal
# (an unconverged Jacobi, a NaN order) keeps its Python class and words.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.classical_arrays import classical_address
from bindings.family_common import family_call
from bindings.linalg_ops_buffers import lin_array, lin_truth
from core.classical_tsqr_impl import tsqr_fill_constant, tsqr_copy_columns, tsqr_fill_eye

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


struct KitMatrix(Copyable, Movable):
    var host: PythonObject
    var device: Int
    var rows: Int
    var cols: Int

    def __init__(out self, host: PythonObject, device: Int, rows: Int, cols: Int):
        self.host = host
        self.device = device
        self.rows = rows
        self.cols = cols


struct LinKit(Movable):
    """`_Kit(mode)` over the x_decomp binding of the process's default mode."""
    var b: PythonObject
    var context: PythonObject
    var matrices: List[KitMatrix]
    var allocations: List[Int]
    var resident: Bool

    def __init__(out self, binding: PythonObject, context: PythonObject) raises:
        var bi = Python.import_module("builtins")
        # `_Kit._raw`: through a profiler proxy's `_b`, never its __getattr__
        var raw = binding
        try:
            raw = py_call(bi.object.__getattribute__, binding, "_b")
        except:
            raw = binding
        self.b = raw
        self.context = context
        self.matrices = List[KitMatrix]()
        self.allocations = List[Int]()
        # `_Kit._res`: the binding has the resident entries (a GPU binding)
        self.resident = lin_truth(py_call(bi.hasattr, raw, "x_decomp_dev_ew"))

    def entry(self, name: String) raises -> PythonObject:
        return family_call(Python.import_module("builtins").getattr, self.b, PythonObject(name))

    def has(self, name: String) raises -> Bool:
        """`_Kit._opt_dev(name)`."""
        return self.resident and lin_truth(py_call(Python.import_module("builtins").hasattr, self.b, name))

    def close(mut self):
        """Return every device buffer this call allocated to the pool."""
        for i in range(len(self.allocations) - 1, -1, -1):  # driver-loop: over native matrix allocation handles
            try:
                _ = family_call(self.entry("x_decomp_dev_free"), PythonObject(self.allocations[i]))
            except:
                pass
        self.allocations.clear()

    def add(mut self, host: PythonObject, rows: Int, cols: Int, device: Int = -1) -> Int:
        var result = len(self.matrices)
        self.matrices.append(KitMatrix(host, device, rows, cols))
        return result

    def n(self, a: Int) -> Int:
        return self.matrices[a].rows * self.matrices[a].cols

    def r(self, a: Int) -> Int:
        return self.matrices[a].rows

    def c(self, a: Int) -> Int:
        return self.matrices[a].cols

    def on_device(self, a: Int) -> Bool:
        return self.matrices[a].device >= 0

    def zeros(mut self, rows: Int, cols: Int) raises -> Int:
        """`_M.zeros(rows, cols)` (one spare word keeps an empty matrix's
        address valid, as `_M._one` did)."""
        var host = lin_array([max(rows * cols, 1)], "<f4", self.context)
        return self.add(host, rows, cols)

    def of_buffer(mut self, addr: Int, rows: Int, cols: Int) raises -> Int:
        """`_xd_matrix`: a host copy of a C-order float32 buffer."""
        var out = self.zeros(rows, cols)
        if rows * cols:
            memcpy(dest=self.ptr(out), src=KIT_F32P(unsafe_from_address=addr), count=rows * cols)
        return out

    def device_output(mut self, rows: Int, cols: Int) raises -> Int:
        """`_Kit._dout`."""
        var handle = Int(py=family_call(self.entry("x_decomp_dev_alloc"), PythonObject(max(rows * cols, 1))))
        self.allocations.append(handle)
        return self.add(PythonObject(None), rows, cols, handle)

    def host(mut self, a: Int) raises -> PythonObject:
        """`_M.s`: the host store, downloading a device matrix once (it moves)."""
        if self.matrices[a].host is None:
            var count = self.n(a)
            var result = lin_array([max(count, 1)], "<f4", self.context)
            if count:
                _ = family_call(self.entry("x_decomp_dev_download"), PythonObject(self.matrices[a].device),
                                classical_address(result), PythonObject(count))
            self.matrices[a].host = result
            self.matrices[a].device = -1
        return self.matrices[a].host

    def ptr(mut self, a: Int) raises -> KIT_F32P:
        var owner = self.host(a)
        var p = KIT_F32P(unsafe_from_address=Int(py=classical_address(owner)))
        _ = owner^
        return p

    def address(mut self, a: Int) raises -> PythonObject:
        var owner = self.host(a)
        var result = classical_address(owner)
        _ = owner^
        return result

    def device(mut self, a: Int) raises -> Int:
        """`_Kit._did`: the device id, uploading a host matrix (it moves)."""
        if self.matrices[a].device < 0:
            var count = self.n(a)
            var owner = self.host(a)
            var handle = Int(py=family_call(self.entry("x_decomp_dev_alloc"), PythonObject(max(count, 1))))
            self.allocations.append(handle)
            if count:
                _ = family_call(self.entry("x_decomp_dev_upload"), PythonObject(handle), classical_address(owner),
                                PythonObject(count))
            self.matrices[a].device = handle
            self.matrices[a].host = PythonObject(None)
            _ = owner^
        return self.matrices[a].device

    def out(mut self, a: Int) raises -> PythonObject:
        """`_M.out()`: a public float32 Array (C order) of the matrix."""
        var rows = self.r(a)
        var cols = self.c(a)
        var result = lin_array([rows, cols], "<f4", self.context)
        if rows * cols:
            memcpy(dest=KIT_F32P(unsafe_from_address=Int(py=classical_address(result))), src=self.ptr(a),
                   count=rows * cols)
        return result

    def out_vector(mut self, a: Int, length: Int) raises -> PythonObject:
        """`_M.out((length,))`."""
        var result = lin_array([length], "<f4", self.context)
        if length:
            memcpy(dest=KIT_F32P(unsafe_from_address=Int(py=classical_address(result))), src=self.ptr(a),
                   count=length)
        return result

    def constant(mut self, value: Float32, rows: Int = 1, cols: Int = 1) raises -> Int:
        """`_M.of([value] * (rows * cols), rows, cols)` / `_Kit.const`."""
        var out = self.zeros(rows, cols)
        tsqr_fill_constant(self.ptr(out), rows * cols, value)
        return out

    def reshape(mut self, a: Int, rows: Int, cols: Int) raises -> Int:
        """`_M.reshape`: the same words (a view of the same store or buffer)."""
        var host = self.matrices[a].host
        var device = self.matrices[a].device
        return self.add(host, rows, cols, device)

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
        var zero = self.zeros(1, 1)
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
        var params = Python.list(PythonObject(op), PythonObject(rows * cols), PythonObject(cols),
                                 PythonObject(self.n(ib)), PythonObject(bm), PythonObject(self.n(ic)), PythonObject(cm))
        if rows * cols and self.resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var db = self.device(ib)
            var dc = self.device(ic)
            _ = family_call(self.entry("x_decomp_dev_ew"), PythonObject(da), PythonObject(db), PythonObject(dc),
                            PythonObject(self.matrices[out].device), params, PythonObject(scalar))
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            _ = family_call(self.entry("x_decomp_ew"), self.address(a), self.address(ib), self.address(ic),
                            self.address(out), params, PythonObject(scalar))
        return out

    def mm(mut self, a: Int, b: Int, ta: Bool = False, tb: Bool = False) raises -> Int:
        """`_Kit.mm(A, B, ta, tb)`."""
        var m = self.c(a) if ta else self.r(a)
        var k = self.r(a) if ta else self.c(a)
        var k2 = self.c(b) if tb else self.r(b)
        var n = self.r(b) if tb else self.c(b)
        if k != k2:
            raise Error("x_decomp: gemm inner dimensions " + String(k) + " and " + String(k2) + " differ")
        var params = Python.list(PythonObject(m), PythonObject(k), PythonObject(n), PythonObject(Int(ta)), PythonObject(Int(tb)))
        if m * n and m * k and k * n and self.resident:
            var out = self.device_output(m, n)
            var da = self.device(a)
            var db = self.device(b)
            _ = family_call(self.entry("x_decomp_dev_gemm"), PythonObject(da), PythonObject(db),
                            PythonObject(self.matrices[out].device), params)
            return out
        var out = self.zeros(m, n)
        if m * n:
            _ = family_call(self.entry("x_decomp_gemm"), self.address(a), self.address(b), self.address(out), params)
        return out

    def _sum(mut self, a: Int, columns: Bool) raises -> Int:
        """`_Kit.colsum` / `rowsum`."""
        var rows = self.r(a)
        var cols = self.c(a)
        var nr = 1 if columns else rows
        var nc = cols if columns else 1
        var name = String("colsum") if columns else String("rowsum")
        var params = Python.list(PythonObject(rows), PythonObject(cols))
        if rows * cols and self.resident:
            var out = self.device_output(nr, nc)
            var da = self.device(a)
            _ = family_call(self.entry("x_decomp_dev_" + name), PythonObject(da), PythonObject(self.matrices[out].device), params)
            return out
        var out = self.zeros(nr, nc)
        _ = family_call(self.entry("x_decomp_" + name), self.address(a), self.address(out), params)
        return out

    def total_word(mut self, a: Int) raises -> Float32:
        """`_Kit.total(A).s[0]`: rows ascending, then the row sums ascending."""
        var rows = self._sum(a, False)
        var t = self._sum(rows, True)
        return self.ptr(t)[0]

    def move(mut self, a: Int, idx: Int, rows: Int, cols: Int, params: PythonObject, resident: Bool) raises -> Int:
        if rows * cols and resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var di = self.device(idx)
            _ = family_call(self.entry("x_decomp_dev_move"), PythonObject(da), PythonObject(di),
                            PythonObject(self.matrices[out].device), params)
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            _ = family_call(self.entry("x_decomp_move"), self.address(a), self.address(idx), self.address(out), params)
        return out

    def take_rows(mut self, a: Int, idx: Int) raises -> Int:
        """`_M.take_rows(idx)` (the kit's TAKE_ROWS move; idx exact floats)."""
        if self.r(a) >= F32_INDEX_MAX:
            raise Error("x_decomp: take_rows exceeds the float32 row-index bound (2^24 rows)")
        var count = self.n(idx)
        var cols = self.c(a)
        return self.move(a, idx, count, cols, Python.list(PythonObject(MV_TAKE_ROWS), PythonObject(count * cols),
            PythonObject(cols), PythonObject(0), PythonObject(0), PythonObject(1), PythonObject(0),
            PythonObject(self.n(a)), PythonObject(count * cols), PythonObject(count)), self.resident)

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
            return self.move(a, idx, rows, width, Python.list(PythonObject(MV_TAKE_COLS), PythonObject(rows * width),
                PythonObject(width), PythonObject(cols), PythonObject(0), PythonObject(0), PythonObject(0),
                PythonObject(rows * cols), PythonObject(rows * width), PythonObject(width)), True)
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
            var host = self.matrices[a].host
            var device = self.matrices[a].device
            return self.add(host, cols, rows, device)
        var one = self.zeros(1, 1)
        var params = Python.list(PythonObject(MV_TRANSPOSE), PythonObject(count), PythonObject(rows), PythonObject(cols),
            PythonObject(0), PythonObject(0), PythonObject(0), PythonObject(count), PythonObject(count), PythonObject(1))
        if count and count < I32_MAX and self.resident:
            # the index operand is unused: the source's own id stands in
            var da = self.device(a)
            var out = self.device_output(cols, rows)
            _ = family_call(self.entry("x_decomp_dev_move"), PythonObject(da), PythonObject(da),
                            PythonObject(self.matrices[out].device), params)
            return out
        var out = self.zeros(cols, rows)
        if count:
            _ = family_call(self.entry("x_decomp_move"), self.address(a), self.address(one), self.address(out), params)
        return out

    def strided(mut self, a: Int, count: Int, stride: Int, offset: Int = 0) raises -> Int:
        """`_Kit.strided(A, count, stride, off)`."""
        var actual = offset + 1 if count == 1 else stride
        var idx = self.constant(Float32(offset))
        return self.move(a, idx, 1, count, Python.list(PythonObject(MV_TAKE_COLS), PythonObject(count), PythonObject(1),
            PythonObject(actual), PythonObject(0), PythonObject(0), PythonObject(0), PythonObject(self.n(a)),
            PythonObject(count), PythonObject(1)), self.resident)

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
            _ = family_call(self.entry("x_decomp_dev_reduce"), PythonObject(da), PythonObject(self.matrices[maximum].device),
                            Python.list(PythonObject(n), PythonObject(1)))
            var v = self.word(maximum)
            if v != v:
                raise Error("x_decomp: a NaN has no order (refused)")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            _ = family_call(self.entry("x_decomp_dev_order_f"), PythonObject(da), PythonObject(da),
                            PythonObject(self.matrices[out].device), PythonObject(self.matrices[count].device),
                            Python.list(PythonObject(n), PythonObject(0), PythonObject(0)))
            return out
        var out = self.zeros(n, 1)
        if n:
            _ = family_call(self.entry("x_decomp_order_f"), self.address(a), PythonObject(n), self.address(out))
        return out

    def order_small(mut self, a: Int) raises -> Int:
        """`_Kit.order_small(A)` (ties to the lower index, NaN last)."""
        var n = self.n(a)
        if n and self.resident:
            var out = self.device_output(n, 1)
            var da = self.device(a)
            _ = family_call(self.entry("x_decomp_dev_order_small"), PythonObject(da),
                            PythonObject(self.matrices[out].device), Python.list(PythonObject(n)))
            return out
        var out = self.zeros(n, 1)
        if n:
            _ = family_call(self.entry("x_decomp_order_small"), self.address(a), self.address(out), Python.list(PythonObject(n)))
        return out

    def take_cols_m(mut self, a: Int, idx: Int, w: Int) raises -> Int:
        """`_Kit.take_cols_m(A, idx, w)`."""
        var rows = self.r(a)
        var cols = self.c(a)
        return self.move(a, idx, rows, w, Python.list(PythonObject(MV_TAKE_COLS), PythonObject(rows * w), PythonObject(w),
            PythonObject(cols), PythonObject(0), PythonObject(0), PythonObject(0), PythonObject(rows * cols),
            PythonObject(rows * w), PythonObject(w)), self.resident)

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
                var params = Python.list(PythonObject(MV_PLACE_COLS), PythonObject(count), PythonObject(self.c(source)),
                    PythonObject(w), PythonObject(offset), PythonObject(0), PythonObject(0), PythonObject(count),
                    PythonObject(rows * w), PythonObject(1))
                var one = self.zeros(1, 1)
                if resident:
                    var ds = self.device(source)
                    var di = self.device(one)
                    _ = family_call(self.entry("x_decomp_dev_move"), PythonObject(ds), PythonObject(di),
                                    PythonObject(self.matrices[out].device), params)
                else:
                    _ = family_call(self.entry("x_decomp_move"), self.address(source), self.address(one),
                                    self.address(out), params)
            offset += self.c(source)
        return out

    def neg_signs(mut self, v: Int) raises -> Int:
        """`_Kit.neg_signs(V)`: -1 where V < 0, else +1 (NaN: +1)."""
        var negated = self.ew(OP_SCALE, v, scalar=-1.0)
        var negative = self.constant(-1.0)
        var positive = self.constant(1.0)
        return self.ew(OP_SELECT, negated, negative, positive, 0.0)

    def copy(mut self, a: Int) raises -> Int:
        """`_M.copy()`: a host copy (a device matrix is downloaded first)."""
        var rows = self.r(a)
        var cols = self.c(a)
        var out = self.zeros(rows, cols)
        if rows * cols:
            memcpy(dest=self.ptr(out), src=self.ptr(a), count=rows * cols)
        return out

    def geqrf(mut self, a: Int, mut tau: Int) raises -> Int:
        """`_Kit.geqrf(A)`: (h, tau), LAPACK geqrf's factored form."""
        var h = self.copy(a)
        var kk = min(self.r(a), self.c(a))
        tau = self.zeros(1, kk)
        _ = family_call(self.entry("x_decomp_geqrf"), self.address(h), self.address(tau),
                        Python.list(PythonObject(self.r(a)), PythonObject(self.c(a))))
        return h

    def orgqr(mut self, h: Int, tau: Int, qc: Int) raises -> Int:
        """`_Kit.orgqr(h, tau, qc)`."""
        var q = self.zeros(self.r(h), qc)
        _ = family_call(self.entry("x_decomp_orgqr"), self.address(h), self.address(tau), self.address(q),
                        Python.list(PythonObject(self.r(h)), PythonObject(self.c(h)), PythonObject(self.c(tau)), PythonObject(qc)))
        return q

    def eigh(mut self, a: Int, uplo: Int, mut v: Int) raises -> Int:
        """`_Kit.eigh(A, uplo)`: (w 1 x n ascending, v n x n columns)."""
        var n = self.r(a)
        var w = self.zeros(1, n)
        v = self.zeros(n, n)
        var params = Python.list(PythonObject(n), PythonObject(uplo))
        if n >= 1 and self.c(a) == n and self.on_device(a) and self.has("x_decomp_dev_eigh"):
            _ = family_call(self.entry("x_decomp_dev_eigh"), PythonObject(self.matrices[a].device), self.address(w),
                            self.address(v), params)
            return w
        _ = family_call(self.entry("x_decomp_eigh"), self.address(a), self.address(w), self.address(v), params)
        return w

    def svd(mut self, a: Int, mut vt: Int) raises -> Int:
        """`_Kit.svd(A)`: (S 1 x n descending, Vt n x n) of a tall A."""
        var m = self.r(a)
        var n = self.c(a)
        var s = self.zeros(1, n)
        var v = self.zeros(n, n)
        var params = Python.list(PythonObject(m), PythonObject(n))
        if m >= n and n >= 1 and self.has("x_decomp_dev_svd") and self.resident:
            var da = self.device(a)
            _ = family_call(self.entry("x_decomp_dev_svd"), PythonObject(da), self.address(s), self.address(v), params)
        else:
            _ = family_call(self.entry("x_decomp_svd"), self.address(a), self.address(s), self.address(v), params)
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
        var params = Python.list(PythonObject(rows), PythonObject(cols))
        if rows * cols and self.resident:
            var q = self.device_output(rows, cols)
            var da = self.device(a)
            _ = family_call(self.entry("x_decomp_dev_orth_diag"), PythonObject(da), PythonObject(self.matrices[q].device),
                            self.address(diag), params)
            return q
        var q = self.copy(a)
        _ = family_call(self.entry("x_decomp_orth_diag"), self.address(q), self.address(diag), params)
        return q

    def eye(mut self, rows: Int, cols: Int) raises -> Int:
        var out = self.zeros(rows, cols)
        tsqr_fill_eye(self.ptr(out), rows, cols)
        return out
