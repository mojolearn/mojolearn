# SPDX-License-Identifier: Apache-2.0
# Native owner of LinearRegression's existing small-R TSQR/SVD launch sequence.
# The numerical entries and all rank/route choices match _Kit/_svd_tall. No
# new solver or performance default is introduced. Matrix descriptors are
# native structs; PythonObjects only own admitted host buffers and C entries.
from std.python import Python, PythonObject
from std.memory import memcpy
from bindings.classical_arrays import classical_array, classical_word
from bindings.sequential_arrays import sequential_address


struct ClassicalMatrix(Movable, Copyable):
    var host: PythonObject
    var device: Int
    var rows: Int
    var cols: Int

    def __init__(out self, host: PythonObject, device: Int, rows: Int, cols: Int):
        self.host = host
        self.device = device
        self.rows = rows
        self.cols = cols


struct ClassicalKit(Movable):
    var entries: PythonObject
    var context: PythonObject
    var matrices: List[ClassicalMatrix]
    var allocations: List[Int]
    var resident: Bool

    def __init__(out self, entries: PythonObject, context: PythonObject) raises:
        self.entries = entries
        self.context = context
        self.matrices = List[ClassicalMatrix]()
        self.allocations = List[Int]()
        self.resident = entries.get("x_decomp_dev_ew") is not None

    def close(mut self) raises:
        # Cleanup owns every successful allocation before the first upload.
        # Views may share an id, so only the allocation ledger frees buffers.
        # driver-loop: over native matrix allocation handles
        for i in range(len(self.allocations) - 1, -1, -1):
            _ = self.entries["x_decomp_dev_free"](self.allocations[i])
        self.allocations.clear()

    def add(mut self, host: PythonObject, rows: Int, cols: Int, device: Int = -1) -> Int:
        var result = len(self.matrices)
        self.matrices.append(ClassicalMatrix(host, device, rows, cols))
        return result

    def zeros(mut self, rows: Int, cols: Int) raises -> Int:
        # One spare word gives empty matrices the same safe pointer as _M._one.
        var host = classical_array(max(rows * cols, 1), 0, self.context)
        return self.add(host, rows, cols)

    def device_output(mut self, rows: Int, cols: Int) raises -> Int:
        var handle = Int(py=self.entries["x_decomp_dev_alloc"](max(rows * cols, 1)))
        self.allocations.append(handle)
        return self.add(PythonObject(None), rows, cols, handle)

    def host(mut self, index: Int) raises -> PythonObject:
        if self.matrices[index].host is None:
            var rows = self.matrices[index].rows
            var cols = self.matrices[index].cols
            var result = classical_array(max(rows * cols, 1), 0, self.context)
            if rows * cols:
                _ = self.entries["x_decomp_dev_download"](self.matrices[index].device,
                        sequential_address(result), rows * cols)
            self.matrices[index].host = result
            self.matrices[index].device = -1
        return self.matrices[index].host

    def address(mut self, index: Int) raises -> PythonObject:
        var owner = self.host(index)
        var result = sequential_address(owner)
        _ = owner^
        return result

    def device(mut self, index: Int) raises -> Int:
        if self.matrices[index].device < 0:
            var n = self.matrices[index].rows * self.matrices[index].cols
            var owner = self.host(index)
            var handle = Int(py=self.entries["x_decomp_dev_alloc"](max(n, 1)))
            self.allocations.append(handle)
            if n:
                _ = self.entries["x_decomp_dev_upload"](handle, sequential_address(owner), n)
            self.matrices[index].device = handle
            self.matrices[index].host = PythonObject(None)
            _ = owner^
        return self.matrices[index].device

    def constant(mut self, value: Float32, rows: Int = 1, cols: Int = 1) raises -> Int:
        var out = self.zeros(rows, cols)
        var owner = self.host(out)
        var data = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(owner)))
        # Native host constant construction, matching the original input array.
        for i in range(rows * cols):
            data[i] = value
        _ = owner^
        return out

    def indices(mut self, start: Int, stop: Int) raises -> Int:
        var out = self.zeros(stop - start, 1)
        var owner = self.host(out)
        var data = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(owner)))
        # Native index metadata, bounded by the TSQR width (512).
        for i in range(stop - start):
            data[i] = Float32(start + i)
        _ = owner^
        return out

    def broadcast(self, a: Int, b: Int) raises -> Int:
        if self.matrices[a].rows == self.matrices[b].rows and self.matrices[a].cols == self.matrices[b].cols:
            return 0
        if self.matrices[b].rows == 1 and self.matrices[a].cols == self.matrices[b].cols:
            return 1
        if self.matrices[b].cols == 1 and self.matrices[a].rows == self.matrices[b].rows:
            return 2
        if self.matrices[b].rows == 1 and self.matrices[b].cols == 1:
            return 3
        raise Error("classical TSQR: invalid internal broadcast")

    def ew(mut self, op: Int, a: Int, b: Int = -1, c: Int = -1, scalar: Float64 = 0.0) raises -> Int:
        var zero = self.zeros(1, 1)
        var ib = zero if b < 0 else b
        var ic = zero if c < 0 else c
        var bm = 3 if b < 0 else self.broadcast(a, ib)
        var cm = 3 if c < 0 else self.broadcast(a, ic)
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var params = Python.list(op, rows * cols, cols,
            self.matrices[ib].rows * self.matrices[ib].cols, bm,
            self.matrices[ic].rows * self.matrices[ic].cols, cm)
        if rows * cols and self.resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var db = self.device(ib)
            var dc = self.device(ic)
            _ = self.entries["x_decomp_dev_ew"](da, db, dc, self.matrices[out].device, params, PythonObject(scalar))
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var aa = self.address(a)
            var ab = self.address(ib)
            var ac = self.address(ic)
            var ao = self.address(out)
            _ = self.entries["x_decomp_ew"](aa, ab, ac, ao, params, PythonObject(scalar))
        return out

    def mm(mut self, a: Int, b: Int, ta: Bool = False, tb: Bool = False) raises -> Int:
        var m = self.matrices[a].cols if ta else self.matrices[a].rows
        var k = self.matrices[a].rows if ta else self.matrices[a].cols
        var k2 = self.matrices[b].cols if tb else self.matrices[b].rows
        var n = self.matrices[b].rows if tb else self.matrices[b].cols
        if k != k2:
            raise Error("classical TSQR: invalid internal matrix product")
        var params = Python.list(m, k, n, Int(ta), Int(tb))
        if m * n and m * k and k * n and self.resident:
            var out = self.device_output(m, n)
            var da = self.device(a)
            var db = self.device(b)
            _ = self.entries["x_decomp_dev_gemm"](da, db, self.matrices[out].device, params)
            return out
        var out = self.zeros(m, n)
        if m * n:
            var aa = self.address(a)
            var ab = self.address(b)
            var ao = self.address(out)
            _ = self.entries["x_decomp_gemm"](aa, ab, ao, params)
        return out

    def sum(mut self, a: Int, columns: Bool) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var nr = 1 if columns else rows
        var nc = cols if columns else 1
        var name = "colsum" if columns else "rowsum"
        if rows * cols and self.resident:
            var out = self.device_output(nr, nc)
            var da = self.device(a)
            _ = self.entries["x_decomp_dev_" + name](da, self.matrices[out].device, Python.list(rows, cols))
            return out
        var out = self.zeros(nr, nc)
        var aa = self.address(a)
        var ao = self.address(out)
        _ = self.entries["x_decomp_" + name](aa, ao, Python.list(rows, cols))
        return out

    def total(mut self, a: Int) raises -> Float64:
        var rows = self.sum(a, False)
        var total = self.sum(rows, True)
        var owner = self.host(total)
        var value = classical_word(owner)
        _ = owner^
        return value

    def move(mut self, a: Int, idx: Int, rows: Int, cols: Int, params: PythonObject,
             resident: Bool) raises -> Int:
        if rows * cols and resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var di = self.device(idx)
            _ = self.entries["x_decomp_dev_move"](da, di, self.matrices[out].device, params)
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var aa = self.address(a)
            var ai = self.address(idx)
            var ao = self.address(out)
            _ = self.entries["x_decomp_move"](aa, ai, ao, params)
        return out

    def take_rows(mut self, a: Int, idx: Int) raises -> Int:
        var rows = self.matrices[idx].rows * self.matrices[idx].cols
        var cols = self.matrices[a].cols
        return self.move(a, idx, rows, cols, Python.list(0, rows * cols, cols, 0, 0, 1, 0,
            self.matrices[a].rows * cols, rows * cols, rows), self.resident)

    def columns(mut self, a: Int, start: Int, stop: Int) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var width = stop - start
        # _M.take_cols only uses resident gather if A is already resident.
        if self.matrices[a].device >= 0 and width:
            var idx = self.indices(start, stop)
            return self.move(a, idx, rows, width, Python.list(4, rows * width, width, cols, 0, 0, 0,
                rows * cols, rows * width, width), True)
        var out = self.zeros(rows, width)
        var source = self.host(a)
        var result = self.host(out)
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(source)))
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
        # Host ABI copy of existing host words; no arithmetic route changes.
        for r in range(rows):
            for c in range(width):
                dst[r * width + c] = src[r * cols + start + c]
        _ = source^
        _ = result^
        return out

    def transpose(mut self, a: Int) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        if rows == 1 or cols == 1:
            return self.add(self.matrices[a].host, cols, rows, self.matrices[a].device)
        var one = self.zeros(1, 1)
        return self.move(a, one, cols, rows, Python.list(1, rows * cols, rows, cols, 0, 0, 0,
            rows * cols, rows * cols, 1), self.resident)

    def strided(mut self, a: Int, count: Int, stride: Int, offset: Int = 0) raises -> Int:
        var idx = self.constant(Float32(offset))
        var actual_stride = offset + 1 if count == 1 else stride
        return self.move(a, idx, 1, count, Python.list(4, count, 1, actual_stride, 0, 0, 0,
            self.matrices[a].rows * self.matrices[a].cols, count, 1), self.resident)

    def word(mut self, a: Int) raises -> Float64:
        var index = a
        if self.matrices[a].device >= 0 and self.matrices[a].rows * self.matrices[a].cols > 1:
            index = self.strided(a, 1, 1)
        var owner = self.host(index)
        var value = classical_word(owner)
        _ = owner^
        return value

    def order(mut self, a: Int) raises -> Int:
        var n = self.matrices[a].rows * self.matrices[a].cols
        if n and self.resident:
            # The same device max/NaN check and stable radix order as _Kit.
            var maximum = self.device_output(1, 1)
            var da = self.device(a)
            _ = self.entries["x_decomp_dev_reduce"](da, self.matrices[maximum].device, Python.list(n, 1))
            var v = self.word(maximum)
            if v != v:
                raise Error("x_decomp: a NaN has no order (refused)")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            _ = self.entries["x_decomp_dev_order_f"](da, da, self.matrices[out].device,
                                                    self.matrices[count].device, Python.list(n, 0, 0))
            return out
        var out = self.zeros(n, 1)
        if n:
            var aa = self.address(a)
            var ao = self.address(out)
            _ = self.entries["x_decomp_order_f"](aa, n, ao)
        return out

    def signs(mut self, a: Int) raises -> Int:
        var negated = self.ew(14, a, scalar=-1.0)
        var negative = self.constant(-1.0)
        var positive = self.constant(1.0)
        return self.ew(35, negated, negative, positive)

    def copy_host(mut self, a: Int) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var out = self.zeros(rows, cols)
        var source = self.host(a)
        var result = self.host(out)
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(source)))
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
        memcpy(dest=dst, src=src, count=rows * cols)
        _ = source^
        _ = result^
        return out

    def svd(mut self, a: Int) raises -> Tuple[Int, Int, Int]:
        var m = self.matrices[a].rows
        var n = self.matrices[a].cols
        var s = self.zeros(1, n)
        var v = self.zeros(n, n)
        var sa = self.address(s)
        var va = self.address(v)
        if self.resident and self.entries.get("x_decomp_dev_svd") is not None:
            var da = self.device(a)
            _ = self.entries["x_decomp_dev_svd"](da, sa, va, Python.list(m, n))
        else:
            var aa = self.address(a)
            _ = self.entries["x_decomp_svd"](aa, sa, va, Python.list(m, n))
        var negated = self.ew(14, s, scalar=-1.0)
        var order = self.order(negated)
        var st = self.transpose(s)
        var sorted_s = self.take_rows(st, order)
        s = self.transpose(sorted_s)
        var vt = self.transpose(v)
        vt = self.take_rows(vt, order)
        sa = self.address(s)
        var rank = Int(py=self.entries["x_decomp_rank_above"](sa, Python.list(n), PythonObject(0.00000095367431640625)))
        var av = self.mm(a, vt, tb=True)
        var used_av = av
        var used_s = s
        if rank < n:
            used_av = self.columns(av, 0, rank)
            used_s = self.columns(s, 0, rank)
        var ug = self.ew(3, used_av, used_s)
        if rank == n:
            var diag = self.zeros(1, n)
            var q: Int
            var dia = self.address(diag)
            if self.resident and m * n:
                q = self.device_output(m, n)
                var du = self.device(ug)
                _ = self.entries["x_decomp_dev_orth_diag"](du, self.matrices[q].device, dia, Python.list(m, n))
            else:
                q = self.copy_host(ug)
                var qa = self.address(q)
                _ = self.entries["x_decomp_orth_diag"](qa, dia, Python.list(m, n))
            var absolute = self.ew(13, diag)
            var zero = self.zeros(1, 1)
            var nonpositive = self.ew(34, absolute, zero)
            if self.total(nonpositive) == 0:
                var sign = self.signs(diag)
                var u = self.ew(2, q, sign)
                return (u, s, vt)
        if rank == 0:
            var eye = self.zeros(m, n)
            var owner = self.host(eye)
            var data = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(owner)))
            for i in range(min(m, n)):
                data[i * (n + 1)] = 1.0
            _ = owner^
            return (eye, s, vt)
        var h = self.copy_host(ug)
        var tau = self.zeros(1, min(m, rank))
        var ha = self.address(h)
        var ta = self.address(tau)
        _ = self.entries["x_decomp_geqrf"](ha, ta, Python.list(m, rank))
        var qc = self.zeros(m, n)
        var qa = self.address(qc)
        _ = self.entries["x_decomp_orgqr"](ha, ta, qa, Python.list(m, rank, min(m, rank), n))
        var diagonal = self.strided(h, rank, rank + 1)
        var sign = self.signs(diagonal)
        if n > rank:
            var remaining = self.constant(1.0, 1, n - rank)
            var joined = self.device_output(1, n) if self.resident else self.zeros(1, n)
            var offset = 0
            # driver-loop: over the two sign-vector fragments
            for part in range(2):
                var source = sign if part == 0 else remaining
                var count = self.matrices[source].cols
                var params = Python.list(2, count, count, n, offset, 0, 0, count, n, 1)
                var one = self.zeros(1, 1)
                if self.resident:
                    var ds = self.device(source)
                    var di = self.device(one)
                    _ = self.entries["x_decomp_dev_move"](ds, di, self.matrices[joined].device, params)
                else:
                    var aa = self.address(source)
                    var ai = self.address(one)
                    var ao = self.address(joined)
                    _ = self.entries["x_decomp_move"](aa, ai, ao, params)
                offset += count
            sign = joined
        var u = self.ew(2, qc, sign)
        return (u, s, vt)


def classical_tsqr_solve(ra: PythonObject, rows: Int, cols: Int,
                         entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var kit = ClassicalKit(entries, context)
    try:
        # R_aug is host output from the existing one-upload TSQR factor entry.
        # Its first cols rows and two column slices match the old _M copies.
        var top = kit.add(ra, cols, cols + 1)
        var r = kit.columns(top, 0, cols)
        var c = kit.columns(top, cols, cols + 1)
        var squared = kit.ew(8, r)
        var sums = kit.sum(squared, True)
        var scale = kit.ew(38, sums)
        r = kit.ew(2, r, scale)
        var decomposition = kit.svd(r)
        var u = decomposition[0]
        var s = decomposition[1]
        var vt = decomposition[2]
        # Same binary64 scalar multiply then one Float32 narrowing as _f32.
        var cut = Float64(Float32(kit.word(s) * (Float64(cols) * 0.00000011920928955078125)))
        var kept = kit.ew(23, s, scalar=cut)
        var rank = Int(kit.total(kept))
        var zero = kit.zeros(1, 1)
        var selected = kit.ew(35, s, s, zero, cut)
        var inverse = kit.ew(16, selected)
        var utc = kit.mm(u, c, ta=True)
        var invt = kit.transpose(inverse)
        var weighted = kit.ew(2, utc, invt)
        var x = kit.mm(vt, weighted, ta=True)
        var st = kit.transpose(scale)
        x = kit.ew(2, x, st)
        # Preserve the old residual/rank execution, although LinearRegression
        # only publishes coefficients from this internal least-squares result.
        if rank == cols and rows > cols:
            var tail = classical_array(cols + 1, 0, context)
            var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(ra)))
            var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(tail)))
            memcpy(dest=dst, src=src + cols * (cols + 1), count=cols + 1)
            var tail_id = kit.add(tail, 1, cols + 1)
            var last = kit.columns(tail_id, cols, cols + 1)
            var residual_squared = kit.ew(8, last)
            _ = kit.sum(residual_squared, True)
            _ = tail^
        var result = kit.host(x)
        # The final matrix is cols x 1, already a flat native ndarray buffer.
        return result
    finally:
        kit.close()
