# SPDX-License-Identifier: Apache-2.0
"""Typed owner of LinearRegression's existing TSQR/SVD operation sequence.

Host storage has native owners and reference-counted views. Device storage
uses the existing pool and an allocation ledger, including on exceptions.
The numeric operations and selected kernels are unchanged.
"""
from std.memory import memcpy
from core.hostptr import f32_ptr
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.classical_tsqr_impl import tsqr_fill_constant, tsqr_fill_indices, tsqr_copy_columns, tsqr_fill_eye
from x_decomp.descriptor_schema import (
    XD_CAPABILITIES, XD_CAP_RESIDENT, XD_EW, XD_GEMM, XD_COLSUM, XD_ROWSUM, XD_MOVE,
    XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD, XD_DEV_DOWNLOAD, XD_DEV_EW, XD_DEV_GEMM,
    XD_DEV_COLSUM, XD_DEV_ROWSUM, XD_DEV_MOVE, XD_ORDER, XD_DEV_REDUCE, XD_DEV_ORDER,
    XD_SVD, XD_DEV_SVD, XD_RANK, XD_ORTH_DIAG, XD_DEV_ORTH_DIAG, XD_GEQRF, XD_ORGQR,
)


struct LinearHostStorage(Movable):
    var words: List[Float32]
    var address: Int
    var references: Int

    def __init__(out self, var words: List[Float32], borrowed: Int = 0):
        self.address = borrowed if borrowed else Int(words.unsafe_ptr())
        self.words = words^
        self.references = 0


struct LinearMatrix(Copyable, Movable):
    var host: Int
    var device: Int
    var rows: Int
    var cols: Int

    def __init__(out self, host: Int, device: Int, rows: Int, cols: Int):
        self.host = host
        self.device = device
        self.rows = rows
        self.cols = cols


struct LinearKit(Movable):
    var entry: NativeDescriptorCall
    var matrices: List[LinearMatrix]
    var storage: List[LinearHostStorage]
    var allocations: List[Int]
    var resident: Bool
    var zero: Int

    def __init__(out self, entry: NativeDescriptorCall) raises:
        self.entry = entry
        self.matrices = List[LinearMatrix]()
        self.storage = List[LinearHostStorage]()
        self.allocations = List[Int]()
        self.resident = (native_descriptor_call(entry, XD_CAPABILITIES, [], [], []) & XD_CAP_RESIDENT) != 0
        self.zero = -1

    def call(self, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
        return native_descriptor_call(self.entry, operation, addresses, ints, floats)

    def prefixed(self, var prefix: List[Int], values: List[Int]) -> List[Int]:
        for i in range(len(values)):  # driver-loop: over fixed primitive metadata slots
            prefix.append(values[i])
        return prefix^

    def shared_zero(mut self) raises -> Int:
        if self.zero < 0:
            self.zero = self.zeros(1, 1)
        return self.zero

    def close(mut self) raises:
        var failure = String("")
        while len(self.allocations):  # driver-loop: over owned native allocation handles
            var handle = self.allocations.pop()
            try:
                _ = self.call(XD_DEV_FREE, [], [handle], [])
            except problem:
                if failure.byte_length() == 0:
                    failure = String(problem)
        if failure.byte_length() != 0:
            raise Error(failure)

    def add(mut self, host: Int, rows: Int, cols: Int, device: Int = -1) -> Int:
        if host >= 0:
            self.storage[host].references += 1
        var result = len(self.matrices)
        self.matrices.append(LinearMatrix(host, device, rows, cols))
        return result

    def borrowed(mut self, address: Int, rows: Int, cols: Int) -> Int:
        var host = len(self.storage)
        self.storage.append(LinearHostStorage(List[Float32](), address))
        return self.add(host, rows, cols)

    def allocate_host(mut self, cells: Int) -> Int:
        var result = len(self.storage)
        self.storage.append(LinearHostStorage(List[Float32](length=max(cells, 1), fill=0.0)))
        return result

    def release_host(mut self, index: Int):
        var slot = self.matrices[index].host
        if slot >= 0:
            self.storage[slot].references -= 1
            if self.storage[slot].references == 0:
                self.storage[slot].words = List[Float32]()
                self.storage[slot].address = 0
            self.matrices[index].host = -1

    def zeros(mut self, rows: Int, cols: Int) -> Int:
        var host = self.allocate_host(rows * cols)
        return self.add(host, rows, cols)

    def device_output(mut self, rows: Int, cols: Int) raises -> Int:
        var handle = self.call(XD_DEV_ALLOC, [], [max(rows * cols, 1)], [])
        self.allocations.append(handle)
        return self.add(-1, rows, cols, handle)

    def host(mut self, index: Int) raises -> Int:
        if self.matrices[index].host < 0:
            var cells = self.matrices[index].rows * self.matrices[index].cols
            var slot = self.allocate_host(cells)
            var address = self.storage[slot].address
            if cells:
                _ = self.call(XD_DEV_DOWNLOAD, [address], [self.matrices[index].device, cells], [])
            self.matrices[index].host = slot
            self.storage[slot].references = 1
            self.matrices[index].device = -1
        return self.storage[self.matrices[index].host].address

    def address(mut self, index: Int) raises -> Int:
        return self.host(index)

    def device(mut self, index: Int) raises -> Int:
        if self.matrices[index].device < 0:
            var cells = self.matrices[index].rows * self.matrices[index].cols
            var address = self.host(index)
            var handle = self.call(XD_DEV_ALLOC, [], [max(cells, 1)], [])
            self.allocations.append(handle)
            if cells:
                _ = self.call(XD_DEV_UPLOAD, [address], [handle, cells], [])
            self.matrices[index].device = handle
            self.release_host(index)
        return self.matrices[index].device

    def constant(mut self, value: Float32, rows: Int = 1, cols: Int = 1) raises -> Int:
        var result = self.zeros(rows, cols)
        tsqr_fill_constant(f32_ptr(self.host(result)), rows * cols, value)
        return result

    def indices(mut self, start: Int, stop: Int) raises -> Int:
        var result = self.zeros(stop - start, 1)
        tsqr_fill_indices(f32_ptr(self.host(result)), start, stop)
        return result

    def write_result(mut self, index: Int, address: Int, count: Int) raises:
        if self.matrices[index].device >= 0:
            _ = self.call(XD_DEV_DOWNLOAD, [address], [self.matrices[index].device, count], [])
        else:
            memcpy(dest=f32_ptr(address), src=f32_ptr(self.host(index)), count=count)

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
        var ib = b
        var ic = c
        if b < 0 or c < 0:
            var zero = self.shared_zero()
            ib = zero if b < 0 else b
            ic = zero if c < 0 else c
        var bm = 3 if b < 0 else self.broadcast(a, ib)
        var cm = 3 if c < 0 else self.broadcast(a, ic)
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var params: List[Int] = [op, rows * cols, cols,
            self.matrices[ib].rows * self.matrices[ib].cols, bm,
            self.matrices[ic].rows * self.matrices[ic].cols, cm]
        if rows * cols and self.resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var db = self.device(ib)
            var dc = self.device(ic)
            _ = self.call(XD_DEV_EW, [], self.prefixed([da, db, dc, self.matrices[out].device], params), [scalar])
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var aa = self.address(a)
            var ab = self.address(ib)
            var ac = self.address(ic)
            var ao = self.address(out)
            _ = self.call(XD_EW, [aa, ab, ac, ao], params, [scalar])
        return out

    def mm(mut self, a: Int, b: Int, ta: Bool = False, tb: Bool = False) raises -> Int:
        var m = self.matrices[a].cols if ta else self.matrices[a].rows
        var k = self.matrices[a].rows if ta else self.matrices[a].cols
        var k2 = self.matrices[b].cols if tb else self.matrices[b].rows
        var n = self.matrices[b].rows if tb else self.matrices[b].cols
        if k != k2:
            raise Error("classical TSQR: invalid internal matrix product")
        var params: List[Int] = [m, k, n, Int(ta), Int(tb)]
        if m * n and m * k and k * n and self.resident:
            var out = self.device_output(m, n)
            var da = self.device(a)
            var db = self.device(b)
            _ = self.call(XD_DEV_GEMM, [], self.prefixed([da, db, self.matrices[out].device], params), [])
            return out
        var out = self.zeros(m, n)
        if m * n:
            var aa = self.address(a)
            var ab = self.address(b)
            var ao = self.address(out)
            _ = self.call(XD_GEMM, [aa, ab, ao], params, [])
        return out

    def sum(mut self, a: Int, columns: Bool) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var nr = 1 if columns else rows
        var nc = cols if columns else 1
        var op = XD_COLSUM if columns else XD_ROWSUM
        var dev_op = XD_DEV_COLSUM if columns else XD_DEV_ROWSUM
        if rows * cols and self.resident:
            var out = self.device_output(nr, nc)
            var da = self.device(a)
            _ = self.call(dev_op, [], [da, self.matrices[out].device, rows, cols], [])
            return out
        var out = self.zeros(nr, nc)
        var aa = self.address(a)
        var ao = self.address(out)
        _ = self.call(op, [aa, ao], [rows, cols], [])
        return out

    def total(mut self, a: Int) raises -> Float64:
        var rows = self.sum(a, False)
        var total = self.sum(rows, True)
        var owner = self.host(total)
        var value = Float64(f32_ptr(owner)[])
        return value

    def move(mut self, a: Int, idx: Int, rows: Int, cols: Int, params: List[Int],
             resident: Bool) raises -> Int:
        if rows * cols and resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var di = self.device(idx)
            _ = self.call(XD_DEV_MOVE, [], self.prefixed([da, di, self.matrices[out].device], params), [])
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var aa = self.address(a)
            var ai = self.address(idx)
            var ao = self.address(out)
            _ = self.call(XD_MOVE, [aa, ai, ao], params, [])
        return out

    def take_rows(mut self, a: Int, idx: Int) raises -> Int:
        var rows = self.matrices[idx].rows * self.matrices[idx].cols
        var cols = self.matrices[a].cols
        return self.move(a, idx, rows, cols, [0, rows * cols, cols, 0, 0, 1, 0,
            self.matrices[a].rows * cols, rows * cols, rows], self.resident)

    def columns(mut self, a: Int, start: Int, stop: Int) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var width = stop - start
        # _M.take_cols only uses resident gather if A is already resident.
        if self.matrices[a].device >= 0 and width:
            var idx = self.indices(start, stop)
            return self.move(a, idx, rows, width, [4, rows * width, width, cols, 0, 0, 0,
                rows * cols, rows * width, width], True)
        var out = self.zeros(rows, width)
        var source = self.host(a)
        var result = self.host(out)
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=source)
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=result)
        tsqr_copy_columns(src, dst, rows, cols, start, width)
        return out

    def transpose(mut self, a: Int) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        if rows == 1 or cols == 1:
            var host = self.matrices[a].host
            var handle = self.matrices[a].device
            return self.add(host, cols, rows, handle)
        var one = self.zeros(1, 1)
        return self.move(a, one, cols, rows, [1, rows * cols, rows, cols, 0, 0, 0,
            rows * cols, rows * cols, 1], self.resident)

    def strided(mut self, a: Int, count: Int, stride: Int, offset: Int = 0) raises -> Int:
        var idx = self.constant(Float32(offset))
        var actual_stride = offset + 1 if count == 1 else stride
        return self.move(a, idx, 1, count, [4, count, 1, actual_stride, 0, 0, 0,
            self.matrices[a].rows * self.matrices[a].cols, count, 1], self.resident)

    def word(mut self, a: Int) raises -> Float64:
        var index = a
        if self.matrices[a].device >= 0 and self.matrices[a].rows * self.matrices[a].cols > 1:
            index = self.strided(a, 1, 1)
        var owner = self.host(index)
        var value = Float64(f32_ptr(owner)[])
        return value

    def order(mut self, a: Int) raises -> Int:
        var n = self.matrices[a].rows * self.matrices[a].cols
        if n and self.resident:
            # The same device max/NaN check and stable radix order as _Kit.
            var maximum = self.device_output(1, 1)
            var da = self.device(a)
            _ = self.call(XD_DEV_REDUCE, [], [da, self.matrices[maximum].device, n, 1], [])
            var v = self.word(maximum)
            if v != v:
                raise Error("x_decomp: a NaN has no order (refused)")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            _ = self.call(XD_DEV_ORDER, [], [da, da, self.matrices[out].device, self.matrices[count].device, n, 0, 0], [])
            return out
        var out = self.zeros(n, 1)
        if n:
            var aa = self.address(a)
            var ao = self.address(out)
            _ = self.call(XD_ORDER, [aa, ao], [n], [])
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
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=source)
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=result)
        memcpy(dest=dst, src=src, count=rows * cols)
        return out

    def svd(mut self, a: Int) raises -> Tuple[Int, Int, Int]:
        var m = self.matrices[a].rows
        var n = self.matrices[a].cols
        var s = self.zeros(1, n)
        var v = self.zeros(n, n)
        var sa = self.address(s)
        var va = self.address(v)
        if self.resident:
            var da = self.device(a)
            _ = self.call(XD_DEV_SVD, [sa, va], [da, m, n], [])
        else:
            var aa = self.address(a)
            _ = self.call(XD_SVD, [aa, sa, va], [m, n], [])
        var negated = self.ew(14, s, scalar=-1.0)
        var order = self.order(negated)
        var st = self.transpose(s)
        var sorted_s = self.take_rows(st, order)
        s = self.transpose(sorted_s)
        var vt = self.transpose(v)
        vt = self.take_rows(vt, order)
        sa = self.address(s)
        var rank = self.call(XD_RANK, [sa], [n], [0.00000095367431640625])
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
                _ = self.call(XD_DEV_ORTH_DIAG, [dia], [du, self.matrices[q].device, m, n], [])
            else:
                q = self.copy_host(ug)
                var qa = self.address(q)
                _ = self.call(XD_ORTH_DIAG, [qa, dia], [m, n], [])
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
            var data = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=owner)
            tsqr_fill_eye(data, m, n)
            return (eye, s, vt)
        var h = self.copy_host(ug)
        var tau = self.zeros(1, min(m, rank))
        var ha = self.address(h)
        var ta = self.address(tau)
        _ = self.call(XD_GEQRF, [ha, ta], [m, rank], [])
        var qc = self.zeros(m, n)
        var qa = self.address(qc)
        _ = self.call(XD_ORGQR, [ha, ta, qa], [m, rank, min(m, rank), n], [])
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
                var params: List[Int] = [2, count, count, n, offset, 0, 0, count, n, 1]
                var one = self.zeros(1, 1)
                if self.resident:
                    var ds = self.device(source)
                    var di = self.device(one)
                    _ = self.call(XD_DEV_MOVE, [], self.prefixed([ds, di, self.matrices[joined].device], params), [])
                else:
                    var aa = self.address(source)
                    var ai = self.address(one)
                    var ao = self.address(joined)
                    _ = self.call(XD_MOVE, [aa, ai, ao], params, [])
                offset += count
            sign = joined
        var u = self.ew(2, qc, sign)
        return (u, s, vt)



def linear_tsqr_solve(ra: Int, cols: Int, coef: Int, entry: NativeDescriptorCall) raises:
    var kit = LinearKit(entry)
    try:
        # R_aug is host output from the existing one-upload TSQR factor entry.
        # Its first cols rows and two column slices match the old _M copies.
        var top = kit.borrowed(ra, cols, cols + 1)
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
        # The kept-rank count (`ew 23` + `total`, a device sum and a scalar
        # download) and the residual-norm sum that followed it were read by
        # nothing: LinearRegression publishes only the coefficients `x`. Both
        # are gone; `x` is formed by the same ops on the same words.
        var zero = kit.zeros(1, 1)
        var selected = kit.ew(35, s, s, zero, cut)
        var inverse = kit.ew(16, selected)
        var utc = kit.mm(u, c, ta=True)
        var invt = kit.transpose(inverse)
        var weighted = kit.ew(2, utc, invt)
        var x = kit.mm(vt, weighted, ta=True)
        var st = kit.transpose(scale)
        x = kit.ew(2, x, st)
        kit.write_result(x, coef, cols)
    finally:
        kit.close()
