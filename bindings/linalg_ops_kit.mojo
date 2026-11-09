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
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays import classical_address
from bindings.family_common import family_call
from bindings.linalg_ops_buffers import lin_truth
from bindings.admitted_input import admitted_zeros
from core.classical_tsqr_impl import tsqr_fill_constant, tsqr_copy_columns, tsqr_fill_eye
# The x_decomp native kernel-call table (lane ne3-f-kernel-table, rows N293,
# N294): alloc / upload / download / free, ew, gemm, the sums and the moves
# call one abi("C") dispatch function (bindings/kit_call.mojo) instead of a
# Python entry call with a Python parameter list per operation.
from bindings.kit_call import kit_connect, kit_provider, kit_call
from x_decomp.descriptor_schema import (
    XD_ABI_NAME, XD_OPERATION_COUNT, XD_MODE, XD_CAPABILITIES,
    XD_EW, XD_GEMM, XD_COLSUM, XD_ROWSUM, XD_MOVE,
    XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD, XD_DEV_DOWNLOAD, XD_DEV_EW, XD_DEV_GEMM,
    XD_DEV_COLSUM, XD_DEV_ROWSUM, XD_DEV_MOVE,
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


struct KitMatrix(Copyable, Movable):
    # `borrowed` (row N293): the address of an admitted caller buffer the
    # matrix reads in place (no host copy), kept alive by `keep`; 0 when the
    # matrix owns its words. A borrowed matrix is a host matrix for every
    # route decision, exactly as the copy it replaces was; it is never a
    # write target (every kit operation writes a fresh output).
    var host: PythonObject
    var device: Int
    var rows: Int
    var cols: Int
    var borrowed: Int
    var keep: PythonObject

    def __init__(out self, host: PythonObject, device: Int, rows: Int, cols: Int):
        self.host = host
        self.device = device
        self.rows = rows
        self.cols = cols
        self.borrowed = 0
        self.keep = PythonObject(None)


struct LinKit(Movable):
    """`_Kit(mode)` over the x_decomp binding of the process's default mode."""
    var b: PythonObject
    var context: PythonObject
    var matrices: List[KitMatrix]
    var allocations: List[Int]
    var resident: Bool
    var handles: Dict[String, PythonObject]  # entry handles resolved by this kit
    var table: PythonObject  # the binding's x_decomp_native_api_v1 provider, connected once per kit
    var zero: Int  # the kit's one 1 x 1 zero operand (row N294), -1 until made
    var minus_one: Int  # neg_signs' constants, made once per kit (-1 until made)
    var plus_one: Int

    def __init__(out self, binding: PythonObject, context: PythonObject) raises:
        if py_type(binding) is not py_type(native_loaded_module("builtins")):
            raise Error("TypeError: linalg decomposition requires an exact native provider module")
        self.b = binding
        self.context = context
        self.matrices = List[KitMatrix]()
        self.allocations = List[Int]()
        self.handles = Dict[String, PythonObject]()
        self.table = PythonObject(None)
        self.zero = -1
        self.minus_one = -1
        self.plus_one = -1
        # `_Kit._res`: the binding has the resident entries (a GPU binding)
        self.resident = neural_descriptor_entry(binding, "x_decomp_dev_ew", False) is not None

    # ---- the native kernel-call table (x_decomp/descriptor_schema.mojo)
    def xd(mut self) raises -> PythonObject:
        if self.table is None:
            var snapshot = kit_connect(self.b, "x_decomp_native_api_v1", String(XD_ABI_NAME),
                                       XD_OPERATION_COUNT, XD_MODE, XD_CAPABILITIES)
            self.table = kit_provider(snapshot)
        return self.table

    def dcall(mut self, op: Int, ints: List[Int], s: Float64 = 0.0) raises -> Int:
        """A device-resident operation on pool ids (under the GIL, as the
        `dev_*_py` doors ran)."""
        var floats = List[Float64]()
        if op == XD_DEV_EW:
            floats.append(s)
        return kit_call(self.xd(), op, List[Int](), ints, floats, False)

    def hcall(mut self, op: Int, addresses: List[Int], ints: List[Int], s: Float64 = 0.0) raises -> Int:
        """A host-address operation (the GIL released, as the doors did)."""
        var floats = List[Float64]()
        if op == XD_EW:
            floats.append(s)
        return kit_call(self.xd(), op, addresses, ints, floats, True)

    def addr(mut self, a: Int) raises -> Int:
        """The host address of matrix a (a device matrix moves to the host)."""
        if self.matrices[a].borrowed != 0 and self.matrices[a].device < 0:
            return self.matrices[a].borrowed
        return Int(py=self.address(a))

    def shared_zero(mut self) raises -> Int:
        """The kit's 1 x 1 zero (row N294): the absent-operand and unused-index
        constant every ew / move used to allocate (and, on a resident kit,
        upload) once per operation. Read-only, so one per kit serves all."""
        if self.zero < 0:
            self.zero = self.zeros(1, 1)
        return self.zero

    def entry(self, name: String) raises -> PythonObject:
        return neural_descriptor_entry(self.b, name)

    def _hot(mut self, name: String) raises -> PythonObject:
        # The kit's own kernel ops resolve each binding entry once per kit
        # (one descriptor admission), not once per op; a refusal is raised at
        # first use exactly as `entry` raises it.
        if name in self.handles:
            return self.handles[name]
        var handle = neural_descriptor_entry(self.b, name)
        self.handles[name] = handle
        return handle

    def has(self, name: String) raises -> Bool:
        """`_Kit._opt_dev(name)`."""
        return self.resident and neural_descriptor_entry(self.b, name, False) is not None

    def close(mut self):
        """Return every device buffer this call allocated to the pool."""
        for i in range(len(self.allocations) - 1, -1, -1):  # driver-loop: over native matrix allocation handles
            try:
                _ = self.dcall(XD_DEV_FREE, [self.allocations[i]])
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
        var host = admitted_zeros([max(rows * cols, 1)], "<f4", self.context).keep
        return self.add(host, rows, cols)

    def borrow(mut self, keep: PythonObject, addr: Int, rows: Int, cols: Int) raises -> Int:
        """`_xd_matrix` without the copy (row N293): a host matrix that reads
        the admitted C-order float32 buffer at `addr` in place; `keep` (the
        admission's owner) is retained with the matrix. A resident operation
        uploads straight from `addr`."""
        var out = self.add(PythonObject(None), rows, cols)
        self.matrices[out].borrowed = addr
        self.matrices[out].keep = keep
        if rows * cols == 0 or addr == 0:
            # an empty operand keeps of_buffer's owned spare word
            self.matrices[out].borrowed = 0
            self.matrices[out].keep = PythonObject(None)
            self.matrices[out].host = admitted_zeros([max(rows * cols, 1)], "<f4", self.context).keep
        return out

    def of_buffer(mut self, addr: Int, rows: Int, cols: Int) raises -> Int:
        """`_xd_matrix`: a host copy of a C-order float32 buffer."""
        var out = self.zeros(rows, cols)
        if rows * cols:
            memcpy(dest=self.ptr(out), src=KIT_F32P(unsafe_from_address=addr), count=rows * cols)
            _ = out  # keep the Python object alive through the native call above
        return out

    def device_output(mut self, rows: Int, cols: Int) raises -> Int:
        """`_Kit._dout`."""
        var handle = self.dcall(XD_DEV_ALLOC, [max(rows * cols, 1)])
        self.allocations.append(handle)
        return self.add(PythonObject(None), rows, cols, handle)

    def host(mut self, a: Int) raises -> PythonObject:
        """`_M.s`: the host store, downloading a device matrix once (it moves)."""
        if self.matrices[a].host is None:
            var count = self.n(a)
            var result = admitted_zeros([max(count, 1)], "<f4", self.context).keep
            if self.matrices[a].borrowed != 0 and self.matrices[a].device < 0:
                # a borrowed matrix asked for its owned store (no kit path
                # does today): one copy of the admitted words, as of_buffer made
                if count:
                    memcpy(dest=KIT_F32P(unsafe_from_address=Int(py=classical_address(result))),
                           src=KIT_F32P(unsafe_from_address=self.matrices[a].borrowed), count=count)
                self.matrices[a].borrowed = 0
                self.matrices[a].keep = PythonObject(None)
                self.matrices[a].host = result
                return result
            if count:
                _ = kit_call(self.xd(), XD_DEV_DOWNLOAD, [Int(py=classical_address(result))],
                             [self.matrices[a].device, count], List[Float64](), True)
                _ = result  # keep the Python object alive through the native call above
            self.matrices[a].host = result
            self.matrices[a].device = -1
        return self.matrices[a].host

    def ptr(mut self, a: Int) raises -> KIT_F32P:
        if self.matrices[a].borrowed != 0 and self.matrices[a].device < 0:
            return KIT_F32P(unsafe_from_address=self.matrices[a].borrowed)
        var owner = self.host(a)
        var p = KIT_F32P(unsafe_from_address=Int(py=classical_address(owner)))
        _ = owner  # keep the Python object alive through the native call above
        _ = owner^
        return p

    def address(mut self, a: Int) raises -> PythonObject:
        if self.matrices[a].borrowed != 0 and self.matrices[a].device < 0:
            return PythonObject(self.matrices[a].borrowed)
        var owner = self.host(a)
        var result = classical_address(owner)
        _ = owner  # keep the Python object alive through the native call above
        _ = owner^
        return result

    def device(mut self, a: Int) raises -> Int:
        """`_Kit._did`: the device id, uploading a host matrix (it moves)."""
        if self.matrices[a].device < 0:
            var count = self.n(a)
            var handle = self.dcall(XD_DEV_ALLOC, [max(count, 1)])
            self.allocations.append(handle)
            if self.matrices[a].borrowed != 0:
                # straight from the admitted buffer (row N293)
                if count:
                    _ = kit_call(self.xd(), XD_DEV_UPLOAD, [self.matrices[a].borrowed], [handle, count],
                                 List[Float64](), True)
                _ = self.matrices[a].keep  # keep the admitted owner alive through the upload above
                self.matrices[a].borrowed = 0
                self.matrices[a].keep = PythonObject(None)
            else:
                var owner = self.host(a)
                if count:
                    _ = kit_call(self.xd(), XD_DEV_UPLOAD, [Int(py=classical_address(owner))], [handle, count],
                                 List[Float64](), True)
                    _ = owner  # keep the Python object alive through the native call above
                _ = owner^
            self.matrices[a].device = handle
            self.matrices[a].host = PythonObject(None)
        return self.matrices[a].device

    def out(mut self, a: Int) raises -> PythonObject:
        """`_M.out()`: a public float32 Array (C order) of the matrix."""
        var rows = self.r(a)
        var cols = self.c(a)
        var result = admitted_zeros([rows, cols], "<f4", self.context).keep
        if rows * cols:
            memcpy(dest=KIT_F32P(unsafe_from_address=Int(py=classical_address(result))), src=self.ptr(a),
                   count=rows * cols)
            _ = result  # keep the Python object alive through the native call above
        return result

    def out_vector(mut self, a: Int, length: Int) raises -> PythonObject:
        """`_M.out((length,))`."""
        var result = admitted_zeros([length], "<f4", self.context).keep
        if length:
            memcpy(dest=KIT_F32P(unsafe_from_address=Int(py=classical_address(result))), src=self.ptr(a),
                   count=length)
            _ = result  # keep the Python object alive through the native call above
        return result

    def constant(mut self, value: Float32, rows: Int = 1, cols: Int = 1) raises -> Int:
        """`_M.of([value] * (rows * cols), rows, cols)` / `_Kit.const`."""
        var out = self.zeros(rows, cols)
        tsqr_fill_constant(self.ptr(out), rows * cols, value)
        _ = out  # keep the Python object alive through the native call above
        return out

    def reshape(mut self, a: Int, rows: Int, cols: Int) raises -> Int:
        """`_M.reshape`: the same words (a view of the same store or buffer)."""
        var host = self.matrices[a].host
        var device = self.matrices[a].device
        var out = self.add(host, rows, cols, device)
        self.matrices[out].borrowed = self.matrices[a].borrowed
        self.matrices[out].keep = self.matrices[a].keep
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
            # every owner sits in self.matrices, alive through the call
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
        var _keep_ret_t_r279 = self.ptr(t)[0]
        _ = t  # keep the Python object alive through the native call above
        var _keep_ret_t = _keep_ret_t_r279^
        _ = t  # keep the Python object alive through the native call above
        return _keep_ret_t^

    def move(mut self, a: Int, idx: Int, rows: Int, cols: Int, words: List[Int], resident: Bool) raises -> Int:
        """A move into a fresh rows x cols output; words = [op, count, a1,
        a2, a3, ist, ioff, nsrc, ndst, nidx]."""
        if rows * cols and resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var di = self.device(idx)
            var ints: List[Int] = [da, di, self.matrices[out].device]
            for i in range(len(words)):  # small-loop(params: the ten move words): 10
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
            _ = idx  # keep the Python object alive through the native call above
            for t in range(width):  # driver-loop: over the selected column indices (matrix columns)
                p[t] = Float32(start + t)
                _ = idx  # keep the Python object alive through the native call above
            return self.move(a, idx, rows, width, [MV_TAKE_COLS, rows * width, width, cols, 0, 0, 0, rows * cols,
                                                   rows * width, width], True)
        var out = self.zeros(rows, width)
        if rows * width:
            tsqr_copy_columns(self.ptr(a), self.ptr(out), rows, cols, start, width)
            _ = out  # keep the Python object alive through the native call above
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
            for i in range(len(words)):  # small-loop(params: the ten move words): 10
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
        var _keep_ret_index_r363 = self.ptr(index)[0]
        _ = index  # keep the Python object alive through the native call above
        var _keep_ret_index = _keep_ret_index_r363^
        _ = index  # keep the Python object alive through the native call above
        return _keep_ret_index^

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
            _ = family_call(self._hot("x_decomp_dev_reduce"), PythonObject(da), PythonObject(self.matrices[maximum].device),
                            Python.list(PythonObject(n), PythonObject(1)))
            var v = self.word(maximum)
            if v != v:
                raise Error("x_decomp: a NaN has no order (refused)")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            _ = family_call(self._hot("x_decomp_dev_order_f"), PythonObject(da), PythonObject(da),
                            PythonObject(self.matrices[out].device), PythonObject(self.matrices[count].device),
                            Python.list(PythonObject(n), PythonObject(0), PythonObject(0)))
            return out
        var out = self.zeros(n, 1)
        if n:
            _ = family_call(self._hot("x_decomp_order_f"), self.address(a), PythonObject(n), self.address(out))
            _ = out  # keep the Python object alive through the native call above
        return out

    def order_small(mut self, a: Int) raises -> Int:
        """`_Kit.order_small(A)` (ties to the lower index, NaN last)."""
        var n = self.n(a)
        if n and self.resident:
            var out = self.device_output(n, 1)
            var da = self.device(a)
            _ = family_call(self._hot("x_decomp_dev_order_small"), PythonObject(da),
                            PythonObject(self.matrices[out].device), Python.list(PythonObject(n)))
            return out
        var out = self.zeros(n, 1)
        if n:
            _ = family_call(self._hot("x_decomp_order_small"), self.address(a), self.address(out), Python.list(PythonObject(n)))
            _ = out  # keep the Python object alive through the native call above
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
                    for i in range(len(words)):  # small-loop(params: the ten move words): 10
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
            memcpy(dest=self.ptr(out), src=self.ptr(a), count=rows * cols)
            _ = out  # keep the Python object alive through the native call above
        return out

    def geqrf(mut self, a: Int, mut tau: Int) raises -> Int:
        """`_Kit.geqrf(A)`: (h, tau), LAPACK geqrf's factored form."""
        var h = self.copy(a)
        var kk = min(self.r(a), self.c(a))
        tau = self.zeros(1, kk)
        _ = family_call(self._hot("x_decomp_geqrf"), self.address(h), self.address(tau),
                        Python.list(PythonObject(self.r(a)), PythonObject(self.c(a))))
        _ = h  # keep the Python object alive through the native call above
        return h

    def orgqr(mut self, h: Int, tau: Int, qc: Int) raises -> Int:
        """`_Kit.orgqr(h, tau, qc)`."""
        var q = self.zeros(self.r(h), qc)
        _ = family_call(self._hot("x_decomp_orgqr"), self.address(h), self.address(tau), self.address(q),
                        Python.list(PythonObject(self.r(h)), PythonObject(self.c(h)), PythonObject(self.c(tau)), PythonObject(qc)))
        _ = q  # keep the Python object alive through the native call above
        return q

    def eigh(mut self, a: Int, uplo: Int, mut v: Int) raises -> Int:
        """`_Kit.eigh(A, uplo)`: (w 1 x n ascending, v n x n columns)."""
        var n = self.r(a)
        var w = self.zeros(1, n)
        v = self.zeros(n, n)
        var params = Python.list(PythonObject(n), PythonObject(uplo))
        # TOMBSTONE: the device-resident x_decomp_dev_eigh branch (dead code: both callers,
        # linalg_ops_decomp `_lin_eigh` and `svd(hermitian=True)`, pass a freshly borrowed host
        # matrix, so `on_device(a)` was never true) deleted 2026-11-09 by __SHA__; code recoverable
        # at __SHA__^.
        # Restore: git apply experiments/removed/dead_code_p5_20261109.patch
        _ = family_call(self._hot("x_decomp_eigh"), self.address(a), self.address(w), self.address(v), params)
        _ = w  # keep the Python object alive through the native call above
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
            _ = family_call(self._hot("x_decomp_dev_svd"), PythonObject(da), self.address(s), self.address(v), params)
            _ = v  # keep the Python object alive through the native call above
            _ = s  # keep the Python object alive through the native call above
        else:
            _ = family_call(self._hot("x_decomp_svd"), self.address(a), self.address(s), self.address(v), params)
            _ = v  # keep the Python object alive through the native call above
            _ = s  # keep the Python object alive through the native call above
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
            _ = family_call(self._hot("x_decomp_dev_orth_diag"), PythonObject(da), PythonObject(self.matrices[q].device),
                            self.address(diag), params)
            return q
        var q = self.copy(a)
        _ = family_call(self._hot("x_decomp_orth_diag"), self.address(q), self.address(diag), params)
        _ = q  # keep the Python object alive through the native call above
        return q

    def eye(mut self, rows: Int, cols: Int) raises -> Int:
        var out = self.zeros(rows, cols)
        tsqr_fill_eye(self.ptr(out), rows, cols)
        _ = out  # keep the Python object alive through the native call above
        return out
