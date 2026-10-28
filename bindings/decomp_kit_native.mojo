# SPDX-License-Identifier: Apache-2.0
# The native twin of python/mojolearn/_expansion_decomp.py's `_Kit` and `_M`
# for the native decomposition owners: the same x_decomp binding entries,
# with the same parameter words, in the same order, under the same route
# rules (a GPU binding, one with the resident `x_decomp_dev_ew`, runs every
# non-empty operation on the device, as `_Kit._use` with `_RES_MIN` 1 does;
# a CPU-only host binding takes the host-address entries). So the bits are
# the ones the Python kit produced.
#
# One difference is data movement only: a matrix keeps both its host and its
# device words once it has both (the Python `_M` dropped one and moved the
# same words again when it next needed them). The words are equal, so no
# value changes; every route `_Kit` chose by residency reads `on_device`,
# which follows the Python `_M` exactly; an operation that writes a matrix
# in place drops the copy that went stale (`wrote_device` / `wrote_host`).
from std.python import Python, PythonObject
from std.memory import memcpy
from providers.python.array_metadata_pyapi import ArrayMetadata
from bindings.py_balanced import py_call, py_get, py_type, py_contains
from bindings.classical_arrays import classical_array, classical_typed, classical_address, classical_host_f32, classical_kind
from bindings.classical_validate import classical_view
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from core.classical_tsqr_impl import tsqr_fill_constant, tsqr_fill_indices, tsqr_copy_columns, tsqr_fill_eye

comptime DecompF32 = MutPointer[Float32, MutUntrackedOrigin]

# x_decomp/cells.mojo elementwise codes (`_expansion_decomp._OP`).
comptime XD_ADD = 0
comptime XD_SUB = 1
comptime XD_MUL = 2
comptime XD_DIV = 3
comptime XD_MAXS = 5
comptime XD_SQRT = 7
comptime XD_SQ = 8
comptime XD_SCALE = 14
comptime XD_SOFT = 17
comptime XD_LOGS = 10
comptime XD_ABS = 13
comptime XD_ADDS = 22
comptime XD_GTS = 23
comptime XD_LE = 34
comptime XD_RECIP = 16
comptime XD_MINS = 19
comptime XD_SELECT = 35

# x_decomp/moves.mojo move codes.
comptime XD_TAKE_ROWS = 0
comptime XD_TRANSPOSE = 1
comptime XD_PLACE_COLS = 2
comptime XD_TAKE_COLS = 4
comptime XD_F32_INDEX_MAX = 1 << 24
comptime XD_I32_MAX = (1 << 31) - 1

comptime XD_SPARSE_ALGOS = "('lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold')"
comptime XD_RESHAPE_HINT = ". Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."


def _repr(value: PythonObject) raises -> String:
    return String(py_call(Python.import_module("builtins").repr, value))


def decomp_int(value: PythonObject) raises -> Int:
    """`int(value)`."""
    return Int(py=py_call(Python.import_module("builtins").int, value))


def decomp_float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=py_call(Python.import_module("builtins").float, value))


def decomp_seed(random_state: PythonObject) raises -> PythonObject:
    """`_expansion_decomp._seed_of`: None is 0; anything but an int refused."""
    var b = Python.import_module("builtins")
    if random_state is None:
        return PythonObject(0)
    if Bool(py=py_call(b.isinstance, random_state, b.bool)) or not Bool(py=py_call(b.isinstance, random_state, b.int)):
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    return random_state


def decomp_seed_word(seed: PythonObject) raises -> Int:
    """`int(seed) & 0xFFFFFFFF`, with Python's integer semantics."""
    var masked = py_call(Python.import_module("operator").and_, py_call(Python.import_module("builtins").int, seed),
                         PythonObject(4294967295))
    return Int(py=masked)


def decomp_is_a(model: PythonObject, cls: PythonObject) raises -> Bool:
    """The model's class MRO, read through type's own descriptor, holds cls."""
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        if mro[k] is cls:
            return True
    return False


def decomp_class_attr(model: PythonObject, name: String) raises -> PythonObject:
    """A class attribute (a fixed estimator constant such as `_deflation`),
    read from the class dictionaries along the MRO, None when absent."""
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[k])
        if py_contains(fields, PythonObject(name)):
            return fields[name]
    return PythonObject(None)


def decomp_mode(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_decomp._mode`: the estimator's numeric_mode, None the
    process default (the value the estimator records as numeric_mode_)."""
    if value is None:
        return context["backend_fields"]["_DEFAULT_MODE"]
    return value


def decomp_entries(mode_value: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected x_decomp entries of a numeric mode, refused as
    `_backend.binding` refuses the mode (or as it refused at connection)."""
    var b = Python.import_module("builtins")
    var mode = decomp_mode(mode_value, context)
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    var name = String(py=py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["decomp_bindings"][name]
    if py_type(entries) is b.str:
        raise Error(String(py=entries))
    return entries


def decomp_input(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_M.from_input`: a SciPy sparse X densified (exact), then a nonempty
    2-D input cast once to C-order float32 (the input itself when it is one),
    non-finite values refused, in `_buffer.as_f32_c`'s and `_M`'s words."""
    var value = source
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        value = host_sparse_dense(source, format, context)
    var view = classical_view(value, context, name)
    var nd = Int(py=view.ndim)
    var shape = _repr(view.shape)
    if nd != 2:
        raise Error("mojolearn: " + name + " must be 2-D, got " + String(nd) + "-D shape " + shape
                    + (String(XD_RESHAPE_HINT) if nd < 2 else String("")))
    var rows = Int(py=view.shape[0])
    var cols = Int(py=view.shape[1])
    if rows * cols == 0:
        var what = String("feature(s)") if rows > 0 else String("sample(s)")
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    var kind = classical_kind(view, context)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f":
        raise Error("TypeError: mojolearn: " + name + " has a non-numeric dtype")
    try:
        return classical_host_f32(view, 2, name, context)
    except error:
        if String(error) == "Input " + name + " contains NaN or infinity":
            raise Error(name + ": input must be finite; NaN/inf are unsupported")
        raise Error(String(error))


def decomp_shape(source: PythonObject, name: String, context: PythonObject) raises -> Tuple[Int, Int]:
    """The (rows, cols) `_buffer.as_f32_c(source, ndim=2)` accepts, with its
    refusals, and no conversion (SparseCoder.fit reads only the width)."""
    var view = classical_view(source, context, name)
    var nd = Int(py=view.ndim)
    var shape = _repr(view.shape)
    if nd != 2:
        raise Error("mojolearn: " + name + " must be 2-D, got " + String(nd) + "-D shape " + shape
                    + (String(XD_RESHAPE_HINT) if nd < 2 else String("")))
    var rows = Int(py=view.shape[0])
    var cols = Int(py=view.shape[1])
    if rows * cols == 0:
        var what = String("feature(s)") if rows > 0 else String("sample(s)")
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    return (rows, cols)


struct DecompMatrix(Copyable, Movable):
    var host: PythonObject   # a float32 Array of rows * cols words (at least one), or None
    var host_address: Int    # -1 until captured, otherwise operation-local native address
    var device: Int          # an x_decomp device buffer id, or -1
    var rows: Int
    var cols: Int
    # Whether the Python `_M` would be on the device (its `_d` set): every
    # route `_Kit` chose by residency reads this, never the buffers kept.
    var on_device: Bool

    def __init__(out self, host: PythonObject, device: Int, rows: Int, cols: Int, on_device: Bool):
        self.host = host
        self.host_address = -1
        self.device = device
        self.rows = rows
        self.cols = cols
        self.on_device = on_device


struct DecompKit(Movable):
    var entries: PythonObject
    var context: PythonObject   # the NumPy-free output context: every buffer is an Array
    var matrices: List[DecompMatrix]
    var allocations: List[Int]
    var input_owners: List[PythonObject]
    var resident: Bool
    var one_device: Int
    var one_host: Int

    def __init__(out self, entries: PythonObject, context: PythonObject) raises:
        self.entries = entries
        self.context = context
        self.matrices = List[DecompMatrix]()
        self.allocations = List[Int]()
        self.input_owners = List[PythonObject]()
        self.resident = py_get(entries, "x_decomp_dev_ew") is not None
        self.one_device = -1
        self.one_host = -1
        # `_M._one`: the unused broadcast operand, a 0.
        self.one_host = self.zeros(1, 1)

    def close(mut self) raises:
        # Views share ids, so only the allocation ledger frees buffers.
        for i in range(len(self.allocations) - 1, -1, -1):  # driver-loop: over native device allocation handles
            _ = py_call(self.entries["x_decomp_dev_free"], self.allocations[i])
        self.allocations.clear()

    def has(self, name: String) raises -> Bool:
        return py_get(self.entries, PythonObject(name)) is not None

    def flags(self, name: String) raises -> Int:
        # A binding's compiled-route word (`x_decomp_s_flags`, ...): 0 when
        # the binding has no such entry, as `_Kit.s_flags` and its siblings.
        var entry = py_get(self.entries, PythonObject(name))
        if entry is None:
            return 0
        return Int(py=py_call(entry))

    def opt_dev(self, name: String) raises -> Bool:
        # `_Kit._opt_dev`: an optional resident entry this binding exports.
        return self.resident and self.has(name)

    # ---- matrices
    def add(mut self, host: PythonObject, rows: Int, cols: Int, device: Int = -1,
            host_address: Int = -1) -> Int:
        var result = len(self.matrices)
        self.matrices.append(DecompMatrix(host, device, rows, cols, device >= 0))
        self.matrices[result].host_address = host_address
        return result

    def size(self, a: Int) -> Int:
        return self.matrices[a].rows * self.matrices[a].cols

    def nrows(self, a: Int) -> Int:
        return self.matrices[a].rows

    def ncols(self, a: Int) -> Int:
        return self.matrices[a].cols

    def zeros(mut self, rows: Int, cols: Int) raises -> Int:
        # One spare word gives an empty matrix a safe address, as `_M._one`.
        return self.add(classical_array(max(rows * cols, 1), 0, self.context), rows, cols)

    def input(mut self, array: PythonObject, rows: Int, cols: Int) -> Int:
        # A C-order float32 Array the caller converted (decomp_input): read,
        # never written (every operation writes a fresh output).
        return self.add(array, rows, cols)

    def input_snapshot(mut self, snapshot: ArrayMetadata, rows: Int, cols: Int) -> Int:
        # The provider admitted shape/dtype/address once. Preserve the original
        # storage owner independently of mutable caller Array slots.
        self.input_owners.append(snapshot.owner)
        var index = self.add(snapshot.keep, rows, cols)
        self.matrices[index].host_address = snapshot.source.addr
        return index

    def const(mut self, value: Float32, rows: Int = 1, cols: Int = 1) raises -> Int:
        # `_M.of([v] * (r * c), r, c)`: the value rounded once to float32.
        var out = self.zeros(rows, cols)
        tsqr_fill_constant(DecompF32(unsafe_from_address=Int(py=classical_address(self.matrices[out].host))),
                           rows * cols, value)
        return out

    def indices(mut self, start: Int, stop: Int) raises -> Int:
        # `array.array("f", range(start, stop))`: row or column numbers as exact floats.
        var out = self.zeros(stop - start, 1)
        tsqr_fill_indices(DecompF32(unsafe_from_address=Int(py=classical_address(self.matrices[out].host))),
                          start, stop)
        return out

    def device_output(mut self, rows: Int, cols: Int) raises -> Int:
        var handle = Int(py=py_call(self.entries["x_decomp_dev_alloc"], max(rows * cols, 1)))
        self.allocations.append(handle)
        return self.add(PythonObject(None), rows, cols, handle)

    def host(mut self, a: Int) raises -> PythonObject:
        # `_M.s`: a device-only matrix comes down once; as in `_M.s`, the
        # matrix then counts as a host matrix (its device words stay valid
        # and are reused by a later upload, the same words).
        self.matrices[a].on_device = False
        if self.matrices[a].host is None:
            var n = self.size(a)
            var result = classical_array(max(n, 1), 0, self.context)
            if n:
                _ = py_call(self.entries["x_decomp_dev_download"], self.matrices[a].device,
                            classical_address(result), n)
                _ = result  # keep the Python object alive through the native call above
            self.matrices[a].host = result
        return self.matrices[a].host

    def address(mut self, a: Int) raises -> PythonObject:
        var owner = self.host(a)
        if self.matrices[a].host_address < 0:
            self.matrices[a].host_address = Int(py=classical_address(owner))
        return PythonObject(self.matrices[a].host_address)

    def device(mut self, a: Int) raises -> Int:
        # `_Kit._did`: a host matrix goes up once.
        if self.matrices[a].device < 0:
            var n = self.size(a)
            var owner = self.host(a)
            var handle = Int(py=py_call(self.entries["x_decomp_dev_alloc"], max(n, 1)))
            self.allocations.append(handle)
            if n:
                _ = py_call(self.entries["x_decomp_dev_upload"], handle, self.address(a), n)
                _ = owner  # keep the Python object alive through the native call above
            self.matrices[a].device = handle
            _ = owner^
        self.matrices[a].on_device = True
        return self.matrices[a].device

    def dev_one(mut self) raises -> Int:
        # `_dev_one`: the unused operand on the device, uploaded once.
        if self.one_device < 0:
            var handle = Int(py=py_call(self.entries["x_decomp_dev_alloc"], 1))
            self.allocations.append(handle)
            _ = py_call(self.entries["x_decomp_dev_upload"], handle, classical_address(self.matrices[self.one_host].host), 1)
            self.one_device = handle
        return self.one_device

    def wrote_device(mut self, a: Int):
        # The device copy was written in place: the host words are stale.
        self.matrices[a].host = PythonObject(None)
        self.matrices[a].host_address = -1

    def wrote_host(mut self, a: Int):
        # The host copy was written in place: the device words are stale.
        self.matrices[a].device = -1
        self.matrices[a].on_device = False

    def use(self, a: Int, b: Int = -1, c: Int = -1) -> Bool:
        # `_Kit._use` with `_RES_MIN` 1: a GPU binding, and an operand on the
        # device or with at least one value.
        if not self.resident:
            return False
        var found = False
        for k in range(3):  # small-loop(operands): the three operand slots of one call
            var m = a if k == 0 else (b if k == 1 else c)
            if m < 0:
                continue
            if self.matrices[m].device >= 0 or self.size(m) >= 1:
                found = True
        return found

    def publish(mut self, a: Int) raises -> PythonObject:
        # `_M.out()`: a fresh (rows, cols) float32 Array of the matrix.
        var shape = Python.tuple(self.matrices[a].rows, self.matrices[a].cols)
        return self.out_shape(a, shape)

    def out_shape(mut self, a: Int, shape: PythonObject) raises -> PythonObject:
        var n = self.size(a)
        var result = classical_typed(shape, "<f4", self.context)
        if n:
            var source = self.host(a)
            memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(result))),
                   src=DecompF32(unsafe_from_address=Int(py=self.address(a))), count=n)
            _ = result  # keep the Python object alive through the native call above
            _ = source  # keep the Python object alive through the native call above
            _ = source^
        return result

    def copy_host(mut self, a: Int) raises -> Int:
        # `_M.copy()`: a host copy (a device-only matrix's download is the copy).
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var n = rows * cols
        self.matrices[a].on_device = False
        if self.matrices[a].host is None:
            var result = classical_array(max(n, 1), 0, self.context)
            if n:
                _ = py_call(self.entries["x_decomp_dev_download"], self.matrices[a].device,
                            classical_address(result), n)
                _ = result  # keep the Python object alive through the native call above
            return self.add(result, rows, cols)
        var out = self.zeros(rows, cols)
        if n:
            var source = self.host(a)
            memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(self.matrices[out].host))),
                   src=DecompF32(unsafe_from_address=Int(py=self.address(a))), count=n)
            _ = source  # keep the Python object alive through the native call above
            _ = source^
        return out

    # ---- elementwise
    def _mode_of(self, a: Int, x: Int) raises -> Int:
        if x < 0:
            return 3
        var ar = self.matrices[a].rows
        var ac = self.matrices[a].cols
        var xr = self.matrices[x].rows
        var xc = self.matrices[x].cols
        if xr == ar and xc == ac:
            return 0
        if xr == 1 and xc == ac:
            return 1
        if xc == 1 and xr == ar:
            return 2
        if xr == 1 and xc == 1:
            return 3
        raise Error("x_decomp: cannot broadcast " + String(xr) + "x" + String(xc) + " against "
                    + String(ar) + "x" + String(ac))

    def ew(mut self, op: Int, a: Int, b: Int = -1, c: Int = -1, scalar: Float64 = 0.0) raises -> Int:
        var bm = self._mode_of(a, b)
        var cm = self._mode_of(a, c)
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var nb = 1 if b < 0 else self.size(b)
        var nc = 1 if c < 0 else self.size(c)
        var params = Python.list(op, rows * cols, cols, nb, bm, nc, cm)
        if rows * cols and self.use(a, b, c):
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var db = self.dev_one() if b < 0 else self.device(b)
            var dc = self.dev_one() if c < 0 else self.device(c)
            _ = py_call(self.entries["x_decomp_dev_ew"], da, db, dc, self.matrices[out].device, params,
                        PythonObject(scalar))
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var aa = self.address(a)
            var one = self.one_host
            var ib = one if b < 0 else b
            var ic = one if c < 0 else c
            var ab = self.address(ib)
            _ = ib  # keep the Python object alive through the native call above
            var ac = self.address(ic)
            _ = ic  # keep the Python object alive through the native call above
            _ = py_call(self.entries["x_decomp_ew"], aa, ab, ac, self.address(out), params, PythonObject(scalar))
            _ = ic  # keep the Python object alive through the native call above
            _ = ib  # keep the Python object alive through the native call above
            _ = out  # keep the Python object alive through the native call above
        return out

    # ---- products and sums
    def mm(mut self, a: Int, b: Int, ta: Bool = False, tb: Bool = False) raises -> Int:
        var m = self.matrices[a].cols if ta else self.matrices[a].rows
        var k = self.matrices[a].rows if ta else self.matrices[a].cols
        var k2 = self.matrices[b].cols if tb else self.matrices[b].rows
        var n = self.matrices[b].rows if tb else self.matrices[b].cols
        if k != k2:
            raise Error("x_decomp: gemm inner dimensions " + String(k) + " and " + String(k2) + " differ")
        var params = Python.list(m, k, n, Int(ta), Int(tb))
        if m * n and m * k and k * n and self.use(a, b):
            var out = self.device_output(m, n)
            var da = self.device(a)
            var db = self.device(b)
            _ = py_call(self.entries["x_decomp_dev_gemm"], da, db, self.matrices[out].device, params)
            return out
        var out = self.zeros(m, n)
        if m * n:
            var aa = self.address(a)
            var ab = self.address(b)
            _ = py_call(self.entries["x_decomp_gemm"], aa, ab, self.address(out), params)
            _ = out  # keep the Python object alive through the native call above
        return out

    def _sum(mut self, a: Int, columns: Bool) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var nr = 1 if columns else rows
        var nc = cols if columns else 1
        var name = String("colsum") if columns else String("rowsum")
        if rows * cols and self.use(a):
            var out = self.device_output(nr, nc)
            var da = self.device(a)
            _ = py_call(self.entries["x_decomp_dev_" + name], da, self.matrices[out].device, Python.list(rows, cols))
            return out
        var out = self.zeros(nr, nc)
        var aa = self.address(a)
        _ = py_call(self.entries["x_decomp_" + name], aa, self.address(out), Python.list(rows, cols))
        _ = out  # keep the Python object alive through the native call above
        return out

    def colsum(mut self, a: Int) raises -> Int:
        return self._sum(a, True)

    def rowsum(mut self, a: Int) raises -> Int:
        return self._sum(a, False)

    def colmean(mut self, a: Int) raises -> Int:
        # `_Kit.colmean`: the column sums scaled by 1.0 / rows (binary64, rounded by the cell).
        var sums = self.colsum(a)
        return self.ew(XD_SCALE, sums, scalar=1.0 / Float64(self.matrices[a].rows))

    # ---- moves
    def _move(mut self, a: Int, idx: Int, rows: Int, cols: Int, params: PythonObject, resident: Bool) raises -> Int:
        if rows * cols and resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var di = self.device(idx)
            _ = py_call(self.entries["x_decomp_dev_move"], da, di, self.matrices[out].device, params)
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var aa = self.address(a)
            var ai = self.address(idx)
            _ = py_call(self.entries["x_decomp_move"], aa, ai, self.address(out), params)
            _ = out  # keep the Python object alive through the native call above
        return out

    def take_rows(mut self, a: Int, idx: Int, radd: Int = 0) raises -> Int:
        # `_Kit.take_rows(A, idx, radd=radd)`: rows idx[t] + radd of A.
        var m = self.size(idx)
        var cols = self.matrices[a].cols
        var count = m * cols
        var params = Python.list(XD_TAKE_ROWS, count, cols, radd, 0, 1, 0, self.size(a), count, m)
        return self._move(a, idx, m, cols, params, count > 0 and self.use(a, idx))

    def _moves_on_device(self, a: Int, count: Int) -> Bool:
        # `_res_moves`: a device matrix's move runs on the device.
        return (self.matrices[a].on_device and count > 0 and self.matrices[a].rows < XD_F32_INDEX_MAX
                and self.matrices[a].cols < XD_F32_INDEX_MAX)

    def rows_of(mut self, a: Int, start: Int, stop: Int) raises -> Int:
        # `_M.rows(a, b)`: gathered on the device, a slice of the host store otherwise.
        var cols = self.matrices[a].cols
        var count = (stop - start) * cols
        if 0 <= start and start < stop and stop <= self.matrices[a].rows and self._moves_on_device(a, count):
            var idx = self.indices(start, stop)
            var w = stop - start
            return self._move(a, idx, w, cols, Python.list(XD_TAKE_ROWS, count, cols, 0, 0, 1, 0, self.size(a), count, w), True)
        var lo = max(0, min(start, self.matrices[a].rows))
        var hi = max(lo, min(stop, self.matrices[a].rows))
        var out = self.zeros(hi - lo, cols)
        var source = self.host(a)      # `self.s[...]`: the store is read (A is a host matrix after)
        if (hi - lo) * cols:
            memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(self.matrices[out].host))),
                   src=DecompF32(unsafe_from_address=Int(py=self.address(a))) + lo * cols,
                   count=(hi - lo) * cols)
            _ = source  # keep the Python object alive through the native call above
        _ = source^
        return out

    def cols_of(mut self, a: Int, start: Int, stop: Int) raises -> Int:
        # `_M.cols(a, b)` (`take_cols(range(a, b))`).
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var width = stop - start
        if start < 0 or stop > cols:
            raise Error("x_decomp: column range " + String(start) + ".." + String(stop) + " exceeds "
                        + String(cols) + " columns")
        if width > 0 and self._moves_on_device(a, rows * width):
            var idx = self.indices(start, stop)
            return self._move(a, idx, rows, width, Python.list(XD_TAKE_COLS, rows * width, width, cols, 0, 0, 0,
                              rows * cols, rows * width, width), True)
        var out = self.zeros(rows, max(width, 0))
        if rows * width > 0:
            var source = self.host(a)
            tsqr_copy_columns(DecompF32(unsafe_from_address=Int(py=self.address(a))),
                              DecompF32(unsafe_from_address=Int(py=classical_address(self.matrices[out].host))),
                              rows, cols, start, width)
            _ = source  # keep the Python object alive through the native call above
            _ = source^
        return out

    def transpose(mut self, a: Int) raises -> Int:
        # `_M.T`.
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var n = rows * cols
        var vector = rows == 1 or cols == 1
        if not self.matrices[a].on_device and n and not vector and n < XD_I32_MAX and self.resident:
            _ = self.device(a)
        var d = self.matrices[a].device
        if vector:
            # The same words, a view: of the device words for a device
            # matrix, else of the host store.
            if self.matrices[a].on_device:
                var shared = self.matrices[a].host
                return self.add(shared, cols, rows, d, self.matrices[a].host_address)
            var store = self.host(a)
            var view = self.add(store, cols, rows, d, self.matrices[a].host_address)
            self.matrices[view].on_device = False
            return view
        if self.matrices[a].on_device and n < XD_I32_MAX:
            var out = self.device_output(cols, rows)
            _ = py_call(self.entries["x_decomp_dev_move"], d, d, self.matrices[out].device,
                        Python.list(XD_TRANSPOSE, n, rows, cols, 0, 0, 0, n, n, 1))
            return out
        var out = self.zeros(cols, rows)
        var aa = self.address(a)
        var one = self.one_host
        var ai = self.address(one)
        _ = one  # keep the Python object alive through the native call above
        var ao = self.address(out)
        _ = out  # keep the Python object alive through the native call above
        _ = py_call(self.entries["x_decomp_move"], aa, ai, ao,
                    Python.list(XD_TRANSPOSE, n, rows, cols, 0, 0, 0, n, n, 1))
        _ = one  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        return out

    def reshape(mut self, a: Int, rows: Int, cols: Int) raises -> Int:
        # `_M.reshape`: a view of the same words (of the host store for a host matrix).
        if self.matrices[a].on_device:
            var shared = self.matrices[a].host
            var handle = self.matrices[a].device
            return self.add(shared, rows, cols, handle, self.matrices[a].host_address)
        var store = self.host(a)
        var view = self.add(store, rows, cols, self.matrices[a].device, self.matrices[a].host_address)
        self.matrices[view].on_device = False
        return view

    def hstack(mut self, ms: List[Int]) raises -> Int:
        # `_Kit.hstack`: PLACE_COLS moves, copies only.
        var rows = self.matrices[ms[0]].rows
        var width = 0
        for i in range(len(ms)):  # driver-loop: over the stacked matrices
            width += self.matrices[ms[i]].cols
        var resident = self.resident and rows * width > 0
        var out = self.device_output(rows, width) if resident else self.zeros(rows, width)
        var offset = 0
        for i in range(len(ms)):  # driver-loop: over the stacked matrices
            var m = ms[i]
            var count = self.size(m)
            if count:
                var params = Python.list(XD_PLACE_COLS, count, self.matrices[m].cols, width, offset, 0, 0, count, rows * width, 1)
                if resident:
                    var dm = self.device(m)
                    var one = self.dev_one()
                    _ = py_call(self.entries["x_decomp_dev_move"], dm, one, self.matrices[out].device, params)
                else:
                    var am = self.address(m)
                    _ = m  # keep the Python object alive through the native call above
                    var one = self.one_host
                    var ai = self.address(one)
                    _ = one  # keep the Python object alive through the native call above
                    var ao = self.address(out)
                    _ = out  # keep the Python object alive through the native call above
                    _ = py_call(self.entries["x_decomp_move"], am, ai, ao, params)
                    _ = one  # keep the Python object alive through the native call above
                    _ = m  # keep the Python object alive through the native call above
                    _ = out  # keep the Python object alive through the native call above
            offset += self.matrices[m].cols
        return out

    def vstack(mut self, ms: List[Int]) raises -> Int:
        # `_Kit.vstack`: PLACE_COLS moves on the device, the stores joined on the host.
        var cols = self.matrices[ms[0]].cols
        var rows = 0
        var values = 0
        for i in range(len(ms)):  # driver-loop: over the stacked matrices
            rows += self.matrices[ms[i]].rows
            values += self.size(ms[i])
        if self.resident and values:
            var out = self.device_output(rows, cols)
            var offset = 0
            for i in range(len(ms)):  # driver-loop: over the stacked matrices
                var m = ms[i]
                var count = self.size(m)
                if count:
                    var dm = self.device(m)
                    var one = self.dev_one()
                    _ = py_call(self.entries["x_decomp_dev_move"], dm, one, self.matrices[out].device,
                                Python.list(XD_PLACE_COLS, count, cols, cols, offset * cols, 0, 0, count, rows * cols, 1))
                offset += self.matrices[m].rows
            return out
        var out = self.zeros(rows, cols)
        var at = 0
        for i in range(len(ms)):  # driver-loop: over the stacked matrices
            var count = self.size(ms[i])
            if count:
                var source = self.host(ms[i])
                memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(self.matrices[out].host))) + at,
                       src=DecompF32(unsafe_from_address=Int(py=self.address(ms[i]))), count=count)
                _ = source  # keep the Python object alive through the native call above
                _ = source^
            at += count
        return out

    def strided(mut self, a: Int, count: Int, stride: Int, off: Int = 0) raises -> Int:
        # `_Kit.strided`: A[off + t * stride] for t < count, 1 x count.
        var step = off + 1 if count == 1 else stride
        var idx = self.const(Float32(off))
        var n = self.size(a)
        return self._move(a, idx, 1, count, Python.list(XD_TAKE_COLS, count, 1, step, 0, 0, 0, n, count, 1),
                          count > 0 and self.use(a))

    def place_strided(mut self, d: Int, v: Int, stride: Int, off: Int = 0) raises -> Int:
        # `_Kit.place_strided`: D[off + t * stride] = V[t], in place.
        var count = self.size(v)
        if count == 0:
            return d
        var params = Python.list(XD_PLACE_COLS, count, 1, stride, off, 0, 0, count, self.size(d), 1)
        if self.use(d, v):
            var dv = self.device(v)
            var dd = self.device(d)
            var one = self.dev_one()
            _ = py_call(self.entries["x_decomp_dev_move"], dv, one, dd, params)
            self.wrote_device(d)
            return d
        var av = self.address(v)
        var ad = self.address(d)
        var one = self.one_host
        var ai = self.address(one)
        _ = one  # keep the Python object alive through the native call above
        _ = py_call(self.entries["x_decomp_move"], av, ai, ad, params)
        _ = one  # keep the Python object alive through the native call above
        self.wrote_host(d)
        return d

    def copy(mut self, a: Int) raises -> Int:
        # `_Kit.copy`: a device copy of a device matrix, else `_M.copy()`.
        var n = self.size(a)
        if n and self.matrices[a].on_device and self.use(a):
            var rows = self.matrices[a].rows
            var cols = self.matrices[a].cols
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var one = self.dev_one()
            _ = py_call(self.entries["x_decomp_dev_move"], da, one, self.matrices[out].device,
                        Python.list(XD_PLACE_COLS, n, cols, cols, 0, 0, 0, n, n, 1))
            return out
        return self.copy_host(a)

    def diag_add(mut self, a: Int, v: Int) raises -> Int:
        # `_Kit.diag_add`: a copy of square A with v added to its diagonal.
        var rows = self.matrices[a].rows
        var copied = self.copy(a)
        var diagonal = self.strided(a, rows, rows + 1, 0)
        var added = self.ew(XD_ADD, diagonal, v)
        return self.place_strided(copied, added, rows + 1, 0)

    # ---- orders and signs
    def word(mut self, a: Int, i: Int = 0) raises -> Float64:
        # `_Kit.word`: one value down.
        var index = a
        var at = i
        if self.matrices[a].on_device and self.size(a) > 1:
            index = self.strided(a, 1, 1, i)
            at = 0
        var owner = self.host(index)
        var value = Float64(DecompF32(unsafe_from_address=Int(py=self.address(index)))[at])
        _ = owner  # keep the Python object alive through the native call above
        _ = owner^
        return value

    def order(mut self, a: Int) raises -> Int:
        # `_Kit.order`: the stable ascending order, NaN refused.
        var n = self.size(a)
        if n and self.use(a):
            if n >= XD_F32_INDEX_MAX:
                raise Error("x_decomp: order_f exceeds the float32 index bound")
            var maximum = self.device_output(1, 1)
            var da = self.device(a)
            _ = py_call(self.entries["x_decomp_dev_reduce"], da, self.matrices[maximum].device, Python.list(n, 1))
            var v = self.word(maximum)
            if v != v:
                raise Error("x_decomp: a NaN has no order (refused)")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            _ = py_call(self.entries["x_decomp_dev_order_f"], da, da, self.matrices[out].device,
                        self.matrices[count].device, Python.list(n, 0, 0))
            return out
        var out = self.zeros(n, 1)
        if n:
            var aa = self.address(a)
            _ = py_call(self.entries["x_decomp_order_f"], aa, n, self.address(out))
            _ = out  # keep the Python object alive through the native call above
        return out

    def neg_signs(mut self, v: Int) raises -> Int:
        # `_Kit.neg_signs`: -1 where V < 0, else +1.
        var negated = self.ew(XD_SCALE, v, scalar=-1.0)
        var minus = self.const(-1.0)
        var plus = self.const(1.0)
        return self.ew(XD_SELECT, negated, minus, plus, 0.0)

    def absmax_signs(mut self, a: Int, by_col: Bool) raises -> Int:
        # `_Kit.absmax_signs`.
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var count = cols if by_col else rows
        var params = Python.list(rows, cols, 1 if by_col else 0)
        var out: Int
        if count and rows * cols and self.use(a):
            out = self.device_output(1, count)
            var da = self.device(a)
            _ = py_call(self.entries["x_decomp_dev_absmax"], da, self.matrices[out].device, params)
        else:
            out = self.zeros(1, count)
            if count and rows * cols:
                var aa = self.address(a)
                _ = py_call(self.entries["x_decomp_absmax_sign"], aa, self.address(out), params)
        return self.neg_signs(out if by_col else self.reshape(out, count, 1))

    def rand(mut self, rows: Int, cols: Int, seed: Int, stream: Int, kind: Int) raises -> Int:
        # `_Kit.rand` (seed already masked to 32 bits).
        var params = Python.list(rows * cols, seed, stream, kind)
        if rows * cols >= 1024 and self.resident:
            var out = self.device_output(rows, cols)
            _ = py_call(self.entries["x_decomp_dev_rand"], self.matrices[out].device, params)
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            _ = py_call(self.entries["x_decomp_rand"], self.address(out), params)
            _ = out  # keep the Python object alive through the native call above
        return out

    # ---- small dense linear algebra
    def svd(mut self, a: Int) raises -> Tuple[Int, Int]:
        # `_Kit.svd`: (S 1 x n descending, Vt n x n) of a tall A.
        var m = self.matrices[a].rows
        var n = self.matrices[a].cols
        var s = self.zeros(1, n)
        var v = self.zeros(n, n)
        if m >= n and n >= 1 and self.opt_dev("x_decomp_dev_svd") and self.use(a):
            var da = self.device(a)
            _ = py_call(self.entries["x_decomp_dev_svd"], da, self.address(s), self.address(v), Python.list(m, n))
            _ = v  # keep the Python object alive through the native call above
            _ = s  # keep the Python object alive through the native call above
        else:
            var aa = self.address(a)
            _ = py_call(self.entries["x_decomp_svd"], aa, self.address(s), self.address(v), Python.list(m, n))
            _ = v  # keep the Python object alive through the native call above
            _ = s  # keep the Python object alive through the native call above
        var negated = self.ew(XD_SCALE, s, scalar=-1.0)
        var o = self.order(negated)
        var st = self.transpose(s)
        var sorted = self.take_rows(st, o)
        var s_desc = self.transpose(sorted)
        var vt = self.transpose(v)
        return (s_desc, self.take_rows(vt, o))

    def thin_svd(mut self, x: Int, nc: Int, u_based: Bool) raises -> Tuple[Int, Int, Int]:
        # `_thin_svd`: (U n x nc, S 1 x nc, Vt nc x d) with svd_flip.
        var n = self.matrices[x].rows
        var d = self.matrices[x].cols
        var u: Int
        var s: Int
        var vt: Int
        if d <= n:
            var sv = self.svd(x)
            s = self.cols_of(sv[0], 0, nc)
            vt = self.rows_of(sv[1], 0, nc)
            var xv = self.mm(x, vt, tb=True)
            u = self.ew(XD_DIV, xv, s)
        else:
            var xt = self.transpose(x)
            var su = self.svd(xt)
            s = self.cols_of(su[0], 0, nc)
            var ut = self.rows_of(su[1], 0, nc)
            u = self.transpose(ut)
            var ux = self.mm(u, x, ta=True)
            var st = self.transpose(s)
            vt = self.ew(XD_DIV, ux, st)
        if u_based:
            var sg = self.absmax_signs(u, True)
            var sgt = self.transpose(sg)
            var u2 = self.ew(XD_MUL, u, sg)
            var vt2 = self.ew(XD_MUL, vt, sgt)
            return (u2, s, vt2)
        var sg = self.absmax_signs(vt, False)
        var sgt = self.transpose(sg)
        var u2 = self.ew(XD_MUL, u, sgt)
        var vt2 = self.ew(XD_MUL, vt, sg)
        return (u2, s, vt2)

    def lu(mut self, a: Int) raises -> Tuple[Int, PythonObject, Int]:
        # `_Kit.lu`: (the factor, its int32 pivots, info).
        var n = self.matrices[a].rows
        var piv = py_call(self.context["array_type"], PythonObject("i"), py_call(self.context["native_bytes_type"], 4 * max(n, 1)))
        var info = self.zeros(1, 1)
        var pa = py_call(piv.buffer_info)[0]
        var lu: Int
        if n >= 1 and self.matrices[a].cols == n and self.opt_dev("x_decomp_dev_lu") and self.use(a):
            lu = self.device_output(n, n)
            var da = self.device(a)
            var info_address = self.address(info)
            _ = info  # keep the Python object alive through the native call above
            var lu_device = self.matrices[lu].device
            _ = py_call(self.entries["x_decomp_dev_lu"], da, lu_device, pa, info_address, Python.list(n))
            _ = info  # keep the Python object alive through the native call above
        else:
            lu = self.copy_host(a)
            _ = py_call(self.entries["x_decomp_lu"], self.address(lu), pa, self.address(info), Python.list(n))
            _ = info  # keep the Python object alive through the native call above
            self.wrote_host(lu)
        return (lu, piv, Int(self.word(info)))

    def lu_resid(mut self, a: Int, x: Int, b: Int) raises -> Int:
        # `_Kit.lu_resid`: B - A X, folded in float-float on the device.
        var out = self.zeros(self.matrices[b].rows, self.matrices[b].cols)
        var aa = self.address(a)
        var ax = self.address(x)
        var ab = self.address(b)
        _ = py_call(self.entries["x_decomp_lu_resid"], aa, ax, ab, self.address(out),
                    Python.list(self.matrices[a].rows, self.matrices[b].cols))
        _ = out  # keep the Python object alive through the native call above
        return out

    def lu_solve(mut self, lu: Int, piv: PythonObject, b: Int) raises -> Int:
        # `_Kit.lu_solve(lu, piv, B)` (trans 0): on a host copy of B.
        var out = self.copy_host(b)
        var al = self.address(lu)
        _ = py_call(self.entries["x_decomp_lu_solve"], al, py_call(piv.buffer_info)[0], self.address(out),
                    Python.list(self.matrices[lu].rows, self.matrices[b].cols))
        _ = out  # keep the Python object alive through the native call above
        self.wrote_host(out)
        return out

    # ---- sparse coding rows
    def lasso_rows(mut self, g: Int, q: Int, w: Int, alpha: Float64, max_iter: Int, tol: Float64, positive: Int) raises -> Int:
        # `_Kit.lasso_rows`: W the warm start, updated in place.
        var rows = self.matrices[q].rows
        var cols = self.matrices[q].cols
        if rows * cols and self.opt_dev("x_decomp_dev_code_rows") and self.use(g, q, w):
            var dg = self.device(g)
            var dq = self.device(q)
            var dw = self.device(w)
            _ = py_call(self.entries["x_decomp_dev_code_rows"], dg, dq, dw,
                        Python.list(0, rows, cols, max_iter, positive), Python.list(alpha, tol))
            self.wrote_device(w)
            return w
        var its = self.zeros(rows, 1)
        var ag = self.address(g)
        var aq = self.address(q)
        var aw = self.address(w)
        _ = py_call(self.entries["x_decomp_lasso_rows"], ag, aq, aw, self.address(its),
                    Python.list(rows, cols, max_iter, positive), Python.list(alpha, tol))
        _ = its  # keep the Python object alive through the native call above
        self.wrote_host(w)
        return w

    def lars_rows(mut self, g: Int, q: Int, m: Int, nnz: Int) raises -> Int:
        # `_Kit.lars_rows`.
        var rows = self.matrices[q].rows
        var cols = self.matrices[q].cols
        if rows * cols and self.opt_dev("x_decomp_dev_code_rows") and self.use(g, q):
            var w = self.device_output(rows, cols)
            var dg = self.device(g)
            var dq = self.device(q)
            _ = py_call(self.entries["x_decomp_dev_code_rows"], dg, dq, self.matrices[w].device,
                        Python.list(1, rows, cols, m, nnz), Python.list(Float64(0.0), Float64(0.0)))
            return w
        var w = self.zeros(rows, cols)
        var na = self.zeros(rows, 1)
        if rows * cols:
            var ag = self.address(g)
            var aq = self.address(q)
            _ = py_call(self.entries["x_decomp_lars_rows"], ag, aq, self.address(w), self.address(na),
                        Python.list(rows, cols, m, nnz))
            _ = na  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
        return w

    def omp_rows(mut self, g: Int, q: Int, nnz: Int) raises -> Int:
        # `_Kit.omp_rows`.
        var rows = self.matrices[q].rows
        var cols = self.matrices[q].cols
        if rows * cols and self.opt_dev("x_decomp_dev_code_rows") and self.use(g, q):
            var w = self.device_output(rows, cols)
            var dg = self.device(g)
            var dq = self.device(q)
            _ = py_call(self.entries["x_decomp_dev_code_rows"], dg, dq, self.matrices[w].device,
                        Python.list(2, rows, cols, nnz, 0), Python.list(Float64(0.0), Float64(0.0)))
            return w
        var w = self.zeros(rows, cols)
        var na = self.zeros(rows, 1)
        var ag = self.address(g)
        var aq = self.address(q)
        _ = py_call(self.entries["x_decomp_omp_rows"], ag, aq, self.address(w), self.address(na),
                    Python.list(rows, cols, nnz))
        _ = na  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        return w

    # ---- reductions, selections and factor companions (MinCovDet)
    def reduce(mut self, a: Int, op: Int) raises -> Int:
        # `_Kit.reduce`: max |A| (0), max A (1) or min A (2) as 1 x 1.
        var n = self.size(a)
        if n == 0:
            raise Error("x_decomp: reduce of an empty matrix")
        if self.use(a):
            var out = self.device_output(1, 1)
            var da = self.device(a)
            _ = py_call(self.entries["x_decomp_dev_reduce"], da, self.matrices[out].device, Python.list(n, op))
            return out
        var out = self.zeros(1, 1)
        var aa = self.address(a)
        _ = py_call(self.entries["x_decomp_reduce"], aa, self.address(out), Python.list(n, op))
        _ = out  # keep the Python object alive through the native call above
        return out

    def total_word(mut self, a: Int) raises -> Float64:
        # `k.total(A).s[0]`: the sum of every entry, rows ascending, one word down.
        var rows = self.rowsum(a)
        var total = self.colsum(rows)
        return self.word(total)

    def eigh(mut self, a: Int) raises -> Tuple[Int, Int]:
        # `_Kit.eigh(A)` (uplo 0): ascending values (1 x n), vectors in columns.
        var n = self.matrices[a].rows
        var w = self.zeros(1, n)
        var v = self.zeros(n, n)
        var params = Python.list(n, 0)
        if n >= 1 and self.matrices[a].cols == n and self.matrices[a].on_device and self.opt_dev("x_decomp_dev_eigh"):
            var da = self.matrices[a].device
            _ = py_call(self.entries["x_decomp_dev_eigh"], da, self.address(w), self.address(v), params)
            _ = v  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
            return (w, v)
        var aa = self.address(a)
        _ = py_call(self.entries["x_decomp_eigh"], aa, self.address(w), self.address(v), params)
        _ = v  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        return (w, v)

    def refuse_nan(mut self, a: Int) raises:
        # `_Kit._refuse_nan`: a device operand's max is NaN when any value is.
        var maximum = self.reduce(a, 1)
        var v = self.word(maximum)
        if v != v:
            raise Error("x_decomp: a NaN has no order (refused)")

    def argmin_all(mut self, a: Int) raises -> Int:
        # `_Kit.argmin_all`: every position (exact floats, c x 1) of the minimum.
        var n = self.size(a)
        if n and self.use(a):
            if n >= XD_F32_INDEX_MAX:
                raise Error("x_decomp: argmin_all exceeds the float32 index bound")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            var da = self.device(a)
            _ = py_call(self.entries["x_decomp_dev_argmin_all"], da, self.matrices[out].device,
                        self.matrices[count].device, Python.list(n))
            var c = Int(self.word(count))
            return self.rows_of(out, 0, c) if c else self.zeros(0, 1)
        var buffer = self.zeros(max(n, 1), 1)
        var c = 0
        if n:
            var aa = self.address(a)
            c = Int(py=py_call(self.entries["x_decomp_argmin_all"], aa, n, self.address(buffer)))
            _ = buffer  # keep the Python object alive through the native call above
        return self.rows_of(buffer, 0, c)

    def select_smallest(mut self, a: Int, h: Int) raises -> Tuple[Int, Int, PythonObject]:
        # `_Kit.select_smallest`: (the h smallest positions ascending, h x 1;
        # the membership as an n x 1 0/1 device matrix, or -1 and the host
        # int32 membership store).
        var n = self.size(a)
        if n and self.use(a):
            if n >= XD_F32_INDEX_MAX or h < 0 or h > n:
                raise Error("x_decomp: select_smallest out of range")
            self.refuse_nan(a)
            var sel = self.device_output(h, 1)
            var mask = self.device_output(n, 1)
            var da = self.device(a)
            _ = py_call(self.entries["x_decomp_dev_select_smallest"], da, self.matrices[sel].device,
                        self.matrices[mask].device, Python.list(n, h))
            return (sel, mask, PythonObject(None))
        var sel = self.zeros(h, 1)
        var store = py_call(self.context["array_type"], PythonObject("i"), py_call(self.context["native_bytes_type"], 4 * max(n, 1)))
        if n:
            var aa = self.address(a)
            _ = py_call(self.entries["x_decomp_select_smallest"], aa, Python.list(n, h), self.address(sel),
                        py_call(store.buffer_info)[0])
            _ = sel  # keep the Python object alive through the native call above
        return (sel, Int(-1), store)

    def lu_aux(mut self, lu: Int, piv: PythonObject) raises -> Tuple[Int, Int]:
        # `_Kit.lu_aux(lu, piv)` (clamp False): (stats 1 x 4 = [max |u_ii|,
        # zero pivots, negative pivots, swaps], the diagonal 1 x n), host.
        var n = self.matrices[lu].rows
        var pm = self.zeros(n, 1)
        var im = self.zeros(n, 1)
        var diag = self.zeros(1, n)
        var st = self.zeros(1, 4)
        var pa = py_call(piv.buffer_info)[0]
        var params = Python.list(n, 0)
        if n >= 1 and self.matrices[lu].on_device and self.opt_dev("x_decomp_dev_lu"):
            var dl = self.matrices[lu].device
            _ = py_call(self.entries["x_decomp_dev_lu_aux"], dl, pa, self.address(pm), self.address(im),
                        self.address(diag), self.address(st), params)
            _ = im  # keep the Python object alive through the native call above
            _ = pm  # keep the Python object alive through the native call above
            _ = st  # keep the Python object alive through the native call above
            _ = diag  # keep the Python object alive through the native call above
            return (st, diag)
        var al = self.address(lu)
        _ = py_call(self.entries["x_decomp_lu_aux"], al, pa, self.address(pm), self.address(im),
                    self.address(diag), self.address(st), params)
        _ = im  # keep the Python object alive through the native call above
        _ = pm  # keep the Python object alive through the native call above
        _ = st  # keep the Python object alive through the native call above
        _ = diag  # keep the Python object alive through the native call above
        return (st, diag)

    # ---- top-N orders (AlternatingLeastSquares)
    def order_dev(mut self, a: Int, neg: Int, skip: Int) raises -> Tuple[Int, Int]:
        # `_Kit._order_dev(A, neg, skip)` on a resident kit: (n x 1 order, 1 x 1 count).
        var n = self.size(a)
        if n >= XD_F32_INDEX_MAX:
            raise Error("x_decomp: order_f exceeds the float32 index bound")
        self.refuse_nan(a)
        var out = self.device_output(n, 1)
        var count = self.device_output(1, 1)
        var sid = self.device(skip) if skip >= 0 else self.device(a)
        var da = self.device(a)
        _ = py_call(self.entries["x_decomp_dev_order_f"], da, sid, self.matrices[out].device,
                    self.matrices[count].device, Python.list(n, neg, 1 if skip >= 0 else 0))
        return (out, count)

    def take_cols_m(mut self, a: Int, idx: Int, w: Int) raises -> Int:
        # `_Kit.take_cols_m`: columns idx[0 .. w) of A.
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var params = Python.list(XD_TAKE_COLS, rows * w, w, cols, 0, 0, 0, rows * cols, rows * w, w)
        return self._move(a, idx, rows, w, params, rows * w > 0 and self.use(a, idx))

    # ---- orthonormal bases and small orders (randomized_svd)
    def orth(mut self, a: Int) raises -> Int:
        # `_Kit.orth`: a copy of A with its columns orthonormalized.
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        if rows * cols and self.use(a):
            var q = self.device_output(rows, cols)
            var da = self.device(a)
            var name = String("x_decomp_dev_orth_ws") if (self.flags("x_decomp_s_flags") & 2) != 0 else String("x_decomp_dev_orth")
            _ = py_call(self.entries[name], da, self.matrices[q].device, Python.list(rows, cols))
            return q
        var q = self.copy_host(a)
        _ = py_call(self.entries["x_decomp_orth"], self.address(q), Python.list(rows, cols))
        _ = q  # keep the Python object alive through the native call above
        self.wrote_host(q)
        return q

    def order_small(mut self, a: Int) raises -> Int:
        # `_Kit.order_small`: the stable ascending order as n x 1 exact floats.
        var n = self.size(a)
        if n and self.use(a):
            var out = self.device_output(n, 1)
            var da = self.device(a)
            _ = py_call(self.entries["x_decomp_dev_order_small"], da, self.matrices[out].device, Python.list(n))
            return out
        var out = self.zeros(n, 1)
        if n:
            var aa = self.address(a)
            _ = py_call(self.entries["x_decomp_order_small"], aa, self.address(out), Python.list(n))
            _ = out  # keep the Python object alive through the native call above
        return out

    def count_gt(mut self, a: Int, s: Float64) raises -> Int:
        # `_Kit.count_gt`: how many values of A are > s.
        if not self.size(a):
            return 0
        var flags = self.ew(XD_GTS, a, scalar=s)
        return Int(self.total_word(flags))

    def grown(mut self, a: Int, rows: Int, cols: Int) raises -> Int:
        # `_M(A.s + zeros, rows, cols)`: A's words then zeros, as a host matrix.
        var out = self.zeros(rows, cols)
        var n = self.size(a)
        if n:
            var source = self.host(a)
            memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(self.matrices[out].host))),
                   src=DecompF32(unsafe_from_address=Int(py=self.address(a))), count=n)
            _ = source  # keep the Python object alive through the native call above
            _ = source^
        return out

    # ---- the tall SVD of `_linalg_impl._svd_tall` (lstsq)
    def orth_diag(mut self, a: Int) raises -> Tuple[Int, Int]:
        # `_Kit.orth_diag`: (orth(A), the passes' R-diagonal products 1 x c).
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var diag = self.zeros(1, cols)
        var params = Python.list(rows, cols)
        if rows * cols and self.use(a):
            var q = self.device_output(rows, cols)
            var da = self.device(a)
            var diag_address = self.address(diag)
            _ = diag  # keep the Python object alive through the native call above
            var q_device = self.matrices[q].device
            _ = py_call(self.entries["x_decomp_dev_orth_diag"], da, q_device, diag_address, params)
            _ = diag  # keep the Python object alive through the native call above
            return (q, diag)
        var q = self.copy_host(a)
        _ = py_call(self.entries["x_decomp_orth_diag"], self.address(q), self.address(diag), params)
        _ = q  # keep the Python object alive through the native call above
        _ = diag  # keep the Python object alive through the native call above
        self.wrote_host(q)
        return (q, diag)

    def svd_tall(mut self, a: Int) raises -> Tuple[Int, Int, Int]:
        # `_linalg_impl._svd_tall(k, A, full=False)`: (U m x n, S 1 x n, Vt n x n).
        var m = self.matrices[a].rows
        var n = self.matrices[a].cols
        var sv = self.svd(a)
        var s = sv[0]
        var vt = sv[1]
        var r = 0
        if n:
            _ = self.word(s, 0)      # `S.s[0]`: S comes to the host, as before
            var sa = self.address(s)
            _ = s  # keep the Python object alive through the native call above
            r = Int(py=py_call(self.entries["x_decomp_rank_above"], sa, Python.list(n), PythonObject(Float64(0.00000095367431640625))))
            _ = s  # keep the Python object alive through the native call above
        var av = self.mm(a, vt, tb=True)
        var left = self.cols_of(av, 0, r) if r < n else av
        var right = self.cols_of(s, 0, r) if r < n else s
        var ug = self.ew(XD_DIV, left, right)
        if r == n:
            var qd = self.orth_diag(ug)
            var absolute = self.ew(XD_ABS, qd[1])
            var zero = self.zeros(1, 1)
            var guarded = self.ew(XD_LE, absolute, zero)
            if self.total_word(guarded) == 0:
                var signs = self.neg_signs(qd[1])
                return (self.ew(XD_MUL, qd[0], signs), s, vt)
        if r == 0:
            var eye = self.zeros(m, n)
            tsqr_fill_eye(DecompF32(unsafe_from_address=Int(py=classical_address(self.matrices[eye].host))), m, n)
            return (eye, s, vt)
        # geqrf + orgqr of A V / s, Q's columns signed by R's diagonal.
        var h = self.copy_host(ug)
        var tau = self.zeros(1, min(self.matrices[ug].rows, self.matrices[ug].cols))
        _ = py_call(self.entries["x_decomp_geqrf"], self.address(h), self.address(tau),
                    Python.list(self.matrices[ug].rows, self.matrices[ug].cols))
        _ = tau  # keep the Python object alive through the native call above
        _ = h  # keep the Python object alive through the native call above
        self.wrote_host(h)
        var qc = self.zeros(self.matrices[h].rows, n)
        var ah = self.address(h)
        _ = h  # keep the Python object alive through the native call above
        var at = self.address(tau)
        _ = tau  # keep the Python object alive through the native call above
        _ = py_call(self.entries["x_decomp_orgqr"], ah, at, self.address(qc),
                    Python.list(self.matrices[h].rows, self.matrices[h].cols, self.matrices[tau].cols, n))
        _ = qc  # keep the Python object alive through the native call above
        _ = tau  # keep the Python object alive through the native call above
        _ = h  # keep the Python object alive through the native call above
        self.wrote_host(qc)
        var diagonal = self.strided(h, r, r + 1, 0)
        var sg = self.neg_signs(diagonal)
        if n > r:
            var ones = self.const(1.0, 1, n - r)
            sg = self.hstack([sg, ones])
        return (self.ew(XD_MUL, qc, sg), s, vt)
