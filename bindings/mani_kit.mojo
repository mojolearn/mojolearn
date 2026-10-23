# SPDX-License-Identifier: Apache-2.0
# The decomp kit (`_expansion_decomp._Kit` and its `_M` matrices) for the
# compiled manifold owners (bindings/mani_*_native.mojo): the same x_decomp
# entries, called with the same operands, broadcast modes, float32 scalars and
# parameter lists, in the same order, so every word is the one the Python kit
# produced.
#
# The matrices ARE the Python kit's `_M` objects and their device buffers its
# `_DevBuf` objects, assembled from their slots (no Python runs): a matrix
# keeps its host store (`_s`, an `array.array('f')`) or its device buffer
# (`_d`), moves to the device when an entry needs it there (`mk_did`) and back
# when a host-address entry reads it (`mk_store`), exactly as `_M.s` and
# `_Kit._did` move it. Views share the store or the buffer as `reshape` and a
# vector's `T` shared them, and a buffer returns to the binding's pool when its
# last holder dies (`_DevBuf.__del__`), as before. The kit's choices (resident
# entries on a GPU binding, `_RES_MIN` 1, the optional resident entries, the
# Apple FAST precise LLE entries) are the Python kit's own tests.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setattr, py_setitem
from bindings.py_attr import py_attr
from bindings.native_buffer_owner import native_storage_allocate
from bindings.classical_arrays import classical_address, classical_wrap_array
from bindings.family_common import FAMILY_F32P, FAMILY_F64P, family_call, family_entry, family_as_f32, family_as_f32_dense
from core.family_cast_impl import family_all_finite_f32

# x_decomp/cells.mojo op codes (`_OP` in _expansion_decomp.py)
comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_MAXS = 5
comptime OP_SQRT = 7
comptime OP_SQ = 8
comptime OP_ABS = 13
comptime OP_SCALE = 14
comptime OP_FMA = 15
comptime OP_RECIP = 16
comptime OP_ADDS = 22
comptime OP_GTS = 23
comptime OP_MAX = 30
comptime OP_MIN = 31
comptime OP_LE = 34
comptime OP_SELECT = 35

# x_decomp/moves.mojo move ops and bounds (`_MV_*` in _expansion_decomp.py)
comptime MV_TAKE_ROWS = 0
comptime MV_TRANSPOSE = 1
comptime MV_PLACE_COLS = 2
comptime MV_FILL0 = 3
comptime MV_TAKE_COLS = 4
comptime F32_INDEX_MAX = 1 << 24
comptime I32_MAX = (1 << 31) - 1

# `_Kit._SEL_*` (x_decomp/select_ops.mojo)
comptime SEL_MAXABS = 0
comptime SEL_MAX = 1
comptime SEL_MIN = 2


def mk_ints(values: List[Int]) raises -> PythonObject:
    """A Python list of ints (an entry's integer parameter list)."""
    var out = Python.list()
    for i in range(len(values)):  # small-loop(params: one entry's parameter list): bounded by the call site
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def mk_append(items: PythonObject, value: PythonObject) raises:
    _ = py_call(items.append, value)


def mk_store_zeros(n: Int, ctx: PythonObject) raises -> PythonObject:
    """`array.array('f', [0.0]) * n`: n zero float32 words."""
    return py_call(ctx["array_type"], PythonObject("f"), py_call(ctx["native_bytes_type"], PythonObject(n * 4)))


def mk_store_len(store: PythonObject) raises -> Int:
    return Int(py=py_call(Python.import_module("builtins").len, store))


def mk_store_addr(store: PythonObject) raises -> PythonObject:
    """`store.buffer_info()[0]`."""
    return py_call(store.buffer_info)[0]


def mk_store_ptr(store: PythonObject) raises -> FAMILY_F32P:
    return FAMILY_F32P(unsafe_from_address=Int(py=mk_store_addr(store)))


def mk_store_copy(store: PythonObject, ctx: PythonObject) raises -> PythonObject:
    """`array.array('f', store)`: an exact copy."""
    return py_call(ctx["array_type"], PythonObject("f"), store)


def mk_r(m: PythonObject) raises -> Int:
    return Int(py=m.r)


def mk_c(m: PythonObject) raises -> Int:
    return Int(py=m.c)


def mk_n(m: PythonObject) raises -> Int:
    return Int(py=m.r) * Int(py=m.c)


def mk_on_dev(m: PythonObject) raises -> Bool:
    return not (m._d is None)


def mk_new(store: PythonObject, r: Int, c: Int, ctx: PythonObject) raises -> PythonObject:
    """`_M(store, r, c)`, assembled from its slots."""
    if mk_store_len(store) != r * c:
        raise Error("x_decomp: matrix store does not match its shape")
    var m = native_storage_allocate(ctx["family_mani_M"])
    py_setattr(m, "_s", store)
    py_setattr(m, "r", PythonObject(r))
    py_setattr(m, "c", PythonObject(c))
    py_setattr(m, "_d", PythonObject(None))
    return m


def mk_dev_view(d: PythonObject, r: Int, c: Int, ctx: PythonObject) raises -> PythonObject:
    """`_M._on_device(d, r, c)`."""
    var m = native_storage_allocate(ctx["family_mani_M"])
    py_setattr(m, "_s", PythonObject(None))
    py_setattr(m, "r", PythonObject(r))
    py_setattr(m, "c", PythonObject(c))
    py_setattr(m, "_d", d)
    return m


def mk_devbuf(raw: PythonObject, n: Int, ctx: PythonObject) raises -> PythonObject:
    """`_DevBuf(raw, n)`: a pooled device buffer of the binding, freed (back to
    the pool) when its last holder dies."""
    var d = native_storage_allocate(ctx["family_mani_devbuf"])
    py_setattr(d, "b", raw)
    py_setattr(d, "n", PythonObject(n))
    py_setattr(d, "id", family_call(family_entry(raw, "x_decomp_dev_alloc", ctx), PythonObject(max(n, 1))))
    return d


def mk_zeros(r: Int, c: Int, ctx: PythonObject) raises -> PythonObject:
    """`_M.zeros(r, c)`."""
    return mk_new(mk_store_zeros(r * c, ctx), r, c, ctx)


def mk_of(values: List[Float64], r: Int, c: Int, ctx: PythonObject) raises -> PythonObject:
    """`_M.of(values, r, c)`: each value rounded once to float32."""
    var store = mk_store_zeros(len(values), ctx)
    if len(values) > 0:
        var p = mk_store_ptr(store)
        for i in range(len(values)):  # small-loop(values: a constant operand the caller spells out): bounded by the call site
            p[i] = Float32(values[i])
    return mk_new(store, r, c, ctx)


def mk_fill(value: Float64, r: Int, c: Int, ctx: PythonObject) raises -> PythonObject:
    """`_M.of([value] * (r * c), r, c)`."""
    var values = List[Float64](length=r * c, fill=value)
    return mk_of(values, r, c, ctx)


def mk_one(ctx: PythonObject) raises -> PythonObject:
    """`_M._one`, the shared host 1 x 1 zero."""
    return ctx["family_mani_M"]._one


def mk_store(m: PythonObject, ctx: PythonObject) raises -> PythonObject:
    """`m.s`: the host store, downloaded once if the matrix is on the device;
    the matrix is a host matrix afterwards."""
    if m._s is None:
        var d = m._d
        var n = mk_n(m)
        var a = mk_store_zeros(n, ctx)
        if n > 0:
            _ = family_call(family_entry(d.b, "x_decomp_dev_download", ctx), d.id, mk_store_addr(a), PythonObject(n))
        py_setattr(m, "_s", a)
    py_setattr(m, "_d", PythonObject(None))
    return m._s


def mk_addr(m: PythonObject, ctx: PythonObject) raises -> PythonObject:
    """`m.addr`: its host store's address (`_M._one`'s for an empty one)."""
    var store = mk_store(m, ctx)
    if mk_store_len(store):
        return mk_store_addr(store)
    return mk_store_addr(mk_store(mk_one(ctx), ctx))


def mk_ptr(m: PythonObject, ctx: PythonObject) raises -> FAMILY_F32P:
    return FAMILY_F32P(unsafe_from_address=Int(py=mk_addr(m, ctx)))


def mk_at(m: PythonObject, i: Int, ctx: PythonObject) raises -> Float64:
    """`float(m.s[i])`."""
    var store = mk_store(m, ctx)
    var v = Float64(mk_store_ptr(store)[i])
    _ = store^
    return v


def mk_copy(m: PythonObject, ctx: PythonObject) raises -> PythonObject:
    """`m.copy()`: a host copy."""
    return mk_new(mk_store_copy(mk_store(m, ctx), ctx), mk_r(m), mk_c(m), ctx)


def mk_out(m: PythonObject, shape: PythonObject, ctx: PythonObject) raises -> PythonObject:
    """`m.out(shape)`: a public float32 Array (C order) holding a copy."""
    var store = mk_store_copy(mk_store(m, ctx), ctx)
    return classical_wrap_array(store, shape, "<f4", "C", ctx)


def mk_out2(m: PythonObject, ctx: PythonObject) raises -> PythonObject:
    """`m.out()`: shape (r, c)."""
    return mk_out(m, Python.tuple(mk_r(m), mk_c(m)), ctx)


def mk_reshape(m: PythonObject, r: Int, c: Int, ctx: PythonObject) raises -> PythonObject:
    """`m.reshape(r, c)`: a view of the same buffer or the same store."""
    if mk_on_dev(m) and r * c == mk_n(m):
        return mk_dev_view(m._d, r, c, ctx)
    return mk_new(mk_store(m, ctx), r, c, ctx)


def mk_res_moves(m: PythonObject, count: Int) raises -> Bool:
    """`_res_moves(M, count)`."""
    return count > 0 and mk_r(m) < F32_INDEX_MAX and mk_c(m) < F32_INDEX_MAX


def mk_dev_gather(m: PythonObject, op: Int, idx: List[Int], ctx: PythonObject) raises -> PythonObject:
    """`_dev_gather(M, op, idx)`: rows (TAKE_ROWS) or columns (TAKE_COLS) of
    a device matrix, on its own binding."""
    var d = m._d
    var raw = d.b
    var w = len(idx)
    var ia = mk_store_zeros(w, ctx)
    var p = mk_store_ptr(ia)
    for t in range(w):  # small-loop(idx: the gathered row or column numbers): bounded by the caller's index list
        p[t] = Float32(Float64(idx[t]))
    var di = mk_devbuf(raw, w, ctx)
    _ = family_call(family_entry(raw, "x_decomp_dev_upload", ctx), di.id, mk_store_addr(ia), PythonObject(w))
    var r = mk_r(m)
    var c = mk_c(m)
    var n = r * c
    var out: PythonObject
    if op == MV_TAKE_ROWS:
        var count = w * c
        out = mk_dev_view(mk_devbuf(raw, count, ctx), w, c, ctx)
        _ = family_call(family_entry(raw, "x_decomp_dev_move", ctx), d.id, di.id, out._d.id,
                        mk_ints([op, count, c, 0, 0, 1, 0, n, count, w]))
    else:
        var count = r * w
        out = mk_dev_view(mk_devbuf(raw, count, ctx), r, w, ctx)
        _ = family_call(family_entry(raw, "x_decomp_dev_move", ctx), d.id, di.id, out._d.id,
                        mk_ints([op, count, w, c, 0, 0, 0, n, count, w]))
    _ = ia^
    _ = di^
    return out


def mk_rows(m: PythonObject, a: Int, b: Int, ctx: PythonObject) raises -> PythonObject:
    """`m.rows(a, b)`."""
    var c = mk_c(m)
    if mk_on_dev(m) and 0 <= a and a < b and b <= mk_r(m) and mk_res_moves(m, (b - a) * c):
        var idx = List[Int]()
        for i in range(a, b):  # small-loop(rows: the requested row range): bounded by the caller
            idx.append(i)
        return mk_dev_gather(m, MV_TAKE_ROWS, idx, ctx)
    var store = mk_store(m, ctx)
    var part = py_call(py_attr(store, "__getitem__"), py_call(Python.import_module("builtins").slice,
                                                  PythonObject(a * c), PythonObject(b * c)))
    return mk_new(part, b - a, c, ctx)


def mk_take_cols(m: PythonObject, idx: List[Int], ctx: PythonObject) raises -> PythonObject:
    """`m.take_cols(idx)`: on the device for a device matrix, else one host
    copy per selected column."""
    var w = len(idx)
    var r = mk_r(m)
    var c = mk_c(m)
    var in_range = True
    for t in range(w):  # small-loop(idx: the selected column numbers): bounded by the caller's index list
        if idx[t] < 0 or idx[t] >= c:
            in_range = False
    if mk_on_dev(m) and in_range and w > 0 and mk_res_moves(m, r * w):
        return mk_dev_gather(m, MV_TAKE_COLS, idx, ctx)
    if not in_range:
        raise Error("x_decomp: column index out of range")
    var src = mk_store(m, ctx)
    var out = mk_store_zeros(r * w, ctx)
    if r * w > 0:
        var sp = mk_store_ptr(src)
        var dp = mk_store_ptr(out)
        for t in range(w):  # small-loop(idx: one host column copy per selected column): bounded by the caller's index list
            for i in range(r):  # native data movement: the column's values of a host matrix
                dp[i * w + t] = sp[i * c + idx[t]]
    _ = src^
    return mk_new(out, r, w, ctx)


def mk_col_range(a: Int, b: Int) -> List[Int]:
    var idx = List[Int]()
    for j in range(a, b):  # small-loop(columns: a column range): bounded by the caller
        idx.append(j)
    return idx^


def mk_desc_range(start: Int, count: Int) -> List[Int]:
    """`list(range(start, start - count, -1))`."""
    var idx = List[Int]()
    for t in range(count):  # small-loop(components: the selected positions): bounded by n_components
        idx.append(start - t)
    return idx^


def mk_vstack_host(ms: List[PythonObject], ctx: PythonObject) raises -> PythonObject:
    """`_vstack(*ms)`: the stores concatenated on the host."""
    var s = py_call(ctx["array_type"], PythonObject("f"))
    var rows = 0
    for i in range(len(ms)):  # small-loop(ms: the stacked matrices): bounded by the caller
        _ = py_call(s.extend, mk_store(ms[i], ctx))
        rows += mk_r(ms[i])
    return mk_new(s, rows, mk_c(ms[0]), ctx)


struct MKit(Movable):
    """`_Kit(mode, binding)`: the binding's cells, called on `_M` matrices."""
    var mode: String
    var b: PythonObject
    var raw: PythonObject
    var ctx: PythonObject
    var res_known: Bool
    var res_ok: Bool
    var opt: PythonObject
    var lle_mm: PythonObject
    var lle_lu: PythonObject

    def __init__(out self, mode: String, binding: PythonObject, ctx: PythonObject) raises:
        self.mode = mode
        self.b = binding
        self.ctx = ctx
        # `_raw()`: a profiler proxy's wrapped binding, else the binding
        var raw = binding
        try:
            raw = py_call(Python.import_module("builtins").object.__getattribute__, binding, PythonObject("_b"))
        except:
            raw = binding
        self.raw = raw
        self.res_known = False
        self.res_ok = False
        self.opt = Python.dict()
        self.lle_mm = PythonObject(None)
        self.lle_lu = PythonObject(None)

    def fn(self, name: String) raises -> PythonObject:
        """`self.b.<name>`."""
        return family_entry(self.b, name, self.ctx)

    def has(self, name: String) -> Bool:
        """Whether the raw binding exports `name` (any refusal: no)."""
        try:
            _ = py_call(Python.import_module("builtins").getattr, self.raw, PythonObject(name))
            return True
        except:
            return False

    def res(mut self) -> Bool:
        """`_res()`: the binding has the resident entries (a GPU binding)."""
        if not self.res_known:
            self.res_ok = self.has("x_decomp_dev_ew")
            self.res_known = True
        return self.res_ok

    def opt_dev(mut self, name: String) raises -> Bool:
        """`_opt_dev(name)`: an optional resident entry, asked once."""
        var key = PythonObject(name)
        var got = py_get(self.opt, key, PythonObject(None))
        if got is None:
            var r = False
            if self.res():
                r = self.has(name)
            py_setitem(self.opt, key, PythonObject(r))
            return r
        return Bool(py=got)

    def vendor(self) -> String:
        """`_kit_vendor`: the binding's compiled vendor, "" when it cannot say."""
        try:
            var f = py_call(Python.import_module("builtins").getattr, self.raw, PythonObject("x_decomp_vendor"))
            return String(py=py_call(Python.import_module("builtins").str, py_call(f)))
        except:
            return String("")

    def w4_flags(self) -> Int:
        """`w4_flags()`: 0 on a binding without the entry."""
        try:
            var f = py_call(Python.import_module("builtins").getattr, self.raw, PythonObject("x_decomp_w4_flags"))
            return Int(py=py_call(Python.import_module("builtins").int, py_call(f)))
        except:
            return 0

    def enable_lle_precision(mut self) raises:
        """The Apple FAST binding's ordered GEMM and LU for LLE's near-null
        solve; every other binding keeps its entries."""
        if self.mode != "fast" or self.vendor() != "metal":
            return
        try:
            var g = Python.import_module("builtins").getattr
            var mm = py_call(g, self.b, PythonObject("x_decomp_lle_gemm_ordered"))
            var lu = py_call(g, self.b, PythonObject("x_decomp_lle_lu_ordered"))
            self.lle_mm = mm
            self.lle_lu = lu
        except:
            return

    # ---- placement
    def use(mut self, a: PythonObject, b: PythonObject, c: PythonObject) raises -> Bool:
        """`_use(a, b, c)` (None for an absent operand)."""
        if not self.res():
            return False
        var big = False
        if not (a is None):
            if mk_on_dev(a):
                return True
            if mk_n(a) >= 1:
                big = True
        if not (b is None):
            if mk_on_dev(b):
                return True
            if mk_n(b) >= 1:
                big = True
        if not (c is None):
            if mk_on_dev(c):
                return True
            if mk_n(c) >= 1:
                big = True
        return big

    def use1(mut self, a: PythonObject) raises -> Bool:
        return self.use(a, PythonObject(None), PythonObject(None))

    def use2(mut self, a: PythonObject, b: PythonObject) raises -> Bool:
        return self.use(a, b, PythonObject(None))

    def did(mut self, m: PythonObject) raises -> PythonObject:
        """`_did(M)`: M's device id on this binding, uploading a host matrix
        (it moves)."""
        var d = m._d
        if not (d is None) and d.b is self.raw:
            return d.id
        var a = mk_store(m, self.ctx)
        var n = mk_store_len(a)
        var nd = mk_devbuf(self.raw, n, self.ctx)
        if n > 0:
            _ = family_call(self.fn("x_decomp_dev_upload"), nd.id, mk_store_addr(a), PythonObject(n))
        py_setattr(m, "_s", PythonObject(None))
        py_setattr(m, "_d", nd)
        _ = a^
        return nd.id

    def dout(mut self, r: Int, c: Int) raises -> PythonObject:
        return mk_dev_view(mk_devbuf(self.raw, r * c, self.ctx), r, c, self.ctx)

    def dev_one(mut self) raises -> PythonObject:
        """`_dev_one(kit)`: the unused broadcast operand on the device, one
        per binding (the Python kit's own cache)."""
        var cache = self.ctx["family_mani_dev_one"]
        var key = py_call(Python.import_module("builtins").id, self.raw)
        var d = py_get(cache, key, PythonObject(None))
        if d is None or not (d.b is self.raw):
            d = mk_devbuf(self.raw, 1, self.ctx)
            var z = mk_store_zeros(1, self.ctx)
            _ = family_call(self.fn("x_decomp_dev_upload"), d.id, mk_store_addr(z), PythonObject(1))
            _ = z^
            py_setitem(cache, key, d)
        return d.id

    # ---- elementwise
    def _mode_of(self, x: PythonObject, a: PythonObject) raises -> Int:
        if x is None:
            return 3
        var xr = mk_r(x)
        var xc = mk_c(x)
        var ar = mk_r(a)
        var ac = mk_c(a)
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

    def ew_full(mut self, op: Int, a: PythonObject, b: PythonObject, c: PythonObject,
                s: Float64) raises -> PythonObject:
        """`ew(op, A, B, C, s)`."""
        var bmode = self._mode_of(b, a)
        var cmode = self._mode_of(c, a)
        var one = mk_one(self.ctx)
        var bm = one if b is None else b
        var cm = one if c is None else c
        var n = mk_n(a)
        if n > 0 and self.use(a, b, c):
            var out = self.dout(mk_r(a), mk_c(a))
            var one_id = PythonObject(None)
            if b is None or c is None:
                one_id = self.dev_one()
            var ib = one_id if b is None else self.did(bm)
            var ic = one_id if c is None else self.did(cm)
            var ia = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_ew"), ia, ib, ic, out._d.id,
                            mk_ints([op, n, mk_c(a), mk_n(bm), bmode, mk_n(cm), cmode]), PythonObject(s))
            return out
        var out = mk_zeros(mk_r(a), mk_c(a), self.ctx)
        if n > 0:
            var pa = mk_addr(a, self.ctx)
            var pb = mk_addr(bm, self.ctx)
            var pc = mk_addr(cm, self.ctx)
            var po = mk_addr(out, self.ctx)
            var params = mk_ints([op, n, mk_c(a), mk_store_len(mk_store(bm, self.ctx)), bmode,
                                  mk_store_len(mk_store(cm, self.ctx)), cmode])
            _ = family_call(self.fn("x_decomp_ew"), pa, pb, pc, po, params, PythonObject(s))
        return out

    def ew1(mut self, op: Int, a: PythonObject, s: Float64) raises -> PythonObject:
        return self.ew_full(op, a, PythonObject(None), PythonObject(None), s)

    def ew2(mut self, op: Int, a: PythonObject, b: PythonObject) raises -> PythonObject:
        return self.ew_full(op, a, b, PythonObject(None), 0.0)

    def ew2s(mut self, op: Int, a: PythonObject, b: PythonObject, s: Float64) raises -> PythonObject:
        return self.ew_full(op, a, b, PythonObject(None), s)

    def ew3(mut self, op: Int, a: PythonObject, b: PythonObject, c: PythonObject, s: Float64) raises -> PythonObject:
        return self.ew_full(op, a, b, c, s)

    def neg_signs(mut self, v: PythonObject) raises -> PythonObject:
        """-1 where V < 0, else +1: select(-V > 0)."""
        var neg = self.ew1(OP_SCALE, v, -1.0)
        var lo = mk_of([-1.0], 1, 1, self.ctx)
        var hi = mk_of([1.0], 1, 1, self.ctx)
        return self.ew3(OP_SELECT, neg, lo, hi, 0.0)

    # ---- products and folds
    def mm(mut self, a: PythonObject, b: PythonObject, ta: Bool, tb: Bool) raises -> PythonObject:
        var m = mk_c(a) if ta else mk_r(a)
        var k = mk_r(a) if ta else mk_c(a)
        var k2 = mk_c(b) if tb else mk_r(b)
        var n = mk_r(b) if tb else mk_c(b)
        if k != k2:
            raise Error("x_decomp: gemm inner dimensions " + String(k) + " and " + String(k2) + " differ")
        var params = mk_ints([m, k, n, 1 if ta else 0, 1 if tb else 0])
        if not (self.lle_mm is None):
            var out = mk_zeros(m, n, self.ctx)
            var pa = mk_addr(a, self.ctx)
            var pb = mk_addr(b, self.ctx)
            _ = family_call(self.lle_mm, pa, pb, mk_addr(out, self.ctx), params)
            return out
        if m * n > 0 and m * k > 0 and k * n > 0 and self.use2(a, b):
            var out = self.dout(m, n)
            var ia = self.did(a)
            var ib = self.did(b)
            _ = family_call(self.fn("x_decomp_dev_gemm"), ia, ib, out._d.id, params)
            return out
        var out = mk_zeros(m, n, self.ctx)
        if m * n > 0:
            var pa = mk_addr(a, self.ctx)
            var pb = mk_addr(b, self.ctx)
            _ = family_call(self.fn("x_decomp_gemm"), pa, pb, mk_addr(out, self.ctx), params)
        return out

    def colsum(mut self, a: PythonObject) raises -> PythonObject:
        var r = mk_r(a)
        var c = mk_c(a)
        if r * c > 0 and self.use1(a):
            var out = self.dout(1, c)
            var ia = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_colsum"), ia, out._d.id, mk_ints([r, c]))
            return out
        var out = mk_zeros(1, c, self.ctx)
        var pa = mk_addr(a, self.ctx)
        _ = family_call(self.fn("x_decomp_colsum"), pa, mk_addr(out, self.ctx), mk_ints([r, c]))
        return out

    def rowsum(mut self, a: PythonObject) raises -> PythonObject:
        var r = mk_r(a)
        var c = mk_c(a)
        if r * c > 0 and self.use1(a):
            var out = self.dout(r, 1)
            var ia = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_rowsum"), ia, out._d.id, mk_ints([r, c]))
            return out
        var out = mk_zeros(r, 1, self.ctx)
        var pa = mk_addr(a, self.ctx)
        _ = family_call(self.fn("x_decomp_rowsum"), pa, mk_addr(out, self.ctx), mk_ints([r, c]))
        return out

    def total(mut self, a: PythonObject) raises -> PythonObject:
        """The sum of every entry (1 x 1): rows ascending, then the row sums."""
        var rs = self.rowsum(a)
        return self.colsum(rs)

    def sqdist(mut self, a: PythonObject, b: PythonObject) raises -> PythonObject:
        var ar = mk_r(a)
        var br = mk_r(b)
        var ac = mk_c(a)
        var params = mk_ints([ar, br, ac])
        if ar * br > 0 and ac > 0 and self.use2(a, b):
            var out = self.dout(ar, br)
            var ia = self.did(a)
            var ib = self.did(b)
            _ = family_call(self.fn("x_decomp_dev_sqdist"), ia, ib, out._d.id, params)
            return out
        var out = mk_zeros(ar, br, self.ctx)
        var pa = mk_addr(a, self.ctx)
        var pb = mk_addr(b, self.ctx)
        _ = family_call(self.fn("x_decomp_sqdist"), pa, pb, mk_addr(out, self.ctx), params)
        return out

    def pdist(mut self, a: PythonObject, b: PythonObject, kind: Int, pw: Float64) raises -> PythonObject:
        """A non-Euclidean distance matrix: kind 1 manhattan, 2 chebyshev,
        3 minkowski pw, 4 cosine."""
        var ar = mk_r(a)
        var br = mk_r(b)
        var ac = mk_c(a)
        var params = mk_ints([ar, br, ac, kind])
        mk_append(params, PythonObject(pw))
        if ar * br > 0 and ac > 0 and self.use2(a, b):
            var out = self.dout(ar, br)
            var ia = self.did(a)
            var ib = self.did(b)
            _ = family_call(self.fn("x_decomp_dev_sqdist"), ia, ib, out._d.id, params)
            return out
        var out = mk_zeros(ar, br, self.ctx)
        var pa = mk_addr(a, self.ctx)
        var pb = mk_addr(b, self.ctx)
        _ = family_call(self.fn("x_decomp_sqdist"), pa, pb, mk_addr(out, self.ctx), params)
        return out

    def rand(mut self, r: Int, c: Int, seed: PythonObject, stream: Int, kind: Int) raises -> PythonObject:
        """`rand(r, c, seed, stream, kind)` (the Philox stream)."""
        var op = Python.import_module("operator")
        var b = Python.import_module("builtins")
        var params = Python.list()
        mk_append(params, PythonObject(r * c))
        mk_append(params, py_call(op.and_, py_call(b.int, seed), PythonObject(0xFFFFFFFF)))
        mk_append(params, PythonObject(stream & 0xFFFFFFFF))
        mk_append(params, PythonObject(kind))
        if r * c >= 1024 and self.res():
            var out = self.dout(r, c)
            _ = family_call(self.fn("x_decomp_dev_rand"), out._d.id, params)
            return out
        var out = mk_zeros(r, c, self.ctx)
        if r * c > 0:
            _ = family_call(self.fn("x_decomp_rand"), mk_addr(out, self.ctx), params)
        return out

    # ---- small dense linear algebra
    def eigh(mut self, a: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
        """Ascending eigenvalues (1 x n) and eigenvectors in COLUMNS (n x n)."""
        var n = mk_r(a)
        var w = mk_zeros(1, n, self.ctx)
        var v = mk_zeros(n, n, self.ctx)
        var params = mk_ints([n, 0])
        var d = a._d
        if n >= 1 and mk_c(a) == n and not (d is None) and d.b is self.raw and self.opt_dev("x_decomp_dev_eigh"):
            _ = family_call(self.fn("x_decomp_dev_eigh"), d.id, mk_addr(w, self.ctx), mk_addr(v, self.ctx), params)
            return (w, v)
        var pa = mk_addr(a, self.ctx)
        _ = family_call(self.fn("x_decomp_eigh"), pa, mk_addr(w, self.ctx), mk_addr(v, self.ctx), params)
        return (w, v)

    def lu(mut self, a: PythonObject) raises -> Tuple[PythonObject, PythonObject, Int]:
        """(lu, piv int32 array, info)."""
        var n = mk_r(a)
        var piv = py_call(self.ctx["array_type"], PythonObject("i"),
                          py_call(self.ctx["native_bytes_type"], PythonObject(n * 4)))
        var info = mk_zeros(1, 1, self.ctx)
        if not (self.lle_lu is None):
            var lu = mk_copy(a, self.ctx)
            _ = family_call(self.lle_lu, mk_addr(lu, self.ctx), mk_store_addr(piv), mk_addr(info, self.ctx),
                            mk_ints([n]))
            return (lu, piv, Int(mk_at(info, 0, self.ctx)))
        if n >= 1 and mk_c(a) == n and self.opt_dev("x_decomp_dev_lu") and self.use1(a):
            var lu = self.dout(n, n)
            var ia = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_lu"), ia, lu._d.id, mk_store_addr(piv), mk_addr(info, self.ctx),
                            mk_ints([n]))
            return (lu, piv, Int(mk_at(info, 0, self.ctx)))
        var lu = mk_copy(a, self.ctx)
        _ = family_call(self.fn("x_decomp_lu"), mk_addr(lu, self.ctx), mk_store_addr(piv), mk_addr(info, self.ctx),
                        mk_ints([n]))
        return (lu, piv, Int(mk_at(info, 0, self.ctx)))

    def lu_aux(mut self, lu: PythonObject, piv: PythonObject,
               clamp: Bool) raises -> Tuple[List[Float64], PythonObject, PythonObject, PythonObject]:
        """(stats [max |u_ii|, zero pivots, negative pivots, swaps], diag
        1 x n, pm n x 1, im n x 1)."""
        var n = mk_r(lu)
        var pm = mk_zeros(n, 1, self.ctx)
        var im = mk_zeros(n, 1, self.ctx)
        var diag = mk_zeros(1, n, self.ctx)
        var st = mk_zeros(1, 4, self.ctx)
        var params = mk_ints([n, 1 if clamp else 0])
        var d = lu._d
        if n >= 1 and not (d is None) and d.b is self.raw and self.opt_dev("x_decomp_dev_lu"):
            _ = family_call(self.fn("x_decomp_dev_lu_aux"), d.id, mk_store_addr(piv), mk_addr(pm, self.ctx),
                            mk_addr(im, self.ctx), mk_addr(diag, self.ctx), mk_addr(st, self.ctx), params)
        else:
            var pl = mk_addr(lu, self.ctx)
            _ = family_call(self.fn("x_decomp_lu_aux"), pl, mk_store_addr(piv), mk_addr(pm, self.ctx),
                            mk_addr(im, self.ctx), mk_addr(diag, self.ctx), mk_addr(st, self.ctx), params)
        var stats = List[Float64]()
        for i in range(4):  # small-loop(stats: the four lu_aux status words): 4
            stats.append(mk_at(st, i, self.ctx))
        return (stats^, diag, pm, im)

    def lu_dev_aux(mut self, a: PythonObject,
                   clamp: Bool) raises -> Tuple[PythonObject, PythonObject, PythonObject, List[Float64]]:
        """LLE_FAST_DEV_LU: `lu` then `lu_aux` on the device, A untouched:
        (lu, pm, im) device-resident and the four stats."""
        var n = mk_r(a)
        var lu = self.dout(n, n)
        var pm = self.dout(n, 1)
        var im = self.dout(n, 1)
        var diag = self.dout(1, n)
        var st = self.dout(1, 4)
        var ia = self.did(a)
        _ = family_call(self.fn("x_decomp_dev_lu_aux"), ia, lu._d.id, pm._d.id, im._d.id, diag._d.id, st._d.id,
                        mk_ints([n, 1 if clamp else 0]))
        var stats = List[Float64]()
        for i in range(4):  # small-loop(stats: the four lu_aux status words): 4
            stats.append(mk_at(st, i, self.ctx))
        return (lu, pm, im, stats^)

    def trisolve(mut self, lu: PythonObject, idx: PythonObject, b: PythonObject, trans: Int) raises -> PythonObject:
        var n = mk_r(lu)
        var w = mk_c(b)
        var params = mk_ints([n, w, trans])
        if n * w > 0 and self.use2(lu, b):
            var out = self.dout(n, w)
            var il = self.did(lu)
            var ii = self.did(idx)
            var ib = self.did(b)
            _ = family_call(self.fn("x_decomp_dev_trisolve"), il, ii, ib, out._d.id, params)
            return out
        var out = mk_zeros(n, w, self.ctx)
        if n * w > 0:
            var pl = mk_addr(lu, self.ctx)
            var pi = mk_addr(idx, self.ctx)
            var pb = mk_addr(b, self.ctx)
            _ = family_call(self.fn("x_decomp_trisolve"), pl, pi, pb, mk_addr(out, self.ctx), params)
        return out

    def svd(mut self, a: PythonObject, mut hk: MKit) raises -> Tuple[PythonObject, PythonObject]:
        """(S 1 x n DESCENDING, Vt n x n) of a tall A: values sorted
        descending, ties to the lower index."""
        var m = mk_r(a)
        var n = mk_c(a)
        var s = mk_zeros(1, n, self.ctx)
        var v = mk_zeros(n, n, self.ctx)
        var params = mk_ints([m, n])
        if m >= n and n >= 1 and self.opt_dev("x_decomp_dev_svd") and self.use1(a):
            var ia = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_svd"), ia, mk_addr(s, self.ctx), mk_addr(v, self.ctx), params)
        else:
            var pa = mk_addr(a, self.ctx)
            _ = family_call(self.fn("x_decomp_svd"), pa, mk_addr(s, self.ctx), mk_addr(v, self.ctx), params)
        var neg = self.ew1(OP_SCALE, s, -1.0)
        var o = self.order(neg)
        var st = mk_t(s, hk, self.ctx)
        var s_sorted = mk_t(self.take_rows(st, o), hk, self.ctx)
        var vt = mk_t(v, hk, self.ctx)
        var v_sorted = self.take_rows(vt, o)
        return (s_sorted, v_sorted)

    # ---- moves and orders
    def take_rows_at(mut self, a: PythonObject, idx: PythonObject, m: Int, ist: Int, ioff: Int,
                     radd: Int) raises -> PythonObject:
        """`take_rows(A, idx, m, ist, ioff, radd)`: rows Int(idx[t * ist +
        ioff]) + radd of A for t < m."""
        var c = mk_c(a)
        var count = m * c
        var params = mk_ints([MV_TAKE_ROWS, count, c, radd, 0, ist, ioff, mk_n(a), count, mk_n(idx)])
        if count > 0 and self.use2(a, idx):
            var out = self.dout(m, c)
            var ia = self.did(a)
            var ii = self.did(idx)
            _ = family_call(self.fn("x_decomp_dev_move"), ia, ii, out._d.id, params)
            return out
        var out = mk_zeros(m, c, self.ctx)
        if count > 0:
            var pa = mk_addr(a, self.ctx)
            var pi = mk_addr(idx, self.ctx)
            _ = family_call(self.fn("x_decomp_move"), pa, pi, mk_addr(out, self.ctx), params)
        return out

    def take_rows(mut self, a: PythonObject, idx: PythonObject) raises -> PythonObject:
        return self.take_rows_at(a, idx, mk_n(idx), 1, 0, 0)

    def fill0(mut self, a: PythonObject, start: Int, stride: Int, count: Int) raises:
        """A[start + t * stride] = 0 for t < count, in place."""
        if count == 0:
            return
        var n = mk_n(a)
        var params = mk_ints([MV_FILL0, count, start, stride, 0, 0, 0, n, n, 1])
        if self.use1(a):
            var ia = self.did(a)
            var one = self.dev_one()
            _ = family_call(self.fn("x_decomp_dev_move"), ia, one, ia, params)
            return
        var pa = mk_addr(a, self.ctx)
        _ = family_call(self.fn("x_decomp_move"), pa, mk_addr(mk_one(self.ctx), self.ctx), pa, params)

    def reduce(mut self, a: PythonObject, op: Int) raises -> PythonObject:
        """max |A| (0), max A (1) or min A (2) as a 1 x 1 matrix."""
        var n = mk_n(a)
        if n == 0:
            raise Error("x_decomp: reduce of an empty matrix")
        var params = mk_ints([n, op])
        if self.use1(a):
            var out = self.dout(1, 1)
            var ia = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_reduce"), ia, out._d.id, params)
            return out
        var out = mk_zeros(1, 1, self.ctx)
        var pa = mk_addr(a, self.ctx)
        _ = family_call(self.fn("x_decomp_reduce"), pa, mk_addr(out, self.ctx), params)
        return out

    def strided(mut self, a: PythonObject, count: Int, stride_in: Int, off: Int) raises -> PythonObject:
        """A's values at off + t * stride (t < count) as 1 x count."""
        var stride = off + 1 if count == 1 else stride_in
        var idx = mk_of([Float64(off)], 1, 1, self.ctx)
        var n = mk_n(a)
        var params = mk_ints([MV_TAKE_COLS, count, 1, stride, 0, 0, 0, n, count, 1])
        if count > 0 and self.use1(a):
            var out = self.dout(1, count)
            var ia = self.did(a)
            var ii = self.did(idx)
            _ = family_call(self.fn("x_decomp_dev_move"), ia, ii, out._d.id, params)
            return out
        var out = mk_zeros(1, count, self.ctx)
        if count > 0:
            var pa = mk_addr(a, self.ctx)
            var pi = mk_addr(idx, self.ctx)
            _ = family_call(self.fn("x_decomp_move"), pa, pi, mk_addr(out, self.ctx), params)
        return out

    def word(mut self, a: PythonObject, i: Int) raises -> Float64:
        """A's value at flat position i: one word down."""
        if mk_on_dev(a) and mk_n(a) > 1:
            return mk_at(self.strided(a, 1, 1, i), 0, self.ctx)
        return mk_at(a, i, self.ctx)

    def place_strided(mut self, d: PythonObject, v: PythonObject, stride: Int, off: Int) raises:
        """D[off + t * stride] = V[t], in place."""
        var cnt = mk_n(v)
        if cnt == 0:
            return
        var params = mk_ints([MV_PLACE_COLS, cnt, 1, stride, off, 0, 0, cnt, mk_n(d), 1])
        if self.use2(d, v):
            var iv = self.did(v)
            var one = self.dev_one()
            var id = self.did(d)
            _ = family_call(self.fn("x_decomp_dev_move"), iv, one, id, params)
            return
        var pv = mk_addr(v, self.ctx)
        var po = mk_addr(mk_one(self.ctx), self.ctx)
        var pd = mk_addr(d, self.ctx)
        _ = family_call(self.fn("x_decomp_move"), pv, po, pd, params)

    def place_rows(mut self, d: PythonObject, v: PythonObject, row: Int) raises:
        """V (r x D.c) into rows row .. row + r - 1 of D, in place."""
        if mk_n(v) > 0:
            self.place_strided(d, mk_reshape(v, 1, mk_n(v), self.ctx), 1, row * mk_c(d))

    def vstack(mut self, ms: List[PythonObject]) raises -> PythonObject:
        """`vstack(ms)`: device PLACE_COLS moves for a resident kit."""
        var total = 0
        for i in range(len(ms)):  # small-loop(ms: the stacked matrices): bounded by the caller
            total += mk_n(ms[i])
        if not (self.res() and total > 0):
            return mk_vstack_host(ms, self.ctx)
        var c = mk_c(ms[0])
        var rows = 0
        for i in range(len(ms)):  # small-loop(ms: the stacked matrices): bounded by the caller
            rows += mk_r(ms[i])
        var out = self.dout(rows, c)
        var one = self.dev_one()
        var off = 0
        for i in range(len(ms)):  # small-loop(ms: one Mojo move per stacked matrix): bounded by the caller
            var cnt = mk_n(ms[i])
            if cnt > 0:
                var im = self.did(ms[i])
                _ = family_call(self.fn("x_decomp_dev_move"), im, one, out._d.id,
                                mk_ints([MV_PLACE_COLS, cnt, c, c, off * c, 0, 0, cnt, rows * c, 1]))
            off += mk_r(ms[i])
        return out

    def absmax_signs(mut self, a: PythonObject, by_col: Bool) raises -> PythonObject:
        """Per column (by_col, 1 x c) or row (r x 1): -1 where the largest-|.|
        entry (ties to the lower index) is negative, else +1."""
        var r = mk_r(a)
        var c = mk_c(a)
        var cnt = c if by_col else r
        var params = mk_ints([r, c, 1 if by_col else 0])
        var out: PythonObject
        if cnt > 0 and r * c > 0 and self.use1(a):
            out = self.dout(1, cnt)
            var ia = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_absmax"), ia, out._d.id, params)
        else:
            out = mk_zeros(1, cnt, self.ctx)
            if cnt > 0 and r * c > 0:
                var pa = mk_addr(a, self.ctx)
                _ = family_call(self.fn("x_decomp_absmax_sign"), pa, mk_addr(out, self.ctx), params)
        if by_col:
            return self.neg_signs(out)
        return self.neg_signs(mk_reshape(out, cnt, 1, self.ctx))

    def refuse_nan(mut self, a: PythonObject) raises:
        var v = self.word(self.reduce(a, SEL_MAX), 0)
        if v != v:
            raise Error("x_decomp: a NaN has no order (refused)")

    def order(mut self, a: PythonObject) raises -> PythonObject:
        """The stable ascending order of A's values (ties to the lower index)
        as n x 1 exact floats; NaN refused."""
        var n = mk_n(a)
        if n > 0 and self.use1(a):
            if n >= F32_INDEX_MAX:
                raise Error("x_decomp: order_f exceeds the float32 index bound")
            self.refuse_nan(a)
            var out = self.dout(n, 1)
            var cnt = self.dout(1, 1)
            var ia = self.did(a)
            var sid = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_order_f"), ia, sid, out._d.id, cnt._d.id, mk_ints([n, 0, 0]))
            return out
        var out = mk_zeros(n, 1, self.ctx)
        if n > 0:
            var pa = mk_addr(a, self.ctx)
            _ = family_call(self.fn("x_decomp_order_f"), pa, PythonObject(n), mk_addr(out, self.ctx))
        return out

    def argmin_all(mut self, a: PythonObject) raises -> PythonObject:
        """Every position (exact floats, c x 1) whose value is the minimum."""
        var n = mk_n(a)
        if n > 0 and self.use1(a):
            if n >= F32_INDEX_MAX:
                raise Error("x_decomp: argmin_all exceeds the float32 index bound")
            var out = self.dout(n, 1)
            var cnt = self.dout(1, 1)
            var ia = self.did(a)
            _ = family_call(self.fn("x_decomp_dev_argmin_all"), ia, out._d.id, cnt._d.id, mk_ints([n]))
            var c = Int(self.word(cnt, 0))
            if c > 0:
                return mk_rows(out, 0, c, self.ctx)
            return mk_zeros(0, 1, self.ctx)
        var buf = mk_zeros(max(n, 1), 1, self.ctx)
        var c = 0
        if n > 0:
            var pa = mk_addr(a, self.ctx)
            c = Int(py=py_call(Python.import_module("builtins").int,
                               family_call(self.fn("x_decomp_argmin_all"), pa, PythonObject(n),
                                           mk_addr(buf, self.ctx))))
        var store = mk_store(buf, self.ctx)
        var part = py_call(py_attr(store, "__getitem__"), py_call(Python.import_module("builtins").slice,
                                                      PythonObject(0), PythonObject(c)))
        return mk_new(part, c, 1, self.ctx)

    def pad_zero_row(mut self, x: PythonObject) raises -> PythonObject:
        """[X; 0]: X over one zero row (on the device for a device X)."""
        var r = mk_r(x)
        var c = mk_c(x)
        if c > 0 and mk_on_dev(x) and self.use1(x):
            var tot = (r + 1) * c
            var out = self.dout(r + 1, c)
            var one = self.dev_one()
            var cnt = r * c
            if cnt > 0:
                var ix = self.did(x)
                _ = family_call(self.fn("x_decomp_dev_move"), ix, one, out._d.id,
                                mk_ints([MV_PLACE_COLS, cnt, c, c, 0, 0, 0, cnt, tot, 1]))
            var oid = out._d.id
            _ = family_call(self.fn("x_decomp_dev_move"), oid, one, oid,
                            mk_ints([MV_FILL0, c, cnt, 1, 0, 0, 0, tot, tot, 1]))
            return out
        var store = mk_store_copy(mk_store(x, self.ctx), self.ctx)
        _ = py_call(store.extend, mk_store_zeros(c, self.ctx))
        return mk_new(store, r + 1, c, self.ctx)

    def dsum_sq(mut self, m: PythonObject) raises -> Float64:
        """`_dsum_sq(k, M)`: the binary64 sum of the squares of M's values."""
        var n = mk_n(m)
        if n == 0:
            return 0.0
        if self.use1(m):
            var out = self.dout(1, 2)
            var im = self.did(m)
            _ = family_call(self.fn("x_decomp_dev_dsum_sq"), im, out._d.id, mk_ints([n]))
            var store = mk_store(out, self.ctx)
            # the binary64 result's two float32 words, read as one double
            var v = FAMILY_F64P(unsafe_from_address=Int(py=mk_store_addr(store)))[0]
            _ = store^
            return v
        # cpu-route: host binding only (CPU-only installs); a GPU binding sums on the device above
        var pm = mk_addr(m, self.ctx)
        return Float64(py=py_call(Python.import_module("builtins").float,
                                  family_call(self.fn("x_decomp_dsum_sq"), pm, PythonObject(n))))

    # ---- neighbor graphs (x_decomp/graph_cells.mojo)
    def dijkstra(mut self, w: PythonObject) raises -> PythonObject:
        var n = mk_r(w)
        if self.res():
            var dist = self.dout(n, n)
            var iw = self.did(w)
            _ = family_call(self.fn("x_decomp_dev_graph_dijkstra"), iw, dist._d.id, mk_ints([n]))
            return dist
        var dist = mk_zeros(n, n, self.ctx)
        var pw = mk_addr(w, self.ctx)
        _ = family_call(self.fn("x_decomp_graph_dijkstra"), pw, mk_addr(dist, self.ctx), mk_ints([n]))
        return dist

    def graph_knn(mut self, d: PythonObject, nn: Int, exclude_self: Bool) raises -> Tuple[PythonObject, PythonObject]:
        var n = mk_r(d)
        var m = mk_c(d)
        var params = mk_ints([n, m, nn, 1 if exclude_self else 0])
        if self.res():
            var idx = self.dout(n, nn)
            var dst = self.dout(n, nn)
            if n * nn > 0:
                var id = self.did(d)
                _ = family_call(self.fn("x_decomp_dev_graph_knn"), id, idx._d.id, dst._d.id, params)
            return (idx, dst)
        var idx = mk_zeros(n, nn, self.ctx)
        var dst = mk_zeros(n, nn, self.ctx)
        if n * nn > 0:
            var pd = mk_addr(d, self.ctx)
            _ = family_call(self.fn("x_decomp_graph_knn"), pd, mk_addr(idx, self.ctx), mk_addr(dst, self.ctx), params)
        return (idx, dst)

    def graph_knn_dense(mut self, idx: PythonObject, w: PythonObject, n: Int) raises -> PythonObject:
        var nn = mk_c(idx)
        if self.res():
            var out = self.dout(n, n)
            var ii = self.did(idx)
            var iw = self.did(w)
            _ = family_call(self.fn("x_decomp_dev_graph_knn_dense"), ii, iw, out._d.id, mk_ints([n, nn]))
            return out
        var out = mk_zeros(n, n, self.ctx)
        var pi = mk_addr(idx, self.ctx)
        var pw = mk_addr(w, self.ctx)
        _ = family_call(self.fn("x_decomp_graph_knn_dense"), pi, pw, mk_addr(out, self.ctx), mk_ints([n, nn]))
        return out

    def graph_radius(mut self, d: PythonObject, r: Float64) raises -> PythonObject:
        var n = mk_r(d)
        if self.res():
            var out = self.dout(n, n)
            var id = self.did(d)
            _ = family_call(self.fn("x_decomp_dev_graph_radius"), id, out._d.id, mk_ints([n]), PythonObject(r))
            return out
        var out = mk_zeros(n, n, self.ctx)
        var pd = mk_addr(d, self.ctx)
        _ = family_call(self.fn("x_decomp_graph_radius"), pd, mk_addr(out, self.ctx), mk_ints([n]), PythonObject(r))
        return out

    def graph_radius_geo(mut self, dq: PythonObject, d: PythonObject, r: Float64) raises -> PythonObject:
        var nq = mk_r(dq)
        var n = mk_c(dq)
        if self.res():
            var out = self.dout(nq, n)
            if nq * n > 0:
                var iq = self.did(dq)
                var id = self.did(d)
                _ = family_call(self.fn("x_decomp_dev_graph_radius_geo"), iq, id, out._d.id, mk_ints([nq, n]),
                                PythonObject(r))
            return out
        var out = mk_zeros(nq, n, self.ctx)
        if nq * n > 0:
            var pq = mk_addr(dq, self.ctx)
            var pd = mk_addr(d, self.ctx)
            _ = family_call(self.fn("x_decomp_graph_radius_geo"), pq, pd, mk_addr(out, self.ctx), mk_ints([nq, n]),
                            PythonObject(r))
        return out

    def graph_lle_iw(mut self, idx: PythonObject, wb: PythonObject, n: Int) raises -> PythonObject:
        var nn = mk_c(idx)
        if self.res():
            var out = self.dout(n, n)
            var ii = self.did(idx)
            var iw = self.did(wb)
            _ = family_call(self.fn("x_decomp_dev_graph_lle_iw"), ii, iw, out._d.id, mk_ints([n, nn]))
            return out
        var out = mk_zeros(n, n, self.ctx)
        var pi = mk_addr(idx, self.ctx)
        var pw = mk_addr(wb, self.ctx)
        _ = family_call(self.fn("x_decomp_graph_lle_iw"), pi, pw, mk_addr(out, self.ctx), mk_ints([n, nn]))
        return out

    def graph_components(mut self, w: PythonObject) raises -> Tuple[PythonObject, Int]:
        var n = mk_r(w)
        var b = Python.import_module("builtins")
        if self.res():
            var comp = self.dout(n, 1)
            var iw = self.did(w)
            var c = family_call(self.fn("x_decomp_dev_graph_components"), iw, comp._d.id, mk_ints([n]))
            return (comp, Int(py=py_call(b.int, c)))
        var comp = mk_zeros(n, 1, self.ctx)
        var pw = mk_addr(w, self.ctx)
        var c = family_call(self.fn("x_decomp_graph_components"), pw, mk_addr(comp, self.ctx), mk_ints([n]))
        return (comp, Int(py=py_call(b.int, c)))

    def graph_join(mut self, w: PythonObject, d: PythonObject, comp: PythonObject, count: Int) raises:
        """sklearn `_fix_connected_components`, W changed in place."""
        var n = mk_r(w)
        if self.res():
            var iw = self.did(w)
            var id = self.did(d)
            var ic = self.did(comp)
            _ = family_call(self.fn("x_decomp_dev_graph_join"), iw, id, ic, mk_ints([n, count]))
            return
        var pw = mk_addr(w, self.ctx)
        var pd = mk_addr(d, self.ctx)
        var pc = mk_addr(comp, self.ctx)
        _ = family_call(self.fn("x_decomp_graph_join"), pw, pd, pc, mk_ints([n, count]))

    # ---- LLE cells
    def barycenter(mut self, x: PythonObject, y: PythonObject, nbr: PythonObject, reg: PythonObject) raises -> PythonObject:
        """sklearn barycenter_weights: (n x k) weights of each row of X on
        its k neighbors in Y."""
        var n = mk_r(x)
        var k = mk_c(nbr)
        var w = mk_zeros(n, k, self.ctx)
        var flags = mk_zeros(n, 1, self.ctx)
        var px = mk_addr(x, self.ctx)
        var py = mk_addr(y, self.ctx)
        var pi = mk_addr(nbr, self.ctx)
        _ = family_call(self.fn("x_decomp_barycenter_rows"), px, py, pi, mk_addr(w, self.ctx), mk_addr(flags, self.ctx),
                        mk_ints([n, mk_r(y), mk_c(x), k]), py_call(Python.import_module("builtins").float, reg))
        return w

    def lle_apply(mut self, wb: PythonObject, idm: PythonObject, e: PythonObject) raises -> PythonObject:
        """out (nq x nc) = sum_a Wb[i, a] E[idm[i, a]]."""
        var nq = mk_r(wb)
        var nn = mk_c(wb)
        var nc = mk_c(e)
        var out = mk_zeros(nq, nc, self.ctx)
        if nq * nc > 0:
            var pw = mk_addr(wb, self.ctx)
            var pi = mk_addr(idm, self.ctx)
            var pe = mk_addr(e, self.ctx)
            _ = family_call(self.fn("x_decomp_lle_apply"), pw, pi, pe, mk_addr(out, self.ctx),
                            mk_ints([nq, mk_r(e), nn, nc]))
        return out

    def lle_local(mut self, m: PythonObject, idm: PythonObject, method: Int, nn: Int, nc: Int,
                  tol: Float64) raises -> PythonObject:
        """LTSA (0), Hessian (1) or modified (2) LLE's stacked factor B."""
        var n = mk_r(m)
        var d = mk_c(m)
        var rows = n * (nn - 1 - nc) if method == 1 else n * nn
        var bm = mk_zeros(rows, n, self.ctx)
        var pm = mk_addr(m, self.ctx)
        var pi = mk_addr(idm, self.ctx)
        var tols = Python.list()
        mk_append(tols, PythonObject(tol))
        _ = family_call(self.fn("x_decomp_lle_local"), pm, pi, mk_addr(bm, self.ctx), mk_ints([method, n, d, nn, nc]),
                        tols)
        return bm


def mk_t(m: PythonObject, mut hk: MKit, ctx: PythonObject) raises -> PythonObject:
    """`m.T`. `hk` is `_host_kit()`, the default-mode kit: a host matrix on a
    GPU binding goes up once on it (it moves) and is transposed there."""
    var r = mk_r(m)
    var c = mk_c(m)
    var n = r * c
    var d = m._d
    if d is None and n > 0 and r != 1 and c != 1 and n < I32_MAX:
        if hk.res():
            _ = hk.did(m)
            d = m._d
    if not (d is None) and (r == 1 or c == 1):
        return mk_dev_view(d, c, r, ctx)
    if not (d is None) and n < I32_MAX:
        var out = mk_dev_view(mk_devbuf(d.b, n, ctx), c, r, ctx)
        _ = family_call(family_entry(d.b, "x_decomp_dev_move", ctx), d.id, d.id, out._d.id,
                        mk_ints([MV_TRANSPOSE, n, r, c, 0, 0, 0, n, n, 1]))
        return out
    if r == 1 or c == 1:
        return mk_new(mk_store(m, ctx), c, r, ctx)
    var out = mk_zeros(c, r, ctx)
    var pm = mk_addr(m, ctx)
    # cpu-route: host binding only (CPU-only installs); a GPU binding transposed above
    _ = family_call(hk.fn("x_decomp_move"), pm, mk_addr(mk_one(ctx), ctx), mk_addr(out, ctx),
                    mk_ints([MV_TRANSPOSE, n, r, c, 0, 0, 0, n, n, 1]))
    return out


def mk_take_rows_list(m: PythonObject, idx: List[Int], mut hk: MKit, ctx: PythonObject) raises -> PythonObject:
    """`m.take_rows(idx)` (a list of row numbers): the default-mode kit's
    move."""
    if mk_r(m) >= F32_INDEX_MAX:
        raise Error("x_decomp: take_rows exceeds the float32 row-index bound (2^24 rows)")
    var values = List[Float64]()
    for t in range(len(idx)):  # small-loop(idx: the taken row numbers): bounded by the caller's index list
        values.append(Float64(idx[t]))
    var im = mk_of(values, len(idx), 1, ctx)
    var count = len(idx)
    var c = mk_c(m)
    if count * c > 0 and hk.res():
        return hk.take_rows(m, im)
    var out = mk_zeros(count, c, ctx)
    if count * c > 0:
        var pm = mk_addr(m, ctx)
        var pi = mk_addr(im, ctx)
        # cpu-route: host binding only (CPU-only installs), the GPU binding moved above
        _ = family_call(hk.fn("x_decomp_move"), pm, pi, mk_addr(out, ctx),
                        mk_ints([MV_TAKE_ROWS, count * c, c, 0, 0, 1, 0, mk_store_len(mk_store(m, ctx)), count * c, count]))
    return out


def mk_from_input(source: PythonObject, name: String, ctx: PythonObject) raises -> PythonObject:
    """`_M.from_input(X, name)`: a scipy.sparse input densified exactly, the
    `as_f32_c` cast and refusals, one copy into a host store, and the
    finiteness refusal."""
    var a = family_as_f32_dense(source, 2, name, ctx)
    var r = Int(py=a.shape[0])
    var c = Int(py=a.shape[1])
    if Int(py=a.ndim) != 2 or r == 0 or c == 0:
        raise Error(name + ": a nonempty two-dimensional input is required")
    var store = mk_store_zeros(r * c, ctx)
    var src = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(a)))
    memcpy(dest=mk_store_ptr(store), src=src, count=r * c)
    if not family_all_finite_f32(src, r * c):
        raise Error(name + ": input must be finite; NaN/inf are unsupported")
    _ = a^
    return mk_new(store, r, c, ctx)
