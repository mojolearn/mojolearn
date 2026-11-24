# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`linalg.eigh`, `qr` ('reduced' on and off the TSQR route, 'complete', 'raw', a wide 'r') and `svd` with compute_uv
(TSQR economy, tall full, wide, hermitian) on a TrainDeviceArray (lane vdf-linalg6, 2026-11-24): the door's
sequences (bindings/linalg_ops_decomp_door.mojo) on the typed kit (bindings/linalg_ops_kit_run.mojo), with the
operand either borrowed from host memory (a host operand, the GPU image), adopted from a training pool handle
(`LinKit.adopt`, XD_DEV_FROM_TRAIN; the TSQR on pool ids XD_DEV_TSQR_R / XD_DEV_TSQR_Q) or borrowed on the host
column (x_decomp_host_v1). Every result word of the device operand must equal the host operand's and the host
column's, and each result copied into a training handle (`LinKit.to_train`, XD_DEV_TO_TRAIN) must download to the
same words. The adopted training array must be left unchanged.
"""
from core.entry_table import EntryRef, entry_ref_word, entry_ref_call
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.descriptor_schema import (
    XD_OPERATION_COUNT, XD_CAPABILITIES, XD_TSQR_R, XD_TSQR_Q, XD_R_SIGNS, XD_DEV_TSQR_R, XD_DEV_TSQR_Q,
)
from x_decomp.descriptor_device import x_decomp_device_v1
from x_decomp.descriptor_host import x_decomp_host_v1
from x_decomp.device import xd_ctx
from core.train_dev_pool import train_dev_alloc, train_dev_free, train_dev_put, train_dev_get
from bindings.linalg_ops_kit_run import LinKit, lin_svd_tall_run, lin_svd_hermitian_run, _lin_tsqr_on
from std.testing import TestSuite

comptime _FP = MutPointer[Float32, MutUntrackedOrigin]
comptime RTOL = 1.1920928955078125e-07  # a fixed rank cut for the check (the door reads the module's)

#: operand sources
comptime HOST_OP = 0    # host operand, GPU image
comptime DEVICE_OP = 1  # training device array, GPU image
comptime COLUMN = 2     # host operand, host column


def _kit(source: Int) raises -> LinKit:
    var word = entry_ref_word(Int(GLOBAL_NUMERIC_MODE), 0, XD_OPERATION_COUNT)
    if source == COLUMN:
        return LinKit(EntryRef(x_decomp_host_v1, word), 0)
    var entry = EntryRef(x_decomp_device_v1, word)
    var caps = entry_ref_call(entry, XD_CAPABILITIES, List[Int](), List[Int](), List[Float64]())
    return LinKit(entry, caps)


def _fixture(rows: Int, cols: Int, symmetric: Bool = False) -> List[Float32]:
    var a = List[Float32](length=rows * cols, fill=0.0)
    for i in range(rows * cols):  # small-loop: a small fixture
        a[i] = Float32(((i * 53 + 7) % 89) - 44) / Float32(8.0)
    if symmetric:
        for i in range(rows):  # small-loop: a small fixture
            for j in range(i):  # small-loop: a small fixture
                a[i * cols + j] = a[j * cols + i]
    return a^


struct Operand(Movable):
    """The fixture in host memory and in a training handle (the pool keeps it until `close`)."""
    var words: List[Float32]
    var handle: Int
    var rows: Int
    var cols: Int

    def __init__(out self, var words: List[Float32], rows: Int, cols: Int) raises:
        self.words = words^
        self.rows = rows
        self.cols = cols
        var ctx = xd_ctx()
        self.handle = train_dev_alloc(ctx, rows * cols)
        train_dev_put(ctx, self.handle, _FP(unsafe_from_address=Int(self.words.unsafe_ptr())), rows * cols)

    def unchanged(self, label: String) raises:
        var back = List[Float32](length=self.rows * self.cols, fill=0.0)
        train_dev_get(xd_ctx(), self.handle, _FP(unsafe_from_address=Int(back.unsafe_ptr())), self.rows * self.cols)
        _same(label + " (the adopted training array)", back, self.words)

    def close(self) raises:
        train_dev_free(xd_ctx(), self.handle)


def _take(mut kit: LinKit, source: Int, a: Operand) raises -> Int:
    if source == DEVICE_OP:
        return kit.adopt(a.handle, a.rows, a.cols)
    return kit.borrow(Int(a.words.unsafe_ptr()), a.rows, a.cols)


def _words(mut kit: LinKit, m: Int, source: Int) raises -> List[Float32]:
    """Matrix m's words; for the device operand also through a training handle (the PACK), which must agree."""
    var count = kit.r(m) * kit.c(m)
    var out = List[Float32](length=max(count, 1), fill=0.0)
    if source == DEVICE_OP and count > 0:
        var ctx = xd_ctx()
        var h = train_dev_alloc(ctx, count)
        kit.to_train(m, h)
        train_dev_get(ctx, h, _FP(unsafe_from_address=Int(out.unsafe_ptr())), count)
        train_dev_free(ctx, h)
        var direct = List[Float32](length=count, fill=0.0)
        var src = kit.ptr(m)
        for i in range(count):  # small-loop: a small result
            direct[i] = src[i]
        _same("to_train vs the kit's words", out, direct)
        return out^
    var src = kit.ptr(m)
    for i in range(count):  # small-loop: a small result
        out[i] = src[i]
    return out^


def _same(label: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        raise Error(label + ": lengths differ")
    for i in range(len(a)):  # small-loop: a small result
        if a[i].to_bits() != b[i].to_bits():
            raise Error(label + ": word " + String(i) + " differs")


def _eigh(source: Int, a: Operand, uplo: Int) raises -> List[Float32]:
    var kit = _kit(source)
    try:
        var v = 0
        var x = _take(kit, source, a)
        var w = kit.eigh(x, uplo, v)
        var out = _words(kit, w, source)
        out.extend(_words(kit, v, source))
        return out^
    finally:
        kit.close()


def _qr_kit(source: Int, a: Operand, mode: Int) raises -> List[Float32]:
    """mode 0 reduced (off the TSQR), 1 complete, 2 raw, 3 wide 'r': `_lin_qr_kit`."""
    var kit = _kit(source)
    try:
        var tau = 0
        var x = _take(kit, source, a)
        var h = kit.geqrf(x, tau)
        var kk = min(a.rows, a.cols)
        var out = List[Float32]()
        if mode == 2:
            var ht = kit.transpose(h)
            out = _words(kit, ht, source)
            out.extend(_words(kit, tau, source))
        elif mode == 3:
            var rw = kit.triu(h, a.rows)
            out = _words(kit, rw, source)
        else:
            var q = kit.orgqr(h, tau, a.rows if mode == 1 else kk)
            var r = kit.triu(h, a.rows if mode == 1 else kk)
            out = _words(kit, q, source)
            out.extend(_words(kit, r, source))
        return out^
    finally:
        kit.close()


def _qr_tsqr(source: Int, a: Operand) raises -> List[Float32]:
    """`_qr_tsqr` (host operand: XD_TSQR_R / XD_R_SIGNS / XD_TSQR_Q; device: `_lin_qr_tsqr_dev`)."""
    var rows = a.rows
    var cols = a.cols
    var kit = _kit(source)
    try:
        var r = kit.zeros(cols, cols)
        var signs = kit.zeros(cols, cols)
        var pr = kit.addr(r)
        var ps = kit.addr(signs)
        var owner = 0
        var q = -1
        if source == DEVICE_OP:
            var x = _take(kit, source, a)
            owner = kit.call(XD_DEV_TSQR_R, [pr], [kit.device(x), rows, cols, 1], List[Float64]())
            _ = kit.call(XD_R_SIGNS, [pr, ps], [cols], List[Float64]())
            q = kit.device_output(rows, cols)
            _ = kit.call(XD_DEV_TSQR_Q, [ps], [owner, -1, kit.matrices[q].device, rows, cols, cols], List[Float64]())
        else:
            var pa = Int(a.words.unsafe_ptr())
            owner = kit.call(XD_TSQR_R, [pa, pa, pr], [rows, cols, 0, 1], List[Float64]())
            _ = kit.call(XD_R_SIGNS, [pr, ps], [cols], List[Float64]())
            q = kit.zeros(rows, cols)
            _ = kit.call(XD_TSQR_Q, [ps, kit.addr(q)], [rows, cols, cols, owner], List[Float64]())
        var out = _words(kit, q, source)
        out.extend(_words(kit, r, source))
        return out^
    finally:
        kit.close()


def _svd_tsqr(source: Int, a: Operand) raises -> List[Float32]:
    """`_svd_tsqr` (host operand: U_R downloaded and handed to XD_TSQR_Q; device: `_lin_svd_tsqr_dev`)."""
    var rows = a.rows
    var cols = a.cols
    var kit = _kit(source)
    try:
        var r = kit.zeros(cols, cols)
        var pr = kit.addr(r)
        var s = 0
        var vt = 0
        var u = -1
        if source == DEVICE_OP:
            var x = _take(kit, source, a)
            var owner = kit.call(XD_DEV_TSQR_R, [pr], [kit.device(x), rows, cols, 1], List[Float64]())
            var ur = lin_svd_tall_run(kit, r, False, RTOL, s, vt)
            var dc = kit.device(ur)
            u = kit.device_output(rows, cols)
            _ = kit.call(XD_DEV_TSQR_Q, [pr], [owner, dc, kit.matrices[u].device, rows, cols, cols], List[Float64]())
        else:
            var pa = Int(a.words.unsafe_ptr())
            var owner = kit.call(XD_TSQR_R, [pa, pa, pr], [rows, cols, 0, 1], List[Float64]())
            var ur = lin_svd_tall_run(kit, r, False, RTOL, s, vt)
            var pc = kit.addr(ur)
            u = kit.zeros(rows, cols)
            _ = kit.call(XD_TSQR_Q, [pc, kit.addr(u)], [rows, cols, cols, owner], List[Float64]())
        var out = _words(kit, u, source)
        out.extend(_words(kit, s, source))
        out.extend(_words(kit, vt, source))
        return out^
    finally:
        kit.close()


def _svd_kit(source: Int, a: Operand, full: Bool, hermitian: Bool) raises -> List[Float32]:
    """`_lin_svd_kit`: hermitian, tall, or wide (through the transpose)."""
    var kit = _kit(source)
    try:
        var s = 0
        var first = 0
        var last = 0
        var m = _take(kit, source, a)
        if hermitian:
            first = lin_svd_hermitian_run(kit, m, a.rows, s, last)
        elif a.rows >= a.cols:
            first = lin_svd_tall_run(kit, m, full, RTOL, s, last)
        else:
            var vt2 = 0
            var mt = kit.transpose(m)
            var u2 = lin_svd_tall_run(kit, mt, full, RTOL, s, vt2)
            first = kit.transpose(vt2)
            last = kit.transpose(u2)
        var out = _words(kit, first, source)
        out.extend(_words(kit, s, source))
        out.extend(_words(kit, last, source))
        return out^
    finally:
        kit.close()


def _three(label: String, device: List[Float32], host: List[Float32], column: List[Float32]) raises:
    _same(label + ": device operand vs host operand", device, host)
    _same(label + ": device operand vs host column", device, column)


def test_eigh_device_operand() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var sizes: List[Int] = [3, 8, 40]
    for si in range(len(sizes)):  # small-loop: three fixture sizes (one block, the round-robin Jacobi, the tridiagonal route)
        var n = sizes[si]
        var a = Operand(_fixture(n, n, True), n, n)
        for uplo in range(1, 3):  # small-loop: 'L' (1) and 'U' (2)
            var tag = "eigh " + String(n) + " uplo " + String(uplo)
            _three(tag, _eigh(DEVICE_OP, a, uplo), _eigh(HOST_OP, a, uplo), _eigh(COLUMN, a, uplo))
        a.unchanged("eigh")
        a.close()


def test_qr_device_operand() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var tall = Operand(_fixture(9, 4), 9, 4)
    var wide = Operand(_fixture(4, 9), 4, 9)
    for mode in range(1, 3):  # small-loop: complete (1), raw (2)
        _three("qr tall mode " + String(mode), _qr_kit(DEVICE_OP, tall, mode), _qr_kit(HOST_OP, tall, mode),
               _qr_kit(COLUMN, tall, mode))
    for mode in range(4):  # small-loop: reduced (wide: off the TSQR), complete, raw, wide 'r'
        _three("qr wide mode " + String(mode), _qr_kit(DEVICE_OP, wide, mode), _qr_kit(HOST_OP, wide, mode),
               _qr_kit(COLUMN, wide, mode))
    if not _lin_tsqr_on(9, 4):
        raise Error("the 9 x 4 fixture must take the TSQR route")
    _three("qr reduced TSQR", _qr_tsqr(DEVICE_OP, tall), _qr_tsqr(HOST_OP, tall), _qr_tsqr(COLUMN, tall))
    var many = Operand(_fixture(700, 5), 700, 5)
    _three("qr reduced TSQR 700 x 5", _qr_tsqr(DEVICE_OP, many), _qr_tsqr(HOST_OP, many), _qr_tsqr(COLUMN, many))
    tall.unchanged("qr")
    wide.unchanged("qr")
    many.unchanged("qr")
    tall.close()
    wide.close()
    many.close()


def test_svd_device_operand() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var tall = Operand(_fixture(9, 4), 9, 4)
    var wide = Operand(_fixture(4, 9), 4, 9)
    var square = Operand(_fixture(5, 5, True), 5, 5)
    _three("svd TSQR economy", _svd_tsqr(DEVICE_OP, tall), _svd_tsqr(HOST_OP, tall), _svd_tsqr(COLUMN, tall))
    _three("svd tall full", _svd_kit(DEVICE_OP, tall, True, False), _svd_kit(HOST_OP, tall, True, False),
           _svd_kit(COLUMN, tall, True, False))
    for fi in range(2):  # small-loop: economy and full
        var full = fi == 1
        _three("svd wide full " + String(full), _svd_kit(DEVICE_OP, wide, full, False),
               _svd_kit(HOST_OP, wide, full, False), _svd_kit(COLUMN, wide, full, False))
    _three("svd hermitian", _svd_kit(DEVICE_OP, square, False, True), _svd_kit(HOST_OP, square, False, True),
           _svd_kit(COLUMN, square, False, True))
    tall.unchanged("svd")
    wide.unchanged("svd")
    square.unchanged("svd")
    tall.close()
    wide.close()
    square.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
