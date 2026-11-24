# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`solve`, `lstsq`, `randomized_svd` and `sparse_encode` on TrainDeviceArray operands (lane vdf-linalg7, 2026-11-24):
the doors' RUN sequences (bindings/linalg_decomp_dev_door.mojo, bindings/dict_learning_native_unpack.mojo
`_sparse_encode_dev`) on the typed kit (bindings/decomp_kit_run.mojo), with every operand either adopted from a
training pool handle (`DKit.adopt`: XD_DEV_FROM_TRAIN + XD_DEV_FINITE; the pool-id operations XD_DEV_LU_GESV,
XD_DEV_LU_FACTOR_IN / XD_DEV_LU_SOLVE_IN, XD_DEV_TSQR_R with [X | B] and an R id, XD_RANDOMIZED_SVD with pool ids)
or borrowed from host memory (a host operand: the GPU image, and the host column x_decomp_host_v1 where the route
exists there). Every result word of the device operand must equal the host operand's on the GPU image and the host
column's, and every result copied into a training handle (`DKit.to_train`) must download to the same words. Also:
XD_DEV_R_SIGNS == XD_R_SIGNS word for word (negative diagonals, -0.0), XD_DEV_TSQR_R's R id == its host R, a
non-finite device operand refused in the host admission's words, and the adopted arrays left unchanged.

A fast build has no host column to match (`fast` promises no bits across columns): there the column is the host
operand on the GPU image again.
"""
from std.testing import TestSuite
from std.math import inf
from x_decomp.s_linalg_fast import LU_RESIDENT_ENTRIES
from core.entry_table import EntryRef, entry_ref_word, entry_ref_call
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.descriptor_schema import (
    XD_OPERATION_COUNT, XD_CAPABILITIES, XD_CAP_RESIDENT, XD_CAP_DEV_SVD, XD_R_SIGNS, XD_DEV_R_SIGNS, XD_TSQR_R,
    XD_DEV_TSQR_R, XD_RANDOMIZED_SVD, XD_DEV_LU_FACTOR, XD_DEV_LU_SOLVE,
)
from x_decomp.descriptor_mani_schema import XD_CAP_DEV_LU, XD_CAP_DEV_EIGH
from x_decomp.descriptor_device import x_decomp_device_v1
from x_decomp.descriptor_host import x_decomp_host_v1
from x_decomp.device import xd_ctx
from x_decomp.resident import IDN_CODE_RESIDENT
from core.train_dev_pool import train_dev_alloc, train_dev_free, train_dev_put, train_dev_get
from bindings.decomp_kit_run import DKit, DK_HAS_DEV_SVD, DK_HAS_DEV_LU, DK_HAS_DEV_EIGH_TO, DK_HAS_DEV_CODE_ROWS
from bindings.linalg_decomp_run import (
    LstsqOut, ld_gesv_run, ld_dev_gesv_run, ld_dev_lu_pair_run, ld_device_buffers, ld_lstsq_run,
)
from bindings.dict_learning_run import EncodeWords, Gram, dl_sparse_encode

comptime _FP = MutPointer[Float32, MutUntrackedOrigin]

comptime HOST_OP = 0    # host operand, GPU image
comptime DEVICE_OP = 1  # training device array, GPU image
comptime COLUMN = 2     # host operand, host column


def _kit(source: Int) raises -> DKit:
    var word = entry_ref_word(Int(GLOBAL_NUMERIC_MODE), 0, XD_OPERATION_COUNT)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if source == COLUMN:
            return DKit(EntryRef(x_decomp_host_v1, word), True, False, 0)
    var entry = EntryRef(x_decomp_device_v1, word)
    var caps = entry_ref_call(entry, XD_CAPABILITIES, List[Int](), List[Int](), List[Float64]())
    # the optional-entry bits the door's UNPACK reads from the image's entries (bindings/decomp_kit_native_door.mojo
    # `decomp_kit_unpack`)
    var present = 0
    if (caps & XD_CAP_DEV_SVD) != 0:
        present |= DK_HAS_DEV_SVD
    if (caps & XD_CAP_DEV_LU) != 0:
        present |= DK_HAS_DEV_LU
    if (caps & XD_CAP_DEV_EIGH) != 0:
        present |= DK_HAS_DEV_EIGH_TO
    comptime if IDN_CODE_RESIDENT:
        present |= DK_HAS_DEV_CODE_ROWS
    return DKit(entry, True, (caps & XD_CAP_RESIDENT) != 0, present)


def _fixture(rows: Int, cols: Int, salt: Int = 0, diagonal: Float32 = 0.0) -> List[Float32]:
    var a = List[Float32](length=rows * cols, fill=0.0)
    for i in range(rows * cols):  # small-loop: a small fixture
        a[i] = Float32(((i * 53 + 7 + salt * 31) % 89) - 44) / Float32(8.0)
    if diagonal != 0.0:
        for i in range(min(rows, cols)):  # small-loop: a small fixture
            a[i * cols + i] += diagonal
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


def _take(mut kit: DKit, source: Int, a: Operand, name: String) raises -> Int:
    if source == DEVICE_OP:
        return kit.adopt(a.handle, a.rows, a.cols, name)
    return kit.borrow(Int(a.words.unsafe_ptr()), a.rows, a.cols)


def _words(mut kit: DKit, m: Int, source: Int) raises -> List[Float32]:
    """Matrix m's words; for the device operand also through a training handle (the PACK), which must agree."""
    var count = kit.size(m)
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


def _three(label: String, device: List[Float32], host: List[Float32], column: List[Float32]) raises:
    _same(label + ": device operand vs host operand", device, host)
    _same(label + ": device operand vs host column", device, column)


def _solve_gesv(source: Int, a: Operand, b: Operand) raises -> List[Float32]:
    var kit = _kit(source)
    try:
        var am = _take(kit, source, a, "a")
        var bm = _take(kit, source, b, "b")
        var solved = (0, 0)
        if source == DEVICE_OP:
            solved = ld_dev_gesv_run(kit, am, bm)
        else:
            solved = ld_gesv_run(kit, am, bm)
        if solved[1] != 0:
            raise Error("the gesv fixture must be regular")
        return _words(kit, solved[0], source)
    finally:
        kit.close()


def _solve_pair(source: Int, a: Operand, b: Operand, refine: Bool) raises -> List[Float32]:
    """The resident LU pair: the device form (XD_DEV_LU_FACTOR_IN / _SOLVE_IN) or the host operand's
    XD_DEV_LU_FACTOR / XD_DEV_LU_SOLVE on the GPU image (the pair is a GPU route: no host column)."""
    var kit = _kit(source)
    try:
        var n = a.rows
        var nrhs = b.cols
        if source == DEVICE_OP:
            var am = _take(kit, source, a, "a")
            var bm = _take(kit, source, b, "b")
            var pair = ld_dev_lu_pair_run(kit, am, bm, refine)
            if pair[0] < 0:
                raise Error("the pair fixture must be regular")
            return _words(kit, pair[0], source)
        var ids = ld_device_buffers(kit, n)
        var lu = List[Float32](length=n * n, fill=0.0)
        var piv = List[Float32](length=n, fill=0.0)
        var info = List[Float32](length=1, fill=0.0)
        var addresses: List[Int] = [Int(a.words.unsafe_ptr()), Int(lu.unsafe_ptr()), Int(piv.unsafe_ptr()),
                                    Int(info.unsafe_ptr())]
        _ = kit.xd(XD_DEV_LU_FACTOR, addresses, [ids[0], ids[1], ids[2], n], List[Float64]())
        var x = List[Float32](length=n * nrhs, fill=0.0)
        var solve: List[Int] = [Int(b.words.unsafe_ptr()), Int(x.unsafe_ptr())]
        _ = kit.xd(XD_DEV_LU_SOLVE, solve, [ids[1], ids[2], ids[0], n, nrhs, 1 if refine else 0], List[Float64]())
        _ = lu^  # keep the factor's host words alive through the calls above
        _ = piv^
        _ = info^
        return x^
    finally:
        kit.close()


def _lstsq(source: Int, a: Operand, b: Operand) raises -> List[Float32]:
    var kit = _kit(source)
    try:
        var am = _take(kit, source, a, "a")
        var bm = _take(kit, source, b, "b")
        var rcond = 1.1920928955078125e-07 * Float64(max(a.rows, a.cols))
        var out = ld_lstsq_run(kit, am, bm, rcond)
        var words = _words(kit, out.x, source)
        words.extend(_words(kit, out.s, source))
        if out.resid >= 0:
            words.extend(_words(kit, out.resid, source))
        words.append(Float32(out.rank))
        return words^
    finally:
        kit.close()


def _rsvd(source: Int, m: Operand, kc: Int, transpose: Bool) raises -> List[Float32]:
    var kit = _kit(source)
    try:
        var rows = m.rows
        var cols = m.cols
        var params: List[Int] = [rows, cols, kc, 4, 2, 0, Int(transpose), 1, 11, 0, -1, -1, -1, -1]
        if source == DEVICE_OP:
            var x = _take(kit, source, m, "M")
            var u = kit.device_output(rows, kc)
            var s = kit.device_output(1, kc)
            var vt = kit.device_output(kc, cols)
            params[10] = kit.device(x)
            params[11] = kit.matrices[u].device
            params[12] = kit.matrices[s].device
            params[13] = kit.matrices[vt].device
            _ = kit.xd(XD_RANDOMIZED_SVD, [0, 0, 0, 0], params, List[Float64]())
            var out = _words(kit, u, source)
            out.extend(_words(kit, s, source))
            out.extend(_words(kit, vt, source))
            return out^
        var u = List[Float32](length=rows * kc, fill=0.0)
        var s = List[Float32](length=kc, fill=0.0)
        var vt = List[Float32](length=kc * cols, fill=0.0)
        var addresses: List[Int] = [Int(m.words.unsafe_ptr()), Int(u.unsafe_ptr()), Int(s.unsafe_ptr()),
                                    Int(vt.unsafe_ptr())]
        _ = kit.xd(XD_RANDOMIZED_SVD, addresses, params, List[Float64]())
        u.extend(s.copy())
        u.extend(vt.copy())
        return u^
    finally:
        kit.close()


def _encode(source: Int, x: Operand, d: Operand, algorithm: Int) raises -> List[Float32]:
    """`dl_sparse_encode`: 1 lasso_cd, 3 omp (two atoms), 4 threshold (alpha 0.5)."""
    var kit = _kit(source)
    try:
        var xm = _take(kit, source, x, "X")
        var dm = _take(kit, source, d, "dictionary")
        var words = EncodeWords(algorithm, 2 if algorithm == 3 else 0, 0.5 if algorithm != 3 else 1.0,
                                200 if algorithm == 1 else 0, 0, False, False)
        var gram = Gram(-1)
        var code = dl_sparse_encode(kit, xm, dm, words, -1, gram)
        return _words(kit, code, source)
    finally:
        kit.close()


def test_r_signs_device() raises:
    """XD_DEV_R_SIGNS == XD_R_SIGNS word for word: negative, positive and -0.0 diagonals."""
    var n = 5
    var r = _fixture(n, n)
    r[0] = -2.0
    r[1 * n + 1] = 3.0
    r[2 * n + 2] = Float32(-0.0)
    r[3 * n + 3] = -0.25
    r[4 * n + 4] = 1.5
    var host_r = r.copy()
    var host_c = List[Float32](length=n * n, fill=7.0)
    var kit = _kit(HOST_OP)
    try:
        var addresses: List[Int] = [Int(host_r.unsafe_ptr()), Int(host_c.unsafe_ptr())]
        _ = kit.xd(XD_R_SIGNS, addresses, [n], List[Float64]())
        var dr = kit.borrow(Int(r.unsafe_ptr()), n, n)
        var rid = kit.device(dr)
        var cm = kit.device_output(n, n)
        _ = kit.xd(XD_DEV_R_SIGNS, List[Int](), [rid, kit.matrices[cm].device, n], List[Float64]())
        kit.wrote_device(dr)
        _ = r^  # keep the borrowed words alive through the upload above
        _same("R signs: R", _words(kit, dr, HOST_OP), host_r)
        _same("R signs: C", _words(kit, cm, HOST_OP), host_c)
    finally:
        kit.close()


def test_tsqr_r_device_id() raises:
    """XD_DEV_TSQR_R with an R pool id (R never on the host) == its host R, with and without B beside X."""
    var x = Operand(_fixture(700, 5), 700, 5)
    var b = Operand(_fixture(700, 2, 3), 700, 2)
    for nrhs in range(0, 3, 2):  # small-loop: no right-hand side, then two
        var n = 5 + nrhs
        var kit = _kit(HOST_OP)
        try:
            var xm = _take(kit, DEVICE_OP, x, "X")
            var bm = _take(kit, DEVICE_OP, b, "b")
            var dx = kit.device(xm)
            var db = kit.device(bm)
            var host_r = List[Float32](length=n * n, fill=0.0)
            _ = kit.xd(XD_DEV_TSQR_R, [Int(host_r.unsafe_ptr())], [dx, 700, 5, 0, -1, db, nrhs], List[Float64]())
            var rm = kit.device_output(n, n)
            _ = kit.xd(XD_DEV_TSQR_R, [0], [dx, 700, 5, 0, kit.matrices[rm].device, db, nrhs], List[Float64]())
            _same("tsqr R id nrhs " + String(nrhs), _words(kit, rm, HOST_OP), host_r)
            # and the host operand's XD_TSQR_R
            var col = List[Float32](length=n * n, fill=0.0)
            var addresses: List[Int] = [Int(x.words.unsafe_ptr()), Int(b.words.unsafe_ptr()), Int(col.unsafe_ptr())]
            _ = kit.xd(XD_TSQR_R, addresses, [700, 5, nrhs, 0], List[Float64]())
            _same("tsqr R device vs host operand nrhs " + String(nrhs), host_r, col)
        finally:
            kit.close()
    x.unchanged("tsqr X")
    b.unchanged("tsqr B")
    x.close()
    b.close()


def test_solve_device_operands() raises:
    var a = Operand(_fixture(6, 6, 1, 20.0), 6, 6)
    var b = Operand(_fixture(6, 3, 2), 6, 3)
    var v = Operand(_fixture(6, 1, 4), 6, 1)  # a 1-D b: the n x 1 column of its words
    _three("solve gesv", _solve_gesv(DEVICE_OP, a, b), _solve_gesv(HOST_OP, a, b), _solve_gesv(COLUMN, a, b))
    _three("solve gesv vector", _solve_gesv(DEVICE_OP, a, v), _solve_gesv(HOST_OP, a, v), _solve_gesv(COLUMN, a, v))
    comptime if LU_RESIDENT_ENTRIES:
        for ri in range(2):  # small-loop: IDN_LU_RESIDENT_PAIR (no refinement), LU_FAST_RESIDENT (one step)
            var refine = ri == 1
            _same("solve pair refine " + String(refine), _solve_pair(DEVICE_OP, a, b, refine),
                  _solve_pair(HOST_OP, a, b, refine))
    a.unchanged("solve a")
    b.unchanged("solve b")
    a.close()
    b.close()
    v.close()


def test_lstsq_device_operands() raises:
    var tall = Operand(_fixture(40, 4, 1), 40, 4)
    var tb = Operand(_fixture(40, 2, 5), 40, 2)
    var short = Operand(_fixture(5, 4, 2), 5, 4)  # m < n + nrhs: the thin SVD route
    var sb = Operand(_fixture(5, 2, 6), 5, 2)
    _three("lstsq TSQR", _lstsq(DEVICE_OP, tall, tb), _lstsq(HOST_OP, tall, tb), _lstsq(COLUMN, tall, tb))
    _three("lstsq thin SVD", _lstsq(DEVICE_OP, short, sb), _lstsq(HOST_OP, short, sb), _lstsq(COLUMN, short, sb))
    tall.unchanged("lstsq a")
    tb.unchanged("lstsq b")
    tall.close()
    tb.close()
    short.close()
    sb.close()


def test_randomized_svd_device_operand() raises:
    var m = Operand(_fixture(60, 12, 3), 60, 12)
    for ti in range(2):  # small-loop: the sketch on M, then on M^T
        var transpose = ti == 1
        _three("rsvd transpose " + String(transpose), _rsvd(DEVICE_OP, m, 3, transpose),
               _rsvd(HOST_OP, m, 3, transpose), _rsvd(COLUMN, m, 3, transpose))
    m.unchanged("rsvd M")
    m.close()


def test_sparse_encode_device_operands() raises:
    var x = Operand(_fixture(30, 8, 1), 30, 8)
    var d = Operand(_fixture(5, 8, 2), 5, 8)
    var algorithms: List[Int] = [1, 3, 4]
    for ai in range(len(algorithms)):  # small-loop: lasso_cd, omp, threshold
        var algorithm = algorithms[ai]
        _three("sparse_encode " + String(algorithm), _encode(DEVICE_OP, x, d, algorithm),
               _encode(HOST_OP, x, d, algorithm), _encode(COLUMN, x, d, algorithm))
    x.unchanged("sparse_encode X")
    d.unchanged("sparse_encode dictionary")
    x.close()
    d.close()


def test_device_operand_finite_refusal() raises:
    var words = _fixture(4, 4)
    words[5] = inf[DType.float32]()
    var a = Operand(words^, 4, 4)
    var kit = _kit(DEVICE_OP)
    try:
        try:
            _ = kit.adopt(a.handle, 4, 4, "a")
            raise Error("a non-finite device operand was not refused")
        except problem:
            if String(problem) != "a: input must be finite; NaN/inf are unsupported":
                raise Error("finite refusal words: " + String(problem))
    finally:
        kit.close()
    a.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
