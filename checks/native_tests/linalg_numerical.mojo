# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Original linalg numerical cases, independent of the Python binding layer.

Inputs preserve NumPy seed 5 and every original dimension. QR uses its Gram
identity; the diagonal and duplicated-column expectations are analytic. The
constant-column spectrum is recorded from independent NumPy Float64 SVD.
No output hashes, repeated-execution comparison or alternative solver.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr
from decomposition.linalg_host_impl import qr_r_host_run, svdvals_host_run
from x_decomp.cells import F32Ptr
from x_decomp.host import HostExec
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from decomposition.linalg_door_impl import qr_r_device_into, svdvals_device_into
from x_decomp.device import DevExec
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from checks.native_tests.linalg_fixture_inputs import input_values
# test_numpy_contract (lane py2mojo-D6, 2026-11-22)
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word
from x_decomp.descriptor_schema import XD_OPERATION_COUNT, XD_CAPABILITIES, XD_TSQR_R, XD_TSQR_Q, XD_R_SIGNS
from x_decomp.descriptor_host import x_decomp_host_v1
from x_decomp.descriptor_device import x_decomp_device_v1
from bindings.linalg_ops_kit_run import LinKit, lin_svd_tall_run, lin_svd_hermitian_run
from bindings.linalg_ops_decomp_run import _lin_tsqr_on
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.testsuite_serial import check_cases_serial


# The column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds the host column).
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


comptime LINALG_NUMERICAL_FIXTURES = 5
comptime LINALG_GUARD = Float32(27182.5)


trait LinalgNumericalCalls:
    @staticmethod
    def qr(source: List[Float32], rows: Int, cols: Int, dst: Int) raises: ...
    @staticmethod
    def singular(source: List[Float32], rows: Int, cols: Int, dst: Int) raises: ...
    @staticmethod
    def eigen(source: Int, values: Int, vectors: Int, n: Int) raises: ...


def constant_reference() raises -> List[Float64]:
    var result = List[Float64](unsafe_uninit_length=10)
    with open("checks/native_tests/fixtures/linalg_numerical/constant-64x10-singular.f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 80, "linalg reference fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed-width file decoding, not a numerical oracle.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(byte * 8)
            rp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(word, 10)
        _ = bytes^
    return result^


def guards(value: List[Float32], name: String) raises:
    require(value[0] == LINALG_GUARD and value[len(value) - 1] == LINALG_GUARD, name + " output guards")


def source_unchanged(source: List[Float32], original: List[Float32]) raises:
    # Storage ownership assertion, not a repeatability/identity run.
    var count = len(source)
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = source.unsafe_ptr()
    var bp = original.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm bp}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](bp[index])))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, "linalg operation changed its caller's source")
    _ = source
    _ = original


def qr_expectations(source: List[Float32], result: List[Float32], rows: Int, cols: Int) raises:
    var failures = List[Int](length=cols * cols, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = source.unsafe_ptr()
    var rp = result.unsafe_ptr() + 1
    def cell(index: Int) {imm fp, imm ap, imm rp, imm rows, imm cols}:
        var i = index // cols
        var j = index % cols
        var expected = Float64(0)
        var actual = Float32(0)
        for row in range(rows):  # Independent Gram cell; columns/cells split across the host.
            expected += Float64(ap[row * cols + i]) * Float64(ap[row * cols + j])
        for row in range(cols):
            actual += rp[row * cols + i] * rp[row * cols + j]
        var triangular = i <= j or abs(rp[index]) <= Float32(1e-5)
        var close = abs(Float64(actual) - expected) <= Float64(1e-4) + Float64(1e-4) * abs(expected)
        fp.unsafe_store(index, Int(not triangular or not close))
    host_parallelize(cell, cols * cols)
    for failure in failures:
        require(failure == 0, "QR triangularity or original Gram tolerance")
    _ = source
    _ = result


def singular_expectations(result: List[Float32], reference: List[Float64], cols: Int, constant: Bool) raises:
    var failures = List[Int](length=cols, fill=0)
    var fp = failures.unsafe_ptr()
    var sp = result.unsafe_ptr() + 1
    var rp = reference.unsafe_ptr()
    def cell(index: Int) {imm fp, imm sp, imm rp, imm cols, imm constant}:
        var ok = index == 0 or sp[index - 1] >= sp[index]
        if constant:
            if index < 5:
                ok = ok and abs(Float64(sp[index]) - rp[index]) <= Float64(1e-4) * abs(rp[index]) + Float64(1e-5) * rp[0]
            else:
                ok = ok and abs(Float64(sp[index])) <= Float64(1e-3) * rp[0]
        elif index == cols - 1:
            ok = ok and abs(sp[index]) < Float32(1e-3)
        fp.unsafe_store(index, Int(not ok))
    host_parallelize(cell, cols)
    for failure in failures:
        require(failure == 0, "SVD ordering/rank/independent spectrum tolerance")
    _ = result
    _ = reference


def fixture[Calls: LinalgNumericalCalls](index: Int) raises:
    if index < 2:
        var rows = 48 if index == 0 else 16
        var cols = 5 if index == 0 else 4
        var source = input_values("qr-48x5" if index == 0 else "qr-16x4", rows * cols)
        var original = source.copy()
        var result = List[Float32](length=cols * cols + 2, fill=LINALG_GUARD)
        Calls.qr(source, rows, cols, Int(result.unsafe_ptr()) + 4)
        guards(result, "QR")
        qr_expectations(source, result, rows, cols)
        source_unchanged(source, original)
        _ = result^
    elif index == 2:
        var source: List[Float32] = [3, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 9]
        var original = source.copy()
        var values = List[Float32](length=6, fill=LINALG_GUARD)
        var vectors = List[Float32](length=18, fill=LINALG_GUARD)
        Calls.eigen(Int(source.unsafe_ptr()), Int(values.unsafe_ptr()) + 4, Int(vectors.unsafe_ptr()) + 4, 4)
        guards(values, "eigh values")
        guards(vectors, "eigh vectors")
        var expected: List[Float32] = [0.5, 2, 3, 9]
        var failures = List[Int](length=4, fill=0)
        var fp = failures.unsafe_ptr()
        var vp = values.unsafe_ptr() + 1
        var ep = expected.unsafe_ptr()
        def cell(at: Int) {imm fp, imm vp, imm ep}:
            fp.unsafe_store(at, Int(not (abs(vp[at] - ep[at]) <= Float32(1e-6))))
        host_parallelize(cell, 4)
        for failure in failures:
            require(failure == 0, "diagonal eigh must return ascending diagonal")
        source_unchanged(source, original)
        _ = expected^
        _ = values^
        _ = vectors^
    else:
        var rows = 48 if index == 3 else 64
        var cols = 4 if index == 3 else 10
        var source = List[Float32]()
        var reference = List[Float64](length=1, fill=0)
        if index == 3:
            var base = input_values("rank-48x3", 144)
            source = List[Float32](unsafe_uninit_length=192)
            var bp = base.unsafe_ptr()
            var sp = source.unsafe_ptr()
            def duplicate(at: Int) {imm bp, imm sp}:
                var column = at % 4
                sp.unsafe_store(at, bp[(at // 4) * 3 + (column if column < 2 else column - 1)])
            host_parallelize(duplicate, 192)
            _ = base^
        else:
            source = input_values("constant-64x10", 640)
            reference = constant_reference()
        var original = source.copy()
        var result = List[Float32](length=cols + 2, fill=LINALG_GUARD)
        Calls.singular(source, rows, cols, Int(result.unsafe_ptr()) + 4)
        guards(result, "svdvals")
        singular_expectations(result, reference, cols, index == 4)
        source_unchanged(source, original)
        _ = result^


# ---- The host column (was linalg_numerical_cpu_suite.mojo) ----
# Five original numerical fixtures over the actual CPU linalg entry bodies.
struct HostLinalgNumerical(LinalgNumericalCalls):
    @staticmethod
    def qr(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        qr_r_host_run(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def singular(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        svdvals_host_run(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def eigen(source: Int, values: Int, vectors: Int, n: Int) raises:
        HostExec.eigh(F32Ptr(unsafe_from_address=source), F32Ptr(unsafe_from_address=values),
                      F32Ptr(unsafe_from_address=vectors), n, 1)


def _host_fixture(index: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        fixture[HostLinalgNumerical](index)


# ---- The device column (was linalg_numerical_gpu_suite.mojo) ----
# Original fixtures on actual GPU linalg operations, with unchanged settings.
#
# The production device contexts/pools are shared, so graph submissions are
# ordered. Independent input decoding and numerical oracle cells use the host
# pool; the numerical device operations retain their existing parallel kernels.
struct DeviceLinalgNumerical(LinalgNumericalCalls):
    @staticmethod
    def qr(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        qr_r_device_into(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def singular(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        svdvals_device_into(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def eigen(source: Int, values: Int, vectors: Int, n: Int) raises:
        DevExec.eigh(F32Ptr(unsafe_from_address=source), F32Ptr(unsafe_from_address=values),
                     F32Ptr(unsafe_from_address=vectors), n, 1)


def _device_fixtures_in_order() raises:
    comptime if not CPU_COLUMN_BUILD:
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
        var failed = 0
        for index in range(LINALG_NUMERICAL_FIXTURES):  # shared device contexts: graph submissions in order
            try:
                fixture[DeviceLinalgNumerical](index)
            except error:
                failed += 1
                print("FAIL linalg_numerical gpu fixture", index, String(error))
        if failed:
            raise Error("linalg numerical GPU failed " + String(failed) + " of " + String(LINALG_NUMERICAL_FIXTURES) + " fixtures")


def test_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("fixture", LINALG_NUMERICAL_FIXTURES, _host_fixture)
    else:
        _device_fixtures_in_order()


# ---- NumPy's contract for qr, eigh, svdvals and svd (lane py2mojo-D6, 2026-11-22) ----
# Ported from python/mojolearn/tests/test_linalg_decompositions.py: `test_eigh_is_ascending_like_numpy_and_solves_
# its_own_equation` (unit 0), `test_eigh_reads_one_triangle_like_numpy` (1), `test_svdvals_is_descending_like_numpy`
# (2), `test_the_same_input_gives_the_same_bits` (3), `test_qr_modes_are_numpys` (4-7: 16x4, 48x5, 5x9, 6x6),
# `test_svd_is_numpys` (8-15: the same shapes, full_matrices True then False), `test_svd_rank_deficient_and_hermitian`
# (16) and `test_a_wide_matrix_takes_the_transpose` (17). Inputs are the original `_matrix(rows, cols)` draws (seed 5)
# and the references NumPy's float64 calls, stored as words in fixtures/linalg_numpy/ by
# tools/py2mojo_native_fixtures_b6.py. The routes are the door's (bindings/linalg_ops_decomp_door.mojo `_lin_qr`,
# `_lin_qr_q`, `_lin_eigh`, `lin_svdvals_of`, `_lin_svd`) on the typed x_decomp kit (bindings/linalg_ops_kit_run.mojo)
# and the linalg entries of the column, in the door's order; the tolerances and assertion senses are the original's
# (`_close`: max |a - b| <= tol max(1, max |b|); np.allclose: |a - e| <= atol + rtol |e|). What stays in Python: the
# refusals (UPLO 'X', unknown qr mode, float64, non-square), the returned Array / QRResult / SVDResult objects, the
# 'UPLO defaults to L' keyword default, and `compute_uv=False` returning svdvals (the door calls the same function).
# The units run one after another on both columns: the TSQR route keeps its factorization in ONE process slot between
# XD_TSQR_R and XD_TSQR_Q (x_decomp/tsqr_host.mojo), so two units on that route may not overlap, as two public calls
# may not; the matrix oracles inside a unit run across the host cores.
comptime NP_DIR = "checks/native_tests/fixtures/linalg_numpy/"
comptime NUMPY_CONTRACT_UNITS = 18
# python/mojolearn/_linalg_impl.py `_SVD_NULL_RTOL_MODULE` (2 ** -20), the rank cut the door reads for `_svd_tall`
comptime SVD_NULL_RTOL = Float64(1.0) / Float64(1048576.0)


struct Mat(Movable):
    """A row-major float64 matrix of the oracle side (the kit's float32 words widened exactly)."""
    var v: List[Float64]
    var r: Int
    var c: Int

    def __init__(out self, r: Int, c: Int):
        self.v = List[Float64](length=max(r * c, 1), fill=0.0)
        self.r = r
        self.c = c

    def at(self, i: Int, j: Int) -> Float64:
        return self.v[i * self.c + j]


@fieldwise_init
struct MatPair(Movable):
    var first: Mat
    var second: Mat


@fieldwise_init
struct SvdParts(Movable):
    var u: Mat
    var s: Mat
    var vh: Mat


def mat_f32(values: List[Float32], r: Int, c: Int) -> Mat:
    var m = Mat(r, c)
    var vp = values.unsafe_ptr()
    var mp = m.v.unsafe_ptr()
    def cell(i: Int) {imm vp, imm mp}:
        mp[i] = Float64(vp[i])
    host_parallelize(cell, r * c)
    return m^


def mat_ref(name: String, r: Int, c: Int) raises -> Mat:
    var m = Mat(r, c)
    m.v = fixture_f64(NP_DIR + name, r * c)
    return m^


def load_x(rows: Int, cols: Int) raises -> List[Float32]:
    return fixture_f32(NP_DIR + "x-" + String(rows) + "x" + String(cols) + ".f32", rows * cols)


def kit_words(mut kit: LinKit, a: Int) raises -> List[Float32]:
    var count = kit.r(a) * kit.c(a)
    var words = List[Float32](length=max(count, 1), fill=0.0)
    var src = kit.ptr(a)
    var dp = words.unsafe_ptr()
    def cell(i: Int) {imm src, imm dp}:
        dp[i] = src[i]
    host_parallelize(cell, count)
    return words^


def mat_kit(mut kit: LinKit, a: Int) raises -> Mat:
    var rows = kit.r(a)
    var cols = kit.c(a)
    var words = kit_words(kit, a)
    return mat_f32(words, rows, cols)


def mm(a: Mat, ta: Bool, b: Mat, tb: Bool) -> Mat:
    """op(a) @ op(b) in float64, one independent output cell per task."""
    var rows = a.c if ta else a.r
    var inner = a.r if ta else a.c
    var cols = b.r if tb else b.c
    var res = Mat(rows, cols)
    var ap = a.v.unsafe_ptr()
    var bp = b.v.unsafe_ptr()
    var rp = res.v.unsafe_ptr()
    var ac = a.c
    var bc = b.c
    def cell(index: Int) {imm ap, imm bp, imm rp, imm inner, imm cols, imm ac, imm bc, imm ta, imm tb}:
        var i = index // cols
        var j = index % cols
        var total = Float64(0)
        for k in range(inner):  # the one output cell's dot product
            var x = ap[k * ac + i] if ta else ap[i * ac + k]
            var y = bp[j * bc + k] if tb else bp[k * bc + j]
            total += x * y
        rp[index] = total
    host_parallelize(cell, rows * cols)
    return res^


def eye(n: Int) -> Mat:
    var m = Mat(n, n)
    for i in range(n):  # small-loop: n diagonal words of an oracle identity
        m.v[i * n + i] = 1.0
    return m^


def scale_cols(a: Mat, s: List[Float64]) -> Mat:
    """a[:, j] * s[j] (numpy's `a * s` over the last axis)."""
    var res = Mat(a.r, a.c)
    var ap = a.v.unsafe_ptr()
    var sp = s.unsafe_ptr()
    var rp = res.v.unsafe_ptr()
    var c = a.c
    def cell(i: Int) {imm ap, imm sp, imm rp, imm c}:
        rp[i] = ap[i] * sp[i % c]
    host_parallelize(cell, a.r * a.c)
    return res^


def scale_rows(a: Mat, s: List[Float64]) -> Mat:
    """a[i, :] * s[i] (numpy's `a * s[:, None]`)."""
    var res = Mat(a.r, a.c)
    var ap = a.v.unsafe_ptr()
    var sp = s.unsafe_ptr()
    var rp = res.v.unsafe_ptr()
    var c = a.c
    def cell(i: Int) {imm ap, imm sp, imm rp, imm c}:
        rp[i] = ap[i] * sp[i // c]
    host_parallelize(cell, a.r * a.c)
    return res^


def sub(a: Mat, r0: Int, r1: Int, c0: Int, c1: Int) -> Mat:
    """a[r0:r1, c0:c1]."""
    var res = Mat(r1 - r0, c1 - c0)
    var ap = a.v.unsafe_ptr()
    var rp = res.v.unsafe_ptr()
    var ac = a.c
    var w = c1 - c0
    def cell(i: Int) {imm ap, imm rp, imm ac, imm w, imm r0, imm c0}:
        rp[i] = ap[(r0 + i // w) * ac + c0 + i % w]
    host_parallelize(cell, (r1 - r0) * w)
    return res^


def triu_rows(h: Mat, rows: Int) -> Mat:
    """`_triu(h, r)`: the first `rows` rows of h, everything below the diagonal zero."""
    var res = sub(h, 0, rows, 0, h.c)
    var rp = res.v.unsafe_ptr()
    var c = h.c
    def cell(i: Int) {imm rp, imm c}:
        if i % c < i // c:
            rp[i] = 0.0
    host_parallelize(cell, rows * c)
    return res^


def np_sign(x: Float64) -> Float64:
    return Float64(1.0) if x > 0 else (Float64(-1.0) if x < 0 else Float64(0.0))


def close_m(tag: String, a: Mat, b: Mat, tol: Float64) raises:
    """test_linalg_decompositions.py `_close`: same shape, max |a - b| <= tol max(1, max |b|)."""
    require(a.r == b.r and a.c == b.c, tag + ": shape (" + String(a.r) + ", " + String(a.c) + ") expected ("
            + String(b.r) + ", " + String(b.c) + ")")
    var err = Float64(0)
    var scale = Float64(1)
    for i in range(a.r * a.c):  # small-loop: one max reduction over an oracle matrix
        err = max(err, abs(a.v[i] - b.v[i]))
        scale = max(scale, abs(b.v[i]))
    require(err <= tol * scale, tag + ": max |diff| " + String(err) + " above " + String(tol * scale))


def all_close_m(tag: String, a: Mat, b: Mat, rtol: Float64, atol: Float64) raises:
    """np.allclose over the flat values."""
    require(a.r * a.c == b.r * b.c, tag + ": size")
    for i in range(a.r * a.c):  # small-loop: a handful of eigen / singular values
        require(abs(a.v[i] - b.v[i]) <= atol + rtol * abs(b.v[i]),
                tag + "[" + String(i) + "]: " + String(a.v[i]) + " expected " + String(b.v[i]))


def same_bits(tag: String, a: Mat, b: Mat) raises:
    """Byte equality of two float32 results (their exact float64 widenings, sign of zero included)."""
    require(a.r == b.r and a.c == b.c, tag + ": shape")
    for i in range(a.r * a.c):  # small-loop: a word comparison over one result
        require(bitcast[DType.uint64](a.v[i]) == bitcast[DType.uint64](b.v[i]),
                tag + ": word " + String(i) + " " + String(a.v[i]) + " vs " + String(b.v[i]))


def np_entry(device: Bool) -> EntryRef:
    var word = entry_ref_word(Int(GLOBAL_NUMERIC_MODE), 0, XD_OPERATION_COUNT)
    if device:
        return EntryRef(x_decomp_device_v1, word)
    return EntryRef(x_decomp_host_v1, word)


def tsqr_r(entry: EntryRef, a: Int, rows: Int, cols: Int) raises -> List[Float32]:
    """The door's `_lin_tsqr_r(keep=True)`: addresses (a, b, r_out), ints (m, d, nrhs, keep)."""
    var r = List[Float32](length=cols * cols, fill=0.0)
    var addresses: List[Int] = [a, a, Int(r.unsafe_ptr())]
    _ = entry_ref_call(entry, XD_TSQR_R, addresses, [rows, cols, 0, 1], List[Float64]())
    return r^


def tsqr_q(entry: EntryRef, c_addr: Int, rows: Int, cols: Int, k: Int) raises -> List[Float32]:
    """The door's `_lin_tsqr_q`: Q C (rows x k) for the kept factorization."""
    var q = List[Float32](length=rows * k, fill=0.0)
    var addresses: List[Int] = [c_addr, Int(q.unsafe_ptr())]
    _ = entry_ref_call(entry, XD_TSQR_Q, addresses, [rows, cols, k], List[Float64]())
    return q^


def qr_mode(mut kit: LinKit, a: List[Float32], rows: Int, cols: Int, mode: Int) raises -> MatPair:
    """`_lin_qr_q`: mode 0 'reduced' (Q, R), 1 'complete' (Q, R), 2 'raw' (h, tau)."""
    var addr = Int(a.unsafe_ptr())
    if mode == 0 and _lin_tsqr_on(rows, cols):
        var r = tsqr_r(kit.entry, addr, rows, cols)
        var signs = List[Float32](length=cols * cols, fill=0.0)
        var addresses: List[Int] = [Int(r.unsafe_ptr()), Int(signs.unsafe_ptr())]
        _ = entry_ref_call(kit.entry, XD_R_SIGNS, addresses, [cols], List[Float64]())
        var q = tsqr_q(kit.entry, Int(signs.unsafe_ptr()), rows, cols, cols)
        var result = MatPair(mat_f32(q, rows, cols), mat_f32(r, cols, cols))
        _ = signs^
        return result^
    var tau = 0
    var held = kit.borrow(addr, rows, cols)
    var h = kit.geqrf(held, tau)
    var kk = min(rows, cols)
    if mode == 2:
        var ht = kit.transpose(h)
        var hm = mat_kit(kit, ht)
        var tm = mat_kit(kit, tau)
        return MatPair(hm^, tm^)
    var factor = kit.orgqr(h, tau, rows if mode == 1 else kk)
    var q = mat_kit(kit, factor)
    var hm = mat_kit(kit, h)
    return MatPair(q^, triu_rows(hm, rows if mode == 1 else kk))


def qr_r_route[Calls: LinalgNumericalCalls](mut kit: LinKit, a: List[Float32], rows: Int, cols: Int) raises -> Mat:
    """`_lin_qr` mode 'r': a wide R is geqrf's; a tall one the qr_r entry of the column."""
    if rows < cols:
        var tau = 0
        var held = kit.borrow(Int(a.unsafe_ptr()), rows, cols)
        var h = kit.geqrf(held, tau)
        var hm = mat_kit(kit, h)
        return triu_rows(hm, rows)
    var r = List[Float32](length=cols * cols, fill=0.0)
    Calls.qr(a, rows, cols, Int(r.unsafe_ptr()))
    return mat_f32(r, cols, cols)


def svdvals_route[Calls: LinalgNumericalCalls](mut kit: LinKit, a: List[Float32], rows: Int, cols: Int) raises -> Mat:
    """`lin_svdvals_of`: a wide A is read through its transpose, then the svdvals entry of the column."""
    var r = rows
    var c = cols
    var src = a.copy()
    if rows < cols:
        var held = kit.borrow(Int(a.unsafe_ptr()), rows, cols)
        var t = kit.transpose(held)
        src = kit_words(kit, t)
        r = cols
        c = rows
    var values = List[Float32](length=c, fill=0.0)
    Calls.singular(src, r, c, Int(values.unsafe_ptr()))
    _ = src^
    return mat_f32(values, 1, c)


def svd_route(mut kit: LinKit, a: List[Float32], rows: Int, cols: Int, full: Bool, hermitian: Bool) raises -> SvdParts:
    """`_lin_svd` with compute_uv=True: the TSQR route, the hermitian route, the tall route or the transposed one."""
    var addr = Int(a.unsafe_ptr())
    var s = 0
    var vt = 0
    if not hermitian and not full and _lin_tsqr_on(rows, cols):
        var r = tsqr_r(kit.entry, addr, rows, cols)
        var r_held = kit.borrow(Int(r.unsafe_ptr()), cols, cols)
        var ur = lin_svd_tall_run(kit, r_held, False, SVD_NULL_RTOL, s, vt)
        var c_store = kit_words(kit, ur)
        var u = tsqr_q(kit.entry, Int(c_store.unsafe_ptr()), rows, cols, cols)
        var sm = mat_kit(kit, s)
        var vm = mat_kit(kit, vt)
        _ = r^
        _ = c_store^
        return SvdParts(mat_f32(u, rows, cols), sm^, vm^)
    var m = kit.borrow(addr, rows, cols)
    if hermitian:
        var u = lin_svd_hermitian_run(kit, m, rows, s, vt)
        var um = mat_kit(kit, u)
        var sm = mat_kit(kit, s)
        var vm = mat_kit(kit, vt)
        return SvdParts(um^, sm^, vm^)
    if rows >= cols:
        var u = lin_svd_tall_run(kit, m, full, SVD_NULL_RTOL, s, vt)
        var um = mat_kit(kit, u)
        var sm = mat_kit(kit, s)
        var vm = mat_kit(kit, vt)
        return SvdParts(um^, sm^, vm^)
    var mt = kit.transpose(m)
    var u2 = lin_svd_tall_run(kit, mt, full, SVD_NULL_RTOL, s, vt)
    var v2 = kit.transpose(vt)
    var u2t = kit.transpose(u2)
    var um = mat_kit(kit, v2)
    var sm = mat_kit(kit, s)
    var vm = mat_kit(kit, u2t)
    return SvdParts(um^, sm^, vm^)


def eigh_route(mut kit: LinKit, a: List[Float32], n: Int, uplo: Int) raises -> MatPair:
    """`_lin_eigh`: (w 1 x n ascending, v n x n), uplo 1 'L', 2 'U'."""
    var v = 0
    var held = kit.borrow(Int(a.unsafe_ptr()), n, n)
    var w = kit.eigh(held, uplo, v)
    var wm = mat_kit(kit, w)
    var vm = mat_kit(kit, v)
    return MatPair(wm^, vm^)


def shape_tag(rows: Int, cols: Int) -> String:
    return String(rows) + "x" + String(cols)


def numpy_case[Calls: LinalgNumericalCalls](mut kit: LinKit, unit: Int) raises:
    var shape_rows: Array[Int, 4] = [16, 48, 5, 6]
    var shape_cols: Array[Int, 4] = [4, 5, 9, 6]
    if unit == 0:
        var sym = fixture_f32(NP_DIR + "eigh-sym.f32", 25)
        var e = eigh_route(kit, sym, 5, 1)
        for i in range(1, 5):  # small-loop: the ascending order of five eigenvalues
            require(e.first.v[i] >= e.first.v[i - 1], "eigh must be ascending like numpy, got w[" + String(i - 1)
                    + "] " + String(e.first.v[i - 1]) + " > w[" + String(i) + "] " + String(e.first.v[i]))
        all_close_m("eigh vs numpy eigvalsh", e.first, mat_ref("eigh-ref.f64", 1, 5), 1e-4, 1e-4)
        var a64 = mat_f32(sym, 5, 5)
        var av = mm(a64, False, e.second, False)
        var vw = scale_cols(e.second, e.first.v)
        var residual = Float64(0)
        for i in range(25):  # small-loop: one max reduction
            residual = max(residual, abs(av.v[i] - vw.v[i]))
        require(residual < 1e-3, "v[:, i] is not the eigenvector of w[i]: residual " + String(residual))
        _ = sym^
    elif unit == 1:
        var sym = fixture_f32(NP_DIR + "eigh-sym.f32", 25)
        var lower = fixture_f32(NP_DIR + "eigh-lower.f32", 25)
        var upper = fixture_f32(NP_DIR + "eigh-upper.f32", 25)
        var base = eigh_route(kit, sym, 5, 1)
        var lo = eigh_route(kit, lower, 5, 1)
        var up = eigh_route(kit, upper, 5, 2)
        same_bits("eigh L of the lower triangle: w", lo.first, base.first)
        same_bits("eigh L of the lower triangle: v", lo.second, base.second)
        same_bits("eigh U of the upper triangle: w", up.first, base.first)
        same_bits("eigh U of the upper triangle: v", up.second, base.second)
        all_close_m("eigh L vs numpy", lo.first, mat_ref("eigh-ref-L.f64", 1, 5), 1e-4, 1e-4)
        all_close_m("eigh U vs numpy", up.first, mat_ref("eigh-ref-U.f64", 1, 5), 1e-4, 1e-4)
        _ = sym^
        _ = lower^
        _ = upper^
    elif unit == 2:
        var a = load_x(48, 5)
        var s = svdvals_route[Calls](kit, a, 48, 5)
        for i in range(1, 5):  # small-loop: the descending order of five singular values
            require(s.v[i] <= s.v[i - 1], "svdvals must be descending like numpy, got s[" + String(i - 1) + "] "
                    + String(s.v[i - 1]) + " < s[" + String(i) + "] " + String(s.v[i]))
        all_close_m("svdvals vs numpy", s, mat_ref("svdvals-48x5.f64", 1, 5), 1e-4, 1e-4)
        _ = a^
    elif unit == 3:
        var a = load_x(48, 5)
        var s1 = svdvals_route[Calls](kit, a, 48, 5)
        var s2 = svdvals_route[Calls](kit, a, 48, 5)
        same_bits("svdvals twice", s1, s2)
        var r1 = qr_r_route[Calls](kit, a, 48, 5)
        var r2 = qr_r_route[Calls](kit, a, 48, 5)
        same_bits("qr twice", r1, r2)
        var q1 = qr_mode(kit, a, 48, 5, 0)
        var q2 = qr_mode(kit, a, 48, 5, 0)
        same_bits("qr reduced Q twice", q1.first, q2.first)
        var u1 = svd_route(kit, a, 48, 5, True, False)
        var u2 = svd_route(kit, a, 48, 5, True, False)
        same_bits("svd U twice", u1.u, u2.u)
        _ = a^
    elif unit < 8:
        var rows = shape_rows[unit - 4]
        var cols = shape_cols[unit - 4]
        var kk = min(rows, cols)
        var tag = "qr " + shape_tag(rows, cols)
        var name = "qr-" + shape_tag(rows, cols)
        var a = load_x(rows, cols)
        var a64 = mat_f32(a, rows, cols)
        var nq = mat_ref(name + "-q.f64", rows, kk)
        var nr = mat_ref(name + "-r.f64", kk, cols)
        var red = qr_mode(kit, a, rows, cols, 0)
        if rows >= cols:
            var sign = List[Float64](length=kk, fill=1.0)
            for i in range(kk):  # small-loop: R's diagonal (at most 6)
                require(red.second.at(i, i) >= 0, tag + " reduced: R's diagonal must be non-negative, R[" + String(i)
                        + ", " + String(i) + "] = " + String(red.second.at(i, i)))
                if np_sign(nr.at(i, i)) < 0:
                    sign[i] = -1.0
            close_m(tag + " reduced Q (signed)", scale_cols(red.first, sign), nq, 2e-4)
            close_m(tag + " reduced R (signed)", scale_rows(red.second, sign), nr, 2e-4)
            close_m(tag + " reduced Q R = a", mm(red.first, False, red.second, False), a64, 2e-4)
        else:
            close_m(tag + " reduced Q", red.first, nq, 2e-4)
            close_m(tag + " reduced R", red.second, nr, 2e-4)
        var comp = qr_mode(kit, a, rows, cols, 1)
        close_m(tag + " complete R", comp.second, mat_ref(name + "-rc.f64", rows, cols), 2e-4)
        close_m(tag + " complete Q^T Q = I", mm(comp.first, True, comp.first, False), eye(rows), 1e-5)
        close_m(tag + " complete Q R = a", mm(comp.first, False, comp.second, False), a64, 2e-4)
        var raw = qr_mode(kit, a, rows, cols, 2)
        close_m(tag + " raw h", raw.first, mat_ref(name + "-h.f64", cols, rows), 2e-4)
        close_m(tag + " raw tau", raw.second, mat_ref(name + "-tau.f64", 1, kk), 2e-4)
        if rows >= cols:
            var rr = qr_r_route[Calls](kit, a, rows, cols)
            var sign = List[Float64](length=cols, fill=1.0)
            for i in range(cols):  # small-loop: R's diagonal
                sign[i] = np_sign(rr.at(i, i) * nr.at(i, i))
            close_m(tag + " mode r (row signs)", scale_rows(rr, sign), nr, 2e-4)
        _ = a^
    elif unit < 16:
        var rows = shape_rows[(unit - 8) // 2]
        var cols = shape_cols[(unit - 8) // 2]
        var full = (unit - 8) % 2 == 0
        var kk = min(rows, cols)
        var name = "svd-" + shape_tag(rows, cols) + ("-full" if full else "-reduced")
        var tag = "svd " + shape_tag(rows, cols) + (" full" if full else " reduced")
        var a = load_x(rows, cols)
        var a64 = mat_f32(a, rows, cols)
        var res = svd_route(kit, a, rows, cols, full, False)
        var u_cols = rows if full else kk
        var vh_rows = cols if full else kk
        var nu = mat_ref(name + "-u.f64", rows, u_cols)
        var nvh = mat_ref(name + "-vh.f64", vh_rows, cols)
        close_m(tag + " S", res.s, mat_ref(name + "-s.f64", 1, kk), 2e-4)
        require(res.u.r == nu.r and res.u.c == nu.c and res.vh.r == nvh.r and res.vh.c == nvh.c,
                tag + ": U (" + String(res.u.r) + ", " + String(res.u.c) + ") Vh (" + String(res.vh.r) + ", "
                + String(res.vh.c) + ") expected numpy's shapes")
        close_m(tag + " U^T U = I", mm(res.u, True, res.u, False), eye(res.u.c), 1e-5)
        close_m(tag + " Vh Vh^T = I", mm(res.vh, False, res.vh, True), eye(res.vh.r), 1e-5)
        var us = scale_cols(sub(res.u, 0, rows, 0, kk), res.s.v)
        close_m(tag + " U diag(S) Vh = a", mm(us, False, sub(res.vh, 0, kk, 0, cols), False), a64, 2e-4)
        for j in range(kk):  # driver-loop: over the singular pairs (at most 6)
            var uj = sub(res.u, 0, rows, j, j + 1)
            var nuj = sub(nu, 0, rows, j, j + 1)
            var sg = List[Float64](length=1, fill=np_sign(mm(uj, True, nuj, False).v[0]))
            close_m(tag + " U[:, " + String(j) + "] up to sign", scale_cols(uj, sg), nuj, 1e-3)
            close_m(tag + " Vh[" + String(j) + "] up to sign", scale_rows(sub(res.vh, j, j + 1, 0, cols), sg),
                    sub(nvh, j, j + 1, 0, cols), 1e-3)
        _ = a^
    elif unit == 16:
        var base = load_x(24, 3)
        var dup = List[Float32](length=96, fill=0.0)
        var bp = base.unsafe_ptr()
        var dp = dup.unsafe_ptr()
        def duplicate(at: Int) {imm bp, imm dp}:
            var column = at % 4
            dp[at] = bp[(at // 4) * 3 + (column if column < 2 else column - 1)]
        host_parallelize(duplicate, 96)
        var res = svd_route(kit, dup, 24, 4, True, False)
        close_m("rank-deficient svd U^T U = I", mm(res.u, True, res.u, False), eye(24), 1e-5)
        var us = scale_cols(sub(res.u, 0, 24, 0, 4), res.s.v)
        close_m("rank-deficient svd U diag(S) Vh = a", mm(us, False, res.vh, False), mat_f32(dup, 24, 4), 2e-4)
        var sym = fixture_f32(NP_DIR + "herm-sym.f32", 16)
        var h = svd_route(kit, sym, 4, 4, False, True)
        close_m("hermitian svd S", h.s, mat_ref("herm-s.f64", 1, 4), 2e-4)
        close_m("hermitian svd U diag(S) Vh = a", mm(scale_cols(h.u, h.s.v), False, h.vh, False), mat_f32(sym, 4, 4),
                2e-4)
        _ = base^
        _ = dup^
        _ = sym^
    else:
        var wide = load_x(4, 16)
        close_m("wide svdvals", svdvals_route[Calls](kit, wide, 4, 16), mat_ref("wide-svdvals.f64", 1, 4), 2e-4)
        close_m("wide qr r", qr_r_route[Calls](kit, wide, 4, 16), mat_ref("wide-r.f64", 4, 16), 2e-4)
        _ = wide^


def numpy_unit[Calls: LinalgNumericalCalls](device: Bool, unit: Int) raises:
    var entry = np_entry(device)
    var kit = LinKit(entry, entry_ref_call(entry, XD_CAPABILITIES, List[Int](), List[Int](), List[Float64]()))
    try:
        numpy_case[Calls](kit, unit)
    finally:
        kit.close()


def _host_numpy(unit: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        numpy_unit[HostLinalgNumerical](False, unit)


def _device_numpy(unit: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        numpy_unit[DeviceLinalgNumerical](True, unit)


def test_numpy_contract() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime if CPU_COLUMN_BUILD:
        check_cases_serial("numpy_contract", NUMPY_CONTRACT_UNITS, _host_numpy)
    else:
        check_cases_serial("numpy_contract", NUMPY_CONTRACT_UNITS, _device_numpy)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
