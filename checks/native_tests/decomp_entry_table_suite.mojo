# SPDX-License-Identifier: Apache-2.0
"""The x_decomp operations the decomposition kit reaches through the
published table (core/decomp_entry_table.mojo), natively.

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the same publication
`_mojolearn_x_decomp_host` makes at module init, then a schema mismatch, an
arity mismatch and an unknown operation refuse. Then one case per operation
lane calltable-prep-decomp added to the ABI (x_decomp/descriptor_schema.mojo
XD_RAND .. XD_DEV_SELECT_SMALLEST) and lane pyout-decomp-linalg added
(XD_ALS_FIT .. XD_RANDOMIZED_SVD: the owner doors that were Python entries), each through `decomp_table_call` and the
published host pointer, with literal expected values; the device-only and
GPU-only operations refuse on the host image with its text. No Python
participates.

The cases are independent units: on the host column they run across the
native host worker pool; on a device column they run one after another.
Run in a fresh process (the first check needs an unpublished table).
"""
from core.entry_table import entry_ref_same
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.linear_entry_table import LinearEntryKey, linear_table_publish, linear_table_entry, LINEAR_TABLE_DECOMP
from core.decomp_entry_table import decomp_ref, decomp_table_call
from x_decomp.descriptor_host import x_decomp_host_v1, X_DECOMP_HOST_API
from x_decomp.descriptor_schema import (
    XD_ABI_NAME, XD_OPERATION_COUNT, XD_RAND, XD_LU, XD_LU_SOLVE, XD_EIGH, XD_LASSO_ROWS, XD_LARS_ROWS,
    XD_OMP_ROWS, XD_LU_AUX, XD_ABSMAX_SIGN, XD_REDUCE, XD_SELECT_SMALLEST, XD_ARGMIN_ALL, XD_LU_RESID,
    XD_S_FLAGS, XD_QFIX_FLAGS, XD_DEV_RAND, XD_DEV_ABSMAX, XD_DEV_LU, XD_DEV_LU_AUX, XD_DEV_CODE_ROWS,
    XD_DEV_EIGH_TO, XD_DEV_ARGMIN_ALL, XD_DEV_SELECT_SMALLEST, XD_GEMM,
    XD_GATHER, XD_TOPN_DESC, XD_LU_GESV, XD_CHI2_QUANTILE, XD_CONSISTENCY_FACTOR, XD_RANDOMIZED_SVD,
    XD_DEV_LU_FACTOR, XD_DEV_LU_SOLVE, XD_ALS_FIT,
)

comptime DECOMP_TABLE_CASES = 35


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> LinearEntryKey:
    return LinearEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _f32(mut l: List[Float32]) -> Int:
    return Int(l.unsafe_ptr())


def _i32(mut l: List[Int32]) -> Int:
    return Int(l.unsafe_ptr())


def _near(a: Float32, b: Float64, tol: Float64 = 1e-5) -> Bool:
    return abs(Float64(a) - b) <= tol


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _call(op: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    return decomp_table_call(decomp_ref(_key()), op, addresses, ints, floats)


def precondition_unpublished() raises:
    try:
        _ = _call(XD_S_FLAGS, List[Int](), List[Int](), List[Float64]())
    except problem:
        _refuses(problem, "provider is not published")
        return
    raise Error("an unpublished x_decomp table did not refuse")


def publish_host() raises:
    # The publication bindings/_mojolearn_x_decomp_host.mojo makes at init.
    var api = X_DECOMP_HOST_API.get_or_create_ptr()
    linear_table_publish[LINEAR_TABLE_DECOMP](_key(), Int(api[].entries.unsafe_ptr()),
        [String(XD_ABI_NAME)], [XD_OPERATION_COUNT], [x_decomp_host_v1])


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = decomp_ref(_key())
    var second = decomp_ref(_key())
    require(entry_ref_same(first, second), "x_decomp: a second resolution returned another entry")
    try:
        _ = linear_table_entry[LINEAR_TABLE_DECOMP](_key(), 0, String(XD_ABI_NAME), XD_OPERATION_COUNT + 1)
        raise Error("x_decomp: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "x_decomp: unexpected refusal: " + String(problem))


def preconditions_published() raises:
    ref_resolution()
    try:
        _ = linear_table_entry[LINEAR_TABLE_DECOMP](_key(), 0, String("mojolearn.x_decomp.other.v1"), XD_OPERATION_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        _ = _call(XD_REDUCE, List[Int](), [1, 0], List[Float64]())
        raise Error("a wrong arity did not refuse")
    except problem:
        _refuses(problem, "takes 2 addresses")
    try:
        _ = _call(XD_OPERATION_COUNT, List[Int](), List[Int](), List[Float64]())
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")


def _device_refuses(op: Int, na: Int, ni: Int, nf: Int, text: String) raises:
    var a = List[Int](length=na, fill=0)
    var i = List[Int](length=ni, fill=1)
    var f = List[Float64](length=nf, fill=0.0)
    try:
        _ = _call(op, a, i, f)
    except problem:
        _refuses(problem, text)
        return
    raise Error("the host image ran device operation " + String(op))


def case(unit: Int) raises:
    comptime HOST_ONLY = "has no device-resident operation"
    if unit == 0:
        var out: List[Float32] = [9.0, 9.0, 9.0, 9.0]
        require(_call(XD_RAND, [_f32(out)], [4, 7, 0, 0], List[Float64]()) == 4, "rand count")
        for v in out:  # small-loop(four drawn words)
            require(v == v and v != 9.0, "rand words")
        _ = out
    elif unit == 1 or unit == 2:
        # A = [[4, 3], [6, 3]]: |det| = 6, the product of U's diagonal.
        var a: List[Float32] = [4.0, 3.0, 6.0, 3.0]
        var piv: List[Int32] = [0, 0]
        var info: List[Float32] = [9.0]
        require(_call(XD_LU, [_f32(a), _i32(piv), _f32(info)], [2], List[Float64]()) == 2, "lu n")
        require(info[0] == 0.0 and _near(abs(a[0] * a[3]), 6.0), "lu factor")
        if unit == 2:
            # b = A [1, 1]^T: the solve returns [1, 1].
            var b: List[Float32] = [7.0, 9.0]
            require(_call(XD_LU_SOLVE, [_f32(a), _i32(piv), _f32(b)], [2, 1, 0], List[Float64]()) == 2, "lu_solve n")
            require(_near(b[0], 1.0) and _near(b[1], 1.0), "lu_solve values")
            _ = b
        _ = a
        _ = piv
        _ = info
    elif unit == 3:
        var a: List[Float32] = [2.0, 0.0, 0.0, 1.0]
        var w: List[Float32] = [0.0, 0.0]
        var v: List[Float32] = [0.0, 0.0, 0.0, 0.0]
        require(_call(XD_EIGH, [_f32(a), _f32(w), _f32(v)], [2, 0], List[Float64]()) == 2, "eigh n")
        require(_near(w[0], 1.0) and _near(w[1], 2.0), "eigh ascending values")
        _ = a
        _ = v
    elif unit == 4 or unit == 5 or unit == 6:
        # One row, one atom, G = [1], Q = [0.5]: every coder returns 0.5.
        var g: List[Float32] = [1.0]
        var q: List[Float32] = [0.5]
        var w: List[Float32] = [0.0]
        var side: List[Float32] = [0.0]
        var addresses: List[Int] = [_f32(g), _f32(q), _f32(w), _f32(side)]
        if unit == 4:
            require(_call(XD_LASSO_ROWS, addresses, [1, 1, 100, 0], [0.0, 1e-8]) == 1, "lasso n")
        elif unit == 5:
            require(_call(XD_LARS_ROWS, addresses, [1, 1, 1, 1], List[Float64]()) == 1, "lars n")
        else:
            require(_call(XD_OMP_ROWS, addresses, [1, 1, 1], List[Float64]()) == 1, "omp n")
        require(_near(w[0], 0.5), "coder value " + String(unit))
        _ = g
        _ = q
        _ = side
    elif unit == 7:
        # diag(2, -3): no swaps, max |u_ii| 3, no zero pivot, one negative.
        var a: List[Float32] = [2.0, 0.0, 0.0, -3.0]
        var piv: List[Int32] = [0, 0]
        var info: List[Float32] = [9.0]
        _ = _call(XD_LU, [_f32(a), _i32(piv), _f32(info)], [2], List[Float64]())
        var pm: List[Float32] = [0.0, 0.0]
        var im: List[Float32] = [0.0, 0.0]
        var diag: List[Float32] = [0.0, 0.0]
        var stats: List[Float32] = [9.0, 9.0, 9.0, 9.0]
        require(_call(XD_LU_AUX, [_f32(a), _i32(piv), _f32(pm), _f32(im), _f32(diag), _f32(stats)], [2, 0],
                      List[Float64]()) == 2, "lu_aux n")
        require(diag[0] == 2.0 and diag[1] == -3.0, "lu_aux diagonal")
        require(stats[0] == 3.0 and stats[1] == 0.0 and stats[2] == 1.0 and stats[3] == 0.0, "lu_aux stats")
        _ = a
        _ = piv
        _ = info
        _ = pm
        _ = im
    elif unit == 8:
        # by column: column 0's largest |x| is +2, column 1's is -3.
        var a: List[Float32] = [1.0, -3.0, 2.0, 1.0]
        var dst: List[Float32] = [0.0, 0.0]
        require(_call(XD_ABSMAX_SIGN, [_f32(a), _f32(dst)], [2, 2, 1], List[Float64]()) == 2, "absmax d")
        require(dst[0] > 0.0 and dst[1] < 0.0, "absmax signs")
        _ = a
    elif unit == 9:
        var a: List[Float32] = [1.0, -5.0, 3.0]
        var dst: List[Float32] = [0.0]
        require(_call(XD_REDUCE, [_f32(a), _f32(dst)], [3, 0], List[Float64]()) == 1 and dst[0] == 5.0, "max |x|")
        _ = _call(XD_REDUCE, [_f32(a), _f32(dst)], [3, 1], List[Float64]())
        require(dst[0] == 3.0, "max")
        _ = _call(XD_REDUCE, [_f32(a), _f32(dst)], [3, 2], List[Float64]())
        require(dst[0] == -5.0, "min")
        _ = a
    elif unit == 10:
        var x: List[Float32] = [3.0, 1.0, 2.0]
        var sel: List[Float32] = [9.0, 9.0]
        var mask: List[Int32] = [9, 9, 9]
        require(_call(XD_SELECT_SMALLEST, [_f32(x), _f32(sel), _i32(mask)], [3, 2], List[Float64]()) == 2, "select h")
        require(sel[0] == 1.0 and sel[1] == 2.0, "select positions")
        require(mask[0] == 0 and mask[1] == 1 and mask[2] == 1, "select membership")
        _ = x
    elif unit == 11:
        var x: List[Float32] = [2.0, 1.0, 1.0]
        var dst: List[Float32] = [9.0, 9.0, 9.0]
        require(_call(XD_ARGMIN_ALL, [_f32(x), _f32(dst)], [3], List[Float64]()) == 2, "argmin count")
        require(dst[0] == 1.0 and dst[1] == 2.0, "argmin positions")
        _ = x
    elif unit == 12:
        require(_call(XD_S_FLAGS, List[Int](), List[Int](), List[Float64]()) == 0, "host s_flags")
    elif unit == 13:
        require(_call(XD_QFIX_FLAGS, List[Int](), List[Int](), List[Float64]()) == 0, "host qfix_flags")
    elif unit == 14:
        _device_refuses(XD_LU_RESID, 4, 2, 0, HOST_ONLY)
    elif unit == 15:
        _device_refuses(XD_DEV_RAND, 0, 5, 0, HOST_ONLY)
    elif unit == 16:
        _device_refuses(XD_DEV_ABSMAX, 0, 5, 0, HOST_ONLY)
    elif unit == 17:
        _device_refuses(XD_DEV_LU, 2, 3, 0, HOST_ONLY)
    elif unit == 18:
        _device_refuses(XD_DEV_LU_AUX, 5, 3, 0, HOST_ONLY)
    elif unit == 19:
        _device_refuses(XD_DEV_CODE_ROWS, 0, 8, 2, HOST_ONLY)
    elif unit == 20:
        _device_refuses(XD_DEV_EIGH_TO, 0, 5, 0, HOST_ONLY)
    elif unit == 21:
        _device_refuses(XD_DEV_ARGMIN_ALL, 0, 4, 0, HOST_ONLY)
    elif unit == 22:
        _device_refuses(XD_DEV_SELECT_SMALLEST, 0, 5, 0, HOST_ONLY)
    elif unit == 23:
        # The kit's product through the same table (XD_GEMM, ta = tb = 0).
        var a: List[Float32] = [1.0, 2.0]
        var b: List[Float32] = [3.0, 4.0]
        var c: List[Float32] = [0.0]
        _ = _call(XD_GEMM, [_f32(a), _f32(b), _f32(c)], [1, 2, 1, 0, 0], List[Float64]())
        require(c[0] == 11.0, "gemm")
        _ = a
        _ = b
    elif unit == 24:
        # dst[a] = src[idx[a]] (exact copies)
        var src: List[Float32] = [10.0, 20.0, 30.0]
        var idx: List[Int32] = [2, 0]
        var dst: List[Float32] = [0.0, 0.0]
        require(_call(XD_GATHER, [_f32(src), _i32(idx), _f32(dst)], [2], List[Float64]()) == 2, "gather count")
        require(dst[0] == 30.0 and dst[1] == 10.0, "gather values")
        _ = src
        _ = idx
    elif unit == 25 or unit == 26:
        # the two best positions by (-x, index); unit 26 skips position 1 (a nonzero skip word)
        var x: List[Float32] = [1.0, 3.0, 2.0]
        var skip: List[Float32] = [0.0, 1.0, 0.0]
        var dst: List[Int32] = [9, 9]
        var skip_address = _f32(skip) if unit == 26 else 0
        require(_call(XD_TOPN_DESC, [_f32(x), skip_address, _i32(dst)], [3, 2], List[Float64]()) == 2, "topn count")
        if unit == 25:
            require(dst[0] == 1 and dst[1] == 2, "topn positions")
        else:
            require(dst[0] == 2 and dst[1] == 0, "topn skipped positions")
        _ = x
        _ = skip
    elif unit == 27:
        # gesv: A = [[4, 3], [6, 3]], b = A [1, 1]^T, solved in place
        var a: List[Float32] = [4.0, 3.0, 6.0, 3.0]
        var b: List[Float32] = [7.0, 9.0]
        var info: List[Float32] = [9.0]
        require(_call(XD_LU_GESV, [_f32(a), _f32(b), _f32(info)], [2, 1], List[Float64]()) == 2, "gesv n")
        require(info[0] == 0.0 and _near(b[0], 1.0) and _near(b[1], 1.0), "gesv solution")
        _ = a
    elif unit == 28:
        # P(chi2_2 > m) = 0.025 at m = -2 ln 0.025 = 7.3778
        var word: List[Float64] = [0.0]
        _ = _call(XD_CHI2_QUANTILE, [Int(word.unsafe_ptr())], List[Int](), [2.0, 0.025])
        require(abs(word[0] - 7.377759) < 1e-3, "chi2 quantile " + String(word[0]))
    elif unit == 29:
        # the Pison factor at alpha 0.975, p 1: alpha over a cdf just below 1
        var word: List[Float64] = [0.0]
        _ = _call(XD_CONSISTENCY_FACTOR, [Int(word.unsafe_ptr())], List[Int](), [1.0, 0.975, 1.0])
        require(word[0] > 1.0 and word[0] < 1.5, "consistency factor " + String(word[0]))
        try:
            _ = _call(XD_CONSISTENCY_FACTOR, [Int(word.unsafe_ptr())], List[Int](), [1.0, 0.975, 0.0])
            raise Error("a zero den did not refuse")
        except problem:
            _refuses(problem, "den must be > 0")
        _ = word
    elif unit == 30 or unit == 31:
        # randomized_svd of diag(3, 1), one component: S = [3]; unit 31 asks the host image
        # for a deferred finite test, which needs a resident binding
        var m: List[Float32] = [3.0, 0.0, 0.0, 1.0]
        var u: List[Float32] = [0.0, 0.0]
        var sv: List[Float32] = [0.0]
        var vt: List[Float32] = [0.0, 0.0]
        var params: List[Int] = [2, 2, 1, 1, 2, 0, 0, 1, 7, 1 if unit == 31 else 0]
        if unit == 30:
            require(_call(XD_RANDOMIZED_SVD, [_f32(m), _f32(u), _f32(sv), _f32(vt)], params, List[Float64]()) == 1,
                    "rsvd components")
            require(_near(sv[0], 3.0, 1e-4) and _near(abs(u[0]), 1.0, 1e-4), "rsvd values")
        else:
            try:
                _ = _call(XD_RANDOMIZED_SVD, [_f32(m), _f32(u), _f32(sv), _f32(vt)], params, List[Float64]())
                raise Error("a host deferred finite test did not refuse")
            except problem:
                _refuses(problem, "needs a resident binding")
        _ = m
        _ = u
        _ = vt
    elif unit == 32:
        _device_refuses(XD_DEV_LU_FACTOR, 4, 4, 0, HOST_ONLY)
    elif unit == 33:
        _device_refuses(XD_DEV_LU_SOLVE, 2, 6, 0, HOST_ONLY)
    elif unit == 34:
        # ALS on one user and one item, one factor: the call runs twice in one
        # process (the old Python repeat test's purpose) and returns the same words
        var first = List[Float32](length=2, fill=0.0)
        for repeat in range(2):  # small-loop(two repeats of the same fit)
            var c: List[Float32] = [1.0]
            var x: List[Float32] = [0.5]
            var y: List[Float32] = [0.5]
            var losses: List[Float64] = [0.0]
            var count = _call(XD_ALS_FIT, [_f32(c), _f32(x), _f32(y), Int(losses.unsafe_ptr())], [1, 1, 1, 1, 0, 1, 1],
                              [0.01])
            require(count >= 0 and count <= 1, "als loss count")
            require(x[0] == x[0] and y[0] == y[0], "als factors finite")
            if repeat == 0:
                first[0] = x[0]
                first[1] = y[0]
            else:
                require(first[0] == x[0] and first[1] == y[0], "als repeat bits")
            _ = c
            _ = losses


def run_checks() raises:
    precondition_unpublished()
    publish_host()
    preconditions_published()
    var failures = List[Int](length=DECOMP_TABLE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL decomp entry table case", unit, String(problem))
    comptime if TARGET_COLUMN == COLUMN_CPU:
        host_parallelize(task, DECOMP_TABLE_CASES)
    else:
        for unit in range(DECOMP_TABLE_CASES):  # driver-loop: device cases share one device context
            task(unit)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "decomp entry table cases failed: " + String(failed))
    print("PASS decomp entry table", DECOMP_TABLE_CASES)
