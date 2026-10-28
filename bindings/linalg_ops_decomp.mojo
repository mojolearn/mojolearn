# SPDX-License-Identifier: Apache-2.0
# `mojolearn.linalg.qr`, `eigh`, `svdvals` and `svd`, one native call each
# (`family_lin_qr` ...): python/mojolearn/_linalg_impl.py's routes (the blocked
# TSQR, geqrf + orgqr, the round-robin Jacobi eigh, the QR + one-sided Jacobi
# SVD and `_svd_tall`'s U) driven here on the same x_decomp cells and linalg
# binding entries, with the same operands in the same order
# (bindings/linalg_ops_kit.mojo is `_M` / `_Kit`). Same kernels, same words:
# the bits do not move. The bindings are acquired the way the Python acquired
# them: the linalg binding through `_load()`'s cache, the x_decomp binding of
# the default mode, each only on the route that used it.
from std.memory import memcpy, memset_zero
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_pandas import classical_admit_value
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.estimator_binding_pyapi import estimator_binding
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.py_balanced import py_call, py_get, py_setitem
from bindings.classical_arrays import classical_address
from bindings.family_common import family_call, family_repr
from bindings.linalg_ops_buffers import LinBuf, lin_truth, lin_new, lin_array
from bindings.linalg_ops_lowbit import lin_operand, lin_state, lin_ints, lin_connected_binding
from providers.python.linalg_result_pyapi import linalg_result_new
from bindings.linalg_ops_kit import LinKit, KIT_F32P, OP_MUL, OP_DIV, OP_ABS, OP_SCALE, OP_ADDS, OP_LE

comptime TS_MAX_N = 512


def _lin_door(context: PythonObject) raises -> PythonObject:
    """`_door()` / `_load()`: the linalg binding of the selected tier, cached
    in the module's `_binding_cache` as the Python cached it."""
    return lin_connected_binding(context)


def _lin_xd(context: PythonObject) raises -> LinKit:
    """The import-connected x_decomp provider selected natively for this call."""
    return LinKit(estimator_binding(PythonObject(None), "_mojolearn_x_decomp", context), context)


def _lin_eq(a: PythonObject, b: String) raises -> Bool:
    return lin_truth(py_call(native_loaded_module("operator").eq, a, PythonObject(b)))


def _lin_named(cls: String, items: PythonObject, context: PythonObject) raises -> PythonObject:
    """`QRResult(...)` / `SVDResult(...)`: the tuple subclass over `items`."""
    return linalg_result_new(lin_state(context)[cls], items, context)


def _lin_tsqr_on(rows: Int, cols: Int) -> Bool:
    """`_tsqr_on(rows, cols)`: tall and at most TS_MAX_N wide."""
    return cols >= 1 and cols <= TS_MAX_N and rows >= cols


def _lin_triu(mut kit: LinKit, h: Int, r: Int, context: PythonObject) raises -> PythonObject:
    """`_triu(h, r).out()`: the first r rows of h, everything below the
    diagonal zeroed (pure data movement)."""
    var c = kit.c(h)
    var out = lin_array([r, c], "<f4", context)
    var dst = KIT_F32P(unsafe_from_address=Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the native call above
    if r * c:
        memcpy(dest=dst, src=kit.ptr(h), count=r * c)
        _ = out  # keep the Python object alive through the native call above
    for i in range(1, r):  # cpu-route: one zero fill per row of a host factor (the Python _triu's slice store)
        var w = min(i, c)
        if w:
            memset_zero(dst + i * c, w)
            _ = out  # keep the Python object alive through the native call above
    return out


def _lin_tsqr_release(mut kit: LinKit, context: PythonObject):
    """`_tsqr_release`: drop the factorization the binding kept."""
    try:
        var one = lin_new([1], "<f4", context)
        _ = family_call(kit.entry("x_decomp_tsqr_q"), PythonObject(one.addr), PythonObject(one.addr), lin_ints(1, 1, 0))
        _ = one  # keep the Python object alive through the native call above
    except:
        pass


def _lin_tsqr_r(mut kit: LinKit, a: LinBuf, rows: Int, cols: Int, keep: Bool, context: PythonObject) raises -> LinBuf:
    """`_tsqr_r`: R (cols x cols) of the TSQR of a; `keep` holds the
    factorization in the binding for one `_tsqr_q`."""
    var r = lin_new([cols, cols], "<f4", context)
    # ORDER MATCHES x_decomp/api.mojo tsqr_r_py: (a, b, r_out), params
    # (m, d, nrhs, keep); b is not read when nrhs is 0
    _ = family_call(kit.entry("x_decomp_tsqr_r"), PythonObject(a.addr), PythonObject(a.addr), PythonObject(r.addr),
                    lin_ints(rows, cols, 0, 1 if keep else 0))
    _ = r  # keep the Python object alive through the native call above
    return r^


def _lin_tsqr_q(mut kit: LinKit, c_addr: Int, rows: Int, cols: Int, k: Int, context: PythonObject) raises -> LinBuf:
    """`_tsqr_q`: Q C (rows x k) for the kept factorization; C is cols x k."""
    var out = lin_new([rows, k], "<f4", context)
    _ = family_call(kit.entry("x_decomp_tsqr_q"), PythonObject(c_addr), PythonObject(out.addr), lin_ints(rows, cols, k))
    _ = out  # keep the Python object alive through the native call above
    return out^


def _lin_svd_tall(mut kit: LinKit, a: Int, full: Bool, context: PythonObject, mut s_out: Int, mut vt_out: Int) raises -> Int:
    """`_svd_tall(k, A, full)`: (U, S, Vt) of a tall A; returns U."""
    var m = kit.r(a)
    var n = kit.c(a)
    var vt = 0
    var s = kit.svd(a, vt)
    s_out = s
    vt_out = vt
    var rtol = Float64(py=lin_state(context)["_SVD_NULL_RTOL_MODULE"])
    var r = 0
    if n:
        # the numerical rank, counted in Mojo (x_decomp/api.mojo rank_above_py)
        r = Int(py=family_call(native_loaded_module("builtins").int,
                               family_call(kit.entry("x_decomp_rank_above"), kit.address(s), lin_ints(n), PythonObject(rtol))))
        _ = s  # keep the Python object alive through the native call above
    var av = kit.mm(a, vt, tb=True)
    var num = kit.take_cols(av, 0, r) if r < n else av
    var den = kit.take_cols(s, 0, r) if r < n else s
    var ug = kit.ew(OP_DIV, num, den)
    var width = m if full else n
    if r == n and width == n:
        var d = 0
        var qo = kit.orth_diag(ug, d)
        var zero = kit.zeros(1, 1)
        var magnitude = kit.ew(OP_ABS, d)
        var guarded = kit.ew(OP_LE, magnitude, zero)
        if kit.total_word(guarded) == 0:
            var column_signs = kit.neg_signs(d)
            return kit.ew(OP_MUL, qo, column_signs)
    if r == 0:
        return kit.eye(m, width)
    var tau = 0
    var h = kit.geqrf(ug, tau)
    var qc = kit.orgqr(h, tau, width)
    var diagonal = kit.strided(h, r, r + 1, 0)
    var sg = kit.neg_signs(diagonal)
    if width > r:
        var trailing = kit.constant(1.0, 1, width - r)
        sg = kit.hstack2(sg, trailing)
    return kit.ew(OP_MUL, qc, sg)


# ---------------------------------------------------------------------------
# the public functions


def lin_svdvals_of(a: LinBuf, context: PythonObject) raises -> PythonObject:
    """`svdvals` of an admitted operand."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    var buf = LinBuf(a.keep, a.addr, a.shape.copy())
    if rows < cols:
        # the singular values of a wide A are those of A^T, exactly: the
        # transpose is data movement
        var kit = _lin_xd(context)
        try:
            var held = kit.of_buffer(a.addr, rows, cols)
            var t = kit.transpose(held)
            var moved = kit.out(t)
            buf = LinBuf(moved, Int(py=classical_address(moved)), [cols, rows])
            _ = moved  # keep the Python object alive through the native call above
        finally:
            kit.close()
        var swap = rows
        rows = cols
        cols = swap
    var out = lin_new([cols], "<f4", context)
    _ = family_call(neural_descriptor_entry(_lin_door(context), "svdvals"),
                    Python.list(PythonObject(buf.addr), PythonObject(out.addr)), lin_ints(rows, cols))
    _ = out  # keep the Python object alive through the native call above
    _ = buf  # keep the Python object alive through the native call above
    _ = buf^
    return out.keep


def _lin_svdvals(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_operand(args[0], "a", context)
    var result = lin_svdvals_of(a, context)
    _ = a^
    return result


def _lin_eigh(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`eigh(a, UPLO)`."""
    var a = lin_operand(args[0], "a", context)
    var rows = a.dim(0)
    var cols = a.dim(1)
    if rows != cols:
        raise Error("mojolearn.linalg.eigh: a must be square, got " + String(rows) + " x " + String(cols))
    var lower = _lin_eq(args[1], "L")
    if not lower and not _lin_eq(args[1], "U"):
        raise Error("mojolearn.linalg.eigh: UPLO argument must be 'L' or 'U'")
    var kit = _lin_xd(context)
    try:
        var v = 0
        var held = kit.of_buffer(a.addr, rows, rows)
        _ = a  # keep the Python object alive through the native call above
        var w = kit.eigh(held, 1 if lower else 2, v)
        _ = a  # keep the Python object alive through the native call above
        var result = Python.tuple(kit.out_vector(w, rows), kit.out(v))
        _ = a^
        return result
    finally:
        kit.close()


def _lin_qr_q(a: LinBuf, mode: String, context: PythonObject) raises -> PythonObject:
    """`_qr_q(a, mode)`: numpy.linalg.qr's 'reduced', 'complete' and 'raw'."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    var kit = _lin_xd(context)
    try:
        # TOMBSTONE: MOJOLEARN_QR_FAST_DEV (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1 (the FAST kit's
        # grid-fold geqrf / orgqr ahead of the TSQR); code recoverable at b639a2bd2.
        # Restore: git apply experiments/removed/MOJOLEARN_QR_FAST_DEV.patch; record in docs/TOMBSTONES.md.
        if mode == "reduced" and _lin_tsqr_on(rows, cols):
            # `_qr_tsqr`: (Q, R), R's diagonal non-negative
            var r = _lin_tsqr_r(kit, a, rows, cols, True, context)
            var signs = lin_new([cols, cols], "<f4", context)
            var signed = False
            try:
                # R's rows signed so its diagonal is non-negative, and C the
                # diagonal of those signs (x_decomp/api.mojo r_signs_py)
                _ = family_call(kit.entry("x_decomp_r_signs"), PythonObject(r.addr), PythonObject(signs.addr), lin_ints(cols))
                _ = signs  # keep the Python object alive through the native call above
                _ = r  # keep the Python object alive through the native call above
                signed = True
            finally:
                if not signed:
                    _lin_tsqr_release(kit, context)
            var q = _lin_tsqr_q(kit, signs.addr, rows, cols, cols, context)
            _ = signs  # keep the Python object alive through the native call above
            var _keep_ret_signs = _lin_named("QRResult", Python.tuple(q.keep, r.keep), context)
            _ = signs  # keep the Python object alive through the native call above
            return _keep_ret_signs^
        var tau = 0
        var held = kit.of_buffer(a.addr, rows, cols)
        var h = kit.geqrf(held, tau)
        var kk = min(rows, cols)
        if mode == "raw":
            # numpy returns geqrf's Fortran-ordered array seen in C order: the
            # factored matrix transposed, (N, M); tau (K,)
            var transposed = kit.transpose(h)
            var ht = kit.out(transposed)
            return Python.tuple(ht, kit.out_vector(tau, kk))
        if mode == "complete":
            var full_q = kit.orgqr(h, tau, rows)
            var q = kit.out(full_q)
            return _lin_named("QRResult", Python.tuple(q, _lin_triu(kit, h, rows, context)), context)
        var reduced_q = kit.orgqr(h, tau, kk)
        var q = kit.out(reduced_q)
        return _lin_named("QRResult", Python.tuple(q, _lin_triu(kit, h, kk, context)), context)
    finally:
        kit.close()


def _lin_qr(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`qr(a, mode)`."""
    var mode = args[1]
    var named = String("")
    if _lin_eq(mode, "reduced"):
        named = "reduced"
    elif _lin_eq(mode, "complete"):
        named = "complete"
    elif _lin_eq(mode, "raw"):
        named = "raw"
    if named != "":
        var operand = lin_operand(args[0], "a", context)
        var result = _lin_qr_q(operand, named, context)
        _ = operand^
        return result
    if not _lin_eq(mode, "r"):
        raise Error("mojolearn.linalg.qr: unrecognized mode " + family_repr(mode) + "; numpy's modes "
                    + "are 'reduced', 'complete', 'r' and 'raw'")
    var a = lin_operand(args[0], "a", context)
    var rows = a.dim(0)
    var cols = a.dim(1)
    if rows < cols:
        # a wide R (M x N, upper trapezoidal) is geqrf's; the TSQR route is tall only
        var kit = _lin_xd(context)
        try:
            var tau = 0
            var held = kit.of_buffer(a.addr, rows, cols)
            _ = a  # keep the Python object alive through the native call above
            var h = kit.geqrf(held, tau)
            _ = a  # keep the Python object alive through the native call above
            var wide = _lin_triu(kit, h, rows, context)
            _ = a^
            return wide
        finally:
            kit.close()
    var out = lin_new([cols, cols], "<f4", context)
    # ORDER MATCHES qr_r_binding IN BOTH BINDINGS: addrs (a, r_out), params
    # (n_rows, n_cols); swapping the addresses writes R over the caller's matrix
    _ = family_call(neural_descriptor_entry(_lin_door(context), "qr_r"),
                    Python.list(PythonObject(a.addr), PythonObject(out.addr)), lin_ints(rows, cols))
    _ = out  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def _lin_svd(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`svd(a, full_matrices, compute_uv, hermitian)`."""
    var a = lin_operand(args[0], "a", context)
    var rows = a.dim(0)
    var cols = a.dim(1)
    if not lin_truth(args[2]):
        var values = lin_svdvals_of(a, context)
        _ = a^
        return values
    var full = lin_truth(args[1])
    var hermitian = lin_truth(args[3])
    var kit = _lin_xd(context)
    try:
        # TOMBSTONE: MOJOLEARN_SVD_FAST_CHOLQR (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1 (the FAST kit's
        # CholeskyQR2 whole-matrix route ahead of the TSQR); code recoverable at b639a2bd2.
        # Restore: git apply experiments/removed/MOJOLEARN_SVD_FAST_CHOLQR.patch; record in docs/TOMBSTONES.md.
        if not hermitian and not full and _lin_tsqr_on(rows, cols):
            # `_svd_tsqr`: the SVD of the TSQR's R, then U = Q U_R
            var r = _lin_tsqr_r(kit, a, rows, cols, True, context)
            var s = 0
            var vt = 0
            var c_store = PythonObject(None)
            var done = False
            try:
                # TOMBSTONE: MOJOLEARN_SVD_QFIX (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
                # Tried: the 2^-40 cut + Householder U_R branch here (qfix bit 1).
                # Restore: git apply experiments/removed/MOJOLEARN_SVD_QFIX.patch; record in docs/TOMBSTONES.md.
                var r_held = kit.of_buffer(r.addr, cols, cols)
                _ = r  # keep the Python object alive through the native call above
                var ur = _lin_svd_tall(kit, r_held, False, context, s, vt)
                _ = r  # keep the Python object alive through the native call above
                c_store = kit.out(ur)
                done = True
            finally:
                if not done:
                    _lin_tsqr_release(kit, context)
            var u = _lin_tsqr_q(kit, Int(py=classical_address(c_store)), rows, cols, cols, context)
            _ = c_store  # keep the Python object alive through the native call above
            var result = _lin_named("SVDResult", Python.tuple(u.keep, kit.out_vector(s, cols), kit.out(vt)), context)
            _ = c_store  # keep the Python object alive through the native call above
            _ = a^
            return result
        var m = kit.of_buffer(a.addr, rows, cols)
        if hermitian:
            if rows != cols:
                raise Error("mojolearn.linalg.svd: hermitian=True needs a square matrix")
            var v = 0
            var w = kit.eigh(m, 1, v)
            # numpy: argsort(|w|) (ascending) reversed, so equal magnitudes come
            # higher index first: the stable ascending order of -|w| read back
            # to front, on the kit
            var rev = kit.zeros(1, rows)
            var rp = kit.ptr(rev)
            _ = rev  # keep the Python object alive through the native call above
            for t in range(rows):  # driver-loop: over the index table n-1 .. 0 (matrix columns)
                rp[t] = Float32(rows - 1 - t)
                _ = rev  # keep the Python object alive through the native call above
            var wr = kit.take_cols_m(w, rev, rows)
            var wr_abs = kit.ew(OP_ABS, wr)
            var keys = kit.ew(OP_SCALE, wr_abs, scalar=-1.0)
            var o = kit.order_small(keys)
            var o_row = kit.reshape(o, 1, rows)
            var o_neg = kit.ew(OP_SCALE, o_row, scalar=-1.0)
            var order = kit.ew(OP_ADDS, o_neg, scalar=Float64(rows - 1))
            var w_abs = kit.ew(OP_ABS, w)
            var s = kit.take_cols_m(w_abs, order, rows)
            var u = kit.take_cols_m(v, order, rows)
            var ut = kit.transpose(u)
            var w_ordered = kit.take_cols_m(w, order, rows)
            var w_signs = kit.neg_signs(w_ordered)
            var signs = kit.reshape(w_signs, rows, 1)
            var vt = kit.ew(OP_MUL, ut, signs)
            var result = _lin_named("SVDResult", Python.tuple(kit.out(u), kit.out_vector(s, rows), kit.out(vt)), context)
            _ = a^
            return result
        if rows >= cols:
            var s = 0
            var vt = 0
            var u = _lin_svd_tall(kit, m, full, context, s, vt)
            var result = _lin_named("SVDResult", Python.tuple(kit.out(u), kit.out_vector(s, cols), kit.out(vt)), context)
            _ = a^
            return result
        var s = 0
        var vt2 = 0
        var mt = kit.transpose(m)
        var u2 = _lin_svd_tall(kit, mt, full, context, s, vt2)
        var v2 = kit.transpose(vt2)
        var first = kit.out(v2)
        var values = kit.out_vector(s, rows)
        var u2t = kit.transpose(u2)
        var last = kit.out(u2t)
        _ = a^
        return _lin_named("SVDResult", Python.tuple(first, values, last), context)
    finally:
        kit.close()


def lin_decomp_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # Admit public scalar controls before equality/truth conversion; input
    # exporters are captured by the shared native linalg buffer boundary.
    classical_admit_value(args[0], context)
    for i in range(1, len(args)):  # small-loop(public decomposition option metadata)
        host_inference_scalar(args[i], context, True)
    if operation == "family_lin_qr":
        return _lin_qr(args, context)
    if operation == "family_lin_eigh":
        return _lin_eigh(args, context)
    if operation == "family_lin_svdvals":
        return _lin_svdvals(args, context)
    if operation == "family_lin_svd":
        return _lin_svd(args, context)
    raise Error("RuntimeError: unknown linalg operation " + operation)
