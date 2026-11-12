# SPDX-License-Identifier: Apache-2.0
# `mojolearn.linalg.qr`, `eigh`, `svdvals` and `svd`, one native call each
# (`family_lin_qr` ...): python/mojolearn/_linalg_impl.py's routes (the blocked
# TSQR, geqrf + orgqr, the round-robin Jacobi eigh, the QR + one-sided Jacobi
# SVD and `_svd_tall`'s U) driven here on the same x_decomp cells and linalg
# binding entries, with the same operands in the same order
# (bindings/linalg_ops_kit_run.mojo is `_M` / `_Kit`, typed). Same kernels,
# same words: the bits do not move. The bindings are acquired the way the
# Python acquired them: the linalg binding through `_load()`'s cache, the
# x_decomp binding of the default mode, each only on the route that used it.
#
# Door shape (lane entangled-kits, 2026-11-12): UNPACK admits the operand and
# resolves the x_decomp image's dispatch once (`_lin_kit`); the kit sequences
# are RUN (typed, the GIL released around each, the admitted owners named
# after it); PACK copies the kit's matrices into public Arrays (`_lin_out`).
from std.memory import unsafe_memcpy, unsafe_memset_zero
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_pandas import classical_admit_value
from bindings.host_inference_scalars import host_inference_scalar
from providers.python.estimator_binding_pyapi import estimator_binding
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays import classical_address
from bindings.family_common import family_call, family_repr
from bindings.linalg_ops_buffers import lin_truth
from bindings.admitted_input import AdmittedBuf, admitted_zeros
from bindings.linalg_ops_lowbit import lin_operand, lin_state, lin_ints, lin_connected_binding
from providers.python.linalg_result_pyapi import linalg_result_new
from bindings.kit_call import kit_connect, kit_capabilities, kit_entry_ref
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT, XD_MODE, XD_CAPABILITIES
from bindings.linalg_ops_kit_run import LinKit, KIT_F32P, lin_svd_tall_run, lin_svd_hermitian_run
from bindings.linalg_ops_decomp_run import _lin_tsqr_on


def _lin_door(context: PythonObject) raises -> PythonObject:
    """`_door()` / `_load()`: the linalg binding of the selected tier, cached
    in the module's `_binding_cache` as the Python cached it."""
    return lin_connected_binding(context)


def _lin_xd(context: PythonObject) raises -> PythonObject:
    """UNPACK: the import-connected x_decomp binding selected natively for
    this call (an exact native module)."""
    var binding = estimator_binding(PythonObject(None), "_mojolearn_x_decomp", context)
    if py_type(binding) is not py_type(native_loaded_module("builtins")):
        raise Error("TypeError: linalg decomposition requires an exact native provider module")
    return binding


def _lin_kit(binding: PythonObject) raises -> LinKit:
    """UNPACK: the typed kit over the binding's x_decomp dispatch, connected
    once per binding (bindings/kit_call.mojo) and read here as an EntryRef
    and the image's capability word."""
    var snapshot = kit_connect(binding, "x_decomp_native_api_v1", String(XD_ABI_NAME),
                               XD_OPERATION_COUNT, XD_MODE, XD_CAPABILITIES)
    var kit = LinKit(kit_entry_ref(snapshot, XD_OPERATION_COUNT), kit_capabilities(snapshot))
    _ = snapshot  # keep the snapshot alive through the reads above
    return kit^


def _lin_out(mut kit: LinKit, a: Int, context: PythonObject) raises -> PythonObject:
    """PACK `_M.out()`: a public float32 Array (C order) of matrix a; a
    device matrix is downloaded once (it moves), the GIL released."""
    var rows = kit.r(a)
    var cols = kit.c(a)
    var result = admitted_zeros([rows, cols], "<f4", context).keep
    _lin_copy_out(kit, a, result, rows * cols)
    return result


def _lin_out_vector(mut kit: LinKit, a: Int, length: Int, context: PythonObject) raises -> PythonObject:
    """PACK `_M.out((length,))`."""
    var result = admitted_zeros([length], "<f4", context).keep
    _lin_copy_out(kit, a, result, length)
    return result


def _lin_copy_out(mut kit: LinKit, a: Int, result: PythonObject, count: Int) raises:
    if count:
        var src = 0
        var py = Python()
        with GILReleased(py):
            src = kit.addr(a)
        unsafe_memcpy(dest=KIT_F32P(unsafe_from_address=Int(py=classical_address(result))),
                      src=KIT_F32P(unsafe_from_address=src), count=count)
        _ = result  # keep the Python object alive through the copy above


def _lin_rtol(context: PythonObject) raises -> Float64:
    """UNPACK: `_SVD_NULL_RTOL_MODULE`, the rank cut `_svd_tall` reads."""
    return Float64(py=lin_state(context)["_SVD_NULL_RTOL_MODULE"])


def _lin_eq(a: PythonObject, b: String) raises -> Bool:
    return lin_truth(py_call(native_loaded_module("operator").eq, a, PythonObject(b)))


def _lin_named(cls: String, items: PythonObject, context: PythonObject) raises -> PythonObject:
    """`QRResult(...)` / `SVDResult(...)`: the tuple subclass over `items`."""
    return linalg_result_new(lin_state(context)[cls], items, context)


def _lin_triu(mut kit: LinKit, h: Int, r: Int, context: PythonObject) raises -> PythonObject:
    """PACK `_triu(h, r).out()`: the first r rows of h (a host factor),
    everything below the diagonal zeroed (pure data movement)."""
    var c = kit.c(h)
    var out = admitted_zeros([r, c], "<f4", context).keep
    var dst = KIT_F32P(unsafe_from_address=Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the native call above
    if r * c:
        unsafe_memcpy(dest=dst, src=kit.ptr(h), count=r * c)
        _ = out  # keep the Python object alive through the native call above
    for i in range(1, r):  # cpu-route: one zero fill per row of a host factor (the Python _triu's slice store)
        var w = min(i, c)
        if w:
            unsafe_memset_zero(dst + i * c, w)
            _ = out  # keep the Python object alive through the native call above
    return out


def _lin_tsqr_release(binding: PythonObject, context: PythonObject):
    """`_tsqr_release`: drop the factorization the binding kept."""
    try:
        var one = admitted_zeros([1], "<f4", context)
        _ = family_call(neural_descriptor_entry(binding, "x_decomp_tsqr_q"), PythonObject(one.addr),
                        PythonObject(one.addr), lin_ints(1, 1, 0))
        _ = one  # keep the Python object alive through the native call above
    except:
        pass


def _lin_tsqr_r(binding: PythonObject, a: AdmittedBuf, rows: Int, cols: Int, keep: Bool,
                context: PythonObject) raises -> AdmittedBuf:
    """`_tsqr_r`: R (cols x cols) of the TSQR of a; `keep` holds the
    factorization in the binding for one `_tsqr_q`."""
    var r = admitted_zeros([cols, cols], "<f4", context)
    # ORDER MATCHES x_decomp/api.mojo tsqr_r_py: (a, b, r_out), params
    # (m, d, nrhs, keep); b is not read when nrhs is 0
    _ = family_call(neural_descriptor_entry(binding, "x_decomp_tsqr_r"), PythonObject(a.addr), PythonObject(a.addr),
                    PythonObject(r.addr), lin_ints(rows, cols, 0, 1 if keep else 0))
    _ = r  # keep the Python object alive through the native call above
    return r^


def _lin_tsqr_q(binding: PythonObject, c_addr: Int, rows: Int, cols: Int, k: Int,
                context: PythonObject) raises -> AdmittedBuf:
    """`_tsqr_q`: Q C (rows x k) for the kept factorization; C is cols x k."""
    var out = admitted_zeros([rows, k], "<f4", context)
    _ = family_call(neural_descriptor_entry(binding, "x_decomp_tsqr_q"), PythonObject(c_addr), PythonObject(out.addr),
                    lin_ints(rows, cols, k))
    _ = out  # keep the Python object alive through the native call above
    return out^


# ---------------------------------------------------------------------------
# the public functions


def lin_svdvals_of(a: AdmittedBuf, context: PythonObject) raises -> PythonObject:
    """`svdvals` of an admitted operand."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    var buf = AdmittedBuf(a.keep, a.addr, a.shape.copy())
    if rows < cols:
        # the singular values of a wide A are those of A^T, exactly: the
        # transpose is data movement
        var kit = _lin_kit(_lin_xd(context))
        try:
            var t = 0
            var py = Python()
            with GILReleased(py):
                var held = kit.borrow(a.addr, rows, cols)
                t = kit.transpose(held)
            _ = a  # keep the admitted owner alive through the RUN above
            var moved = _lin_out(kit, t, context)
            buf = AdmittedBuf(moved, Int(py=classical_address(moved)), [cols, rows])
            _ = moved  # keep the Python object alive through the native call above
        finally:
            kit.close()
        var swap = rows
        rows = cols
        cols = swap
    var out = admitted_zeros([cols], "<f4", context)
    _ = family_call(neural_descriptor_entry(_lin_door(context), "svdvals"),
                    Python.list(PythonObject(buf.addr), PythonObject(out.addr)), lin_ints(rows, cols))
    _ = out  # keep the Python object alive through the native call above
    _ = buf  # keep the Python object alive through the native call above
    _ = buf^
    return out.keep


def _lin_svdvals(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_operand(args[0], "a", context, True)
    var result = lin_svdvals_of(a, context)
    _ = a^
    return result


def _lin_eigh(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`eigh(a, UPLO)`."""
    var a = lin_operand(args[0], "a", context, True)
    var rows = a.dim(0)
    var cols = a.dim(1)
    if rows != cols:
        raise Error("mojolearn.linalg.eigh: a must be square, got " + String(rows) + " x " + String(cols))
    var lower = _lin_eq(args[1], "L")
    if not lower and not _lin_eq(args[1], "U"):
        raise Error("mojolearn.linalg.eigh: UPLO argument must be 'L' or 'U'")
    var kit = _lin_kit(_lin_xd(context))
    try:
        var v = 0
        var w = 0
        var py = Python()
        with GILReleased(py):
            var held = kit.borrow(a.addr, rows, rows)
            w = kit.eigh(held, 1 if lower else 2, v)
        _ = a  # keep the admitted owner alive through the RUN above
        var result = Python.tuple(_lin_out_vector(kit, w, rows, context), _lin_out(kit, v, context))
        _ = a^
        return result
    finally:
        kit.close()


def _lin_qr_q(a: AdmittedBuf, mode: String, context: PythonObject) raises -> PythonObject:
    """`_qr_q(a, mode)`: numpy.linalg.qr's 'reduced', 'complete' and 'raw'."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    var binding = _lin_xd(context)
    # TOMBSTONE: MOJOLEARN_QR_FAST_DEV (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1 (the FAST kit's
    # grid-fold geqrf / orgqr ahead of the TSQR); code recoverable at b639a2bd2.
    # Restore: git apply experiments/removed/MOJOLEARN_QR_FAST_DEV.patch; record in docs/TOMBSTONES.md.
    if mode == "reduced" and _lin_tsqr_on(rows, cols):
        # `_qr_tsqr`: (Q, R), R's diagonal non-negative
        var r = _lin_tsqr_r(binding, a, rows, cols, True, context)
        var signs = admitted_zeros([cols, cols], "<f4", context)
        var signed = False
        try:
            # R's rows signed so its diagonal is non-negative, and C the
            # diagonal of those signs (x_decomp/api.mojo r_signs_py)
            _ = family_call(neural_descriptor_entry(binding, "x_decomp_r_signs"), PythonObject(r.addr),
                            PythonObject(signs.addr), lin_ints(cols))
            _ = signs  # keep the Python object alive through the native call above
            _ = r  # keep the Python object alive through the native call above
            signed = True
        finally:
            if not signed:
                _lin_tsqr_release(binding, context)
        var q = _lin_tsqr_q(binding, signs.addr, rows, cols, cols, context)
        _ = signs  # keep the Python object alive through the native call above
        var _keep_ret_signs = _lin_named("QRResult", Python.tuple(q.keep, r.keep), context)
        _ = signs  # keep the Python object alive through the native call above
        return _keep_ret_signs^
    var kit = _lin_kit(binding)
    try:
        var tau = 0
        var h = 0
        var factor = 0
        var kk = min(rows, cols)
        var py = Python()
        with GILReleased(py):
            var held = kit.borrow(a.addr, rows, cols)
            h = kit.geqrf(held, tau)
            if mode == "raw":
                # numpy returns geqrf's Fortran-ordered array seen in C order:
                # the factored matrix transposed, (N, M); tau (K,)
                factor = kit.transpose(h)
            elif mode == "complete":
                factor = kit.orgqr(h, tau, rows)
            else:
                factor = kit.orgqr(h, tau, kk)
        _ = a  # keep the admitted owner alive through the RUN above
        if mode == "raw":
            var ht = _lin_out(kit, factor, context)
            return Python.tuple(ht, _lin_out_vector(kit, tau, kk, context))
        var q = _lin_out(kit, factor, context)
        var r_rows = rows if mode == "complete" else kk
        return _lin_named("QRResult", Python.tuple(q, _lin_triu(kit, h, r_rows, context)), context)
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
        var operand = lin_operand(args[0], "a", context, True)
        var result = _lin_qr_q(operand, named, context)
        _ = operand^
        return result
    if not _lin_eq(mode, "r"):
        raise Error("mojolearn.linalg.qr: unrecognized mode " + family_repr(mode) + "; numpy's modes "
                    + "are 'reduced', 'complete', 'r' and 'raw'")
    var a = lin_operand(args[0], "a", context, True)
    var rows = a.dim(0)
    var cols = a.dim(1)
    if rows < cols:
        # a wide R (M x N, upper trapezoidal) is geqrf's; the TSQR route is tall only
        var kit = _lin_kit(_lin_xd(context))
        try:
            var tau = 0
            var h = 0
            var py = Python()
            with GILReleased(py):
                var held = kit.borrow(a.addr, rows, cols)
                h = kit.geqrf(held, tau)
            _ = a  # keep the admitted owner alive through the RUN above
            var wide = _lin_triu(kit, h, rows, context)
            _ = a^
            return wide
        finally:
            kit.close()
    var out = admitted_zeros([cols, cols], "<f4", context)
    # ORDER MATCHES qr_r_binding IN BOTH BINDINGS: addrs (a, r_out), params
    # (n_rows, n_cols); swapping the addresses writes R over the caller's matrix
    _ = family_call(neural_descriptor_entry(_lin_door(context), "qr_r"),
                    Python.list(PythonObject(a.addr), PythonObject(out.addr)), lin_ints(rows, cols))
    _ = out  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def _lin_svd(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`svd(a, full_matrices, compute_uv, hermitian)`."""
    var a = lin_operand(args[0], "a", context, True)
    var rows = a.dim(0)
    var cols = a.dim(1)
    if not lin_truth(args[2]):
        var values = lin_svdvals_of(a, context)
        _ = a^
        return values
    var full = lin_truth(args[1])
    var hermitian = lin_truth(args[3])
    var binding = _lin_xd(context)
    var kit = _lin_kit(binding)
    try:
        var py = Python()
        # TOMBSTONE: MOJOLEARN_SVD_FAST_CHOLQR (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1 (the FAST kit's
        # CholeskyQR2 whole-matrix route ahead of the TSQR); code recoverable at b639a2bd2.
        # Restore: git apply experiments/removed/MOJOLEARN_SVD_FAST_CHOLQR.patch; record in docs/TOMBSTONES.md.
        if not hermitian and not full and _lin_tsqr_on(rows, cols):
            # `_svd_tsqr`: the SVD of the TSQR's R, then U = Q U_R
            var rtol = _lin_rtol(context)
            var r = _lin_tsqr_r(binding, a, rows, cols, True, context)
            var s = 0
            var vt = 0
            var c_store = PythonObject(None)
            var done = False
            try:
                # TOMBSTONE: MOJOLEARN_SVD_QFIX (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
                # Tried: the 2^-40 cut + Householder U_R branch here (qfix bit 1).
                # Restore: git apply experiments/removed/MOJOLEARN_SVD_QFIX.patch; record in docs/TOMBSTONES.md.
                var ur = 0
                with GILReleased(py):
                    var r_held = kit.borrow(r.addr, cols, cols)
                    ur = lin_svd_tall_run(kit, r_held, False, rtol, s, vt)
                _ = r  # keep the Python object alive through the RUN above
                c_store = _lin_out(kit, ur, context)
                done = True
            finally:
                if not done:
                    _lin_tsqr_release(binding, context)
            var u = _lin_tsqr_q(binding, Int(py=classical_address(c_store)), rows, cols, cols, context)
            _ = c_store  # keep the Python object alive through the native call above
            var result = _lin_named("SVDResult", Python.tuple(u.keep, _lin_out_vector(kit, s, cols, context),
                                                              _lin_out(kit, vt, context)), context)
            _ = c_store  # keep the Python object alive through the native call above
            _ = a^
            return result
        var m = kit.borrow(a.addr, rows, cols)
        if hermitian:
            if rows != cols:
                raise Error("mojolearn.linalg.svd: hermitian=True needs a square matrix")
            var s = 0
            var vt = 0
            var u = 0
            with GILReleased(py):
                u = lin_svd_hermitian_run(kit, m, rows, s, vt)
            _ = a  # keep the admitted owner alive through the RUN above
            var result = _lin_named("SVDResult", Python.tuple(_lin_out(kit, u, context),
                                                              _lin_out_vector(kit, s, rows, context),
                                                              _lin_out(kit, vt, context)), context)
            _ = a^
            return result
        var rtol = _lin_rtol(context)
        if rows >= cols:
            var s = 0
            var vt = 0
            var u = 0
            with GILReleased(py):
                u = lin_svd_tall_run(kit, m, full, rtol, s, vt)
            _ = a  # keep the admitted owner alive through the RUN above
            var result = _lin_named("SVDResult", Python.tuple(_lin_out(kit, u, context),
                                                              _lin_out_vector(kit, s, cols, context),
                                                              _lin_out(kit, vt, context)), context)
            _ = a^
            return result
        var s = 0
        var v2 = 0
        var u2t = 0
        with GILReleased(py):
            var vt2 = 0
            var mt = kit.transpose(m)
            var u2 = lin_svd_tall_run(kit, mt, full, rtol, s, vt2)
            v2 = kit.transpose(vt2)
            u2t = kit.transpose(u2)
        _ = a  # keep the admitted owner alive through the RUN above
        var first = _lin_out(kit, v2, context)
        var values = _lin_out_vector(kit, s, rows, context)
        var last = _lin_out(kit, u2t, context)
        _ = a^
        return _lin_named("SVDResult", Python.tuple(first, values, last), context)
    finally:
        kit.close()


def lin_decomp_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # Admit public scalar controls before equality/truth conversion; input
    # exporters are captured by the shared native linalg buffer boundary.
    # The one admission of the operand: every _lin_* below passes
    # admitted=True to lin_operand (row N237).
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
