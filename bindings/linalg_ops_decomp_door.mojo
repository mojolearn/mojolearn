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
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_pandas_door import classical_admit_value
from bindings.host_inference_scalars_door import host_inference_scalar
from providers.python.estimator_binding_pyapi_door import estimator_binding, estimator_binding_device
from core.function_device import FunctionDevice, function_device_default
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays_door import classical_address
from bindings.family_common_unpack import family_repr
from bindings.family_common_pack import family_call
from bindings.linalg_ops_buffers_unpack import lin_truth
from bindings.admitted_input_unpack import AdmittedBuf, admitted_zeros, admitted_unfilled
from bindings.linalg_ops_lowbit_door import (
    lin_operand, lin_state, lin_ints, lin_device, lin_device_binding, lin_device_context, LIN_HAS_GPU, LIN_HAS_CPU,
)
from providers.python.linalg_result_pyapi_door import linalg_result_new
from bindings.kit_call_unpack import kit_capabilities, kit_entry_ref
from bindings.kit_call_pack import kit_connect
from x_decomp.descriptor_schema import (
    XD_ABI_NAME, XD_OPERATION_COUNT, XD_MODE, XD_CAPABILITIES, XD_TSQR_R, XD_TSQR_Q, XD_R_SIGNS,
    XD_DEV_TSQR_R, XD_DEV_TSQR_Q, XD_DEV_R_SIGNS,
)
from core.entry_table import EntryRef, entry_ref_call
from bindings.linalg_ops_kit_run import LinKit, KIT_F32P, lin_svd_tall_run, lin_svd_hermitian_run
from bindings.linalg_ops_kit_run import _lin_tsqr_on
from bindings.linalg_keys_unpack import linalg_keys
from bindings.py_balanced import py_item
from bindings.operation_names_linops import (
    LINOPS_OPN_FAMILY_LIN_QR, LINOPS_OPN_FAMILY_LIN_EIGH, LINOPS_OPN_FAMILY_LIN_SVDVALS, LINOPS_OPN_FAMILY_LIN_SVD)
from bindings.linalg_dev_arrays_unpack import (
    lin_train_device, lin_is_device_array, lin_dev_device, lin_dev_entries, lin_dev_operand, lin_dev_new,
    lin_dev_handle,
)
from bindings.linalg_ops_lowbit_door import lin_call
from bindings.linalg_ops_lowbit_run import LIN_OP_SVDVALS_DEV, LIN_OP_QR_R_DEV
from core.train_dev_pool import TRAIN_DEV_F32


def _lin_door(device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """`_door()` / `_load()`: the linalg binding of the selected tier, cached
    in the module's `_binding_cache` as the Python cached it; for an explicit
    "cpu" beside a GPU set the host image (lane vdf-linalg,
    bindings/linalg_ops_lowbit_door.mojo `lin_device_binding`)."""
    return lin_device_binding(device, context)


def _lin_xd(device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """UNPACK: the import-connected x_decomp binding selected natively for
    this call (an exact native module). An explicit "cpu" beside a GPU set
    reads the x_decomp host image connected at import
    (`_mojolearn_x_decomp_host`, python/mojolearn/_native_family_bridge.py
    `_CPU_DEVICE_IMAGES`), refused in words where it is not connected."""
    var binding = PythonObject(None)
    if device.host_beside:
        binding = estimator_binding_device(device, PythonObject(None), "_mojolearn_x_decomp", context)
    else:
        binding = estimator_binding(PythonObject(None), "_mojolearn_x_decomp", context)
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
    # lane fix-output-zero-fill (2026-11-14): unfilled; _lin_copy_out writes all rows * cols.
    var result = admitted_unfilled([rows, cols], "<f4", context).keep
    _lin_copy_out(kit, a, result, rows * cols)
    return result


def _lin_out_vector(mut kit: LinKit, a: Int, length: Int, context: PythonObject) raises -> PythonObject:
    """PACK `_M.out((length,))`."""
    var result = admitted_unfilled([length], "<f4", context).keep
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
    var nk = linalg_keys()
    return Float64(py=py_item(lin_state(context), nk[].k__SVD_NULL_RTOL_MODULE))


def _lin_eq(a: PythonObject, b: String) raises -> Bool:
    return lin_truth(py_call(native_loaded_module("operator").eq, a, PythonObject(b)))


def _lin_named(cls: String, items: PythonObject, context: PythonObject) raises -> PythonObject:
    """`QRResult(...)` / `SVDResult(...)`: the tuple subclass over `items`."""
    return linalg_result_new(lin_state(context)[cls], items, context)


# lane vdf-linalg2 (2026-11-24): `_lin_triu` (R cut from the downloaded h on
# the host, one zero fill per row, tagged cpu-route) is gone: the kit's
# `triu` cuts R on the device where h now stays (XD_DEV_TRIU), and PACK
# downloads R once (`_lin_out`). The host column cuts it in the kit.


def _lin_entry(binding: PythonObject) raises -> EntryRef:
    """UNPACK: the x_decomp image's dispatch alone (the TSQR route needs no
    kit), from the binding's connected snapshot (bindings/kit_call.mojo)."""
    var snapshot = kit_connect(binding, "x_decomp_native_api_v1", String(XD_ABI_NAME),
                               XD_OPERATION_COUNT, XD_MODE, XD_CAPABILITIES)
    var entry = kit_entry_ref(snapshot, XD_OPERATION_COUNT)
    _ = snapshot  # keep the snapshot alive through the read above
    return entry


def _lin_tsqr_call(entry: EntryRef, op: Int, addresses: List[Int], ints: List[Int]) raises -> Int:
    """One TSQR table operation (XD_TSQR_R / XD_TSQR_Q / XD_R_SIGNS; were the
    binding's Python entries x_decomp_tsqr_r / _tsqr_q / _r_signs), the GIL
    released as the doors released it. The caller keeps every address alive."""
    var result = 0
    var py = Python()
    with GILReleased(py):
        result = entry_ref_call(entry, op, addresses, ints, List[Float64]())
    return result


def _lin_tsqr_release(entry: EntryRef, owner: Int):
    """`_tsqr_release`: free the factorization `owner` names (XD_TSQR_Q with
    k == 0 reads no word; one native word stands in for both buffers)."""
    try:
        var one = List[Float32](length=1, fill=0.0)
        var addresses: List[Int] = [Int(one.unsafe_ptr()), Int(one.unsafe_ptr())]
        _ = _lin_tsqr_call(entry, XD_TSQR_Q, addresses, [1, 1, 0, owner])
        _ = one^  # keep the word alive through the call above
    except:
        pass


def _lin_tsqr_release_run(mut kit: LinKit, owner: Int):
    """RUN `_lin_tsqr_release` for a caller that already released the GIL (the device forms' sequences)."""
    try:
        var one = List[Float32](length=1, fill=0.0)
        var addresses: List[Int] = [Int(one.unsafe_ptr()), Int(one.unsafe_ptr())]
        _ = kit.call(XD_TSQR_Q, addresses, [1, 1, 0, owner], List[Float64]())
        _ = one^  # keep the word alive through the call above
    except:
        pass


def _lin_tsqr_r(entry: EntryRef, a: AdmittedBuf, rows: Int, cols: Int, mut owner: Int,
                context: PythonObject) raises -> AdmittedBuf:
    """`_tsqr_r`: R (cols x cols) of the TSQR of a. The factorization is kept
    in a native owner this call carries: `owner` gets its handle for the one
    `_lin_tsqr_q` (or `_lin_tsqr_release`) that frees it. Lane linalg-owner
    (2026-11-22): the image held ONE kept factorization per process, so two
    concurrent qr / svd calls could take each other's factor."""
    var r = admitted_zeros([cols, cols], "<f4", context)
    # ORDER MATCHES x_decomp/api.mojo tsqr_r_run: (a, b, r_out), params
    # (m, d, nrhs, keep); b is not read when nrhs is 0
    var addresses: List[Int] = [a.addr, a.addr, r.addr]
    owner = _lin_tsqr_call(entry, XD_TSQR_R, addresses, [rows, cols, 0, 1])
    _ = r  # keep the Python object alive through the native call above
    return r^


def _lin_tsqr_q(entry: EntryRef, owner: Int, c_addr: Int, rows: Int, cols: Int, k: Int,
                context: PythonObject) raises -> AdmittedBuf:
    """`_tsqr_q`: Q C (rows x k) for the factorization `owner` holds (C is
    cols x k); the owner is freed on every path."""
    var applied = False
    try:
        var out = admitted_zeros([rows, k], "<f4", context)
        var addresses: List[Int] = [c_addr, out.addr]
        applied = True  # XD_TSQR_Q frees the owner whether it applies or refuses
        _ = _lin_tsqr_call(entry, XD_TSQR_Q, addresses, [rows, cols, k, owner])
        _ = out  # keep the Python object alive through the native call above
        return out^
    finally:
        if not applied:
            _lin_tsqr_release(entry, owner)


# ---------------------------------------------------------------------------
# the public functions


def lin_svdvals_of(a: AdmittedBuf, device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """`svdvals` of an admitted operand."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    # the singular values of a wide A are those of A^T, exactly: the linalg
    # entry reads A once and transposes it itself (on the device in the GPU
    # image, in its read in the host image; the `wide` word). Lane
    # vdf-linalg2 (2026-11-24): was the x_decomp kit's device transpose,
    # downloaded here and handed back to the entry, which uploaded it again.
    var wide = rows < cols
    var params = lin_ints(cols, rows, 1) if wide else lin_ints(rows, cols)
    var n_values = rows if wide else cols
    var out = admitted_zeros([n_values], "<f4", context)
    _ = family_call(neural_descriptor_entry(_lin_door(device, context), "svdvals"),
                    Python.list(PythonObject(a.addr), PythonObject(out.addr)), params)
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the admitted owner alive through the native call above
    return out.keep


def _lin_svdvals(args: PythonObject, device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    var a = lin_operand(args[0], "a", context, True)
    var result = lin_svdvals_of(a, device, context)
    _ = a^
    return result


@fieldwise_init
struct LinSource(Copyable, Movable):
    """The operand of one kit decomposition (lane vdf-linalg6, 2026-11-24): an admitted host buffer the door keeps
    alive (`addr`, `handle` -1) or a training device array's pool handle (`handle` >= 0, `addr` 0). Its matrix is
    the same words either way: borrowed in place, or copied device to device into the kit's pool (`_lin_take`)."""
    var addr: Int
    var handle: Int
    var rows: Int
    var cols: Int

    def on_device(self) -> Bool:
        return self.handle >= 0


def _lin_host_source(a: AdmittedBuf) -> LinSource:
    return LinSource(a.addr, -1, a.dim(0), a.dim(1))


def _lin_take(mut kit: LinKit, src: LinSource) raises -> Int:
    """RUN: the kit matrix of the operand. Call with the GIL released."""
    if src.on_device():
        return kit.adopt(src.handle, src.rows, src.cols)
    return kit.borrow(src.addr, src.rows, src.cols)


def _lin_pack(mut kit: LinKit, m: Int, src: LinSource, entries: PythonObject, held: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """PACK of matrix m: a public Array for a host operand (`_lin_out`), a new TrainDeviceArray (rows, cols) for a
    device operand (one device copy, `LinKit.to_train`)."""
    if not src.on_device():
        return _lin_out(kit, m, context)
    return _lin_dev_out(kit, m, [kit.r(m), kit.c(m)], entries, held)


def _lin_pack_vector(mut kit: LinKit, m: Int, length: Int, src: LinSource, entries: PythonObject,
                     held: PythonObject, context: PythonObject) raises -> PythonObject:
    """PACK `_M.out((length,))`, or a new 1-D TrainDeviceArray for a device operand."""
    if not src.on_device():
        return _lin_out_vector(kit, m, length, context)
    return _lin_dev_out(kit, m, [length], entries, held)


def _lin_dev_out(mut kit: LinKit, m: Int, dims: List[Int], entries: PythonObject, held: PythonObject) raises -> PythonObject:
    """PACK: a new float32 TrainDeviceArray of `dims` (one training pool handle, `lin_dev_new`) holding matrix m's
    words; a device matrix is copied device to device, a host one (a small host-made result) uploaded once."""
    var out = lin_dev_new(entries, held, dims, TRAIN_DEV_F32)
    var handle = lin_dev_handle(out)
    var py = Python()
    with GILReleased(py):
        kit.to_train(m, handle)
    return out


def _lin_eigh_of(src: LinSource, lower: Bool, device: FunctionDevice, entries: PythonObject, held: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """`eigh(a, UPLO)` of a square operand (checked by the caller)."""
    var rows = src.rows
    var kit = _lin_kit(_lin_xd(device, context))
    try:
        var v = 0
        var w = 0
        var py = Python()
        with GILReleased(py):
            var x = _lin_take(kit, src)
            w = kit.eigh(x, 1 if lower else 2, v)
        return Python.tuple(_lin_pack_vector(kit, w, rows, src, entries, held, context),
                            _lin_pack(kit, v, src, entries, held, context))
    finally:
        kit.close()


def _lin_eigh_admit(uplo: PythonObject, rows: Int, cols: Int) raises -> Bool:
    """UNPACK of eigh's square shape and UPLO; True for 'L'."""
    if rows != cols:
        raise Error("mojolearn.linalg.eigh: a must be square, got " + String(rows) + " x " + String(cols))
    var lower = _lin_eq(uplo, "L")
    if not lower and not _lin_eq(uplo, "U"):
        raise Error("mojolearn.linalg.eigh: UPLO argument must be 'L' or 'U'")
    return lower


def _lin_eigh(args: PythonObject, device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """`eigh(a, UPLO)`."""
    var a = lin_operand(args[0], "a", context, True)
    var lower = _lin_eigh_admit(args[1], a.dim(0), a.dim(1))
    var result = _lin_eigh_of(_lin_host_source(a), lower, device, PythonObject(None), PythonObject(None), context)
    _ = a^  # keep the admitted owner alive through the RUN above
    return result


def _lin_qr_tsqr_dev(src: LinSource, device: FunctionDevice, entries: PythonObject, held: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`_qr_tsqr` of a device operand: the host operand's TSQR route (XD_TSQR_R, XD_R_SIGNS, XD_TSQR_Q) on pool ids
    (XD_DEV_TSQR_R, XD_DEV_R_SIGNS, XD_DEV_TSQR_Q; x_decomp/resident_train.mojo): X adopted device to device, R
    written into a pool id (the factor's NaN refusal is a device scan; one status word comes down), signed there, and
    the signs matrix C applied there, so R, C and Q never cross the host (lane vdf-linalg7; R came down for the signs
    and went up again for its result)."""
    var rows = src.rows
    var cols = src.cols
    var kit = _lin_kit(_lin_xd(device, context))
    try:
        var owner = 0
        var r = 0
        var q = 0
        var py = Python()
        with GILReleased(py):
            var x = _lin_take(kit, src)
            r = kit.device_output(cols, cols)
            var signs = kit.device_output(cols, cols)
            var dx = kit.device(x)
            var dr = kit.matrices[r].device
            var ds = kit.matrices[signs].device
            # (x id, rows, cols, keep, r id, b id, nrhs): R into pool id dr, the host word unread
            owner = kit.call(XD_DEV_TSQR_R, [0], [dx, rows, cols, 1, dr, -1, 0], List[Float64]())
            var signed = False
            try:
                _ = kit.call(XD_DEV_R_SIGNS, List[Int](), [dr, ds, cols], List[Float64]())
                q = kit.device_output(rows, cols)
                signed = True
            finally:
                if not signed:
                    _lin_tsqr_release_run(kit, owner)
            # XD_DEV_TSQR_Q frees the owner whether it applies or refuses; C is the pool matrix ds (host word unread)
            _ = kit.call(XD_DEV_TSQR_Q, [0], [owner, ds, kit.matrices[q].device, rows, cols, cols], List[Float64]())
        return _lin_named("QRResult", Python.tuple(_lin_pack(kit, q, src, entries, held, context),
                                                   _lin_pack(kit, r, src, entries, held, context)), context)
    finally:
        kit.close()


def _lin_qr_kit(src: LinSource, mode: String, device: FunctionDevice, entries: PythonObject, held: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`_qr_q(a, mode)` off the TSQR route ('reduced' wide or past TS_MAX_N, 'complete', 'raw') and the wide 'r'
    (mode "r"): geqrf, then orgqr / transpose / triu, on the kit."""
    var rows = src.rows
    var cols = src.cols
    var kit = _lin_kit(_lin_xd(device, context))
    try:
        var tau = 0
        var h = 0
        var factor = 0
        var kk = min(rows, cols)
        var py = Python()
        with GILReleased(py):
            var x = _lin_take(kit, src)
            h = kit.geqrf(x, tau)
            if mode == "raw":
                # numpy returns geqrf's Fortran-ordered array seen in C order:
                # the factored matrix transposed, (N, M); tau (K,)
                factor = kit.transpose(h)
            elif mode == "complete":
                factor = kit.orgqr(h, tau, rows)
            elif mode == "reduced":
                factor = kit.orgqr(h, tau, kk)
        if mode == "raw":
            var ht = _lin_pack(kit, factor, src, entries, held, context)
            return Python.tuple(ht, _lin_pack_vector(kit, tau, kk, src, entries, held, context))
        # a wide R (M x N, upper trapezoidal) for 'r'; R of 'complete' has M rows, of 'reduced' K
        var r_rows = rows if (mode == "complete" or mode == "r") else kk
        var r_cut = 0
        with GILReleased(py):
            r_cut = kit.triu(h, r_rows)
        if mode == "r":
            return _lin_pack(kit, r_cut, src, entries, held, context)
        var q = _lin_pack(kit, factor, src, entries, held, context)
        return _lin_named("QRResult", Python.tuple(q, _lin_pack(kit, r_cut, src, entries, held, context)), context)
    finally:
        kit.close()


def _lin_qr_q(a: AdmittedBuf, mode: String, device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """`_qr_q(a, mode)`: numpy.linalg.qr's 'reduced', 'complete' and 'raw'."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    # TOMBSTONE: MOJOLEARN_QR_FAST_DEV (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1 (the FAST kit's
    # grid-fold geqrf / orgqr ahead of the TSQR); code recoverable at b639a2bd2.
    # Restore: git apply experiments/removed/MOJOLEARN_QR_FAST_DEV.patch; record in docs/TOMBSTONES.md.
    if mode == "reduced" and _lin_tsqr_on(rows, cols):
        var binding = _lin_xd(device, context)
        # `_qr_tsqr`: (Q, R), R's diagonal non-negative
        var entry = _lin_entry(binding)
        # the signs matrix first: nothing that can raise sits between the
        # kept factorization and the try that frees it
        var signs = admitted_zeros([cols, cols], "<f4", context)
        var owner = 0
        var r = _lin_tsqr_r(entry, a, rows, cols, owner, context)
        var signed = False
        try:
            # R's rows signed so its diagonal is non-negative, and C the
            # diagonal of those signs (x_decomp/api.mojo r_signs_py)
            var addresses: List[Int] = [r.addr, signs.addr]
            _ = _lin_tsqr_call(entry, XD_R_SIGNS, addresses, [cols])
            _ = signs  # keep the Python object alive through the native call above
            _ = r  # keep the Python object alive through the native call above
            signed = True
        finally:
            if not signed:
                _lin_tsqr_release(entry, owner)
        var q = _lin_tsqr_q(entry, owner, signs.addr, rows, cols, cols, context)
        _ = signs  # keep the Python object alive through the native call above
        var _keep_ret_signs = _lin_named("QRResult", Python.tuple(q.keep, r.keep), context)
        _ = signs  # keep the Python object alive through the native call above
        return _keep_ret_signs^
    var result = _lin_qr_kit(_lin_host_source(a), mode, device, PythonObject(None), PythonObject(None), context)
    _ = a  # keep the admitted owner alive through the RUN above
    return result


def _lin_qr_mode(mode: PythonObject) raises -> String:
    """UNPACK of qr's mode: 'reduced', 'complete', 'raw' or 'r', refused in words otherwise."""
    if _lin_eq(mode, "reduced"):
        return "reduced"
    if _lin_eq(mode, "complete"):
        return "complete"
    if _lin_eq(mode, "raw"):
        return "raw"
    if _lin_eq(mode, "r"):
        return "r"
    raise Error("mojolearn.linalg.qr: unrecognized mode " + family_repr(mode) + "; numpy's modes "
                + "are 'reduced', 'complete', 'r' and 'raw'")


def _lin_qr(args: PythonObject, device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """`qr(a, mode)`."""
    var named = _lin_qr_mode(args[1])
    if named != "r":
        var operand = lin_operand(args[0], "a", context, True)
        var result = _lin_qr_q(operand, named, device, context)
        _ = operand^
        return result
    var a = lin_operand(args[0], "a", context, True)
    var rows = a.dim(0)
    var cols = a.dim(1)
    if rows < cols:
        # a wide R (M x N, upper trapezoidal) is geqrf's; the TSQR route is tall only
        var wide = _lin_qr_kit(_lin_host_source(a), "r", device, PythonObject(None), PythonObject(None), context)
        _ = a^  # keep the admitted owner alive through the RUN above
        return wide
    var out = admitted_zeros([cols, cols], "<f4", context)
    # ORDER MATCHES qr_r_binding IN BOTH BINDINGS: addrs (a, r_out), params
    # (n_rows, n_cols); swapping the addresses writes R over the caller's matrix
    _ = family_call(neural_descriptor_entry(_lin_door(device, context), "qr_r"),
                    Python.list(PythonObject(a.addr), PythonObject(out.addr)), lin_ints(rows, cols))
    _ = out  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def _lin_svd_tsqr_dev(src: LinSource, device: FunctionDevice, entries: PythonObject, held: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`_svd_tsqr` of a device operand: the host operand's sequence (the SVD of the TSQR's R on the kit, then
    U = Q U_R) with X adopted and R, U_R and U on pool ids (XD_DEV_TSQR_R / XD_DEV_TSQR_Q): R is written into a
    pool id and read there by the kit's small SVD (lane vdf-linalg7: it came down at the factor and went up again for
    the SVD), U_R is not downloaded and uploaded again; the words are the same."""
    var rows = src.rows
    var cols = src.cols
    var rtol = _lin_rtol(context)
    var kit = _lin_kit(_lin_xd(device, context))
    try:
        var s = 0
        var vt = 0
        var u = 0
        var py = Python()
        with GILReleased(py):
            var x = _lin_take(kit, src)
            var r = kit.device_output(cols, cols)
            var dx = kit.device(x)
            # (x id, rows, cols, keep, r id, b id, nrhs): R into the pool id of r, the host word unread
            var owner = kit.call(XD_DEV_TSQR_R, [0], [dx, rows, cols, 1, kit.matrices[r].device, -1, 0],
                                 List[Float64]())
            var dc = -1
            var done = False
            try:
                var ur = lin_svd_tall_run(kit, r, False, rtol, s, vt)
                dc = kit.device(ur)
                u = kit.device_output(rows, cols)
                done = True
            finally:
                if not done:
                    _lin_tsqr_release_run(kit, owner)
            # the host word is unread (C is the pool matrix dc); XD_DEV_TSQR_Q frees the owner on every path
            _ = kit.call(XD_DEV_TSQR_Q, [0], [owner, dc, kit.matrices[u].device, rows, cols, cols], List[Float64]())
        return _lin_named("SVDResult", Python.tuple(_lin_pack(kit, u, src, entries, held, context),
                                                    _lin_pack_vector(kit, s, cols, src, entries, held, context),
                                                    _lin_pack(kit, vt, src, entries, held, context)), context)
    finally:
        kit.close()


def _lin_svd_kit(src: LinSource, full: Bool, hermitian: Bool, device: FunctionDevice, entries: PythonObject,
                 held: PythonObject, context: PythonObject) raises -> PythonObject:
    """`svd(a, full_matrices, compute_uv=True, hermitian)` off the TSQR route, on the kit."""
    var rows = src.rows
    var cols = src.cols
    if hermitian and rows != cols:
        raise Error("mojolearn.linalg.svd: hermitian=True needs a square matrix")
    var rtol = 0.0 if hermitian else _lin_rtol(context)
    var kit = _lin_kit(_lin_xd(device, context))
    try:
        var py = Python()
        var s = 0
        var first = 0
        var last = 0
        var n_values = cols
        with GILReleased(py):
            var m = _lin_take(kit, src)
            if hermitian:
                first = lin_svd_hermitian_run(kit, m, rows, s, last)
                n_values = rows
            elif rows >= cols:
                first = lin_svd_tall_run(kit, m, full, rtol, s, last)
            else:
                var vt2 = 0
                var mt = kit.transpose(m)
                var u2 = lin_svd_tall_run(kit, mt, full, rtol, s, vt2)
                first = kit.transpose(vt2)
                last = kit.transpose(u2)
                n_values = rows
        var u = _lin_pack(kit, first, src, entries, held, context)
        var values = _lin_pack_vector(kit, s, n_values, src, entries, held, context)
        var vt = _lin_pack(kit, last, src, entries, held, context)
        return _lin_named("SVDResult", Python.tuple(u, values, vt), context)
    finally:
        kit.close()


def _lin_svd(args: PythonObject, device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """`svd(a, full_matrices, compute_uv, hermitian)`."""
    var a = lin_operand(args[0], "a", context, True)
    var rows = a.dim(0)
    var cols = a.dim(1)
    if not lin_truth(args[2]):
        var values = lin_svdvals_of(a, device, context)
        _ = a^
        return values
    var full = lin_truth(args[1])
    var hermitian = lin_truth(args[3])
    # TOMBSTONE: MOJOLEARN_SVD_FAST_CHOLQR (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1 (the FAST kit's
    # CholeskyQR2 whole-matrix route ahead of the TSQR); code recoverable at b639a2bd2.
    # Restore: git apply experiments/removed/MOJOLEARN_SVD_FAST_CHOLQR.patch; record in docs/TOMBSTONES.md.
    if not hermitian and not full and _lin_tsqr_on(rows, cols):
        var binding = _lin_xd(device, context)
        var kit = _lin_kit(binding)
        try:
            var py = Python()
            # `_svd_tsqr`: the SVD of the TSQR's R, then U = Q U_R
            var rtol = _lin_rtol(context)
            var owner = 0
            var r = _lin_tsqr_r(kit.entry, a, rows, cols, owner, context)
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
                    _lin_tsqr_release(kit.entry, owner)
            var u = _lin_tsqr_q(kit.entry, owner, Int(py=classical_address(c_store)), rows, cols, cols, context)
            _ = c_store  # keep the Python object alive through the native call above
            var result = _lin_named("SVDResult", Python.tuple(u.keep, _lin_out_vector(kit, s, cols, context),
                                                              _lin_out(kit, vt, context)), context)
            _ = c_store  # keep the Python object alive through the native call above
            _ = a^
            return result
        finally:
            kit.close()
    var result = _lin_svd_kit(_lin_host_source(a), full, hermitian, device, PythonObject(None), PythonObject(None),
                              context)
    _ = a^  # keep the admitted owner alive through the RUN above
    return result


def _lin_decomp_name(operation: Int) -> String:
    """The public name a `device=` refusal of `operation` leads with."""
    if operation == LINOPS_OPN_FAMILY_LIN_QR:
        return "mojolearn.linalg.qr"
    if operation == LINOPS_OPN_FAMILY_LIN_EIGH:
        return "mojolearn.linalg.eigh"
    if operation == LINOPS_OPN_FAMILY_LIN_SVDVALS:
        return "mojolearn.linalg.svdvals"
    return "mojolearn.linalg.svd"


# lane vdf-linalg5 (2026-11-24): a TrainDeviceArray operand (bindings/linalg_dev_arrays_unpack.mojo, the training
# ops' one device-array convention) is read in place by the linalg binding's device entries (gemm/linalg_dev_arrays.mojo
# `svdvals_dev_native`, `qr_r_dev_native`: the host-address entries' kernels on a pool view); the result is a new
# TrainDeviceArray. Lane vdf-linalg6 (2026-11-24): the forms whose host route runs on the x_decomp kit or the TSQR
# table (eigh, qr 'reduced' / 'complete' / 'raw', a wide qr 'r', svd with compute_uv) read it too: the kit adopts the
# array device to device (`LinKit.adopt`, XD_DEV_FROM_TRAIN), runs the host operand's sequence on the same words
# (the TSQR on pool ids: XD_DEV_TSQR_R / XD_DEV_TSQR_Q), and PACK copies each result into a new TrainDeviceArray
# (XD_DEV_TO_TRAIN). Same route, same bits as a host operand. Lane vdf-linalg7: the TSQR's R stays on the device too
# (its NaN refusal is a device scan; XD_DEV_R_SIGNS signs it there), so nothing but status words comes down.


def _lin_decomp_dev(operation: Int, args: PythonObject, device: FunctionDevice, held: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """qr, eigh, svdvals and svd of a TrainDeviceArray; every result a new TrainDeviceArray."""
    var name = _lin_decomp_name(operation)
    lin_dev_device(device, name)
    var who = String("qr") if operation == LINOPS_OPN_FAMILY_LIN_QR else (
        String("eigh") if operation == LINOPS_OPN_FAMILY_LIN_EIGH else (
            String("svdvals") if operation == LINOPS_OPN_FAMILY_LIN_SVDVALS else String("svd")))
    # (outside an identical GPU set the device entries refuse in words: gemm/linalg_dev_arrays.mojo `lin_dev_ready`,
    # x_decomp/resident_train.mojo `_train_ready`)
    var entries = lin_dev_entries(args[0], held)
    var a = lin_dev_operand(args[0], "a", who, entries, held, TRAIN_DEV_F32, 2)
    var rows = a.dim(0)
    var cols = a.dim(1)
    var src = LinSource(0, a.handle, rows, cols)
    var values_only = operation == LINOPS_OPN_FAMILY_LIN_SVDVALS or (
        operation == LINOPS_OPN_FAMILY_LIN_SVD and not lin_truth(args[2]))
    if values_only:
        # the linalg binding of the call and its operations table, published for `lin_call` (`_lin_door`)
        _ = _lin_door(device, context)
        # as `lin_svdvals_of`: a wide A's values are A^T's, transposed by the entry on the device
        var wide = rows < cols
        var n_values = rows if wide else cols
        var out = lin_dev_new(entries, held, [n_values], TRAIN_DEV_F32)
        var params = List[Int]()
        if wide:
            params.append(cols)
            params.append(rows)
            params.append(1)
        else:
            params.append(rows)
            params.append(cols)
        _ = lin_call(context, LIN_OP_SVDVALS_DEV, [a.handle, lin_dev_handle(out)], params)
        return out
    if operation == LINOPS_OPN_FAMILY_LIN_EIGH:
        var lower = _lin_eigh_admit(args[1], rows, cols)
        return _lin_eigh_of(src, lower, device, entries, held, context)
    if operation == LINOPS_OPN_FAMILY_LIN_QR:
        var mode = _lin_qr_mode(args[1])
        if mode == "r" and rows >= cols:
            _ = _lin_door(device, context)
            var r = lin_dev_new(entries, held, [cols, cols], TRAIN_DEV_F32)
            # ORDER as qr_r: (a, r_out), params (n_rows, n_cols)
            _ = lin_call(context, LIN_OP_QR_R_DEV, [a.handle, lin_dev_handle(r)], [rows, cols])
            return r
        if mode == "reduced" and _lin_tsqr_on(rows, cols):
            return _lin_qr_tsqr_dev(src, device, entries, held, context)
        return _lin_qr_kit(src, mode, device, entries, held, context)
    var full = lin_truth(args[1])
    var hermitian = lin_truth(args[3])
    if not hermitian and not full and _lin_tsqr_on(rows, cols):
        return _lin_svd_tsqr_dev(src, device, entries, held, context)
    return _lin_svd_kit(src, full, hermitian, device, entries, held, context)


def lin_decomp_operation(operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # bw2-linalg: `operation` is a leaf code of bindings/operation_names_linops.mojo.
    # Admit public scalar controls before equality/truth conversion; input
    # exporters are captured by the shared native linalg buffer boundary.
    # The one admission of the operand: every _lin_* below passes
    # admitted=True to lin_operand (row N237).
    # lane vdf-linalg: `device` is the last item of the tuple; it is read
    # once (one identity compare for None) after the argument admission and
    # before any work, and selects the context (`lin_device_context`) and the
    # two bindings (`_lin_door`, `_lin_xd`) of this call.
    var last = len(args) - 1
    # lane vdf-linalg5: one dictionary read and one identity compare; a device array never reaches host admission.
    var held = lin_train_device(context)
    if held is not None and lin_is_device_array(args[0], held):
        for i in range(1, last):  # small-loop(public decomposition option metadata)
            host_inference_scalar(args[i], context, True)
        var dev_device = function_device_default()
        var dev_scope = context
        if args[last] is not None:
            dev_device = lin_device(args[last], _lin_decomp_name(operation), LIN_HAS_GPU, LIN_HAS_CPU, context)
            dev_scope = lin_device_context(dev_device, context)
        return _lin_decomp_dev(operation, args, dev_device, held, dev_scope)
    classical_admit_value(args[0], context)
    for i in range(1, last):  # small-loop(public decomposition option metadata)
        host_inference_scalar(args[i], context, True)
    var device = function_device_default()
    var scope = context
    if args[last] is not None:
        device = lin_device(args[last], _lin_decomp_name(operation), LIN_HAS_GPU, LIN_HAS_CPU, context)
        scope = lin_device_context(device, context)
    if operation == LINOPS_OPN_FAMILY_LIN_QR:
        return _lin_qr(args, device, scope)
    if operation == LINOPS_OPN_FAMILY_LIN_EIGH:
        return _lin_eigh(args, device, scope)
    if operation == LINOPS_OPN_FAMILY_LIN_SVDVALS:
        return _lin_svdvals(args, device, scope)
    if operation == LINOPS_OPN_FAMILY_LIN_SVD:
        return _lin_svd(args, device, scope)
    raise Error("RuntimeError: unknown linalg operation " + String(operation))
