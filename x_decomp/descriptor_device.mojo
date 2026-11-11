# SPDX-License-Identifier: Apache-2.0
# The x_decomp kernel-call table of the GPU binding (x_decomp/descriptor_schema.mojo):
# one `abi("C")` dispatch function, exported by bindings/_mojolearn_x_decomp.mojo
# as the capsule `x_decomp_native_api_v1`. Address operations run on DevExec
# (the same executor `ew_py[DevExec]` etc. use); resident operations call the
# typed entries of x_decomp/resident.mojo that the `dev_*_py` doors of
# providers/python/x_decomp_resident_pyapi.mojo call, with the same checks.
# No numeric change: the same kernels, the same operands, the same order.
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from x_decomp.resident import (
    pool_alloc, pool_free, dev_upload_run, dev_download_run, dev_ew_run, dev_gemm_run,
    dev_svd_run, dev_orth_diag_run,
    dev_colsum_run, dev_rowsum_run, dev_sqdist_run, dev_move_run, dev_upload_finite_run,
)
from x_decomp.select_dev import dev_reduce_run, dev_order_f_run
from x_decomp.descriptor_ops import xd_address_op, xd_is_address_op, xd_f32, xd_float
from x_decomp.descriptor_schema import (
    XD_ABI_NAME, XD_MODE, XD_CAPABILITIES, XD_CAP_HOST, XD_CAP_RESIDENT,
    XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD, XD_DEV_DOWNLOAD, XD_DEV_EW, XD_DEV_GEMM,
    XD_DEV_COLSUM, XD_DEV_ROWSUM, XD_DEV_SQDIST, XD_DEV_MOVE, XD_DEV_ISO_GATHER_MIN, XD_DEV_UPLOAD_FINITE,
    XD_CELL_ADD, XD_CELL_MIN, XD_MV_TAKE_ROWS, XD_MV_TAKE_COLS,
    XD_DEV_REDUCE, XD_DEV_ORDER, XD_DEV_SVD, XD_DEV_ORTH_DIAG, XD_VENDOR, xd_vendor_code,
    xd_validate_call, xd_dim,
)
from core.shared_slot import SharedSlot


def xd_iso_gather_min(d: Int, im: Int, sq: Int, q: Int, dc: Int, nn: Int, one: Int,
                      dn: Int, imn: Int) raises -> Int:
    """Isomap.transform's geodesic step (row N295) on device ids: the kit's
    per-rank `take_rows_at(D, im, q, nn, a, 0)`, `take_cols(SQ, [a])`,
    `ew(ADD, rows, col)` and `ew(MIN, G, cand)`, the same launches with the
    same parameter words in the same order, without a host crossing between
    them. Returns G's id (q x dc); the caller owns it. Temporaries go back
    to the pool as the kit's buffers did when their holders died."""
    var count = q * dc
    if count <= 0 or nn <= 0:
        raise Error("x_decomp: the geodesic gather needs a nonempty query and neighbor set")
    # `_mode_of(col, rows)`: a q x 1 column against q x dc rows.
    var bmode = 0 if dc == 1 else 2
    var g = -1
    for a in range(nn):  # driver-loop: over neighbor ranks (n_neighbors), one gather-and-min step each, all enqueued on the device
        var rows = pool_alloc(max(count, 1))
        _ = dev_move_run(d, im, rows, XD_MV_TAKE_ROWS, count, dc, 0, 0, nn, a, dn, count, imn)
        # mk_dev_gather(SQ, TAKE_COLS, [a]): the index word uploaded as an
        # exact float32, then the q x 1 column moved out.
        var word = Array[Float32, 1](fill=Float32(Float64(a)))
        var di = pool_alloc(1)
        _ = dev_upload_run(di, F32Ptr(unsafe_from_address=Int(word.unsafe_ptr())), 1)
        _ = word[0]  # keep the index word alive through the synchronous upload
        var col = pool_alloc(max(q, 1))
        _ = dev_move_run(sq, di, col, XD_MV_TAKE_COLS, q, 1, nn, 0, 0, 0, q * nn, q, 1)
        var cand = pool_alloc(max(count, 1))
        _ = dev_ew_run(rows, col, one, cand, XD_CELL_ADD, count, dc, q, bmode, 1, 3, 0.0)
        pool_free(rows)
        pool_free(di)
        pool_free(col)
        if g < 0:
            g = cand
        else:
            var folded = pool_alloc(max(count, 1))
            _ = dev_ew_run(g, cand, one, folded, XD_CELL_MIN, count, dc, count, 0, 1, 3, 0.0)
            pool_free(g)
            pool_free(cand)
            g = folded
    return g


def xd_resident_op(op: Int, p: List[Int], floats: Int, a: List[Int]) raises -> Int:
    if op == XD_DEV_ALLOC:
        return pool_alloc(p[0])
    if op == XD_DEV_FREE:
        pool_free(p[0])
        return 0
    if op == XD_DEV_UPLOAD:
        return dev_upload_run(p[0], xd_f32(a[0]), p[1])
    if op == XD_DEV_UPLOAD_FINITE:
        return dev_upload_finite_run(p[0], xd_f32(a[0]), p[1])
    if op == XD_DEV_DOWNLOAD:
        return dev_download_run(p[0], xd_f32(a[0]), p[1])
    if op == XD_DEV_EW:
        # p = [a, b, c, dst, op, count, d, lb, bm, lc, cm], as dev_ew_py
        return dev_ew_run(p[0], p[1], p[2], p[3], p[4], xd_dim(p[5]), xd_dim(p[6]), xd_dim(p[7]), p[8],
                          xd_dim(p[9]), p[10], xd_float(floats, 0))
    if op == XD_DEV_GEMM:
        return dev_gemm_run(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]), xd_dim(p[5]), p[6] != 0, p[7] != 0)
    if op == XD_DEV_COLSUM:
        return dev_colsum_run(p[0], p[1], xd_dim(p[2]), xd_dim(p[3]))
    if op == XD_DEV_ROWSUM:
        return dev_rowsum_run(p[0], p[1], xd_dim(p[2]), xd_dim(p[3]))
    if op == XD_DEV_SQDIST:
        return dev_sqdist_run(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]), xd_dim(p[5]), xd_dim(p[6]),
                              xd_float(floats, 0))
    if op == XD_DEV_MOVE:
        return dev_move_run(p[0], p[1], p[2], p[3], xd_dim(p[4]), xd_dim(p[5]), xd_dim(p[6]), xd_dim(p[7]),
                            xd_dim(p[8]), xd_dim(p[9]), xd_dim(p[10]), xd_dim(p[11]), xd_dim(p[12]))
    if op == XD_DEV_ISO_GATHER_MIN:
        return xd_iso_gather_min(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]), xd_dim(p[5]), p[6],
                                 xd_dim(p[7]), xd_dim(p[8]))
    if op == XD_DEV_REDUCE:
        return dev_reduce_run(p[0], p[1], xd_dim(p[2]), p[3])
    if op == XD_DEV_ORDER:
        return dev_order_f_run(p[0], p[1], p[2], p[3], xd_dim(p[4]), p[5], p[6])
    if op == XD_DEV_SVD:
        return dev_svd_run(p[0], xd_f32(a[0]), xd_f32(a[1]), xd_dim(p[1]), xd_dim(p[2]))
    if op == XD_DEV_ORTH_DIAG:
        return dev_orth_diag_run(p[0], p[1], xd_f32(a[0]), xd_dim(p[2]), xd_dim(p[3]))
    raise Error("x_decomp native ABI: unknown resident operation " + String(op))


def x_decomp_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                       ff: Int, nf: Int, result: Int, context: Int,
                       sink: NativeDescriptorError) abi("C") -> Int:
    try:
        xd_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var answer = 0
        if operation == XD_MODE:
            answer = GLOBAL_NUMERIC_MODE
        elif operation == XD_VENDOR:
            answer = xd_vendor_code(String(COMPILED_VENDOR))
        elif operation == XD_CAPABILITIES:
            answer = XD_CAP_HOST | XD_CAP_RESIDENT
        elif xd_is_address_op(operation):
            answer = xd_address_op[DevExec](operation, a, p, ff)
        else:
            answer = xd_resident_op(operation, p, ff, a)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XDecompDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(XD_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(x_decomp_device_v1)


# The extension image owns the table and name borrowed by PyCapsule. A named
# global is process-wide, so the name carries the image's numeric mode (as
# x_neighbors/device_ops.mojo names its context): the identical and fast
# images loaded together must each export their own dispatch function.
comptime _X_DECOMP_DEVICE_API_NAME = "MojolearnXDecompDeviceAPIIdenticalv1V1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXDecompDeviceAPIFastv1V1"
comptime X_DECOMP_DEVICE_API = SharedSlot[XDecompDeviceAPI, _X_DECOMP_DEVICE_API_NAME]
