# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-kapprox: the chi2 samplers' device drivers (GPU binding).

FAST + Apple only, behind `-D MOJOLEARN_KAPPROX_DEVICE` (`XN_KAPPROX_FAST`);
any other build exports the same names and refuses by name, and the Python
layer asks `x_neighbors_kapprox_fast()` once per call before taking this
path, so IDENTICAL runs main's code. Every op is one upload, grid launches
over every cell, one download: no host loop, no host min, no host draw and
no device-to-host round trip inside the op (x_neighbors/kapprox_items.mojo).
"""
from std.python import PythonObject
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_neighbors.items import FP, IP
from x_neighbors.device_ops import xn_ctx, _grid, _tid, BLOCK, skew_transform_kernel
from x_neighbors.kapprox_items import (
    kapprox_check_item,
    kapprox_achi2_item,
    kapprox_skew_fit_item,
    kapprox_skew_log_item,
    kapprox_sparse_rp_item,
)

#: The switch: FAST, an Apple GPU and `-D MOJOLEARN_KAPPROX_DEVICE`. Default
#: OFF; IDENTICAL never takes it.
comptime XN_KAPPROX_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_KAPPROX_DEVICE"]()
)


def kapprox_fast_binding() raises -> PythonObject:
    """1 when this binary takes the device fit/transform of the chi2 samplers."""
    comptime if XN_KAPPROX_FAST:
        return PythonObject(1)
    return PythonObject(0)


#: KernelPCA.fit resident on x_decomp's kit (one upload of X; the kernel
#: matrix, its centering and the top-k solve never leave the device):
#: FAST + Apple + `-D MOJOLEARN_KPCA_RESIDENT`. A Python-only route; this
#: binary only answers whether it is on.
comptime XN_KPCA_RESIDENT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_KPCA_RESIDENT"]()
)

#: SparseRandomProjection.fit: the sparse matrix in one launch
#: (`op_kapprox_sparse_rp`) and no host finiteness pass over X (transform's
#: device flag refuses a non-finite X): FAST + Apple +
#: `-D MOJOLEARN_SPARSE_RP_DEVICE`.
comptime XN_SPARSE_RP_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_SPARSE_RP_DEVICE"]()
)


def kpca_resident_binding() raises -> PythonObject:
    """1 when KernelPCA.fit takes the resident kit route."""
    comptime if XN_KPCA_RESIDENT:
        return PythonObject(1)
    return PythonObject(0)


def sparse_rp_device_binding() raises -> PythonObject:
    """1 when SparseRandomProjection.fit draws its matrix in one launch here."""
    comptime if XN_SPARSE_RP_DEVICE:
        return PythonObject(1)
    return PythonObject(0)


def _refuse() raises:
    raise Error("x_neighbors: the kapprox device ops are FAST + Apple only (-D MOJOLEARN_KAPPROX_DEVICE)")


# One plain enqueue_copy per buffer on the lane's stream (graph_dev.mojo's
# shape): no staged host copy, no host threads.
def _dev_f(ctx: DeviceContext, addr: Int, count: Int, upload: Bool) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
    if upload and count > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=FP(unsafe_from_address=addr))
    return buf^


def _dev_i(ctx: DeviceContext, addr: Int, count: Int, upload: Bool) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](count if count > 0 else 1)
    if upload and count > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=IP(unsafe_from_address=addr))
    return buf^


def _back_f(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], addr: Int, count: Int) raises:
    if count > 0:
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=addr), src_buf=buf)


def _back_i(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], addr: Int, count: Int) raises:
    if count > 0:
        ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=addr), src_buf=buf)


def kapprox_check_kernel(x: FP, flag: IP, n_: Int64, d_: Int64, strict_: Int64, floor_: Float32):
    var n = Int(n_)
    var d = Int(d_)
    var strict = Int(strict_)
    var t = _tid()
    if t < n * d:
        kapprox_check_item(t, x, flag, n, d, strict, floor_)


def kapprox_achi2_kernel(x: FP, res: FP, flag: IP, n_: Int64, d_: Int64, steps_: Int64, interval_: Float32):
    var n = Int(n_)
    var d = Int(d_)
    var steps = Int(steps_)
    var t = _tid()
    if t < n * d:
        kapprox_achi2_item(t, x, res, flag, n, d, steps, interval_)


def kapprox_skew_fit_kernel(w: FP, off: FP, d_: Int64, nc_: Int64, seed_: Int64):
    var d = Int(d_)
    var nc = Int(nc_)
    var seed = Int(seed_)
    var t = _tid()
    if t < d * nc + nc:
        kapprox_skew_fit_item(t, w, off, d, nc, seed)


def kapprox_skew_log_kernel(x: FP, lx: FP, flag: IP, count_: Int64, skew_: Float32):
    var count = Int(count_)
    var t = _tid()
    if t < count:
        kapprox_skew_log_item(t, x, lx, flag, skew_)


def kapprox_sparse_rp_kernel(res: FP, kc_: Int64, d_: Int64, seed_: Int64, dens_: Float32, scale_: Float32):
    var kc = Int(kc_)
    var d = Int(d_)
    var seed = Int(seed_)
    var t = _tid()
    if t < kc * d:
        kapprox_sparse_rp_item(t, res, kc, d, seed, dens_, scale_)


def op_kapprox_sparse_rp(res: Int, kc: Int, d: Int, seed: Int, dens: Float32, scale: Float32) raises:
    """SparseRandomProjection's kc x d matrix in one launch, one thread per entry."""
    comptime if not XN_SPARSE_RP_DEVICE:
        _refuse()
        return
    var ctx = xn_ctx()
    var count = kc * d
    var d_res = _dev_f(ctx, res, count, False)
    ctx.enqueue_function[kapprox_sparse_rp_kernel](
        d_res.unsafe_ptr(), Int64(kc), Int64(d), Int64(seed), dens, scale,
        grid_dim=_grid(count), block_dim=(BLOCK if count > 1 else 1),
    )
    _back_f(ctx, d_res, res, count)
    ctx.synchronize()
    _ = d_res^
    _ = ctx^


def op_kapprox_check(x: Int, flag: Int, n: Int, d: Int, strict: Int, floor: Float32) raises:
    """flag[0] = 1 when any cell of x[n x d] is below `floor` (strict == 0:
    x < floor; else x <= floor). The fit of both samplers."""
    comptime if not XN_KAPPROX_FAST:
        _refuse()
        return
    var ctx = xn_ctx()
    var d_x = _dev_f(ctx, x, n * d, True)
    var d_flag = _dev_i(ctx, flag, 1, True)
    ctx.enqueue_function[kapprox_check_kernel](
        d_x.unsafe_ptr(), d_flag.unsafe_ptr(), Int64(n), Int64(d), Int64(strict), floor,
        grid_dim=_grid(n * d), block_dim=(BLOCK if n * d > 1 else 1),
    )
    _back_i(ctx, d_flag, flag, 1)
    ctx.synchronize()
    _ = d_x^
    _ = d_flag^
    _ = ctx^


def op_kapprox_achi2(x: Int, res: Int, flag: Int, n: Int, d: Int, steps: Int, interval: Float32) raises:
    """AdditiveChi2Sampler.transform: the map and the negative check in one launch."""
    comptime if not XN_KAPPROX_FAST:
        _refuse()
        return
    var ctx = xn_ctx()
    var d_x = _dev_f(ctx, x, n * d, True)
    var d_res = _dev_f(ctx, res, n * d * (2 * steps - 1), False)
    var d_flag = _dev_i(ctx, flag, 1, True)
    ctx.enqueue_function[kapprox_achi2_kernel](
        d_x.unsafe_ptr(), d_res.unsafe_ptr(), d_flag.unsafe_ptr(), Int64(n), Int64(d), Int64(steps), interval,
        grid_dim=_grid(n * d), block_dim=(BLOCK if n * d > 1 else 1),
    )
    _back_f(ctx, d_res, res, n * d * (2 * steps - 1))
    _back_i(ctx, d_flag, flag, 1)
    ctx.synchronize()
    _ = d_x^
    _ = d_res^
    _ = d_flag^
    _ = ctx^


def op_kapprox_skew_fit(w: Int, off: Int, d: Int, nc: Int, seed: Int) raises:
    """SkewedChi2Sampler.fit: random_weights_[d x nc] and random_offset_[nc]
    drawn on the device, one thread per draw."""
    comptime if not XN_KAPPROX_FAST:
        _refuse()
        return
    var ctx = xn_ctx()
    var count = d * nc + nc
    var d_w = _dev_f(ctx, w, d * nc, False)
    var d_off = _dev_f(ctx, off, nc, False)
    ctx.enqueue_function[kapprox_skew_fit_kernel](
        d_w.unsafe_ptr(), d_off.unsafe_ptr(), Int64(d), Int64(nc), Int64(seed),
        grid_dim=_grid(count), block_dim=(BLOCK if count > 1 else 1),
    )
    _back_f(ctx, d_w, w, d * nc)
    _back_f(ctx, d_off, off, nc)
    ctx.synchronize()
    _ = d_w^
    _ = d_off^
    _ = ctx^


def op_kapprox_skew_transform(x: Int, w: Int, off: Int, res: Int, flag: Int, n: Int, d: Int, nc: Int, skew: Float32) raises:
    """SkewedChi2Sampler.transform: log(x + skewedness) into a device scratch
    (with the -skewedness check fused), then the product and cosine
    (`skew_transform_kernel`) on the same stream. One upload of X, one
    download of the map."""
    comptime if not XN_KAPPROX_FAST:
        _refuse()
        return
    var ctx = xn_ctx()
    var d_x = _dev_f(ctx, x, n * d, True)
    var d_w = _dev_f(ctx, w, d * nc, True)
    var d_off = _dev_f(ctx, off, nc, True)
    var d_res = _dev_f(ctx, res, n * nc, False)
    var d_flag = _dev_i(ctx, flag, 1, True)
    var d_lx = _dev_f(ctx, 0, n * d, False)
    ctx.enqueue_function[kapprox_skew_log_kernel](
        d_x.unsafe_ptr(), d_lx.unsafe_ptr(), d_flag.unsafe_ptr(), Int64(n * d), skew,
        grid_dim=_grid(n * d), block_dim=(BLOCK if n * d > 1 else 1),
    )
    ctx.enqueue_function[skew_transform_kernel](
        d_lx.unsafe_ptr(), d_w.unsafe_ptr(), d_off.unsafe_ptr(), d_res.unsafe_ptr(), Int64(n), Int64(d), Int64(nc),
        grid_dim=_grid(n * nc), block_dim=(BLOCK if n * nc > 1 else 1),
    )
    _back_f(ctx, d_res, res, n * nc)
    _back_i(ctx, d_flag, flag, 1)
    ctx.synchronize()
    _ = d_x^
    _ = d_w^
    _ = d_off^
    _ = d_res^
    _ = d_flag^
    _ = d_lx^
    _ = ctx^
