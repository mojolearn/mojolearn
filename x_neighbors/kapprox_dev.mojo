# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-kapprox: SparseRandomProjection's device draw and the
KernelPCA resident switch (GPU binding).

FAST + Apple only, behind `-D MOJOLEARN_SPARSE_RP_DEVICE` /
`-D MOJOLEARN_KPCA_RESIDENT`; any other build exports the same names and
refuses by name, and the Python layer asks the binding's constants before
taking these paths, so IDENTICAL runs main's code. The chi2 samplers' device
ops (`-D MOJOLEARN_KAPPROX_DEVICE`) were DROPPED-quality; their code is on
lane/apple-fast-kapprox @ 10d5a7970.
"""
from std.python import PythonObject
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_neighbors.items import FP
from x_neighbors.device_ops import xn_ctx, _grid, _tid, BLOCK
from x_neighbors.kapprox_items import kapprox_sparse_rp_item

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
    raise Error("x_neighbors: the kapprox device ops are FAST + Apple only (-D MOJOLEARN_SPARSE_RP_DEVICE)")


# One plain enqueue_copy per buffer on the lane's stream (graph_dev.mojo's
# shape): no staged host copy, no host threads.
def _dev_f(ctx: DeviceContext, addr: Int, count: Int, upload: Bool) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
    if upload and count > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=FP(unsafe_from_address=addr))
    return buf^


def _back_f(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], addr: Int, count: Int) raises:
    if count > 0:
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=addr), src_buf=buf)


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
