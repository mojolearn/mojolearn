"""Host-pointer surface for normal-equations ordinary least squares."""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from core.gemm import gemv_n
from glm.ported.linalg.detail.lstsq import lstsq_eig


def _add_scalar_kernel(
    dst: MutPointer[Float32, MutAnyOrigin], n_in: Int32, value: Float32
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i, dst.unsafe_load(i) + value)


def ols_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
) raises:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var w = ctx.enqueue_create_buffer[DType.float32](n_features)
    var cov = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var q = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var qs = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var s = ctx.enqueue_create_buffer[DType.float32](n_features)
    var ab = ctx.enqueue_create_buffer[DType.float32](n_features)
    var inv = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var xa = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.synchronize()
    lstsq_eig(ctx, x, y, w, cov, q, qs, s, ab, inv, xa, xa2, n_rows, n_features)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    for i in range(n_features):
        coef_ptr.unsafe_store(i, hw.unsafe_ptr().unsafe_load(i))


def ols_predict_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    intercept: Float32,
) raises:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var coef = ctx.enqueue_create_buffer[DType.float32](n_features)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=coef, src_ptr=coef_ptr)
    ctx.synchronize()
    gemv_n(ctx, out, x, coef, n_rows, n_features)
    if intercept != Float32(0.0):
        ctx.enqueue_function[_add_scalar_kernel](
            out.unsafe_ptr(), Int32(n_rows), intercept,
            grid_dim=((n_rows + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
    var hout = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=out)
    ctx.synchronize()
    for i in range(n_rows):
        out_ptr.unsafe_store(i, hout.unsafe_ptr().unsafe_load(i))
