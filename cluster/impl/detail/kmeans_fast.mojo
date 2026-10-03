"""FAST + Apple opt-in switches for the k-means fit (lane/apple-fast-core,
2026-10-02). Every switch is a `-D MOJOLEARN_<NAME>` define read with
`is_defined` inside `KMEANS_FAST_APPLE` (no env read: that was a host step)
and defaults OFF; nothing here is compiled under IDENTICAL, whose bits never
move. The device-scale switch this file first carried is gone: main's
`plan_sum_scale(ctx, x, ...)` forms the fixed-point scale on the device.

    -D MOJOLEARN_KMEANS_FAST_ROWNORM
        `row_norm_kernel` is launched ONE BLOCK PER ROW (`grid_dim=(n_samples,
        1, 1)`, `NORM_TPB` threads folding `d` values): 4,000,000 blocks of
        128 threads for 11 features each, three times per fit
        (`cluster/estimator.mojo::kmeans_fit`, `detail/kmeans.mojo::
        kmeans_fit_main_traced`, `kmeans_predict`). `fast_row_sqnorm_kernel`
        is one thread per row, 256 rows per block, a serial fold over the
        row; taken for `d <= KMEANS_FAST_ROWNORM_MAX_D` and the squared
        (`take_sqrt == 0`) norms, which is every shipped metric. The fold
        order differs from the block tree, so the norm bits can move within
        FAST; the assignment argmin is unchanged except on exact ties.
    -D MOJOLEARN_KMEANS_FAST_SKIP_PREDICT
        `kmeans_fit` runs `fit_predict`: the fit, whose loop already ends
        with a fresh assignment against the FINAL centroids
        (`kmeans_fit_main_traced`, `:500-537`), then `predict`, the same
        assignment again over the same `x`, `x_norm` and centroids. With one
        restart (`n_init == 1`, the board's setting) the second pass writes
        the labels the first wrote, so it and the `x_norm` pass that feeds
        it are skipped. Same kernel on the same inputs: same labels.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime KMEANS_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and has_apple_gpu_accelerator()
)

comptime KMEANS_FAST_ROWNORM_TPB = 256
comptime KMEANS_FAST_ROWNORM_MAX_D = 64


def kmeans_fast_rownorm_on(n_features: Int) -> Bool:
    comptime if not (KMEANS_FAST_APPLE and is_defined["MOJOLEARN_KMEANS_FAST_ROWNORM"]()):
        return False
    else:
        return n_features >= 1 and n_features <= KMEANS_FAST_ROWNORM_MAX_D


def kmeans_fast_skip_predict_on() -> Bool:
    comptime if not (KMEANS_FAST_APPLE and is_defined["MOJOLEARN_KMEANS_FAST_SKIP_PREDICT"]()):
        return False
    else:
        return True


def fast_row_sqnorm_kernel(
    out_norm: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """One thread per row: the squared L2 norm of row `row`, a serial fold."""
    var row = Int(block_idx.x) * KMEANS_FAST_ROWNORM_TPB + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    var n_cols = Int(n_cols_in)
    var base = row * n_cols
    var acc = Float32(0.0)
    for c in range(n_cols):
        var v = a.unsafe_load(base + c)
        acc = acc + v * v
    out_norm.unsafe_store(row, acc)


def launch_fast_row_sqnorm(
    ctx: DeviceContext,
    mut out_norm: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
) raises:
    ctx.enqueue_function[fast_row_sqnorm_kernel](
        out_norm.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=((n_rows + KMEANS_FAST_ROWNORM_TPB - 1) // KMEANS_FAST_ROWNORM_TPB, 1, 1),
        block_dim=(KMEANS_FAST_ROWNORM_TPB, 1, 1),
    )
