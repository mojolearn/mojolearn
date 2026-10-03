"""FAST + Apple opt-in switches for the k-means fit (lane/apple-fast-core,
2026-10-02). Every switch is a `-D MOJOLEARN_<NAME>` define read with
`is_defined` inside `KMEANS_FAST_APPLE` (no env read: that was a host step)
and defaults OFF; nothing here is compiled under IDENTICAL, whose bits never
move. The device-scale switch this file first carried is gone: main's
`plan_sum_scale(ctx, x, ...)` forms the fixed-point scale on the device.

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

def kmeans_fast_skip_predict_on() -> Bool:
    comptime if not (KMEANS_FAST_APPLE and is_defined["MOJOLEARN_KMEANS_FAST_SKIP_PREDICT"]()):
        return False
    else:
        return True
