# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for the KPSS test and the differencing-order choice.

This is the entry `bindings/_mojolearn_tsa.mojo` calls. It is shaped like
`kde/estimator.mojo` and `decomposition/estimator.mojo`: raw host pointers
in and out, a `DeviceContext` created per call and destroyed with it, no
pointer retained past the call.

`kpss_test_host` runs `ML::Stationarity::kpss_test`
(`tsa/impl/stationarity.mojo`) on a batch and writes back the per
series stationarity flag and the per series statistic.
`select_d_host` runs auto_arima's "Choose the hyper-parameter d" loop
(`tsa/impl/auto_arima.mojo::select_d`) and writes back the chosen `d`
per series.

LAYOUT, AND IT IS cuML's. `y` is `batch_size * n_obs` float32 with each
SERIES CONTIGUOUS: series `b` occupies `[b * n_obs, (b + 1) * n_obs)`.
That is exactly what `stationarity.pyx` produces, because it calls
`check_array(y, order="F")` on an `(n_obs, batch_size)` array, and it is
what `matrix.cuh:74-76` indexes (`tsa/impl/linalg/batched/matrix.mojo`).
The Python wrapper is responsible for handing over that flat order and
says so.

WHAT THIS SURFACE DOES NOT ADD. Every refusal below already exists one
layer down and is raised there by name: `batch_size < 1` and
`n_obs <= d + s*D` in `stationarity.mojo::kpss_test`, `d + D > 2` and
`D with s < 2` in `tsa/impl/timeSeries/arima_helpers.mojo::prepare_data`,
non-finite input in `stationarity.mojo::_refuse_non_finite`, and
`0 <= d_max <= 2 - D` in `auto_arima.mojo::select_d`. The only check added
here is that `batch_size` and `n_obs` are at least one, which is what this
layer needs before it multiplies them into a read length; the null address
is refused one layer up, in `bindings/_mojolearn_tsa.mojo`.

CROSS-VENDOR STATUS. The gates behind this surface have been run on one
Apple M4 in both numeric modes and on nothing else. This lane has never
been carried on an E1 leg (`tools/e1_bootstrap.sh` phase 8 does not name
it) and has no row in `IDENTITY_PATHS.md`. Nothing here may be described
as certified across vendors.
"""

# DEVIATION 2486: bulk host staging; stream/lifetime boundaries unchanged.
from bindings.hostptr import copy_f32
from max.gpu.host import DeviceBuffer, DeviceContext

from tsa.impl.timeSeries.stationarity import KPSS_ELEM_TPB, download_results
from tsa.impl.auto_arima import select_d
from tsa.impl.stationarity import kpss_test


from core.neural_context import neural_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _CTX_MODE, NUMERIC_IDENTICAL as _CTX_IDENTICAL

#: ONE process-lifetime DeviceContext for this module's binding entries
#: (lane/sequence-apple2, the pattern of a5f27d9c2 / core/neural_context.mojo).
#: A context per call paid Metal's context and pipeline setup on every call
#: (AutoARIMA's search makes 14 ARIMA fits) and can run the M2 Pro out of
#: command queues. Same kernels, same order; each entry synchronizes before it
#: returns, so no bit moves.
comptime _CTX_NAME = "MojoTsaEstimatorContextIdentical" if _CTX_MODE == _CTX_IDENTICAL else "MojoTsaEstimatorContextFast"


def _binding_ctx() raises -> DeviceContext:
    return neural_ctx[_CTX_NAME]()



def _upload_f32(
    ctx: DeviceContext, ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises -> DeviceBuffer[DType.float32]:
    """Copy `n` floats from the caller's memory onto the device.

    The host staging buffer is kept live past the `synchronize` on purpose:
    Mojo frees a buffer at its LAST USE, and `.unsafe_ptr()` is a use, so
    without the trailing `_ = host^` the staging allocation can be gone
    before the copy it feeds has run (mojo-buffer-freed-at-last-use).
    """
    var count = n if n > 0 else 1
    var buf = ctx.enqueue_create_buffer[DType.float32](count)
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    copy_f32(ptr, host.unsafe_ptr(), max(0, n))
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _refuse_empty_shape(batch_size: Int, n_obs: Int, who: String) raises:
    """The two things only this layer can see. A null address is refused in
    `bindings/_mojolearn_tsa.mojo` before it ever reaches here."""
    if batch_size < 1:
        raise Error(
            who + ": batch_size must be >= 1 (batch_size=" + String(batch_size) + ")"
        )
    if n_obs < 1:
        raise Error(who + ": n_obs must be >= 1 (n_obs=" + String(n_obs) + ")")


def kpss_test_host(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    flags_ptr: MutPointer[Int32, MutUntrackedOrigin],
    stat_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval_threshold: Float32,
) raises -> Int:
    """`cuml.tsa.stationarity.kpss_test(y, d, D, s, pval_threshold)`.

    Writes `batch_size` int32 to `flags_ptr` (1 stationary, 0 not) and
    `batch_size` float32 to `stat_ptr` (the KPSS statistic, which cuML does
    not return; it is ours, and the Python wrapper says so). Returns
    `batch_size`.

    The flag is `pvalue > pval_threshold` exactly as
    `stationarity.cuh:249-259` decides it. The statistic of a constant
    series is `0.0` by DEVIATION 672, not a computed NaN; it interpolates
    to their `pvalue = 0.10` and therefore to their decision.
    """
    _refuse_empty_shape(batch_size, n_obs, "kpss_test")
    var ctx = _binding_ctx()
    var y = _upload_f32(ctx, y_ptr, batch_size * n_obs)
    var res = kpss_test(
        ctx, y, batch_size, n_obs, d, D, s, pval_threshold, KPSS_ELEM_TPB
    )
    var rs = download_results(ctx, res, batch_size)
    var flags = rs[0].copy()
    var stats = rs[1].copy()
    for b in range(batch_size):
        flags_ptr.unsafe_store(b, Int32(1) if flags[b] else Int32(0))
        stat_ptr.unsafe_store(b, stats[b])
    _ = res^
    _ = y^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return batch_size


def select_d_host(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    d_ptr: MutPointer[Int32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    D: Int,
    s: Int,
    d_max: Int,
    pval_threshold: Float32,
) raises -> Int:
    """auto_arima's "Choose the hyper-parameter d" block
    (`auto_arima.pyx:318-343`), as a host loop over `kpss_test`.

    Writes `batch_size` int32 to `d_ptr` and returns `batch_size`. `D` is
    NOT chosen here and must be given: their `D` comes from statsmodels'
    STL on the host and is not implemented (`tsa/NOT_IMPLEMENTED.tsv`).
    """
    _refuse_empty_shape(batch_size, n_obs, "select_d")
    var ctx = _binding_ctx()
    var y = _upload_f32(ctx, y_ptr, batch_size * n_obs)
    var chosen = select_d(ctx, y, batch_size, n_obs, D, s, d_max, pval_threshold)
    for b in range(batch_size):
        d_ptr.unsafe_store(b, chosen[b])
    _ = y^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return batch_size
