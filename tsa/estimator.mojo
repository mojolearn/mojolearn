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
from core.hostptr import copy_f32
from core.transfer import transfer_upload_ptr
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojolearnTsaContextIdenticalV3" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojolearnTsaContextFastV3"


from tsa.impl.timeSeries.stationarity import KPSS_ELEM_TPB, download_results
from tsa.impl.timeSeries.kpss_fused import (
    KPSS_FUSED_MAX_N, KPSS_PACK_W, TSA_KPSS_PACK, TSA_SELD_FUSED, kpss_rounds,
)
from tsa.impl.auto_arima import select_d
from tsa.impl.select_d_fast import KPSS_ONE_WAIT, kpss_one_wait
from tsa.impl.stationarity import kpss_test
from tsa.tsa_store import TsaSeriesIn, tsa_series_fill
from core.scope_store_slot import scope_input_block


def _upload_f32(
    ctx: DeviceContext, ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises -> DeviceBuffer[DType.float32]:
    """Copy `n` floats from the caller's memory onto the device (a buffer
    of at least one cell). core/transfer.mojo (lane fx-dup-next, 2026-11-19):
    the copy reads the caller's words in place and is drained before the
    return. Gone: a pinned host buffer of the whole input per call and a
    host pass into it.
    """
    return transfer_upload_ptr[DType.float32](ctx, ptr, max(0, n), n if n > 0 else 1)


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
    flags_ptr: MutPointer[UInt8, MutUntrackedOrigin],
    stat_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval_threshold: Float32,
) raises -> Int:
    """`kpss_test` of the caller's series-major host words (the checks' form)."""
    _refuse_empty_shape(batch_size, n_obs, "kpss_test")
    return _kpss_test_series(TsaSeriesIn(y_ptr), flags_ptr, stat_ptr, batch_size, n_obs, d, D, s, pval_threshold)


def kpss_test_host(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    flags_ptr: MutPointer[UInt8, MutUntrackedOrigin],
    stat_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval_threshold: Float32,
    scope_block: Int,
    layout: Int,
) raises -> Int:
    """The operations table's form (lane gd-resident-2): `scope_block` 0, or
    the door's `resident()` scope block (core/scope_store_slot.mojo): the
    series is then the scope's resident slot of the caller's words (C order
    when `layout` is 1), laid out on the device (tsa/tsa_store.mojo)."""
    _refuse_empty_shape(batch_size, n_obs, "kpss_test")
    var src = TsaSeriesIn(y_ptr, scope_input_block(scope_block, 0), layout, batch_size * n_obs)
    return _kpss_test_series(src, flags_ptr, stat_ptr, batch_size, n_obs, d, D, s, pval_threshold)


def _kpss_test_series(
    src: TsaSeriesIn,
    flags_ptr: MutPointer[UInt8, MutUntrackedOrigin],
    stat_ptr: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval_threshold: Float32,
) raises -> Int:
    """`cuml.tsa.stationarity.kpss_test(y, d, D, s, pval_threshold)`.

    Writes `batch_size` uint8 to `flags_ptr` (1 stationary, 0 not) and
    `batch_size` float32 to `stat_ptr` (the KPSS statistic, which cuML does
    not return; it is ours, and the Python wrapper says so). Returns
    `batch_size`.

    The flag is `pvalue > pval_threshold` exactly as
    `stationarity.cuh:249-259` decides it. The statistic of a constant
    series is `0.0` by DEVIATION 672, not a computed NaN; it interpolates
    to their `pvalue = 0.10` and therefore to their decision.
    """
    var ctx = process_ctx[_DEVCTX_SLOT]()
    comptime if TSA_KPSS_PACK:
        # lane/apple-fast-gap-tsa (TSA_KPSS_PACK, default; FAST +
        # Apple): one launch, one device buffer and one host stage for the
        # input and the packed outputs, one wait (kpss_fused.mojo::kpss_rounds)
        if d == 0 and D == 0 and n_obs <= KPSS_FUSED_MAX_N:
            var res = List[Float32]()
            kpss_rounds(ctx, src, batch_size, n_obs, 1, pval_threshold, res)
            for b in range(batch_size):
                stat_ptr.unsafe_store(b, res[b * KPSS_PACK_W])
                flags_ptr.unsafe_store(b, UInt8(1) if res[b * KPSS_PACK_W + 1] != Float32(0.0) else UInt8(0))
            _ = ctx^
            return batch_size
    comptime if KPSS_ONE_WAIT:
        # TOMBSTONE: MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF (old path, no A/B recorded; non-owner site, owner expdel2-36-neural-series-linear) named here only; removed 2026-11-15 by lane/expdel2-37-series; code recoverable at 7b328fcfd6.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF.expdel2-37-series.patch; record in docs/TOMBSTONES.md.
        # IDENTICAL on every vendor (lane/fam-timeseries): one wait instead of four,
        # the primitive's kernels (select_d_fast.mojo::kpss_one_wait). False
        # is a refusal, raised by name by the sequence below.
        if kpss_one_wait(
            ctx, src, flags_ptr, stat_ptr, batch_size, n_obs, d, D, s, pval_threshold
        ):
            _ = ctx^
            return batch_size
    var y: DeviceBuffer[DType.float32]
    if src.resident():
        # lane gd-resident-2: the scope's resident series, laid out on the device
        y = ctx.enqueue_create_buffer[DType.float32](max(1, batch_size * n_obs))
        tsa_series_fill(ctx, y, src, batch_size, n_obs)
    else:
        y = _upload_f32(ctx, src.host, batch_size * n_obs)
    var res = kpss_test(
        ctx, y, batch_size, n_obs, d, D, s, pval_threshold, KPSS_ELEM_TPB
    )
    var rs = download_results(ctx, res, batch_size)
    var flags = rs[0].copy()
    var stats = rs[1].copy()
    for b in range(batch_size):
        flags_ptr.unsafe_store(b, UInt8(1) if flags[b] else UInt8(0))
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
    """`select_d` of the caller's series-major host words (the checks' form)."""
    _refuse_empty_shape(batch_size, n_obs, "select_d")
    return _select_d_series(TsaSeriesIn(y_ptr), d_ptr, batch_size, n_obs, D, s, d_max, pval_threshold)


def select_d_host(
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    d_ptr: MutPointer[Int32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    D: Int,
    s: Int,
    d_max: Int,
    pval_threshold: Float32,
    scope_block: Int,
    layout: Int,
) raises -> Int:
    """The operations table's form (lane gd-resident-2): `kpss_test_host`'s
    scope block and layout words."""
    _refuse_empty_shape(batch_size, n_obs, "select_d")
    var src = TsaSeriesIn(y_ptr, scope_input_block(scope_block, 0), layout, batch_size * n_obs)
    return _select_d_series(src, d_ptr, batch_size, n_obs, D, s, d_max, pval_threshold)


def _select_d_series(
    src: TsaSeriesIn,
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
    var ctx = process_ctx[_DEVCTX_SLOT]()
    comptime if TSA_SELD_FUSED:
        # lane/apple-fast-gap-tsa (TSA_SELD_FUSED, default; FAST +
        # Apple): every round of a series in its block, the first stationary
        # order chosen there, one upload, one launch, one download, one wait
        # (kpss_fused.mojo::kpss_rounds). Seasonal differencing, d_max past 2
        # and series past the threadgroup's words take the path below.
        if D == 0 and d_max >= 0 and d_max <= 2 and n_obs <= KPSS_FUSED_MAX_N:
            for d_ in range(d_max):
                if n_obs <= d_:
                    raise Error(
                        "stationarity: n_obs (" + String(n_obs)
                        + ") must be greater than d + s*D (" + String(d_) + ")"
                    )
            var res = List[Float32]()
            kpss_rounds(ctx, src, batch_size, n_obs, d_max, pval_threshold, res)
            for b in range(batch_size):
                d_ptr.unsafe_store(b, Int32(Int(res[b * KPSS_PACK_W + 3])))
            _ = ctx^
            return batch_size
    # The upload is queued, not waited on (lane fb15-sync-counts, 2026-11-18;
    # `_upload_f32` waited here, a second wait per call): `select_d`'s rounds
    # follow it on the same stream and end in their one wait. The host stage
    # is kept alive past that wait; on a refusal the stream is drained before
    # the stage is freed.
    var count = batch_size * n_obs
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    var y = ctx.enqueue_create_buffer[DType.float32](count)
    if src.resident():
        # lane gd-resident-2: the scope's resident series, laid out on the device
        tsa_series_fill(ctx, y, src, batch_size, n_obs)
    else:
        copy_f32(src.host, host.unsafe_ptr(), count)
        ctx.enqueue_copy(dst_buf=y, src_ptr=host.unsafe_ptr())
    var chosen = List[Int32]()
    try:
        chosen = select_d(ctx, y, batch_size, n_obs, D, s, d_max, pval_threshold)
    except error:
        ctx.synchronize()
        _ = host^
        _ = y^
        raise error^
    for b in range(batch_size):
        d_ptr.unsafe_store(b, chosen[b])
    _ = host^
    _ = y^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return batch_size
