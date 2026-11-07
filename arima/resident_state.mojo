# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident fitted state of a batched ARIMA (lane ne1-a-decomp-tsa,
plan NATIVE_ENTRY_POINTS_AFTER_0866 section A, rows N028 / N029).

`ARIMA.predict` and `ARIMA.forecast` used to stage and upload the training
series `y`, the fit's regressors and the packed fitted parameters on every
call (`_predict_into`, arima/estimator.mojo). The first query after a fit
now uploads them ONCE through `arima_resident_prepare`; the handle is held
by the model's owner (bindings/tsa_arima_native.mojo, a prepared-owner record
released with the model, a refit, or a replaced fitted array) and every
later query uploads only its own future regressors
(`arima_predict_resident`).

The registry follows `cluster/resident_centers.mojo` (one process-wide
`_Global` keyed by a monotonically increasing handle, one lock word because
the entries run with the GIL released, a query copies the entry's context
and buffer HANDLES under the lock so a concurrent release only drops the
registry's references; the buffers die before their context, DEVIATION
1946). Each entry records the device it was minted on; a query on a thread
bound to another device moves the same bytes there first.

WHAT DOES NOT CHANGE. A resident query runs `arima_predict_device`, the
device half `_predict_into` itself runs after its upload, over the same
words (the regressors already in the filter layout `exog_filter_layout`
returns): the same kernels read the same bytes, so the bits are the same.
"""
from std.ffi import _Global
from std.atomic import Atomic, Ordering

from max.gpu.host import DeviceBuffer, DeviceContext
from core.hostptr import copy_f32
from core.neural_context import process_ctx, neural_device_plan
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from arima.exog_layout import exog_filter_layout
from arima.estimator import arima_predict_device, _order, _refuse_shape

# The context slot `_predict_into` uses (arima/estimator.mojo `_DEVCTX_SLOT`):
# a resident entry and the per-call path share one process context.
comptime _ARIMA_CTX_SLOT = "MojoArimaContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoArimaContextFast"
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _device_of(root: Int) -> Int:
    """The device a context made for plan root `root` runs on: `process_ctx`
    opens `root` when it is positive and device 0 otherwise."""
    return root if root > 0 else 0


struct ResidentArimaFit(Movable):
    """One fitted model's `y` (batch_size x n_obs), its regressors in the
    filter layout and its packed parameters (N x batch_size), on the device."""

    var ctx: DeviceContext
    var y: DeviceBuffer[DType.float32]
    var exog: DeviceBuffer[DType.float32]
    var xin: DeviceBuffer[DType.float32]
    var exog_count: Int
    # batch_size, n_obs, p, d, q, P, D, Q, s, k, n_exog
    var words: List[Int]
    var device: Int

    def __init__(
        out self,
        y_ptr: _F32P,
        exog_host: List[Float32],
        params_ptr: _F32P,
        var words: List[Int],
        params_count: Int,
    ) raises:
        var dplan = neural_device_plan()
        var ctx = process_ctx[_ARIMA_CTX_SLOT](dplan)
        var y_count = words[0] * words[1]
        var exog_count = len(exog_host)
        # The staging `_predict_into` does, once: the same host buffers, the
        # same copies, one synchronize.
        var y = ctx.enqueue_create_buffer[DType.float32](y_count)
        var exog = ctx.enqueue_create_buffer[DType.float32](max(1, exog_count))
        var xin = ctx.enqueue_create_buffer[DType.float32](params_count)
        var y_host = ctx.enqueue_create_host_buffer[DType.float32](y_count)
        var exog_stage = ctx.enqueue_create_host_buffer[DType.float32](max(1, exog_count))
        var params_host = ctx.enqueue_create_host_buffer[DType.float32](params_count)
        copy_f32(y_ptr, y_host.unsafe_ptr(), y_count)
        copy_f32(params_ptr, params_host.unsafe_ptr(), params_count)
        copy_f32(exog_host.unsafe_ptr(), exog_stage.unsafe_ptr(), exog_count)
        ctx.enqueue_copy(dst_buf=y, src_ptr=y_host.unsafe_ptr())
        if exog_count > 0:
            ctx.enqueue_copy(dst_buf=exog, src_ptr=exog_stage.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=xin, src_ptr=params_host.unsafe_ptr())
        ctx.synchronize()
        _ = params_host^
        _ = exog_stage^
        _ = y_host^
        self.exog_count = exog_count
        self.words = words^
        self.device = _device_of(dplan.root)
        self.y = y^
        self.exog = exog^
        self.xin = xin^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.xin^
        _ = self.exog^
        _ = self.y^
        _ = self.ctx^


struct ArimaResidentRegistry(Movable):
    var entries: Dict[Int, ResidentArimaFit]
    var next_id: Int
    # One lock word: queries run with the GIL released (see the module note).
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ResidentArimaFit]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime ARIMA_RESIDENT_REGISTRY = _Global[
    StorageType=ArimaResidentRegistry,
    name=(
        "MojoArimaResidentFitIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else
        "MojoArimaResidentFitFast"
    ),
    init_fn=ArimaResidentRegistry.__init__,
]


def _registry() raises -> UnsafePointer[ArimaResidentRegistry, MutAnyOrigin]:
    return ARIMA_RESIDENT_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _lock(state: UnsafePointer[ArimaResidentRegistry, MutAnyOrigin]):
    # Non-raising (plan section 4, lesson 6).
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _unlock(state: UnsafePointer[ArimaResidentRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


struct _ArimaView(Movable):
    """A query's own copies of one entry's context and buffer HANDLES
    (reference counted, the memory is shared) and its key words."""

    var ctx: DeviceContext
    var y: DeviceBuffer[DType.float32]
    var exog: DeviceBuffer[DType.float32]
    var xin: DeviceBuffer[DType.float32]
    var exog_count: Int
    var words: List[Int]
    var device: Int

    def __init__(out self, ref entry: ResidentArimaFit):
        self.ctx = entry.ctx.copy()
        self.y = entry.y.copy()
        self.exog = entry.exog.copy()
        self.xin = entry.xin.copy()
        self.exog_count = entry.exog_count
        self.words = entry.words.copy()
        self.device = entry.device

    def __deinit__(deinit self):
        _ = self.xin^
        _ = self.exog^
        _ = self.y^
        _ = self.ctx^


def _locked_insert(state: UnsafePointer[ArimaResidentRegistry, MutAnyOrigin],
                   var entry: ResidentArimaFit) raises -> Int:
    _lock(state)
    if state[].next_id == 9223372036854775807:
        _unlock(state)
        raise Error("resident ARIMA handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    _unlock(state)
    return handle


def _locked_pop(state: UnsafePointer[ArimaResidentRegistry, MutAnyOrigin],
                handle: Int) raises -> Optional[ResidentArimaFit]:
    _lock(state)
    var popped = Optional[ResidentArimaFit](None)
    if handle in state[].entries:
        try:
            popped = Optional[ResidentArimaFit](state[].entries.pop(handle))
        except error:
            _unlock(state)
            raise error^
    _unlock(state)
    return popped^


def _locked_view(state: UnsafePointer[ArimaResidentRegistry, MutAnyOrigin],
                 handle: Int) raises -> _ArimaView:
    _lock(state)
    if handle not in state[].entries:
        _unlock(state)
        raise Error("unknown or released resident ARIMA handle")
    try:
        var view = _ArimaView(state[].entries[handle])
        _unlock(state)
        return view^
    except error:
        _unlock(state)
        raise error^


def _locked_replace(state: UnsafePointer[ArimaResidentRegistry, MutAnyOrigin], handle: Int,
                    var moved: ResidentArimaFit) raises -> ResidentArimaFit:
    """Put `moved` under `handle`; the entry it replaces is returned so it
    is destroyed outside the lock. A handle released meanwhile is refused."""
    _lock(state)
    if handle not in state[].entries:
        _unlock(state)
        raise Error("unknown or released resident ARIMA handle")
    try:
        var replaced = state[].entries.pop(handle)
        state[].entries[handle] = moved^
        _unlock(state)
        return replaced^
    except error:
        _unlock(state)
        raise error^


def arima_resident_prepare(
    y_ptr: _F32P,
    exog_address: Int,
    params_ptr: _F32P,
    batch_size: Int,
    n_obs: Int,
    p: Int, d: Int, q: Int, P: Int, D: Int, Q: Int, s: Int, k: Int, n_exog: Int,
) raises -> Int:
    """Upload a fitted model's `y`, regressors and packed parameters once;
    the handle every later `arima_predict_resident` names. The refusals are
    the ones the per-call path makes before its upload, in the same order
    (`_refuse_shape`, the order, the regressors' layout and finite scan)."""
    _refuse_shape(batch_size, n_obs, "arima_predict")
    var order = _order(p, d, q, P, D, Q, s, k, n_exog)
    var exog_host = exog_filter_layout(exog_address, batch_size, n_obs, order.n_exog, "exog")
    var words: List[Int] = [batch_size, n_obs, p, d, q, P, D, Q, s, k, n_exog]
    var entry = ResidentArimaFit(y_ptr, exog_host, params_ptr, words^, order.complexity() * batch_size)
    _ = exog_host^
    # The upload ran outside the lock; only the dictionary step is locked.
    return _locked_insert(_registry(), entry^)


def arima_resident_release(handle: Int) raises:
    """Drop the device copy; quiet for a handle already released (a copied
    instance's record, a release racing the model's death callback)."""
    var released = _locked_pop(_registry(), handle)
    # Destroyed outside the lock: a query still running holds its own handle
    # copies of the context and the buffers.
    _ = released^


def arima_resident_count() raises -> Int:
    """The number of live resident ARIMA fits (the tools/ lifetime check)."""
    var state = _registry()
    _lock(state)
    var n = len(state[].entries)
    _unlock(state)
    return n


def _arima_view_on_device(handle: Int) raises -> _ArimaView:
    """The entry for one query, on the device the calling thread selected
    (moved there first, the same bytes, when it lives on another one)."""
    var device = _device_of(neural_device_plan().root)
    var state = _registry()
    var view = _locked_view(state, handle)
    if view.device == device:
        return view^
    var w = view.words.copy()
    var y_count = w[0] * w[1]
    var params_count = len(view.xin)
    var y_host = List[Float32](length=y_count, fill=Float32(0))
    var exog_host = List[Float32](length=view.exog_count, fill=Float32(0))
    var x_host = List[Float32](length=params_count, fill=Float32(0))
    var y_ptr = rebind[_F32P](y_host.unsafe_ptr())
    var x_ptr = rebind[_F32P](x_host.unsafe_ptr())
    view.ctx.enqueue_copy(dst_ptr=y_ptr, src_buf=view.y)
    if view.exog_count > 0:
        var e_ptr = rebind[_F32P](exog_host.unsafe_ptr())
        view.ctx.enqueue_copy(dst_ptr=e_ptr, src_buf=view.exog.create_sub_buffer[DType.float32](0, view.exog_count))
    view.ctx.enqueue_copy(dst_ptr=x_ptr, src_buf=view.xin)
    view.ctx.synchronize()
    var moved = ResidentArimaFit(y_ptr, exog_host, x_ptr, w^, params_count)
    _ = y_host^
    _ = x_host^
    _ = exog_host^
    var fresh = _ArimaView(moved)
    var replaced = _locked_replace(state, handle, moved^)
    _ = replaced^
    _ = view^
    return fresh^


def arima_predict_resident(
    handle: Int,
    exog_fut_address: Int,
    out_ptr: _F32P,
    start_in: Int,
    end_in: Int,
    forecast: Bool,
    who: String,
) raises -> Int:
    """`_predict_into` over the handle's resident fitted state: the same
    refusals, only the future regressors uploaded, the same device half
    (`arima_predict_device`). Returns `(end - start) * batch_size`. With
    `forecast`, `end_in` is the step count and the bounds are `n_obs` and
    `n_obs + steps` (`arima_forecast_ptr_host`'s)."""
    var view = _arima_view_on_device(handle)
    var w = view.words.copy()
    var batch_size = w[0]
    var n_obs = w[1]
    var start = n_obs if forecast else start_in
    var end = n_obs + end_in if forecast else end_in
    _refuse_shape(batch_size, n_obs, who)
    if start < 0 or end <= start:
        raise Error(
            who + ": need 0 <= start < end (start=" + String(start)
            + ", end=" + String(end) + ")"
        )
    if start > n_obs:
        raise Error(
            who + ": there can't be a gap between the data and the"
            " prediction (start=" + String(start) + ", n_obs="
            + String(n_obs) + ")"
        )
    var order = _order(w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10])
    var num_steps = end - n_obs if end > n_obs else 0
    var fut_host = exog_filter_layout(
        exog_fut_address, batch_size, num_steps, order.n_exog, "exog (future values)"
    )
    var fut_count = len(fut_host)
    var exog_fut = view.ctx.enqueue_create_buffer[DType.float32](max(1, fut_count))
    var fut_stage = view.ctx.enqueue_create_host_buffer[DType.float32](max(1, fut_count))
    copy_f32(fut_host.unsafe_ptr(), fut_stage.unsafe_ptr(), fut_count)
    if fut_count > 0:
        view.ctx.enqueue_copy(dst_buf=exog_fut, src_ptr=fut_stage.unsafe_ptr())
    view.ctx.synchronize()
    _ = fut_stage^
    _ = fut_host^
    var written = arima_predict_device(
        view.ctx, view.y, view.exog, exog_fut, view.xin, out_ptr, batch_size, n_obs, start, end, order
    )
    _ = exog_fut^
    _ = view^
    return written
