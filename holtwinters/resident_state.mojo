# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident fitted components of a Holt-Winters model (lane
ne1-a-decomp-tsa, plan NATIVE_ENTRY_POINTS_AFTER_0866 section A, row N030).

`ExponentialSmoothing.predict` and `forecast` used to upload the packed
fitted components (level, trend, season; `3 * components_len` float32) on
every call, and a predict that spans the sample end and the forecast made two
binding calls with two uploads. The first query after a fit now uploads them
ONCE (`holtwinters_resident_prepare`); a query runs the in-sample kernel and
the forecast in one call over the device copy (`holtwinters_query_resident`).

The registry follows `cluster/resident_centers.mojo` (a process-wide
`_Global` keyed by a monotonically increasing handle, one lock word, the
query copies the entry's context and buffer HANDLES under it; buffers die
before their context, DEVIATION 1946; an entry minted on another device is
moved, the same bytes, to the calling thread's device first).

WHAT DOES NOT CHANGE. The in-sample cells are `hw_predict_in_sample_kernel`
over the same packed words (holtwinters/estimator.mojo
`holtwinters_predict_ptr`), the forecast is `forecast` over the same level,
trend and season words (`holtwinters_forecast_ptr`), and with the identity
trace on the forecast runs `holtwinters_forecast_ptr` itself over a host copy
of the same words: the bits are the per-call path's.
"""
from std.ffi import _Global
from std.atomic import Atomic, Ordering
from std.memory import memcpy

from max.gpu.host import DeviceBuffer, DeviceContext
from core.identity_trace import IdentityTrace
from core.neural_context import process_ctx, neural_device_plan
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from holtwinters.impl.holtwinters import forecast
from holtwinters.impl.tsa.holtwinters_params import SEASONAL_ADDITIVE, seasonal_from_name
from holtwinters.estimator import hw_predict_in_sample_kernel, HW_PREDICT_TPB, holtwinters_forecast_ptr

# The context slot of holtwinters/estimator.mojo `_DEVCTX_SLOT`: resident
# entries and the per-call path share one process context.
comptime _HW_CTX_SLOT = "MojolearnTsaContextIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnTsaContextFastV2"
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _device_of(root: Int) -> Int:
    return root if root > 0 else 0


struct ResidentHWComponents(Movable):
    """One fitted model's packed components (`3 * components_len` float32,
    level then trend then season, each time-major) on the device."""

    var ctx: DeviceContext
    var comps: DeviceBuffer[DType.float32]
    var n: Int
    var batch_size: Int
    var frequency: Int
    var device: Int

    def __init__(out self, comps_ptr: _F32P, n: Int, batch_size: Int, frequency: Int) raises:
        if n <= frequency:
            raise Error(
                "holtwinters predict: n (" + String(n) + ") must exceed frequency ("
                + String(frequency) + "); there would be no fitted components"
            )
        if batch_size < 1:
            raise Error(
                "holtwinters predict: batch_size must be >= 1 (batch_size="
                + String(batch_size) + ")"
            )
        var dplan = neural_device_plan()
        var ctx = process_ctx[_HW_CTX_SLOT](dplan)
        var count = 3 * (n - frequency) * batch_size
        var comps = ctx.enqueue_create_buffer[DType.float32](count)
        ctx.enqueue_copy(dst_buf=comps, src_ptr=comps_ptr)
        ctx.synchronize()
        self.n = n
        self.batch_size = batch_size
        self.frequency = frequency
        self.device = _device_of(dplan.root)
        self.comps = comps^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffer before the context it was created on (DEVIATION 1946).
        _ = self.comps^
        _ = self.ctx^


struct HWResidentRegistry(Movable):
    var entries: Dict[Int, ResidentHWComponents]
    var next_id: Int
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ResidentHWComponents]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime HW_RESIDENT_REGISTRY = _Global[
    StorageType=HWResidentRegistry,
    name=(
        "MojoHWResidentComponentsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else
        "MojoHWResidentComponentsFast"
    ),
    init_fn=HWResidentRegistry.__init__,
]


def _registry() raises -> UnsafePointer[HWResidentRegistry, MutAnyOrigin]:
    return HW_RESIDENT_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _lock(state: UnsafePointer[HWResidentRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _unlock(state: UnsafePointer[HWResidentRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


struct _HWView(Movable):
    var ctx: DeviceContext
    var comps: DeviceBuffer[DType.float32]
    var n: Int
    var batch_size: Int
    var frequency: Int
    var device: Int

    def __init__(out self, ref entry: ResidentHWComponents):
        self.ctx = entry.ctx.copy()
        self.comps = entry.comps.copy()
        self.n = entry.n
        self.batch_size = entry.batch_size
        self.frequency = entry.frequency
        self.device = entry.device

    def __deinit__(deinit self):
        _ = self.comps^
        _ = self.ctx^


def _locked_insert(state: UnsafePointer[HWResidentRegistry, MutAnyOrigin],
                   var entry: ResidentHWComponents) raises -> Int:
    _lock(state)
    if state[].next_id == 9223372036854775807:
        _unlock(state)
        raise Error("resident Holt-Winters handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    _unlock(state)
    return handle


def _locked_pop(state: UnsafePointer[HWResidentRegistry, MutAnyOrigin],
                handle: Int) raises -> Optional[ResidentHWComponents]:
    _lock(state)
    var popped = Optional[ResidentHWComponents](None)
    if handle in state[].entries:
        try:
            popped = Optional[ResidentHWComponents](state[].entries.pop(handle))
        except error:
            _unlock(state)
            raise error^
    _unlock(state)
    return popped^


def _locked_view(state: UnsafePointer[HWResidentRegistry, MutAnyOrigin], handle: Int) raises -> _HWView:
    _lock(state)
    if handle not in state[].entries:
        _unlock(state)
        raise Error("unknown or released resident Holt-Winters handle")
    try:
        var view = _HWView(state[].entries[handle])
        _unlock(state)
        return view^
    except error:
        _unlock(state)
        raise error^


def _locked_replace(state: UnsafePointer[HWResidentRegistry, MutAnyOrigin], handle: Int,
                    var moved: ResidentHWComponents) raises -> ResidentHWComponents:
    _lock(state)
    if handle not in state[].entries:
        _unlock(state)
        raise Error("unknown or released resident Holt-Winters handle")
    try:
        var replaced = state[].entries.pop(handle)
        state[].entries[handle] = moved^
        _unlock(state)
        return replaced^
    except error:
        _unlock(state)
        raise error^


def holtwinters_resident_prepare(comps_ptr: _F32P, n: Int, batch_size: Int, frequency: Int) raises -> Int:
    """Upload the packed fitted components once; the handle every later
    query names."""
    var entry = ResidentHWComponents(comps_ptr, n, batch_size, frequency)
    return _locked_insert(_registry(), entry^)


def holtwinters_resident_release(handle: Int) raises:
    """Drop the device copy; quiet for a handle already released."""
    var released = _locked_pop(_registry(), handle)
    _ = released^


def holtwinters_resident_count() raises -> Int:
    """Live resident component sets (the tools/ lifetime check reads it)."""
    var state = _registry()
    _lock(state)
    var n = len(state[].entries)
    _unlock(state)
    return n


def _hw_view_on_device(handle: Int) raises -> _HWView:
    var device = _device_of(neural_device_plan().root)
    var state = _registry()
    var view = _locked_view(state, handle)
    if view.device == device:
        return view^
    var count = 3 * (view.n - view.frequency) * view.batch_size
    var host = List[Float32](length=count, fill=Float32(0))
    var host_ptr = rebind[_F32P](host.unsafe_ptr())
    view.ctx.enqueue_copy(dst_ptr=host_ptr, src_buf=view.comps)
    view.ctx.synchronize()
    var moved = ResidentHWComponents(host_ptr, view.n, view.batch_size, view.frequency)
    _ = host^
    var fresh = _HWView(moved)
    var replaced = _locked_replace(state, handle, moved^)
    _ = replaced^
    _ = view^
    return fresh^


def holtwinters_query_resident(handle: Int, out_ptr: _F32P, seasonal: String, start: Int, end: Int) raises -> Int:
    """`ExponentialSmoothing.predict(start, end)` (and `forecast(h)` as
    `start = n, end = n + h`) in ONE call over the handle's components:
    `out_ptr` receives `(end - start) * batch_size` float32, time-major, the
    in-sample cells `[start, min(end, n))` first, then the forecast steps
    `[max(start, n), end)`: the words the owner's two per-call entries wrote
    into the same buffer (holtwinters_predict_ptr's guards, in its words)."""
    var view = _hw_view_on_device(handle)
    var n = view.n
    var bs = view.batch_size
    var f = view.frequency
    var st = seasonal_from_name(seasonal)
    if start < 0 or end <= start:
        raise Error(
            "holtwinters predict: need 0 <= start < end (start="
            + String(start) + ", end=" + String(end) + ")"
        )
    var cl = (n - f) * bs
    var steps = end - start
    var k_in = (min(end, n) - start) if start < n else 0
    if k_in > 0:
        var total = k_in * bs
        var out_d = view.ctx.enqueue_create_buffer[DType.float32](total)
        view.ctx.enqueue_function[hw_predict_in_sample_kernel](
            out_d.unsafe_ptr(), view.comps.unsafe_ptr(),
            Int64(cl), Int64(bs), Int64(f),
            Int64(start), Int64(k_in), Int32(1 if st == SEASONAL_ADDITIVE else 0),
            grid_dim=(total + HW_PREDICT_TPB - 1) // HW_PREDICT_TPB,
            block_dim=HW_PREDICT_TPB,
        )
        view.ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out_d)
        view.ctx.synchronize()
        _ = out_d^
    if end > n:
        # The forecast covers `[n, end)`; a query starting past `n` keeps
        # its tail, as the owner's `_shaped(fc, (start - n) * b, ...)` did.
        var h = end - n
        var skip = (start - n) * bs if start > n else 0
        var dst = out_ptr + k_in * bs
        var trace = IdentityTrace()
        if trace.enabled:
            # The traced sequence runs on the host (`holtwinters_forecast_ptr`
            # itself), over a host copy of the same words.
            var host = List[Float32](length=3 * cl, fill=Float32(0))
            var host_ptr = rebind[_F32P](host.unsafe_ptr())
            view.ctx.enqueue_copy(dst_ptr=host_ptr, src_buf=view.comps)
            view.ctx.synchronize()
            var fc = List[Float32](length=h * bs, fill=Float32(0))
            var fc_ptr = rebind[_F32P](fc.unsafe_ptr())
            _ = holtwinters_forecast_ptr(host_ptr, fc_ptr, n, bs, f, seasonal, h)
            memcpy(dest=dst, src=fc_ptr + skip, count=h * bs - skip)
            _ = host^
            _ = fc^
        else:
            var level_d = view.comps.create_sub_buffer[DType.float32](0, cl)
            var trend_d = view.comps.create_sub_buffer[DType.float32](cl, cl)
            var season_d = view.comps.create_sub_buffer[DType.float32](2 * cl, cl)
            var fc_d = view.ctx.enqueue_create_buffer[DType.float32](h * bs)
            forecast(view.ctx, n, bs, f, h, st, level_d, trend_d, season_d, fc_d, trace, -1)
            view.ctx.enqueue_copy(dst_ptr=dst, src_buf=fc_d.create_sub_buffer[DType.float32](skip, h * bs - skip))
            view.ctx.synchronize()
            _ = fc_d^
            _ = season_d^
            _ = trend_d^
            _ = level_d^
    _ = view^
    return steps * bs
