# SPDX-License-Identifier: Apache-2.0
"""The held predict query of a resident kernel model (lane
fix-svc-query-resident, 2026-11-14; core/query_x_held.mojo,
core/device_query_x.mojo `DeviceQueryX.held`, core/fitted_resident_store.mojo
`fitted_query_take` / `fitted_query_put`):

  * the held form puts the same words on the device as the per-call form,
    and its finite word agrees with the per-call word on a finite and on a
    non-finite query;
  * the held status pair is never filled per call, yet a NaN seen by one
    call is not reported by the next finite call (the call's mark);
  * a call with fewer rows reuses the held buffer (same device address, a
    prefix view); a call with more rows replaces it (grow only);
  * a live lease keeps the buffers between calls, a second take while one
    call has them finds none, and the lease's close frees them.

This is the completion test of the lane: the SVC / SVR / GP / KernelRidge
resident predicts call exactly these functions. GPU columns only; the test
owns its context and its store."""
from std.math import nan
from std.memory import bitcast
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_query_x import DeviceQueryX, QueryFold, resident_query_x, resident_query_x_done
from core.query_x_held import IDN_QUERY_X_HELD, QueryXBuffers
from core.fitted_resident_store import (
    FittedResidentStore,
    fitted_lease_open,
    fitted_lease_close,
    fitted_query_take,
    fitted_query_put,
    fitted_resident_view,
    fitted_resident_stats,
)
from core.shared_slot import SharedSlot
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report


# ---- The device column (was query_x_held_gpu_suite.mojo) ----
comptime _TEST_STORE = SharedSlot[FittedResidentStore, "MojolearnQueryXHeldTestStoreV1"]


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _words(n: Int, seed: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=0.0)
    for i in range(n):  # test fixture: n words
        out[i] = Float32(((i * 31 + seed * 7) % 101) - 50) / 16.0
    return out^


def _download(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var host = List[Float32](length=n, fill=0.0)
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(host.unsafe_ptr()))
    ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
    ctx.synchronize()
    return host^


def _same_words(got: List[Float32], want: List[Float32], n: Int, what: String) raises:
    _require(len(got) >= n and len(want) >= n, what + ": too few words")
    for i in range(n):  # test fixture: compare every word
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            raise Error(what + ": word " + String(i) + " differs")


def _held_matches_per_call(ctx: DeviceContext) raises:
    var rows = 37
    var cols = 5
    var n = rows * cols
    var x = _words(n, 1)
    var per_call = DeviceQueryX(ctx, Int(x.unsafe_ptr()), rows, cols, QueryFold.none())
    var held = DeviceQueryX.held(ctx, Int(x.unsafe_ptr()), rows, cols, Optional[QueryXBuffers]())
    ctx.synchronize()
    _require(held.bufs.held() and not per_call.bufs.held(), "held and per-call forms")
    _same_words(_download(ctx, held.x, n), _download(ctx, per_call.x, n), n, "held x vs per-call x")
    _same_words(_download(ctx, held.x, n), x, n, "held x vs host words")
    _require(held.finite_word() == 0 and per_call.finite_word() == 0, "a finite query is not flagged")
    _require(held.range_word() == 0 and per_call.range_word() == 0, "host words have no range flag")
    var bad = x.copy()
    bad[n - 3] = nan[DType.float32]()
    var per_call_bad = DeviceQueryX(ctx, Int(bad.unsafe_ptr()), rows, cols, QueryFold.none())
    var held_bad = DeviceQueryX.held(ctx, Int(bad.unsafe_ptr()), rows, cols, held^.release())
    ctx.synchronize()
    _require(per_call_bad.finite_word() == 1, "per-call form flags a NaN")
    _require(held_bad.finite_word() == 1, "held form flags a NaN")
    _ = per_call^
    _ = per_call_bad^
    _ = held_bad^
    _ = len(x)
    _ = len(bad)
    print("PASS query_x_held held matches per-call")


def _held_marks_and_growth(ctx: DeviceContext) raises:
    var cols = 4
    var big = _words(50 * cols, 2)
    var bad = big.copy()
    bad[7] = Float32(1.0) / Float32(0.0)
    var small = _words(20 * cols, 3)
    var bigger = _words(80 * cols, 4)
    # 1: finite, 50 rows (new held buffers of exactly 50 x 4).
    var q = DeviceQueryX.held(ctx, Int(big.unsafe_ptr()), 50, cols, Optional[QueryXBuffers]())
    ctx.synchronize()
    _require(q.finite_word() == 0, "call 1 finite")
    var addr = Int(q.bufs.x.unsafe_ptr())
    var mark1 = q.bufs.mark
    # 2: an infinity, same rows: same buffer, flagged.
    q = DeviceQueryX.held(ctx, Int(bad.unsafe_ptr()), 50, cols, q^.release())
    ctx.synchronize()
    _require(Int(q.bufs.x.unsafe_ptr()) == addr, "call 2 reuses the held buffer")
    _require(q.bufs.mark != mark1, "call 2 has its own mark")
    _require(q.finite_word() == 1, "call 2 flags the infinity")
    # 3: finite again, no fill in between: call 2's hit is not call 3's.
    q = DeviceQueryX.held(ctx, Int(big.unsafe_ptr()), 50, cols, q^.release())
    ctx.synchronize()
    _require(q.finite_word() == 0, "call 3 does not see call 2's flag")
    # 4: fewer rows: a prefix of the same buffer, the right words.
    q = DeviceQueryX.held(ctx, Int(small.unsafe_ptr()), 20, cols, q^.release())
    ctx.synchronize()
    _require(Int(q.bufs.x.unsafe_ptr()) == addr and q.bufs.capacity == 50 * cols, "call 4 keeps the held buffer")
    _require(Int(q.x.unsafe_ptr()) == addr, "call 4 reads a prefix view")
    _same_words(_download(ctx, q.x, 20 * cols), small, 20 * cols, "call 4 prefix words")
    _require(q.finite_word() == 0, "call 4 finite")
    # 5: more rows: the buffers grow (new buffers of 80 x 4).
    q = DeviceQueryX.held(ctx, Int(bigger.unsafe_ptr()), 80, cols, q^.release())
    ctx.synchronize()
    _require(q.bufs.capacity == 80 * cols, "call 5 grows the held buffer")
    _same_words(_download(ctx, q.x, 80 * cols), bigger, 80 * cols, "call 5 words")
    _require(q.finite_word() == 0, "call 5 finite")
    _ = q^
    _ = len(big)
    _ = len(bad)
    _ = len(small)
    _ = len(bigger)
    print("PASS query_x_held marks and growth")


def _lease_holds(ctx: DeviceContext) raises:
    var store = _TEST_STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var lease = fitted_lease_open(store)
    # One fitted upload under the lease sets the store's cap (a predict
    # makes it before it gives its query back).
    var fitted = _words(16, 5)
    var view = fitted_resident_view(store, ctx, lease, 0, [Int(fitted.unsafe_ptr())], [16])
    _ = view^
    var rows = 30
    var cols = 3
    var x = _words(rows * cols, 6)
    var q = resident_query_x(store, ctx, lease, Int(x.unsafe_ptr()), rows, cols, QueryFold.none())
    ctx.synchronize()
    comptime if IDN_QUERY_X_HELD:
        _require(q.bufs.held(), "a live lease's predict uses held buffers")
        var addr = Int(q.bufs.x.unsafe_ptr())
        _require(not fitted_query_take(store, lease), "a second take while a call holds them finds none")
        resident_query_x_done(store, lease, q^)
        _require(fitted_resident_stats(store)[5] == 1, "the lease keeps its query after the call")
        var q2 = resident_query_x(store, ctx, lease, Int(x.unsafe_ptr()), rows, cols, QueryFold.none())
        ctx.synchronize()
        _require(Int(q2.bufs.x.unsafe_ptr()) == addr, "the next call reuses the lease's buffers")
        _require(fitted_resident_stats(store)[5] == 0, "taken while the call runs")
        _same_words(_download(ctx, q2.x, rows * cols), x, rows * cols, "the reused buffers hold this call's words")
        resident_query_x_done(store, lease, q2^)
        _require(fitted_resident_stats(store)[5] == 1, "given back again")
        fitted_lease_close(store, lease)
        _require(fitted_resident_stats(store)[5] == 0, "the lease's close frees its query")
        # A give-back after the close is dropped.
        var q3 = DeviceQueryX.held(ctx, Int(x.unsafe_ptr()), rows, cols, Optional[QueryXBuffers]())
        ctx.synchronize()
        fitted_query_put(store, lease, q3^.release())
        _require(fitted_resident_stats(store)[5] == 0, "a closed lease keeps nothing")
    else:
        _require(not q.bufs.held(), "the B arm uses per-call buffers")
        resident_query_x_done(store, lease, q^)
        _require(fitted_resident_stats(store)[5] == 0, "the B arm holds nothing")
        fitted_lease_close(store, lease)
    # A fold or lease 0 always takes per-call buffers.
    var q0 = resident_query_x(store, ctx, 0, Int(x.unsafe_ptr()), rows, cols, QueryFold.none())
    ctx.synchronize()
    _require(not q0.bufs.held(), "lease 0 uses per-call buffers")
    resident_query_x_done(store, 0, q0^)
    _ = len(x)
    _ = len(fitted)
    print("PASS query_x_held lease holds")


def _run_gpu(mut report: Report) raises:
    comptime assert TARGET_COLUMN != COLUMN_CPU
    var ctx = DeviceContext()
    _held_matches_per_call(ctx)
    _held_marks_and_growth(ctx)
    _lease_holds(ctx)
    ctx.synchronize()


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("query_x_held", cpu=False, gpu=True, serial="", run=run_cases)
