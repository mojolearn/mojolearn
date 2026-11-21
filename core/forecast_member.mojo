# SPDX-License-Identifier: Apache-2.0
"""ARIMA as a MEMBER (core/member_abi.mojo), family 37
`MEMBER_FAMILY_FORECAST`.

One entry, `forecast_member_v2`, published by each `_mojolearn_arima` image
(`forecast_member_publish`, after the image's ARIMA operations table). It
runs the family's existing typed RUN (core/arima_run.mojo: `arima_fit_run`,
`arima_forecast_run`, the resident prepare / query / release) over the
operations entry of the `DeviceRoute` it is handed, the same entry the
estimator's own door (bindings/tsa_arima_native_door.mojo `_arima_fit`,
`arima_predict` with `forecast`) calls. No kernel is touched.

THE FORECAST FORM (MEMBER_CAN_FORECAST): FIT reads the series in `y`, host
float32, observations x series; optional regressors in `x` (host float32,
observations x n_exog) for ONE series only (a batch of series with
regressors is refused in words: the member view has no third axis).
PREDICT writes `dst` = horizon x series float32: without regressors
`dst.rows` is the horizon; with them `x` holds the future regressors
(horizon x n_exog) and `x.rows == dst.rows`.

KINDS: 0 ARIMA.

ROUTE SLOTS (filled by providers/python/member_forecast_boundary_door.mojo):
  * `entry`: the ARIMA operations entry of the route's image
    (`arima_route`, the door's route); `flags` its ARIMA_HAS_* bits.

MEMBER_OP_NEW parameters (Float64 words, 11): [0] kind 0; [1 .. 7] the
order p, d, q and the seasonal order P, D, Q, s; [8] k (trend term, 0 / 1);
[9] method code (the connected `arima_methods` table's code of `method`);
[10] maxiter.

FITTED-STATE BLOCKS, in index order:
  0 `MEMBER_BLOCK_SCALARS` float64 [kind, series, observations, n_exog,
    complexity N];
  1 `MEMBER_BLOCK_FAMILY` float32 series x observations: `_y`;
  2 `+1` float32 (series * observations) x n_exog: `_exog` (rows 0: none);
  3 `+2` float32 series x N: `params_`;  4 `+3` `x_`;  5 `+4` `x0_`;
  6 `+5` float32 1 x 2 series: the log-likelihoods then `fx_`;
  7 `+6` int32 1 x 2 series: `n_iter_` then `retcode_`;
  8 `+7` float64 1 x 2 series: `aic_` then `bic_`.
An import needs blocks 0, 1, 3 (and 2 with regressors); the rest are
optional (a predict reads only the series, the regressors and params_).

RESIDENT STATE: on a route with the resident operations (a GPU image), the
first PREDICT of a fitted state uploads it once (`arima_resident_prepare_run`)
and every predict queries it, as the estimator's own predict record does;
refit, import and FREE release it.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted; the store's lock covers only mint, free and
the handle lookup. The series copy and the forecast publication split their
series (or rows) across the host workers; the fit and the forecast are the
image's own batched operations.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_MULTI_OUTPUT,
    MEMBER_CAN_FORECAST,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_FORECAST
from core.device_route import DeviceRoute
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.arima_run import (
    arima_fit_run, arima_forecast_run, arima_route_fits, arima_route_resident, arima_resident_prepare_run,
    arima_resident_release_run, arima_resident_query_run,
)
from arima.tsa_arima_run import _penalties
from arima.tsa_arima_resident_run import arima_resident_words

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _KIND_ARIMA = 0
comptime _WORDS = 11
comptime _SCALAR_WORDS = 5
comptime _BLOCK_SERIES = MEMBER_BLOCK_FAMILY
comptime _BLOCK_EXOG = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_PARAMS = MEMBER_BLOCK_FAMILY + 2
comptime _BLOCK_X = MEMBER_BLOCK_FAMILY + 3
comptime _BLOCK_X0 = MEMBER_BLOCK_FAMILY + 4
comptime _BLOCK_STATS = MEMBER_BLOCK_FAMILY + 5
comptime _BLOCK_FLAGS = MEMBER_BLOCK_FAMILY + 6
comptime _BLOCK_ICS = MEMBER_BLOCK_FAMILY + 7
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _A = "ARIMA: "


struct ForecastMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var fitted: Bool
    var words: List[Float64]
    var scalars: List[Float64]
    var series: List[Float32]
    var exog: List[Float32]
    var params: List[Float32]
    var x: List[Float32]
    var x0: List[Float32]
    var stats: List[Float32]
    var flags: List[Int32]
    var ics: List[Float64]
    # The resident upload of the fitted state (0: none) and the route that
    # made it (its release goes through the same entry).
    var resident: Int
    var resident_route: DeviceRoute

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.fitted = False
        self.words = List[Float64]()
        self.scalars = List[Float64]()
        self.series = List[Float32]()
        self.exog = List[Float32]()
        self.params = List[Float32]()
        self.x = List[Float32]()
        self.x0 = List[Float32]()
        self.stats = List[Float32]()
        self.flags = List[Int32]()
        self.ics = List[Float64]()
        self.resident = 0
        self.resident_route = DeviceRoute(0)


struct ForecastMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[ForecastMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[ForecastMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[ForecastMemberStore, "MojolearnForecastMemberStoreV2", 2]

comptime _SlotP = MutPointer[ForecastMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[ForecastMemberSlot]()


def _clear(slot: _SlotP):
    slot[].fitted = False
    slot[].scalars = List[Float64]()
    slot[].series = List[Float32]()
    slot[].exog = List[Float32]()
    slot[].params = List[Float32]()
    slot[].x = List[Float32]()
    slot[].x0 = List[Float32]()
    slot[].stats = List[Float32]()
    slot[].flags = List[Int32]()
    slot[].ics = List[Float64]()


def _release_resident(slot: _SlotP) raises:
    """Release the resident upload (outside the store's lock)."""
    if slot[].resident != 0:
        var handle = slot[].resident
        slot[].resident = 0
        arima_resident_release_run(slot[].resident_route, handle)


def _mint(var words: List[Float64]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live ARIMA member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[ForecastMemberSlot](length=_CHUNK, fill=ForecastMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        _clear(slot)
        slot[].resident = 0
        slot[].words = words^
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a stale or foreign handle is
    refused, never read."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _SlotP(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].words = List[Float64]()
            _clear(slot)
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _dense(view: MemberMatrix, dtype: Int, what: String) raises:
    if not view.dense_host(dtype):
        var kind = String("float32") if dtype == MEMBER_F32 else (
            String("int32") if dtype == MEMBER_I32 else String("float64"))
        raise Error("TypeError: an ARIMA member reads " + what + " as a dense host " + kind + " view")


def _whole(call: _CallP) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: an ARIMA member reads whole host views (no row or column index)")


def _transpose(src: Int, dst: Int, rows: Int, cols: Int):
    """`dst` (cols x rows) = the transpose of `src` (rows x cols), float32,
    split over the destination's rows across the host workers. One row
    (or one column) is the same bytes: one copy."""
    if rows == 1 or cols == 1:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=dst), src=_F32P(unsafe_from_address=src), count=rows * cols)
        return
    var tasks = host_predict_task_count_work(cols, rows * cols)
    var chunk = host_predict_chunk(cols, tasks)
    var source = _F32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)

    def task(unit: Int) {imm source, imm target, imm chunk, imm rows, imm cols}:
        for j in range(unit * chunk, min((unit + 1) * chunk, cols)):  # native data loop (one host task's rows)
            for i in range(rows):  # native data loop (one destination row)
                target[j * rows + i] = source[i * cols + j]

    host_parallelize(task, tasks)


def _order(slot: _SlotP) -> List[Int]:
    """[p, d, q, P, D, Q, s]."""
    var w = slot[].words.copy()
    var out: List[Int] = [Int(w[1]), Int(w[2]), Int(w[3]), Int(w[4]), Int(w[5]), Int(w[6]), Int(w[7])]
    return out^


def _complexity(order: List[Int], k: Int, n_exog: Int) -> Int:
    return order[0] + order[2] + order[3] + order[5] + k + n_exog + 1


def _copy_f32(view: MemberMatrix) -> List[Float32]:
    var count = view.rows * view.cols
    var words = List[Float32](length=count, fill=0.0)
    if count:
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=view.data), count=count)
    return words^


def _fit(slot: _SlotP, call: _CallP) raises:
    _whole(call)
    if call[].fit_output != 0:
        member_unsupported("ARIMA", call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: an ARIMA member fits without sample weights")
    _dense(call[].y, MEMBER_F32, "the series y (observations x series)")
    var n_obs = call[].y.rows
    var batch = call[].y.cols
    var n_exog = 0
    if call[].x.rows > 0:
        _dense(call[].x, MEMBER_F32, "the regressors x (observations x n_exog)")
        if batch != 1:
            raise Error("NotImplementedError: an ARIMA member takes regressors for one series only")
        if call[].x.rows != n_obs:
            raise Error("ValueError: " + String(_A) + "dimensions mismatch, the regressors and y differ in rows")
        n_exog = call[].x.cols
    if not arima_route_fits(call[].route):
        raise Error("AttributeError: the ARIMA binding has no arima_fit")
    _release_resident(slot)
    _clear(slot)
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    var order = _order(slot)
    var k = Int(w[8])
    var big_n = _complexity(order, k, n_exog)
    var total = batch * big_n
    # The estimator's own fit keeps a private series-major copy (`_y`) and,
    # with regressors, a private copy of them (`_exog`).
    var series = List[Float32](length=batch * n_obs, fill=0.0)
    _transpose(call[].y.data, Int(series.unsafe_ptr()), n_obs, batch)
    var exog = _copy_f32(call[].x) if n_exog > 0 else List[Float32](length=1, fill=0.0)
    var params = List[Float32](length=total, fill=0.0)
    var x = List[Float32](length=total, fill=0.0)
    var x0 = List[Float32](length=total, fill=0.0)
    var stats = List[Float32](length=2 * batch, fill=0.0)
    var flags = List[Int32](length=2 * batch, fill=0)
    var ics = List[Float64](length=2 * batch, fill=0.0)
    var pens = _penalties(order, big_n, n_obs)
    var written = arima_fit_run(call[].route, Int(series.unsafe_ptr()), Int(exog.unsafe_ptr()),
        Int(params.unsafe_ptr()), Int(x.unsafe_ptr()), Int(x0.unsafe_ptr()), Int(stats.unsafe_ptr()),
        Int(flags.unsafe_ptr()), Int(ics.unsafe_ptr()), batch, n_obs, order, k, n_exog, Int(w[9]), Int(w[10]),
        pens[0], pens[1])
    if written != total:
        raise Error("RuntimeError: " + String(_A) + "the fit wrote " + String(written)
            + " parameters, batch_size * N is " + String(total))
    slot[].series = series^
    if n_exog > 0:
        slot[].exog = exog^
    slot[].params = params^
    slot[].x = x^
    slot[].x0 = x0^
    slot[].stats = stats^
    slot[].flags = flags^
    slot[].ics = ics^
    slot[].scalars = [Float64(_KIND_ARIMA), Float64(batch), Float64(n_obs), Float64(n_exog), Float64(big_n)]
    slot[].fitted = True


def _predict(slot: _SlotP, call: _CallP) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this ARIMA member is not fitted yet")
    _whole(call)
    _dense(call[].dst, MEMBER_F32, "the forecast buffer (horizon x series)")
    var batch = Int(slot[].scalars[1])
    var n_obs = Int(slot[].scalars[2])
    var n_exog = Int(slot[].scalars[3])
    if call[].dst.cols != batch:
        raise Error("ValueError: the forecast buffer holds one column per fitted series")
    var h = call[].dst.rows
    # An unread future pointer must still be a live address (the door's rule).
    var future = Int(slot[].series.unsafe_ptr())
    if n_exog > 0:
        if call[].x.rows == 0:
            raise Error("ValueError: " + String(_A) + "the model was fit with a regression component, so future "
                + "values must be provided via `exog`")
        _dense(call[].x, MEMBER_F32, "the future regressors x (horizon x n_exog)")
        if call[].x.rows != h or call[].x.cols != n_exog:
            raise Error("ValueError: " + String(_A) + "dimensions mismatch, `exog` has " + String(call[].x.cols)
                + " regressor column(s) and the model was fit with " + String(n_exog))
        future = call[].x.data
    elif call[].x.rows > 0:
        raise Error("ValueError: " + String(_A) + "a value was given for `exog` but the model was fit without any "
            + "regression component")
    var exog = Int(slot[].exog.unsafe_ptr()) if n_exog > 0 else Int(slot[].series.unsafe_ptr())
    # Series-major (batch x h); one series is the caller's layout already.
    var staged = List[Float32]()
    var target = call[].dst.data
    if batch > 1:
        staged = List[Float32](length=batch * h, fill=0.0)
        target = Int(staged.unsafe_ptr())
    var order = _order(slot)
    var k = Int(slot[].words[8])
    if arima_route_resident(call[].route):
        if slot[].resident == 0:
            var words = arima_resident_words(batch, n_obs, order, k, n_exog)
            slot[].resident = arima_resident_prepare_run(call[].route, Int(slot[].series.unsafe_ptr()), exog,
                Int(slot[].params.unsafe_ptr()), words)
            slot[].resident_route = call[].route
        _ = arima_resident_query_run(slot[].resident_route, slot[].resident, future, target, 0, 0, h, True)
    else:
        _ = arima_forecast_run(call[].route, Int(slot[].series.unsafe_ptr()), exog, future,
            Int(slot[].params.unsafe_ptr()), target, batch, n_obs, h, order, k, n_exog)
    if batch > 1:
        _transpose(target, call[].dst.data, batch, h)
    _ = staged^


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block, valid until this state's next fit,
    import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this ARIMA member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 8:
            raise Error("member ABI: an ARIMA member has nine fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else MEMBER_BLOCK_FAMILY + block - 1
    var batch = Int(slot[].scalars[1])
    var n_obs = Int(slot[].scalars[2])
    var n_exog = Int(slot[].scalars[3])
    var big_n = Int(slot[].scalars[4])
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == _BLOCK_SERIES:
        call[].dst = MemberMatrix(Int(slot[].series.unsafe_ptr()), batch, n_obs)
    elif block == _BLOCK_EXOG:
        if n_exog > 0:
            call[].dst = MemberMatrix(Int(slot[].exog.unsafe_ptr()), batch * n_obs, n_exog)
        else:
            call[].dst = MemberMatrix()
    elif block == _BLOCK_PARAMS:
        call[].dst = MemberMatrix(Int(slot[].params.unsafe_ptr()), batch, big_n)
    elif block == _BLOCK_X:
        call[].dst = MemberMatrix(Int(slot[].x.unsafe_ptr()), batch, len(slot[].x) // max(batch, 1))
    elif block == _BLOCK_X0:
        call[].dst = MemberMatrix(Int(slot[].x0.unsafe_ptr()), batch, len(slot[].x0) // max(batch, 1))
    elif block == _BLOCK_STATS:
        call[].dst = MemberMatrix(Int(slot[].stats.unsafe_ptr()), 1, len(slot[].stats))
    elif block == _BLOCK_FLAGS:
        call[].dst = MemberMatrix(Int(slot[].flags.unsafe_ptr()), 1, len(slot[].flags), MEMBER_I32)
    elif block == _BLOCK_ICS:
        call[].dst = MemberMatrix(Int(slot[].ics.unsafe_ptr()), 1, len(slot[].ics), MEMBER_F64)
    else:
        raise Error("member ABI: an ARIMA member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 9


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var block = call[].block
    if block == MEMBER_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "a fitted block")
        if call[].x.rows * call[].x.cols != _SCALAR_WORDS:
            raise Error("member ABI: the ARIMA scalars block is five host float64 words")
        var words = List[Float64](length=_SCALAR_WORDS, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=_SCALAR_WORDS)
        slot[].scalars = words^
    elif block == _BLOCK_FLAGS:
        _dense(call[].x, MEMBER_I32, "a fitted block")
        var count = call[].x.rows * call[].x.cols
        var words = List[Int32](length=count, fill=0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=MutPointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=call[].x.data), count=count)
        slot[].flags = words^
    elif block == _BLOCK_ICS:
        _dense(call[].x, MEMBER_F64, "a fitted block")
        var count = call[].x.rows * call[].x.cols
        var words = List[Float64](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=count)
        slot[].ics = words^
    elif block >= _BLOCK_SERIES and block <= _BLOCK_STATS:
        _dense(call[].x, MEMBER_F32, "a fitted block")
        var words = _copy_f32(call[].x)
        if block == _BLOCK_SERIES:
            slot[].series = words^
        elif block == _BLOCK_EXOG:
            slot[].exog = words^
        elif block == _BLOCK_PARAMS:
            slot[].params = words^
        elif block == _BLOCK_X:
            slot[].x = words^
        elif block == _BLOCK_X0:
            slot[].x0 = words^
        else:
            slot[].stats = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an ARIMA member has no fitted block " + String(block))
    slot[].fitted = False
    _release_resident(slot)
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _SCALAR_WORDS:
        raise Error("member ABI: a fitted ARIMA state needs its scalars block")
    var batch = Int(slot[].scalars[1])
    var n_obs = Int(slot[].scalars[2])
    var n_exog = Int(slot[].scalars[3])
    var big_n = Int(slot[].scalars[4])
    var k = Int(slot[].words[8])
    if batch <= 0 or n_obs <= 0 or n_exog < 0 or big_n != _complexity(_order(slot), k, n_exog):
        raise Error("ValueError: ARIMA fitted dimension metadata is invalid")
    if len(slot[].series) != batch * n_obs or len(slot[].params) != batch * big_n:
        raise Error("ValueError: ARIMA fitted buffer dimensions disagree with model metadata")
    if n_exog > 0 and len(slot[].exog) != batch * n_obs * n_exog:
        raise Error("ValueError: ARIMA fitted exog dimensions disagree")
    slot[].fitted = True


def _capabilities() -> Int:
    return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_MULTI_OUTPUT
        | MEMBER_CAN_FORECAST)


def _new(call: _CallP) raises:
    if call[].param_count != _WORDS or not call[].params:
        raise Error("member ABI: an ARIMA member takes eleven parameters")
    var w = _F64P(unsafe_from_address=call[].params)
    if w[0] != Float64(_KIND_ARIMA):
        raise Error("member ABI: unknown ARIMA member kind")
    for i in range(1, 8):  # driver-loop: over the seven order words
        if w[i] < 0.0:
            raise Error("member ABI: an ARIMA member's order words are non-negative")
    if (w[8] != 0.0 and w[8] != 1.0) or w[9] < 0.0 or w[10] < 0.0:
        raise Error("member ABI: an ARIMA member's k, method or maxiter word is out of range")
    var words = List[Float64](length=_WORDS, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=w, count=_WORDS)
    call[].state = _mint(words^)


def _forecast_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state != 0:
            _ = _lookup(call[].state, False)
        return _capabilities()
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_OUTPUTS and call[].state == 0:
        # No state: one forecast column per series of the fit to come.
        call[].classes = 0
        call[].out_cols = max(1, call[].y.cols)
        return 0
    if operation == MEMBER_OP_FREE:
        var dying = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        _release_resident(dying)
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_OUTPUTS:
        call[].classes = 0
        call[].out_cols = Int(slot[].scalars[1]) if slot[].fitted else max(1, call[].y.cols)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported("ARIMA", operation)
    return 0


def forecast_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of ARIMA (schema `MEMBER_ABI`)."""
    try:
        var word = _forecast_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def forecast_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's ARIMA member entry. One line in each
    `_mojolearn_arima` image's module initialization, after its ARIMA
    operations table. A build with no vendor word (`vendor` < 0) publishes
    nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_FORECAST](mode, vendor, publisher, forecast_member_v2)
