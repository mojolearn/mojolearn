# SPDX-License-Identifier: Apache-2.0
"""The sequence-image forecasters (the theta models) as MEMBERS
(core/member_abi.mojo), family 43 `MEMBER_FAMILY_FORECAST_SEQ`.

One entry, `forecast_seq_member_v2`, published by each `_mojolearn_x_sequence`
image (`forecast_seq_member_publish`, beside the recurrent member). It runs
the family's existing typed RUN (sequence/xseq_forecast_run.mojo: the theta
checks, `forecast_run_theta`, `forecast_run_theta_forecast`) over the glue
entry of the `DeviceRoute` it is handed, the same entry the estimator's own
door (bindings/xseq_forecast_native_door.mojo `_theta_fit` /
`_theta_predict`) calls. No kernel is touched.

THE FORECAST FORM (MEMBER_CAN_FORECAST): FIT reads the series in `y`, host
float32, observations x series (one column per series); `x` must not be
given (exogenous regressors are not implemented, as the estimator's own
fit refuses them). PREDICT with `x` not given writes `dst` = horizon x
series float32: `dst.rows` is the horizon h.

KINDS: 0 theta (AutoTheta and its four fixed-model classes).

ROUTE SLOTS (filled by providers/python/member_forecast_seq_boundary_door.mojo):
  * `entry`: the glue entry of the route's sequence image (`xseq_route`'s
    `route.entry`, the door's `route.glue`).

MEMBER_OP_NEW parameters (Float64 words):
  theta, 8 words: [0] kind 0; [1] season_length; [2] model code (-1 auto
  selection, 0 STM, 1 OTM, 2 DSTM, 3 DOTM: the estimator's `model` mapped by
  the connected code table); [3] decomposition (1 additive, 0
  multiplicative); [4] fixed-setting mask (1 initial level, 2 alpha, 4
  theta); [5] initial_smoothed; [6] alpha; [7] theta (0 when free).

FITTED-STATE BLOCKS, in index order:
  0 `MEMBER_BLOCK_SCALARS` float64 [kind, series, observations, season
    letter (theta: 0)];
  1 `MEMBER_BLOCK_FAMILY` float32 series x observations: the series,
    series-major (the estimator's `_y`).
An import needs both.

WHEN THE MODEL IS ESTIMATED: as in the estimator, the fit stores the series
(one series-major copy: the estimator's own fit copies too) and checks its
length; the first PREDICT runs the whole theta fit and forecast at its
horizon and keeps the fit's info rows; a later PREDICT at another horizon
runs the fit's forecast-only tail from those rows (no Nelder-Mead; the fit
does not read h, so its words are the whole call's at that horizon), and a
PREDICT at the stored horizon publishes the stored forecast, as the
estimator's stored-forecast reuse does.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted; the store's lock covers only mint, free and
the handle lookup. The series copy and the forecast publication split their
series (or rows) across the host workers; the theta runs are the image's
own glue operations (one task per series inside the glue).
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_MULTI_OUTPUT,
    MEMBER_CAN_FORECAST,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_FORECAST_SEQ
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from sequence.xseq_forecast_run import (
    forecast_run_theta_check_n, forecast_run_theta, forecast_run_theta_forecast,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _KIND_THETA = 0
comptime _THETA_WORDS = 8
comptime _SCALAR_WORDS = 4
comptime _THETA_INFO = 8
comptime _BLOCK_SERIES = MEMBER_BLOCK_FAMILY
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


struct ForecastSeqMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    var fitted: Bool
    var words: List[Float64]
    # The series, series-major (batch x n), and its two counts.
    var series: List[Float32]
    var batch: Int
    var n: Int
    # The season letter a fit keeps (ETS; theta 0).
    var letter: Int
    # The fit's info rows (batch x info words), once a predict ran the fit.
    var info: List[Float32]
    var info_ready: Bool
    # The stored forecast (batch x forecast_h), -1: none.
    var forecast: List[Float32]
    var forecast_h: Int
    # The scalars block: kind, batch, n, letter.
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = 0
        self.fitted = False
        self.words = List[Float64]()
        self.series = List[Float32]()
        self.batch = 0
        self.n = 0
        self.letter = 0
        self.info = List[Float32]()
        self.info_ready = False
        self.forecast = List[Float32]()
        self.forecast_h = -1
        self.scalars = List[Float64]()


struct ForecastSeqMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[ForecastSeqMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[ForecastSeqMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[ForecastSeqMemberStore, "MojolearnForecastSeqMemberStoreV2", 2]

comptime _SlotP = MutPointer[ForecastSeqMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[ForecastSeqMemberSlot]()


def _clear(slot: _SlotP):
    slot[].fitted = False
    slot[].series = List[Float32]()
    slot[].batch = 0
    slot[].n = 0
    slot[].letter = 0
    slot[].info = List[Float32]()
    slot[].info_ready = False
    slot[].forecast = List[Float32]()
    slot[].forecast_h = -1
    slot[].scalars = List[Float64]()


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
                raise Error("RuntimeError: mojolearn: too many live forecaster member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[ForecastSeqMemberSlot](length=_CHUNK, fill=ForecastSeqMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].kind = Int(words[0])
        _clear(slot)
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


def _name(slot: _SlotP) -> String:
    return String("Theta")


def _dense(view: MemberMatrix, dtype: Int, what: String, estimator: String) raises:
    if not view.dense_host(dtype):
        var kind = String("float32") if dtype == MEMBER_F32 else String("float64")
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host " + kind + " view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _no_exog(call: _CallP, estimator: String) raises:
    if call[].x.rows > 0:
        raise Error("NotImplementedError: " + estimator + ": exogenous regressors are not implemented")


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


def _fit(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits without sample weights")
    _no_exog(call, estimator)
    _dense(call[].y, MEMBER_F32, "the series y (observations x series)", estimator)
    var n = call[].y.rows
    var batch = call[].y.cols
    _clear(slot)
    forecast_run_theta_check_n(n)
    # The estimator's own fit keeps a private series-major copy (`_y`).
    var series = List[Float32](length=batch * n, fill=0.0)
    _transpose(call[].y.data, Int(series.unsafe_ptr()), n, batch)
    slot[].series = series^
    slot[].batch = batch
    slot[].n = n
    slot[].scalars = [Float64(slot[].kind), Float64(batch), Float64(n), Float64(slot[].letter)]
    slot[].fitted = True


def _theta_run(slot: _SlotP, call: _CallP, h: Int) raises:
    """The stored forecast at horizon h: the whole fit and forecast at the
    first predict, the forecast-only tail from the fit's info rows after."""
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    var batch = slot[].batch
    var n = slot[].n
    var season = Int(w[1])
    var forecast = List[Float32](length=batch * h, fill=0.0)
    if slot[].info_ready:
        forecast_run_theta_forecast(call[].route.entry, Int(slot[].series.unsafe_ptr()),
            Int(slot[].info.unsafe_ptr()), Int(forecast.unsafe_ptr()), batch, n, h, season)
    else:
        var info = List[Float32](length=batch * _THETA_INFO, fill=0.0)
        var settings: List[Float64] = [w[5], w[6], w[7]]
        forecast_run_theta(call[].route.entry, Int(slot[].series.unsafe_ptr()), Int(forecast.unsafe_ptr()),
            Int(info.unsafe_ptr()), batch, n, h, season, Int(w[2]), w[3] != 0.0, Int(w[4]), settings)
        slot[].info = info^
        slot[].info_ready = True
    slot[].forecast = forecast^
    slot[].forecast_h = h


def _predict(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _no_exog(call, estimator)
    _dense(call[].dst, MEMBER_F32, "the forecast buffer (horizon x series)", estimator)
    if call[].dst.cols != slot[].batch:
        raise Error("ValueError: the forecast buffer holds one column per fitted series")
    var h = call[].dst.rows
    if slot[].forecast_h != h:
        _theta_run(slot, call, h)
    # batch x h -> h x batch (the caller's layout).
    _transpose(Int(slot[].forecast.unsafe_ptr()), call[].dst.data, slot[].batch, h)


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block, valid until this state's next fit,
    import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 1:
            raise Error("member ABI: a forecaster member has two fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else _BLOCK_SERIES
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == _BLOCK_SERIES:
        call[].dst = MemberMatrix(Int(slot[].series.unsafe_ptr()), slot[].batch, slot[].n)
    else:
        raise Error("member ABI: a forecaster member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 2


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    if block == _BLOCK_SERIES:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        var scalars = slot[].scalars.copy()
        _clear(slot)
        slot[].scalars = scalars^
        slot[].series = words^
        slot[].batch = call[].x.rows
        slot[].n = call[].x.cols
    elif block == MEMBER_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "a fitted block", estimator)
        if call[].x.rows * call[].x.cols != _SCALAR_WORDS:
            raise Error("member ABI: the forecaster scalars block is four host float64 words")
        var words = List[Float64](length=_SCALAR_WORDS, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=_SCALAR_WORDS)
        slot[].fitted = False
        slot[].scalars = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a forecaster member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _SCALAR_WORDS or len(slot[].series) == 0:
        raise Error("member ABI: a fitted forecaster state needs its scalars and series blocks")
    if Int(slot[].scalars[0]) != slot[].kind:
        raise Error("member ABI: the fitted forecaster state is of another kind")
    if Int(slot[].scalars[1]) != slot[].batch or Int(slot[].scalars[2]) != slot[].n:
        raise Error("member ABI: the forecaster scalars do not match the series block")
    slot[].letter = Int(slot[].scalars[3])
    slot[].info_ready = False
    slot[].forecast_h = -1
    slot[].fitted = True


def _outputs(slot: _SlotP, call: _CallP):
    call[].classes = 0
    call[].out_cols = slot[].batch if slot[].fitted else max(1, call[].y.cols)


def _capabilities() -> Int:
    return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_MULTI_OUTPUT
        | MEMBER_CAN_FORECAST)


def _new(call: _CallP) raises:
    if not call[].params or call[].param_count < 1:
        raise Error("member ABI: a forecaster member takes its kind word first")
    var w = _F64P(unsafe_from_address=call[].params)
    var count: Int
    if w[0] == Float64(_KIND_THETA):
        count = _THETA_WORDS
        if call[].param_count != count:
            raise Error("member ABI: a theta member takes eight parameters")
        if w[1] < 1.0 or w[2] < -1.0 or w[2] > 3.0 or (w[3] != 0.0 and w[3] != 1.0) or w[4] < 0.0 or w[4] > 7.0:
            raise Error("member ABI: a theta member's season, model, decomposition or mask word is out of range")
    else:
        raise Error("member ABI: unknown forecaster member kind")
    var words = List[Float64](length=count, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=w, count=count)
    call[].state = _mint(words^)


def _forecast_seq_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state != 0:
            _ = _lookup(call[].state, False)  # every kind answers the same bits
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
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_OUTPUTS:
        _outputs(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def forecast_seq_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the theta forecasters (schema `MEMBER_ABI`)."""
    try:
        var word = _forecast_seq_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def forecast_seq_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's forecaster member entry. One line in each
    `_mojolearn_x_sequence` image's module initialization, after its glue
    entry. A build with no vendor word (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_FORECAST_SEQ](mode, vendor, publisher, forecast_seq_member_v2)
