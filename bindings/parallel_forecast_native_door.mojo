# SPDX-License-Identifier: Apache-2.0
"""The native owner of `mojolearn.parallel_forecasting`
(python/mojolearn/parallel_forecasting.py): `predict_arima`,
`forecast_arima`, `predict_exponential_smoothing` and
`forecast_exponential_smoothing` as one call each into
`parallel_forecast_call`, on the native local process pool
(bindings/native_process_pool_door.mojo).

The partition is the Python driver's: whole series in `series_per_shard`
ranges; each shard's series-major state (ARIMA's `_y` and `params_` rows,
the Holt-Winters packed components' columns) is copied into a plain state
record, the worker rebuilds the shard's model natively and runs the
native `predict` (bindings/tsa_arima_native.mojo, tsa_hw_native.mojo), and
each shard's block is placed at its own offset in range order. The same
compiled entries on the same words, so no bit moves.

UNPACK and PACK only: the span, step, index and mode refusals, the shard
bounds and the block copies are typed, in
bindings/parallel_forecast_native_run.mojo and
bindings/parallel_shard_common_run.mojo. A worker builds the member once
per generation and answers a predict request from its typed words through
the family's typed-span queries (`arima_predict_span`, `hw_predict_span`);
the fits call the family's `arima_fit` / `hw_fit` on a rebuilt member until
that family publishes an entry table; the frames are the codec's.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, native_interpreter_finalizing
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_fields, py_contains
from std.ffi import external_call
from bindings.native_prepared_owner_unpack import prepared_key_fields, prepared_key_same, prepared_get, prepared_put
from bindings.native_loaded_module_door import NativeModules
from bindings.msel_member_device_unpack import member_says_cpu
from bindings.py_attr import py_attr
from bindings.classical_arrays_door import classical_address
from bindings.native_process_pool_door import (
    process_pool_create, process_pool_close, process_pool_closed, process_pool_pids,
    process_worker_connect, process_worker_next, process_worker_value, process_worker_reply, process_worker_refuse,
    WorkerTurn,
)
from bindings.native_transport_door import transport_close
from bindings.tsa_hw_native_door import (
    tsa_array, tsa_array_unfilled, hw_predict_span, hw_fit, _check_dims, _set_fitted, _commit,
)
from bindings.tsa_admission_door import tsa_hw_fields
from bindings.tsa_arima_native_door import arima_predict_span, arima_fit, _series_major
from bindings.parallel_shard_common_door import (
    shard_ranges,
    shard_devices,
    shard_run,
    shard_map,
    _shard_acquire,
    _shard_keep,
    shard_rows,
    shard_concat,
    shard_state,
    shard_model,
    shard_array,
    shard_identity,
    ShardWave,
    shard_wave,
    SHARD_IDLE,
)
from bindings.parallel_shard_common_run import (
    shard_read_shift, shard_width_admit, shard_bounds, shard_indices_admit, shard_block_copy, SHARD_DEVICE_NOT_AN_INDEX,
    shard_request_decode, shard_batch_admit,
    PARALLEL_SHARD_WORKER_OPN_IDENTITY,
    PARALLEL_FORECAST_WORKER_OPN_GENERATION_STORE,
    PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD,
    PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD,
    PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD,
    PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD,
)
from bindings.parallel_forecast_native_run import (
    forecast_arima_mode_admit, forecast_process_mode_admit, forecast_exog_admit, forecast_span_admit,
    forecast_positive_admit, forecast_out_of_sample_admit, forecast_index_admit, forecast_batch_admit,
    forecast_length_admit, forecast_envelope_admit, forecast_rows_copy, forecast_rows_place, forecast_words_place,
)
from core.parallel_forecast_impl import forecast_gather_cols, forecast_scatter_cols
from core.shared_slot import SharedSlot
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_FORECAST_OPN_FIT_ARIMA,
    PARALLEL_FORECAST_OPN_FIT_EXPONENTIAL_SMOOTHING,
    PARALLEL_FORECAST_OPN_FORECAST_ARIMA,
    PARALLEL_FORECAST_OPN_FORECAST_EXPONENTIAL_SMOOTHING,
    PARALLEL_FORECAST_OPN_INITIALIZE,
    PARALLEL_FORECAST_OPN_PREDICT_ARIMA,
    PARALLEL_FORECAST_OPN_PREDICT_EXPONENTIAL_SMOOTHING,
    PARALLEL_FORECAST_OPN_WORKER,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item

comptime _CHILD = "mojolearn._native_parallel_forecast_worker"


struct ForecastGeneration(Movable):
    """One fitted estimator's predict state as the forecast workers hold it
    (a "generation"), for one kind (ARIMA or exponential smoothing).

    In the coordinating process: `token` names the generation the workers of
    the pool with process ids `pids` hold ("" for none) and `width` is its
    series-per-shard word. Nothing of the estimator is held here: the
    generation's key (the fitted objects it was cut from, compared by
    identity) is the estimator's own prepared record (`_GENERATION_SLOT`),
    which the registry drops when the estimator is freed; the record's
    `ForecastGenerationLifetime` then asks the workers to drop their models.
    `orphaned` says that request is still owed (the pool was in use when the
    estimator was freed); the next call of this door sends it.

    A generation's name on the wire is two words, never a string: `owner`
    (the coordinating process id) and `issued` (its store counter there).

    In a worker process: `owner`, `issued` and `models` (a list, one model
    per shard this worker answers, built once at the store; None for none)
    are the generation it holds; `token`, `pids`, `width` and `orphaned` are
    unused there."""
    var token: String
    var owner: Int
    var issued: Int
    var width: Int
    var pids: List[Int]
    var orphaned: Bool
    var models: PythonObject

    def __init__(out self):
        self.token = String()
        self.owner = 0
        self.issued = 0
        self.width = 0
        self.pids = List[Int]()
        self.orphaned = False
        self.models = PythonObject(None)


struct ForecastGenerationLifetime(Movable, Writable):
    """Held by an estimator's generation record (`_GENERATION_SLOT`), so it
    is freed with the estimator, at a refit's or another shard width's new
    store, or when the record is replaced. Freed while its generation is
    still the one the workers hold, it asks them to drop their models
    (`_generation_released`); a replaced generation needs nothing (the new
    store replaced the workers' models). A finalizer is its own operation
    and stays quiet: a closed or lost pool has no worker state left."""
    var kind: Int
    var token: String

    def __init__(out self, kind: Int, var token: String):
        self.kind = kind
        self.token = token^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_ForecastGenerationLifetime(kind=", self.kind, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        try:
            # At interpreter shutdown the workers end with their sockets;
            # nothing is asked (no loaded-module read is safe then).
            if not native_interpreter_finalizing():
                _generation_released(self.kind, self.token)
        except:
            pass


struct ParallelForecastBoundary(Defaultable, Movable):
    var arima_cls: PythonObject
    var hw_cls: PythonObject
    var arima_context: PythonObject
    var hw_context: PythonObject
    var backend: PythonObject
    var arima_generation: ForecastGeneration
    var hw_generation: ForecastGeneration

    def __init__(out self):
        self.arima_cls = PythonObject(None)
        self.hw_cls = PythonObject(None)
        self.arima_context = PythonObject(None)
        self.hw_context = PythonObject(None)
        self.backend = PythonObject(None)
        self.arima_generation = ForecastGeneration()
        self.hw_generation = ForecastGeneration()


# V2 (lane pools-frame, 2026-11-21): the two predict generations. V3 (same
# lane, round 7): a generation pins nothing of the estimator (its key is the
# estimator's prepared record) and carries `orphaned`.
comptime PARALLEL_FORECAST_BOUNDARY = SharedSlot[ParallelForecastBoundary, "MojolearnParallelForecastOwnerV3"]

#: The prepared-registry slot of an estimator's generation record,
#: `(token, key, lifetime)`: `key` is the flat list of the objects the
#: generation was cut from (None when a fitted array could not be sealed: no
#: claim, the next predict stores again), `lifetime` a
#: `ForecastGenerationLifetime`. The registry holds the estimator weakly and
#: drops the record when it is freed; the record never holds the estimator.
comptime _GENERATION_SLOT = "parallel_forecast_generation_v1"

#: The generation kinds a store request names.
comptime _GENERATION_ARIMA = 0
comptime _GENERATION_HW = 1


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _arima_mode(mode: PythonObject) raises:
    """UNPACK of the model's own mode word for `forecast_arima_mode_admit`.
    Once per driver call, no loop: one type test and one String read."""
    var exact = mode is not None and py_type(mode) is _b().str
    forecast_arima_mode_admit(mode is not None, exact, String(py=mode) if exact else String(""))


def _width(width: PythonObject) raises -> Int:
    var exact = py_type(width) is _b().int
    shard_width_admit(exact, Int(py=width) if exact else 0, "series_per_shard")
    return Int(py=width)


def _ranges(count: Int, width: PythonObject) raises -> List[Int]:
    """`_ranges(count, width)`: the shard bounds [lo0, hi0, lo1, hi1, ...]."""
    return shard_bounds(count, _width(width), "at least one series is required")


def _member_host(estimator: PythonObject) raises -> Bool:
    """UNPACK, once per driver call: True when the model says device='cpu'
    in a process with a GPU set. Its shards then run on a host pool
    (`_shard_acquire(..., host=True)`: workers on the CPU-only set, the
    family's CPU fit and predict); `devices` are logical worker indices
    there. The parent merges copy bytes only. (vdv-gaps, 2026-11-24: the
    model was refused, so the CPU side could not run in a GPU process; the
    shared read is bindings/msel_member_device_unpack.mojo.)
    `_backend._CPU_ONLY` (None in a process with a GPU set) is read from the
    connected backend namespace only for a model that says 'cpu'; on a
    CPU-only install the pool is already the host one."""
    if not member_says_cpu(py_fields(estimator), _b()):
        return False
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    return py_get(f[].backend, PythonObject("_CPU_ONLY")) is None


def _devices(devices: PythonObject) raises:
    """`DevicePool`'s admission of the device list, in its words."""
    var b = _b()
    var selected = py_call(b.tuple, devices)
    var indices = List[Int](capacity=len(selected))
    for i in range(len(selected)):  # driver-loop: over the admitted device indices
        indices.append(Int(py=selected[i]) if py_type(selected[i]) is b.int else SHARD_DEVICE_NOT_AN_INDEX)
    shard_indices_admit(indices)


def _generation_same(generation: ForecastGeneration, token: String, width: Int, pool: PythonObject) raises -> Bool:
    """Whether the workers of `pool` hold the generation `token` (the token
    of the estimator's own record, whose key the caller compared): the same
    worker processes and the same shard width."""
    if generation.token.byte_length() == 0 or generation.token != token or generation.width != width or process_pool_closed(pool):
        return False
    var pids = process_pool_pids(pool)
    if len(pids) == 0 or len(pids) != len(generation.pids):
        return False
    for i in range(len(pids)):  # small-loop(the pool's worker processes)
        if pids[i] != generation.pids[i]:
            return False
    return True


def _generation_frozen(value: PythonObject, context: PythonObject) raises -> Bool:
    """A sealed fitted array is still read-only: one type compare and one
    slot read, once per array per public predict. An Array whose flag was
    turned off since the seal is not trusted: the caller seals a new copy of
    its present words and stores a new generation."""
    var array_class = py_get(context, parallel_keys()[].k_array_class)
    return array_class is not None and py_type(value) is array_class and Bool(py=value._readonly)


def _generation_seal(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """A private read-only copy of fitted Array `value` (same shape, float32,
    C order), or None when `value` is not an exact Array that owns C-ordered
    float32 storage (a caller-assigned ndarray, a view): then no generation
    is claimed. The copy is born read-only, so no writable NumPy view of it
    can exist: `est.params_[...] = v` and `np.asarray(est.params_)[...] = v`
    raise the read-only error. The caller that puts the copy in the
    estimator's field marks `value` read-only too (it is no longer the
    estimator's field: a write through an older reference raises instead of
    being silently ignored). One copy of the array's bytes per sealed
    estimator, never per predict."""
    var array_class = py_get(context, parallel_keys()[].k_array_class)
    if array_class is None or py_type(value) is not array_class:
        return PythonObject(None)
    if value._base is not None or String(py=value.dtype) != "<f4" or String(py=value.order) != "C":
        return PythonObject(None)
    var count = Int(py=value.size)
    var sealed = tsa_array_unfilled("f", "<f4", value.shape, count, 4, context)
    if count > 0:
        shard_block_copy(Int(py=classical_address(sealed)), Int(py=classical_address(value)), count * 4)
    _ = value  # keep the Python object alive through the native call above
    py_setattr(sealed, "_readonly", PythonObject(True))
    return sealed


def _generation_ours(record: PythonObject, key: PythonObject) raises -> Bool:
    """The estimator's record was cut from exactly these objects (identity
    only; nothing is hashed or compared by value)."""
    return record is not None and record[1] is not None and prepared_key_same(record[1], key)


def _generation_claim(estimator: PythonObject, kind: Int, token: String, key: PythonObject) raises -> PythonObject:
    """Record the stored generation on the estimator (replacing its older
    record, whose lifetime sees another token and asks nothing). Returns the
    lifetime: the caller keeps it past `_shard_keep`, so for an estimator
    that takes no weak reference (nothing recorded) it ends after the call,
    with the pool idle, and the workers drop the models then."""
    var lifetime = PythonObject(alloc=ForecastGenerationLifetime(kind, token.copy()))
    var mods = NativeModules()
    prepared_put(mods, _GENERATION_SLOT, estimator, Python.tuple(PythonObject(token), key, lifetime))
    return lifetime


def _generation_released(kind: Int, token: String) raises:
    """An estimator's generation record ended. When it is still the
    generation the workers hold, the coordinator's claim is cleared and the
    workers are asked to drop the models."""
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    if kind == _GENERATION_ARIMA:
        if token.byte_length() == 0 or f[].arima_generation.token != token:
            return
        f[].arima_generation.token = String()
        f[].arima_generation.orphaned = True
    else:
        if token.byte_length() == 0 or f[].hw_generation.token != token:
            return
        f[].hw_generation.token = String()
        f[].hw_generation.orphaned = True
    _generation_sweep()


def _generation_idle(pids: List[Int]) raises -> Int:
    """The idle forecast pool of this process whose workers are `pids`, as
    its index in the idle list, or -1 (in use by a running call, closed or
    never kept)."""
    var pid = Int(external_call["getpid", Int32]())
    var idle = SHARD_IDLE.get_or_create_ptr()
    for i in range(len(idle[].pools)):  # small-loop(the idle pools of this process: a few)
        if idle[].pools[i].pid != pid or idle[].pools[i].child != _CHILD:
            continue
        if process_pool_closed(idle[].pools[i].pool):
            continue
        var held = process_pool_pids(idle[].pools[i].pool)
        if len(held) != len(pids) or len(held) == 0:
            continue
        var same = True
        for j in range(len(held)):  # small-loop(the pool's worker processes)
            if held[j] != pids[j]:
                same = False
                break
        if same:
            return i
    return -1


def _generation_drop(kind: Int, pids: List[Int]) raises -> Bool:
    """Ask the workers `pids` to drop their generation of `kind`: one store
    of no states under the empty token per worker (the store's own
    replacement). True when the request is settled: sent, or the pool is
    gone (its workers ended with it). False when the pool is in use by a
    running call: the request stays owed. The pool leaves the idle list for
    the wave, so no other call takes it meanwhile; a failed wave closes it."""
    if len(pids) == 0:
        return True
    var at = _generation_idle(pids)
    if at < 0:
        # In use by a running call of this door (it is kept again at that
        # call's end), or closed: a closed pool is not seen again under
        # these process ids, and the next store clears `orphaned`.
        return False
    var idle = SHARD_IDLE.get_or_create_ptr()
    var taken = idle[].pools.pop(at)
    var pool = taken.pool
    try:
        var b = _b()
        var drops = shard_wave()
        for j in range(len(pids)):  # driver-loop: one drop per worker
            # owner 0, issued 0: no generation has that name
            var named: List[Int] = [kind, 0, 0]
            drops.add_operands(PARALLEL_FORECAST_WORKER_OPN_GENERATION_STORE, named^, b.list())
        var dropped = shard_map(pool, drops)
        shard_batch_admit(len(dropped), len(pids))
    except:
        try:
            process_pool_close(pool)
        except:
            pass
        return True
    _shard_keep(taken^)
    return True


def _generation_sweep() raises:
    """Send the drops still owed (`orphaned`). Two flag reads when none is."""
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    if f[].arima_generation.orphaned:
        var pids = f[].arima_generation.pids.copy()
        var settled = _generation_drop(_GENERATION_ARIMA, pids)
        # The slot is read again: a wave releases the GIL.
        f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
        if settled and f[].arima_generation.token.byte_length() == 0:
            f[].arima_generation.orphaned = False
    if f[].hw_generation.orphaned:
        var pids = f[].hw_generation.pids.copy()
        var settled = _generation_drop(_GENERATION_HW, pids)
        f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
        if settled and f[].hw_generation.token.byte_length() == 0:
            f[].hw_generation.orphaned = False


def _generation_token(kind: Int, issued: Int) -> String:
    """The coordinator's own name of a generation (the estimator record's
    word; it never travels)."""
    return String("fg-") + String(kind) + "-" + String(Int(external_call["getpid", Int32]())) + "-" + String(issued)


def _generation_store(mut generation: ForecastGeneration, kind: Int, width: Int,
                      states: PythonObject, pool: PythonObject, workers: Int) raises -> Int:
    """Store a new generation on the workers of `pool`: worker j receives,
    once, the states of the shards it answers (r = j mod workers, the order
    of `shard_map`'s waves) and builds their models. The claim is recorded
    only after every worker stored; a failed store leaves none. Workers
    beyond the shard count get a store of no states, so none keeps an older
    generation of this kind. Returns the generation's `issued` word."""
    var b = _b()
    generation.token = String()
    generation.issued += 1
    var issued = generation.issued
    var owner = Int(external_call["getpid", Int32]())
    generation.owner = owner
    var shards = len(states)
    if workers > 0:
        var stores = shard_wave()
        for j in range(workers):  # driver-loop: one generation store per worker
            var mine = b.list()
            var r = j
            while r < shards:  # driver-loop: the shards worker j answers
                _ = py_call(mine.append, states[r])
                r += workers
            # The generation's name rides in the request words; the payload
            # is the list of states alone.
            var named: List[Int] = [kind, owner, issued]
            stores.add_operands(PARALLEL_FORECAST_WORKER_OPN_GENERATION_STORE, named^, mine)
        var stored = shard_map(pool, stores)
        shard_batch_admit(len(stored), workers)
    generation.width = width
    generation.pids = process_pool_pids(pool)
    generation.orphaned = False
    generation.token = _generation_token(kind, issued)
    return issued


def _generation_predict(op: Int, issued: Int, shards: Int, workers: Int, start: Int, end: Int,
                        pool: PythonObject) raises -> PythonObject:
    """One predict request per shard, typed whole: the request words name
    the generation (`owner`, `issued`), the worker's held model (shard r is
    slot r // workers of worker r mod workers) and the span. No payload
    record travels (it was the tuple `((token, slot), (start, end))` built
    per shard, UNPACKed, sent, PACKed by the worker and read back)."""
    var requests = shard_wave()
    var owner = Int(external_call["getpid", Int32]())
    for r in range(shards):  # driver-loop: one worker request per device shard
        var operands: List[Int] = [owner, issued, r // workers, start, end]
        requests.add_operands(op, operands^, PythonObject(None))
    var parts = shard_map(pool, requests)
    shard_batch_admit(len(parts), shards)
    return parts


def _generation_model(generation: ForecastGeneration, operands: List[Int]) raises -> PythonObject:
    """A worker's held model for a predict request's typed operands
    `[owner, issued, slot, start, end]`: three word compares and one list
    read (it was a tuple test, a String built from the token and compared,
    and an integer conversion per request)."""
    if len(operands) != 5:
        raise Error("parallel forecasting: a predict request names a held model and a span")
    if generation.models is None or operands[0] != generation.owner or operands[1] != generation.issued:
        raise Error("parallel forecasting: this worker does not hold the requested fitted state")
    var slot = operands[2]
    if slot < 0 or slot >= len(generation.models):
        raise Error("parallel forecasting: this worker holds no model for the requested shard")
    return generation.models[slot]


def _rows(source: PythonObject, lo: Int, hi: Int, width: Int, context: PythonObject) raises -> PythonObject:
    """`source[lo:hi]` of a C-ordered (rows, width) float32 Array, a copy."""
    var out = tsa_array("f", "<f4", Python.tuple(hi - lo, width), (hi - lo) * width, 4, context)
    if (hi - lo) * width > 0:
        forecast_rows_copy(Int(py=classical_address(out)), Int(py=classical_address(source)), lo, hi, width)
        _ = out  # keep the Python object alive through the native call above
        _ = source
    return out


# ---------------------------------------------------------------- ARIMA

def _arima_key(fields: PythonObject, keep: PythonObject, y: PythonObject, params: PythonObject) raises -> PythonObject:
    """The ARIMA generation key: the scalar fields a shard's predict reads
    (None for an absent one) and the two fitted arrays, by identity."""
    var key = _b().list()
    for k in range(len(keep)):  # small-loop(the scalar fields a shard's predict reads)
        _ = py_call(key.append, py_get(fields, keep[k]))
    _ = py_call(key.append, y)
    _ = py_call(key.append, params)
    return key


def _predict_arima(estimator: PythonObject, start_obj: PythonObject, end_obj: PythonObject, exog: PythonObject,
                   devices: PythonObject, sps: PythonObject) raises -> PythonObject:
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if py_type(estimator) is not f[].arima_cls:
        raise Error("TypeError: predict_arima requires mojolearn.ARIMA")
    var host = _member_host(estimator)
    var fields = py_fields(estimator)
    var nk = parallel_keys()
    if not py_contains(fields, nk[].k_params_):
        raise Error("ValueError: mojolearn ARIMA: call fit() before predict")
    _arima_mode(py_get(fields, nk[].k_numeric_mode))
    forecast_exog_admit(Int(py=py_get(fields, nk[].k_n_exog_, PythonObject(0))), exog is not None)
    var n_obs = Int(py=py_item(fields, nk[].k_n_obs_))
    var start = Int(py=py_call(b.int, start_obj))
    var end = n_obs if end_obj is None else Int(py=py_call(b.int, end_obj))
    forecast_span_admit(start, end, "need 0 <= start < end; end is exclusive")
    var batch = Int(py=py_item(fields, nk[].k_batch_size_))
    var ranges = _ranges(batch, sps)
    _devices(devices)
    var complexity = Int(py=py_item(fields, nk[].k_complexity_))
    var keep = Python.tuple("order", "seasonal_order", "trend", "k_", "method", "maxiter", "n_exog_", "complexity_",
                            "n_obs_", "_y_ndim", "numeric_mode")
    var context = f[].arima_context
    var shards = len(ranges) // 2
    var workers = len(py_call(b.tuple, devices))
    var shard_width = ranges[1] - ranges[0] if shards > 0 else 0
    # The generation key: every object a shard's state is cut from, by
    # identity (once per public call; no row is read). The estimator itself
    # is the registry's weak key, never a key item.
    var y = py_item(fields, nk[].k__y)
    var params = py_item(fields, nk[].k_params_)
    var key = _arima_key(fields, keep, y, params)
    var mods = NativeModules()
    var record = prepared_get(mods, _GENERATION_SLOT, estimator)
    # Sealed by an earlier store of this estimator and still read-only: one
    # identity pass over the key and one flag read per array.
    var ours = (_generation_ours(record, key) and _generation_frozen(y, context)
                and _generation_frozen(params, context))
    var acquired = _shard_acquire(devices, _CHILD, host)
    var pool = acquired.pool
    var parts = PythonObject(None)
    var lifetime = PythonObject(None)
    try:
        # The generation's wire name, read in the same step as its check (or
        # returned by this call's own store): a predict never names a
        # generation another call stored meanwhile.
        var issued = f[].arima_generation.issued
        if not (ours and _generation_same(f[].arima_generation, String(py=record[0]), shard_width, pool)):
            var claimed = ours
            if not ours:
                # Seal: the estimator's two fitted arrays become private
                # read-only copies (never the objects a caller may hold a
                # writable view of), and the key names them. An array that
                # cannot be sealed leaves no claim: the next predict stores
                # again from its words of that moment.
                var sealed_y = _generation_seal(y, context)
                var sealed_params = _generation_seal(params, context)
                if sealed_y is not None and sealed_params is not None:
                    py_setitem(fields, nk[].k__y, sealed_y)
                    py_setitem(fields, nk[].k_params_, sealed_params)
                    py_setattr(y, "_readonly", PythonObject(True))
                    py_setattr(params, "_readonly", PythonObject(True))
                    y = sealed_y
                    params = sealed_params
                    key = _arima_key(fields, keep, y, params)
                    claimed = True
            var states = b.list()
            for r in range(shards):  # driver-loop: one stored state per device shard
                var lo = ranges[2 * r]
                var hi = ranges[2 * r + 1]
                # only the two series-major arrays the prediction kernel reads travel
                var state = b.dict()
                for k in range(len(keep)):  # driver-loop: over the scalar fields a shard's predict reads
                    if py_contains(fields, keep[k]):
                        py_setitem(state, keep[k], fields[keep[k]])
                py_setitem(state, nk[].k_batch_size_, hi - lo)
                py_setitem(state, nk[].k__y, _rows(y, lo, hi, n_obs, context))
                py_setitem(state, nk[].k_params_, _rows(params, lo, hi, complexity, context))
                _ = py_call(states.append, state)
            issued = _generation_store(f[].arima_generation, _GENERATION_ARIMA, shard_width, states, pool, workers)
            var token = _generation_token(_GENERATION_ARIMA, issued)
            var claim = PythonObject(None)
            if claimed:
                claim = key
            lifetime = _generation_claim(estimator, _GENERATION_ARIMA, token, claim)
        parts = _generation_predict(PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD, issued, shards, workers,
                                    start, end, pool)
    except error:
        # A failed call leaves no claim: the pool closes (its workers and their models end with it).
        f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
        f[].arima_generation.token = String()
        process_pool_close(pool)
        raise error^
    _shard_keep(acquired^)
    # An unrecorded lifetime (an estimator that takes no weak reference)
    # ends here, with the pool idle: the workers drop the models then.
    _ = lifetime
    var width = end - start
    var result = tsa_array("f", "<f4", Python.tuple(batch, width), batch * width, 4, context)
    for r in range(len(ranges) // 2):  # driver-loop: places one result buffer per device shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var value = parts[r]
        if not Bool(py=py_call(b.tuple, value.shape) == Python.tuple(hi - lo, width)) or not Bool(py=value.dtype == "<f4"):
            raise Error("ValueError: ARIMA worker returned an invalid prediction shape or dtype")
        forecast_rows_place(Int(py=classical_address(result)), Int(py=classical_address(value)), lo, hi, width)
        _ = result  # keep the Python object alive through the native call above
        _ = value  # keep the Python object alive through the native call above
    return result


def _forecast_arima(estimator: PythonObject, steps_obj: PythonObject, exog: PythonObject, devices: PythonObject,
                    sps: PythonObject) raises -> PythonObject:
    var fields = py_fields(estimator)
    var nk = parallel_keys()
    if not py_contains(fields, nk[].k_params_):
        raise Error("ValueError: mojolearn ARIMA: call fit() before forecast")
    var steps = Int(py=py_call(_b().int, steps_obj))
    forecast_positive_admit(steps, "steps")
    var n_obs = Int(py=py_item(fields, nk[].k_n_obs_))
    return _predict_arima(estimator, PythonObject(n_obs), PythonObject(n_obs + steps), exog, devices, sps)


# ---------------------------------------------------------------- Holt-Winters

def _predict_hw(estimator: PythonObject, start_obj: PythonObject, end_obj: PythonObject, index: PythonObject,
                devices: PythonObject, sps: PythonObject) raises -> PythonObject:
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if py_type(estimator) is not f[].hw_cls:
        raise Error("TypeError: requires mojolearn.ExponentialSmoothing")
    var host = _member_host(estimator)
    var fields = py_fields(estimator)
    var nk = parallel_keys()
    if not Bool(py=py_item(fields, nk[].k_fit_executed_flag)):
        raise Error("ValueError: fit the model before prediction")
    forecast_process_mode_admit(String(py=py_item(f[].backend, nk[].k__DEFAULT_MODE)),
                                "parallel forecasting requires IDENTICAL numeric mode")
    if py_type(start_obj) is not b.int or py_type(end_obj) is not b.int:
        raise Error("TypeError: start and end must be integers")
    var start = Int(py=start_obj)
    var end = Int(py=end_obj)
    forecast_span_admit(start, end, "need 0 <= start < end")
    var n = Int(py=py_item(fields, nk[].k_n))
    forecast_out_of_sample_admit(start, n)
    var batch = Int(py=py_item(fields, nk[].k_ts_num))
    if index is not None:
        if py_type(index) is not b.int:
            raise Error("TypeError: index must be int or None")
        forecast_index_admit(Int(py=index), batch)
    var ranges = _ranges(batch, sps)
    _devices(devices)
    var context = f[].hw_context
    var steps = n - Int(py=py_item(fields, nk[].k_seasonal_periods))
    var comps = py_item(fields, nk[].k__comps)
    var drop = py_call(_hw_fitted().union, Python.list("endog"))
    # prediction reads the component state and the constructor settings the
    # native snapshot admits (start_periods, eps, initialization_method,
    # numeric_mode, ...): every carried scalar field, without the batch's
    # fitted arrays, then each shard's slice. The carried fields are walked
    # once; each shard takes a shallow copy and sets its own words.
    var shards = len(ranges) // 2
    var workers = len(py_call(b.tuple, devices))
    var shard_width = ranges[1] - ranges[0] if shards > 0 else 0
    # The generation key: every field of the estimator's instance
    # dictionary, names and values by identity (once per public call; the
    # carried-field walk and the shard cuts below run only for a new
    # generation). The estimator itself is the registry's weak key, never a
    # key item.
    var mods = NativeModules()
    var key = b.list()
    prepared_key_fields(mods, key, fields, List[String]())
    var record = prepared_get(mods, _GENERATION_SLOT, estimator)
    var ours = _generation_ours(record, key) and _generation_frozen(comps, context)
    var acquired = _shard_acquire(devices, _CHILD, host)
    var pool = acquired.pool
    var parts = PythonObject(None)
    var lifetime = PythonObject(None)
    try:
        # The generation's wire name, read in the same step as its check (or
        # returned by this call's own store): a predict never names a
        # generation another call stored meanwhile.
        var issued = f[].hw_generation.issued
        if not (ours and _generation_same(f[].hw_generation, String(py=record[0]), shard_width, pool)):
            var claimed = ours
            if not ours:
                # Seal (see `_predict_arima`): the packed components become a
                # private read-only copy and the key is walked again over the
                # fields as they now stand (once per sealed estimator).
                var sealed = _generation_seal(comps, context)
                if sealed is not None:
                    py_setitem(fields, nk[].k__comps, sealed)
                    py_setattr(comps, "_readonly", PythonObject(True))
                    comps = sealed
                    key = b.list()
                    prepared_key_fields(mods, key, fields, List[String]())
                    claimed = True
            var base = shard_state(fields, drop, context)
            var states = b.list()
            for r in range(shards):  # driver-loop: one stored state per device shard
                var lo = ranges[2 * r]
                var hi = ranges[2 * r + 1]
                var w = hi - lo
                var state = py_call(b.dict, base)
                py_setitem(state, nk[].k_n, n)
                py_setitem(state, nk[].k_fit_executed_flag, py_item(fields, nk[].k_fit_executed_flag))
                py_setitem(state, nk[].k_ts_num, w)
                py_setitem(state, nk[].k__components_len, steps * w)
                var part = tsa_array("f", "<f4", Python.tuple(3 * steps * w), 3 * steps * w, 4, context)
                # rows r = component * steps + step: dst[r * w + j] = src[r * batch + lo + j]
                forecast_gather_cols(Int(py=classical_address(part)), Int(py=classical_address(comps)), 3 * steps, batch, lo, w)
                _ = part  # keep the Python object alive through the native call above
                _ = comps  # keep the Python object alive through the native call above
                py_setitem(state, nk[].k__comps, part)
                _ = py_call(states.append, state)
            issued = _generation_store(f[].hw_generation, _GENERATION_HW, shard_width, states, pool, workers)
            var token = _generation_token(_GENERATION_HW, issued)
            var claim = PythonObject(None)
            if claimed:
                claim = key
            lifetime = _generation_claim(estimator, _GENERATION_HW, token, claim)
        parts = _generation_predict(PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD, issued, shards, workers,
                                    start, end, pool)
    except error:
        # A failed call leaves no claim: the pool closes (its workers and their models end with it).
        f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
        f[].hw_generation.token = String()
        process_pool_close(pool)
        raise error^
    _shard_keep(acquired^)
    # An unrecorded lifetime ends here, with the pool idle (see `_predict_arima`).
    _ = lifetime
    var width = end - start
    var result = tsa_array("f", "<f4", Python.tuple(width * batch), width * batch, 4, context)
    for r in range(len(ranges) // 2):  # driver-loop: places one result buffer per device shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var value = parts[r]
        var expected = Python.tuple(width) if hi - lo == 1 else Python.tuple(width, hi - lo)
        if not Bool(py=py_call(b.tuple, value.shape) == expected) or not Bool(py=value.dtype == "<f4"):
            raise Error("ValueError: Holt-Winters worker returned an invalid prediction shape or dtype")
        forecast_scatter_cols(Int(py=classical_address(result)), Int(py=classical_address(value)), width, batch, lo, hi - lo)
        _ = value  # keep the Python object alive through the native call above
        _ = result  # keep the Python object alive through the native call above
    # `estimator._shaped(result, width, index)`: one series, the flat
    # block (one series in all), or the (width, ts_num) block
    if index is not None:
        var one = tsa_array("f", "<f4", Python.tuple(width), width, 4, context)
        forecast_gather_cols(Int(py=classical_address(one)), Int(py=classical_address(result)), width, batch, Int(py=index), 1)
        _ = one  # keep the Python object alive through the native call above
        _ = result  # keep the Python object alive through the native call above
        return one
    if batch == 1:
        return result
    var block = tsa_array("f", "<f4", Python.tuple(width, batch), width * batch, 4, context)
    shard_block_copy(Int(py=classical_address(block)), Int(py=classical_address(result)), width * batch * 4)
    _ = block  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    return block


def _forecast_hw(estimator: PythonObject, h: PythonObject, index: PythonObject, devices: PythonObject,
                 sps: PythonObject) raises -> PythonObject:
    if py_type(h) is not _b().int:
        raise Error("TypeError: h must be an integer")
    forecast_positive_admit(Int(py=h), "h")
    var nk = parallel_keys()
    var n = Int(py=py_item(py_fields(estimator), nk[].k_n))
    return _predict_hw(estimator, PythonObject(n), PythonObject(n + Int(py=h)), index, devices, sps)


# ---------------------------------------------------------------- fits
# parallel_classical.fit_arima / fit_exponential_smoothing, one native call
# each (lane/native-parallel-driver-20261024). The partition, the driver
# negative control and the merge are the old Python driver's: whole series in
# `series_per_shard` ranges (each shard after the first on an owner above rank
# 0 reads one series early under the switch), each shard fitted by the plain
# native fit in its own worker, every fitted array placed at its shard's
# series offset. The plain fit's own words on the same series: no bit moves.

#: ARIMA fields the fit writes; a shard worker's state carries the estimator's
#: constructor fields only.
def _arima_fitted() raises -> PythonObject:
    return _b().frozenset(Python.list("params_", "x_", "x0_", "n_iter_", "retcode_", "llf_", "fx_", "aic_", "bic_",
                                      "_y", "_exog", "batch_size_", "n_obs_", "input_copied_", "_y_ndim"))


def _fit_arima(estimator: PythonObject, y: PythonObject, devices: PythonObject, sps: PythonObject,
               exog: PythonObject) raises -> PythonObject:
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if py_type(estimator) is not f[].arima_cls:
        raise Error("TypeError: fit_arima requires mojolearn.ARIMA")
    var host = _member_host(estimator)
    if exog is not None:
        raise Error("NotImplementedError: fit_arima does not shard exogenous regressors; ARIMA.fit(y, exog) fits "
                    + "them on one device")
    var fields = py_fields(estimator)
    var nk = parallel_keys()
    _arima_mode(py_get(fields, nk[].k_numeric_mode))
    _ = _width(sps)
    var context = f[].arima_context
    var series = _series_major(y, context)
    var data = series[0]
    var batch = Int(py=series[1])
    var n_obs = Int(py=series[2])
    var copied = series[3]
    var ranges = shard_ranges(batch, sps, "series_per_shard", "")
    var drop = _arima_fitted()
    var state = shard_state(fields, drop, context)
    py_setitem(state, nk[].k_numeric_mode, "identical")
    var n_devices = shard_devices(devices)
    var requests = shard_wave()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per series shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        requests.add(PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD, state,
                                                  Python.tuple(shard_rows(data, lo - shift, hi - shift, context)))
    var parts = shard_run(requests, devices, _CHILD, host)
    var merged = b.dict()
    var names = [String("params_"), String("x_"), String("x0_"), String("n_iter_"), String("retcode_"),
                 String("llf_"), String("fx_"), String("aic_"), String("bic_")]
    for i in range(len(names)):  # small-loop(the nine series-major fitted arrays)
        py_setitem(merged, names[i], shard_concat(parts, names[i], batch, context))
    # `result = ARIMA(**params); result.__dict__.update(parts[0].__dict__)`,
    # the merged arrays, then the whole batch's series state
    _ = py_call(fields.clear)
    _ = py_call(fields.update, parts[0])
    _ = py_call(fields.update, merged)
    py_setitem(fields, nk[].k_batch_size_, batch)
    py_setitem(fields, nk[].k_n_obs_, n_obs)
    py_setitem(fields, nk[].k__y, data)
    py_setitem(fields, nk[].k__y_ndim, series[4])
    py_setitem(fields, nk[].k_input_copied_, copied)
    return estimator


#: Holt-Winters fields the fit writes.
def _hw_fitted() raises -> PythonObject:
    return _b().frozenset(Python.list("_data", "_comps", "_components_len", "_comp_layout", "_time_major", "sse_",
                                      "alpha_", "beta_", "gamma_", "n_iter_", "criterion_", "n", "level_", "trend_",
                                      "season_"))


def _fit_hw(estimator: PythonObject, devices: PythonObject, sps: PythonObject) raises -> PythonObject:
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if py_type(estimator) is not f[].hw_cls:
        raise Error("TypeError: requires mojolearn.ExponentialSmoothing")
    var host = _member_host(estimator)
    var nk = parallel_keys()
    forecast_process_mode_admit(String(py=py_item(f[].backend, nk[].k__DEFAULT_MODE)),
                                "ExponentialSmoothing requires IDENTICAL numeric mode")
    _ = _width(sps)
    var fields = py_fields(estimator)
    var batch = Int(py=py_item(fields, nk[].k_ts_num))
    forecast_batch_admit(batch)
    var context = f[].hw_context
    var checked = _check_dims(fields, py_item(fields, nk[].k_endog), context)
    var data = checked[0]
    var n = Int(py=checked[1])
    var ranges = shard_ranges(batch, sps, "series_per_shard", "")
    var drop = _hw_fitted()
    var state = shard_state(fields, drop, context)
    py_setitem(state, nk[].k_fit_executed_flag, False)
    var n_devices = shard_devices(devices)
    var requests = shard_wave()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per series shard
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        var shard_state_r = py_call(b.dict, state)
        # `ExponentialSmoothing(data[lo:hi], ts_num=hi - lo, **params)`
        py_setitem(shard_state_r, nk[].k_endog, shard_rows(data, lo - shift, hi - shift, context))
        py_setitem(shard_state_r, nk[].k_ts_num, hi - lo)
        requests.add(PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD, shard_state_r, Python.tuple())
    var parts = shard_run(requests, devices, _CHILD, host)
    # The plain fit's buffers for the whole batch, every shard's columns at
    # its series offset: the packed components (3 * steps rows of `batch`
    # series), the four per-series statistics and the two flag rows.
    var period = Int(py=py_item(fields, nk[].k_seasonal_periods))
    var steps = n - period
    var comps = shard_array("<f4", Python.tuple(3 * steps * batch), 3 * steps * batch, context)
    var stats = shard_array("<f4", Python.tuple(4 * batch), 4 * batch, context)
    var flags = shard_array("<i4", Python.tuple(2 * batch), 2 * batch, context)
    var stat_names = [String("sse_"), String("alpha_"), String("beta_"), String("gamma_")]
    for r in range(len(ranges) // 2):  # driver-loop: places one shard's blocks
        var lo = ranges[2 * r]
        var w = ranges[2 * r + 1] - lo
        var part = parts[r]
        forecast_length_admit(Int(py=py_item(part, nk[].k_n)), n)
        var shard_comps = py_item(part, nk[].k__comps)
        if steps > 0 and w > 0:
            forecast_scatter_cols(Int(py=classical_address(comps)), Int(py=classical_address(shard_comps)),
                                  3 * steps, batch, lo, w)
            _ = comps  # keep the Python object alive through the native call above
            _ = shard_comps  # keep the Python object alive through the native call above
        for i in range(4):  # small-loop(the four per-series statistics)
            var value = part[stat_names[i]]
            forecast_words_place(Int(py=classical_address(stats)), Int(py=classical_address(value)), i, batch, lo, w)
            _ = stats  # keep the Python object alive through the native call above
            _ = value  # keep the Python object alive through the native call above
        var iters = py_item(part, nk[].k_n_iter_)
        var criteria = py_item(part, nk[].k_criterion_)
        forecast_words_place(Int(py=classical_address(flags)), Int(py=classical_address(iters)), 0, batch, lo, w)
        forecast_words_place(Int(py=classical_address(flags)), Int(py=classical_address(criteria)), 1, batch, lo, w)
        _ = flags  # keep the Python object alive through the native call above
        _ = iters  # keep the Python object alive through the native call above
        _ = criteria  # keep the Python object alive through the native call above
    # `_set_fitted` returns the fitted updates; commit them as the plain
    # `hw_fit` does (discarding them left the estimator unfitted: "fit() the
    # model before forecast()" and no fitted components).
    var updates = _set_fitted(fields, n, comps, stats, flags, context)
    py_setitem(updates, nk[].k__data, data)
    _commit(tsa_hw_fields(estimator, context), updates, context, True)
    return estimator


def _fitted_record(model: PythonObject, names: PythonObject) raises -> PythonObject:
    """The fitted fields `names` of a shard worker's model."""
    var b = _b()
    var fields = py_fields(model)
    var out = b.dict()
    for k in range(len(names)):  # driver-loop: over the fitted field names a shard returns
        if py_contains(fields, names[k]):
            py_setitem(out, names[k], fields[names[k]])
    return out


# ---------------------------------------------------------------- the worker

def _worker_generation_store(operands: List[Int], states: PythonObject) raises -> PythonObject:
    """A worker's store of a generation: `operands` is its typed name
    `[kind, owner, issued]`, `states` the list of the states of the shards
    this worker answers. The models are built here, once per generation; the
    kind's previous generation (and whatever the plain owners held for its
    models) is dropped by the replacement. A store of no states (the
    coordinator's drop, or a worker beyond the shard count) leaves this
    worker holding nothing of the kind."""
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    var nk = parallel_keys()
    if len(operands) != 3 or py_type(states) is not b.list:
        raise Error("parallel forecasting: a generation store carries its name and a list of states")
    var kind = operands[0]
    if kind != _GENERATION_ARIMA and kind != _GENERATION_HW:
        raise Error("parallel forecasting: unknown generation kind")
    var models = b.list()
    for i in range(len(states)):  # driver-loop: one model per shard this worker answers
        if kind == _GENERATION_ARIMA:
            var built = shard_model(f[].arima_cls, states[i])
            py_setitem(py_fields(built), nk[].k__exog, PythonObject(None))
            _ = py_call(models.append, built)
        else:
            _ = py_call(models.append, shard_model(f[].hw_cls, states[i]))
    var held = PythonObject(None)
    if len(models) > 0:
        held = models
    # The slot is read again: building a model can run a collection.
    f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    if kind == _GENERATION_ARIMA:
        f[].arima_generation.owner = operands[1]
        f[].arima_generation.issued = operands[2]
        f[].arima_generation.models = held
    else:
        f[].hw_generation.owner = operands[1]
        f[].hw_generation.issued = operands[2]
        f[].hw_generation.models = held
    return PythonObject(len(models))


def forecast_worker_dispatch(operation: Int, request: PythonObject) raises -> PythonObject:
    """One fit shard: the model rebuilt natively from its state record, then
    the native fit. `operation` is the decoded request word (the identity
    code or a `PARALLEL_FORECAST_WORKER_OPN_*_FIT_SHARD` code); `request` is
    the payload record `(state, args)` (None for identity). The predict and
    generation-store requests are typed and never come here
    (`forecast_worker`)."""
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    var b = _b()
    if operation == PARALLEL_SHARD_WORKER_OPN_IDENTITY:
        return shard_identity()
    forecast_envelope_admit(py_type(request) is b.tuple, len(request) if py_type(request) is b.tuple else 0)
    var state = request[0]
    var args = request[1]
    if operation == PARALLEL_FORECAST_WORKER_OPN_ARIMA_FIT_SHARD:
        var model = shard_model(f[].arima_cls, state)
        _ = arima_fit(model, Python.tuple(args[0], PythonObject(None)), f[].arima_context)
        # everything the fit set except the shard's series copy
        var fields = py_call(b.dict, py_fields(model))
        _ = py_call(fields.pop, "_y", PythonObject(None))
        return shard_state(fields, b.frozenset(), f[].arima_context)
    if operation == PARALLEL_FORECAST_WORKER_OPN_HW_FIT_SHARD:
        var model = shard_model(f[].hw_cls, state)
        _ = hw_fit(model, f[].hw_context)
        return _fitted_record(model, Python.list("n", "_comps", "sse_", "alpha_", "beta_", "gamma_", "n_iter_",
                                                 "criterion_"))
    raise Error("parallel forecasting: unsupported worker operation")


def _worker_turn(turn: WorkerTurn) raises -> PythonObject:
    """One request's result. The typed request: one decode of the frame's
    words (a frame without request words is refused), then the code's
    branch. A predict reads its operands from the request words alone: no
    record is PACKed and no Python value is built before the plain owner's
    typed-span query (its result Array is the reply's one tensor). A
    generation store PACKs its states (the models are Python objects the
    plain owners read); a fit PACKs its `(state, args)` record."""
    var request = shard_request_decode(Int(turn.request.unsafe_ptr()), len(turn.request) * 8)
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    if request.op == PARALLEL_FORECAST_WORKER_OPN_ARIMA_PREDICT_SHARD:
        # The model this worker built at the generation's store.
        var model = _generation_model(f[].arima_generation, request.words)
        return arima_predict_span(model, request.words[3], request.words[4], f[].arima_context)
    if request.op == PARALLEL_FORECAST_WORKER_OPN_HW_PREDICT_SHARD:
        var model = _generation_model(f[].hw_generation, request.words)
        return hw_predict_span(model, request.words[3], request.words[4], f[].hw_context)
    if request.op == PARALLEL_FORECAST_WORKER_OPN_GENERATION_STORE:
        return _worker_generation_store(request.words, process_worker_value(turn))
    if len(request.words) != 0:
        raise Error("parallel shards: invalid worker request envelope")
    return forecast_worker_dispatch(request.op, process_worker_value(turn))


def forecast_worker(argv: PythonObject) raises:
    var socket = process_worker_connect(argv)
    try:
        while True:  # driver-loop: over native local forecast shard requests
            var turn = process_worker_next(socket)
            if not turn.open:
                break
            try:
                process_worker_reply(socket, turn.sequence, _worker_turn(turn))
            except error:
                process_worker_refuse(socket, turn.sequence, String(error))
    except error:
        transport_close(socket)
        raise error^
    transport_close(socket)


def parallel_forecast_call(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    """`operation` is a `PARALLEL_FORECAST_OPN_*` code (-1 for a name the table
    does not hold); `name` is read only for the unknown operation's text."""
    var f = PARALLEL_FORECAST_BOUNDARY.get_or_create_ptr()
    if operation == PARALLEL_FORECAST_OPN_INITIALIZE:
        f[].arima_cls = args[0]
        f[].hw_cls = args[1]
        f[].arima_context = args[2]
        f[].hw_context = args[3]
        f[].backend = args[4]
        return PythonObject(None)
    if operation != PARALLEL_FORECAST_OPN_WORKER and (f[].arima_generation.orphaned or f[].hw_generation.orphaned):
        # A drop still owed to the workers of a freed estimator (two flag
        # reads per public call when none is).
        _generation_sweep()
    if operation == PARALLEL_FORECAST_OPN_PREDICT_ARIMA:
        return _predict_arima(args[0], args[1], args[2], args[3], args[4], args[5])
    if operation == PARALLEL_FORECAST_OPN_FORECAST_ARIMA:
        return _forecast_arima(args[0], args[1], args[2], args[3], args[4])
    if operation == PARALLEL_FORECAST_OPN_PREDICT_EXPONENTIAL_SMOOTHING:
        return _predict_hw(args[0], args[1], args[2], args[3], args[4], args[5])
    if operation == PARALLEL_FORECAST_OPN_FORECAST_EXPONENTIAL_SMOOTHING:
        return _forecast_hw(args[0], args[1], args[2], args[3], args[4])
    if operation == PARALLEL_FORECAST_OPN_FIT_ARIMA:
        return _fit_arima(args[0], args[1], args[2], args[3], args[4])
    if operation == PARALLEL_FORECAST_OPN_FIT_EXPONENTIAL_SMOOTHING:
        return _fit_hw(args[0], args[1], args[2])
    if operation == PARALLEL_FORECAST_OPN_WORKER:
        forecast_worker(args)
        return PythonObject(None)
    raise Error("RuntimeError: parallel forecasting: unknown operation " + String(name))


def parallel_forecast_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var nk = parallel_keys()
        var result = parallel_forecast_call(py_op_code(nk[].ops_forecast, args[0]), args[0], args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("IndexError:"):
            return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
