# SPDX-License-Identifier: Apache-2.0
"""Native owner and local workers for existing parallel MLP/Samba training."""
from std.python import Python, PythonObject
from bindings.parallel_neural_native_run import (
    NEURAL_SCHEMA, NEURAL_REDUCTION, NEURAL_MAX_SHARDS,
    NEURAL_OPERATION_MLP, NEURAL_OPERATION_SAMBA, NEURAL_DELTA_SKIP,
    neural_operation_name, neural_accumulation_admit, neural_mode_admit, neural_devices_admit,
    neural_ids_rank_admit, neural_ids_signed, neural_stream_admit, neural_shard_admit,
    neural_resend_full, neural_cache_reach_admit, neural_delta_action,
    neural_cache_arrived_admit, neural_checkpoint_admit, neural_restore_operation, neural_envelope_admit,
    neural_gradient_admit,
    NeuralTrainerState, neural_step_admit, neural_step_withdraw, neural_step_begin, neural_step_version, neural_step_commit,
    neural_history_excess, neural_gradient_words, neural_update_words, neural_gradient_words_admit,
)
from core.parallel_neural_native_impl import samba_ids_into
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_array, generation_copy
from bindings.parallel_byte_native_door import _parallel_int, _parallel_view
from bindings.parallel_mlp_native_door import (
    mlp_snapshot, mlp_from_state, mlp_publish, mlp_validate_state,
    mlp_gradients_open, mlp_gradients_close, mlp_gradients_on,
    mlp_apply_from_state, mlp_batch, mlp_targets, neural_ordered_sum,
    neural_cooperative_admit,
)
from bindings.native_process_pool_door import (
    process_pool_create, process_pool_map_typed, process_pool_close,
    process_worker_connect, process_worker_next, process_worker_value, process_worker_reply, process_worker_refuse,
    process_pool_devices,
)
from bindings.native_transport_door import transport_close
from bindings.py_balanced import py_call, py_fields, py_get, py_setitem, py_type
from bindings.parallel_neural_admission_unpack import neural_admit
from core.sha256 import Sha256
from bindings.py_cpython_table import cpython_table
from bindings.sha256_door import sha256_update, sha256_update_text
from core.shared_slot import SharedSlot
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_NEURAL_OPN_CHECKPOINT,
    PARALLEL_NEURAL_OPN_CLOSE,
    PARALLEL_NEURAL_OPN_ENTER,
    PARALLEL_NEURAL_OPN_EXIT,
    PARALLEL_NEURAL_OPN_EXPORT_GRADIENTS,
    PARALLEL_NEURAL_OPN_INIT,
    PARALLEL_NEURAL_OPN_INITIALIZE,
    PARALLEL_NEURAL_OPN_INIT_WITH_WORKER,
    PARALLEL_NEURAL_OPN_ORDERED_SUM,
    PARALLEL_NEURAL_OPN_RESTORE,
    PARALLEL_NEURAL_OPN_TRAIN,
    PARALLEL_NEURAL_OPN_WORKER,
)
from core.parallel_worker_request import (
    WorkerRequest, worker_request_decode,
    PN_MLP_GRADIENT_CACHED, PN_SAMBA_GRADIENT_CACHED, PN_MLP_UPDATE, PN_SAMBA_UPDATE,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item


struct ParallelNeuralBoundary(Defaultable, Movable):
    var bi: PythonObject
    var owner_cls: PythonObject
    var mlp_cls: PythonObject
    var samba_cls: PythonObject
    var samba_call: PythonObject
    var samba_context: PythonObject
    var admission_context: PythonObject
    var lock: PythonObject
    #: a worker process's resident model (lane migrate-neural-2): a dict
    #: {epoch, version, state, model} kept across requests
    var cache: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.owner_cls = PythonObject(None)
        self.mlp_cls = PythonObject(None)
        self.samba_cls = PythonObject(None)
        self.samba_call = PythonObject(None)
        self.samba_context = PythonObject(None)
        self.admission_context = PythonObject(None)
        self.lock = PythonObject(None)
        self.cache = PythonObject(None)


comptime PARALLEL_NEURAL_BOUNDARY = SharedSlot[ParallelNeuralBoundary, "MojolearnParallelNeuralOwnerV1"]


struct ParallelNeuralOwner(Movable, Writable):
    """One trainer's native owner. Stored once in the instance dictionary as
    `_native`; every operation reads this record, never instance attributes.
    `native` is the typed state RUN decides on
    (bindings/parallel_neural_native_run.mojo). The Python objects here are
    boundary pins only: the model whose snapshot is taken and published, the
    two pool dictionaries, the lock, the committed gradient buffers and the
    committed step deltas (the wire payload of the next request's header).
    `model`, `logical_shards`, `_pool`, `_update_pool` and `_lock` on the
    instance are mirrors for callers and for the admission walk; the door
    writes them once and reads only `model`, to refuse a replaced one."""
    var owner_id: Int
    var native: NeuralTrainerState
    var model: PythonObject
    var pool: PythonObject
    var update_pool: PythonObject
    var lock: PythonObject
    var gradients: PythonObject
    var history: PythonObject

    def __init__(out self, owner_id: Int, operation: Int, logical_shards: Int, shared_pool: Bool,
                 model: PythonObject, pool: PythonObject, update_pool: PythonObject,
                 lock: PythonObject, history: PythonObject):
        self.owner_id = owner_id
        self.native = NeuralTrainerState(operation, logical_shards, shared_pool)
        self.model = model
        self.pool = pool
        self.update_pool = update_pool
        self.lock = lock
        self.gradients = PythonObject(None)
        self.history = history

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ParallelNeuralOwner(operation=", self.native.operation, ", closed=", self.native.closed, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def parallel_neural_initialize(args: PythonObject) raises:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    b[].bi = Python.import_module("builtins")
    b[].owner_cls = args[0]
    b[].mlp_cls = args[1]
    b[].samba_cls = args[2]
    b[].samba_call = args[3]
    b[].samba_context = args[4]
    b[].admission_context = args[5]
    b[].lock = Python.import_module("_thread").RLock
    if b[].samba_call is not None and py_type(b[].samba_call) is not py_type(b[].bi.len):
        raise Error("parallel neural: Samba operations must use the compiled native dispatcher")


def _neural_samba(operation: String, values: PythonObject) raises -> PythonObject:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    if b[].samba_call is None:
        raise Error("ImportError: installed native Samba training dispatcher is unavailable")
    return py_call(b[].samba_call, operation, py_call(b[].bi.tuple, values), b[].samba_context)


def _neural_values(value: PythonObject) raises -> PythonObject:
    var result = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()[].bi.list()
    _ = py_call(result.append, value)
    return result


def _neural_record(owner: PythonObject) raises -> PythonObject:
    """The exact native owner record of an admitted trainer instance."""
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var nk = parallel_keys()
    var record = py_get(py_fields(owner), nk[].k__native)
    if record is None:
        raise Error("RuntimeError: parallel trainer is not initialized")
    var owner_id = Int(py=py_call(b[].bi.id, owner))
    try:
        var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
        if rec[].owner_id != owner_id:
            raise Error("foreign")
    except:
        raise Error("TypeError: parallel trainer storage does not belong to this trainer")
    return record


def _neural_model_admit(owner: PythonObject, record: PythonObject) raises:
    """The admission walk validated the instance's `model`; the operations
    read the record's. Refuse when they are not one object."""
    var nk = parallel_keys()
    var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
    if py_get(py_fields(owner), nk[].k_model) is not rec[].model:
        raise Error("TypeError: ParallelNeuralTrainer.model cannot be replaced; construct a new trainer")


def _neural_snapshot(record: PythonObject) raises -> PythonObject:
    var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
    if rec[].native.operation == NEURAL_OPERATION_MLP:
        return mlp_snapshot(rec[].model)
    var args = _neural_values(rec[].model)
    _ = py_call(args.append, True)
    return _neural_samba("snapshot", args)


def _neural_publish(record: PythonObject, state: PythonObject) raises:
    var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
    if rec[].native.operation == NEURAL_OPERATION_MLP:
        mlp_publish(rec[].model, state)
    else:
        var args = _neural_values(rec[].model)
        _ = py_call(args.append, state)
        _ = _neural_samba("publish", args)


def _neural_init(owner: PythonObject, model: PythonObject, devices: PythonObject, logical: PythonObject, child_module: String) raises:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var shards = _parallel_int(logical, 1, NEURAL_MAX_SHARDS, "logical_shards")
    var operation = NEURAL_OPERATION_MLP
    var nk = parallel_keys()
    if py_type(model) is b[].mlp_cls:
        _ = mlp_snapshot(model)
    elif py_type(model) is b[].samba_cls:
        operation = NEURAL_OPERATION_SAMBA
        var fields = py_call(b[].bi.object.__getattribute__, model, "__dict__")
        var optimizer = py_call(b[].bi.object.__getattribute__, py_item(fields, nk[].k_optimizer), "__dict__")
        neural_accumulation_admit(Int(py=py_item(optimizer, nk[].k_accumulation_steps)))
        var own_mode = py_item(fields, nk[].k_numeric_mode)
        neural_mode_admit(own_mode is not None, String(own_mode) if own_mode is not None else String(""),
                          "parallel training requires identical numeric mode")
        var values = _neural_values(model)
        _ = py_call(values.append, True)
        var snapshot = _neural_samba("snapshot", values)
        neural_mode_admit(True, String(py_item(snapshot, nk[].k_numeric_mode)), "parallel training requires an identical model")
    else:
        raise Error("TypeError: ParallelNeuralTrainer supports SmallMLPTrainer and SambaStack")
    var pool = process_pool_create(devices, False, child_module)
    var pool_devices = process_pool_devices(pool)
    neural_devices_admit(len(pool_devices), shards)
    # One device (lane perf-trees-misc, 2026-11-05): the cooperative update
    # group IS the gradient worker's group (the same device, the same worker
    # environment: `_process_environment` sees the same devices and group),
    # so the update request goes to the gradient pool's worker instead of a
    # second spawned process per owner (the neural-trainer-unfused cell
    # paid two interpreter starts, imports and binding connections). The
    # worker dispatch takes gradient and update requests alike; the update
    # reads only the request's state, never the worker's resident cache.
    var update_pool = pool
    if len(pool_devices) != 1:
        var admitted = b[].bi.list()
        for i in range(len(pool_devices)):  # small-loop(the devices the gradient pool admitted)
            _ = py_call(admitted.append, PythonObject(pool_devices[i]))
        update_pool = process_pool_create(admitted, True, child_module)
    var lock = b[].lock()
    var owner_id = Int(py=py_call(b[].bi.id, owner))
    var record = PythonObject(alloc=ParallelNeuralOwner(owner_id, operation, shards, len(pool_devices) == 1,
                                                        model, pool, update_pool, lock, b[].bi.list()))
    # PACK, once per trainer: the record, and the mirrors callers and the
    # admission walk read. The instance dictionary is written directly (an
    # exact admitted class: no attribute hook exists to run).
    var fields = py_fields(owner)
    py_setitem(fields, nk[].k__native, record)
    py_setitem(fields, nk[].k__pool, pool)
    py_setitem(fields, nk[].k__update_pool, update_pool)
    py_setitem(fields, nk[].k_model, model)
    py_setitem(fields, nk[].k_logical_shards, PythonObject(shards))
    py_setitem(fields, nk[].k__lock, lock)


def _neural_ids(value: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var own = py_type(value) is g[].array_cls
    var view = _parallel_view(value)
    var shape = value.shape if own else view.shape
    neural_ids_rank_admit(len(shape))
    var rows = Int(py=shape[0])
    var columns = Int(py=shape[1])
    var fmt = String(py_call(view.format.lstrip, "=@<"))
    var signed = neural_ids_signed(fmt)
    var packed = py_call(g[].arrays, fmt)
    var item = Int(py=packed.itemsize)
    var out = generation_array([rows, columns], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    var column_major = own and String(value.order) == "F"
    # A native-order C-contiguous export whose item size matches the packing
    # array's is already the packed buffer tobytes + frombytes would build:
    # widen from it in place while the export is held. Other layouts keep the
    # packing copy.
    var done = False
    if Int(py=view.itemsize) == item and Bool(py=view.c_contiguous):
        var buffer = List[Int](length=10, fill=0)
        var bp = buffer.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        # PyBUF_C_CONTIGUOUS (0x38): only a C-contiguous export succeeds
        if cpython_table()[].PyObject_GetBuffer(view._obj_ptr, bp, Int32(0x38)) == 0:
            if buffer[2] == rows * columns * item:
                try:
                    samba_ids_into(buffer[0], item, signed, rows, columns, column_major, dst)
                except error:
                    # release the export before the refusal propagates
                    cpython_table()[].PyBuffer_Release(bp)
                    raise error^
                done = True
            cpython_table()[].PyBuffer_Release(bp)
            _ = buffer^  # the Py_buffer words stay alive through the release above
        else:
            ref cpy = Python().cpython()
            cpython_table()[].PyErr_Clear()
    if not done:
        _ = py_call(packed.frombytes, py_call(view.tobytes, "C"))
        samba_ids_into(Int(py=packed.buffer_info()[0]), item, signed, rows, columns, column_major, dst)
    _ = view  # the exporter stays alive through the reads above
    _ = out  # keep the Python object alive through the native call above
    _ = packed^
    return out


def _neural_request(state: PythonObject, args: PythonObject) raises -> PythonObject:
    """One request's payload: the model slot and the arguments. The
    operation (and, for a gradient, the cache epoch and version) are the
    typed request's words (`neural_gradient_words`, `neural_update_words`)."""
    var bi = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()[].bi
    var request = bi.list()
    _ = py_call(request.append, state)
    _ = py_call(request.append, args)
    return py_call(bi.tuple, request)


def _digest_array_bytes(mut h: Sha256, view: PythonObject) raises:
    """`h.update(view.tobytes("C"))` without the two copies: a C-contiguous
    export is hashed in place (the same bytes tobytes would build), while
    the export is held; any other layout keeps the copy."""
    var buffer = List[Int](length=10, fill=0)
    var bp = buffer.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    # PyBUF_C_CONTIGUOUS (0x38): only a C-contiguous export succeeds
    if cpython_table()[].PyObject_GetBuffer(view._obj_ptr, bp, Int32(0x38)) == 0:
        h.update(buffer[0], buffer[2])
        cpython_table()[].PyBuffer_Release(bp)
        _ = buffer^  # the Py_buffer words stay alive through the release above
        _ = view  # the exporter stays alive through the hash above
        return
    ref cpy = Python().cpython()
    cpython_table()[].PyErr_Clear()
    sha256_update(h, py_call(view.tobytes, "C"))


def _digest_into(mut h: Sha256, value: PythonObject) raises -> Bool:
    """Hash a model state (dicts, lists, tuples, arrays, scalars) into `h`;
    False for a value it cannot describe exactly (the caller then sends the
    full state)."""
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if value is None or py_type(value) is b[].bi.bool or py_type(value) is b[].bi.int or py_type(value) is b[].bi.float or py_type(value) is b[].bi.str:
        sha256_update_text(h, py_call(b[].bi.repr, value))
        h.update_text(String("\n"))
        return True
    if py_type(value) is b[].bi.dict:
        var keys = py_call(b[].bi.sorted, py_call(b[].bi.list, value))
        h.update_text(String("{"))
        for i in range(len(keys)):  # driver-loop: over state record fields
            sha256_update_text(h, py_call(b[].bi.repr, keys[i]))
            if not _digest_into(h, value[keys[i]]):
                return False
        h.update_text(String("}"))
        return True
    if py_type(value) is b[].bi.list or py_type(value) is b[].bi.tuple:
        h.update_text(String("["))
        for i in range(len(value)):  # driver-loop: over state record items
            if not _digest_into(h, value[i]):
                return False
        h.update_text(String("]"))
        return True
    if py_type(value) is g[].array_cls:
        sha256_update_text(h, value.shape)
        sha256_update_text(h, value.dtype)
        _digest_array_bytes(h, value._mv)
        return True
    if py_type(value) is b[].bi.bytes or py_type(value) is b[].bi.bytearray:
        sha256_update(h, value)
        return True
    return False


def _state_digest(state: PythonObject) raises -> String:
    """The committed model state's digest, or "" when it cannot be taken."""
    var h = Sha256()
    if not _digest_into(h, state):
        return String()
    return h.hexdigest()


def _cache_header(record: PythonObject, snapshot: PythonObject, full: Bool) raises -> PythonObject:
    """The request's model slot: the full state (the worker keeps it), or
    the last committed step deltas (the worker brings its resident copy up
    to the request's version by applying them). The owner's epoch and
    version are the typed request's two words, not fields here."""
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
    var header = b[].bi.dict()
    var nk = parallel_keys()
    py_setitem(header, nk[].k_full, snapshot if full else PythonObject(None))
    py_setitem(header, nk[].k_history, b[].bi.list() if full else rec[].history)
    return header


def _worker_model(header: PythonObject, samba: Bool, epoch: Int, version: Int) raises -> PythonObject:
    """The worker's resident state (and, for Samba, its resident model) at
    the header's version: replaced by a full state, else advanced by the
    step deltas. A worker that cannot reach the version refuses with the
    cache-miss words and the owner resends the full state."""
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    if b[].cache is None:
        b[].cache = b[].bi.dict()
    var cache = b[].cache
    var nk = parallel_keys()
    var full = py_item(header, nk[].k_full)
    if full is not None:
        py_setitem(cache, nk[].k_epoch, PythonObject(epoch))
        py_setitem(cache, nk[].k_version, PythonObject(version))
        py_setitem(cache, nk[].k_state, full)
        py_setitem(cache, nk[].k_model, _neural_samba("from_state", _neural_values(full)) if samba else PythonObject(None))
        return cache
    var resident = py_get(cache, nk[].k_state) is not None
    var same_epoch = resident and Int(py=py_item(cache, nk[].k_epoch)) == epoch
    var have = Int(py=py_item(cache, nk[].k_version)) if same_epoch else 0
    var want = version if same_epoch else 0
    neural_cache_reach_admit(resident, same_epoch, have, want)
    var history = py_item(header, nk[].k_history)
    for i in range(len(history)):  # driver-loop: over committed step deltas in version order
        var entry = history[i]
        var version = Int(py=entry[0])
        if neural_delta_action(version, have) == NEURAL_DELTA_SKIP:
            continue
        var state = py_item(cache, nk[].k_state)
        var updated = PythonObject(None)
        if samba:
            var update_state = state
            if entry[2] is not None:
                update_state = py_call(b[].bi.dict, state)
                py_setitem(update_state, nk[].k_rng, entry[2])
            var values = _neural_values(update_state)
            _ = py_call(values.append, entry[1])
            updated = _neural_samba("apply", values)
            var publish = _neural_values(py_item(cache, nk[].k_model))
            _ = py_call(publish.append, updated)
            _ = _neural_samba("publish", publish)
        else:
            updated = mlp_apply_from_state(state, entry[1])
        py_setitem(cache, nk[].k_state, updated)
        have = version
        py_setitem(cache, nk[].k_version, have)
    neural_cache_arrived_admit(have, want)
    return cache


def _neural_requests(header: PythonObject, shard_args: PythonObject) raises -> PythonObject:
    """The payloads of one gradient wave, one per logical shard."""
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var requests = b[].bi.list()
    for i in range(len(shard_args)):  # driver-loop: over logical training shards in immutable reduction order
        _ = py_call(requests.append, _neural_request(header, shard_args[i]))
    return requests


def _neural_train(owner: PythonObject, record: PythonObject, supplied: PythonObject) raises -> PythonObject:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
    var shards = py_call(b[].bi.list, supplied)
    neural_step_admit(rec[].native, len(shards))
    _neural_model_admit(owner, record)
    var snapshot = _neural_snapshot(record)
    # from here a failed step leaves nothing to export
    neural_step_withdraw(rec[].native)
    rec[].gradients = PythonObject(None)
    var nk = parallel_keys()
    try:
        var shard_args = b[].bi.list()
        var stream = PythonObject(None)
        var next_rng = PythonObject(None)
        var samba = rec[].native.operation == NEURAL_OPERATION_SAMBA
        if samba and Float64(py=py_item(py_item(snapshot, nk[].k_config), nk[].k_dropout)) > 0:
            var counter = Int(py=py_item(py_item(snapshot, nk[].k_rng), nk[].k_counter))
            neural_stream_admit(counter)
            stream = counter
            next_rng = py_call(b[].bi.dict, py_item(snapshot, nk[].k_rng))
            py_setitem(next_rng, nk[].k_counter, counter + 1)
        var offset = 0
        for i in range(len(shards)):  # driver-loop: over logical training shards in immutable reduction order
            var shard = py_call(b[].bi.tuple, shards[i])
            neural_shard_admit(len(shard))
            var args = b[].bi.list()
            if samba:
                var inputs = _neural_ids(shard[0])
                _ = py_call(args.append, inputs)
                _ = py_call(args.append, _neural_ids(shard[1]))
                _ = py_call(args.append, stream)
                _ = py_call(args.append, offset)
                offset += Int(py=inputs.size)
            else:
                var inputs = mlp_batch(shard[0])
                _ = py_call(args.append, inputs)
                _ = py_call(args.append, mlp_targets(shard[1], Int(py=inputs.shape[0])))
            _ = py_call(shard_args.append, py_call(b[].bi.tuple, args))
        # The workers keep the model resident (lane migrate-neural-2): when
        # the owner's state is the one it last committed, a request carries
        # only the committed step deltas; otherwise (first step, a change made
        # outside this trainer, a worker that fell behind) the full state.
        var digest = _state_digest(snapshot)
        var cached = neural_step_begin(rec[].native, digest)
        if not cached:
            rec[].history = b[].bi.list()
        var results = PythonObject(None)
        try:
            results = process_pool_map_typed(rec[].pool, neural_gradient_words(samba, rec[].native, len(shard_args)),
                                             _neural_requests(_cache_header(record, snapshot, not cached), shard_args))
        except error:
            if not neural_resend_full(cached, String(error)):
                raise error
            results = process_pool_map_typed(rec[].pool, neural_gradient_words(samba, rec[].native, len(shard_args)),
                                             _neural_requests(_cache_header(record, snapshot, True), shard_args))
        var losses = b[].bi.list()
        var parts = b[].bi.list()
        for i in range(len(results)):  # driver-loop: over returned logical shard loss/gradient records
            _ = py_call(losses.append, results[i][0])
            _ = py_call(parts.append, results[i][1])
        var update_state = snapshot
        if next_rng is not None:
            update_state = py_call(b[].bi.dict, snapshot)
            py_setitem(update_state, nk[].k_rng, next_rng)
        var update_requests = b[].bi.list()
        _ = py_call(update_requests.append, _neural_request(update_state, parts))
        var updated = process_pool_map_typed(rec[].update_pool, neural_update_words(samba), update_requests)[0]
        _neural_publish(record, updated[0])
        rec[].gradients = updated[1]
        # the committed step as a delta the workers replay: the ordered-sum
        # gradient and the generator state the update read. The requests
        # that carried the list have been answered, so it is extended in
        # place (the per-step list copy is gone). The committed digest is
        # taken BEFORE the delta is appended (lane fx-audit, audit-neural
        # finding 2): a raise in the snapshot or digest then leaves no delta
        # for a step that never committed, so no worker replays it; the
        # append, the trim and the commit after it do not fail on a live list.
        var committed = _state_digest(_neural_snapshot(record))
        var history = rec[].history
        _ = py_call(history.append, Python.tuple(neural_step_version(rec[].native), updated[1], next_rng))
        for _ in range(neural_history_excess(len(history))):  # driver-loop: over retained committed steps beyond the bound (at most one)
            _ = py_call(history.pop, 0)
        neural_step_commit(rec[].native, committed)
        var result = b[].bi.dict()
        py_setitem(result, nk[].k_losses, py_call(b[].bi.tuple, losses))
        py_setitem(result, nk[].k_completed_steps, updated[2])
        py_setitem(result, nk[].k_logical_shards, PythonObject(rec[].native.logical_shards))
        py_setitem(result, nk[].k_reduction, NEURAL_REDUCTION)
        return result
    except error:
        # Keep the same complete-model rollback contract, including RNG and
        # schedule state, even if a publication failed after a worker reply.
        try:
            _neural_publish(record, snapshot)
        except rollback_error:
            raise Error("RuntimeError: parallel operation failed: " + String(error) + "; full-state rollback also failed: " + String(rollback_error))
        raise error


def _neural_checkpoint(owner: PythonObject, record: PythonObject) raises -> PythonObject:
    var result = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()[].bi.dict()
    var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
    var nk = parallel_keys()
    _neural_model_admit(owner, record)
    py_setitem(result, nk[].k_schema, NEURAL_SCHEMA)
    py_setitem(result, nk[].k_operation, PythonObject(neural_operation_name(rec[].native.operation)))
    py_setitem(result, nk[].k_logical_shards, PythonObject(rec[].native.logical_shards))
    py_setitem(result, nk[].k_reduction, NEURAL_REDUCTION)
    py_setitem(result, nk[].k_state, _neural_snapshot(record))
    return result


def _neural_restore(cls: PythonObject, checkpoint: PythonObject, devices: PythonObject, child_module: String) raises -> PythonObject:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var nk = parallel_keys()
    neural_checkpoint_admit(String(py_get(checkpoint, nk[].k_schema, "")), String(py_get(checkpoint, nk[].k_reduction, "")))
    var operation = neural_restore_operation(String(py_get(checkpoint, nk[].k_operation, "")))
    var model = PythonObject(None)
    if operation == NEURAL_OPERATION_MLP:
        model = mlp_from_state(py_item(checkpoint, nk[].k_state))
    else:
        model = _neural_samba("from_state", _neural_values(py_item(checkpoint, nk[].k_state)))
    var owner = py_call(b[].bi.object.__new__, cls)
    _neural_init(owner, model, devices, py_item(checkpoint, nk[].k_logical_shards), child_module)
    return owner


def _neural_close(record: PythonObject) raises:
    var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
    rec[].native.closed = True
    var failure = String()
    try:
        process_pool_close(rec[].pool)
    except error:
        failure = String(error)
    try:
        if not rec[].native.shared_pool:  # one device: one shared pool, closed above
            process_pool_close(rec[].update_pool)
    except error:
        if failure == "":
            failure = String(error)
    if failure != "":
        raise Error(failure)


def neural_worker_dispatch(request: WorkerRequest, payload: PythonObject) raises -> PythonObject:
    """One worker request: `request` is the typed operation (and, for a
    gradient, the owner's cache epoch and version); `payload` is the pair
    (model slot, arguments). REMOVED: the operation string in the payload
    and its table lookup, and the uncached `mlp_gradient` / `samba_gradient`
    operations, which no owner sent."""
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    neural_envelope_admit(len(payload))
    var nk = parallel_keys()
    var operation = request.op
    var state = payload[0]
    var args = payload[1]
    if operation == PN_MLP_GRADIENT_CACHED:
        neural_gradient_words_admit(request)
        # The worker's resident state is replaced (never edited in place) when
        # a full state or a committed delta arrives: validate each resident
        # state object once and open ONE gradient slot over it (the
        # SmallMLPTrainer program, weights uploaded once) for every shard.
        var cache = _worker_model(state, False, request.words[0], request.words[1])
        var resident = py_item(cache, nk[].k_state)
        var slot = py_get(cache, nk[].k_mlp_slot)
        if slot is None or py_get(cache, nk[].k_checked_of) is not resident:
            if slot is not None:
                py_setitem(cache, nk[].k_mlp_slot, PythonObject(None))
                mlp_gradients_close(Int(py=slot))
            var opened = mlp_gradients_open(mlp_validate_state(resident))
            slot = PythonObject(opened)
            py_setitem(cache, nk[].k_mlp_slot, slot)
            py_setitem(cache, nk[].k_checked_of, resident)
        return mlp_gradients_on(Int(py=slot), args[0], args[1])
    if operation == PN_SAMBA_GRADIENT_CACHED:
        neural_gradient_words_admit(request)
        var cache = _worker_model(state, True, request.words[0], request.words[1])
        var values = _neural_values(py_item(cache, nk[].k_model))
        _ = py_call(values.append, args[0])
        _ = py_call(values.append, args[1])
        _ = py_call(values.append, PythonObject(None))
        _ = py_call(values.append, args[2])
        _ = py_call(values.append, args[3])
        return _neural_samba("model_gradients", values)
    if operation != PN_MLP_UPDATE and operation != PN_SAMBA_UPDATE:
        raise Error("parallel neural: unsupported worker operation")
    neural_cooperative_admit()
    var grads = neural_ordered_sum(args)
    var updated = PythonObject(None)
    var step = 0
    if operation == PN_MLP_UPDATE:
        updated = mlp_apply_from_state(state, grads)
        step = Int(py=py_item(py_item(updated, nk[].k_optimizer), nk[].k_step))
    else:
        var values = _neural_values(state)
        _ = py_call(values.append, grads)
        updated = _neural_samba("apply", values)
        step = Int(py=py_item(updated, nk[].k_t))
    var result = b[].bi.list()
    _ = py_call(result.append, updated)
    _ = py_call(result.append, grads)
    _ = py_call(result.append, step)
    return py_call(b[].bi.tuple, result)


def neural_worker(argv: PythonObject) raises:
    var socket = process_worker_connect(argv)
    try:
        while True:  # driver-loop: over complete native local model training requests
            var turn = process_worker_next(socket)
            if not turn.open:
                break
            try:
                var typed = worker_request_decode(Int(turn.request.unsafe_ptr()), len(turn.request) * 8)
                var result = neural_worker_dispatch(typed, process_worker_value(turn))
                process_worker_reply(socket, turn.sequence, result)
            except error:
                process_worker_refuse(socket, turn.sequence, String(error))
    except error:
        transport_close(socket)
        raise error
    transport_close(socket)


def parallel_neural_call(operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    """`operation` is a `PARALLEL_NEURAL_OPN_*` code (-1 for a name the table
    does not hold); `name` is read only for the unknown operation's text."""
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var child = String("mojolearn._native_parallel_neural_worker")
    if operation == PARALLEL_NEURAL_OPN_INITIALIZE:
        parallel_neural_initialize(args)
        return PythonObject(None)
    neural_admit(operation, args, b[].owner_cls, b[].mlp_cls, b[].samba_cls, b[].admission_context)
    if operation == PARALLEL_NEURAL_OPN_INIT:
        _neural_init(args[0], args[1], args[2], args[3], child)
        return PythonObject(None)
    if operation == PARALLEL_NEURAL_OPN_INIT_WITH_WORKER:
        # Verification entry chooses an independent child audit controller;
        # the public constructor always selects the installed native worker.
        _neural_init(args[0], args[1], args[2], args[3], String(args[4]))
        return PythonObject(None)
    if operation == PARALLEL_NEURAL_OPN_RESTORE:
        return _neural_restore(args[0], args[1], args[2], child)
    if operation == PARALLEL_NEURAL_OPN_ORDERED_SUM:
        return neural_ordered_sum(args[0])
    if operation == PARALLEL_NEURAL_OPN_WORKER:
        neural_worker(args)
        return PythonObject(None)
    var owner = args[0]
    # One instance-dictionary read per operation; everything after it is the
    # record (the old form looked up up to a dozen instance attributes).
    var record = _neural_record(owner)
    var rec = record.downcast_value_ptr[ParallelNeuralOwner]()
    var lock = rec[].lock
    _ = lock.acquire()
    var result = PythonObject(None)
    try:
        if operation == PARALLEL_NEURAL_OPN_TRAIN:
            result = _neural_train(owner, record, args[1])
        elif operation == PARALLEL_NEURAL_OPN_EXPORT_GRADIENTS:
            neural_gradient_admit(rec[].native.committed)
            var gradients = rec[].gradients
            result = b[].bi.list()
            for i in range(len(gradients)):  # driver-loop: over committed gradient output buffers
                _ = py_call(result.append, generation_copy(gradients[i]))
        elif operation == PARALLEL_NEURAL_OPN_CHECKPOINT:
            result = _neural_checkpoint(owner, record)
        elif operation == PARALLEL_NEURAL_OPN_CLOSE or operation == PARALLEL_NEURAL_OPN_EXIT:
            _neural_close(record)
        elif operation == PARALLEL_NEURAL_OPN_ENTER:
            result = owner
        else:
            raise Error("parallel neural: unknown operation " + String(name))
    except error:
        _ = lock.release()
        raise error
    _ = lock.release()
    _ = record  # the record outlives every read through `rec` above
    return result


def parallel_neural_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var nk = parallel_keys()
        var result = parallel_neural_call(py_op_code(nk[].ops_neural, args[0]), args[0], args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if message.startswith("OverflowError:"):
            return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
