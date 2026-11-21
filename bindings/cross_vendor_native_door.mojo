# SPDX-License-Identifier: Apache-2.0
"""Cross-vendor replicated training at the Python boundary (UNPACK / PACK).

Every public operation of `mojolearn.cross_vendor` enters here. This file
reads the Python objects (exact classes only, no hooks), keeps each owner's
typed state in a native record stored once on the instance as `_native`
(Coordinator, Worker, BatchFeed, LearningRateTable and the readiness event)
and packs rows and events. The work is typed and lives in
`bindings/cross_vendor_native_run.mojo`: frames over descriptors, the
trainer through its `ParallelByteSession`, the fold and the digests.

The instance attributes written at construction (`port`, `shards`,
`trainer`, `log_path`, ...) are read-only mirrors for callers; operations
read the record. Three are still read from the instance, once per call, as
they always were: a Worker's collaborators (`trainer`, `batches`,
`lr_for_step`, `on_commit`, re-admitted at each run), a Coordinator's
`peers` list and both owners' `_events` list.

GIL: released around every socket transfer and the host fold, held around
the trainer's session calls, exactly as before the split.
"""
from std.math import isfinite
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, GILReleased
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_transport_door import (
    transport_initialize,
    transport_receive,
    transport_listen,
    transport_accept,
    transport_close,
    transport_sleep,
    transport_connect,
    transport_port,
    transport_address,
    transport_timeout,
    transport_unpack,
    _transport_json,
)
from bindings.native_transport_run import transport_clock, transport_hostname
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.parallel_byte_native_door import (
    parallel_byte_call, parallel_byte_init, _parallel_array, _parallel_int, _parallel_float, _parallel_record,
    ParallelByteOwner,
)
from bindings.parallel_keys_unpack import PARALLEL_BYTE_OPN_ADMITTED_STATE, PARALLEL_BYTE_OPN_EXPORT_RAW
from bindings.generation_buffers_door import generation_array
from bindings.cross_vendor_admission_unpack import (
    cv_types_initialize,
    cv_int,
    cv_float,
    cv_bool,
    cv_text,
    cv_sequence,
    cv_buffer,
    cv_json,
    cv_state,
)
from bindings.cross_vendor_admission_pack import (
    cv_path,
)
from bindings.byte_checkpoint_native_door import byte_checkpoint_initialize, byte_checkpoint_save, byte_checkpoint_load, byte_checkpoint_raw, byte_checkpoint_validate, byte_checkpoint_file_error
from bindings.byte_checkpoint_native_run import byte_checkpoint_write, ByteCheckpointWriteFailure
from bindings.byte_checkpoint_io_unpack import ByteCheckpointIOError
from bindings.py_balanced import py_call, py_contains, py_fields, py_get, py_setattr, py_setitem, py_type
from core.sha256 import Sha256
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer
from core.shared_slot import SharedSlot
from core.parallel_entry_table import ParallelEntryKey, parallel_ref, parallel_fold_pair_f32, PARALLEL_TABLE_BASE
from bindings.cross_vendor_native_run import (
    CROSS_SCHEME_V1, CROSS_SCHEME_SLICED, CROSS_COMMIT, CROSS_DONE, CROSS_HOST_FOLD,
    CROSS_DEFECT_NONE, CROSS_DEFECT_NAME, CROSS_DEFECT_IDS, CROSS_DEFECT_ID,
    CrossFeed, CrossRates, CrossCoordinatorState, CrossWorkerState,
    cross_feed_address, cross_rate, cross_ready_wait, cross_scheme_name,
    cross_coordinator_clear, cross_coordinator_refuse, cross_coordinator_admit, cross_coordinator_step,
    cross_coordinator_done,
    cross_worker_begin, cross_worker_end, cross_worker_receive, cross_worker_hello, cross_worker_reply,
    cross_worker_host_fold, cross_worker_command,
)


struct CrossVendorBoundary(Defaultable, Movable):
    var bi: PythonObject
    var arrays: PythonObject
    var array_cls: PythonObject
    var mismatch_cls: PythonObject
    var fold_mode: Int
    var fold_vendor: Int
    var parallel_cls: PythonObject
    var feed_cls: PythonObject
    var rate_cls: PythonObject
    var vendor: PythonObject
    var worker_cls: PythonObject
    var coordinator_cls: PythonObject
    var ready_cls: PythonObject
    var argv: PythonObject
    var console_write: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.arrays = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.mismatch_cls = PythonObject(None)
        self.fold_mode = -1
        self.fold_vendor = -1
        self.parallel_cls = PythonObject(None)
        self.feed_cls = PythonObject(None)
        self.rate_cls = PythonObject(None)
        self.vendor = PythonObject(None)
        self.worker_cls = PythonObject(None)
        self.coordinator_cls = PythonObject(None)
        self.ready_cls = PythonObject(None)
        self.argv = PythonObject(None)
        self.console_write = PythonObject(None)


# V2: the fold entry is the base image's parallel table key, not a Python door.
comptime CROSS_VENDOR_BOUNDARY = SharedSlot[CrossVendorBoundary, "MojolearnCrossVendorBoundaryV2", 2]


def cross_vendor_initialize(array_cls: PythonObject, mismatch_cls: PythonObject, fold_key: PythonObject) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    c[].bi = Python.import_module("builtins")
    c[].arrays = Python.import_module("array").array
    c[].array_cls = array_cls
    c[].mismatch_cls = mismatch_cls
    # UNPACK: the base image's (tier, vendor) table key the loader recorded
    # (python/mojolearn/_native_table_key.py); None when the base binding is
    # not built, and then the first fold refuses by the table's word.
    if fold_key is not None:
        c[].fold_mode = Int(py=fold_key[0])
        c[].fold_vendor = Int(py=fold_key[1])
    cv_types_initialize(array_cls)
    byte_checkpoint_initialize(array_cls)
    c[].argv = Python.import_module("sys").argv
    c[].console_write = Python.import_module("posix").write
    transport_initialize()


def cross_vendor_bytes(value: PythonObject) raises -> PythonObject:
    return py_call(cv_buffer(value).cast, "B")


def cross_vendor_f32(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var raw = cross_vendor_bytes(value)
    if len(raw) % 4 != 0:
        raise Error("cross_vendor: gradient bytes must be a multiple of four")
    return py_call(raw.cast, "f")


def cross_vendor_fold(gradients: PythonObject, prefix: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var parts = cv_sequence(gradients, "gradients")
    var first = 0
    var source = prefix
    if source is None:
        if len(parts) == 0:
            raise Error("ordered_fold needs at least one gradient")
        source = parts[0]
        first = 1
    var store = py_call(c[].arrays, "f")
    _ = py_call(store.frombytes, py_call(cross_vendor_f32(source).cast, "B"))
    var n = len(store)
    var total = Int(py=store.buffer_info()[0])
    # Resolved once before the shard loop (lane entry-table-zero-alloc).
    var key = parallel_ref(PARALLEL_TABLE_BASE, ParallelEntryKey(c[].fold_mode, c[].fold_vendor))
    for k in range(first, len(parts)):  # driver-loop: over logical shard gradient buffers in immutable order
        # pass 3 (rank 36): the shard's received float32 view is folded in
        # place (one read-only buffer export); it was copied into a new
        # array('f') per shard first. Same values, same fold order.
        var view = cross_vendor_f32(parts[k])
        var pin = AdmittedBorrowedBuffer(view)
        try:
            var layout = pin.layout.value().copy()
            if layout.count != n:
                raise Error("ordered_fold: gradients differ in length")
            if layout.itemsize != 4 or not layout.contiguous():
                raise Error("ordered_fold: gradients must be contiguous float32")
            # Retain the selected operation's existing GPU/CPU execution route.
            # Mojo owns the immutable fold; the cached compiled entry owns its
            # arithmetic, including FTZ and the one float32 rounding per add.
            # The base image's fold_pair_f32 through its parallel table entry
            # (core/parallel_entry_table.mojo), with the GIL released.
            var py = Python()
            with GILReleased(py):
                _ = parallel_fold_pair_f32(key, total, layout.address, n)
        finally:
            pin.release()  # the export pins the received bytes through the fold above
        _ = view^
    var result = store.tobytes()
    _ = store
    return result



def cross_vendor_state_hash(state: PythonObject) raises -> PythonObject:
    """SHA-256 over a caller's state dictionary, each of the four buffers
    read in place under one read-only buffer export."""
    cv_state(state)
    var hasher = Sha256()
    for key in [String("parameters"), String("m"), String("v"), String("flags")]:  # small-loop(four state keys): one digest update per named state field
        var view = cross_vendor_bytes(state[key])
        var pin = AdmittedBorrowedBuffer(view)
        try:
            var layout = pin.layout.value().copy()
            if not layout.contiguous():
                raise Error("cross_vendor: hashed buffers must be contiguous")
            hasher.update(layout.address, layout.count * layout.itemsize)
        finally:
            pin.release()  # the export pins the hashed bytes through the digest above
        _ = view^
    return PythonObject(hasher.hexdigest())


# ------------------------------------------------------------ native owners
struct CrossReadyOwner(Movable, Writable):
    """The readiness event: one word (0 clear, 1 set) at a stable address."""
    var cell: List[Int]

    def __init__(out self):
        self.cell = List[Int](length=1, fill=0)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CrossReadyOwner(set=", self.cell[0] != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct CrossRateOwner(Movable, Writable):
    var rates: CrossRates

    def __init__(out self, var rates: CrossRates):
        self.rates = rates^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CrossRateOwner(rates=", len(self.rates.values), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct CrossFeedOwner(Movable, Writable):
    """`data` pins the admitted int32 copy that `feed.address` points into."""
    var feed: CrossFeed
    var data: PythonObject

    def __init__(out self, var feed: CrossFeed, data: PythonObject):
        self.feed = feed^
        self.data = data

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CrossFeedOwner(steps=", self.feed.steps, ", shards=", self.feed.shards, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct CrossWorkerOwner(Movable, Writable):
    """`log_path` and `checkpoint_dir` are boundary pins PACK reads."""
    var owner_id: Int
    var native: CrossWorkerState
    var log_path: PythonObject
    var checkpoint_dir: PythonObject

    def __init__(out self, owner_id: Int, var native: CrossWorkerState, log_path: PythonObject, checkpoint_dir: PythonObject):
        self.owner_id = owner_id
        self.native = native^
        self.log_path = log_path
        self.checkpoint_dir = checkpoint_dir

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CrossWorkerOwner(shards=", len(self.native.shards), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct CrossCoordinatorOwner(Movable, Writable):
    """`ready` pins the readiness event object whose record this sets."""
    var owner_id: Int
    var native: CrossCoordinatorState
    var log_path: PythonObject
    var ready: PythonObject

    def __init__(out self, owner_id: Int, var native: CrossCoordinatorState, log_path: PythonObject, ready: PythonObject):
        self.owner_id = owner_id
        self.native = native^
        self.log_path = log_path
        self.ready = ready

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CrossCoordinatorOwner(workers=", self.native.workers, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _cross_native(owner: PythonObject, cls: PythonObject, refusal: String) raises -> PythonObject:
    """The native record of an exact owner instance."""
    if py_type(owner) is not cls:
        raise Error(refusal)
    var record = py_get(py_fields(owner), "_native")
    if record is None:
        raise Error("cross_vendor: owner is not initialized")
    return record


def _cross_worker(owner: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var record = _cross_native(owner, c[].worker_cls, "cross_vendor: worker owner must be an exact Worker")
    if record.downcast_value_ptr[CrossWorkerOwner]()[].owner_id != Int(py=py_call(c[].bi.id, owner)):
        raise Error("cross_vendor: native storage does not belong to this owner")
    return record


def _cross_coordinator(owner: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var record = _cross_native(owner, c[].coordinator_cls, "cross_vendor: coordinator owner must be an exact Coordinator")
    if record.downcast_value_ptr[CrossCoordinatorOwner]()[].owner_id != Int(py=py_call(c[].bi.id, owner)):
        raise Error("cross_vendor: native storage does not belong to this owner")
    if py_get(py_fields(owner), "on_step") is not None:
        raise Error("cross_vendor: on_step callbacks are not accepted; use drain_events or log_path")
    return record


def _cross_events(owner: PythonObject) raises -> PythonObject:
    var events = py_get(py_fields(owner), "_events")
    if py_type(events) is not CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi.list:
        raise Error("cross_vendor: peer and event storage must be native lists")
    return events


def cross_vendor_call(operation: String, args: PythonObject, mut io_error: ByteCheckpointIOError) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if operation == "byte_checkpoint_initialize":
        byte_checkpoint_initialize(args[0])
    elif operation == "byte_checkpoint_save":
        return byte_checkpoint_save(args[0], args[1], io_error)
    elif operation == "byte_checkpoint_load":
        return byte_checkpoint_load(args[0], io_error)
    elif operation == "main":
        return cross_vendor_main(args[0])
    elif operation == "initialize":
        cross_vendor_initialize(args[0], args[1], args[2])
    elif operation == "worker_initialize":
        c[].parallel_cls = args[0]
        c[].feed_cls = args[1]
        c[].rate_cls = args[2]
        c[].vendor = args[3]
        c[].worker_cls = args[4]
        c[].coordinator_cls = args[5]
        c[].ready_cls = args[6]
    elif operation == "worker_init":
        return cross_vendor_worker_init(args)
    elif operation == "worker_connect":
        return cross_vendor_worker_connect(args[0])
    elif operation == "worker_run":
        return cross_vendor_worker_run(args[0], io_error)
    elif operation == "trainer_hash":
        return cross_vendor_state_hash(_cross_vendor_trainer(args[0], PARALLEL_BYTE_OPN_EXPORT_RAW))
    elif operation == "feed_init":
        return cross_vendor_feed_init(args)
    elif operation == "feed_batch":
        return cross_vendor_feed_batch(args[0], cv_int(args[1], "step"), cv_int(args[2], "shard"))
    elif operation == "rates_init":
        return cross_vendor_rates_init(args)
    elif operation == "rates_value":
        return cross_vendor_rate(args[0], cv_int(args[1], "step"))
    elif operation == "drain_events":
        if py_type(args[0]) is not c[].worker_cls and py_type(args[0]) is not c[].coordinator_cls:
            raise Error("cross_vendor: event owner must be an exact Worker or Coordinator")
        var events = args[0]._events
        py_setattr(args[0], "_events", c[].bi.list())
        return events
    elif operation == "coordinator_init":
        return cross_vendor_coordinator_init(args)
    elif operation == "coordinator_run":
        return cross_vendor_coordinator_run(args[0])
    elif operation == "coordinator_drive":
        return cross_vendor_drive(args[0])
    elif operation == "coordinator_refuse":
        cross_vendor_refuse(args[0], cv_text(args[1], "refusal reason"))
    elif operation.startswith("event_"):
        return cross_vendor_event(operation, args)
    elif operation == "fold":
        return cross_vendor_fold(args[0], args[1])
    elif operation == "pair":
        var parts = c[].bi.list()
        _ = py_call(parts.append, args[0])
        _ = py_call(parts.append, args[1])
        return cross_vendor_fold(parts, PythonObject(None))
    elif operation == "state_hash":
        return cross_vendor_state_hash(args[0])
    else:
        raise Error("cross_vendor: unknown native operation " + operation)
    return PythonObject(None)


def cross_vendor_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    var operation = String()
    var io_error = ByteCheckpointIOError()
    try:
        operation = String(args[0])
        var result = cross_vendor_call(operation, args[1], io_error)
        return result.steal_data()
    except error:
        if io_error.pending:
            return io_error^.restore()
        if String(error).startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if String(error).startswith("CrossVendorMismatch:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 2)
                return causal_set_exception(error_without_class(error, "CrossVendorMismatch"), causal_borrowed_item(context, "mismatch"))
            except nested:
                return raise_python_exception(nested, ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


# ------------------------------------------------------------ readiness event
def _cross_ready_attach(event: PythonObject) raises:
    py_setattr(event, "_native", PythonObject(alloc=CrossReadyOwner()))


def cross_vendor_event(operation: String, args: PythonObject) raises -> PythonObject:
    var event = args[0]
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(event) is not c[].ready_cls:
        raise Error("cross_vendor: readiness requires the native event owner")
    if operation == "event_init":
        _cross_ready_attach(event)
        return PythonObject(None)
    var record = _cross_native(event, c[].ready_cls, "cross_vendor: readiness requires the native event owner")
    var rec = record.downcast_value_ptr[CrossReadyOwner]()
    if operation == "event_clear":
        rec[].cell[0] = 0
    elif operation == "event_set":
        rec[].cell[0] = 1
    elif operation == "event_is_set":
        return PythonObject(rec[].cell[0] != 0)
    elif operation == "event_wait":
        # UNPACK
        var unlimited = args[1] is None
        var timeout = Float64(0) if unlimited else cv_float(args[1], "readiness timeout")
        if not unlimited and not isfinite(timeout):
            raise Error("readiness timeout must be finite or None")
        var address = Int(rec[].cell.unsafe_ptr())
        # RUN: the wait holds no Python state; the caller's argument tuple
        # keeps the event, and so the word, alive.
        var signalled = False
        var py = Python()
        with GILReleased(py):
            signalled = cross_ready_wait(address, unlimited, timeout)
        _ = record
        return PythonObject(signalled)
    _ = record
    return PythonObject(None)


# ------------------------------------------------------------ coordinator
def cross_vendor_coordinator_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var owner = args[0]
    if py_type(owner) is not c[].coordinator_cls:
        raise Error("cross_vendor: Coordinator subclasses and Python attribute hooks are not accepted")
    if args[6] is not None:
        raise Error("cross_vendor: on_step callbacks are not accepted; use drain_events or log_path")
    # UNPACK: every value leaves its Python object here, once.
    var state = CrossCoordinatorState()
    state.workers = cv_int(args[2], "workers")
    state.shards = cv_int(args[3], "logical_shards")
    if state.workers < 1 or state.shards < state.workers:
        raise Error("need 1 <= workers <= logical_shards")
    state.port = cv_int(args[1], "port")
    state.steps = cv_int(args[4], "steps")
    if state.port < 0 or state.port > 65535 or state.steps < 0:
        raise Error("coordinator requires port in [0,65535] and nonnegative steps")
    state.host = cv_text(args[5], "host")
    state.timeout = cv_float(args[7], "timeout")
    state.accept_timeout = cv_float(args[8], "accept_timeout")
    state.max_bytes = cv_int(args[9], "max_gradient_bytes")
    if state.timeout <= 0 or state.accept_timeout <= 0 or state.max_bytes < 0:
        raise Error("coordinator timeouts must be positive and byte limit nonnegative")
    state.chained = cv_bool(args[10], "chained")
    state.fold_mode = c[].fold_mode
    state.fold_vendor = c[].fold_vendor
    var log_path = PythonObject(None) if args[12] is None else cv_path(args[12])
    var ready = py_call(c[].bi.object.__new__, args[11])
    _cross_ready_attach(ready)
    var workers = state.workers
    var shards = state.shards
    var record = PythonObject(alloc=CrossCoordinatorOwner(Int(py=py_call(c[].bi.id, owner)), state^, log_path, ready))
    # PACK: the record, then the read-only mirrors.
    py_setattr(owner, "_native", record)
    py_setattr(owner, "port", args[1])
    py_setattr(owner, "n_workers", PythonObject(workers))
    py_setattr(owner, "K", PythonObject(shards))
    py_setattr(owner, "steps", args[4])
    py_setattr(owner, "host", args[5])
    py_setattr(owner, "on_step", args[6])
    py_setattr(owner, "timeout", args[7])
    py_setattr(owner, "accept_timeout", args[8])
    py_setattr(owner, "max_bytes", args[9])
    py_setattr(owner, "chained", py_call(c[].bi.bool, args[10]))
    py_setattr(owner, "peers", c[].bi.list())
    py_setattr(owner, "_events", c[].bi.list())
    py_setattr(owner, "log_path", log_path)
    py_setattr(owner, "bound_port", PythonObject(None))
    py_setattr(owner, "ready", ready)
    return PythonObject(None)


def _cross_peers(owner: PythonObject) raises -> PythonObject:
    var peers = py_get(py_fields(owner), "peers")
    if py_type(peers) is not CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi.list:
        raise Error("cross_vendor: peer and event storage must be native lists")
    return peers


def _cross_unpack_peers(peers: PythonObject, mut state: CrossCoordinatorState, declarations: Bool) raises:
    """UNPACK the connected group: each peer's descriptor and timeout and,
    with `declarations`, what it declared in its hello. A value of the wrong
    Python type is recorded as that peer's defect, so RUN refuses it in the
    same order as before, after telling every worker."""
    var bi = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi
    cross_coordinator_clear(state)
    state.n_total = -1
    state.completed = -1
    for i in range(len(peers)):  # driver-loop: over worker protocol and shard ownership declarations
        var peer = peers[i]
        var fd = -1
        var timeout = Float64(-1)
        try:
            var pair = transport_unpack(py_get(peer, "sock"))
            fd = pair[0]
            timeout = pair[1]
        except:
            pass
        state.fds.append(fd)
        state.timeouts.append(timeout)
        if not declarations:
            continue
        var defect = CROSS_DEFECT_NONE
        var name = py_get(peer, "name")
        if py_type(name) is bi.str:
            state.names.append(String(name))
        else:
            state.names.append(String(""))
            defect = CROSS_DEFECT_NAME
        state.protocols.append(String(py_get(peer, "protocol")))
        var ids = py_get(peer, "shards")
        if py_type(ids) is not bi.list or len(ids) == 0:
            if defect == CROSS_DEFECT_NONE:
                defect = CROSS_DEFECT_IDS
        else:
            for k in range(len(ids)):  # driver-loop: over one worker's declared shard IDs
                if py_type(ids[k]) is not bi.int:
                    if defect == CROSS_DEFECT_NONE:
                        defect = CROSS_DEFECT_ID
                    break
                state.ids.append(Int(py=ids[k]))
        state.starts.append(len(state.ids))
        var same = 0
        if Bool(py=py_get(peer, "state") == py_get(peers[0], "state")):
            same |= 1
        if Bool(py=py_get(peer, "completed") == py_get(peers[0], "completed")):
            same |= 2
        if Bool(py=py_get(peer, "n_total") == py_get(peers[0], "n_total")):
            same |= 4
        state.same.append(same)
        state.defects.append(defect)
        state.peer_chained.append(Bool(py=py_get(peer, "chained", PythonObject(False))))
    if declarations and len(peers) != 0:
        var total = py_get(peers[0], "n_total")
        if py_type(total) is bi.int:
            state.n_total = Int(py=total)
        var completed = py_get(peers[0], "completed")
        if py_type(completed) is bi.int:
            state.completed = Int(py=completed)


def cross_vendor_refuse(owner: PythonObject, reason: String) raises:
    var record = _cross_coordinator(owner)
    var rec = record.downcast_value_ptr[CrossCoordinatorOwner]()
    _cross_unpack_peers(_cross_peers(owner), rec[].native, False)
    var py = Python()
    with GILReleased(py):
        cross_coordinator_refuse(rec[].native, reason)
    _ = record


def _cross_row(state: CrossCoordinatorState, peers: PythonObject, step: Int) raises -> PythonObject:
    """PACK one agreed step."""
    var bi = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi
    var row = bi.dict()
    py_setitem(row, "step", PythonObject(step + 1))
    var losses = bi.list()
    for k in range(len(state.row_losses)):  # driver-loop: over logical shard losses in fixed order
        _ = py_call(losses.append, PythonObject(state.row_losses[k]))
    py_setitem(row, "losses", losses)
    py_setitem(row, "state", PythonObject(state.row_agreed))
    var hashes = bi.dict()
    var vendors = bi.dict()
    for i in range(len(state.row_states)):  # driver-loop: over committed worker state witnesses
        var peer = peers[i]
        py_setitem(hashes, peer["name"], PythonObject(state.row_states[i]))
        py_setitem(vendors, peer["name"], py_get(peer, "vendor"))
    py_setitem(row, "workers", hashes)
    py_setitem(row, "vendors", vendors)
    py_setitem(row, "total_sha256", PythonObject(state.row_total_hash))
    py_setitem(row, "protocol", PythonObject("chained" if state.chained else "gathered"))
    var timing = bi.dict()
    for i in range(len(state.timing_keys)):  # small-loop(timing marks of one step)
        py_setitem(timing, state.timing_keys[i], py_call(bi.round, PythonObject(state.timing_values[i]), 3))
    py_setitem(row, "timing", timing)
    return row


def cross_vendor_drive(owner: PythonObject) raises -> PythonObject:
    var record = _cross_coordinator(owner)
    var rec = record.downcast_value_ptr[CrossCoordinatorOwner]()
    var bi = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi
    # UNPACK
    var peers = _cross_peers(owner)
    var events = _cross_events(owner)
    _cross_unpack_peers(peers, rec[].native, True)
    # RUN: the group is admitted, or every worker is told why not.
    var py = Python()
    with GILReleased(py):
        cross_coordinator_admit(rec[].native)
    if rec[].native.completed < 0:
        raise Error("cross_vendor: worker hello requires an integer completed step")
    if rec[].native.chained:
        # PACK: `peers` takes the order the steps use (by first shard).
        var ordered = bi.list()
        for i in range(len(peers)):  # driver-loop: over worker blocks in immutable shard order
            _ = py_call(ordered.append, peers[rec[].native.order[i]])
        for i in range(len(peers)):  # driver-loop: over worker blocks in immutable shard order
            py_setitem(peers, i, ordered[i])
    var rows = bi.list()
    for step in range(rec[].native.completed, rec[].native.steps):  # driver-loop: over requested lockstep training rounds
        # RUN: one whole step, no Python object, the GIL released.
        with GILReleased(py):
            cross_coordinator_step(rec[].native, step)
        # PACK
        var row = _cross_row(rec[].native, peers, step)
        _ = py_call(rows.append, row)
        _ = py_call(events.append, row)
        cross_vendor_log(rec[].log_path, row)
    with GILReleased(py):
        cross_coordinator_done(rec[].native)
    # The peers' socket objects own the descriptors RUN used.
    _ = peers
    _ = record
    return rows


def _cross_vendor_close_peers(peers: PythonObject) raises:
    for i in range(len(peers)):  # driver-loop: over native socket cleanup
        try:
            transport_close(peers[i]["sock"])
        except:
            pass


def cross_vendor_coordinator_run(owner: PythonObject) raises -> PythonObject:
    var record = _cross_coordinator(owner)
    var rec = record.downcast_value_ptr[CrossCoordinatorOwner]()
    var bi = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi
    var peers = _cross_peers(owner)
    var server = transport_listen(rec[].native.host, rec[].native.port, rec[].native.accept_timeout)
    py_setattr(owner, "bound_port", transport_port(server))
    var ready = _cross_native(rec[].ready, py_type(rec[].ready), "cross_vendor: readiness requires the native event owner")
    ready.downcast_value_ptr[CrossReadyOwner]()[].cell[0] = 1
    var rows = PythonObject(None)
    try:
        # UNPACK, once per connection: the worker's hello is admitted through
        # the boundary reader into `peers`, as it always was.
        while len(peers) < rec[].native.workers:  # driver-loop: over admitted worker connections
            var socket = transport_accept(server, rec[].native.timeout)
            try:
                var response = transport_receive(socket, 0)
                var peer = py_call(bi.dict, response[0])
                py_setitem(peer, "sock", socket)
                py_setitem(peer, "where", transport_address(socket, True))
                _ = py_call(peers.append, peer)
            except error:
                transport_close(socket)
                raise error
        rows = cross_vendor_drive(owner)
    except error:
        _cross_vendor_close_peers(peers)
        transport_close(server)
        raise error
    _cross_vendor_close_peers(peers)
    transport_close(server)
    _ = record
    return rows


# ------------------------------------------------------------ feed and rates
def cross_vendor_feed_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var owner = args[0]
    if py_type(owner) is not c[].feed_cls:
        raise Error("BatchFeed subclasses and Python attribute hooks are not accepted")
    var value = args[1]
    var view = cv_buffer(value)
    var shape = value.shape if py_type(value) is c[].array_cls else view.shape
    if len(shape) != 4:
        raise Error("BatchFeed requires int32[steps,logical_shards,batch,length+1]")
    var dims = List[Int]()
    for i in range(4):  # small-loop(four batch-feed dimensions): steps, shards, batch, length + 1
        dims.append(_parallel_int(shape[i], 0 if i == 0 else 1, 2147483647, "batch feed dimension", False))
    _ = cv_int(args[2], "start_step")
    var feed = CrossFeed()
    feed.start = _parallel_int(args[2], 0, 999999, "start_step", False)
    feed.steps = dims[0]
    feed.shards = dims[1]
    feed.batch = dims[2]
    feed.width = dims[3]
    var data = _parallel_array(value, dims, True)
    feed.address = Int(py=data._addr)
    py_setattr(owner, "_native", PythonObject(alloc=CrossFeedOwner(feed^, data)))
    return PythonObject(None)


def cross_vendor_feed_batch(owner: PythonObject, step: Int, shard: Int) raises -> PythonObject:
    var record = _cross_native(owner, CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].feed_cls,
                               "cross_vendor: batch owner must be an exact BatchFeed")
    var rec = record.downcast_value_ptr[CrossFeedOwner]()
    var address = cross_feed_address(rec[].feed, step, shard)
    var batch = rec[].feed.batch
    var width = rec[].feed.width
    # PACK: the caller gets an owned copy; a Worker reads the block in place.
    var output = generation_array([batch, width], "i")
    var source = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=address)
    var target = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
    unsafe_memcpy(dest=target, src=source, count=batch * width)
    _ = record
    return output


def cross_vendor_rates_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(args[0]) is not c[].rate_cls:
        raise Error("LearningRateTable subclasses and Python attribute hooks are not accepted")
    var values = cv_sequence(args[1], "rates")
    if len(values) == 0:
        raise Error("LearningRateTable requires at least one rate")
    var rates = CrossRates()
    for i in range(len(values)):  # driver-loop: over the declared learning-rate schedule
        _ = cv_float(values[i], "learning rate")
        var value = _parallel_float(values[i], "learning rate")
        if value <= 0:
            raise Error("learning rates must be positive")
        rates.values.append(value)
    _ = cv_int(args[2], "start_step")
    rates.start = _parallel_int(args[2], 0, 999999, "start_step", False)
    py_setattr(args[0], "_native", PythonObject(alloc=CrossRateOwner(rates^)))
    return PythonObject(None)


def cross_vendor_rate(table: PythonObject, step: Int) raises -> PythonObject:
    var record = _cross_native(table, CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].rate_cls,
                               "cross_vendor: rate owner must be an exact LearningRateTable")
    var value = cross_rate(record.downcast_value_ptr[CrossRateOwner]()[].rates, step)
    _ = record
    return PythonObject(Float64(value))


# ------------------------------------------------------------ worker
def _cross_vendor_trainer(trainer: PythonObject, operation: Int) raises -> PythonObject:
    """A boundary read of the exact trainer: its raw state export (which
    opens its session) or a detached copy of its admitted metadata."""
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if py_type(trainer) is not c[].parallel_cls:
        raise Error("cross_vendor: trainer must be an exact ParallelByteLanguageModelTrainer; Python trainer protocols are not accepted")
    var args = c[].bi.list()
    _ = py_call(args.append, trainer)
    if operation == PARALLEL_BYTE_OPN_EXPORT_RAW:
        _ = py_call(args.append, 0)
        _ = py_call(args.append, PythonObject(None))
    return parallel_byte_call(operation, PythonObject(None), args)


def cross_vendor_log(path: PythonObject, row: PythonObject) raises:
    if path is None:
        return
    var bi = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi
    var stream = py_call(bi.open, cv_path(path), "ab")
    try:
        var encoded = py_call(PythonObject(_transport_json(row) + "\n").encode, "ascii")
        if Int(py=py_call(stream.write, encoded)) != len(encoded):
            raise Error("cross_vendor: short event log write")
        stream.close()
    except error:
        stream.close()
        raise error


def cross_vendor_worker_arguments(trainer: PythonObject, batches: PythonObject, rates: PythonObject, callback: PythonObject) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if trainer is not None and py_type(trainer) is not c[].parallel_cls:
        raise Error("cross_vendor: trainer must be an exact ParallelByteLanguageModelTrainer; Python trainer protocols are not accepted")
    if py_type(batches) is not c[].feed_cls:
        raise Error("cross_vendor: batches must be an exact BatchFeed; Python batch callbacks are not accepted")
    if rates is not None and py_type(rates) is not c[].rate_cls:
        raise Error("cross_vendor: lr_for_step must be an exact LearningRateTable; Python schedules are not accepted")
    if callback is not None:
        raise Error("cross_vendor: on_commit callbacks are not accepted; use drain_events or native checkpoint configuration")


def cross_vendor_commit_configuration(mut state: CrossWorkerState, directory: PythonObject, steps: PythonObject,
                                      expected: PythonObject, retain: PythonObject, scheme: PythonObject) raises -> PythonObject:
    """UNPACK the commit policy into `state`; returns the admitted checkpoint
    directory (None when there is none)."""
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var ids = cv_sequence(steps, "checkpoint_steps")
    for i in range(len(ids)):  # driver-loop: over explicit native checkpoint steps
        var step = cv_int(ids[i], "checkpoint step")
        if step < 1:
            raise Error("cross_vendor: checkpoint steps must be positive")
        var seen = False
        for j in range(len(state.checkpoint_steps)):  # small-loop(checkpoint steps admitted so far)
            seen = seen or state.checkpoint_steps[j] == step
        if not seen:
            state.checkpoint_steps.append(step)
    if len(state.checkpoint_steps) != 0 and directory is None:
        raise Error("cross_vendor: checkpoint_steps requires checkpoint_dir")
    var checkpoint_dir = PythonObject(None) if directory is None else cv_path(directory)
    state.retain = cv_bool(retain, "retain_gradients")
    var hash_scheme = cv_text(scheme, "hash_scheme")
    if hash_scheme != "sha256.v1" and hash_scheme != "sliced-sha256-8.v2":
        raise Error("cross_vendor: unsupported native hash_scheme")
    state.scheme = CROSS_SCHEME_V1 if hash_scheme == "sha256.v1" else CROSS_SCHEME_SLICED
    var rows = cv_sequence(expected, "expected_commits")
    var registered = List[Int]()
    for i in range(len(rows)):  # driver-loop: over explicit expected committed hashes
        var row = rows[i]
        cv_json(row)
        if py_type(row) is not c[].bi.dict or not py_contains(row, "step"):
            raise Error("cross_vendor: expected commits require step and state or total_sha256")
        var step = cv_int(row["step"], "expected step")
        var repeated = False
        for j in range(len(registered)):  # small-loop(expected steps admitted so far)
            repeated = repeated or registered[j] == step
        if step < 1 or repeated:
            raise Error("cross_vendor: expected commit steps must be unique and positive")
        # Keys ascending: the first differing field is reported in that order.
        var keys = py_call(c[].bi.sorted, py_call(c[].bi.list, row))
        if len(keys) < 2:
            raise Error("cross_vendor: expected commits require state or total_sha256")
        for j in range(len(keys)):  # small-loop(expected commit fields): the named keys of one configured row
            var key = String(keys[j])
            if key == "step":
                continue
            var value = String("")
            if key == "losses_f32_hex":
                var losses = cv_sequence(row[keys[j]], "expected loss words")
                for k in range(len(losses)):  # driver-loop: over logical shard loss words of one configured row
                    var word = cv_text(losses[k], "expected loss word")
                    if len(PythonObject(word)) != 8:
                        raise Error("cross_vendor: expected loss words must have eight hexadecimal characters")
                    if k != 0:
                        value += ","
                    value += word
            elif key == "lr_f32_hex":
                value = cv_text(row[keys[j]], "expected learning-rate word")
                if len(PythonObject(value)) != 8:
                    raise Error("cross_vendor: expected learning-rate word must have eight hexadecimal characters")
            elif key == "hash_scheme":
                value = cv_text(row[keys[j]], "expected hash scheme")
            else:
                if key != "state" and key != "total_sha256" and key != "state_sha256" and key != "gradient_sha256":
                    raise Error("cross_vendor: unknown expected commit field")
                value = cv_text(row[keys[j]], "expected hash")
                if len(row[keys[j]]) != 64:
                    raise Error("cross_vendor: expected hashes must have 64 hexadecimal characters")
                var raw = py_call(row[keys[j]].encode, "ascii")
                for k in range(64):  # small-loop(64 hex characters): one configured digest string
                    var ch = Int(py=raw[k])
                    if not (ch >= 48 and ch <= 57) and not (ch >= 97 and ch <= 102):
                        raise Error("cross_vendor: expected hashes must have 64 hexadecimal characters")
            state.expected_step.append(step)
            state.expected_key.append(key)
            state.expected_value.append(value)
        registered.append(step)
    return checkpoint_dir


def cross_vendor_worker_init(args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var owner = args[0]
    if py_type(owner) is not c[].worker_cls:
        raise Error("cross_vendor: Worker subclasses and Python attribute hooks are not accepted")
    var trainer = args[7]
    cross_vendor_worker_arguments(trainer, args[3], args[11], args[12])
    # UNPACK: every value leaves its Python object here, once.
    var state = CrossWorkerState()
    var shards = cv_sequence(args[2], "shards")
    for i in range(len(shards)):  # driver-loop: over the worker's logical shard ids
        var shard = cv_int(shards[i], "shard")
        if shard < 0:
            raise Error("cross_vendor: shards must be nonnegative")
        # Kept ascending: a stable insertion of one id.
        state.shards.append(shard)
        var at = len(state.shards) - 1
        while at > 0 and state.shards[at - 1] > state.shards[at]:  # small-loop(ids admitted so far)
            var moved = state.shards[at - 1]
            state.shards[at - 1] = state.shards[at]
            state.shards[at] = moved
            at -= 1
    var address = cv_sequence(args[4], "address")
    if len(address) != 2:
        raise Error("worker address requires host and port in [1,65535]")
    state.host = cv_text(address[0], "worker host")
    state.port = cv_int(address[1], "worker port")
    if state.port < 1 or state.port > 65535:
        raise Error("worker address requires host and port in [1,65535]")
    if args[5] is not None:
        _ = cv_text(args[5], "worker name")
    state.connect_timeout = cv_float(args[8], "connect_timeout")
    state.timeout = cv_float(args[9], "timeout")
    if state.connect_timeout <= 0 or state.timeout <= 0:
        raise Error("worker timeouts must be positive")
    state.chained = cv_bool(args[10], "chained")
    _ = cv_int(args[6], "device")
    # Admit every optional policy before retaining/copying model state. Invalid
    # callback replacements must not construct even a lazy trainer owner.
    var log_path = PythonObject(None) if args[13] is None else cv_path(args[13])
    var checkpoint_dir = cross_vendor_commit_configuration(state, args[14], args[15], args[16], args[17], args[18])
    if trainer is None:
        cv_state(args[1])
        trainer = py_call(c[].bi.object.__new__, c[].parallel_cls)
        var device = c[].bi.list()
        _ = py_call(device.append, args[6])
        _ = parallel_byte_init(trainer, args[1], device, PythonObject(1), PythonObject(False))
    var name = args[5] if Bool(py=args[5]) else PythonObject(transport_hostname())
    state.name = String(name)
    state.vendor = String(c[].vendor)
    state.fold_mode = c[].fold_mode
    state.fold_vendor = c[].fold_vendor
    var ids = c[].bi.list()
    for i in range(len(state.shards)):  # driver-loop: over the worker's logical shard ids
        _ = py_call(ids.append, PythonObject(state.shards[i]))
    var retain = state.retain
    var record = PythonObject(alloc=CrossWorkerOwner(Int(py=py_call(c[].bi.id, owner)), state^, log_path, checkpoint_dir))
    # PACK: the record, then the read-only mirrors.
    py_setattr(owner, "_native", record)
    py_setattr(owner, "log_path", log_path)
    py_setattr(owner, "checkpoint_dir", checkpoint_dir)
    py_setattr(owner, "retain_gradients", PythonObject(retain))
    py_setattr(owner, "hash_scheme", args[18])
    py_setattr(owner, "trainer", trainer)
    py_setattr(owner, "shards", ids)
    py_setattr(owner, "batches", args[3])
    py_setattr(owner, "address", py_call(c[].bi.tuple, address))
    py_setattr(owner, "name", name)
    py_setattr(owner, "connect_timeout", args[8])
    py_setattr(owner, "timeout", args[9])
    py_setattr(owner, "chained", py_call(c[].bi.bool, args[10]))
    py_setattr(owner, "lr_for_step", args[11])
    py_setattr(owner, "on_commit", args[12])
    py_setattr(owner, "_events", c[].bi.list())
    return PythonObject(None)


def _cross_collaborators(owner: PythonObject) raises -> PythonObject:
    """UNPACK at each run, as before: the trainer, the feed, the rate table
    and the retired callback slot are re-admitted by exact class."""
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var fields = py_fields(owner)
    var found = c[].bi.list()
    _ = py_call(found.append, py_get(fields, "trainer"))
    _ = py_call(found.append, py_get(fields, "batches"))
    _ = py_call(found.append, py_get(fields, "lr_for_step"))
    cross_vendor_worker_arguments(found[0], found[1], found[2], py_get(fields, "on_commit"))
    return found


def cross_vendor_worker_connect(owner: PythonObject) raises -> PythonObject:
    var record = _cross_worker(owner)
    var rec = record.downcast_value_ptr[CrossWorkerOwner]()
    _ = _cross_collaborators(owner)
    var deadline = transport_clock() + rec[].native.connect_timeout
    while True:  # driver-loop: over native connection retries within the configured deadline
        try:
            var socket = transport_connect(rec[].native.host, rec[].native.port, 30.0)
            transport_timeout(socket, rec[].native.timeout)
            return socket
        except error:
            if String(error).startswith("InterruptedError:"):
                raise error
            if transport_clock() > deadline:
                raise error
            transport_sleep(2.0)


def cross_vendor_commit(record: PythonObject, trainer: PythonObject, events: PythonObject,
                        witness: PythonObject, mut io_error: ByteCheckpointIOError) raises:
    """PACK the step RUN just committed: the event, the configured
    expectation's refusal, the checkpoint and the log line."""
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var rec = record.downcast_value_ptr[CrossWorkerOwner]()
    var n = rec[].native.n
    var step = rec[].native.commit_step
    var event = bi.dict()
    py_setitem(event, "step", PythonObject(step))
    py_setitem(event, "state", PythonObject(rec[].native.commit_state))
    py_setitem(event, "total_sha256", PythonObject(rec[].native.commit_total_hash) if rec[].native.commit_has_total_hash else PythonObject(None))
    var losses = bi.list()
    for i in range(len(rec[].native.commit_losses)):  # driver-loop: over logical shard loss metadata
        _ = py_call(losses.append, PythonObject(rec[].native.commit_losses[i]))
    py_setitem(event, "losses", losses)
    if rec[].native.retain:
        # One copy of the applied total, only when the caller asked to keep it.
        var kept = generation_array([n])
        if n > 0:
            unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=kept._addr)),
                          src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=rec[].native.applied_address), count=n * 4)
        py_setitem(event, "total_bytes", py_call(kept._mv.cast, "B").tobytes())
    py_setitem(event, "hash_scheme", PythonObject(cross_scheme_name(rec[].native.scheme)))
    py_setitem(event, "state_sha256", PythonObject(rec[].native.commit_state_sha))
    py_setitem(event, "gradient_sha256", PythonObject(rec[].native.commit_gradient_sha) if rec[].native.commit_has_gradient_sha else PythonObject(None))
    py_setitem(event, "lr_f32_hex", PythonObject(rec[].native.commit_lr_word))
    var words = bi.list()
    for i in range(len(rec[].native.commit_loss_words)):  # driver-loop: over logical shard loss metadata
        _ = py_call(words.append, PythonObject(rec[].native.commit_loss_words[i]))
    py_setitem(event, "losses_f32_hex", words)
    py_setitem(event, "hash_seconds", PythonObject(rec[].native.commit_hash_seconds))
    if rec[].native.commit_mismatch.byte_length() != 0:
        var key = rec[].native.commit_mismatch
        py_setitem(event, "expected_mismatch", PythonObject(key))
        _ = py_call(events.append, event)
        cross_vendor_log(rec[].log_path, row_without_gradient(event))
        raise Error("CrossVendorMismatch: committed " + key + " differs from the configured expectation at step " + String(step))
    if rec[].native.commit_checkpoint:
        var checkpoint_start = transport_clock()
        # UNPACK: the archive's metadata text from the trainer's admitted
        # state (the four arrays are the witness buffers RUN just wrote).
        var metadata = py_call(bi.dict, _cross_vendor_trainer(trainer, PARALLEL_BYTE_OPN_ADMITTED_STATE))
        for key in [String("parameters"), String("m"), String("v"), String("flags")]:  # small-loop(four state keys)
            _ = py_call(metadata.pop, PythonObject(key), PythonObject(None))
        # The archive's own admission, as `byte_checkpoint_save` ran it at
        # every commit checkpoint: the exact field set, the schema, the
        # `identical` mode and base profile, the parameter registry, the step
        # words and the four arrays' kinds and sizes against that registry.
        # The trainer's admission does not cover it (it admits every
        # installed mode and its suffixed profile), so a state the archive
        # refuses is refused here, before any file is opened. The digest
        # field is the writer's own 64 hex digits (`byte_checkpoint_write`
        # hashes the arrays once, on four host tasks); the descriptor carries
        # a well-formed stand-in so the arrays are not hashed twice.
        var entries = bi.list()
        var kinds = [String("parameters"), String("m"), String("v"), String("flags")]
        for i in range(4):  # small-loop(the four checkpoint arrays)
            var raw = byte_checkpoint_raw(witness[kinds[i]], i == 3)
            var descriptor = bi.dict()
            py_setitem(descriptor, "name", PythonObject(kinds[i]))
            py_setitem(descriptor, "dtype", PythonObject("<i4" if i == 3 else "<f4"))
            var dims = bi.list()
            _ = py_call(dims.append, len(raw) // 4)
            py_setitem(descriptor, "shape", dims)
            py_setitem(descriptor, "nbytes", PythonObject(len(raw)))
            py_setitem(descriptor, "sha256", PythonObject("0000000000000000000000000000000000000000000000000000000000000000"))
            _ = py_call(entries.append, descriptor)
        _ = byte_checkpoint_validate(metadata, entries)
        var text = _transport_json(metadata)
        var filename = PythonObject("ckpt_") + py_call(PythonObject(String(step)).zfill, 8) + PythonObject(".blm")
        var path = py_call(rec[].checkpoint_dir.rstrip, "/") + PythonObject("/") + filename
        var target = String(cv_path(path))
        var addresses = rec[].native.snapshot.copy()
        var counts: List[Int] = [n, n, n, rec[].native.nt]
        var failure = ByteCheckpointWriteFailure()
        var digest = String("")
        # RUN: the typed archive write, the GIL released.
        var py = Python()
        try:
            with GILReleased(py):
                digest = byte_checkpoint_write(target, Int(text.unsafe_ptr()), text.byte_length(), addresses, counts, failure)
        except error:
            if failure.number != 0:
                byte_checkpoint_file_error(io_error, failure)
            raise error
        _ = text.byte_length()
        # PACK
        py_setitem(event, "checkpoint_path", path)
        py_setitem(event, "checkpoint_sha256", PythonObject(digest))
        py_setitem(event, "checkpoint_seconds", PythonObject(transport_clock() - checkpoint_start))
    py_setitem(event, "seconds", PythonObject(transport_clock() - rec[].native.step_started))
    _ = py_call(events.append, event)
    cross_vendor_log(rec[].log_path, row_without_gradient(event))


def row_without_gradient(row: PythonObject) raises -> PythonObject:
    var result = py_call(CROSS_VENDOR_BOUNDARY.get_or_create_ptr()[].bi.dict, row)
    _ = py_call(result.pop, "total_bytes", PythonObject(None))
    return result


def _cross_worker_serve(record: PythonObject, trainer_record: PythonObject, trainer: PythonObject,
                        events: PythonObject, pins: PythonObject, mut io_error: ByteCheckpointIOError) raises:
    """The command loop of one connected run. `pins` holds the witness
    arrays, the feed and the rate table whose storage RUN reads by address."""
    var rec = record.downcast_value_ptr[CrossWorkerOwner]()
    var session = trainer_record.downcast_value_ptr[ParallelByteOwner]()
    var py = Python()
    var first = True
    while True:  # driver-loop: over native coordinator commands until completion
        if first:
            # The hello already received the first command.
            first = False
        else:
            with GILReleased(py):
                cross_worker_receive(rec[].native)
        # RUN: typed command handling on the trainer's session (GIL held, as
        # every session call was).
        var code = cross_worker_command(rec[].native, session[].native)
        if code == CROSS_DONE:
            break
        if code == CROSS_HOST_FOLD:
            with GILReleased(py):
                cross_worker_host_fold(rec[].native)
        elif code == CROSS_COMMIT:
            cross_vendor_commit(record, trainer, events, pins[0], io_error)
        with GILReleased(py):
            cross_worker_reply(rec[].native)
    _ = len(pins)


def cross_vendor_worker_run(owner: PythonObject, mut io_error: ByteCheckpointIOError) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var record = _cross_worker(owner)
    var rec = record.downcast_value_ptr[CrossWorkerOwner]()
    # UNPACK
    var found = _cross_collaborators(owner)
    var trainer = found[0]
    var events = py_get(py_fields(owner), "_events")
    if py_type(events) is not c[].bi.list:
        raise Error("cross_vendor: policy and event storage must be native containers")
    var feed = _cross_native(found[1], c[].feed_cls, "cross_vendor: batches must be an exact BatchFeed; Python batch callbacks are not accepted")
    var held = feed.downcast_value_ptr[CrossFeedOwner]()
    rec[].native.feed.address = held[].feed.address
    rec[].native.feed.steps = held[].feed.steps
    rec[].native.feed.shards = held[].feed.shards
    rec[].native.feed.batch = held[].feed.batch
    rec[].native.feed.width = held[].feed.width
    rec[].native.feed.start = held[].feed.start
    rec[].native.has_rates = found[2] is not None
    if rec[].native.has_rates:
        var table = _cross_native(found[2], c[].rate_cls, "cross_vendor: lr_for_step must be an exact LearningRateTable; Python schedules are not accepted")
        rec[].native.rates.values = table.downcast_value_ptr[CrossRateOwner]()[].rates.values.copy()
        rec[].native.rates.start = table.downcast_value_ptr[CrossRateOwner]()[].rates.start
    # The raw export opens the trainer's session and makes the four witness
    # arrays every later step is exported into.
    var state = _cross_vendor_trainer(trainer, PARALLEL_BYTE_OPN_EXPORT_RAW)
    var trainer_record = _parallel_record(trainer)
    var session = trainer_record.downcast_value_ptr[ParallelByteOwner]()
    var snapshot = List[Int]()
    for key in [String("parameters"), String("m"), String("v"), String("flags")]:  # small-loop(four state keys)
        snapshot.append(Int(py=state[key]._addr))
    cross_worker_begin(rec[].native, session[].native.n_total, session[].native.n_tensors, snapshot^)
    var pins = c[].bi.list()
    _ = py_call(pins.append, state)
    _ = py_call(pins.append, feed)
    _ = py_call(pins.append, found)
    var py = Python()
    var deadline = transport_clock() + rec[].native.connect_timeout
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546): the loop leaves only through a connected attempt.
    var opened = Optional[PythonObject]()
    while True:  # driver-loop: over initial tunnel handshakes within the retry deadline
        var attempt = cross_vendor_worker_connect(owner)
        try:
            var pair = transport_unpack(attempt)
            rec[].native.fd = pair[0]
            rec[].native.socket_timeout = pair[1]
            var completed = session[].native.completed
            with GILReleased(py):
                cross_worker_hello(rec[].native, completed)
            opened = attempt
            break
        except error:
            transport_close(attempt)
            if String(error).startswith("InterruptedError:"):
                raise error
            if transport_clock() > deadline:
                raise error
            transport_sleep(5.0)
    var socket = opened.value()
    try:
        _cross_worker_serve(record, trainer_record, trainer, events, pins, io_error)
    except error:
        transport_close(socket)
        cross_worker_end(rec[].native)
        raise error
    transport_close(socket)
    var committed = rec[].native.committed
    cross_worker_end(rec[].native)
    _ = len(pins)
    _ = trainer_record
    _ = record
    return PythonObject(committed)


def cross_vendor_console(text: String) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var raw = py_call(PythonObject(text + "\n").encode, "utf-8")
    var sent = 0
    while sent < len(raw):  # driver-loop: over partial native console writes
        var n = Int(py=py_call(c[].console_write, 1, raw[py_call(c[].bi.slice, sent, len(raw))]))
        if n < 1:
            raise Error("cross_vendor: short console write")
        sent += n


def cross_vendor_main(argv: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var values = cv_sequence(c[].argv[py_call(c[].bi.slice, 1, None)] if argv is None else argv, "arguments")
    var usage = "usage: python -m mojolearn.cross_vendor coordinator --workers N --shards K --steps N [--host HOST] [--port PORT] [--log PATH] [--chained]"
    if len(values) == 0:
        raise Error(usage)
    if cv_text(values[0], "command") == "--help" or String(values[0]) == "-h":
        cross_vendor_console(usage)
        return PythonObject(0)
    if String(values[0]) != "coordinator":
        raise Error(usage)
    var host = PythonObject("0.0.0.0")
    var port = PythonObject(7777)
    # Optional, not PythonObject(None) sentinels: an option not given is absent (modular/modular#7546).
    var workers = Optional[PythonObject]()
    var shards = Optional[PythonObject]()
    var steps = Optional[PythonObject]()
    var path = Optional[PythonObject]()
    var chained = PythonObject(False)
    var at = 1
    while at < len(values):  # driver-loop: over native command-line options
        var option = cv_text(values[at], "option")
        at += 1
        if option == "--help" or option == "-h":
            cross_vendor_console(usage)
            return PythonObject(0)
        if option == "--chained":
            chained = PythonObject(True)
            continue
        if at == len(values):
            raise Error("missing value for " + option)
        var value = values[at]
        _ = cv_text(value, "option value")
        at += 1
        if option == "--host":
            host = value
        elif option == "--log":
            path = value
        elif option == "--port":
            port = py_call(c[].bi.int, value)
        elif option == "--workers":
            workers = py_call(c[].bi.int, value)
        elif option == "--shards":
            shards = py_call(c[].bi.int, value)
        elif option == "--steps":
            steps = py_call(c[].bi.int, value)
        else:
            raise Error("unknown coordinator option " + option)
    if not workers or not shards or not steps:
        raise Error("coordinator requires --workers, --shards and --steps")
    var owner = py_call(c[].bi.object.__new__, c[].coordinator_cls)
    var args = c[].bi.list()
    _ = py_call(args.append, owner)
    _ = py_call(args.append, port)
    _ = py_call(args.append, workers.value())
    _ = py_call(args.append, shards.value())
    _ = py_call(args.append, steps.value())
    _ = py_call(args.append, host)
    _ = py_call(args.append, PythonObject(None))
    _ = py_call(args.append, 3600.0)
    _ = py_call(args.append, 3600.0)
    _ = py_call(args.append, 2147483648)
    _ = py_call(args.append, chained)
    _ = py_call(args.append, c[].ready_cls)
    if path:
        _ = py_call(args.append, path.value())
    else:
        _ = py_call(args.append, PythonObject(None))
    _ = cross_vendor_coordinator_init(args)
    var rows = cross_vendor_coordinator_run(owner)
    cross_vendor_console("AGREED on " + String(len(rows)) + " steps across " + String(workers.value()) + " workers")
    return PythonObject(0)
