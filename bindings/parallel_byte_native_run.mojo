# SPDX-License-Identifier: Apache-2.0
"""RUN of the byte-LM multi-device trainer door: typed values only.

`ParallelByteSession` is one trainer's native state (the schedule, the open
session handle, the admitted shape words, the optimizer scalars and the
committed step). Every operation below drives the byte-LM binding's schedule
descriptor table through its entry and integer descriptors; no Python object
is read, built or called here. The boundary door
(`bindings/parallel_byte_native_door.mojo`) admits and pins the buffers
(UNPACK), calls one function of this file, and builds the public result
(PACK). A native caller that already holds addresses (the cross-vendor worker,
the corpus door) calls these functions directly.

Exclusion: the schedule session is not reentrant. The door calls with the GIL
held, which is the session's exclusion lock; a caller that releases the GIL
owns the exclusion itself.

Not split across cores: one call is one descriptor call into the GPU session,
which runs its own devices in parallel.
"""
from std.memory import bitcast
from core.neural_descriptor_run import neural_descriptor_record_execute


# The byte-LM schedule descriptor ABI (bindings/_mojolearn_byte_lm.mojo).
comptime PARALLEL_SCHEDULE_ABI_NAME = "mojolearn.training.byte_lm_schedule.v1"
comptime PARALLEL_SCHEDULE_OPERATIONS = 15
comptime PARALLEL_REPLICA = 0
comptime PARALLEL_MODEL_POOL = 1
comptime PARALLEL_OFFLOAD = 2
comptime PARALLEL_OP_CLOSE = 0
comptime PARALLEL_OP_OPEN = 1
comptime PARALLEL_OP_STEP = 2
comptime PARALLEL_OP_EXPORT = 3
comptime PARALLEL_OP_ROLLBACK = 4
comptime PARALLEL_OP_OWNERSHIP = 5
comptime PARALLEL_OP_SET_LR = 6
comptime PARALLEL_OP_SHARD_GRADIENT = 7
comptime PARALLEL_OP_APPLY_GRADIENT = 8
comptime PARALLEL_OP_FOLD_RESET = 9
comptime PARALLEL_OP_SHARD_GRADIENT_FOLD = 10
comptime PARALLEL_OP_FOLD_ADD = 11
comptime PARALLEL_OP_FOLD_EXPORT = 12

# Index of the learning rate in `ParallelByteSession.scalars` (lr, beta1,
# beta2, eps, weight_decay, momentum, dampening, max_norm).
comptime PARALLEL_SCALAR_LR = 0
comptime PARALLEL_SCALAR_COUNT = 8


struct ParallelByteSession(Movable):
    """One trainer's typed state. `provider` is the id of the schedule
    descriptor table's typed record (core/neural_descriptor_run.mojo); 0: the
    installed binding has no cooperative GPU session. The door unpacked it
    from the connected provider once and keeps that provider pinned for as
    long as this lives."""
    var provider: Int
    var handle: Int
    var schedule: Int
    var closed: Bool
    var lost: Bool
    var pool_optimizer: Bool
    var devices: List[Int]
    var logical_shards: Int
    var dims: List[Int]
    var n_total: Int
    var n_tensors: Int
    # The committed step. The door's state dictionary is a boundary copy,
    # refreshed when it is read (`mirror_completed`, `lr_stale`).
    var completed: Int
    var mirror_completed: Int
    var lr_stale: Bool
    var kind: Int
    var nesterov: Bool
    var scalars: List[Float64]

    def __init__(out self, provider: Int, schedule: Int, pool_optimizer: Bool, var devices: List[Int],
                 logical_shards: Int, var dims: List[Int], n_total: Int, n_tensors: Int, completed: Int,
                 kind: Int, nesterov: Bool, var scalars: List[Float64]):
        self.provider = provider
        self.handle = 0
        self.schedule = schedule
        self.closed = False
        self.lost = False
        self.pool_optimizer = pool_optimizer
        self.devices = devices^
        self.logical_shards = logical_shards
        self.dims = dims^
        self.n_total = n_total
        self.n_tensors = n_tensors
        self.completed = completed
        self.mirror_completed = completed
        self.lr_stale = False
        self.kind = kind
        self.nesterov = nesterov
        self.scalars = scalars^


def parallel_session_execute(session: ParallelByteSession, op: Int, addresses: List[Int], ints: List[Int],
                             floats: List[Float64]) raises -> Int:
    """One descriptor call of the open session: the handle word, then the
    operation's buffer addresses."""
    if session.handle == 0:
        raise Error("RuntimeError: parallel trainer has no open session")
    var words: List[Int] = [session.handle]
    for value in addresses:  # driver-loop: over addresses: the operation's admitted buffer descriptors
        words.append(value)
    # A handle exists only for a connected table. The record admits the
    # operation, counts it and makes the leased call.
    if session.provider == 0:
        raise Error("neural native ABI: unknown operation")
    return neural_descriptor_record_execute(session.provider, op, words, ints, floats)


def parallel_session_schedule_name(session: ParallelByteSession) -> String:
    if session.schedule == PARALLEL_REPLICA:
        return String("parallel")
    if session.schedule == PARALLEL_MODEL_POOL:
        return String("model_pool")
    return String("offload")


def parallel_session_openable(session: ParallelByteSession) raises -> Bool:
    """True when a session must be created and opened; False when one is
    open. Refuses a closed or lost trainer."""
    if session.closed or session.lost:
        raise Error("RuntimeError: parallel trainer is closed or lost; restore a state export")
    return session.handle == 0


def parallel_session_open(mut session: ParallelByteSession, handle: Int, state_addresses: List[Int]) raises:
    """Admit the four state arrays (parameters, m, v, flags by address) into
    the session `handle` the schedule factory returned. On a refusal the
    session is closed and the trainer is left without a handle."""
    if handle == 0:
        raise Error("RuntimeError: byte-LM schedule factory returned a null session")
    var ints: List[Int] = [session.completed, session.kind, 1 if session.nesterov else 0,
                           1 if session.pool_optimizer else 0, len(session.dims)]
    for value in session.dims:  # driver-loop: over dims: nine or ten admitted shape words
        ints.append(value)
    ints.append(session.logical_shards)
    ints.append(len(session.devices))
    for value in session.devices:  # driver-loop: over devices: admitted device indices
        ints.append(value)
    session.handle = handle
    try:
        var opened = parallel_session_execute(session, PARALLEL_OP_OPEN, state_addresses, ints, session.scalars)
        if opened != session.completed:
            raise Error("RuntimeError: parallel admission returned wrong step")
    except error:
        try:
            _ = parallel_session_execute(session, PARALLEL_OP_CLOSE, List[Int](), List[Int](), List[Float64]())
        except:
            pass
        session.handle = 0
        raise error


def parallel_session_close(mut session: ParallelByteSession) -> String:
    """Close the trainer. Returns the session's close failure text (empty:
    none); the handle is dropped either way."""
    session.closed = True
    var failure = String()
    if session.handle != 0:
        try:
            _ = parallel_session_execute(session, PARALLEL_OP_CLOSE, List[Int](), List[Int](), List[Float64]())
        except error:
            failure = String(error)
        session.handle = 0
    return failure


def parallel_session_require_split(session: ParallelByteSession, entry: String) raises:
    if session.schedule != PARALLEL_REPLICA:
        raise Error(entry + " requires schedule='replica'")
    if session.pool_optimizer or len(session.devices) != 1:
        raise Error(entry + " requires devices=(d,) and pool_optimizer=False")
    if session.provider == 0:
        raise Error("RuntimeError: installed byte-LM binding lacks " + entry)


def parallel_session_rollback(mut session: ParallelByteSession, before: Int):
    """Return the session to committed step `before` after a refused step;
    a session that cannot is marked lost."""
    try:
        if parallel_session_execute(session, PARALLEL_OP_ROLLBACK, List[Int](), List[Int](), List[Float64]()) != before:
            session.lost = True
    except:
        session.lost = True


def parallel_session_step_begin(mut session: ParallelByteSession, shard_addresses: List[Int],
                                mut losses: List[Float64]) raises -> Int:
    """Run one step over `logical_shards` int32[batch, length + 1] buffers
    by address, writing one loss per shard into `losses` (which must hold
    `logical_shards` cells). Returns the committed step before it. The
    caller finishes with `parallel_session_step_commit`, or with
    `parallel_session_rollback` if it cannot deliver the result. An entry
    failure already rolled back inside the GPU session."""
    var count = session.logical_shards
    if len(shard_addresses) != count or len(losses) != count:
        raise Error("logical shard count mismatch")
    var addresses: List[Int] = [Int(losses.unsafe_ptr())]
    for address in shard_addresses:  # driver-loop: over count: one admitted buffer per logical shard
        addresses.append(address)
    var before = session.completed
    var ints: List[Int] = [before, count]
    var written = parallel_session_execute(session, PARALLEL_OP_STEP, addresses, ints, List[Float64]())
    if written != count:
        parallel_session_rollback(session, before)
        raise Error("RuntimeError: parallel step returned wrong loss count")
    return before


def parallel_session_step_commit(mut session: ParallelByteSession, before: Int):
    session.completed = before + 1


def parallel_session_step(mut session: ParallelByteSession, shard_addresses: List[Int],
                          mut losses: List[Float64]) raises -> Int:
    """One committed step for a native caller; returns the new step."""
    var before = parallel_session_step_begin(session, shard_addresses, losses)
    parallel_session_step_commit(session, before)
    return before + 1


def parallel_session_export(session: ParallelByteSession, rank: Int, gradients: Bool, addresses: List[Int]) raises:
    """Write replica `rank`'s state (parameters, m, v, flags) or its
    gradient (one float32[n_total]) into the given target addresses."""
    var ints: List[Int] = [rank, 1 if gradients else 0]
    var step = parallel_session_execute(session, PARALLEL_OP_EXPORT, addresses, ints, List[Float64]())
    if step != session.completed:
        raise Error("RuntimeError: parallel export returned wrong step")


def parallel_session_set_lr(mut session: ParallelByteSession, lr: Float32) raises:
    if lr <= 0:
        raise Error("Byte-LM requires lr > 0")
    if session.handle != 0:
        if session.schedule != PARALLEL_REPLICA:
            raise Error("RuntimeError: installed byte-LM binding lacks set_lr")
        var floats: List[Float64] = [Float64(lr)]
        var stored = parallel_session_execute(session, PARALLEL_OP_SET_LR, List[Int](), List[Int](), floats)
        if UInt32(stored) != bitcast[DType.uint32](lr):
            raise Error("RuntimeError: native set_lr stored different float32 bits")
    session.scalars[PARALLEL_SCALAR_LR] = Float64(lr)
    session.lr_stale = True


def parallel_session_shard_gradient(session: ParallelByteSession, ids_address: Int, gradient_address: Int,
                                    fold: Bool) raises -> Float64:
    """The loss of one int32[batch, length + 1] shard at the committed
    step. With `fold` the gradient joins the session's device fold;
    otherwise it is written to float32[n_total] at `gradient_address`.
    The caller has passed `parallel_session_require_split`."""
    var loss = List[Float64](length=1, fill=0)
    var addresses: List[Int] = [ids_address]
    if not fold:
        addresses.append(gradient_address)
    addresses.append(Int(loss.unsafe_ptr()))
    var ints: List[Int] = [session.completed]
    _ = parallel_session_execute(session, PARALLEL_OP_SHARD_GRADIENT_FOLD if fold else PARALLEL_OP_SHARD_GRADIENT,
                                 addresses, ints, List[Float64]())
    # Named after the call: the session wrote this cell.
    var value = loss[0]
    return value


def parallel_session_apply_gradient(mut session: ParallelByteSession, total_address: Int) raises -> Int:
    """Apply the summed float32[n_total] gradient at `total_address` and
    commit the step; returns the new step. The session's device scan
    refuses non-finite values."""
    var before = session.completed
    var addresses: List[Int] = [total_address]
    var ints: List[Int] = [before]
    var completed = parallel_session_execute(session, PARALLEL_OP_APPLY_GRADIENT, addresses, ints, List[Float64]())
    if completed != before + 1:
        session.lost = True
        raise Error("RuntimeError: apply_gradient returned wrong step")
    session.completed = completed
    return completed


def parallel_session_fold(session: ParallelByteSession, add: Bool, address: Int) raises:
    """`fold_add` of the float32[n_total] gradient at `address`, or
    `fold_reset` to that prefix (address 0: an empty fold)."""
    var addresses = List[Int]()
    if address != 0:
        addresses.append(address)
    elif add:
        raise Error("fold_add requires a gradient")
    _ = parallel_session_execute(session, PARALLEL_OP_FOLD_ADD if add else PARALLEL_OP_FOLD_RESET,
                                 addresses, List[Int](), List[Float64]())


def parallel_session_fold_export(session: ParallelByteSession, address: Int) raises:
    """Write the device fold into float32[n_total] at `address`."""
    var addresses: List[Int] = [address]
    _ = parallel_session_execute(session, PARALLEL_OP_FOLD_EXPORT, addresses, List[Int](), List[Float64]())


def parallel_session_ownership_width(session: ParallelByteSession) -> Int:
    return 5 if session.schedule == PARALLEL_REPLICA else 7


def parallel_session_ownership_capacity(session: ParallelByteSession) -> Int:
    return parallel_session_ownership_width(session) * (session.dims[7] + 2 + len(session.devices))


def parallel_session_ownership(session: ParallelByteSession, mut words: List[Int]) raises -> Int:
    """Fill `words` (capacity `parallel_session_ownership_capacity`) with
    the session's ownership rows; returns the row count."""
    var width = parallel_session_ownership_width(session)
    var capacity = len(words)
    var addresses: List[Int] = [Int(words.unsafe_ptr())]
    var ints: List[Int] = [capacity]
    var written = parallel_session_execute(session, PARALLEL_OP_OWNERSHIP, addresses, ints, List[Float64]())
    if written < 0 or written % width != 0 or written > capacity:
        raise Error("RuntimeError: native ownership returned a malformed record")
    return written // width
