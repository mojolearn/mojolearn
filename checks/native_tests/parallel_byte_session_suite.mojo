# SPDX-License-Identifier: Apache-2.0
"""Native state and lifetime contracts of `ParallelByteSession`
(bindings/parallel_byte_native_run.mojo), the typed state of the byte-LM
multi-device trainer door.

The schedule descriptor table is a stand-in entry that keeps a few words of
session state (the committed step, call marks, fault switches) at the handle
address and answers each operation as the byte-LM binding's table does. These
are host ownership and bookkeeping contracts of the typed session: the open /
close / lost transitions, the committed step, the rollback on a refused step,
the per-operation call counts and the request shapes each operation sends.
They are not GPU tests and compute no model arithmetic.

Five independent fixtures, run across the host's cores.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.neural_descriptor_run import NeuralDescriptorRecord
from core.native_descriptor_call import native_descriptor_report
from bindings.parallel_byte_native_run import (
    ParallelByteSession, PARALLEL_SCHEDULE_OPERATIONS, PARALLEL_REPLICA, PARALLEL_MODEL_POOL,
    PARALLEL_OP_CLOSE, PARALLEL_OP_OPEN, PARALLEL_OP_STEP, PARALLEL_OP_EXPORT, PARALLEL_OP_ROLLBACK,
    PARALLEL_OP_OWNERSHIP, PARALLEL_OP_SET_LR, PARALLEL_OP_SHARD_GRADIENT, PARALLEL_OP_APPLY_GRADIENT,
    PARALLEL_OP_FOLD_RESET, PARALLEL_OP_SHARD_GRADIENT_FOLD, PARALLEL_OP_FOLD_ADD, PARALLEL_OP_FOLD_EXPORT,
    PARALLEL_SCALAR_LR, PARALLEL_SCALAR_COUNT,
    parallel_session_execute, parallel_session_schedule_name, parallel_session_openable, parallel_session_open,
    parallel_session_close, parallel_session_require_split, parallel_session_rollback,
    parallel_session_step_begin, parallel_session_step_commit, parallel_session_step, parallel_session_export,
    parallel_session_set_lr, parallel_session_shard_gradient, parallel_session_apply_gradient,
    parallel_session_fold, parallel_session_fold_export, parallel_session_ownership_width,
    parallel_session_ownership_capacity, parallel_session_ownership,
)

comptime PARALLEL_BYTE_SESSION_FIXTURES = 5

# The stand-in session's words, at the handle address.
comptime W_STEP = 0          # the committed step the stand-in answers with
comptime W_CLOSES = 1        # close calls seen
comptime W_ROLLBACKS = 2     # rollback calls seen
comptime W_FAULT_OP = 3      # operation that raises (-1: none)
comptime W_SKEW_OP = 4       # operation whose result is one off (-1: none)
comptime W_LAST_ADDRESSES = 5  # address words of the last call, the handle included
comptime W_LAST_INTS = 6     # integer words of the last call
comptime W_FIRST_INT = 7     # the last call's first integer word (0 when it had none)
comptime W_COUNT = 8

comptime STEP_AT_OPEN = 7
comptime SHARDS = 3
comptime DIMS = 10


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refused(message: String, expected: String) -> Bool:
    return message.find(expected) != -1


def entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
          result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The stand-in schedule table. Address word 0 is the session handle."""
    try:
        require(na >= 1, "stand-in: every schedule call carries its handle")
        var addresses = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=addresses[0])
        var ints = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        var answer = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        words[W_LAST_ADDRESSES] = na
        words[W_LAST_INTS] = np
        words[W_FIRST_INT] = ints[0] if np > 0 else 0
        if words[W_FAULT_OP] == operation:
            raise Error("RuntimeError: stand-in refused operation " + String(operation))
        var value = 0
        if operation == PARALLEL_OP_OPEN:
            # [completed, kind, nesterov, pool_optimizer, n_dims, dims..., shards, n_devices, devices...]
            require(na == 5 and np >= 7 and nf == PARALLEL_SCALAR_COUNT, "stand-in: open request shape")
            var n_dims = ints[4]
            require(np == 5 + n_dims + 2 + ints[5 + n_dims + 1], "stand-in: open integer words")
            words[W_STEP] = ints[0]
            value = ints[0]
        elif operation == PARALLEL_OP_CLOSE:
            words[W_CLOSES] += 1
        elif operation == PARALLEL_OP_ROLLBACK:
            words[W_ROLLBACKS] += 1
            value = words[W_STEP]
        elif operation == PARALLEL_OP_STEP:
            # addresses [handle, losses, shard...]; ints [before, count]
            require(np == 2 and na == 2 + ints[1] and ints[0] == words[W_STEP], "stand-in: step request shape")
            var losses = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addresses[1])
            for i in range(ints[1]):  # small-loop(one loss per logical shard)
                losses[i] = Float64(i) + 0.5
            value = ints[1]
        elif operation == PARALLEL_OP_EXPORT:
            require(np == 2, "stand-in: export request shape")
            value = words[W_STEP]
        elif operation == PARALLEL_OP_SET_LR:
            require(na == 1 and np == 0 and nf == 1, "stand-in: set_lr request shape")
            var lr = Float32(Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)[0])
            value = Int(bitcast[DType.uint32](lr))
        elif operation == PARALLEL_OP_SHARD_GRADIENT or operation == PARALLEL_OP_SHARD_GRADIENT_FOLD:
            # [handle, ids, (gradient,) loss]; the loss cell is the last address
            require(np == 1 and ints[0] == words[W_STEP], "stand-in: shard gradient is taken at the committed step")
            require(na == (3 if operation == PARALLEL_OP_SHARD_GRADIENT_FOLD else 4), "stand-in: shard gradient addresses")
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addresses[na - 1])[0] = 2.5
        elif operation == PARALLEL_OP_APPLY_GRADIENT:
            require(na == 2 and np == 1 and ints[0] == words[W_STEP], "stand-in: apply request shape")
            words[W_STEP] += 1
            value = words[W_STEP]
        elif operation == PARALLEL_OP_FOLD_RESET or operation == PARALLEL_OP_FOLD_ADD or operation == PARALLEL_OP_FOLD_EXPORT:
            require(np == 0, "stand-in: fold request shape")
        elif operation == PARALLEL_OP_OWNERSHIP:
            # addresses [handle, rows]; ints [capacity]: two rows of five words
            require(na == 2 and np == 1 and ints[0] >= 10, "stand-in: ownership request shape")
            var rows = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=addresses[1])
            for i in range(10):  # small-loop(two ownership rows)
                rows[i] = 100 + i
            value = 10
        else:
            raise Error("stand-in: operation outside the schedule table")
        if words[W_SKEW_OP] == operation:
            value += 1
        answer[0] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def stand_in() -> List[Int]:
    var words = List[Int](length=W_COUNT, fill=0)
    words[W_FAULT_OP] = -1
    words[W_SKEW_OP] = -1
    return words^


def provider() -> List[NeuralDescriptorRecord]:
    """The stand-in table's typed provider record (one element, so its
    address is the provider id for as long as the list lives)."""
    var records = List[NeuralDescriptorRecord]()
    records.append(NeuralDescriptorRecord(String("stand-in.schedule"), entry, PARALLEL_SCHEDULE_OPERATIONS))
    return records^


def session(provider_id: Int, schedule: Int, devices: Int, pool_optimizer: Bool) -> ParallelByteSession:
    """A session as the door's constructor builds it: `devices` device
    indices, SHARDS logical shards, DIMS shape words, step STEP_AT_OPEN.
    `provider_id` 0: a binding without the cooperative table."""
    var selected = List[Int]()
    for d in range(devices):  # small-loop(device indices of the fixture)
        selected.append(d)
    var dims = List[Int](length=DIMS, fill=2)
    var scalars = List[Float64](length=PARALLEL_SCALAR_COUNT, fill=0.0)
    scalars[PARALLEL_SCALAR_LR] = 0.125
    return ParallelByteSession(provider_id, schedule, pool_optimizer, selected^,
                               SHARDS, dims^, 64, 12, STEP_AT_OPEN, 2, False, scalars^)


def opened(mut trainer: ParallelByteSession, handle: Int) raises:
    """Open on four state arrays (one cell each: the stand-in reads none)."""
    var state = List[Float32](length=4, fill=0.0)
    var base = Int(state.unsafe_ptr())
    var addresses: List[Int] = [base, base + 4, base + 8, base + 12]
    parallel_session_open(trainer, handle, addresses)
    _ = state  # the addressed cells outlive the open call


def text_of(problem: Error) -> String:
    return String(problem)


def fixture_open_close() raises:
    """No handle, no call; open admits the committed step; a refused open
    closes the factory's session and leaves no handle; close drops the handle
    and a closed trainer is refused."""
    var words = stand_in()
    var handle = Int(words.unsafe_ptr())
    var records = provider()
    var table = Int(records.unsafe_ptr())
    var trainer = session(table, PARALLEL_REPLICA, 2, True)
    require(parallel_session_schedule_name(trainer) == "parallel", "replica schedule is named parallel")
    require(parallel_session_openable(trainer), "a new trainer needs its session opened")
    var message = String()
    try:
        _ = parallel_session_execute(trainer, PARALLEL_OP_EXPORT, List[Int](), List[Int](), List[Float64]())
    except problem:
        message = text_of(problem)
    require(refused(message, "no open session"), "a call before open is refused")
    message = String()
    try:
        opened(trainer, 0)
    except problem:
        message = text_of(problem)
    require(refused(message, "null session") and trainer.handle == 0, "a null factory session is refused")
    # A table that answers another step: the open is refused, the session closed, no handle kept.
    words[W_SKEW_OP] = PARALLEL_OP_OPEN
    message = String()
    try:
        opened(trainer, handle)
    except problem:
        message = text_of(problem)
    require(refused(message, "wrong step") and trainer.handle == 0 and words[W_CLOSES] == 1,
            "a refused open closes the session once and keeps no handle")
    words[W_SKEW_OP] = -1
    opened(trainer, handle)
    require(trainer.handle == handle and words[W_STEP] == STEP_AT_OPEN and trainer.completed == STEP_AT_OPEN,
            "open admits the committed step")
    # [completed, kind, nesterov, pool, n_dims] + dims + [shards, n_devices] + devices
    require(words[W_LAST_ADDRESSES] == 5 and words[W_LAST_INTS] == 5 + DIMS + 2 + 2, "open request words")
    require(not parallel_session_openable(trainer), "an open trainer is not opened twice")
    require(records[0].calls[PARALLEL_OP_OPEN] == 2 and records[0].calls[PARALLEL_OP_CLOSE] == 1 and records[0].calls[PARALLEL_OP_EXPORT] == 0,
            "every admitted call is counted under its operation; a refused one is not")
    message = String()
    try:
        _ = parallel_session_execute(trainer, PARALLEL_SCHEDULE_OPERATIONS, List[Int](), List[Int](), List[Float64]())
    except problem:
        message = text_of(problem)
    require(refused(message, "unknown operation"), "an operation outside the table is refused before the call")
    # Close reports the session's failure text and drops the handle either way.
    words[W_FAULT_OP] = PARALLEL_OP_CLOSE
    var failure = parallel_session_close(trainer)
    require(refused(failure, "stand-in refused") and trainer.handle == 0 and trainer.closed,
            "close keeps the failure text and drops the handle")
    require(parallel_session_close(trainer) == "", "closing twice calls nothing")
    message = String()
    try:
        _ = parallel_session_openable(trainer)
    except problem:
        message = text_of(problem)
    require(refused(message, "closed or lost"), "a closed trainer is refused")
    _ = words  # the stand-in's words outlive every call through the handle
    _ = records  # the provider record outlives every session that holds its id


def fixture_step() raises:
    """A step writes one loss per shard and commits once; a wrong shard
    count is refused before the call; a short loss count rolls back; a
    rollback that answers another step marks the trainer lost."""
    var words = stand_in()
    var handle = Int(words.unsafe_ptr())
    var records = provider()
    var table = Int(records.unsafe_ptr())
    var trainer = session(table, PARALLEL_REPLICA, 2, True)
    opened(trainer, handle)
    var ids = List[Int32](length=SHARDS, fill=0)
    var shards = List[Int]()
    for i in range(SHARDS):  # small-loop(one buffer address per logical shard)
        shards.append(Int(ids.unsafe_ptr()) + 4 * i)
    var losses = List[Float64](length=SHARDS, fill=0.0)
    var before = parallel_session_step_begin(trainer, shards, losses)
    require(before == STEP_AT_OPEN and trainer.completed == STEP_AT_OPEN, "a step is not committed before its commit")
    require(losses[0] == 0.5 and losses[1] == 1.5 and losses[2] == 2.5, "one loss per shard, in shard order")
    parallel_session_step_commit(trainer, before)
    require(trainer.completed == STEP_AT_OPEN + 1, "the commit advances the step once")
    words[W_STEP] = trainer.completed
    require(parallel_session_step(trainer, shards, losses) == STEP_AT_OPEN + 2 and trainer.completed == STEP_AT_OPEN + 2,
            "a native caller's step commits and returns the new step")
    words[W_STEP] = trainer.completed
    var short = List[Float64](length=SHARDS - 1, fill=0.0)
    var message = String()
    try:
        _ = parallel_session_step_begin(trainer, shards, short)
    except problem:
        message = text_of(problem)
    require(refused(message, "shard count mismatch") and records[0].calls[PARALLEL_OP_STEP] == 2,
            "a loss buffer of another length is refused before the call")
    # The table answers one loss too many: rolled back to the committed step, not lost.
    words[W_SKEW_OP] = PARALLEL_OP_STEP
    message = String()
    try:
        _ = parallel_session_step_begin(trainer, shards, losses)
    except problem:
        message = text_of(problem)
    require(refused(message, "wrong loss count") and words[W_ROLLBACKS] == 1 and not trainer.lost
            and trainer.completed == STEP_AT_OPEN + 2, "a wrong loss count rolls back and commits nothing")
    # A rollback that answers another step: the session is lost.
    words[W_SKEW_OP] = PARALLEL_OP_ROLLBACK
    parallel_session_rollback(trainer, trainer.completed)
    require(trainer.lost and words[W_ROLLBACKS] == 2, "a rollback to another step marks the trainer lost")
    message = String()
    try:
        _ = parallel_session_openable(trainer)
    except problem:
        message = text_of(problem)
    require(refused(message, "closed or lost"), "a lost trainer is refused")
    # A rollback the table refuses also marks the trainer lost.
    var other = session(table, PARALLEL_REPLICA, 2, True)
    words[W_SKEW_OP] = -1
    words[W_STEP] = STEP_AT_OPEN
    opened(other, handle)
    words[W_FAULT_OP] = PARALLEL_OP_ROLLBACK
    parallel_session_rollback(other, STEP_AT_OPEN)
    require(other.lost, "a refused rollback marks the trainer lost")
    _ = ids
    _ = words
    _ = records  # the provider record outlives every session that holds its id


def fixture_set_lr() raises:
    """set_lr stores the float32 the session stored, marks the boundary copy
    stale, refuses a non-positive rate and another schedule, and refuses a
    session that stored other bits."""
    var words = stand_in()
    var handle = Int(words.unsafe_ptr())
    var records = provider()
    var table = Int(records.unsafe_ptr())
    var trainer = session(table, PARALLEL_REPLICA, 1, False)
    # Before open: the rate is kept for the open, no call is made.
    parallel_session_set_lr(trainer, Float32(0.5))
    require(trainer.scalars[PARALLEL_SCALAR_LR] == 0.5 and trainer.lr_stale and records[0].calls[PARALLEL_OP_SET_LR] == 0,
            "a rate set before open is kept without a call")
    opened(trainer, handle)
    trainer.lr_stale = False
    parallel_session_set_lr(trainer, Float32(0.1))
    require(trainer.scalars[PARALLEL_SCALAR_LR] == Float64(Float32(0.1)) and trainer.lr_stale
            and records[0].calls[PARALLEL_OP_SET_LR] == 1, "the stored rate is the float32 the session stored")
    var message = String()
    try:
        parallel_session_set_lr(trainer, Float32(0.0))
    except problem:
        message = text_of(problem)
    require(refused(message, "lr > 0") and records[0].calls[PARALLEL_OP_SET_LR] == 1, "a non-positive rate is refused before the call")
    words[W_SKEW_OP] = PARALLEL_OP_SET_LR
    message = String()
    try:
        parallel_session_set_lr(trainer, Float32(0.25))
    except problem:
        message = text_of(problem)
    require(refused(message, "different float32 bits") and trainer.scalars[PARALLEL_SCALAR_LR] == Float64(Float32(0.1)),
            "a session that stored other bits is refused and the rate is unchanged")
    words[W_SKEW_OP] = -1
    var pool = session(table, PARALLEL_MODEL_POOL, 2, True)
    require(parallel_session_schedule_name(pool) == "model_pool", "the pool schedule's name")
    opened(pool, handle)
    message = String()
    try:
        parallel_session_set_lr(pool, Float32(0.25))
    except problem:
        message = text_of(problem)
    require(refused(message, "lacks set_lr"), "set_lr on another schedule's open session is refused")
    _ = words
    _ = records  # the provider record outlives every session that holds its id


def fixture_require_split() raises:
    """The split entries need the replica schedule, one device, no pool
    optimizer and a connected table."""
    var records = provider()
    var table = Int(records.unsafe_ptr())
    var good = session(table, PARALLEL_REPLICA, 1, False)
    parallel_session_require_split(good, "shard_gradient")
    var cases = List[String]()
    try:
        parallel_session_require_split(session(table, PARALLEL_MODEL_POOL, 1, False), "shard_gradient")
    except problem:
        cases.append(text_of(problem))
    try:
        parallel_session_require_split(session(table, PARALLEL_REPLICA, 2, False), "shard_gradient")
    except problem:
        cases.append(text_of(problem))
    try:
        parallel_session_require_split(session(table, PARALLEL_REPLICA, 1, True), "shard_gradient")
    except problem:
        cases.append(text_of(problem))
    try:
        parallel_session_require_split(session(0, PARALLEL_REPLICA, 1, False), "shard_gradient")
    except problem:
        cases.append(text_of(problem))
    require(len(cases) == 4, "each of the four conditions refuses")
    require(refused(cases[0], "requires ParallelByteLanguageModelTrainer"), "another schedule is refused by name")
    require(refused(cases[1], "devices=(d,)") and refused(cases[2], "pool_optimizer=False"),
            "more than one device or the pool optimizer is refused")
    require(refused(cases[3], "binding lacks shard_gradient"), "an unconnected table is refused")
    require(parallel_session_ownership_width(good) == 5
            and parallel_session_ownership_width(session(table, PARALLEL_MODEL_POOL, 1, False)) == 7,
            "ownership rows are five words for replicas, seven otherwise")
    require(parallel_session_ownership_capacity(good) == 5 * (2 + 2 + 1), "ownership capacity from layers and devices")
    _ = records  # the provider record outlives every session that holds its id


def fixture_gradients() raises:
    """The split trainer's operations: shard gradient with and without the
    device fold, the fold operations, the apply that commits, the export and
    ownership checks."""
    var words = stand_in()
    var handle = Int(words.unsafe_ptr())
    var records = provider()
    var table = Int(records.unsafe_ptr())
    var trainer = session(table, PARALLEL_REPLICA, 1, False)
    opened(trainer, handle)
    var cells = List[Float32](length=8, fill=0.0)
    var address = Int(cells.unsafe_ptr())
    require(parallel_session_shard_gradient(trainer, address, address + 16, False) == 2.5
            and words[W_LAST_ADDRESSES] == 4 and words[W_FIRST_INT] == STEP_AT_OPEN,
            "a shard gradient sends ids, the gradient target and the loss cell at the committed step")
    require(parallel_session_shard_gradient(trainer, address, 0, True) == 2.5 and words[W_LAST_ADDRESSES] == 3,
            "a folded shard gradient sends no gradient target")
    require(records[0].calls[PARALLEL_OP_SHARD_GRADIENT] == 1 and records[0].calls[PARALLEL_OP_SHARD_GRADIENT_FOLD] == 1,
            "the two shard gradient operations are counted apart")
    parallel_session_fold(trainer, False, 0)
    require(words[W_LAST_ADDRESSES] == 1 and records[0].calls[PARALLEL_OP_FOLD_RESET] == 1, "an empty fold reset sends the handle only")
    parallel_session_fold(trainer, False, address)
    parallel_session_fold(trainer, True, address)
    require(words[W_LAST_ADDRESSES] == 2 and records[0].calls[PARALLEL_OP_FOLD_RESET] == 2 and records[0].calls[PARALLEL_OP_FOLD_ADD] == 1,
            "a fold reset to a prefix and a fold add send one gradient")
    var message = String()
    try:
        parallel_session_fold(trainer, True, 0)
    except problem:
        message = text_of(problem)
    require(refused(message, "fold_add requires a gradient") and records[0].calls[PARALLEL_OP_FOLD_ADD] == 1,
            "a fold add without a gradient is refused before the call")
    parallel_session_fold_export(trainer, address)
    require(records[0].calls[PARALLEL_OP_FOLD_EXPORT] == 1, "the fold export is one call")
    require(parallel_session_apply_gradient(trainer, address) == STEP_AT_OPEN + 1
            and trainer.completed == STEP_AT_OPEN + 1 and not trainer.lost, "apply commits the step it returns")
    var targets: List[Int] = [address]
    parallel_session_export(trainer, 0, True, targets)
    require(words[W_LAST_INTS] == 2 and words[W_FIRST_INT] == 0, "export sends the rank and the gradient flag")
    words[W_SKEW_OP] = PARALLEL_OP_EXPORT
    message = String()
    try:
        parallel_session_export(trainer, 0, False, targets)
    except problem:
        message = text_of(problem)
    require(refused(message, "export returned wrong step"), "an export at another step is refused")
    words[W_SKEW_OP] = -1
    var rows = List[Int](length=parallel_session_ownership_capacity(trainer), fill=0)
    require(parallel_session_ownership(trainer, rows) == 2 and rows[0] == 100 and rows[9] == 109,
            "ownership returns whole rows written into the caller's words")
    words[W_SKEW_OP] = PARALLEL_OP_OWNERSHIP
    message = String()
    try:
        _ = parallel_session_ownership(trainer, rows)
    except problem:
        message = text_of(problem)
    require(refused(message, "malformed record"), "a partial ownership row is refused")
    # An apply that answers another step: nothing committed, the trainer is lost.
    words[W_SKEW_OP] = PARALLEL_OP_APPLY_GRADIENT
    message = String()
    try:
        _ = parallel_session_apply_gradient(trainer, address)
    except problem:
        message = text_of(problem)
    require(refused(message, "apply_gradient returned wrong step") and trainer.lost
            and trainer.completed == STEP_AT_OPEN + 1, "a wrong apply step commits nothing and marks the trainer lost")
    _ = cells
    _ = words
    _ = records  # the provider record outlives every session that holds its id


def fixture(unit: Int) raises:
    if unit == 0:
        fixture_open_close()
    elif unit == 1:
        fixture_step()
    elif unit == 2:
        fixture_set_lr()
    elif unit == 3:
        fixture_require_split()
    else:
        fixture_gradients()


def run_checks() raises:
    var failures = List[Int](length=PARALLEL_BYTE_SESSION_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except problem:
            fp[unit] = 1
            print("FAIL parallel_byte_session fixture", unit, String(problem))
    host_parallelize(run, PARALLEL_BYTE_SESSION_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: fixture summaries
        failed += value
    require(failed == 0, "native parallel byte session failed " + String(failed) + " fixtures")
    print("PASS parallel_byte_session:", PARALLEL_BYTE_SESSION_FIXTURES, "native state fixtures")
