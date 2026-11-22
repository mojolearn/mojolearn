# SPDX-License-Identifier: Apache-2.0
"""The stand-in schedule entry of `ParallelByteSession` (bindings/parallel_byte_native_run.mojo), shared by
the `parallel_byte_session` and `cross_vendor_loopback` checks.

The schedule descriptor table is a stand-in entry that keeps a few words of
session state (the committed step, call marks, fault switches) at the handle
address and answers each operation as the byte-LM binding's table does. A helper
module: no `main()` and no `test_*` (docs/plans/TESTSUITE_CONVERSION.md).
"""
from std.memory import bitcast
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
from checks.native_tests.testsuite_cases import require



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


