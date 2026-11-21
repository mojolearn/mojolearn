# SPDX-License-Identifier: Apache-2.0
"""Typed rules of the parallel neural trainer (RUN): the owner's operation
codes, the admission refusals of construction, training and restore, the
generator stream bound, and the version rule of the workers' resident model
cache (which committed step deltas a worker replays, and when it must ask
for the full state). `bindings/parallel_neural_native_door.mojo` unpacks the
Python values into these arguments, moves the frames and packs the results;
no Python object reaches this file. A few words per training step: nothing
to split across cores."""
from core.mode_answer import MODE_FAMILY_NEURAL, word_runs_identical

comptime NEURAL_SCHEMA = "mojolearn.parallel-neural.v1"
comptime NEURAL_REDUCTION = "ordered_sum"
#: The refusal a worker answers when it cannot reach the header's version;
#: the owner then resends the full state.
comptime NEURAL_CACHE_MISS = "parallel neural: worker cache miss"
#: Committed steps a request carries as deltas.
comptime NEURAL_CACHE_HISTORY = 4
comptime NEURAL_MAX_SHARDS = 1024
comptime NEURAL_STREAM_LIMIT = 0xFFFFFFFF

# The owner's gradient operation.
comptime NEURAL_OPERATION_OTHER = 0
comptime NEURAL_OPERATION_MLP = 1
comptime NEURAL_OPERATION_SAMBA = 2

# What a worker does with one committed step delta (`neural_delta_action`).
comptime NEURAL_DELTA_SKIP = 0    # already applied
comptime NEURAL_DELTA_APPLY = 1   # the next step


def neural_operation_code(operation: String) -> Int:
    if operation == "mlp_gradient":
        return NEURAL_OPERATION_MLP
    if operation == "samba_gradient":
        return NEURAL_OPERATION_SAMBA
    return NEURAL_OPERATION_OTHER


def neural_operation_name(code: Int) -> String:
    return String("mlp_gradient") if code == NEURAL_OPERATION_MLP else String("samba_gradient")


def neural_accumulation_admit(steps: Int) raises:
    if steps != 1:
        raise Error("parallel Samba requires accumulation_steps=1; logical_shards owns accumulation")


def neural_mode_admit(present: Bool, word: String, refusal: String) raises:
    """`fast` that the identical set answers (NEURAL) trains the identical
    stack, as the stack's own fit does (lane fx-fast-mask)."""
    if present and not word_runs_identical(word, MODE_FAMILY_NEURAL):
        raise Error(refusal)


def neural_devices_admit(devices: Int, shards: Int) raises:
    if devices > shards:
        raise Error("physical device count exceeds logical shard count")


def neural_ids_rank_admit(rank: Int) raises:
    if rank != 2:
        raise Error("mojolearn.SambaStack: inputs and targets must be (B,L) integer ids")


def neural_ids_signed(format: String) raises -> Bool:
    """Admits a native integer buffer format; True for a signed one."""
    if format == "b" or format == "h" or format == "i" or format == "l" or format == "q":
        return True
    if format == "B" or format == "H" or format == "I" or format == "L" or format == "Q":
        return False
    raise Error("mojolearn.SambaStack: inputs and targets must be native integer buffers")


def neural_train_admit(closed: Bool, supplied: Int, logical: Int) raises:
    """`supplied` is read only for an open trainer."""
    if closed:
        raise Error("RuntimeError: parallel trainer is closed")
    if supplied != logical:
        raise Error("logical shard count mismatch")


def neural_stream_admit(counter: Int) raises:
    if counter > NEURAL_STREAM_LIMIT:
        raise Error("RuntimeError: mojolearn.Generator stream counter exhausted")


def neural_shard_admit(fields: Int) raises:
    if fields != 2:
        raise Error("parallel shard must contain inputs and targets")


def neural_cache_current(digest: String, committed: String, version: Int) -> Bool:
    """The owner's state is the one it last committed: a request then
    carries only the committed step deltas."""
    return digest != "" and committed == digest and version > 0


def neural_resend_full(cached: Bool, failure: String) -> Bool:
    """A delta request a worker answered with the cache-miss words is sent
    again with the full state; any other failure is raised."""
    return cached and failure.find(NEURAL_CACHE_MISS) >= 0


def neural_cache_reach_admit(resident: Bool, same_epoch: Bool, have: Int, want: Int) raises:
    """A worker advances by deltas only from a resident state of the same
    epoch that is not ahead of the header. `have` and `want` are read only
    for such a state."""
    if not resident or not same_epoch or have > want:
        raise Error(NEURAL_CACHE_MISS)


def neural_delta_action(version: Int, have: Int) raises -> Int:
    """A delta at or below the resident version is skipped; the next version
    is applied; a gap is a cache miss."""
    if version <= have:
        return NEURAL_DELTA_SKIP
    if version != have + 1:
        raise Error(NEURAL_CACHE_MISS)
    return NEURAL_DELTA_APPLY


def neural_cache_arrived_admit(have: Int, want: Int) raises:
    if have != want:
        raise Error(NEURAL_CACHE_MISS)


def neural_checkpoint_admit(schema: String, reduction: String) raises:
    if schema != NEURAL_SCHEMA or reduction != NEURAL_REDUCTION:
        raise Error("unsupported parallel neural checkpoint contract")


def neural_restore_operation(operation: String) raises -> Int:
    var code = neural_operation_code(operation)
    if code == NEURAL_OPERATION_OTHER:
        raise Error("unsupported parallel neural checkpoint operation")
    return code


def neural_envelope_admit(fields: Int) raises:
    if fields != 3:
        raise Error("parallel neural: invalid worker request envelope")


def neural_gradient_admit(committed: Bool) raises:
    if not committed:
        raise Error("RuntimeError: no committed parallel gradient")
