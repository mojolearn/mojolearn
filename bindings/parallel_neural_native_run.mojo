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
from core.parallel_worker_request import (
    WorkerRequest, PN_MLP_GRADIENT_CACHED, PN_SAMBA_GRADIENT_CACHED, PN_MLP_UPDATE, PN_SAMBA_UPDATE, PN_GRADIENT_WORDS,
)

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


struct NeuralTrainerState(Movable):
    """One trainer's typed state: what the operations decide on. The door's
    owner record holds it beside the boundary pins (the model, the pools, the
    lock, the committed gradient buffers and the step deltas on their way to
    the workers); no operation reads any of it from the instance."""
    var operation: Int
    var logical_shards: Int
    #: one device: the update request goes to the gradient pool's worker
    var shared_pool: Bool
    var closed: Bool
    #: a step committed since the last failed or started one: its gradient
    #: buffers can be exported
    var committed: Bool
    var cache_epoch: Int
    var cache_version: Int
    var cache_digest: String

    def __init__(out self, operation: Int, logical_shards: Int, shared_pool: Bool):
        self.operation = operation
        self.logical_shards = logical_shards
        self.shared_pool = shared_pool
        self.closed = False
        self.committed = False
        self.cache_epoch = 0
        self.cache_version = 0
        self.cache_digest = String()


def neural_step_admit(state: NeuralTrainerState, supplied: Int) raises:
    """A step needs an open trainer and one microbatch per logical shard."""
    neural_train_admit(state.closed, 0, 0)
    neural_train_admit(False, supplied, state.logical_shards)


def neural_step_withdraw(mut state: NeuralTrainerState):
    """The last committed gradient is withdrawn once a step has taken its
    snapshot: a failed step leaves nothing to export."""
    state.committed = False


def neural_step_begin(mut state: NeuralTrainerState, digest: String) -> Bool:
    """Starts a step on the owner's state of `digest`. True when that state
    is the one last committed (the request carries deltas); otherwise a new
    cache epoch starts and the caller drops the retained deltas and sends
    the full state."""
    var cached = neural_cache_current(digest, state.cache_digest, state.cache_version)
    if not cached:
        state.cache_epoch += 1
    return cached


def neural_step_version(state: NeuralTrainerState) -> Int:
    """The version the step in flight commits as."""
    return state.cache_version + 1


def neural_step_commit(mut state: NeuralTrainerState, digest: String):
    """The step is published: `digest` is the committed state's digest."""
    state.cache_version += 1
    state.cache_digest = digest
    state.committed = True


def neural_history_excess(retained: Int) -> Int:
    """How many of the oldest retained step deltas to drop."""
    return retained - NEURAL_CACHE_HISTORY if retained > NEURAL_CACHE_HISTORY else 0


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
    """A request's payload is the pair (model slot, arguments); the
    operation and the cache words ride in the typed request."""
    if fields != 2:
        raise Error("parallel neural: invalid worker request envelope")


def neural_gradient_words(samba: Bool, state: NeuralTrainerState, count: Int) -> List[List[Int64]]:
    """The typed requests of one gradient wave: the same operation and cache
    words (epoch, version) for each of the `count` logical shards."""
    var request = WorkerRequest(PN_SAMBA_GRADIENT_CACHED if samba else PN_MLP_GRADIENT_CACHED)
    request.words.append(state.cache_epoch)
    request.words.append(state.cache_version)
    var wire = request.encode()
    var words = List[List[Int64]]()
    for _ in range(count):  # driver-loop: over logical training shards in immutable reduction order
        words.append(wire.copy())
    return words^


def neural_update_words(samba: Bool) -> List[List[Int64]]:
    """The typed request of the one cooperative update."""
    var request = WorkerRequest(PN_SAMBA_UPDATE if samba else PN_MLP_UPDATE)
    var words = List[List[Int64]]()
    words.append(request.encode())
    return words^


def neural_gradient_words_admit(request: WorkerRequest) raises:
    if len(request.words) != PN_GRADIENT_WORDS:
        raise Error("parallel neural: invalid worker request envelope")


def neural_gradient_admit(committed: Bool) raises:
    if not committed:
        raise Error("RuntimeError: no committed parallel gradient")
