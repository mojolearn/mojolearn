# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The typed rules of the CausalLM model owner's words (core/causal_owner.mojo).

Also the decode state's words: the state check's four refusals in the
door's order and the in-place check of one state buffer.

Host work, no Python and no device: the position check keeps its three
refusals in their order, and the model halves of the engine selection and of
the resident admission answer as the door's record reads did. The cases are
a few word compares each (microseconds together), so they run in one unit:
splitting them across the host pool would cost more than the work.

The resident session lease (`causal_lease_*`): the word's rule case by
case, then the test-and-set under contention: every host worker tries to
take ONE word many times, and no two are ever inside it together.
"""
from std.testing import TestSuite
from core.causal_layer_run import (
    CAUSAL_LAYER_TRANSFORMER, CAUSAL_LAYER_MAMBA1, CAUSAL_LAYER_MAMBA2, CAUSAL_LAYER_OTHER,
)
from core.causal_owner import (
    CausalModelWords, causal_positions_admit, causal_engine_candidate, causal_resident_candidate,
    CausalStateWords, CausalBufferWords, CausalLayerStateWords, causal_state_admit,
    causal_state_buffer_used, causal_state_buffer_contiguous, causal_state_buffer_address,
    CAUSAL_STATE_K, CAUSAL_STATE_V, CAUSAL_STATE_CONV, CAUSAL_STATE_AFFINE_B, CAUSAL_STATE_XBC,
    CAUSAL_STATE_DTRAW, CAUSAL_STATE_BUFFERS,
    CausalLeaseWord, causal_lease_load, causal_lease_take, causal_lease_release, causal_lease_expose,
    CAUSAL_LEASE_BUSY, CAUSAL_LEASE_EXPOSED, CAUSAL_LEASE_TAKEN, CAUSAL_LEASE_HELD, CAUSAL_LEASE_PRIVATE,
)
from core.host_parallel import host_parallelize
from std.atomic import Atomic, Ordering
from checks.native_tests.testsuite_cases import require


def _refusal(words: CausalModelWords, has_state: Bool, have: Int, capacity: Int, length: Int) -> String:
    try:
        causal_positions_admit(words, has_state, have, capacity, length)
    except error:
        return String(error)
    return String("")


def _local(kind: Int) -> CausalModelWords:
    var words = CausalModelWords()
    words.generation = 1
    words.kind = kind
    words.width = 8
    words.vocab = 16
    words.layers = 2
    words.norm = True
    words.primitives = True
    words.provider = True
    words.block_operations = kind == CAUSAL_LAYER_TRANSFORMER
    return words^


def run_admission_checks() raises:
    var words = _local(CAUSAL_LAYER_TRANSFORMER)
    # positions: an allocation has no state and no capacity word
    require(_refusal(words, False, 0, 0, 64) == "", "allocation without a limit is admitted")
    require(_refusal(words, False, 0, 0, -1) == "CausalLM: negative sequence position", "negative length")
    require(_refusal(words, True, -1, 8, 1) == "CausalLM: negative sequence position", "negative position")
    words.limited = True
    words.max_positions = 6
    require(_refusal(words, False, 0, 0, 7) == "CausalLM: max_tokens or sequence exceeds max_positions", "limit, allocation")
    # the limit is reported before the capacity, as the door did
    require(_refusal(words, True, 4, 5, 3) == "CausalLM: max_tokens or sequence exceeds max_positions", "limit before capacity")
    require(_refusal(words, True, 4, 5, 2) == "CausalLM: sequence exceeds allocated state capacity", "capacity")
    require(_refusal(words, True, 4, 6, 2) == "", "exactly full is admitted")
    # engine: a local fp32 model of a known kind with both providers
    require(causal_engine_candidate(_local(CAUSAL_LAYER_TRANSFORMER)), "transformer engine candidate")
    require(causal_engine_candidate(_local(CAUSAL_LAYER_MAMBA2)), "mamba2 engine candidate")
    require(not causal_engine_candidate(_local(CAUSAL_LAYER_OTHER)), "unknown kind")
    var m1 = _local(CAUSAL_LAYER_MAMBA1)
    require(not causal_engine_candidate(m1), "mamba1 without the plain profile")
    m1.mamba1_profile = 1
    require(causal_engine_candidate(m1), "mamba1 plain profile")
    m1.mamba1_profile = 2
    require(not causal_engine_candidate(m1), "mamba1 affine profile")
    var held = _local(CAUSAL_LAYER_TRANSFORMER)
    held.parallel = True
    require(not causal_engine_candidate(held), "parallel model")
    held.parallel = False
    held.head_packed = True
    require(not causal_engine_candidate(held), "packed head")
    held.head_packed = False
    held.provider = False
    require(not causal_engine_candidate(held), "no layer provider")
    held.provider = True
    held.primitives = False
    require(not causal_engine_candidate(held), "no primitives provider")
    # resident: a transformer with a norm and the resident entries
    var r = _local(CAUSAL_LAYER_TRANSFORMER)
    require(not causal_resident_candidate(r), "no resident entries")
    r.resident = True
    require(causal_resident_candidate(r), "resident transformer")
    r.norm = False
    require(not causal_resident_candidate(r), "no final norm")
    r.norm = True
    r.head_packed = True
    require(not causal_resident_candidate(r), "packed head off CUDA")
    r.resident_cuda = True
    require(causal_resident_candidate(r), "packed head on CUDA")
    var m2 = _local(CAUSAL_LAYER_MAMBA2)
    m2.resident = True
    require(not causal_resident_candidate(m2), "mamba is never resident")


def _state_refusal(model: CausalModelWords, state: CausalStateWords, batch: Int, length: Int) -> String:
    try:
        causal_state_admit(model, state, batch, length)
    except error:
        return String(error)
    return String("")


def _buffer_refusal(buffer: CausalBufferWords, expected: List[Int]) -> String:
    try:
        _ = causal_state_buffer_address(buffer, expected, "k_cache")
    except error:
        return String(error)
    return String("")


def run_state_checks() raises:
    # the state check: four refusals in the door's order
    var model = _local(CAUSAL_LAYER_TRANSFORMER)
    model.generation = 7
    var state = CausalStateWords()
    state.model_generation = 7
    state.layers = 2
    state.batch = 3
    state.capacity = 10
    state.positions = 4
    require(_state_refusal(model, state, 3, 6) == "", "a bound usable state with room")
    require(_state_refusal(model, state, 3, 7) == "CausalLM: sequence exceeds allocated state capacity", "capacity")
    state.batch = 2
    require(_state_refusal(model, state, 3, 7) == "CausalLM: state batch/layer counts disagree with model and input", "batch before positions")
    state.usable = False
    require(_state_refusal(model, state, 3, 7) == "CausalLM: state lost after a failed call; reset or restore it", "lost before counts")
    state.model_generation = 6
    require(_state_refusal(model, state, 3, 7) == "CausalLM: state belongs to another model", "another model first")
    # which buffers a layer kind has
    require(causal_state_buffer_used(CAUSAL_LAYER_TRANSFORMER, CAUSAL_STATE_K)
            and causal_state_buffer_used(CAUSAL_LAYER_TRANSFORMER, CAUSAL_STATE_V)
            and not causal_state_buffer_used(CAUSAL_LAYER_TRANSFORMER, CAUSAL_STATE_CONV), "transformer caches")
    require(causal_state_buffer_used(CAUSAL_LAYER_MAMBA1, CAUSAL_STATE_AFFINE_B)
            and not causal_state_buffer_used(CAUSAL_LAYER_MAMBA1, CAUSAL_STATE_XBC), "mamba1 buffers")
    require(causal_state_buffer_used(CAUSAL_LAYER_MAMBA2, CAUSAL_STATE_DTRAW)
            and not causal_state_buffer_used(CAUSAL_LAYER_MAMBA2, CAUSAL_STATE_AFFINE_B)
            and not causal_state_buffer_used(CAUSAL_LAYER_OTHER, CAUSAL_STATE_K), "mamba2 buffers")
    var layer = CausalLayerStateWords()
    require(len(layer.buffers) == CAUSAL_STATE_BUFFERS and not layer.buffers[CAUSAL_STATE_K].admitted, "empty slots")
    # one buffer: the in-place check's refusals in the door's order
    var expected: List[Int] = [2, 3, 4]
    var buffer = CausalBufferWords()
    require(_buffer_refusal(buffer, expected) == "CausalLM: state buffer k_cache is missing; allocate_state() makes a complete state", "missing")
    buffer.present = True
    buffer.contiguous = True
    require(_buffer_refusal(buffer, expected) == "CausalLM: state buffer k_cache must be writable C-contiguous float32 (it is updated in place)", "layout")
    buffer.writable = True
    var short: List[Int] = [2, 3]
    buffer.shape = short^
    buffer.shape_text = String("(2, 3)")
    require(_buffer_refusal(buffer, expected) == "CausalLM: state buffer k_cache has shape (2, 3), want (2, 3, 4)", "rank")
    buffer.shape = expected.copy()
    buffer.nbytes = 95
    require(_buffer_refusal(buffer, expected) == "CausalLM: state buffer k_cache span differs from its shape", "span")
    buffer.nbytes = 96
    buffer.address = 4096
    require(_buffer_refusal(buffer, expected) == "", "admitted")
    require(causal_state_buffer_address(buffer, expected, "k_cache") == 4096, "address")
    causal_state_buffer_contiguous(buffer)
    causal_state_buffer_contiguous(CausalBufferWords())
    buffer.contiguous = False
    var refused = String("")
    try:
        causal_state_buffer_contiguous(buffer)
    except error:
        refused = String(error)
    require(refused == "generation: internal model and state arrays must be C-contiguous", "contiguity")


comptime _LEASE_TASKS = 64
comptime _LEASE_ROUNDS = 20000


def run_lease_checks() raises:
    # The rule, one holder at a time.
    var storage = List[Int64](length=4, fill=0)
    var word = CausalLeaseWord(unsafe_from_address=Int(storage.unsafe_ptr()))
    require(causal_lease_load(word) == 0, "a new lease is free")
    require(causal_lease_take(word) == CAUSAL_LEASE_TAKEN, "a free lease is taken")
    require(causal_lease_load(word) == CAUSAL_LEASE_BUSY, "taken is busy")
    require(causal_lease_take(word) == CAUSAL_LEASE_HELD, "a second caller does not get the kept sessions (it runs private)")
    require(causal_lease_take(word) == CAUSAL_LEASE_HELD, "nor a third: losing changes nothing")
    require(causal_lease_load(word) == CAUSAL_LEASE_BUSY, "a loser leaves the word to its holder")
    require(not causal_lease_release(word), "released, not exposed")
    require(causal_lease_load(word) == 0, "free again")
    require(causal_lease_take(word) == CAUSAL_LEASE_TAKEN, "taken after a release")
    # Exposed under a running holder: the holder keeps running, learns it
    # at its release, and nobody takes the lease afterwards.
    require(not causal_lease_expose(word), "exposing a held lease does not take it")
    require(causal_lease_load(word) == (CAUSAL_LEASE_BUSY | CAUSAL_LEASE_EXPOSED), "busy and exposed")
    require(causal_lease_take(word) == CAUSAL_LEASE_PRIVATE, "an exposed lease is never taken, even while held")
    require(causal_lease_release(word), "the holder's release reports the exposure")
    require(causal_lease_load(word) == CAUSAL_LEASE_EXPOSED, "exposed stays")
    require(causal_lease_take(word) == CAUSAL_LEASE_PRIVATE, "exposed: private sessions, no lease")
    require(causal_lease_load(word) == CAUSAL_LEASE_EXPOSED, "a private caller holds nothing")
    require(causal_lease_release(word), "a release with nothing held keeps the exposure")
    require(causal_lease_expose(word), "exposing a free, exposed lease again holds it for the drop")
    require(causal_lease_release(word), "and gives it back exposed")
    require(causal_lease_load(word) == CAUSAL_LEASE_EXPOSED, "exposure is monotonic")
    # Exposing a free lease takes it for the drop of the kept sessions.
    var second = word + 1
    require(causal_lease_expose(second), "exposing a free lease holds it")
    require(causal_lease_load(second) == (CAUSAL_LEASE_BUSY | CAUSAL_LEASE_EXPOSED), "held for the drop")
    require(causal_lease_release(second), "then released, exposed")
    # The test-and-set under contention: slot 2 is the lease, slot 3 counts
    # the holders inside it. Every task wins the kept sessions or loses
    # (and would run private); a winner that ever finds another holder
    # inside records it.
    var lease = word + 2
    var inside = word + 3
    var taken = List[Int64](length=_LEASE_TASKS, fill=0)
    var refused = List[Int64](length=_LEASE_TASKS, fill=0)
    var broken = List[Int64](length=_LEASE_TASKS, fill=0)
    var tp = taken.unsafe_ptr()
    var rp = refused.unsafe_ptr()
    var bp = broken.unsafe_ptr()
    def contend(task: Int) {imm lease, imm inside, imm tp, imm rp, imm bp}:
        var mine = Int64(0)
        var lost = Int64(0)
        var bad = Int64(0)
        for _ in range(_LEASE_ROUNDS):  # driver-loop: over this task's attempts on the one shared word
            var verdict = causal_lease_take(lease)
            if verdict == CAUSAL_LEASE_TAKEN:
                if Atomic[Int64].fetch_add(inside, Int64(1)) != 0:
                    bad += 1
                mine += 1
                if Atomic[Int64].fetch_add(inside, Int64(-1)) != 1:
                    bad += 1
                _ = causal_lease_release(lease)
            elif verdict == CAUSAL_LEASE_HELD:
                lost += 1
            else:
                bad += 1
        tp.unsafe_store(task, mine)
        rp.unsafe_store(task, lost)
        bp.unsafe_store(task, bad)
    host_parallelize(contend, _LEASE_TASKS)
    var total = Int64(0)
    var holders = Int64(0)
    for task in range(_LEASE_TASKS):  # small-loop(the tasks' three counters)
        require(broken[task] == 0, "two holders were inside the lease together")
        total += taken[task] + refused[task]
        holders += taken[task]
    require(total == Int64(_LEASE_TASKS * _LEASE_ROUNDS), "every attempt won the lease or lost it")
    require(holders > 0, "the lease was taken")
    require(causal_lease_load(lease) == 0, "free after the last release")
    require(causal_lease_load(inside) == 0, "nobody left inside")
    _ = storage


def test_admission() raises:
    run_admission_checks()


def test_state() raises:
    run_state_checks()


def test_lease() raises:
    run_lease_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
