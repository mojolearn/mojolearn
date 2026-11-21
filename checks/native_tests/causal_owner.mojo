# SPDX-License-Identifier: Apache-2.0
"""The typed rules of the CausalLM model owner's words (core/causal_owner.mojo).

Also the decode state's words: the state check's four refusals in the
door's order and the in-place check of one state buffer.

Host work, no Python and no device: the position check keeps its three
refusals in their order, and the model halves of the engine selection and of
the resident admission answer as the door's record reads did. The cases are
a few word compares each (microseconds together), so they run in one unit:
splitting them across the host pool would cost more than the work.
"""
from checks.native_tests.runner import Check, Column, Report, require
from core.causal_layer_run import (
    CAUSAL_LAYER_TRANSFORMER, CAUSAL_LAYER_MAMBA1, CAUSAL_LAYER_MAMBA2, CAUSAL_LAYER_OTHER,
)
from core.causal_owner import (
    CausalModelWords, causal_positions_admit, causal_engine_candidate, causal_resident_candidate,
    CausalStateWords, CausalBufferWords, CausalLayerStateWords, causal_state_admit,
    causal_state_buffer_used, causal_state_buffer_contiguous, causal_state_buffer_address,
    CAUSAL_STATE_K, CAUSAL_STATE_V, CAUSAL_STATE_CONV, CAUSAL_STATE_AFFINE_B, CAUSAL_STATE_XBC,
    CAUSAL_STATE_DTRAW, CAUSAL_STATE_BUFFERS,
)


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


def run_checks() raises:
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


def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
        run_state_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("causal_owner", cpu=True, gpu=False, serial="", run=run_cases)
