# SPDX-License-Identifier: Apache-2.0
"""The typed rules of the CausalLM model owner's words (core/causal_owner.mojo).

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


def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("causal_owner", cpu=True, gpu=False, serial="", run=run_cases)
