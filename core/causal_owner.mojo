# SPDX-License-Identifier: Apache-2.0
"""Typed words of the CausalLM native model owner (no Python).

A public CausalLM operation decides its route and checks its state from
these words, never from the model's Python record. The owner that holds
them is built once per model generation by the causal door
(bindings/causal_operation_owner_door.mojo `CausalModelOwner`) and lives in
the model's prepared record: a capture whose identity key still matches
hands the same owner over; a changed model gets a new owner with a new
`generation`.

The rules below are the model-only halves of the door's route decisions
and the position check of a state, word for word what the door tested on
the Python record before (lane dw-causal, 2026-11-21).
"""
from core.causal_layer_run import (
    CAUSAL_LAYER_TRANSFORMER, CAUSAL_LAYER_MAMBA1, CAUSAL_LAYER_OTHER,
)


struct CausalModelWords(Copyable, Movable):
    """What is fixed for one generation of a model."""

    # Process counter value given when the owner was built; 0 = not built.
    var generation: Int
    # `causal_layer_kind` word of the model's layer kind.
    var kind: Int
    var width: Int
    var vocab: Int
    var layers: Int
    # The model has a position limit (`max_positions` is not None).
    var limited: Bool
    var max_positions: Int
    # The model has a final norm.
    var norm: Bool
    # The head is the packed (int15) plane triple.
    var head_packed: Bool
    # Entry registry words (the admitted execution registry of the model).
    # The model runs through the parallel door.
    var parallel: Bool
    # The primitives provider is admitted.
    var primitives: Bool
    # The layer kind's provider is admitted (block operations for a
    # Transformer, the Mamba provider otherwise).
    var provider: Bool
    # The Transformer block operations provider is admitted (any kind).
    var block_operations: Bool
    # Mamba-1 state profile of the admitted image (0 when not a Mamba-1).
    var mamba1_profile: Int
    # The registry carries the resident session route, and its vendor is CUDA.
    var resident: Bool
    var resident_cuda: Bool

    def __init__(out self):
        self.generation = 0
        self.kind = CAUSAL_LAYER_OTHER
        self.width = 0
        self.vocab = 0
        self.layers = 0
        self.limited = False
        self.max_positions = 0
        self.norm = False
        self.head_packed = False
        self.parallel = False
        self.primitives = False
        self.provider = False
        self.block_operations = False
        self.mamba1_profile = 0
        self.resident = False
        self.resident_cuda = False


def causal_positions_admit(words: CausalModelWords, has_state: Bool, have: Int, capacity: Int, length: Int) raises:
    """The position check of an allocation (`has_state` False, `have` 0) or
    of a forward on a state at `have` tokens with room for `capacity`."""
    if length < 0 or have < 0:
        raise Error("CausalLM: negative sequence position")
    if words.limited and have + length > words.max_positions:
        raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
    if has_state and have + length > capacity:
        raise Error("CausalLM: sequence exceeds allocated state capacity")


def causal_engine_candidate(words: CausalModelWords) -> Bool:
    """The model-only half of the native generation engine's selection: a
    local fp32 model of a known kind whose providers are admitted. The
    caller adds the build's dispatch word, the setup arm and the resident
    verdict (which reads the state)."""
    if words.parallel or words.head_packed:
        return False
    if words.kind == CAUSAL_LAYER_OTHER:
        return False
    if not words.primitives or not words.provider:
        return False
    if words.kind == CAUSAL_LAYER_MAMBA1 and words.mamba1_profile != 1:
        return False
    return True


def causal_resident_candidate(words: CausalModelWords) -> Bool:
    """The model-only half of the resident forward's admission: a
    Transformer stack with a final norm, the resident entries, and a head
    the resident session takes (a packed head only on CUDA)."""
    if words.kind != CAUSAL_LAYER_TRANSFORMER or not words.norm:
        return False
    if not words.resident:
        return False
    if words.head_packed and not words.resident_cuda:
        return False
    return True
