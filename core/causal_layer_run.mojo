# SPDX-License-Identifier: Apache-2.0
"""Typed values of one CausalLM layer run (no Python).

A layer run of the block door and of the whole-model path is three steps
(bindings/causal_lm_public_door.mojo `_causal_layer_run_bound`):

- UNPACK reads the state object and the weights into a `CausalLayerRequest`:
  the table entry, the operation code and the address and scalar words.
- RUN is one table call over the request's words, with the GIL released.
- PACK writes the returned word and the copied state buffers back.

`CausalLayerRoute` holds what is fixed for a block plan or a model snapshot:
the layer kind, the host/device answer, the Mamba-1 profile and the table
entry. It is resolved once by its owner; no layer call reads it by name.
"""
from core.entry_table import EntryRef, entry_ref_unresolved


comptime CAUSAL_LAYER_TRANSFORMER = 0
comptime CAUSAL_LAYER_MAMBA1 = 1
comptime CAUSAL_LAYER_MAMBA2 = 2
comptime CAUSAL_LAYER_OTHER = 3


def causal_layer_kind(kind: String) -> Int:
    """The kind word of a layer kind name. A name that is none of the three
    is `CAUSAL_LAYER_OTHER`: the layer run gives it the Mamba-2 arm, as its
    `else` always did, and refuses a stateless call for it."""
    if kind == "transformer":
        return CAUSAL_LAYER_TRANSFORMER
    if kind == "mamba1":
        return CAUSAL_LAYER_MAMBA1
    if kind == "mamba2":
        return CAUSAL_LAYER_MAMBA2
    return CAUSAL_LAYER_OTHER


struct CausalLayerRoute(Copyable, Movable):
    """The fixed words of a layer's route. `bound` is False until the owner
    resolved it (providers/python/generation_metadata_pyapi_door.mojo
    `causal_layer_route`)."""

    var bound: Bool
    var kind: Int
    # Transformer: the layer runs the host image's block entry (`entry`).
    var cpu: Bool
    # Transformer, device: the route has a block session.
    var block_session: Bool
    # Mamba-1: the state profile of the admitted image (2 = affine).
    var mamba1_profile: Int
    # Mamba: the image is a device image (entry vendor not 0).
    var device_mamba: Bool
    # The CausalLM table entry the layer calls: the Mamba state entry for a
    # Mamba layer, the block entry for a host Transformer layer; unresolved
    # (word 0) for a device Transformer layer, which calls its session.
    var entry: EntryRef

    def __init__(out self):
        self.bound = False
        self.kind = CAUSAL_LAYER_OTHER
        self.cpu = False
        self.block_session = False
        self.mamba1_profile = 0
        self.device_mamba = False
        self.entry = entry_ref_unresolved()


struct CausalLayerRequest(Movable):
    """One layer call as words: what RUN needs and nothing else."""

    # True: a Transformer block session operation on `handle` (reserved and
    # released around the call); False: one CausalLM table call.
    var session: Bool
    var entry: EntryRef
    var handle: Int
    var operation: Int
    var addresses: List[Int]
    var words: List[Int]
    var floats: List[Float64]

    def __init__(out self):
        self.session = False
        self.entry = entry_ref_unresolved()
        self.handle = 0
        self.operation = 0
        self.addresses = List[Int]()
        self.words = List[Int]()
        self.floats = List[Float64]()
