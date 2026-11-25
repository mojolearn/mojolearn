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

The decode state has its own words (`CausalStateWords`, and per layer
state `CausalLayerStateWords`: where each state buffer is and what shape it
has), held by `CausalStateOwner` of the same door. `causal_state_admit` is
the state check of a forward; `causal_state_buffer_address` is the in-place
check of one state buffer of a layer call. Both keep the door's refusals,
words and order.
"""
from core.causal_layer_run import (
    CAUSAL_LAYER_TRANSFORMER, CAUSAL_LAYER_MAMBA1, CAUSAL_LAYER_MAMBA2, CAUSAL_LAYER_OTHER,
)
from core.generation_buffers_impl import generation_shape_size, generation_shape_text
from std.atomic import Atomic, Ordering


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
    # lane fixed15-parts-native (2026-11-25): some part of the model (head
    # or layers) computes under fixed15_v1; the head has its own word above.
    var fixed15: Bool
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
    # The decode program's session family word of the registry
    # (`program_read`); -1 when the install has no GPU decode program.
    var program_read: Int
    # lane fixed15-decode (2026-11-25): the registry carries the fixed15
    # planes entry ("forward15": a GPU block image with op 35), so a
    # fixed15_v1 layer can run on its planes.
    var planes: Bool
    # The final norm's eps (read only when `norm`).
    var norm_eps: Float64

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
        self.fixed15 = False
        self.parallel = False
        self.primitives = False
        self.provider = False
        self.block_operations = False
        self.mamba1_profile = 0
        self.resident = False
        self.resident_cuda = False
        self.program_read = -1
        self.planes = False
        self.norm_eps = 0.0


def causal_positions_admit(words: CausalModelWords, has_state: Bool, have: Int, capacity: Int, length: Int) raises:
    """The position check of an allocation (`has_state` False, `have` 0) or
    of a forward on a state at `have` tokens with room for `capacity`."""
    if length < 0 or have < 0:
        raise Error("CausalLM: negative sequence position")
    if words.limited and have + length > words.max_positions:
        raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
    if has_state and have + length > capacity:
        raise Error("CausalLM: sequence exceeds allocated state capacity")


# The route word of a read the decode program serves (the door's route
# witness says the same word).
comptime CAUSAL_ROUTE_PROGRAM = "program_decode"


def causal_program_route(words: CausalModelWords, greedy: Bool, sampling_policy: Int, layers: Int) -> String:
    """The decode program's admission of a read, on typed words: its route
    word (`CAUSAL_ROUTE_PROGRAM`) when the program serves it, else why not
    (the door's refusal text; "layer kind" is completed by the door with the
    kind's name). Lane fixed15-decode (2026-11-25): fixed15_v1 layers (any
    parts mask) and a fixed15_v1 head are served (adopted planes, the
    `OPS_LINEAR15` head), so the fixed15_v1 default generates as one program
    on the GPU as fp32_v1 does; before, both were refused and every default
    read took the per-token route."""
    if not greedy and sampling_policy != 1:
        return "the Float64 sampling policy runs on the host"
    if words.program_read < 0:
        return "no GPU decode program in this install"
    if words.parallel:
        return "distributed execution"
    if words.fixed15 and not words.planes:
        return "fixed15_v1 without the planes entry"
    if words.kind != CAUSAL_LAYER_TRANSFORMER and words.kind != CAUSAL_LAYER_MAMBA1 and words.kind != CAUSAL_LAYER_MAMBA2:
        return "layer kind"
    if words.kind == CAUSAL_LAYER_MAMBA1 and words.mamba1_profile != 1:
        return "the Mamba-1 profile"
    if layers == 0:
        return "a model without layers"
    if words.norm and not (words.norm_eps > 0):
        return "a final norm with eps 0"
    return String(CAUSAL_ROUTE_PROGRAM)


def causal_engine_candidate(words: CausalModelWords) -> Bool:
    """The model-only half of the native generation engine's selection: a
    local model of a known kind whose providers are admitted (lane
    fixed15-decode: fixed15_v1 layers and heads included, the engine runs
    their entries). The caller adds the build's dispatch word, the setup arm
    and the resident verdict (which reads the state)."""
    if words.parallel:
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


# The state buffers of a layer state, in the order of the door's capture
# schema (bindings/causal_operation_owner_door.mojo `layer_buffers`).
comptime CAUSAL_STATE_K = 0
comptime CAUSAL_STATE_V = 1
comptime CAUSAL_STATE_CONV = 2
comptime CAUSAL_STATE_H = 3
comptime CAUSAL_STATE_AFFINE_BOUNDARY = 4
comptime CAUSAL_STATE_AFFINE_A = 5
comptime CAUSAL_STATE_AFFINE_B = 6
comptime CAUSAL_STATE_XBC = 7
comptime CAUSAL_STATE_DTRAW = 8
comptime CAUSAL_STATE_BUFFERS = 9


def causal_state_buffer_used(kind: Int, index: Int) -> Bool:
    """Whether a layer of `kind` has the state buffer `index` at all."""
    if kind == CAUSAL_LAYER_TRANSFORMER:
        return index == CAUSAL_STATE_K or index == CAUSAL_STATE_V
    if kind == CAUSAL_LAYER_MAMBA1:
        return index >= CAUSAL_STATE_CONV and index <= CAUSAL_STATE_AFFINE_B
    if kind == CAUSAL_LAYER_MAMBA2:
        return (index == CAUSAL_STATE_CONV or index == CAUSAL_STATE_H
                or index == CAUSAL_STATE_XBC or index == CAUSAL_STATE_DTRAW)
    return False


struct CausalBufferWords(Copyable, Movable):
    """One state buffer as the owner admitted it from its storage pin."""

    # The words below were read (False: never admitted, or the pin changed).
    var admitted: Bool
    # The state has this buffer (False: the field is None).
    var present: Bool
    # C-contiguous.
    var contiguous: Bool
    # float32 and writable (`float32` and not read-only).
    var writable: Bool
    # float32 (fixed for the pinned Array; read once at admission).
    var float32: Bool
    var address: Int
    var nbytes: Int
    var shape: List[Int]
    # The shape as a refusal prints it (built once, at admission).
    var shape_text: String

    def __init__(out self):
        self.admitted = False
        self.present = False
        self.contiguous = False
        self.writable = False
        self.float32 = False
        self.address = 0
        self.nbytes = 0
        self.shape = List[Int]()
        self.shape_text = String("")


struct CausalLayerStateWords(Copyable, Movable):
    """The buffers of one layer state object (`CAUSAL_STATE_BUFFERS` slots;
    a slot the layer kind does not use stays unadmitted)."""

    var buffers: List[CausalBufferWords]

    def __init__(out self):
        self.buffers = List[CausalBufferWords](capacity=CAUSAL_STATE_BUFFERS)
        for _ in range(CAUSAL_STATE_BUFFERS):  # small-loop(the fixed state buffer slots)
            self.buffers.append(CausalBufferWords())


struct CausalStateWords(Copyable, Movable):
    """The words of one decode state. `model_generation` and `layers` hold
    for the owner's life; the other four are written by the UNPACK of each
    public operation from the state's fields (a user may edit those)."""

    # `CausalModelWords.generation` of the model this state was bound to.
    var model_generation: Int
    var layers: Int
    var batch: Int
    var capacity: Int
    var positions: Int
    var usable: Bool

    def __init__(out self):
        self.model_generation = 0
        self.layers = 0
        self.batch = 0
        self.capacity = 0
        self.positions = 0
        self.usable = True


def causal_state_admit(model: CausalModelWords, state: CausalStateWords, batch: Int, length: Int) raises:
    """The state check of a forward of `length` tokens for `batch` rows:
    the door's four refusals in their order."""
    if state.model_generation != model.generation:
        raise Error("CausalLM: state belongs to another model")
    if not state.usable:
        raise Error("CausalLM: state lost after a failed call; reset or restore it")
    if state.batch != batch or state.layers != model.layers:
        raise Error("CausalLM: state batch/layer counts disagree with model and input")
    causal_positions_admit(model, True, state.positions, state.capacity, length)


def causal_state_buffer_contiguous(buffer: CausalBufferWords) raises:
    """The layout check the address read of a Transformer cache makes before
    its in-place check (a missing buffer passes here and is refused there)."""
    if buffer.present and not buffer.contiguous:
        raise Error("generation: internal model and state arrays must be C-contiguous")


def causal_state_buffer_address(buffer: CausalBufferWords, expected: List[Int], name: String) raises -> Int:
    """The address of a state buffer a layer call updates in place, after
    the door's checks of it (present; writable C-contiguous float32; exactly
    `expected`; a span of its shape). Nothing is built unless it refuses."""
    if not buffer.present:
        raise Error("CausalLM: state buffer " + name + " is missing; allocate_state() makes a complete state")
    if not buffer.writable or not buffer.contiguous:
        raise Error("CausalLM: state buffer " + name + " must be writable C-contiguous float32 (it is updated in place)")
    var size = generation_shape_size(buffer.shape, expected)
    if size < 0:
        raise Error("CausalLM: state buffer " + name + " has shape " + buffer.shape_text + ", want " + generation_shape_text(expected))
    if buffer.nbytes != size * 4:
        raise Error("CausalLM: state buffer " + name + " span differs from its shape")
    return buffer.address


# The resident session lease of one model: ONE word, read and written only
# through the functions below. The bits:
#   CAUSAL_LEASE_BUSY     a holder has the model's kept resident sessions
#                         (a resident forward, a resident generate, a read
#                         that keeps them across its tokens);
#   CAUSAL_LEASE_EXPOSED  `parameters()` handed the live arrays out
#                         (monotonic for the lease's life): nothing is kept
#                         any more, every call opens and closes its own.
# The rule: taking the lease is one compare-and-set (free -> busy). The
# caller that wins uses the model's kept sessions. A caller that loses
# (another holder has them, or the model is exposed) holds no lease: it
# opens a private session set of its own, proceeds, and closes it when
# done; it never takes, loads, closes or republishes the kept sessions.
# Two threads generating on one model both work. No lock is held across a
# forward: the word is the whole test-and-set.
comptime CAUSAL_LEASE_BUSY = 1
comptime CAUSAL_LEASE_EXPOSED = 2

comptime CAUSAL_LEASE_TAKEN = 0
comptime CAUSAL_LEASE_HELD = 1
comptime CAUSAL_LEASE_PRIVATE = 2

comptime CausalLeaseWord = MutPointer[Int64, MutUntrackedOrigin]


@always_inline
def causal_lease_load(word: CausalLeaseWord) -> Int:
    """The live lease word."""
    return Int(Atomic.load[ordering=Ordering.ACQUIRE](word))


def causal_lease_take(word: CausalLeaseWord) -> Int:
    """The test-and-set: CAUSAL_LEASE_TAKEN when this caller now holds the
    kept sessions (the word went free -> busy in one step),
    CAUSAL_LEASE_PRIVATE when the model is exposed and CAUSAL_LEASE_HELD
    when another holder has it: in both this caller holds no lease and runs
    on private sessions of its own."""
    var expected = Int64(0)
    if Atomic.compare_exchange[
        success_ordering=Ordering.SEQUENTIAL,
        failure_ordering=Ordering.RELAXED,
    ](word, expected, Int64(CAUSAL_LEASE_BUSY)):
        return CAUSAL_LEASE_TAKEN
    # Exposure is monotonic, so a word seen exposed stays exposed; a word
    # seen busy and not exposed was another holder's at the test.
    if (causal_lease_load(word) & CAUSAL_LEASE_EXPOSED) != 0:
        return CAUSAL_LEASE_PRIVATE
    return CAUSAL_LEASE_HELD


def causal_lease_release(word: CausalLeaseWord) -> Bool:
    """The holder gives the lease back. True when the model was exposed
    while it held (or before): the holder then closes its sessions instead
    of leaving them kept; no taker can reach them, because an exposed word
    is never taken."""
    var exposed = False
    while True:  # driver-loop: compare-and-set retry on the one lease word
        var current = Atomic.load[ordering=Ordering.ACQUIRE](word)
        var expected = current
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, current & ~Int64(CAUSAL_LEASE_BUSY)):
            exposed = (Int(current) & CAUSAL_LEASE_EXPOSED) != 0
            break
    return exposed


def causal_lease_expose(word: CausalLeaseWord) -> Bool:
    """Mark the model exposed. True when the word was free and this caller
    now ALSO holds it busy (it drops the kept sessions and then calls
    `causal_lease_release`); False when a holder is running: that holder
    sees the bit at its next test and at its release, and closes."""
    var held = False
    while True:  # driver-loop: compare-and-set retry on the one lease word
        var current = Atomic.load[ordering=Ordering.ACQUIRE](word)
        var expected = current
        var free = (Int(current) & CAUSAL_LEASE_BUSY) == 0
        var wanted = current | Int64(CAUSAL_LEASE_EXPOSED)
        if free:
            wanted = wanted | Int64(CAUSAL_LEASE_BUSY)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, wanted):
            held = free
            break
    return held
