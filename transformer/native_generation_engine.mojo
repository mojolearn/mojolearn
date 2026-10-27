# SPDX-License-Identifier: Apache-2.0
# Operation-local nonresident fp32 Transformer execution. Numerical operations
# use the existing sibling ABI entries in their original order. No provider,
# Python state, Python allocator, or callback is reachable from this module.
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from transformer.session_abi import GenerationSessionForward
from transformer.native_session_call import generation_session_forward
from transformer.causal_lm_public_impl import causal_lm_pick_step, causal_lm_rows_done


struct NativeGenerationLayer(Movable):
    var forward: GenerationSessionForward
    var handle: Int
    var addresses: List[Int]
    var parameters: List[Int]
    var cached: Int
    var attempted: Bool
    var visited: Bool
    var profile_admitted: Bool
    var failure: String

    def __init__(out self, forward: GenerationSessionForward, handle: Int,
                 var addresses: List[Int], var parameters: List[Int], cached: Int):
        self.forward = forward
        self.handle = handle
        self.addresses = addresses^
        self.parameters = parameters^
        self.cached = cached
        self.attempted = False
        self.visited = False
        self.profile_admitted = False
        self.failure = String("")


struct NativeGenerationModel(Movable):
    var primitive: NativeDescriptorCall
    var embedding: Int
    var norm: Int
    var head: Int
    var norm_eps: Float64
    var batch: Int
    var width: Int
    var vocab: Int
    var position: Int
    var capacity: Int
    var limited: Bool
    var max_positions: Int
    var layers: List[NativeGenerationLayer]
    var calls: List[Int]
    var failed: Bool
    var forwards: Int

    def __init__(out self, primitive: NativeDescriptorCall, embedding: Int, norm: Int, head: Int,
                 norm_eps: Float64, batch: Int, width: Int, vocab: Int, position: Int,
                 capacity: Int, limited: Bool, max_positions: Int):
        self.primitive = primitive
        self.embedding = embedding
        self.norm = norm
        self.head = head
        self.norm_eps = norm_eps
        self.batch = batch
        self.width = width
        self.vocab = vocab
        self.position = position
        self.capacity = capacity
        self.limited = limited
        self.max_positions = max_positions
        self.layers = List[NativeGenerationLayer]()
        self.calls = List[Int](length=6, fill=0)
        self.failed = False
        self.forwards = 0

    def primitive_call(mut self, op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises:
        self.calls[op] += 1
        _ = native_descriptor_call(self.primitive, op, a, p, f)

    def forward(mut self, ids: Int, length: Int) raises -> List[Float32]:
        # These checks precede execution/failure invalidation, matching _state.
        if length < 0 or self.position < 0:
            raise Error("CausalLM: negative sequence position")
        if self.limited and self.position + length > self.max_positions:
            raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
        if self.position + length > self.capacity:
            raise Error("CausalLM: sequence exceeds allocated state capacity")
        var rows = self.batch * length
        try:
            var hidden = List[Float32](length=rows * self.width, fill=0)
            self.primitive_call(0, [Int(hidden.unsafe_ptr()), self.embedding, ids], [rows, self.vocab, self.width], [])
            for i in range(len(self.layers)):  # driver-loop: ordered native layer descriptors
                self.layers[i].visited = True
                if self.layers[i].failure.byte_length() != 0:
                    raise Error(self.layers[i].failure)
                var next = List[Float32](length=rows * self.width, fill=0)
                var addresses = self.layers[i].addresses.copy()
                addresses[0] = Int(hidden.unsafe_ptr())
                addresses[12] = Int(next.unsafe_ptr())
                var parameters = self.layers[i].parameters.copy()
                parameters[0] = self.batch
                parameters[1] = length
                parameters[8] = self.layers[i].cached
                self.layers[i].attempted = True
                self.layers[i].cached = generation_session_forward(1, self.layers[i].forward,
                    self.layers[i].handle, addresses, parameters)
                _ = hidden
                hidden = next^
            if self.norm != 0:
                var normalized = List[Float32](length=rows * self.width, fill=0)
                self.primitive_call(2, [Int(normalized.unsafe_ptr()), Int(hidden.unsafe_ptr()), self.norm],
                                    [rows, self.width], [self.norm_eps])
                _ = hidden
                hidden = normalized^
            var logits = List[Float32](length=rows * self.vocab, fill=0)
            self.primitive_call(4, [Int(logits.unsafe_ptr()), Int(hidden.unsafe_ptr()), self.head],
                                [rows, self.vocab, self.width], [])
            _ = hidden
            self.position += length
            self.forwards += 1
            return logits^
        except problem:
            self.failed = True
            raise problem


struct NativeGenerationProgress(Movable):
    var current: List[Int32]
    var length: Int
    var produced: Int
    var wrote: Int
    var tokens: List[Int32]

    def __init__(out self, address: Int, batch: Int, length: Int, produced: Int, capacity: Int):
        self.current = List[Int32]()
        var input = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
        for i in range(batch * length):  # driver-loop: copy admitted current-token storage once
            self.current.append(input[i])
        self.length = length
        self.produced = produced
        self.wrote = 0
        self.tokens = List[Int32](capacity=batch * capacity)


def native_generation_read(
    mut model: NativeGenerationModel, mut progress: NativeGenerationProgress,
    count: Int, maximum: Int, mut finished: List[Bool], mut lengths: List[Int], eos: List[Int],
    pad: Int, greedy: Bool, temperature: Float32, top_k: Int, top_p: Float64, seed: UInt64,
) raises:
    # The entire token/layer loop owns native state. Sampling is the exact
    # existing native policy; partial counters remain in progress on failure.
    while progress.wrote < count and progress.produced < maximum and not causal_lm_rows_done(finished):  # driver-loop: native autoregressive tokens
        var logits = model.forward(Int(progress.current.unsafe_ptr()), progress.length)
        var next = List[Int32](length=model.batch, fill=0)
        causal_lm_pick_step(
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(logits.unsafe_ptr())),
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(next.unsafe_ptr())),
            model.batch, progress.length, model.vocab, finished, lengths, eos, pad, greedy,
            temperature, top_k, top_p, seed, progress.produced, progress.tokens)
        _ = logits
        progress.current = next^
        progress.length = 1
        progress.produced += 1
        progress.wrote += 1
