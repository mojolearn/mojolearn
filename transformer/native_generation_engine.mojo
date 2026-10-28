# SPDX-License-Identifier: Apache-2.0
# Operation-local nonresident fp32 Transformer execution. Numerical operations
# use the existing sibling ABI entries in their original order. No provider,
# Python state, Python allocator, or callback is reachable from this module.
from core.native_execution import NativeExecutionEndpoint
from core.borrowed_buffer import BorrowedBufferLayout
from transformer.causal_lm_public_impl import causal_lm_pick_step, causal_lm_rows_done


struct NativeGenerationLayer(Movable):
    var endpoint: NativeExecutionEndpoint
    var operation: Int
    var addresses: List[Int]
    var parameters: List[Int]
    var cached: Int
    var attempted: Bool
    var visited: Bool
    var profile_admitted: Bool
    var failure: String
    var refresh_layouts: List[BorrowedBufferLayout]
    var refresh_addresses: List[Int]
    var writeback_layouts: List[BorrowedBufferLayout]
    var writeback_addresses: List[Int]
    var state_index: Int
    var kind: Int
    var output_slot: Int
    var floats: List[Float64]
    var report_size: Int
    var report: List[Float32]
    var completed: Bool

    def __init__(out self, endpoint: NativeExecutionEndpoint, operation: Int,
                 var addresses: List[Int], var parameters: List[Int], cached: Int):
        self.endpoint = endpoint
        self.operation = operation
        self.addresses = addresses^
        self.parameters = parameters^
        self.cached = cached
        self.attempted = False
        self.visited = False
        self.profile_admitted = False
        self.failure = String("")
        self.refresh_layouts = List[BorrowedBufferLayout]()
        self.refresh_addresses = List[Int]()
        self.writeback_layouts = List[BorrowedBufferLayout]()
        self.writeback_addresses = List[Int]()
        self.state_index = -1
        self.kind = 0
        self.output_slot = 12
        self.floats = List[Float64]()
        self.report_size = 0
        self.report = List[Float32]()
        self.completed = False


struct NativeGenerationModel(Movable):
    var primitive: NativeExecutionEndpoint
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
    var embedding_refresh: List[BorrowedBufferLayout]
    var embedding_targets: List[Int]
    var head_refresh: List[BorrowedBufferLayout]
    var head_targets: List[Int]
    var state_counters: List[Int]
    var embedding_failure: String
    var head_failure: String

    def __init__(out self, primitive: NativeExecutionEndpoint, embedding: Int, norm: Int, head: Int,
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
        self.embedding_refresh = List[BorrowedBufferLayout]()
        self.embedding_targets = List[Int]()
        self.head_refresh = List[BorrowedBufferLayout]()
        self.head_targets = List[Int]()
        self.state_counters = List[Int]()
        self.embedding_failure = String("")
        self.head_failure = String("")

    def primitive_call(mut self, op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises:
        self.calls[op] += 1
        _ = self.primitive.call(op, a, p, f)

    def forward(mut self, ids: Int, length: Int, step: Bool = False) raises -> List[Float32]:
        # These checks precede execution/failure invalidation, matching _state.
        if length < 0 or self.position < 0:
            raise Error("CausalLM: negative sequence position")
        if self.limited and self.position + length > self.max_positions:
            raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
        if self.position + length > self.capacity:
            raise Error("CausalLM: sequence exceeds allocated state capacity")
        var rows = self.batch * length
        try:
            if self.embedding_failure.byte_length() != 0:
                raise Error(self.embedding_failure)
            for i in range(len(self.embedding_refresh)):  # driver-loop: admitted embedding copy descriptors
                self.embedding_refresh[i].copy_to(self.embedding_targets[i])
            var hidden = List[Float32](length=rows * self.width, fill=0)
            self.primitive_call(0, [Int(hidden.unsafe_ptr()), self.embedding, ids], [rows, self.vocab, self.width], [])
            for i in range(len(self.layers)):  # driver-loop: ordered native layer descriptors
                self.layers[i].visited = True
                if self.layers[i].failure.byte_length() != 0:
                    raise Error(self.layers[i].failure)
                for j in range(len(self.layers[i].refresh_layouts)):  # driver-loop: ordered deferred weight/state byte copies
                    self.layers[i].refresh_layouts[j].copy_to(self.layers[i].refresh_addresses[j])
                var next = List[Float32](length=rows * self.width, fill=0)
                var addresses = self.layers[i].addresses.copy()
                addresses[0] = Int(hidden.unsafe_ptr())
                addresses[self.layers[i].output_slot] = Int(next.unsafe_ptr())
                var parameters = self.layers[i].parameters.copy()
                parameters[0] = self.batch
                parameters[1] = length
                if self.layers[i].state_index >= 0:
                    self.layers[i].cached = self.state_counters[self.layers[i].state_index]
                var kind = self.layers[i].kind
                var operation = self.layers[i].operation
                if kind == 0:
                    parameters[8] = self.layers[i].cached
                elif kind == 2:
                    parameters[3] = self.layers[i].cached
                if kind != 0 and step:
                    operation += 1
                    var decode = List[Int]()
                    for index in range(len(parameters)):  # small-loop(Mamba decode omits the length word)
                        if index != 1:
                            decode.append(parameters[index])
                    parameters = decode^
                var report = List[Float32](length=self.layers[i].report_size, fill=0)
                if kind == 2:
                    addresses[15] = Int(report.unsafe_ptr())
                self.layers[i].attempted = True
                var result = self.layers[i].endpoint.call(operation, addresses, parameters, self.layers[i].floats)
                self.layers[i].cached = self.layers[i].cached + length if kind == 1 else result
                if kind == 2:
                    self.layers[i].report = report^
                self.layers[i].completed = True
                for j in range(len(self.layers[i].writeback_layouts)):  # driver-loop: successful layer publication before the next alias reads
                    self.layers[i].writeback_layouts[j].copy_from(self.layers[i].writeback_addresses[j])
                if self.layers[i].state_index >= 0:
                    self.state_counters[self.layers[i].state_index] = self.layers[i].cached
                _ = hidden
                hidden = next^
            if self.head_failure.byte_length() != 0:
                raise Error(self.head_failure)
            for i in range(len(self.head_refresh)):  # driver-loop: preserve head snapshot copy before norm and projection
                self.head_refresh[i].copy_to(self.head_targets[i])
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
        var logits = model.forward(Int(progress.current.unsafe_ptr()), progress.length, progress.produced != 0)
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
