# SPDX-License-Identifier: Apache-2.0
# Operation-local nonresident fp32 Transformer execution. Numerical operations
# use the existing sibling ABI entries in their original order. No provider,
# Python state, Python allocator, or callback is reachable from this module.
from std.memory import memset_zero
from core.generation_policy_impl import GenerationSamplingScratch
from core.native_execution import NativeExecutionEndpoint
from core.borrowed_buffer import BorrowedBufferLayout
from transformer.causal_lm_public_impl import causal_lm_pick_step, causal_lm_rows_done


def _generation_zero_workspace(mut storage: List[Float32], count: Int) -> Int:
    # Native calls borrow, and finish consuming, these host buffers before
    # returning. Reuse only an exactly matching extent; a prompt/decode shape
    # transition replaces storage. Keep the former fresh-list zero semantics,
    # including for kernels that write only part of their output on failure.
    if len(storage) != count:
        storage = List[Float32](length=count, fill=0)
    elif count > 0:
        memset_zero(storage.unsafe_ptr(), count)
    return Int(storage.unsafe_ptr())


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
    var report_workspace: List[Float32]
    var completed: Bool

    def __init__(out self, endpoint: NativeExecutionEndpoint, operation: Int,
                 var addresses: List[Int], var parameters: List[Int], cached: Int):
        self.endpoint = endpoint.copy()
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
        self.report_workspace = List[Float32]()
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
    var hidden_workspace: List[Float32]
    var output_workspace: List[Float32]

    def __init__(out self, primitive: NativeExecutionEndpoint, embedding: Int, norm: Int, head: Int,
                 norm_eps: Float64, batch: Int, width: Int, vocab: Int, position: Int,
                 capacity: Int, limited: Bool, max_positions: Int):
        self.primitive = primitive.copy()
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
        self.hidden_workspace = List[Float32]()
        self.output_workspace = List[Float32]()

    def primitive_call(mut self, op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises:
        self.calls[op] += 1
        _ = self.primitive.call(op, a, p, f)

    def forward(mut self, ids: Int, length: Int) raises -> List[Float32]:
        # Public forward retains fresh result ownership. Only private generation
        # progress supplies reusable output storage to forward_into.
        var logits = List[Float32]()
        self.forward_into(ids, length, logits)
        return logits^

    def forward_into(mut self, ids: Int, length: Int, mut logits: List[Float32]) raises:
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
            var count = rows * self.width
            # One operation owns two disjoint activation buffers. Ping-pong
            # through every layer and the final norm, then reuse at the next
            # token of the same extent. No caller/state buffer enters this pair.
            var hidden = _generation_zero_workspace(self.hidden_workspace, count)
            var output_slot = 1
            self.primitive_call(0, [hidden, self.embedding, ids], [rows, self.vocab, self.width], [])
            for i in range(len(self.layers)):  # driver-loop: ordered native layer descriptors
                ref layer = self.layers[i]
                layer.visited = True
                if layer.failure.byte_length() != 0:
                    raise Error(layer.failure)
                for j in range(len(layer.refresh_layouts)):  # driver-loop: ordered deferred weight/state byte copies
                    layer.refresh_layouts[j].copy_to(layer.refresh_addresses[j])
                var next = _generation_zero_workspace(self.output_workspace, count) if output_slot == 1 else _generation_zero_workspace(self.hidden_workspace, count)
                # Admission owns these vectors exclusively for this operation.
                # Only token-dependent slots change; the synchronous endpoint
                # consumes them before the next layer/token. Keep stable weights
                # and launch words in place instead of copying every token.
                layer.addresses[0] = hidden
                layer.addresses[layer.output_slot] = next
                layer.parameters[1] = length
                if layer.state_index >= 0:
                    layer.cached = self.state_counters[layer.state_index]
                var kind = layer.kind
                var operation = layer.operation
                if kind == 0:
                    layer.parameters[8] = layer.cached
                elif kind == 2:
                    layer.parameters[3] = layer.cached
                # Local _run_snapshot always uses forward(step=False), including
                # one-token continuations. remote_step selects only distributed
                # execution; changing to decode here would change the algorithm.
                if kind == 2:
                    # Preserve the last successful report until this call succeeds.
                    layer.addresses[15] = _generation_zero_workspace(layer.report_workspace, layer.report_size)
                layer.attempted = True
                var result = layer.endpoint.call(operation, layer.addresses, layer.parameters, layer.floats)
                layer.cached = layer.cached + length if kind == 1 else result
                if kind == 2:
                    swap(layer.report, layer.report_workspace)
                layer.completed = True
                for j in range(len(layer.writeback_layouts)):  # driver-loop: successful layer publication before the next alias reads
                    layer.writeback_layouts[j].copy_from(layer.writeback_addresses[j])
                if layer.state_index >= 0:
                    self.state_counters[layer.state_index] = layer.cached
                hidden = next
                output_slot = 1 - output_slot
            if self.head_failure.byte_length() != 0:
                raise Error(self.head_failure)
            for i in range(len(self.head_refresh)):  # driver-loop: preserve head snapshot copy before norm and projection
                self.head_refresh[i].copy_to(self.head_targets[i])
            if self.norm != 0:
                var normalized = _generation_zero_workspace(self.output_workspace, count) if output_slot == 1 else _generation_zero_workspace(self.hidden_workspace, count)
                self.primitive_call(2, [normalized, hidden, self.norm],
                                    [rows, self.width], [self.norm_eps])
                hidden = normalized
            var output = _generation_zero_workspace(logits, rows * self.vocab)
            self.primitive_call(4, [output, hidden, self.head],
                                [rows, self.vocab, self.width], [])
            self.position += length
            self.forwards += 1
        except problem:
            self.failed = True
            raise problem


struct NativeGenerationProgress(Movable):
    var current: List[Int32]
    var next: List[Int32]
    var logits: List[Float32]
    var sampling: GenerationSamplingScratch
    var length: Int
    var produced: Int
    var wrote: Int
    var tokens: List[Int32]

    def __init__(out self, address: Int, batch: Int, length: Int, produced: Int, capacity: Int):
        self.current = List[Int32]()
        self.next = List[Int32]()
        self.logits = List[Float32]()
        self.sampling = GenerationSamplingScratch()
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
        model.forward_into(Int(progress.current.unsafe_ptr()), progress.length, progress.logits)
        progress.next.resize(model.batch, 0)
        causal_lm_pick_step(
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(progress.logits.unsafe_ptr())),
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(progress.next.unsafe_ptr())),
            model.batch, progress.length, model.vocab, finished, lengths, eos, pad, greedy,
            temperature, top_k, top_p, seed, progress.produced, progress.tokens, progress.sampling)
        # Failed selection leaves current intact for partial-progress publication.
        # Both owners survive every pointer use; next is private until success.
        swap(progress.current, progress.next)
        progress.length = 1
        progress.produced += 1
        progress.wrote += 1
