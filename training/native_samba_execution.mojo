# SPDX-License-Identifier: Apache-2.0
# Native mixed-layer execution state. All object/buffer metadata is admitted
# before this engine runs. Endpoints manage only the common GIL/lock lifecycle;
# every numerical operation remains in its existing table implementation.
from core.native_execution import NativeExecutionEndpoint
from core.borrowed_buffer import BorrowedBufferLayout
from core.samba_count_impl import samba_target_count
from core.host_helpers_impl import all_finite_f32
from core.hostptr import f32_ptr, i32_ptr
from std.memory import memcpy


struct NativeSambaLayer(Movable):
    var endpoint: NativeExecutionEndpoint
    var backward_endpoint: NativeExecutionEndpoint
    var close_endpoint: NativeExecutionEndpoint
    var backward_operation: Int
    var close_operation: Int
    var reset_operation: Int
    var tape_started: Bool
    var backward_parameters: List[Int]
    var backward_weights: List[Int]
    var gradient_sizes: List[Int]
    var gradients: List[List[Float32]]
    var tape: Bool
    var tape_live: Bool
    var owned_backward: Bool
    var validate_gradient: Bool
    var backward_failure: String
    var operation: Int
    var addresses: List[Int]
    var parameters: List[Int]
    var scalars: List[Float64]
    var input_slot: Int
    var output_slot: Int
    var report_slot: Int
    var report_sizes: List[Int]
    var reports: List[List[Float32]]
    var input: List[Float32]
    var refresh_layouts: List[BorrowedBufferLayout]
    var refresh_addresses: List[Int]
    var prepare_operation: Int
    var prepare_addresses: List[Int]
    var prepare_parameters: List[Int]
    var generation_slot: Int
    var prepared: Bool
    var prepared_result: Int
    var state_index: Int
    var counter_slot: Int
    var pending_slot: Int
    var result: Int
    var visited: Bool
    var attempted: Bool
    var completed: Bool
    var failure: String

    def __init__(out self, endpoint: NativeExecutionEndpoint, operation: Int,
                 var addresses: List[Int], var parameters: List[Int], var scalars: List[Float64],
                 input_slot: Int, output_slot: Int, report_slot: Int, var report_sizes: List[Int]):
        self.endpoint = endpoint
        self.backward_endpoint = endpoint
        self.close_endpoint = endpoint
        self.backward_operation = -1
        self.close_operation = -1
        self.backward_parameters = List[Int]()
        self.backward_weights = List[Int]()
        self.gradient_sizes = List[Int]()
        self.gradients = List[List[Float32]]()
        self.tape = False
        self.reset_operation = -1
        self.tape_started = False
        self.tape_live = False
        self.owned_backward = False
        self.validate_gradient = False
        self.backward_failure = String("")
        self.operation = operation
        self.addresses = addresses^
        self.parameters = parameters^
        self.scalars = scalars^
        self.input_slot = input_slot
        self.output_slot = output_slot
        self.report_slot = report_slot
        self.report_sizes = report_sizes^
        self.reports = List[List[Float32]]()
        self.input = List[Float32]()
        self.refresh_layouts = List[BorrowedBufferLayout]()
        self.refresh_addresses = List[Int]()
        self.prepare_operation = -1
        self.prepare_addresses = List[Int]()
        self.prepare_parameters = List[Int]()
        self.generation_slot = -1
        self.prepared = False
        self.prepared_result = 0
        self.state_index = -1
        self.counter_slot = -1
        self.pending_slot = -1
        self.result = 0
        self.visited = False
        self.attempted = False
        self.completed = False
        self.failure = String("")

    def forward(mut self, input_address: Int, elements: Int) raises -> List[Float32]:
        self.visited = True
        if self.failure.byte_length() != 0:
            raise Error(self.failure)
        if self.tape and self.tape_started:
            _ = self.close_endpoint.call(self.reset_operation, [], [], [])
        if self.prepare_operation >= 0 and not self.prepared:
            self.prepared_result = self.endpoint.call(self.prepare_operation, self.prepare_addresses, self.prepare_parameters, [])
            self.prepared = True
            self.parameters[self.generation_slot] = self.prepared_result
        for i in range(len(self.refresh_layouts)):  # driver-loop: refresh admitted aliased foreign state at original layer position
            self.refresh_layouts[i].copy_to(self.refresh_addresses[i])
        var output = List[Float32](length=elements, fill=0)
        self.addresses[self.input_slot] = input_address
        self.addresses[self.output_slot] = Int(output.unsafe_ptr())
        var reports = List[List[Float32]]()
        for i in range(len(self.report_sizes)):  # driver-loop: four optional native Mamba report buffers
            reports.append(List[Float32](length=self.report_sizes[i], fill=0))
            self.addresses[self.report_slot + i] = Int(reports[i].unsafe_ptr()) if self.report_sizes[i] != 0 else 0
        self.attempted = True
        self.result = self.endpoint.call(self.operation, self.addresses, self.parameters, self.scalars)
        self.tape_started = self.tape
        self.tape_live = self.tape
        self.reports = reports^
        for i in range(len(self.refresh_layouts)):  # driver-loop: publish completed layer state before the next native layer
            self.refresh_layouts[i].copy_from(self.refresh_addresses[i])
        self.completed = True
        return output^


    def backward(mut self, dy: Int, elements: Int) raises -> List[Float32]:
        if self.backward_failure.byte_length() != 0:
            raise Error(self.backward_failure)
        if self.validate_gradient and all_finite_f32(f32_ptr(dy), elements) != 1:
            raise Error("neural backward: grad_output must be finite")
        var dx = List[Float32](length=elements, fill=0)
        var a = List[Int]()
        if not self.tape:
            a.append(Int(self.input.unsafe_ptr()))
            if not self.owned_backward:
                for address in self.backward_weights:  # driver-loop: admitted immutable gradient input slots
                    a.append(address)
        a.append(dy)
        a.append(Int(dx.unsafe_ptr()))
        self.gradients.clear()
        for count in self.gradient_sizes:  # driver-loop: native parameter gradient allocation
            self.gradients.append(List[Float32](length=count, fill=0))
            a.append(Int(self.gradients[len(self.gradients) - 1].unsafe_ptr()))
        if self.owned_backward and self.prepared:
            self.backward_parameters[3] = self.prepared_result
        _ = self.backward_endpoint.call(self.backward_operation, a, self.backward_parameters, [])
        if self.tape_live:
            self.close()
        return dx^

    def close(mut self) raises:
        if self.tape_live:
            _ = self.close_endpoint.call(self.close_operation, [], [], [])
            self.tape_live = False


struct NativeSambaModel(Movable):
    var primitive: NativeExecutionEndpoint
    var embedding: Int
    var norm: Int
    var head: Int
    var norm_eps: Float64
    var rows: Int
    var width: Int
    var vocab: Int
    var layers: List[NativeSambaLayer]
    var state_counters: List[Int]
    var state_pending: List[Int]
    var normalized: List[Float32]
    var last_hidden: List[Float32]
    var forward_complete: Bool
    var dropout: Bool
    var dropout_words: List[Int]
    var dropout_scalars: List[Float64]
    var norm_gradient: List[Float32]
    var head_gradient: List[Float32]
    var embedding_gradient: List[Float32]

    def __init__(out self, primitive: NativeExecutionEndpoint, embedding: Int, norm: Int, head: Int,
                 norm_eps: Float64, rows: Int, width: Int, vocab: Int):
        self.primitive = primitive
        self.embedding = embedding
        self.norm = norm
        self.head = head
        self.norm_eps = norm_eps
        self.rows = rows
        self.width = width
        self.vocab = vocab
        self.layers = List[NativeSambaLayer]()
        self.state_counters = List[Int]()
        self.state_pending = List[Int]()
        self.normalized = List[Float32]()
        self.last_hidden = List[Float32]()
        self.forward_complete = False
        self.dropout = False
        self.dropout_words = List[Int]()
        self.dropout_scalars = List[Float64]()
        self.norm_gradient = List[Float32]()
        self.head_gradient = List[Float32]()
        self.embedding_gradient = List[Float32]()

    def forward_hidden(mut self, ids: Int, retain_inputs: Bool) raises:
        var elements = self.rows * self.width
        var hidden = List[Float32](length=elements, fill=0)
        _ = self.primitive.call(0, [Int(hidden.unsafe_ptr()), self.embedding, ids], [self.rows, self.vocab, self.width], [])
        if retain_inputs and self.dropout:
            var dropped = List[Float32](length=elements, fill=0)
            var words = self.dropout_words.copy()
            words[5] = 2
            _ = self.primitive.call(6, [Int(dropped.unsafe_ptr()), Int(hidden.unsafe_ptr())], words, self.dropout_scalars)
            hidden = dropped^
        for i in range(len(self.layers)):  # driver-loop: native ordered mixed-layer descriptors
            var state_index = self.layers[i].state_index
            if state_index >= 0:
                self.layers[i].parameters[self.layers[i].counter_slot] = self.state_counters[state_index]
                if self.layers[i].pending_slot >= 0:
                    self.layers[i].parameters[self.layers[i].pending_slot] = self.state_pending[state_index]
            var next = self.layers[i].forward(Int(hidden.unsafe_ptr()), elements)
            if state_index >= 0:
                self.state_counters[state_index] = self.layers[i].result
                self.state_pending[state_index] = 0
            if retain_inputs:
                self.layers[i].input = hidden^
            else:
                _ = hidden^
            hidden = next^
        self.last_hidden = hidden^
        self.normalized = List[Float32](length=elements, fill=0)
        _ = self.primitive.call(2, [Int(self.normalized.unsafe_ptr()), Int(self.last_hidden.unsafe_ptr()), self.norm],
            [self.rows, self.width], [self.norm_eps])

    def forward(mut self, ids: Int) raises -> List[Float32]:
        self.forward_hidden(ids, False)
        var logits = List[Float32](length=self.rows * self.vocab, fill=0)
        _ = self.primitive.call(4, [Int(logits.unsafe_ptr()), Int(self.normalized.unsafe_ptr()), self.head],
            [self.rows, self.vocab, self.width], [])
        self.forward_complete = True
        return logits^


    def close_tapes(mut self) raises:
        for i in range(len(self.layers)):  # driver-loop: release only live native forward snapshots
            self.layers[i].close()

    def gradients(mut self, ids: Int, targets: Int, requested_items: Int, automatic_items: Bool, tied: Bool,
                  fused: Bool, chunked: Bool) raises -> Float64:
        var count = samba_target_count(targets, self.rows)
        var items = count if automatic_items else requested_items
        try:
            self.forward_hidden(ids, True)
            var elements = self.rows * self.width
            var loss = List[Float32](length=1, fill=0)
            var row_loss = List[Float32](length=self.rows, fill=0)
            var dh = List[Float32](length=elements, fill=0)
            var dnormalized = List[Float32](length=elements, fill=0)
            self.norm_gradient = List[Float32](length=self.width, fill=0)
            self.head_gradient = List[Float32](length=self.vocab * self.width, fill=0)
            if chunked and count == self.rows and items == count:
                var denominator = List[Float32](length=self.rows, fill=0)
                _ = self.primitive.call(11, [Int(loss.unsafe_ptr()), Int(row_loss.unsafe_ptr()), Int(denominator.unsafe_ptr()),
                    Int(dnormalized.unsafe_ptr()), Int(self.head_gradient.unsafe_ptr()), Int(self.normalized.unsafe_ptr()),
                    self.head, targets], [self.rows, self.vocab, self.width], [])
            elif fused:
                _ = self.primitive.call(7, [Int(loss.unsafe_ptr()), Int(row_loss.unsafe_ptr()), Int(dnormalized.unsafe_ptr()),
                    Int(self.head_gradient.unsafe_ptr()), Int(self.normalized.unsafe_ptr()), self.head, targets],
                    [self.rows, self.vocab, self.width, -100, 1, items], [0.0])
            else:
                var logits = List[Float32](length=self.rows * self.vocab, fill=0)
                var dlogits = List[Float32](length=self.rows * self.vocab, fill=0)
                _ = self.primitive.call(4, [Int(logits.unsafe_ptr()), Int(self.normalized.unsafe_ptr()), self.head],
                    [self.rows, self.vocab, self.width], [])
                _ = self.primitive.call(8, [Int(loss.unsafe_ptr()), Int(row_loss.unsafe_ptr()), Int(dlogits.unsafe_ptr()),
                    Int(logits.unsafe_ptr()), targets], [self.rows, self.vocab, -100, 1, items, 1], [0.0])
                _ = self.primitive.call(5, [Int(dnormalized.unsafe_ptr()), Int(self.head_gradient.unsafe_ptr()), Int(dlogits.unsafe_ptr()),
                    Int(self.normalized.unsafe_ptr()), self.head], [self.rows, self.vocab, self.width], [])
            _ = self.primitive.call(3, [Int(dh.unsafe_ptr()), Int(self.norm_gradient.unsafe_ptr()), Int(dnormalized.unsafe_ptr()),
                Int(self.last_hidden.unsafe_ptr()), self.norm], [self.rows, self.width], [self.norm_eps])
            for reverse in range(len(self.layers)):  # driver-loop: complete native reverse layer/tape walk
                var i = len(self.layers) - reverse - 1
                var next = self.layers[i].backward(Int(dh.unsafe_ptr()), elements)
                dh = next^
            if self.dropout:
                var dropped = List[Float32](length=elements, fill=0)
                var words = self.dropout_words.copy()
                words[5] = 3
                _ = self.primitive.call(6, [Int(dropped.unsafe_ptr()), Int(dh.unsafe_ptr())], words, self.dropout_scalars)
                dh = dropped^
            self.embedding_gradient = List[Float32](length=self.vocab * self.width, fill=0)
            _ = self.primitive.call(1, [Int(self.embedding_gradient.unsafe_ptr()), Int(dh.unsafe_ptr()), ids],
                [self.rows, self.vocab, self.width], [])
            if tied:
                var n = self.vocab * self.width
                var pair = List[Float32](length=2 * n, fill=0)
                memcpy(dest=pair.unsafe_ptr(), src=self.embedding_gradient.unsafe_ptr(), count=n)
                memcpy(dest=pair.unsafe_ptr() + n, src=self.head_gradient.unsafe_ptr(), count=n)
                _ = self.primitive.call(9, [Int(self.embedding_gradient.unsafe_ptr()), Int(pair.unsafe_ptr())], [n, 2, -1], [])
            self.close_tapes()
            return Float64(loss[0])
        except problem:
            self.close_tapes()
            raise problem


    def gradient_address(self, kind: Int, layer: Int, slot: Int) -> Int:
        if kind == 0:
            return Int(self.embedding_gradient.unsafe_ptr())
        if kind == 1:
            return Int(self.norm_gradient.unsafe_ptr())
        if kind == 2:
            return Int(self.head_gradient.unsafe_ptr())
        return Int(self.layers[layer].gradients[slot].unsafe_ptr())

    def accumulated_gradients(mut self, ids: Int, targets: Int, accumulation: Int, total_count: Int,
                              tied: Bool, fused: Bool, chunked: Bool, offsets: List[Int],
                              kinds: List[Int], layers: List[Int], slots: List[Int], output: Int, registry_failure: String) raises -> Float64:
        var n = offsets[len(offsets) - 1] if len(offsets) != 0 else 0
        var stacked = List[Float32](length=accumulation * n, fill=0)
        var losses = List[Float32](length=accumulation, fill=0)
        var base_offset = self.dropout_words[1] if self.dropout else 0
        for k in range(accumulation):  # driver-loop: whole native microbatch/tape/gradient traversal
            if self.dropout:
                self.dropout_words[1] = base_offset + k * self.rows * self.width
            var xpart = List[Int32](length=self.rows, fill=0)
            var ypart = List[Int32](length=self.rows, fill=0)
            memcpy(dest=xpart.unsafe_ptr(), src=i32_ptr(ids) + k * self.rows, count=self.rows)
            memcpy(dest=ypart.unsafe_ptr(), src=i32_ptr(targets) + k * self.rows, count=self.rows)
            losses[k] = Float32(self.gradients(Int(xpart.unsafe_ptr()), Int(ypart.unsafe_ptr()),
                total_count, False, tied, fused, chunked))
            if registry_failure.byte_length() != 0:
                raise Error(registry_failure)
            for j in range(len(kinds)):  # driver-loop: native ordered gradient-registry copies
                memcpy(dest=stacked.unsafe_ptr() + k * n + offsets[j],
                    src=f32_ptr(self.gradient_address(kinds[j], layers[j], slots[j])), count=offsets[j + 1] - offsets[j])
        if accumulation == 1:
            memcpy(dest=f32_ptr(output), src=stacked.unsafe_ptr(), count=n)
            return Float64(losses[0])
        _ = self.primitive.call(9, [output, Int(stacked.unsafe_ptr())], [n, accumulation, self.rows * accumulation], [])
        var total = List[Float32](length=1, fill=0)
        _ = self.primitive.call(9, [Int(total.unsafe_ptr()), Int(losses.unsafe_ptr())], [1, accumulation, -1], [])
        return Float64(total[0])
