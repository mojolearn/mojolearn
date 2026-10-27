# SPDX-License-Identifier: Apache-2.0
# Native mixed-layer execution state. All object/buffer metadata is admitted
# before this engine runs. Endpoints manage only the common GIL/lock lifecycle;
# every numerical operation remains in its existing table implementation.
from core.native_execution import NativeExecutionEndpoint
from core.borrowed_buffer import BorrowedBufferLayout


struct NativeSambaLayer(Movable):
    var endpoint: NativeExecutionEndpoint
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
        if self.prepare_operation >= 0:
            self.prepared_result = self.endpoint.call(self.prepare_operation, self.prepare_addresses, self.prepare_parameters, [])
            self.prepared = True
            self.parameters[self.generation_slot] = self.prepared_result
        for i in range(len(self.refresh_layouts)):  # driver-loop: refresh admitted aliased foreign state at original layer position
            self.refresh_layouts[i].copy_to(self.refresh_addresses[i])
        var output = List[Float32](length=elements, fill=0)
        self.addresses[self.input_slot] = input_address
        self.addresses[self.output_slot] = Int(output.unsafe_ptr())
        for i in range(len(self.report_sizes)):  # driver-loop: four optional native Mamba report buffers
            self.reports.append(List[Float32](length=self.report_sizes[i], fill=0))
            self.addresses[self.report_slot + i] = Int(self.reports[i].unsafe_ptr()) if self.report_sizes[i] != 0 else 0
        self.attempted = True
        self.result = self.endpoint.call(self.operation, self.addresses, self.parameters, self.scalars)
        for i in range(len(self.refresh_layouts)):  # driver-loop: publish completed layer state before the next native layer
            self.refresh_layouts[i].copy_from(self.refresh_addresses[i])
        self.completed = True
        return output^


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

    def forward_hidden(mut self, ids: Int, retain_inputs: Bool) raises:
        var elements = self.rows * self.width
        var hidden = List[Float32](length=elements, fill=0)
        _ = self.primitive.call(0, [Int(hidden.unsafe_ptr()), self.embedding, ids], [self.rows, self.vocab, self.width], [])
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
