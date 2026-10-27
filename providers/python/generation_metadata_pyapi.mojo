# SPDX-License-Identifier: Apache-2.0
# Operation-local Python ownership/admission for native generation metadata.
# No snapshot survives a public forward/generate/read call. A stream's next
# read recaptures current model fields and owners; this adds no thread-safety
# promise for concurrent model mutation during execution.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_type, py_call, py_contains
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_address
from core.generation_metadata import GenerationLayerWords


struct GenerationWeightSnapshot(Copyable, Movable):
    var owners: List[PythonObject]
    var addresses: List[Int]
    var refresh: List[Int]

    def __init__(out self, values: PythonObject, zero_from: Int = -1) raises:
        self.owners = List[PythonObject]()
        self.addresses = List[Int]()
        self.refresh = List[Int]()
        var cls = GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls
        for j in range(len(values)):  # driver-loop: admit each stable weight slot once
            var value = values[j]
            self.owners.append(value)
            if value is None or (zero_from >= 0 and j >= zero_from):
                self.addresses.append(0)
            elif py_type(value) is cls:
                # generation_address checks C order. The exact Array's owner
                # remains in self.owners after its native address is captured.
                var keep = List[PythonObject]()
                self.addresses.append(generation_address(value, keep))
            else:
                # The legacy foreign-buffer route copies bytes on each use.
                # Preserve that behavior (including aliases to mutable state),
                # rather than silently freezing a copied weight for this call.
                self.addresses.append(0)
                self.refresh.append(j)

    def resolve(self, mut keep: List[PythonObject]) raises -> List[Int]:
        var result = self.addresses.copy()
        for k in range(len(self.refresh)):  # driver-loop: retained foreign-buffer copy compatibility
            var index = self.refresh[k]
            result[index] = generation_address(self.owners[index], keep)
        return result^


struct GenerationLayerSnapshot(Copyable, Movable):
    var block: PythonObject
    var words: GenerationLayerWords
    var weights: GenerationWeightSnapshot
    var optional: GenerationWeightSnapshot
    var packed: GenerationWeightSnapshot
    var profile: PythonObject
    var profile_changed: Bool

    def __init__(out self, block: PythonObject, kind: String, planes: Bool) raises:
        self.block = block
        self.weights = GenerationWeightSnapshot(block["_w"], 2 if planes else -1)
        self.profile = PythonObject(None)
        self.profile_changed = False
        var parameters = List[Int]()
        var inner = 0
        var heads = 0
        var conv = 0
        var low = Float64(0)
        var high = Float64(0)
        var optional = Python.list()
        var packed = Python.list()
        if kind == "transformer":
            parameters = [0, 0, Int(py=block["d_model"]), Int(py=block["n_heads"]),
                          Int(py=block["n_kv_heads"]), Int(py=block["head_dim"]),
                          Int(py=block["intermediate"]), 0, 0, Int(py=block["window"])]
            var tail = block["_opts_tail"]
            for j in range(len(tail)):  # driver-loop: admitted immutable transformer option words
                parameters.append(Int(py=tail[j]))
            optional = block["_wopt"]
            if planes:
                var triples = block["_int15"]
                for j in range(len(triples)):  # driver-loop: packed weight plane registry
                    var trip = triples[j]
                    for k in range(3):  # small-loop(int15 plane triple)
                        _ = py_call(packed.append, PythonObject(None) if trip is None else trip[k])
            self.profile = py_get(block, "_arithmetic_profile", PythonObject(None))
            self.profile_changed = Bool(py=py_get(block, "_arithmetic_profile_changed", PythonObject(False)))
        elif kind == "mamba1":
            inner = Int(py=block["d_inner"])
        else:
            heads = Int(py=block["nheads"])
            conv = Int(py=block["conv_dim"])
            var limits = block["dt_limit"]
            low = Float64(py=limits[0])
            high = Float64(py=limits[1])
        self.words = GenerationLayerWords(parameters^, inner, heads, conv, low, high)
        self.optional = GenerationWeightSnapshot(optional)
        self.packed = GenerationWeightSnapshot(packed)


struct GenerationHeadSnapshot(Copyable, Movable):
    var entries: PythonObject
    var native: PythonObject
    var width: Int
    var vocab: Int
    var normalize: Bool
    var packed: Bool
    var norm_eps: Float64
    var weights: GenerationWeightSnapshot

    def __init__(out self, model: PythonObject, entries: PythonObject, width: Int, vocab: Int,
                 normalize: Bool, packed: Bool) raises:
        self.entries = entries
        self.native = py_get(entries, "native_primitives")
        self.width = width
        self.vocab = vocab
        var norm = model._norm if normalize else PythonObject(None)
        var planes = model._head_int15
        self.normalize = norm is not None
        self.packed = planes is not None and packed
        self.norm_eps = Float64(py=model.norm_eps) if self.normalize else Float64(0)
        var values = Python.list(norm, PythonObject(None) if self.packed else model._head)
        if self.packed:
            for k in range(3):  # small-loop(int15 head plane triple)
                _ = py_call(values.append, planes[k])
        self.weights = GenerationWeightSnapshot(values)


struct GenerationOperationSnapshot(Copyable, Movable):
    var model: PythonObject
    var entries: PythonObject
    var native: PythonObject
    var kind: String
    var width: Int
    var vocab: Int
    var planes: Bool
    var embedding: GenerationWeightSnapshot
    var head: GenerationHeadSnapshot
    var layers: List[GenerationLayerSnapshot]

    def __init__(out self, model: PythonObject, entries: PythonObject, blocks: List[PythonObject]) raises:
        self.model = model
        self.entries = entries
        self.native = py_get(entries, "native_primitives")
        self.kind = String(model.kind)
        self.width = Int(py=model.d_model)
        self.vocab = Int(py=model.vocab_size)
        self.embedding = GenerationWeightSnapshot(Python.list(model._embed))
        self.head = GenerationHeadSnapshot(model, entries, self.width, self.vocab, True, True)
        self.planes = self.head.packed and py_contains(entries, "forward15")
        self.layers = List[GenerationLayerSnapshot]()
        for i in range(len(blocks)):  # driver-loop: capture stable layer metadata once per operation
            self.layers.append(GenerationLayerSnapshot(blocks[i], self.kind, self.planes))
