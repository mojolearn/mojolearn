# SPDX-License-Identifier: Apache-2.0
# The lm family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime LM_OPN_CONNECT = 0
comptime LM_OPN_PREPARE = 1
comptime LM_OPN_TOKENIZE = 2
comptime LM_OPN_CORPUS_INIT = 3
comptime LM_OPN_CORPUS_BATCHES = 4
comptime LM_OPN_CORPUS_VOCABULARY = 5
comptime LM_OPN_CORPUS_N_VOCAB = 6
comptime LM_OPN_CORPUS_TOKENIZER = 7
comptime LM_OPN_CORPUS_REPR = 8
comptime LM_OPN_BATCHES_INIT = 9
comptime LM_OPN_BATCHES_IDS = 10
comptime LM_OPN_BATCHES_PARALLEL_STEP = 11
comptime LM_OPN_BATCHES_PREFETCH = 12
comptime LM_OPN_BATCHES_VOCABULARY = 13
comptime LM_OPN_BATCHES_DESCRIBE = 14
comptime LM_OPN_BATCHES_SCHEDULE = 15
comptime LM_OPN_ITERATOR_ITER = 16
comptime LM_OPN_ITERATOR_NEXT = 17
comptime LM_OPN_EXPECTED_VOCABULARY = 18
comptime LM_OPN_REQUIRE_VOCABULARY = 19
comptime LM_OPN_TOKENIZER_FOR = 20


struct LmKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k___del__: PythonObject
    var k___get__: PythonObject
    var k___getattr__: PythonObject
    var k___mro__: PythonObject
    var k__drv: PythonObject
    var k__handle: PythonObject
    var k__parts: PythonObject
    var k__raw_paths: PythonObject
    var k__root: PythonObject
    var k__tokens_verified: PythonObject
    var k_admission_context: PythonObject
    var k_array_class: PythonObject
    var k_array_slots: PythonObject
    var k_batch: PythonObject
    var k_batch_ids_method: PythonObject
    var k_batches_class: PythonObject
    var k_class_dtypes: PythonObject
    var k_corpus_class: PythonObject
    var k_data_schedule: PythonObject
    var k_dir: PythonObject
    var k_hi: PythonObject
    var k_ids: PythonObject
    var k_ids_all: PythonObject
    var k_iterator_class: PythonObject
    var k_length: PythonObject
    var k_lo: PythonObject
    var k_manifest: PythonObject
    var k_manifest_sha256: PythonObject
    var k_mmap_class: PythonObject
    var k_mmap_hooks: PythonObject
    var k_modulus: PythonObject
    var k_n_vocab: PythonObject
    var k_native_module_type: PythonObject
    var k_scalar_connection: PythonObject
    var k_schema: PythonObject
    var k_sha256: PythonObject
    var k_source: PythonObject
    var k_tokens: PythonObject
    var k_tokens_dir: PythonObject
    var k_vocabulary: PythonObject
    var k_vocabulary_path: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___get__ = PythonObject(None)
        self.k___getattr__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__drv = PythonObject(None)
        self.k__handle = PythonObject(None)
        self.k__parts = PythonObject(None)
        self.k__raw_paths = PythonObject(None)
        self.k__root = PythonObject(None)
        self.k__tokens_verified = PythonObject(None)
        self.k_admission_context = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_slots = PythonObject(None)
        self.k_batch = PythonObject(None)
        self.k_batch_ids_method = PythonObject(None)
        self.k_batches_class = PythonObject(None)
        self.k_class_dtypes = PythonObject(None)
        self.k_corpus_class = PythonObject(None)
        self.k_data_schedule = PythonObject(None)
        self.k_dir = PythonObject(None)
        self.k_hi = PythonObject(None)
        self.k_ids = PythonObject(None)
        self.k_ids_all = PythonObject(None)
        self.k_iterator_class = PythonObject(None)
        self.k_length = PythonObject(None)
        self.k_lo = PythonObject(None)
        self.k_manifest = PythonObject(None)
        self.k_manifest_sha256 = PythonObject(None)
        self.k_mmap_class = PythonObject(None)
        self.k_mmap_hooks = PythonObject(None)
        self.k_modulus = PythonObject(None)
        self.k_n_vocab = PythonObject(None)
        self.k_native_module_type = PythonObject(None)
        self.k_scalar_connection = PythonObject(None)
        self.k_schema = PythonObject(None)
        self.k_sha256 = PythonObject(None)
        self.k_source = PythonObject(None)
        self.k_tokens = PythonObject(None)
        self.k_tokens_dir = PythonObject(None)
        self.k_vocabulary = PythonObject(None)
        self.k_vocabulary_path = PythonObject(None)


comptime _LM_KEYS = SharedSlot[LmKeys, "MojolearnLmKeysV1"]


def _lm_keys_fill(nk: UnsafePointer[LmKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k___del__ = py_key("__del__")
    nk[].k___get__ = py_key("__get__")
    nk[].k___getattr__ = py_key("__getattr__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__drv = py_key("_drv")
    nk[].k__handle = py_key("_handle")
    nk[].k__parts = py_key("_parts")
    nk[].k__raw_paths = py_key("_raw_paths")
    nk[].k__root = py_key("_root")
    nk[].k__tokens_verified = py_key("_tokens_verified")
    nk[].k_admission_context = py_key("admission_context")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_slots = py_key("array_slots")
    nk[].k_batch = py_key("batch")
    nk[].k_batch_ids_method = py_key("batch_ids_method")
    nk[].k_batches_class = py_key("batches_class")
    nk[].k_class_dtypes = py_key("class_dtypes")
    nk[].k_corpus_class = py_key("corpus_class")
    nk[].k_data_schedule = py_key("data_schedule")
    nk[].k_dir = py_key("dir")
    nk[].k_hi = py_key("hi")
    nk[].k_ids = py_key("ids")
    nk[].k_ids_all = py_key("ids_all")
    nk[].k_iterator_class = py_key("iterator_class")
    nk[].k_length = py_key("length")
    nk[].k_lo = py_key("lo")
    nk[].k_manifest = py_key("manifest")
    nk[].k_manifest_sha256 = py_key("manifest_sha256")
    nk[].k_mmap_class = py_key("mmap_class")
    nk[].k_mmap_hooks = py_key("mmap_hooks")
    nk[].k_modulus = py_key("modulus")
    nk[].k_n_vocab = py_key("n_vocab")
    nk[].k_native_module_type = py_key("native_module_type")
    nk[].k_scalar_connection = py_key("scalar_connection")
    nk[].k_schema = py_key("schema")
    nk[].k_sha256 = py_key("sha256")
    nk[].k_source = py_key("source")
    nk[].k_tokens = py_key("tokens")
    nk[].k_tokens_dir = py_key("tokens_dir")
    nk[].k_vocabulary = py_key("vocabulary")
    nk[].k_vocabulary_path = py_key("vocabulary_path")
    var ops = Python.dict()
    py_op_add(ops, "connect", LM_OPN_CONNECT)
    py_op_add(ops, "prepare", LM_OPN_PREPARE)
    py_op_add(ops, "tokenize", LM_OPN_TOKENIZE)
    py_op_add(ops, "corpus_init", LM_OPN_CORPUS_INIT)
    py_op_add(ops, "corpus_batches", LM_OPN_CORPUS_BATCHES)
    py_op_add(ops, "corpus_vocabulary", LM_OPN_CORPUS_VOCABULARY)
    py_op_add(ops, "corpus_n_vocab", LM_OPN_CORPUS_N_VOCAB)
    py_op_add(ops, "corpus_tokenizer", LM_OPN_CORPUS_TOKENIZER)
    py_op_add(ops, "corpus_repr", LM_OPN_CORPUS_REPR)
    py_op_add(ops, "batches_init", LM_OPN_BATCHES_INIT)
    py_op_add(ops, "batches_ids", LM_OPN_BATCHES_IDS)
    py_op_add(ops, "batches_parallel_step", LM_OPN_BATCHES_PARALLEL_STEP)
    py_op_add(ops, "batches_prefetch", LM_OPN_BATCHES_PREFETCH)
    py_op_add(ops, "batches_vocabulary", LM_OPN_BATCHES_VOCABULARY)
    py_op_add(ops, "batches_describe", LM_OPN_BATCHES_DESCRIBE)
    py_op_add(ops, "batches_schedule", LM_OPN_BATCHES_SCHEDULE)
    py_op_add(ops, "iterator_iter", LM_OPN_ITERATOR_ITER)
    py_op_add(ops, "iterator_next", LM_OPN_ITERATOR_NEXT)
    py_op_add(ops, "expected_vocabulary", LM_OPN_EXPECTED_VOCABULARY)
    py_op_add(ops, "require_vocabulary", LM_OPN_REQUIRE_VOCABULARY)
    py_op_add(ops, "tokenizer_for", LM_OPN_TOKENIZER_FOR)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def lm_keys() raises -> UnsafePointer[LmKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _LM_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _lm_keys_fill(nk)
    return nk
