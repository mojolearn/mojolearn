# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the neural-blocks family (lane
calltable-neural-blocks, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. The public neural block
operations (`bindings/neural_block_public.mojo`: `TransformerBlock`,
`Mamba1Block`, `Mamba2Block`, `Mamba3Block` forward and backward) call two
sibling images per route. Each image publishes the descriptor entry it already
carries (the `*_api_v1` capsule transport of `core/native_descriptor_abi.mojo`)
here at module init, once per (tier, vendor) key:

  NEURAL_TABLE_BLOCK_FORWARD  the Transformer block image a block's forward
                              connects (`_model_bindings`): GPU
                              `transformer/native_block_dispatch.mojo::
                              transformer_block_v1` (`_mojolearn_transformer`),
                              CPU `core/neural_host_dispatch.mojo` family 2
                              (`_mojolearn_neural_host`). TRANSFORMER_BLOCK_ABI_NAME.
  NEURAL_TABLE_BLOCK_TRAIN    the Transformer image a block's backward connects
                              (`_native_neural_bridge.connect_training`): GPU
                              `_mojolearn_transformer` (the same entry), CPU
                              `transformer/host/native_block_dispatch.mojo::
                              transformer_host_block_v1` (`_mojolearn_transformer_host`).
  NEURAL_TABLE_MAMBA_TRAIN    the Mamba image a block's backward connects: GPU
                              `mamba/native_block_dispatch.mojo` (`_mojolearn_mamba`),
                              CPU `mamba/host/native_block_dispatch.mojo`
                              (`_mojolearn_mamba_host`). MAMBA_BLOCK_ABI_NAME.
  NEURAL_TABLE_MLP_PROGRAM    SmallMLPTrainer's captured program
                              (`autograd/small_mlp_program.mojo`): GPU
                              `sequence/small_mlp_entry_device.mojo`
                              (`_mojolearn_x_sequence`), CPU
                              `sequence/small_mlp_entry_host.mojo`
                              (`_mojolearn_x_sequence_host`). NEURAL_MLP_PROGRAM_ABI;
                              operations are the program's own codes
                              (SMALL_MLP_PROGRAM_OPEN .. SMALL_MLP_PROGRAM_SESSIONS).
  NEURAL_TABLE_EMBEDDING      the Embedding layer's training binding
                              (`_backend.binding("_mojolearn_embedding", mode)`):
                              GPU `_mojolearn_embedding`, CPU
                              `_mojolearn_embedding_host`. NEURAL_EMBEDDING_ABI.
  NEURAL_TABLE_EMBEDDING_INFER the host inference lookup binding
                              (`_mojolearn_embedding_infer_host`, CPU only):
                              forward only.

The operation codes are the images' own (documented beside each dispatch):
Transformer 0/1 reserve/release a session, 2/3 session forward/backward,
4/5 ephemeral forward/backward; Mamba 0/1 reserve/release, 2/3/4 Mamba1/2/3
backward, 7 session backward, 9 owned backward, 15 owned snapshot install.

A caller keys the read by the block's admitted tier and vendor
(`NeuralEntryKey`) and calls the pointer with plain spans; no Python object
crosses. Extending a table's layout requires a new V suffix, never a new
loader. Publication and lookup are small metadata operations with no
independent numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word, entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall, TRANSFORMER_BLOCK_ABI_NAME, MAMBA_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_words


def neural_entry_spans(aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    max_floats: Int) raises -> List[List[Int]]:
    """A descriptor entry's admission of its spans: [addresses, ints]."""
    if result == 0 or nf < 0 or nf > max_floats or (nf > 0 and ff == 0):
        raise Error("native neural entry: null result or invalid float span")
    var spans = List[List[Int]]()
    spans.append(native_descriptor_words(aa, na))
    spans.append(native_descriptor_words(pp, np))
    return spans^


def neural_entry_result(result: Int, value: Int):
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value

comptime NEURAL_TABLE_BLOCK_FORWARD = 0
comptime NEURAL_TABLE_BLOCK_TRAIN = 1
comptime NEURAL_TABLE_MAMBA_TRAIN = 2
comptime NEURAL_TABLE_MLP_PROGRAM = 3
comptime NEURAL_TABLE_EMBEDDING = 4
comptime NEURAL_TABLE_EMBEDDING_INFER = 5
comptime NEURAL_TABLE_LAST = 5

# The operation counts every publisher and reader agree on: the readers'
# former capsule admissions (`neural_descriptor_reuse(..., 36)` and `(..., 44)`).
# The block table gained op 36, NEURAL_BLOCK_SESSION_CREATE (lane
# transformer-session-native, 2026-11-12); the capsule admissions still name 36
# operations and never call it.
comptime NEURAL_BLOCK_OPERATIONS = 37
# The Mamba table gained ops 44..46 (create, info, capabilities of the Mamba-3
# prefill session registry; lane mamba3-prefill-handle, 2026-11-12); the
# capsule admissions still name 44 operations and never call them.
comptime NEURAL_MAMBA_OPERATIONS = 47
# The small MLP program: open, step, download, apply, close, sessions.
comptime NEURAL_MLP_PROGRAM_ABI = "mojolearn.neural.small_mlp_program.v1"
comptime NEURAL_MLP_PROGRAM_OPERATIONS = 6
# The Embedding entries (a: addresses, p: Int words; the former Python
# entries' words in their order, resident tensor handles first):
#   EMB_OP_FORWARD        a [weight, ids, y]        p [V, d, T(, token)]     -> T * d
#   EMB_OP_BACKWARD       a [dy, ids, dw]           p [V, d, T, pad, acc(, plan)] -> V * d
#   EMB_OP_TABLE_RELEASE  -                         p [token]                -> 1 dropped, 0 not
#   EMB_OP_FORWARD_DEV    a [weight, ids]           p [y_h, V, d, T, token]  -> T * d
#   EMB_OP_BACKWARD_DEV   a [ids]                   p [dy_h, dw_h, V, d, T, pad, acc, plan] -> V * d
#   EMB_OP_TENSOR_ALLOC   -                         p [n, zero]              -> handle
#   EMB_OP_TENSOR_FREE    -                         p [handle]               -> handle
#   EMB_OP_TENSOR_UPLOAD  a [source]                p [handle, n]            -> n
#   EMB_OP_TENSOR_COPY    -                         p [dst, src, n]          -> n
# A host image serves FORWARD (and BACKWARD on the training host) only.
comptime NEURAL_EMBEDDING_ABI = "mojolearn.neural.embedding.v1"
comptime NEURAL_EMB_OPERATIONS = 9
comptime EMB_OP_FORWARD = 0
comptime EMB_OP_BACKWARD = 1
comptime EMB_OP_TABLE_RELEASE = 2
comptime EMB_OP_FORWARD_DEV = 3
comptime EMB_OP_BACKWARD_DEV = 4
comptime EMB_OP_TENSOR_ALLOC = 5
comptime EMB_OP_TENSOR_FREE = 6
comptime EMB_OP_TENSOR_UPLOAD = 7
comptime EMB_OP_TENSOR_COPY = 8

# Transformer block operation codes (transformer/native_block_dispatch.mojo).
comptime NEURAL_BLOCK_RESERVE = 0
comptime NEURAL_BLOCK_RELEASE = 1
comptime NEURAL_BLOCK_SESSION_FORWARD_FRESH = 2
comptime NEURAL_BLOCK_SESSION_BACKWARD = 3
comptime NEURAL_BLOCK_FORWARD_FRESH = 4
comptime NEURAL_BLOCK_BACKWARD = 5
# The block session is a native registry handle of the GPU Transformer image
# (lane transformer-session-native, 2026-11-12; core/transformer_block_session.mojo):
# every session operation's first word is the handle word, never an address.
comptime NEURAL_BLOCK_SESSION_FORWARD = 11
"""a: 13 (+ options tail); words [h, 10 scalars (+ tail)] -> cached tokens."""
comptime NEURAL_BLOCK_SESSION_CLOSE = 13
"""words [h]: close the session and release its handle (a busy session refuses
and keeps the handle)."""
comptime NEURAL_BLOCK_SESSION_FORWARD_INT15 = 35
"""a: INT15_ENTRY_ADDRS; words [h, INT15_ENTRY_PARAMS] -> cached tokens."""
comptime NEURAL_BLOCK_SESSION_CREATE = 36
"""No arguments -> a new session's handle word (creation does no device work)."""
# Mamba block operation codes (mamba/native_block_dispatch.mojo).
comptime NEURAL_MAMBA_RESERVE = 0
comptime NEURAL_MAMBA_RELEASE = 1
comptime NEURAL_MAMBA1_BACKWARD = 2
comptime NEURAL_MAMBA2_BACKWARD = 3
comptime NEURAL_MAMBA3_BACKWARD = 4
comptime NEURAL_MAMBA3_FRESH_FORWARD = 5
# The Mamba-3 prefill session is a native registry handle of its training
# image (lane mamba3-prefill-handle, 2026-11-12; core/mamba3_prefill_session.mojo):
# every session operation's first word is the handle word, never an address.
comptime NEURAL_MAMBA3_SESSION_FORWARD = 6
"""a: 15 [x, nine weights, y, four reports]; words [h, B, L, d] (GPU image only)."""
comptime NEURAL_MAMBA3_SESSION_BACKWARD = 7
comptime NEURAL_MAMBA3_OWNED_FORWARD = 8
"""a: 6 [x, y, four reports]; words [h, B, L, d, generation]."""
comptime NEURAL_MAMBA3_OWNED_BACKWARD = 9
comptime NEURAL_MAMBA3_OWNED_INSTALL = 15
comptime NEURAL_MAMBA3_OWNED_STATE = 16
"""a: 25 (the carried-state buffers; weight slots unread); words [h, B, L, d,
buffered, pending, generation, step] -> buffered tokens (the former Python
entry `mamba3_state_owned`)."""
comptime NEURAL_MAMBA3_OWNED_EXPORT = 17
"""a: nine writable buffers; words [h, generation] -> generation."""
comptime NEURAL_MAMBA3_SESSION_CLOSE = 18
"""words [h]: close the session and release its handle (a busy session
refuses and keeps the handle). No reserve."""
comptime NEURAL_MAMBA3_SESSION_CREATE = 44
"""No arguments -> a new session's handle word (no device work)."""
comptime NEURAL_MAMBA3_SESSION_INFO = 45
"""a: [address of six Int words]; words [h]: weight_uploads, weight_recopies,
weight_reuses, backward_reuses, backward_recomputes, stages_held. No reserve
(a busy host session refuses, as its former Python entry did)."""
comptime NEURAL_MAMBA3_SESSION_CAPS = 46
"""No arguments -> capability bits of the image (fixed per build):"""
comptime MAMBA3_CAP_SESSION_PREFILL = 1
"""the non-owned session forward/backward (ops 6/7; the GPU image)."""
comptime MAMBA3_CAP_REPORTS_OPTIONAL = 2
"""op 6 admits a null report address (IDN_MAMBA3_REPORTS_ON_REQUEST)."""
comptime MAMBA3_CAP_OWNED_WEIGHTS = 4
"""the owned weight snapshot (NN40_OWNED_WEIGHTS)."""


@fieldwise_init
struct NeuralEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def neural_entry_vendor(vendor: String) -> Int:
    """The vendor word of a key for the vendor string an image reports
    (`*_vendor`): 0 cpu, 1 metal, 2 cuda, 3 hip, else -1 (a build with no GPU
    vendor publishes nothing)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _neural_entry_key_check(key: NeuralEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native neural entries: invalid tier/vendor key")



def neural_table_publish[family: Int](key: NeuralEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish this image's one descriptor entry of `family` under `key`
    (the provider's module init). A repeated init publishes the same pointer;
    a second image under the same key refuses (core/entry_table.mojo)."""
    comptime assert family >= NEURAL_TABLE_BLOCK_FORWARD and family <= NEURAL_TABLE_LAST, "unknown neural table family"
    _neural_entry_key_check(key)
    comptime if family == NEURAL_TABLE_BLOCK_FORWARD:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
    comptime if family == NEURAL_TABLE_BLOCK_TRAIN:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(TRANSFORMER_BLOCK_ABI_NAME)], [NEURAL_BLOCK_OPERATIONS], [entry])
            return
    comptime if family == NEURAL_TABLE_MAMBA_TRAIN:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(MAMBA_BLOCK_ABI_NAME)], [NEURAL_MAMBA_OPERATIONS], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(MAMBA_BLOCK_ABI_NAME)], [NEURAL_MAMBA_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(MAMBA_BLOCK_ABI_NAME)], [NEURAL_MAMBA_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(MAMBA_BLOCK_ABI_NAME)], [NEURAL_MAMBA_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(MAMBA_BLOCK_ABI_NAME)], [NEURAL_MAMBA_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(MAMBA_BLOCK_ABI_NAME)], [NEURAL_MAMBA_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(MAMBA_BLOCK_ABI_NAME)], [NEURAL_MAMBA_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(MAMBA_BLOCK_ABI_NAME)], [NEURAL_MAMBA_OPERATIONS], [entry])
            return
    comptime if family == NEURAL_TABLE_MLP_PROGRAM:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_MLP_PROGRAM_ABI)], [NEURAL_MLP_PROGRAM_OPERATIONS], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_MLP_PROGRAM_ABI)], [NEURAL_MLP_PROGRAM_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_MLP_PROGRAM_ABI)], [NEURAL_MLP_PROGRAM_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_MLP_PROGRAM_ABI)], [NEURAL_MLP_PROGRAM_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_MLP_PROGRAM_ABI)], [NEURAL_MLP_PROGRAM_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_MLP_PROGRAM_ABI)], [NEURAL_MLP_PROGRAM_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_MLP_PROGRAM_ABI)], [NEURAL_MLP_PROGRAM_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_MLP_PROGRAM_ABI)], [NEURAL_MLP_PROGRAM_OPERATIONS], [entry])
            return
    comptime if family == NEURAL_TABLE_EMBEDDING:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
    comptime if family == NEURAL_TABLE_EMBEDDING_INFER:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [String(NEURAL_EMBEDDING_ABI)], [NEURAL_EMB_OPERATIONS], [entry])
            return
    raise Error("native neural entries: unsupported execution key")


def neural_table_entry[family: Int](key: NeuralEntryKey) raises -> NativeDescriptorCall:
    """The entry the image published for `key`; refuses when that image is
    not loaded or published another schema."""
    comptime assert family >= NEURAL_TABLE_BLOCK_FORWARD and family <= NEURAL_TABLE_LAST, "unknown neural table family"
    _neural_entry_key_check(key)
    comptime if family == NEURAL_TABLE_BLOCK_FORWARD:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockForwardFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
    comptime if family == NEURAL_TABLE_BLOCK_TRAIN:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralBlockTrainFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(TRANSFORMER_BLOCK_ABI_NAME), NEURAL_BLOCK_OPERATIONS)
    comptime if family == NEURAL_TABLE_MAMBA_TRAIN:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(MAMBA_BLOCK_ABI_NAME), NEURAL_MAMBA_OPERATIONS)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(MAMBA_BLOCK_ABI_NAME), NEURAL_MAMBA_OPERATIONS)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(MAMBA_BLOCK_ABI_NAME), NEURAL_MAMBA_OPERATIONS)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(MAMBA_BLOCK_ABI_NAME), NEURAL_MAMBA_OPERATIONS)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(MAMBA_BLOCK_ABI_NAME), NEURAL_MAMBA_OPERATIONS)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(MAMBA_BLOCK_ABI_NAME), NEURAL_MAMBA_OPERATIONS)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(MAMBA_BLOCK_ABI_NAME), NEURAL_MAMBA_OPERATIONS)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMambaTrainFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(MAMBA_BLOCK_ABI_NAME), NEURAL_MAMBA_OPERATIONS)
    comptime if family == NEURAL_TABLE_MLP_PROGRAM:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_MLP_PROGRAM_ABI), NEURAL_MLP_PROGRAM_OPERATIONS)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_MLP_PROGRAM_ABI), NEURAL_MLP_PROGRAM_OPERATIONS)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_MLP_PROGRAM_ABI), NEURAL_MLP_PROGRAM_OPERATIONS)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_MLP_PROGRAM_ABI), NEURAL_MLP_PROGRAM_OPERATIONS)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_MLP_PROGRAM_ABI), NEURAL_MLP_PROGRAM_OPERATIONS)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_MLP_PROGRAM_ABI), NEURAL_MLP_PROGRAM_OPERATIONS)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_MLP_PROGRAM_ABI), NEURAL_MLP_PROGRAM_OPERATIONS)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralMlpProgramFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_MLP_PROGRAM_ABI), NEURAL_MLP_PROGRAM_OPERATIONS)
    comptime if family == NEURAL_TABLE_EMBEDDING:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
    comptime if family == NEURAL_TABLE_EMBEDDING_INFER:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesNeuralEmbeddingInferFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, String(NEURAL_EMBEDDING_ABI), NEURAL_EMB_OPERATIONS)
    raise Error("native neural entries: unsupported execution key")


def _neural_operations[family: Int]() -> Int:
    comptime if family == NEURAL_TABLE_BLOCK_FORWARD or family == NEURAL_TABLE_BLOCK_TRAIN:
        return NEURAL_BLOCK_OPERATIONS
    elif family == NEURAL_TABLE_MAMBA_TRAIN:
        return NEURAL_MAMBA_OPERATIONS
    elif family == NEURAL_TABLE_MLP_PROGRAM:
        return NEURAL_MLP_PROGRAM_OPERATIONS
    else:
        return NEURAL_EMB_OPERATIONS


def neural_ref[family: Int](key: NeuralEntryKey) raises -> EntryRef:
    """The `family` entry of the image `key` names, resolved once (lane
    entry-table-zero-alloc): a connection owner holds it; the table refusals
    raise here, never per call."""
    return EntryRef(neural_table_entry[family](key), entry_ref_word(key.mode, key.vendor, _neural_operations[family]()))


def neural_entry_call(entry: EntryRef, operation: Int, addresses: List[Int],
    ints: List[Int], floats: List[Float64]) raises -> Int:
    """One synchronous descriptor call through a resolved entry: one indirect
    call. The caller owns every buffer the addresses name until this returns
    and decides the GIL (this module imports no Python)."""
    return entry_ref_call(entry, operation, addresses, ints, floats)
