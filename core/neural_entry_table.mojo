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
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall, TRANSFORMER_BLOCK_ABI_NAME, MAMBA_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_call

comptime NEURAL_TABLE_BLOCK_FORWARD = 0
comptime NEURAL_TABLE_BLOCK_TRAIN = 1
comptime NEURAL_TABLE_MAMBA_TRAIN = 2
comptime NEURAL_TABLE_LAST = 2

# The operation counts every publisher and reader agree on: the readers'
# former capsule admissions (`neural_descriptor_reuse(..., 36)` and `(..., 44)`).
comptime NEURAL_BLOCK_OPERATIONS = 36
comptime NEURAL_MAMBA_OPERATIONS = 44

# Transformer block operation codes (transformer/native_block_dispatch.mojo).
comptime NEURAL_BLOCK_RESERVE = 0
comptime NEURAL_BLOCK_RELEASE = 1
comptime NEURAL_BLOCK_SESSION_FORWARD_FRESH = 2
comptime NEURAL_BLOCK_SESSION_BACKWARD = 3
comptime NEURAL_BLOCK_FORWARD_FRESH = 4
comptime NEURAL_BLOCK_BACKWARD = 5
# Mamba block operation codes (mamba/native_block_dispatch.mojo).
comptime NEURAL_MAMBA_RESERVE = 0
comptime NEURAL_MAMBA_RELEASE = 1
comptime NEURAL_MAMBA1_BACKWARD = 2
comptime NEURAL_MAMBA2_BACKWARD = 3
comptime NEURAL_MAMBA3_BACKWARD = 4
comptime NEURAL_MAMBA3_SESSION_BACKWARD = 7
comptime NEURAL_MAMBA3_OWNED_BACKWARD = 9
comptime NEURAL_MAMBA3_OWNED_INSTALL = 15


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
    raise Error("native neural entries: unsupported execution key")


def neural_entry_call[family: Int](key: NeuralEntryKey, operation: Int, addresses: List[Int],
    ints: List[Int], floats: List[Float64]) raises -> Int:
    """One synchronous descriptor call into the image `key` names. The caller
    owns every buffer the addresses name until this returns and decides the
    GIL (this module imports no Python)."""
    var entry = neural_table_entry[family](key)
    return native_descriptor_call(entry, operation, addresses, ints, floats)
