# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the CausalLM family (lane
calltable-causal-lm, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. CausalLM's whole-model runtime
(`bindings/causal_lm_public.mojo`, `bindings/causal_lm_construct.mojo`) calls
four sibling images. Each publishes the descriptor entry it already carries
(the `*_api_v1` capsule transport of `core/native_descriptor_abi.mojo`) here
at module init, once per (tier, vendor) key:

  CAUSAL_TABLE_LINALG      `gemm/linalg_dispatch.mojo::linalg_native_v1`
                           (GPU `_mojolearn_linalg`) and its host twin
                           (`_mojolearn_linalg_host`), schema
                           "mojolearn.linalg.v1", 15 operations.
  CAUSAL_TABLE_PRIMITIVES  `training/native_primitive_dispatch.mojo::
                           native_primitive_v1` (GPU `_mojolearn_training`)
                           and `core/neural_host_dispatch.mojo`'s family 0
                           (`_mojolearn_neural_host`), PRIMITIVE_ABI_NAME.
  CAUSAL_TABLE_BLOCK       `transformer/native_block_dispatch.mojo::
                           transformer_block_v1` (GPU `_mojolearn_transformer`)
                           and the neural host family 2, TRANSFORMER_BLOCK_ABI_NAME.
  CAUSAL_TABLE_MAMBA       `mamba/native_state_dispatch.mojo::mamba_state_v1`
                           (GPU `_mojolearn_mamba`) and the neural host
                           family 1, MAMBA_STATE_ABI_NAME.

A caller keys the read by the model's admitted tier and vendor
(`CausalEntryKey`) and calls the pointer with plain spans; no Python object
crosses. Only the images CausalLM connects publish (the Python package's
`_causal_native_connections` and `causal_lm.py` target lists): a second image
with the same key and another pointer refuses (core/entry_table.mojo).

Extending a table's layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import (
    NativeDescriptorCall, PRIMITIVE_ABI_NAME, TRANSFORMER_BLOCK_ABI_NAME, MAMBA_STATE_ABI_NAME,
)
from core.native_descriptor_call import native_descriptor_call

comptime CAUSAL_TABLE_LINALG = 0
comptime CAUSAL_TABLE_PRIMITIVES = 1
comptime CAUSAL_TABLE_BLOCK = 2
comptime CAUSAL_TABLE_MAMBA = 3

# The schema strings and operation counts each family's publisher and reader
# agree on. LINALG mirrors `gemm/linalg_dispatch.mojo` LINALG_NATIVE_ABI_NAME
# and LINALG_NATIVE_OPERATION_COUNT (repeated here so a caller does not import
# the linalg kernels); the others are the counts the capsule readers already
# admit (`neural_descriptor_reuse(..., 29)`, `(..., 36)`, default 6).
comptime CAUSAL_LINALG_SCHEMA = "mojolearn.linalg.v1"
comptime CAUSAL_LINALG_OPERATIONS = 15
comptime CAUSAL_PRIMITIVE_OPERATIONS = 29
comptime CAUSAL_BLOCK_OPERATIONS = 36
comptime CAUSAL_MAMBA_OPERATIONS = 6

# Linalg operation codes (gemm/linalg_dispatch.mojo linalg_native_operation).
comptime CAUSAL_LIN_QUANTIZE_INT8 = 3
comptime CAUSAL_LIN_DEQUANTIZE_INT8 = 4
comptime CAUSAL_LIN_GEMM_INT15 = 5
comptime CAUSAL_LIN_QUANTIZE_INT15 = 6
comptime CAUSAL_LIN_TO_BF16 = 8
comptime CAUSAL_LIN_FROM_BF16 = 9
# Primitive operation codes (native_primitive_operation / neural host family 0).
comptime CAUSAL_PRIM_EMBED_FORWARD = 0
comptime CAUSAL_PRIM_NORM_FORWARD = 2
comptime CAUSAL_PRIM_LINEAR_FORWARD = 4
# Block operation codes the CPU image serves (neural_host_transformer_operation).
comptime CAUSAL_BLOCK_HOST_FORWARD = 33
comptime CAUSAL_BLOCK_HOST_FORWARD_INT15 = 34
# Mamba state operation codes: 2 * (family) + step.
comptime CAUSAL_MAMBA1_FORWARD = 0
comptime CAUSAL_MAMBA1_DECODE = 1
comptime CAUSAL_MAMBA2_FORWARD = 2
comptime CAUSAL_MAMBA2_DECODE = 3


@fieldwise_init
struct CausalEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def causal_entry_vendor(vendor: String) -> Int:
    """The vendor word of a key for the vendor string an image reports
    (`*_vendor`: "cpu" for a host image, `COMPILED_VENDOR` for a GPU one):
    0 cpu, 1 metal, 2 cuda, 3 hip, else -1 (a build with no GPU vendor
    publishes nothing)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _causal_entry_key_check(key: CausalEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native CausalLM entries: invalid tier/vendor key")


def _causal_schema[family: Int]() -> String:
    comptime if family == CAUSAL_TABLE_LINALG:
        return String(CAUSAL_LINALG_SCHEMA)
    elif family == CAUSAL_TABLE_PRIMITIVES:
        return String(PRIMITIVE_ABI_NAME)
    elif family == CAUSAL_TABLE_BLOCK:
        return String(TRANSFORMER_BLOCK_ABI_NAME)
    else:
        return String(MAMBA_STATE_ABI_NAME)


def _causal_operations[family: Int]() -> Int:
    comptime if family == CAUSAL_TABLE_LINALG:
        return CAUSAL_LINALG_OPERATIONS
    elif family == CAUSAL_TABLE_PRIMITIVES:
        return CAUSAL_PRIMITIVE_OPERATIONS
    elif family == CAUSAL_TABLE_BLOCK:
        return CAUSAL_BLOCK_OPERATIONS
    else:
        return CAUSAL_MAMBA_OPERATIONS


def causal_table_publish[family: Int](key: CausalEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish this image's one descriptor entry of `family` under `key`
    (the provider's module init). A repeated init publishes the same pointer."""
    comptime assert family >= CAUSAL_TABLE_LINALG and family <= CAUSAL_TABLE_MAMBA, "unknown CausalLM table family"
    _causal_entry_key_check(key)
    comptime if family == CAUSAL_TABLE_LINALG:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalLinalgIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalLinalgFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalLinalgIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalLinalgFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalLinalgIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalLinalgFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalLinalgIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalLinalgFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
    comptime if family == CAUSAL_TABLE_PRIMITIVES:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalPrimitivesIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalPrimitivesFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalPrimitivesIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalPrimitivesFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalPrimitivesIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalPrimitivesFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalPrimitivesIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalPrimitivesFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
    comptime if family == CAUSAL_TABLE_BLOCK:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalBlockIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalBlockFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalBlockIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalBlockFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalBlockIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalBlockFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalBlockIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalBlockFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
    comptime if family == CAUSAL_TABLE_MAMBA:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalMambaStateIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalMambaStateFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalMambaStateIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalMambaStateFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalMambaStateIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalMambaStateFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalMambaStateIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalMambaStateFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, [_causal_schema[family]()],
                [_causal_operations[family]()], [entry])
            return
    raise Error("native CausalLM entries: unsupported execution key")


def causal_table_entry[family: Int](key: CausalEntryKey) raises -> NativeDescriptorCall:
    """The published entry of `family` for `key`; a missing or mismatched
    publication refuses with the table's error."""
    comptime assert family >= CAUSAL_TABLE_LINALG and family <= CAUSAL_TABLE_MAMBA, "unknown CausalLM table family"
    _causal_entry_key_check(key)
    comptime if family == CAUSAL_TABLE_LINALG:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalLinalgIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalLinalgFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalLinalgIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalLinalgFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalLinalgIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalLinalgFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalLinalgIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalLinalgFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
    comptime if family == CAUSAL_TABLE_PRIMITIVES:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalPrimitivesIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalPrimitivesFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalPrimitivesIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalPrimitivesFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalPrimitivesIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalPrimitivesFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalPrimitivesIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalPrimitivesFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
    comptime if family == CAUSAL_TABLE_BLOCK:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalBlockIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalBlockFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalBlockIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalBlockFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalBlockIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalBlockFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalBlockIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalBlockFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
    comptime if family == CAUSAL_TABLE_MAMBA:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalMambaStateIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalMambaStateFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalMambaStateIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalMambaStateFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalMambaStateIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalMambaStateFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesCausalMambaStateIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesCausalMambaStateFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, _causal_schema[family](), _causal_operations[family]())
    raise Error("native CausalLM entries: unsupported execution key")


def causal_entry_call[family: Int](key: CausalEntryKey, operation: Int, addresses: List[Int],
    ints: List[Int], floats: List[Float64]) raises -> Int:
    """One synchronous descriptor call into the image `key` names. The caller
    owns every buffer the addresses name until this returns. The entry reports
    a refusal with its former Python entry's text."""
    var entry = causal_table_entry[family](key)
    return native_descriptor_call(entry, operation, addresses, ints, floats)


def causal_tables_admit(key: CausalEntryKey, block_family: Int, packed: Bool) raises:
    """Construction-time admission: every table the model will call is
    published for its key (`block_family` CAUSAL_TABLE_BLOCK for a
    Transformer, CAUSAL_TABLE_MAMBA for Mamba, -1 for neither; `packed` also
    requires linalg). Refuses with the table's error, as the former entry
    lookups refused a missing compiled function."""
    _ = causal_table_entry[CAUSAL_TABLE_PRIMITIVES](key)
    if block_family == CAUSAL_TABLE_BLOCK:
        _ = causal_table_entry[CAUSAL_TABLE_BLOCK](key)
    elif block_family == CAUSAL_TABLE_MAMBA:
        _ = causal_table_entry[CAUSAL_TABLE_MAMBA](key)
    if packed:
        _ = causal_table_entry[CAUSAL_TABLE_LINALG](key)
