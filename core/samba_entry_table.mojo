# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the Samba/Mamba family.

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Each family publishes the ONE
existing `NativeDescriptorCall` of an image's neural primitive provider
(`PRIMITIVE_ABI_NAME`, 29 operations, `core/native_descriptor_abi.mojo`); no
new operation, no respelled arithmetic:

  SAMBA_TABLE_TRAINING     `_mojolearn_training` (vendor 1 metal, 2 cuda,
                           3 hip; `training/native_primitive_dispatch.mojo`)
                           and `_mojolearn_training_host` (vendor 0;
                           `training/host/native_primitive_dispatch.mojo`).
                           The Samba stack's own binding (`context["entries"]`).
  SAMBA_TABLE_NEURAL_HOST  `_mojolearn_neural_host` (vendor 0;
                           `core/neural_host_dispatch.mojo` family 0), the
                           checkpoint CPU inference model's primitives.

A caller derives the key at UNPACK from the mode and vendor words the Samba
bridge already admitted (`bindings/samba_state_native.mojo::samba_entries`)
and calls the pointer with plain spans; no Python object crosses.

Extending a table's layout requires a new V suffix, never a new loader.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_word, entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall, PRIMITIVE_ABI_NAME

comptime SAMBA_TABLE_TRAINING = 0
comptime SAMBA_TABLE_NEURAL_HOST = 1
comptime SAMBA_PRIMITIVE_OPERATIONS = 29

# Neural primitive operation ids of PRIMITIVE_ABI_NAME (the provider
# dispatchers' `op` words; the former Python entry name in parentheses).
comptime SAMBA_PRIM_EMBEDDING_FORWARD = 0  # (embedding_forward)
comptime SAMBA_PRIM_RMS_NORM_FORWARD = 2  # (rms_norm_forward)
comptime SAMBA_PRIM_LINEAR_FORWARD = 4  # (linear_forward)
comptime SAMBA_PRIM_CE_LOSS = 8  # (ce_loss)
comptime SAMBA_PRIM_ACCUMULATE = 9  # (accumulate, the ordered pair fold)
comptime SAMBA_PRIM_OPTIMIZER_STEP = 25  # (optimizer_step, per-call step)


@fieldwise_init
struct SambaEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def samba_vendor_word(vendor: String) -> Int:
    """The table vendor word of a binding vendor string: 0 for the host
    images ("cpu", or "none"/"" for a host-only install), 1 metal, 2 cuda,
    3 hip, else -1."""
    if vendor == "cpu" or vendor == "none" or vendor == "":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _samba_entry_key_check[family: Int](key: SambaEntryKey) raises:
    comptime assert family == SAMBA_TABLE_TRAINING or family == SAMBA_TABLE_NEURAL_HOST, "unknown samba table family"
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native samba entries: invalid tier/vendor key")
    comptime if family == SAMBA_TABLE_NEURAL_HOST:
        if key.vendor != 0:
            raise Error("native samba entries: the CPU inference primitives are host only")


def samba_table_publish[family: Int](key: SambaEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    _samba_entry_key_check[family](key)
    var schemas: List[String] = [String(PRIMITIVE_ABI_NAME)]
    var operations: List[Int] = [SAMBA_PRIMITIVE_OPERATIONS]
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    comptime if family == SAMBA_TABLE_NEURAL_HOST:
        if key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaNeuralHostIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaNeuralHostFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
    else:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaTrainingIdenticalCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaTrainingFastCPUV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaTrainingIdenticalMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaTrainingFastMetalV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaTrainingIdenticalCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaTrainingFastCUDAV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaTrainingIdenticalHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaTrainingFastHIPV1"
            entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
            return
    raise Error("native samba entries: unsupported execution key")


def samba_table_entry[family: Int](key: SambaEntryKey) raises -> NativeDescriptorCall:
    _samba_entry_key_check[family](key)
    var schema = String(PRIMITIVE_ABI_NAME)
    comptime if family == SAMBA_TABLE_NEURAL_HOST:
        if key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaNeuralHostIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
        if key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaNeuralHostFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
    else:
        if key.vendor == 0 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaTrainingIdenticalCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
        if key.vendor == 0 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaTrainingFastCPUV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
        if key.vendor == 1 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaTrainingIdenticalMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
        if key.vendor == 1 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaTrainingFastMetalV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
        if key.vendor == 2 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaTrainingIdenticalCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
        if key.vendor == 2 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaTrainingFastCUDAV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
        if key.vendor == 3 and key.mode == 1:
            comptime NAME = "MojolearnEntriesSambaTrainingIdenticalHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
        if key.vendor == 3 and key.mode == 0:
            comptime NAME = "MojolearnEntriesSambaTrainingFastHIPV1"
            return entry_table_read[NAME](key.mode, key.vendor, 0, schema, SAMBA_PRIMITIVE_OPERATIONS)
    raise Error("native samba entries: unsupported execution key")


def samba_ref[family: Int](key: SambaEntryKey) raises -> EntryRef:
    """The Samba primitive entry of `family`, resolved once (lane
    entry-table-zero-alloc); the table refusals raise here."""
    return EntryRef(samba_table_entry[family](key), entry_ref_word(key.mode, key.vendor, SAMBA_PRIMITIVE_OPERATIONS))
