# SPDX-License-Identifier: Apache-2.0
"""The Gaussian process family's native entry table (lane gp-entry-table,
2026-11-12).

Each GP image publishes its existing typed descriptor
(`core/gp_native_provider.mojo` `gp_native_v1[K]`, ABI `GP_NATIVE_ABI`, the
`GP_*` operations of `core/gp_native_api.mojo`) once at module init, through
the same `core/entry_table.mojo` publication and lookup as
`core/decomp_entry_table_pca.mojo`, in a slot named by column, tier and vendor:

  column 0  `_mojolearn_gp`             the device image (vendor 1 metal, 2 cuda,
                                          3 hip; 0 for a build with no GPU vendor)
  column 1  `_mojolearn_gp_host`        the CPU reference image (vendor 0)
  column 2  `_mojolearn_gp_infer_host`  the CPU inference image (vendor 0)

The two CPU images can be loaded in one process (the host surface binds the
regressor to one and the classifier's inference to the other), so the column
word keeps their slots apart. Each image records its key as the module
attribute `gp_native_key` = `(mode, vendor, column)`; a caller reads it once
at UNPACK (`bindings/gp_native_boundary.mojo` `gp_entry_key`), resolves the
entry once (`gp_table_resolve`) and calls it natively (`GPDriver`). No
`gp_native_api_v1` capsule factory is called and no Python object crosses
into RUN. The GPC shard worker (`bindings/parallel_shard_native.mojo`) calls
the same entry.

Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import EntryRef, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall
from core.gp_native_api import GP_NATIVE_ABI, GP_NATIVE_COUNT

comptime GP_COLUMN_DEVICE = 0
comptime GP_COLUMN_HOST = 1
comptime GP_COLUMN_HOST_INFER = 2


@fieldwise_init
struct GPEntryKey(ImplicitlyCopyable, Movable):
    var mode: Int
    var vendor: Int
    var column: Int


def gp_entry_vendor(vendor: StaticString) -> Int:
    """The vendor word of the key for an image's compiled vendor string
    (`COMPILED_VENDOR`, or "cpu" for a host image): 0 cpu (and "none", a
    device build with no GPU vendor), 1 metal, 2 cuda, 3 hip, else -1."""
    if vendor == "cpu" or vendor == "none":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _gp_entry_key_check(key: GPEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3 or key.column < 0 or key.column > 2:
        raise Error("native GP entries: invalid tier/vendor/column key")
    if key.column != GP_COLUMN_DEVICE and key.vendor != 0:
        raise Error("native GP entries: a host column has vendor 0 only")


def _gp_publish_slot[name: StaticString](key: GPEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(GP_NATIVE_ABI)]
    var operations: List[Int] = [GP_NATIVE_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    entry_table_publish[name](key.mode, key.vendor, publisher, schemas, operations, entries)
    _ = schemas
    _ = operations
    _ = entries


def _gp_resolve_slot[name: StaticString](key: GPEntryKey) raises -> EntryRef:
    return entry_table_resolve[name](key.mode, key.vendor, 0, String(GP_NATIVE_ABI), GP_NATIVE_COUNT)


def gp_table_publish(key: GPEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish an image's GP descriptor in its (column, tier, vendor) slot.
    A repeated module initialization republishes the same pointer."""
    _gp_entry_key_check(key)
    if key.column == 0 and key.vendor == 0 and key.mode == 1:
        _gp_publish_slot["MojolearnEntriesGPIdenticalCPUV1"](key, publisher, entry)
        return
    if key.column == 0 and key.vendor == 0 and key.mode == 0:
        _gp_publish_slot["MojolearnEntriesGPFastCPUV1"](key, publisher, entry)
        return
    if key.column == 0 and key.vendor == 1 and key.mode == 1:
        _gp_publish_slot["MojolearnEntriesGPIdenticalMetalV1"](key, publisher, entry)
        return
    if key.column == 0 and key.vendor == 1 and key.mode == 0:
        _gp_publish_slot["MojolearnEntriesGPFastMetalV1"](key, publisher, entry)
        return
    if key.column == 0 and key.vendor == 2 and key.mode == 1:
        _gp_publish_slot["MojolearnEntriesGPIdenticalCUDAV1"](key, publisher, entry)
        return
    if key.column == 0 and key.vendor == 2 and key.mode == 0:
        _gp_publish_slot["MojolearnEntriesGPFastCUDAV1"](key, publisher, entry)
        return
    if key.column == 0 and key.vendor == 3 and key.mode == 1:
        _gp_publish_slot["MojolearnEntriesGPIdenticalHIPV1"](key, publisher, entry)
        return
    if key.column == 0 and key.vendor == 3 and key.mode == 0:
        _gp_publish_slot["MojolearnEntriesGPFastHIPV1"](key, publisher, entry)
        return
    if key.column == 1 and key.vendor == 0 and key.mode == 1:
        _gp_publish_slot["MojolearnEntriesGPHostIdenticalCPUV1"](key, publisher, entry)
        return
    if key.column == 1 and key.vendor == 0 and key.mode == 0:
        _gp_publish_slot["MojolearnEntriesGPHostFastCPUV1"](key, publisher, entry)
        return
    if key.column == 2 and key.vendor == 0 and key.mode == 1:
        _gp_publish_slot["MojolearnEntriesGPHostInferIdenticalCPUV1"](key, publisher, entry)
        return
    if key.column == 2 and key.vendor == 0 and key.mode == 0:
        _gp_publish_slot["MojolearnEntriesGPHostInferFastCPUV1"](key, publisher, entry)
        return
    raise Error("native GP entries: unsupported execution key")


def gp_table_resolve(key: GPEntryKey) raises -> EntryRef:
    """The published GP descriptor of the image `key` names, admitted once
    (not published, key differs and ABI differs refuse here)."""
    _gp_entry_key_check(key)
    if key.column == 0 and key.vendor == 0 and key.mode == 1:
        return _gp_resolve_slot["MojolearnEntriesGPIdenticalCPUV1"](key)
    if key.column == 0 and key.vendor == 0 and key.mode == 0:
        return _gp_resolve_slot["MojolearnEntriesGPFastCPUV1"](key)
    if key.column == 0 and key.vendor == 1 and key.mode == 1:
        return _gp_resolve_slot["MojolearnEntriesGPIdenticalMetalV1"](key)
    if key.column == 0 and key.vendor == 1 and key.mode == 0:
        return _gp_resolve_slot["MojolearnEntriesGPFastMetalV1"](key)
    if key.column == 0 and key.vendor == 2 and key.mode == 1:
        return _gp_resolve_slot["MojolearnEntriesGPIdenticalCUDAV1"](key)
    if key.column == 0 and key.vendor == 2 and key.mode == 0:
        return _gp_resolve_slot["MojolearnEntriesGPFastCUDAV1"](key)
    if key.column == 0 and key.vendor == 3 and key.mode == 1:
        return _gp_resolve_slot["MojolearnEntriesGPIdenticalHIPV1"](key)
    if key.column == 0 and key.vendor == 3 and key.mode == 0:
        return _gp_resolve_slot["MojolearnEntriesGPFastHIPV1"](key)
    if key.column == 1 and key.vendor == 0 and key.mode == 1:
        return _gp_resolve_slot["MojolearnEntriesGPHostIdenticalCPUV1"](key)
    if key.column == 1 and key.vendor == 0 and key.mode == 0:
        return _gp_resolve_slot["MojolearnEntriesGPHostFastCPUV1"](key)
    if key.column == 2 and key.vendor == 0 and key.mode == 1:
        return _gp_resolve_slot["MojolearnEntriesGPHostInferIdenticalCPUV1"](key)
    if key.column == 2 and key.vendor == 0 and key.mode == 0:
        return _gp_resolve_slot["MojolearnEntriesGPHostInferFastCPUV1"](key)
    raise Error("native GP entries: unsupported execution key")
