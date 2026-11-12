# SPDX-License-Identifier: Apache-2.0
"""The PCA owned-fit table (lane calltable-cn-ann, 2026-11-11): the
`pca_fit_owned` body of the GPU binding `_mojolearn_estimators`
(decomposition/pca_ops_entry_device.mojo) and of its CPU column
`_mojolearn_estimators_host` (decomposition/pca_ops_entry_host.mojo) as one
`NativeDescriptorCall` per image. The TSNE owner's 'pca' start
(bindings/mani_tsne_native.mojo) calls it with plain values instead of the
Python `pca_fit_owned` export. A sibling lane owns core/decomp_entry_table.mojo
(the decomposition family's table); this file holds this one operation only.

The same core/entry_table.mojo publication and lookup as
core/neighbors_entry_table.mojo, separate slot names per tier and vendor,
and the same key shape: the `(mode, vendor)` an image records as its module
attribute `decomp_pca_native_key` when it publishes (vendor words 0 cpu,
1 metal, 2 cuda, 3 hip).

Request layout (a: addresses, p: Int words, f: Float64 words):
  PCA_OP_FIT_OWNED  a [x, components, mean, explained, ratio, singular, out (0: no projection),
                       noise (one Float64), status (one Int32, 0: no finite test)]
                    p [n_rows, n_features, n_components, dense, keep mode, whiten]
                    f [fraction]                                                    -> kept count
The export's optional finite flag is the status address here (nonzero: X's
finite status word is written there; a flagged X is not fit and the kept
count and noise are 0). The fraction stays a Float64, as the export read it.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_entry_table import NeighborsEntryKey

comptime DECOMP_PCA_ABI = "mojolearn.decomposition.pca_owned.v1"
comptime PCA_OP_FIT_OWNED = 0
comptime PCA_OPERATION_COUNT = 1


def pca_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                        ff: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the layout before any word
    is read."""
    if not result:
        raise Error("pca ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff) or na < 0 or np < 0 or nf < 0:
        raise Error("pca ops: null or invalid request span")
    if operation != PCA_OP_FIT_OWNED:
        raise Error("pca ops: unknown operation " + String(operation))
    if na != 9 or np != 6 or nf != 1:
        raise Error("pca_fit_owned: params must contain n_rows, n_features, n_components, dense, keep mode, whiten, fraction")


def pca_status_word(address: Int) raises -> Int:
    """0, or a positive int32-aligned address."""
    if address < 0 or address % 4 != 0:
        raise Error("pca_fit_owned: the status address must be a positive int32-aligned address")
    return address


def pca_ops_publish(key: NeighborsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(DECOMP_PCA_ABI)]
    var operations: List[Int] = [PCA_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesDecompPcaIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesDecompPcaFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesDecompPcaIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesDecompPcaFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesDecompPcaIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesDecompPcaFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesDecompPcaIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesDecompPcaFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native pca ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def pca_ops_entry(key: NeighborsEntryKey) raises -> NativeDescriptorCall:
    var schema = String(DECOMP_PCA_ABI)
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesDecompPcaIdenticalCPUV1"](key.mode, key.vendor, 0, schema, PCA_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesDecompPcaFastCPUV1"](key.mode, key.vendor, 0, schema, PCA_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesDecompPcaIdenticalMetalV1"](key.mode, key.vendor, 0, schema, PCA_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesDecompPcaFastMetalV1"](key.mode, key.vendor, 0, schema, PCA_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesDecompPcaIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, PCA_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesDecompPcaFastCUDAV1"](key.mode, key.vendor, 0, schema, PCA_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesDecompPcaIdenticalHIPV1"](key.mode, key.vendor, 0, schema, PCA_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesDecompPcaFastHIPV1"](key.mode, key.vendor, 0, schema, PCA_OPERATION_COUNT)
    raise Error("native pca ops: unsupported execution key")


def pca_fit_owned_call(entry: NativeDescriptorCall, x: Int, components: Int, mean: Int, explained: Int,
                       ratio: Int, singular: Int, projection: Int, noise: Int, status: Int, n_rows: Int,
                       n_features: Int, n_components: Int, dense: Bool, keep_mode: Int, whiten: Bool,
                       fraction: Float64) raises -> Int:
    """Synchronous; the caller keeps every addressed buffer alive until it
    returns and releases the GIL around it. Returns the kept count."""
    var addresses: List[Int] = [x, components, mean, explained, ratio, singular, projection, noise, status]
    var ints: List[Int] = [n_rows, n_features, n_components, Int(dense), keep_mode, Int(whiten)]
    var floats: List[Float64] = [fraction]
    return native_descriptor_call(entry, PCA_OP_FIT_OWNED, addresses, ints, floats)
