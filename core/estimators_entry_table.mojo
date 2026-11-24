# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables of the `_mojolearn_estimators`
entries that other modules call (lane calltable-solver-estimators,
2026-11-11).

Modeled on core/trees_entry_table.mojo: the same core/entry_table.mojo
publication and lookup, separate slot names. Two families:

  ESTIMATORS_TABLE_QN       the QN entry of `_mojolearn_estimators` (device)
                            and `_mojolearn_estimators_host` (CPU): the
                            existing typed `LOGISTIC_NATIVE_ABI` dispatch
                            (core/logistic_native_api.mojo: fit, decision,
                            binary / multiclass / proba, links, log, mode,
                            host, multiclass) that `qn_fit` and
                            `qn_decision_function` share their bodies with.
  ESTIMATORS_TABLE_QN_FOLD  the resident-fold QN entry (`LOGISTIC_FOLD_ABI`,
                            glm/logistic_fold_native.mojo) the GPU base
                            binding `_mojolearn` exports as `qn_fit_fold`,
                            next to the model-selection store it reads.

A caller reads the slot of the image its bridge connected (the key is the
bridge's `(mode, vendor)`, vendor words 0 cpu, 1 metal, 2 cuda, 3 hip) and
drives the entry with `QnNativeDriver` / `QnFoldDriver`; no Python object
crosses. Extending a table's layout requires a new V suffix. Publication and
lookup are small metadata operations with no independent numerical work to
split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.logistic_native_api import LOGISTIC_NATIVE_ABI, QN_NATIVE_COUNT, LOGISTIC_FOLD_ABI, QN_FOLD_COUNT
from core.native_descriptor_abi import NativeDescriptorCall
from core.pca_native_api import PCA_NATIVE_ABI, PCA_NATIVE_COUNT
from core.tsvd_native_api import TSVD_NATIVE_ABI, TSVD_NATIVE_COUNT
from x_prep.transport_abi import XP_OP_COUNT

comptime ESTIMATORS_TABLE_QN = 0
comptime ESTIMATORS_TABLE_QN_FOLD = 1


@fieldwise_init
struct EstimatorsEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def estimators_vendor_word(vendor: String) -> Int:
    """The table's vendor word for an image's reported vendor string ("cpu"
    for a host image, `COMPILED_VENDOR` for a GPU one) or the selected
    execution vendor: 0 cpu, 1 metal, 2 cuda, 3 hip, else -1."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _estimators_key_check(key: EstimatorsEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native estimators entries: unsupported execution key")


def _estimators_schema[family: Int]() -> Tuple[String, Int]:
    comptime if family == ESTIMATORS_TABLE_QN:
        return (String(LOGISTIC_NATIVE_ABI), QN_NATIVE_COUNT)
    else:
        return (String(LOGISTIC_FOLD_ABI), QN_FOLD_COUNT)


def _publish_named[name: StaticString](key: EstimatorsEntryKey, publisher: Int, schema: String, count: Int,
                                       entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [schema]
    var operations: List[Int] = [count]
    var entries: List[NativeDescriptorCall] = [entry]
    entry_table_publish[name](key.mode, key.vendor, publisher, schemas, operations, entries)
    _ = schemas
    _ = operations
    _ = entries


def estimators_table_publish[family: Int](key: EstimatorsEntryKey, publisher: Int,
    entry: NativeDescriptorCall) raises:
    """Publish an image's one entry of `family` under its tier and vendor."""
    comptime assert family == ESTIMATORS_TABLE_QN or family == ESTIMATORS_TABLE_QN_FOLD, "unknown estimators table family"
    _estimators_key_check(key)
    var schema = _estimators_schema[family]()
    comptime if family == ESTIMATORS_TABLE_QN:
        if key.mode == 1 and key.vendor == 0:
            _publish_named["MojolearnEntriesEstimatorsQnIdenticalCPUV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 0 and key.vendor == 0:
            _publish_named["MojolearnEntriesEstimatorsQnFastCPUV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 1 and key.vendor == 1:
            _publish_named["MojolearnEntriesEstimatorsQnIdenticalMetalV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 0 and key.vendor == 1:
            _publish_named["MojolearnEntriesEstimatorsQnFastMetalV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 1 and key.vendor == 2:
            _publish_named["MojolearnEntriesEstimatorsQnIdenticalCUDAV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 0 and key.vendor == 2:
            _publish_named["MojolearnEntriesEstimatorsQnFastCUDAV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 1 and key.vendor == 3:
            _publish_named["MojolearnEntriesEstimatorsQnIdenticalHIPV1"](key, publisher, schema[0], schema[1], entry)
        else:
            _publish_named["MojolearnEntriesEstimatorsQnFastHIPV1"](key, publisher, schema[0], schema[1], entry)
    else:
        if key.mode == 1 and key.vendor == 0:
            _publish_named["MojolearnEntriesQnFoldIdenticalCPUV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 0 and key.vendor == 0:
            _publish_named["MojolearnEntriesQnFoldFastCPUV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 1 and key.vendor == 1:
            _publish_named["MojolearnEntriesQnFoldIdenticalMetalV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 0 and key.vendor == 1:
            _publish_named["MojolearnEntriesQnFoldFastMetalV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 1 and key.vendor == 2:
            _publish_named["MojolearnEntriesQnFoldIdenticalCUDAV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 0 and key.vendor == 2:
            _publish_named["MojolearnEntriesQnFoldFastCUDAV1"](key, publisher, schema[0], schema[1], entry)
        elif key.mode == 1 and key.vendor == 3:
            _publish_named["MojolearnEntriesQnFoldIdenticalHIPV1"](key, publisher, schema[0], schema[1], entry)
        else:
            _publish_named["MojolearnEntriesQnFoldFastHIPV1"](key, publisher, schema[0], schema[1], entry)


def estimators_table_entry[family: Int](key: EstimatorsEntryKey) raises -> NativeDescriptorCall:
    """The published entry of `family` for the image the key names; a
    missing or mismatched publication refuses with the table's error."""
    comptime assert family == ESTIMATORS_TABLE_QN or family == ESTIMATORS_TABLE_QN_FOLD, "unknown estimators table family"
    _estimators_key_check(key)
    var schema = _estimators_schema[family]()
    comptime if family == ESTIMATORS_TABLE_QN:
        if key.mode == 1 and key.vendor == 0:
            return entry_table_read["MojolearnEntriesEstimatorsQnIdenticalCPUV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 0 and key.vendor == 0:
            return entry_table_read["MojolearnEntriesEstimatorsQnFastCPUV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 1 and key.vendor == 1:
            return entry_table_read["MojolearnEntriesEstimatorsQnIdenticalMetalV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 0 and key.vendor == 1:
            return entry_table_read["MojolearnEntriesEstimatorsQnFastMetalV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 1 and key.vendor == 2:
            return entry_table_read["MojolearnEntriesEstimatorsQnIdenticalCUDAV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 0 and key.vendor == 2:
            return entry_table_read["MojolearnEntriesEstimatorsQnFastCUDAV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 1 and key.vendor == 3:
            return entry_table_read["MojolearnEntriesEstimatorsQnIdenticalHIPV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        return entry_table_read["MojolearnEntriesEstimatorsQnFastHIPV1"](key.mode, key.vendor, 0, schema[0], schema[1])
    else:
        if key.mode == 1 and key.vendor == 0:
            return entry_table_read["MojolearnEntriesQnFoldIdenticalCPUV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 0 and key.vendor == 0:
            return entry_table_read["MojolearnEntriesQnFoldFastCPUV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 1 and key.vendor == 1:
            return entry_table_read["MojolearnEntriesQnFoldIdenticalMetalV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 0 and key.vendor == 1:
            return entry_table_read["MojolearnEntriesQnFoldFastMetalV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 1 and key.vendor == 2:
            return entry_table_read["MojolearnEntriesQnFoldIdenticalCUDAV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 0 and key.vendor == 2:
            return entry_table_read["MojolearnEntriesQnFoldFastCUDAV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        if key.mode == 1 and key.vendor == 3:
            return entry_table_read["MojolearnEntriesQnFoldIdenticalHIPV1"](key.mode, key.vendor, 0, schema[0], schema[1])
        return entry_table_read["MojolearnEntriesQnFoldFastHIPV1"](key.mode, key.vendor, 0, schema[0], schema[1])


# ---- ESTIMATORS_TABLE_DECOMP_FOLD (lane vdf-scope-neural2, 2026-11-24) ----
# The resident-fold decomposition entries the GPU base binding `_mojolearn`
# owns beside the identical model-selection store: entry 0 the PCA fold entry
# (core/pca_native_fold.mojo), 1 the TruncatedSVD fold entry
# (core/tsvd_native_fold.mojo), 2 that store as a scope transport entry
# (core/msel_fold_transport.mojo). The PCA / TruncatedSVD members' UNPACK
# resolves all three once (THE RESIDENT ROUTE). The store is the identical
# one: only an identical GPU build publishes, so only identical keys exist.
comptime ESTIMATORS_TABLE_DECOMP_FOLD = 2
comptime DECOMP_FOLD_PCA = 0
comptime DECOMP_FOLD_TSVD = 1
comptime DECOMP_FOLD_STORE = 2
comptime DECOMP_FOLD_STORE_ABI = "mojolearn.msel.fold_store.v1"


def _decomp_fold_schemas() -> List[String]:
    return [String(PCA_NATIVE_ABI), String(TSVD_NATIVE_ABI), String(DECOMP_FOLD_STORE_ABI)]


def _decomp_fold_counts() -> List[Int]:
    return [PCA_NATIVE_COUNT, TSVD_NATIVE_COUNT, XP_OP_COUNT]


def decomp_fold_table_publish(key: EstimatorsEntryKey, publisher: Int, pca: NativeDescriptorCall,
                              tsvd: NativeDescriptorCall, store: NativeDescriptorCall) raises:
    """Publish the identical GPU base image's three resident-fold entries."""
    _estimators_key_check(key)
    if key.mode != 1 or key.vendor == 0:
        raise Error("native estimators entries: the decomposition fold table is identical GPU only")
    var schemas = _decomp_fold_schemas()
    var operations = _decomp_fold_counts()
    var entries: List[NativeDescriptorCall] = [pca, tsvd, store]
    if key.vendor == 1:
        entry_table_publish["MojolearnEntriesDecompFoldIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas,
                                                                         operations, entries)
    elif key.vendor == 2:
        entry_table_publish["MojolearnEntriesDecompFoldIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas,
                                                                        operations, entries)
    else:
        entry_table_publish["MojolearnEntriesDecompFoldIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas,
                                                                       operations, entries)
    _ = schemas
    _ = operations
    _ = entries


def decomp_fold_entry(key: EstimatorsEntryKey, index: Int) raises -> NativeDescriptorCall:
    """Entry `index` (DECOMP_FOLD_*) of the decomposition fold table of the
    image the key names; a missing publication (a host or fast key, or a
    build without the base binding) refuses with the table's error."""
    _estimators_key_check(key)
    if key.mode != 1 or key.vendor == 0 or index < 0 or index > DECOMP_FOLD_STORE:
        raise Error("native estimators entries: no decomposition fold entry for this key")
    var schemas = _decomp_fold_schemas()
    var counts = _decomp_fold_counts()
    var schema = schemas[index].copy()
    var count = counts[index]
    if key.vendor == 1:
        return entry_table_read["MojolearnEntriesDecompFoldIdenticalMetalV1"](key.mode, key.vendor, index, schema, count)
    if key.vendor == 2:
        return entry_table_read["MojolearnEntriesDecompFoldIdenticalCUDAV1"](key.mode, key.vendor, index, schema, count)
    return entry_table_read["MojolearnEntriesDecompFoldIdenticalHIPV1"](key.mode, key.vendor, index, schema, count)
