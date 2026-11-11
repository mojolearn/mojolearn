# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native tables used by Ridge and LinearRegression.

Slot spellings are explicit so every reader and publisher shares the same
name; extending a table's layout requires a new V suffix, never a new loader.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall

comptime LINEAR_TABLE_ESTIMATORS = 0
comptime LINEAR_TABLE_X_LINEAR = 1
comptime LINEAR_TABLE_DECOMP = 2
comptime LINEAR_TABLE_METRICS = 3


@fieldwise_init
struct LinearEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def _linear_entry_key_check(key: LinearEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native linear entries: invalid tier/vendor key")


def linear_table_publish[family: Int](key: LinearEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    comptime assert family >= 0 and family <= 3, "unknown linear table family"
    _linear_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = (
            "MojolearnEntriesEstimatorsIdenticalCPUV1" if family == 0 else
            "MojolearnEntriesXLinearIdenticalCPUV1" if family == 1 else
            "MojolearnEntriesXDecompIdenticalCPUV1" if family == 2 else
            "MojolearnEntriesXMetricsIdenticalCPUV1"
        )
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = (
            "MojolearnEntriesEstimatorsFastCPUV1" if family == 0 else
            "MojolearnEntriesXLinearFastCPUV1" if family == 1 else
            "MojolearnEntriesXDecompFastCPUV1" if family == 2 else
            "MojolearnEntriesXMetricsFastCPUV1"
        )
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = (
            "MojolearnEntriesEstimatorsIdenticalMetalV1" if family == 0 else
            "MojolearnEntriesXLinearIdenticalMetalV1" if family == 1 else
            "MojolearnEntriesXDecompIdenticalMetalV1" if family == 2 else
            "MojolearnEntriesXMetricsIdenticalMetalV1"
        )
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = (
            "MojolearnEntriesEstimatorsFastMetalV1" if family == 0 else
            "MojolearnEntriesXLinearFastMetalV1" if family == 1 else
            "MojolearnEntriesXDecompFastMetalV1" if family == 2 else
            "MojolearnEntriesXMetricsFastMetalV1"
        )
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = (
            "MojolearnEntriesEstimatorsIdenticalCUDAV1" if family == 0 else
            "MojolearnEntriesXLinearIdenticalCUDAV1" if family == 1 else
            "MojolearnEntriesXDecompIdenticalCUDAV1" if family == 2 else
            "MojolearnEntriesXMetricsIdenticalCUDAV1"
        )
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = (
            "MojolearnEntriesEstimatorsFastCUDAV1" if family == 0 else
            "MojolearnEntriesXLinearFastCUDAV1" if family == 1 else
            "MojolearnEntriesXDecompFastCUDAV1" if family == 2 else
            "MojolearnEntriesXMetricsFastCUDAV1"
        )
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = (
            "MojolearnEntriesEstimatorsIdenticalHIPV1" if family == 0 else
            "MojolearnEntriesXLinearIdenticalHIPV1" if family == 1 else
            "MojolearnEntriesXDecompIdenticalHIPV1" if family == 2 else
            "MojolearnEntriesXMetricsIdenticalHIPV1"
        )
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = (
            "MojolearnEntriesEstimatorsFastHIPV1" if family == 0 else
            "MojolearnEntriesXLinearFastHIPV1" if family == 1 else
            "MojolearnEntriesXDecompFastHIPV1" if family == 2 else
            "MojolearnEntriesXMetricsFastHIPV1"
        )
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native linear entries: unsupported execution key")


def linear_table_entry[family: Int](key: LinearEntryKey, index: Int,
    schema: String, operations: Int) raises -> NativeDescriptorCall:
    comptime assert family >= 0 and family <= 3, "unknown linear table family"
    _linear_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = (
            "MojolearnEntriesEstimatorsIdenticalCPUV1" if family == 0 else
            "MojolearnEntriesXLinearIdenticalCPUV1" if family == 1 else
            "MojolearnEntriesXDecompIdenticalCPUV1" if family == 2 else
            "MojolearnEntriesXMetricsIdenticalCPUV1"
        )
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = (
            "MojolearnEntriesEstimatorsFastCPUV1" if family == 0 else
            "MojolearnEntriesXLinearFastCPUV1" if family == 1 else
            "MojolearnEntriesXDecompFastCPUV1" if family == 2 else
            "MojolearnEntriesXMetricsFastCPUV1"
        )
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = (
            "MojolearnEntriesEstimatorsIdenticalMetalV1" if family == 0 else
            "MojolearnEntriesXLinearIdenticalMetalV1" if family == 1 else
            "MojolearnEntriesXDecompIdenticalMetalV1" if family == 2 else
            "MojolearnEntriesXMetricsIdenticalMetalV1"
        )
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = (
            "MojolearnEntriesEstimatorsFastMetalV1" if family == 0 else
            "MojolearnEntriesXLinearFastMetalV1" if family == 1 else
            "MojolearnEntriesXDecompFastMetalV1" if family == 2 else
            "MojolearnEntriesXMetricsFastMetalV1"
        )
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = (
            "MojolearnEntriesEstimatorsIdenticalCUDAV1" if family == 0 else
            "MojolearnEntriesXLinearIdenticalCUDAV1" if family == 1 else
            "MojolearnEntriesXDecompIdenticalCUDAV1" if family == 2 else
            "MojolearnEntriesXMetricsIdenticalCUDAV1"
        )
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = (
            "MojolearnEntriesEstimatorsFastCUDAV1" if family == 0 else
            "MojolearnEntriesXLinearFastCUDAV1" if family == 1 else
            "MojolearnEntriesXDecompFastCUDAV1" if family == 2 else
            "MojolearnEntriesXMetricsFastCUDAV1"
        )
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = (
            "MojolearnEntriesEstimatorsIdenticalHIPV1" if family == 0 else
            "MojolearnEntriesXLinearIdenticalHIPV1" if family == 1 else
            "MojolearnEntriesXDecompIdenticalHIPV1" if family == 2 else
            "MojolearnEntriesXMetricsIdenticalHIPV1"
        )
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = (
            "MojolearnEntriesEstimatorsFastHIPV1" if family == 0 else
            "MojolearnEntriesXLinearFastHIPV1" if family == 1 else
            "MojolearnEntriesXDecompFastHIPV1" if family == 2 else
            "MojolearnEntriesXMetricsFastHIPV1"
        )
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native linear entries: unsupported execution key")


