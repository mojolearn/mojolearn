# SPDX-License-Identifier: Apache-2.0
"""The multi-output Ridge table of the linear family: the
`_mojolearn_estimators` entries `MultiOutputRegressor`'s one-program ridge
(`bindings/trees_public_meta.mojo::_multioutput_ridge_fit` and its
predict) reaches, as one native entry (lane calltable-base-helpers,
2026-11-11).

It appends to the linear table family (core/linear_entry_table.mojo,
core/linear_entry_table_ext.mojo) with its own slot names and the same
`LinearEntryKey`, so the existing slots keep their names and layouts. The
publisher is the GPU estimators binding when it carries the multi-output
ridge (`glm/impl/ridge_multi.mojo::MULTIOUT_RIDGE`); its runs are in
`core/linear_multi_device.mojo`. A build without it publishes nothing and a
caller's read refuses with "provider is not published", which the caller
takes as "no one-program ridge" exactly as it took a missing
`ridge_fit_multi` export.

Request layouts (a: addresses, p: Int words, f: Float64 words):
  RM_COL_SUMS     a [x, sums]                       p [rows, cols]              f []
  RM_MEANS_FINISH a [sums, mean64, mean32]          p [cols, rows, weighted]    f [total]
  RM_CENTER       a [x, mu, out]                    p [rows, cols]              f []
  RM_FIT          a [x, y, coef, ymean, xmean, icpt] p [n, d, m, center_y, want_icpt] f [alpha]
                  (xmean and icpt are 0 unless want_icpt)
  RM_PREDICT      a [x, coef, icpt, out]            p [n, d, m]                 f []
Every operation returns 0; a refusal is the former Python export's words.
Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.linear_entry_table import LinearEntryKey
from core.native_descriptor_abi import NativeDescriptorCall

comptime RIDGE_MULTI_ABI = "mojolearn.linear.ridge_multi.v1"
comptime RM_COL_SUMS = 0
comptime RM_MEANS_FINISH = 1
comptime RM_CENTER = 2
comptime RM_FIT = 3
comptime RM_PREDICT = 4
comptime RM_OP_COUNT = 5


def _ridge_multi_key_check(key: LinearEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native ridge multi entries: invalid tier/vendor key")


def ridge_multi_table_publish(key: LinearEntryKey, publisher: Int,
    schemas: List[String], operations: List[Int], entries: List[NativeDescriptorCall]) raises:
    _ridge_multi_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesRidgeMultiIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesRidgeMultiFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesRidgeMultiIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesRidgeMultiFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesRidgeMultiIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesRidgeMultiFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesRidgeMultiIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesRidgeMultiFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native ridge multi entries: unsupported execution key")


def ridge_multi_table_entry(key: LinearEntryKey, index: Int,
    schema: String, operations: Int) raises -> NativeDescriptorCall:
    _ridge_multi_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesRidgeMultiIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesRidgeMultiFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesRidgeMultiIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesRidgeMultiFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesRidgeMultiIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesRidgeMultiFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesRidgeMultiIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesRidgeMultiFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native ridge multi entries: unsupported execution key")


def ridge_multi_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                            ff: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if operation < 0 or operation >= RM_OP_COUNT:
        raise Error("ridge multi: unknown operation " + String(operation))
    if not result:
        raise Error("ridge multi: null result word")
    if na < 0 or np < 0 or nf < 0 or (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff):
        raise Error("ridge multi: null or invalid request span")
    var want_a = 2
    var want_p = 2
    var want_f = 0
    if operation == RM_MEANS_FINISH:
        want_a = 3
        want_p = 3
        want_f = 1
    elif operation == RM_CENTER:
        want_a = 3
    elif operation == RM_FIT:
        want_a = 6
        want_p = 5
        want_f = 1
    elif operation == RM_PREDICT:
        want_a = 4
        want_p = 3
    if na != want_a or np != want_p or nf != want_f:
        raise Error("ridge multi: request spans do not match operation " + String(operation))
