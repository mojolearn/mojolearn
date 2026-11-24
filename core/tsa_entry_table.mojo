# SPDX-License-Identifier: Apache-2.0
"""The Holt-Winters operations table: every native entry the
ExponentialSmoothing owner (bindings/tsa_hw_native_door.mojo) calls in
`_mojolearn_tsa` (device), `_mojolearn_tsa_host` (CPU, with the fit) or
`_mojolearn_forecast_host` (CPU inference, no fit), as one
`NativeDescriptorCall` dispatching on its operation id.

Two table families, one slot per tier and vendor each:

  TSA_TABLE_TSA       `_mojolearn_tsa` and `_mojolearn_tsa_host` (the binary
                      the `_mojolearn_tsa` name resolves to)
  TSA_TABLE_FORECAST  `_mojolearn_forecast_host` (the explicit host
                      inference class); a separate slot because it and
                      `_mojolearn_tsa_host` are both CPU identical images and
                      may be loaded in one process

Publication and lookup are metadata operations with no independent numerical
work to split.

Request layouts (a: addresses, p: Int words, f: Float64 words). `seasonal` is
the SEASONAL_* code of holtwinters/impl/tsa/holtwinters_params.mojo (the door
maps the name with `seasonal_from_name`, so an unknown name is refused in the
same words before the call).

  TSA_OP_FLAGS             -                                   -> TSA_HAS_* bits
  TSA_OP_VENDOR            -                                   -> 0 CPU, 1 Metal, 2 CUDA, 3 HIP
  TSA_OP_MODE              -                                   -> 1 identical, 0 fast
  TSA_OP_HW_FIT            a [data, comps, stats, flags]
                           p [n, batch, frequency, start_periods, init_method, seasonal]
                           f [eps]                             -> components_len
  TSA_OP_HW_FORECAST       a [comps, out]
                           p [n, batch, frequency, h, seasonal]            -> h * batch
  TSA_OP_HW_PREDICT        a [comps, out]
                           p [n, batch, frequency, start, end, seasonal]   -> written
  TSA_OP_HW_RESIDENT_PREPARE  a [comps]  p [n, batch, frequency]           -> handle
  TSA_OP_HW_RESIDENT_RELEASE  -          p [handle]                        -> 0
  TSA_OP_HW_RESIDENT_COUNT    -                                            -> live handles
  TSA_OP_HW_QUERY_RESIDENT    a [out]    p [handle, start, end, seasonal]  -> written
  TSA_OP_KPSS_TEST         a [y, flags, stat]  p [batch, n_obs, d, D, s]      f [pval_threshold]  -> batch
                           (flags: batch uint8, 1 stationary, 0 not; stat: batch float32)
  TSA_OP_SELECT_D          a [y, out]          p [batch, n_obs, D, s, d_max]  f [pval_threshold]  -> batch

The two stationarity operations (`kpss_test`, `select_d`; every image holds
them) are functions with no estimator: bindings/tsa_public_native_door.mojo
calls them on the process default tier's entry, and AutoARIMA's search on
the entry of its own device.
"""
from core.entry_table import EntryRef, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall

comptime TSA_OPS_ABI = "mojolearn.tsa.hw.ops.v1"

comptime TSA_TABLE_TSA = 0
comptime TSA_TABLE_FORECAST = 1

comptime TSA_OP_FLAGS = 0
comptime TSA_OP_VENDOR = 1
comptime TSA_OP_MODE = 2
comptime TSA_OP_HW_FIT = 3
comptime TSA_OP_HW_FORECAST = 4
comptime TSA_OP_HW_PREDICT = 5
comptime TSA_OP_HW_RESIDENT_PREPARE = 6
comptime TSA_OP_HW_RESIDENT_RELEASE = 7
comptime TSA_OP_HW_RESIDENT_COUNT = 8
comptime TSA_OP_HW_QUERY_RESIDENT = 9
comptime TSA_OP_KPSS_TEST = 10
comptime TSA_OP_SELECT_D = 11
comptime TSA_OPERATION_COUNT = 12

# TSA_OP_FLAGS bits.
comptime TSA_HAS_FIT = 1
comptime TSA_HAS_RESIDENT = 2


def tsa_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                        ff: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if not result:
        raise Error("tsa ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff) or na < 0 or np < 0 or nf < 0:
        raise Error("tsa ops: null or invalid request span")
    var want_a = 0
    var want_p = 0
    var want_f = 0
    if operation == TSA_OP_FLAGS or operation == TSA_OP_VENDOR or operation == TSA_OP_MODE:
        pass
    elif operation == TSA_OP_HW_FIT:
        want_a = 4
        want_p = 6
        want_f = 1
    elif operation == TSA_OP_HW_FORECAST:
        want_a = 2
        want_p = 5
    elif operation == TSA_OP_HW_PREDICT:
        want_a = 2
        want_p = 6
    elif operation == TSA_OP_HW_RESIDENT_PREPARE:
        want_a = 1
        want_p = 3
    elif operation == TSA_OP_HW_RESIDENT_RELEASE:
        want_p = 1
    elif operation == TSA_OP_HW_RESIDENT_COUNT:
        pass
    elif operation == TSA_OP_HW_QUERY_RESIDENT:
        want_a = 1
        want_p = 4
    elif operation == TSA_OP_KPSS_TEST:
        want_a = 3
        want_p = 5
        want_f = 1
    elif operation == TSA_OP_SELECT_D:
        want_a = 2
        want_p = 5
        want_f = 1
    else:
        raise Error("tsa ops: unknown operation " + String(operation))
    if na != want_a or np != want_p or nf != want_f:
        raise Error("tsa ops: request spans differ from operation " + String(operation))


def tsa_ops_publish[family: Int](mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    comptime assert family == TSA_TABLE_TSA or family == TSA_TABLE_FORECAST, "unknown tsa table family"
    var schemas: List[String] = [String(TSA_OPS_ABI)]
    var operations: List[Int] = [TSA_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    comptime if family == TSA_TABLE_FORECAST:
        # The forecast inference image is CPU identical only.
        if mode != 1 or vendor != 0:
            raise Error("native tsa ops: the forecast host table is CPU identical only")
        entry_table_publish["MojolearnEntriesForecastHostHwOpsIdenticalCPUV1"](mode, vendor, publisher, schemas, operations, entries)
    else:
        if mode == 1 and vendor == 0:
            entry_table_publish["MojolearnEntriesTsaHwOpsIdenticalCPUV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 0 and vendor == 0:
            entry_table_publish["MojolearnEntriesTsaHwOpsFastCPUV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 1 and vendor == 1:
            entry_table_publish["MojolearnEntriesTsaHwOpsIdenticalMetalV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 0 and vendor == 1:
            entry_table_publish["MojolearnEntriesTsaHwOpsFastMetalV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 1 and vendor == 2:
            entry_table_publish["MojolearnEntriesTsaHwOpsIdenticalCUDAV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 0 and vendor == 2:
            entry_table_publish["MojolearnEntriesTsaHwOpsFastCUDAV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 1 and vendor == 3:
            entry_table_publish["MojolearnEntriesTsaHwOpsIdenticalHIPV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 0 and vendor == 3:
            entry_table_publish["MojolearnEntriesTsaHwOpsFastHIPV1"](mode, vendor, publisher, schemas, operations, entries)
        else:
            raise Error("native tsa ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def tsa_ops_resolve(family: Int, mode: Int, vendor: Int) raises -> EntryRef:
    """Read the published table of (`family`, `mode`, `vendor`) once and
    admit it; every refusal of the table happens here, never at call time."""
    var schema = String(TSA_OPS_ABI)
    if family == TSA_TABLE_FORECAST:
        if mode != 1 or vendor != 0:
            raise Error("native tsa ops: the forecast host table is CPU identical only")
        return entry_table_resolve["MojolearnEntriesForecastHostHwOpsIdenticalCPUV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    if family != TSA_TABLE_TSA:
        raise Error("native tsa ops: unknown table family")
    if mode == 1 and vendor == 0:
        return entry_table_resolve["MojolearnEntriesTsaHwOpsIdenticalCPUV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    if mode == 0 and vendor == 0:
        return entry_table_resolve["MojolearnEntriesTsaHwOpsFastCPUV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    if mode == 1 and vendor == 1:
        return entry_table_resolve["MojolearnEntriesTsaHwOpsIdenticalMetalV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    if mode == 0 and vendor == 1:
        return entry_table_resolve["MojolearnEntriesTsaHwOpsFastMetalV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    if mode == 1 and vendor == 2:
        return entry_table_resolve["MojolearnEntriesTsaHwOpsIdenticalCUDAV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    if mode == 0 and vendor == 2:
        return entry_table_resolve["MojolearnEntriesTsaHwOpsFastCUDAV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    if mode == 1 and vendor == 3:
        return entry_table_resolve["MojolearnEntriesTsaHwOpsIdenticalHIPV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    if mode == 0 and vendor == 3:
        return entry_table_resolve["MojolearnEntriesTsaHwOpsFastHIPV1"](mode, vendor, 0, schema, TSA_OPERATION_COUNT)
    raise Error("native tsa ops: unsupported execution key")
