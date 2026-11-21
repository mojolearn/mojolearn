# SPDX-License-Identifier: Apache-2.0
"""The ARIMA operations table: every native entry the ARIMA and AutoARIMA
owners (bindings/tsa_arima_native_door.mojo, bindings/tsa_autoarima_native_door.mojo,
providers/python/autoarima_*_pyapi_door.mojo) call in `_mojolearn_arima`
(device), `_mojolearn_arima_host` (the CPU reference binary, with the fit) or
`_mojolearn_forecast_host` (CPU inference, no fit), as one
`NativeDescriptorCall` dispatching on its operation id.

Two table families, one slot per tier and vendor each:

  ARIMA_TABLE_ARIMA     `_mojolearn_arima` and `_mojolearn_arima_host`
  ARIMA_TABLE_FORECAST  `_mojolearn_forecast_host`; a separate slot because
                        it and `_mojolearn_arima_host` are both CPU identical
                        images and may be loaded in one process

Publication and lookup are metadata operations with no independent numerical
work to split.

Request layouts (a: addresses, p: Int words, f: Float64 words). `order9` is
[p, d, q, P, D, Q, s, k, n_exog].

  ARIMA_OP_FLAGS             -                              -> ARIMA_HAS_* bits (fixed per image)
  ARIMA_OP_VENDOR            -                              -> 0 CPU, 1 Metal, 2 CUDA, 3 HIP
  ARIMA_OP_MODE              -                              -> 1 identical, 0 fast
  ARIMA_OP_FIT               a [y, exog, params, x, x0, stats, flags, ic (0: none)]
                             p [batch, n_obs, order9..., method, max_iterations]
                             f [pen_aic, pen_bic]           -> N * batch
  ARIMA_OP_PREDICT           a [y, exog, exog_fut, params, out]
                             p [batch, n_obs, start, end, order9...]   -> (end - start) * batch
  ARIMA_OP_FORECAST          a [y, exog, exog_fut, params, out]
                             p [batch, n_obs, n_steps, order9...]      -> n_steps * batch
  ARIMA_OP_IC                a [loglike, ic]  p [batch]  f [pen_aic, pen_bic]  -> batch
  ARIMA_OP_RESIDENT_PREPARE  a [y, exog, params]  p [batch, n_obs, order9...]  -> handle
  ARIMA_OP_RESIDENT_RELEASE  -   p [handle]                 -> 0
  ARIMA_OP_RESIDENT_COUNT    -                              -> live handles
  ARIMA_OP_PREDICT_RESIDENT  a [exog_fut, out]  p [handle, start, end]   -> written
  ARIMA_OP_FORECAST_RESIDENT a [exog_fut, out]  p [handle, n_steps]      -> written
  ARIMA_OP_NDIFFS            a [y, out]  p [batch, n_obs, d_max]         -> batch
  ARIMA_OP_ORDER_SEARCH      a [y, out]
                             p [batch, n_obs, d, maxiter, then (p, q, k) per order]   -> written, 0: per-order fits
  ARIMA_OP_ORDER_SEARCH_FIT  a [y, out, fit_f32, fit_i32]
                             p [batch, n_obs, d, maxiter, then (p, q, k) per order]   -> written
  ARIMA_OP_ORDER_SEARCH_DEVICE  a [y, out, best]
                             p [batch, n_obs, d, D, s, maxiter, want_ic, then (p, q, P, Q, k) per order]
                             f [one criterion penalty per order, or none]            -> written
  ARIMA_OP_FAST_SEARCH_MODE  -                              -> `fast_search_mode` bits (0 with a trace on)
  ARIMA_OP_ORDER_CAPS        -                              -> `order_search_caps` bits
  ARIMA_OP_SEARCH_SWITCHES   -                              -> ARIMA_SWITCH_* bits (read per search: a trace turns them off)
"""
from core.entry_table import EntryRef, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall

comptime ARIMA_OPS_ABI = "mojolearn.arima.ops.v1"

comptime ARIMA_TABLE_ARIMA = 0
comptime ARIMA_TABLE_FORECAST = 1

comptime ARIMA_OP_FLAGS = 0
comptime ARIMA_OP_VENDOR = 1
comptime ARIMA_OP_MODE = 2
comptime ARIMA_OP_FIT = 3
comptime ARIMA_OP_PREDICT = 4
comptime ARIMA_OP_FORECAST = 5
comptime ARIMA_OP_IC = 6
comptime ARIMA_OP_RESIDENT_PREPARE = 7
comptime ARIMA_OP_RESIDENT_RELEASE = 8
comptime ARIMA_OP_RESIDENT_COUNT = 9
comptime ARIMA_OP_PREDICT_RESIDENT = 10
comptime ARIMA_OP_FORECAST_RESIDENT = 11
comptime ARIMA_OP_NDIFFS = 12
comptime ARIMA_OP_ORDER_SEARCH = 13
comptime ARIMA_OP_ORDER_SEARCH_FIT = 14
comptime ARIMA_OP_ORDER_SEARCH_DEVICE = 15
comptime ARIMA_OP_FAST_SEARCH_MODE = 16
comptime ARIMA_OP_ORDER_CAPS = 17
comptime ARIMA_OP_SEARCH_SWITCHES = 18
comptime ARIMA_OPERATION_COUNT = 19

# ARIMA_OP_FLAGS bits.
comptime ARIMA_HAS_FIT = 1
comptime ARIMA_HAS_RESIDENT = 2
comptime ARIMA_HAS_SEARCH = 4
comptime ARIMA_HAS_IC = 8

# ARIMA_OP_SEARCH_SWITCHES bits.
comptime ARIMA_SWITCH_ORDER_BATCH = 1
comptime ARIMA_SWITCH_SEARCH_REUSE = 2


def arima_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                          ff: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if not result:
        raise Error("arima ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff) or na < 0 or np < 0 or nf < 0:
        raise Error("arima ops: null or invalid request span")
    var want_a = 0
    var want_p = 0
    var want_f = 0
    if (operation == ARIMA_OP_FLAGS or operation == ARIMA_OP_VENDOR or operation == ARIMA_OP_MODE
            or operation == ARIMA_OP_RESIDENT_COUNT or operation == ARIMA_OP_FAST_SEARCH_MODE
            or operation == ARIMA_OP_ORDER_CAPS or operation == ARIMA_OP_SEARCH_SWITCHES):
        pass
    elif operation == ARIMA_OP_FIT:
        want_a = 8
        want_p = 13
        want_f = 2
    elif operation == ARIMA_OP_PREDICT:
        want_a = 5
        want_p = 13
    elif operation == ARIMA_OP_FORECAST:
        want_a = 5
        want_p = 12
    elif operation == ARIMA_OP_IC:
        want_a = 2
        want_p = 1
        want_f = 2
    elif operation == ARIMA_OP_RESIDENT_PREPARE:
        want_a = 3
        want_p = 11
    elif operation == ARIMA_OP_RESIDENT_RELEASE:
        want_p = 1
    elif operation == ARIMA_OP_PREDICT_RESIDENT:
        want_a = 2
        want_p = 3
    elif operation == ARIMA_OP_FORECAST_RESIDENT:
        want_a = 2
        want_p = 2
    elif operation == ARIMA_OP_NDIFFS:
        want_a = 2
        want_p = 3
    elif operation == ARIMA_OP_ORDER_SEARCH or operation == ARIMA_OP_ORDER_SEARCH_FIT:
        # A variable tail: one (p, q, k) triple per candidate order.
        var fixed_a = 2 if operation == ARIMA_OP_ORDER_SEARCH else 4
        if na != fixed_a or np < 4 or (np - 4) % 3 != 0 or nf != 0:
            raise Error("arima ops: expected [batch,nobs,d,maxiter] and (p,q,k) triples")
        return
    elif operation == ARIMA_OP_ORDER_SEARCH_DEVICE:
        # A variable tail: one (p, q, P, Q, k) row per candidate order, and
        # no penalty or one per order.
        if na != 3 or np < 7 or (np - 7) % 5 != 0 or (nf != 0 and nf != (np - 7) // 5):
            raise Error("arima ops: expected [batch,nobs,d,D,s,maxiter,want_ic], (p,q,P,Q,k) rows and one penalty per order")
        return
    else:
        raise Error("arima ops: unknown operation " + String(operation))
    if na != want_a or np != want_p or nf != want_f:
        raise Error("arima ops: request spans differ from operation " + String(operation))


def arima_ops_publish[family: Int](mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    comptime assert family == ARIMA_TABLE_ARIMA or family == ARIMA_TABLE_FORECAST, "unknown arima table family"
    var schemas: List[String] = [String(ARIMA_OPS_ABI)]
    var operations: List[Int] = [ARIMA_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    comptime if family == ARIMA_TABLE_FORECAST:
        # The forecast inference image is CPU identical only.
        if mode != 1 or vendor != 0:
            raise Error("native arima ops: the forecast host table is CPU identical only")
        entry_table_publish["MojolearnEntriesForecastHostArimaOpsIdenticalCPUV1"](mode, vendor, publisher, schemas, operations, entries)
    else:
        if mode == 1 and vendor == 0:
            entry_table_publish["MojolearnEntriesArimaOpsIdenticalCPUV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 0 and vendor == 0:
            entry_table_publish["MojolearnEntriesArimaOpsFastCPUV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 1 and vendor == 1:
            entry_table_publish["MojolearnEntriesArimaOpsIdenticalMetalV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 0 and vendor == 1:
            entry_table_publish["MojolearnEntriesArimaOpsFastMetalV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 1 and vendor == 2:
            entry_table_publish["MojolearnEntriesArimaOpsIdenticalCUDAV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 0 and vendor == 2:
            entry_table_publish["MojolearnEntriesArimaOpsFastCUDAV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 1 and vendor == 3:
            entry_table_publish["MojolearnEntriesArimaOpsIdenticalHIPV1"](mode, vendor, publisher, schemas, operations, entries)
        elif mode == 0 and vendor == 3:
            entry_table_publish["MojolearnEntriesArimaOpsFastHIPV1"](mode, vendor, publisher, schemas, operations, entries)
        else:
            raise Error("native arima ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def arima_ops_resolve(family: Int, mode: Int, vendor: Int) raises -> EntryRef:
    """Read the published table of (`family`, `mode`, `vendor`) once and
    admit it; every refusal of the table happens here, never at call time."""
    var schema = String(ARIMA_OPS_ABI)
    if family == ARIMA_TABLE_FORECAST:
        if mode != 1 or vendor != 0:
            raise Error("native arima ops: the forecast host table is CPU identical only")
        return entry_table_resolve["MojolearnEntriesForecastHostArimaOpsIdenticalCPUV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    if family != ARIMA_TABLE_ARIMA:
        raise Error("native arima ops: unknown table family")
    if mode == 1 and vendor == 0:
        return entry_table_resolve["MojolearnEntriesArimaOpsIdenticalCPUV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    if mode == 0 and vendor == 0:
        return entry_table_resolve["MojolearnEntriesArimaOpsFastCPUV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    if mode == 1 and vendor == 1:
        return entry_table_resolve["MojolearnEntriesArimaOpsIdenticalMetalV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    if mode == 0 and vendor == 1:
        return entry_table_resolve["MojolearnEntriesArimaOpsFastMetalV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    if mode == 1 and vendor == 2:
        return entry_table_resolve["MojolearnEntriesArimaOpsIdenticalCUDAV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    if mode == 0 and vendor == 2:
        return entry_table_resolve["MojolearnEntriesArimaOpsFastCUDAV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    if mode == 1 and vendor == 3:
        return entry_table_resolve["MojolearnEntriesArimaOpsIdenticalHIPV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    if mode == 0 and vendor == 3:
        return entry_table_resolve["MojolearnEntriesArimaOpsFastHIPV1"](mode, vendor, 0, schema, ARIMA_OPERATION_COUNT)
    raise Error("native arima ops: unsupported execution key")
