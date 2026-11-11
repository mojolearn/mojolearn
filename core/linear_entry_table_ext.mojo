# SPDX-License-Identifier: Apache-2.0
"""The x_linear operations table: every native entry the linear expansion
estimators' owner (bindings/x_linear_native.mojo) calls in `_mojolearn_x_linear`
(device) or `_mojolearn_x_linear_host` (CPU), as one `NativeDescriptorCall`
dispatching on its operation id.

This extends the Ridge/OLS table family (core/linear_entry_table.mojo) with a
separate slot per tier and vendor, so the existing V1 slots keep their names
and layouts. Publication and lookup are metadata operations with no
independent numerical work to split.

Request layouts (a: addresses, p: Int words, f: Float64 words):
  XL_OP_FLAGS          -           -> XL_HAS_* bits
  XL_OP_FIT            a [x, y, out]
                       p [algo, n, d, n_x, n_y, n_out, n_fw, n_iw, n_ip, n_fp, ip...]
                       f [fp...]                                    -> n_out
  XL_OP_FIT_FOLD       a [y, out, words (2 Int: finite, range)]
                       p [algo, base_id, base_rows, index_id, n, d, n_x, n_y, n_out, n_fw, n_iw, n_ip, n_fp, ip...]
                       f [fp...]                                    -> n_out
  XL_OP_DECISION       a [x, wb, out]        p [n, d, k, link, status]                     -> n * k
  XL_OP_DECISION_CODES a [x, wb, out]        p [n, d, k, link, strict, below, above, status] -> n
  XL_OP_DECISION_ROWS  a [x, wb, out]        p [n, d, k, link, rows_mode, status]          -> n
  XL_OP_HUBER_OUT      a [x, wb, y, flags]   p [n, d, link]         f [thr]               -> n
  XL_OP_CLASS_PREP     a [codes, sw, cw, out] p [n, k, balanced, weighted]                 -> largest count
  XL_OP_SPEARMAN       a [x, y]              p [n]                                        -> sign
  XL_OP_BEST_ALPHA     a [scores]            p [n]                                        -> index or -1
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.linear_entry_table import LinearEntryKey
from core.native_descriptor_abi import NativeDescriptorCall

comptime XLINEAR_OPS_ABI = "mojolearn.x_linear.ops.v1"
comptime XL_OP_FLAGS = 0
comptime XL_OP_FIT = 1
comptime XL_OP_FIT_FOLD = 2
comptime XL_OP_DECISION = 3
comptime XL_OP_DECISION_CODES = 4
comptime XL_OP_DECISION_ROWS = 5
comptime XL_OP_HUBER_OUT = 6
comptime XL_OP_CLASS_PREP = 7
comptime XL_OP_SPEARMAN = 8
comptime XL_OP_BEST_ALPHA = 9
comptime XL_OPERATION_COUNT = 10
# XL_OP_FLAGS bits.
comptime XL_HAS_FIT_FOLD = 1
# Fixed header words before the integer hyperparameters.
comptime XL_FIT_HEADER = 10
comptime XL_FOLD_HEADER = 13


def xl_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                       ff: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if not result:
        raise Error("x_linear ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff) or na < 0 or np < 0 or nf < 0:
        raise Error("x_linear ops: null or invalid request span")
    var want_a = 0
    var want_p = 0
    var want_f = 0
    if operation == XL_OP_FLAGS:
        pass
    elif operation == XL_OP_FIT:
        if np < XL_FIT_HEADER:
            raise Error("x_linear ops: fit header is short")
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        want_a = 3
        want_p = XL_FIT_HEADER + p[8]
        want_f = p[9]
    elif operation == XL_OP_FIT_FOLD:
        if np < XL_FOLD_HEADER:
            raise Error("x_linear ops: fold fit header is short")
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        want_a = 3
        want_p = XL_FOLD_HEADER + p[11]
        want_f = p[12]
    elif operation == XL_OP_DECISION:
        want_a = 3
        want_p = 5
    elif operation == XL_OP_DECISION_CODES:
        want_a = 3
        want_p = 8
    elif operation == XL_OP_DECISION_ROWS:
        want_a = 3
        want_p = 6
    elif operation == XL_OP_HUBER_OUT:
        want_a = 4
        want_p = 3
        want_f = 1
    elif operation == XL_OP_CLASS_PREP:
        want_a = 4
        want_p = 4
    elif operation == XL_OP_SPEARMAN:
        want_a = 2
        want_p = 1
    elif operation == XL_OP_BEST_ALPHA:
        want_a = 1
        want_p = 1
    else:
        raise Error("x_linear ops: unknown operation " + String(operation))
    if na != want_a or np != want_p or nf != want_f:
        raise Error("x_linear ops: request spans differ from operation " + String(operation))


def xl_ops_int32(pp: Int, offset: Int, count: Int) -> List[Int32]:
    """The integer hyperparameters as the kernels' Int32 words (the same
    `Int32(int)` conversion the Python door applied)."""
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var words = List[Int32](capacity=max(count, 1))
    for i in range(count):  # small-loop(the route's integer hyperparameters, not data)
        words.append(Int32(p[offset + i]))
    return words^


def xl_ops_float32(ff: Int, count: Int) -> List[Float32]:
    """The float hyperparameters as the kernels' Float32 words (the same
    `Float32(float)` rounding the Python door applied)."""
    var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var words = List[Float32](capacity=max(count, 1))
    for i in range(count):  # small-loop(the route's float hyperparameters, not data)
        words.append(Float32(f[i]))
    return words^


def xl_ops_publish(key: LinearEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(XLINEAR_OPS_ABI)]
    var operations: List[Int] = [XL_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesXLinearOpsIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesXLinearOpsFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesXLinearOpsIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesXLinearOpsFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesXLinearOpsIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesXLinearOpsFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesXLinearOpsIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesXLinearOpsFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native x_linear ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def xl_ops_entry(key: LinearEntryKey) raises -> NativeDescriptorCall:
    var schema = String(XLINEAR_OPS_ABI)
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesXLinearOpsIdenticalCPUV1"](key.mode, key.vendor, 0, schema, XL_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesXLinearOpsFastCPUV1"](key.mode, key.vendor, 0, schema, XL_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesXLinearOpsIdenticalMetalV1"](key.mode, key.vendor, 0, schema, XL_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesXLinearOpsFastMetalV1"](key.mode, key.vendor, 0, schema, XL_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesXLinearOpsIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, XL_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesXLinearOpsFastCUDAV1"](key.mode, key.vendor, 0, schema, XL_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesXLinearOpsIdenticalHIPV1"](key.mode, key.vendor, 0, schema, XL_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesXLinearOpsFastHIPV1"](key.mode, key.vendor, 0, schema, XL_OPERATION_COUNT)
    raise Error("native x_linear ops: unsupported execution key")
