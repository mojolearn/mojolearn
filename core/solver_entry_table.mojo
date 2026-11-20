# SPDX-License-Identifier: Apache-2.0
"""The solver operations table (lane calltable-solver-estimators,
2026-11-11): the coordinate-descent entries of `_mojolearn_solver` (device)
and `_mojolearn_solver_host` (CPU) as one `NativeDescriptorCall` per image,
dispatching on its operation id. The linear owner
(bindings/x_linear_native_door.mojo, ElasticNet / Lasso) calls it with plain
values instead of the binding's Python `cd_fit` / `cd_predict` exports.

Modeled on core/trees_entry_table.mojo: the same core/entry_table.mojo
publication and lookup, separate slot names per tier and vendor. The key is
the bridge's `(mode, vendor)` of the connected solver image
(python/mojolearn/_native_x_linear_bridge.py `_connect_cd`), vendor words
0 cpu, 1 metal, 2 cuda, 3 hip. Publication and lookup are metadata
operations with no independent numerical work to split.

Request layouts (a: addresses, p: Int words, f: Float64 words):
  SV_OP_CD_FIT      a [x, y, coef, info, sample_weight (0: none)]
                    p [n_rows, n_cols, fit_intercept, epochs, shuffle, has_sample_weight, row_major]
                    f [alpha, l1_ratio, tol]                     -> n_iter
  SV_OP_CD_PREDICT  a [x, coef, out]  p [n_rows, n_cols]  f [intercept]  -> 0
The float words are rounded to Float32 by the entry, the same
`Float32(Float64(py=...))` rounding the Python exports apply, so the bits
are unchanged.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call

comptime SOLVER_OPS_ABI = "mojolearn.solver.ops.v1"
comptime SV_OP_CD_FIT = 0
comptime SV_OP_CD_PREDICT = 1
comptime SV_OPERATION_COUNT = 2


@fieldwise_init
struct SolverEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def solver_vendor_word(vendor: StaticString) -> Int:
    """The table's vendor word for the string a solver image reports as
    `solver_vendor` ("cpu" for the host image, `COMPILED_VENDOR` for the
    GPU one): 0 cpu, 1 metal, 2 cuda, 3 hip, else -1 (publishes nothing)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def sv_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                       ff: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if not result:
        raise Error("solver ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff) or na < 0 or np < 0 or nf < 0:
        raise Error("solver ops: null or invalid request span")
    var want_a: Int
    var want_p: Int
    var want_f: Int
    if operation == SV_OP_CD_FIT:
        want_a = 5
        want_p = 7
        want_f = 3
    elif operation == SV_OP_CD_PREDICT:
        want_a = 3
        want_p = 2
        want_f = 1
    else:
        raise Error("solver ops: unknown operation " + String(operation))
    if na != want_a or np != want_p or nf != want_f:
        raise Error("solver ops: request spans differ from operation " + String(operation))


def solver_ops_publish(key: SolverEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(SOLVER_OPS_ABI)]
    var operations: List[Int] = [SV_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesSolverOpsIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesSolverOpsFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesSolverOpsIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesSolverOpsFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesSolverOpsIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesSolverOpsFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesSolverOpsIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesSolverOpsFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native solver ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def solver_ops_entry(key: SolverEntryKey) raises -> NativeDescriptorCall:
    var schema = String(SOLVER_OPS_ABI)
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesSolverOpsIdenticalCPUV1"](key.mode, key.vendor, 0, schema, SV_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesSolverOpsFastCPUV1"](key.mode, key.vendor, 0, schema, SV_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesSolverOpsIdenticalMetalV1"](key.mode, key.vendor, 0, schema, SV_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesSolverOpsFastMetalV1"](key.mode, key.vendor, 0, schema, SV_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesSolverOpsIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, SV_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesSolverOpsFastCUDAV1"](key.mode, key.vendor, 0, schema, SV_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesSolverOpsIdenticalHIPV1"](key.mode, key.vendor, 0, schema, SV_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesSolverOpsFastHIPV1"](key.mode, key.vendor, 0, schema, SV_OPERATION_COUNT)
    raise Error("native solver ops: unsupported execution key")


def solver_cd_fit_call(entry: NativeDescriptorCall, x: Int, y: Int, coef: Int, info: Int, sample_weight: Int,
                       n_rows: Int, n_cols: Int, fit_intercept: Bool, epochs: Int, shuffle: Bool,
                       row_major: Bool, alpha: Float64, l1_ratio: Float64, tol: Float64) raises -> Int:
    """`cd_fit` through an operations entry: n_iter. The caller keeps every
    addressed buffer alive until the call returns."""
    var addresses: List[Int] = [x, y, coef, info, sample_weight]
    var ints: List[Int] = [n_rows, n_cols, Int(fit_intercept), epochs, Int(shuffle),
                           Int(sample_weight != 0), Int(row_major)]
    var floats: List[Float64] = [alpha, l1_ratio, tol]
    return native_descriptor_call(entry, SV_OP_CD_FIT, addresses, ints, floats)


def solver_cd_predict_call(entry: NativeDescriptorCall, x: Int, coef: Int, dst: Int, n_rows: Int, n_cols: Int,
                           intercept: Float64) raises:
    """`cd_predict` through an operations entry into `dst` (n_rows float32)."""
    var addresses: List[Int] = [x, coef, dst]
    var ints: List[Int] = [n_rows, n_cols]
    var floats: List[Float64] = [intercept]
    _ = native_descriptor_call(entry, SV_OP_CD_PREDICT, addresses, ints, floats)
