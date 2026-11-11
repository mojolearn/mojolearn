# SPDX-License-Identifier: Apache-2.0
"""The x_prep program transport table (core/prep_entry_table.mojo), natively.

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the publication `_mojolearn_x_prep_host`
makes at module init (`publish_x_prep_transport_host`), then a schema
mismatch, an arity mismatch and an unknown operation refuse. Then one case
per transport operation of x_prep/transport_abi.mojo through
`xp_transport_call` and the published host pointer: XP_RUN runs (an empty
program returns its stage count; null buffers refuse with the entry's text),
and every device-only operation refuses on the host image. The device image's
operations are exercised by the public prep tests on a GPU column. No Python
participates.

The cases are independent units: on the host column they run across the
native host worker pool; on a device column they run one after another.
Run in a fresh process (the first check needs an unpublished table).
"""
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.prep_entry_table import PrepEntryKey, prep_table_entry, xp_transport_call, PREP_TABLE_X_PREP
from x_prep.transport_abi import (
    XP_TRANSPORT_ABI, XP_OP_COUNT, XP_RUN, XP_RUN_SCRATCH, XP_RUN_OUT, XP_RUN_RANGES, XP_DEV_PUT, XP_DEV_FREE,
    XP_DEV_TAKE_ROWS,
)
from x_prep.transport_host import publish_x_prep_transport_host

comptime PREP_TABLE_CASES = 8


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> PrepEntryKey:
    return PrepEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def precondition_unpublished() raises:
    try:
        _ = xp_transport_call(_key(), XP_DEV_FREE, List[Int](), [0])
    except problem:
        _refuses(problem, "provider is not published")
        return
    raise Error("an unpublished x_prep table did not refuse")


def preconditions_published() raises:
    try:
        _ = prep_table_entry[PREP_TABLE_X_PREP](_key(), 0, String("mojolearn.x_prep.other.v1"), XP_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        _ = xp_transport_call(_key(), XP_RUN, List[Int](), [0, 0])
        raise Error("a wrong arity did not refuse")
    except problem:
        _refuses(problem, "takes 2 addresses")
    try:
        _ = xp_transport_call(_key(), XP_OP_COUNT, List[Int](), List[Int]())
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")


def _device_refuses(op: Int, na: Int, ni: Int) raises:
    var a = List[Int](length=na, fill=0)
    var i = List[Int](length=ni, fill=0)
    try:
        _ = xp_transport_call(_key(), op, a, i)
    except problem:
        _refuses(problem, "host binding has no device operation")
        return
    raise Error("the host image ran device operation " + String(op))


def case(unit: Int) raises:
    if unit == 0:
        # An empty program on a one-word arena: nothing runs, the stage count returns.
        var arena: List[Float32] = [7.0]
        var program = List[Int32](length=16, fill=0)
        var addresses: List[Int] = [Int(arena.unsafe_ptr()), Int(program.unsafe_ptr())]
        require(xp_transport_call(_key(), XP_RUN, addresses, [1, 0]) == 0, "empty program")
        require(arena[0] == 7.0, "empty program left the arena")
        _ = program
    elif unit == 1:
        try:
            _ = xp_transport_call(_key(), XP_RUN, [0, 0], [1, 0])
            raise Error("null program buffers did not refuse")
        except problem:
            _refuses(problem, "x_prep: invalid program buffers")
    elif unit == 2:
        _device_refuses(XP_RUN_SCRATCH, 2, 3)
    elif unit == 3:
        _device_refuses(XP_RUN_OUT, 3, 4)
    elif unit == 4:
        _device_refuses(XP_RUN_RANGES, 5, 6)
    elif unit == 5:
        _device_refuses(XP_DEV_PUT, 1, 1)
    elif unit == 6:
        _device_refuses(XP_DEV_FREE, 0, 1)
    elif unit == 7:
        _device_refuses(XP_DEV_TAKE_ROWS, 1, 3)


def run_checks() raises:
    precondition_unpublished()
    publish_x_prep_transport_host()
    preconditions_published()
    var failures = List[Int](length=PREP_TABLE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL prep entry table case", unit, String(problem))
    comptime if TARGET_COLUMN == COLUMN_CPU:
        host_parallelize(task, PREP_TABLE_CASES)
    else:
        for unit in range(PREP_TABLE_CASES):  # driver-loop: cases share one device context
            task(unit)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "prep entry table cases failed: " + String(failed))
    print("PASS prep entry table", PREP_TABLE_CASES)
