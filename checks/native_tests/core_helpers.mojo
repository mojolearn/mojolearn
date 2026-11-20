# SPDX-License-Identifier: Apache-2.0
"""Row byte gather fixtures moved from test_native_helpers.py
(`test_gather_rows_bytes_checks_and_copies_exactly`).

The source is the 5 x 7 byte matrix 0..34; the expectation is the row copy
`source[indices]` written out by construction, never another product call.
Fixture 0 gathers rows [4, 0, 4, 2]; fixtures 1 and 2 put -1 and 5 in the
last index and require the out-of-bounds report with the output bytes
unchanged. The host walk (core/host_helpers_impl.mojo) and the device gather
(core/rows_bytes_device.mojo) take the same fixtures. The binding's empty
short cut and its refusal wording stay in the Python test (boundary).
"""
from core.host_parallel import host_parallelize
from core.host_helpers_impl import gather_rows_bytes
from max.gpu.host import DeviceContext
from core.rows_bytes_device import device_gather_rows_bytes
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime CORE_HELPERS_FIXTURES = 3
comptime SOURCE_ROWS = 5
comptime WIDTH = 7
comptime OUT_ROWS = 4


@fieldwise_init
struct GatherFixture(Movable):
    var source: List[UInt8]
    var idx: List[Int64]
    var dst: List[UInt8]
    var before: List[UInt8]


def gather_fixture(fixture_id: Int) raises -> GatherFixture:
    require(fixture_id >= 0 and fixture_id < CORE_HELPERS_FIXTURES, "unknown helpers fixture")
    var source = List[UInt8](length=SOURCE_ROWS * WIDTH, fill=0)
    for i in range(SOURCE_ROWS * WIDTH):
        source[i] = UInt8(i)
    var idx: List[Int64] = [4, 0, 4, 2]
    if fixture_id != 0:
        idx[OUT_ROWS - 1] = Int64(-1) if fixture_id == 1 else Int64(SOURCE_ROWS)
    var dst = List[UInt8](length=OUT_ROWS * WIDTH, fill=255)
    var before = dst.copy()
    return GatherFixture(source^, idx^, dst^, before^)


def check_gather(fixture_id: Int, f: GatherFixture, bad: Bool) raises:
    if fixture_id == 0:
        require(not bad, "gather_rows_bytes: valid indices reported out of bounds")
        for r in range(OUT_ROWS):
            for c in range(WIDTH):
                var want = UInt8(Int(f.idx[r]) * WIDTH + c)
                require(f.dst[r * WIDTH + c] == want,
                        "gather_rows_bytes: row " + String(r) + " byte " + String(c))
    else:
        require(bad, "gather_rows_bytes: index " + String(f.idx[OUT_ROWS - 1]) + " not reported out of bounds")
        for i in range(len(f.dst)):
            require(f.dst[i] == f.before[i], "gather_rows_bytes: output written on an out-of-bounds index")
    for i in range(SOURCE_ROWS * WIDTH):
        require(f.source[i] == UInt8(i), "gather_rows_bytes: source mutated")


# ---- The host column (was core_helpers_cpu_suite.mojo) ----
# Independent host row-gather fixtures run across host cores.
def gather_host(mut f: GatherFixture) -> Bool:
    """The host walk; True when it reports an index out of bounds."""
    return gather_rows_bytes(
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(f.source.unsafe_ptr())),
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(f.dst.unsafe_ptr())),
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(f.idx.unsafe_ptr())),
        SOURCE_ROWS, OUT_ROWS, WIDTH,
    )


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(CORE_HELPERS_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            var f = gather_fixture(unit)
            var bad = gather_host(f)
            check_gather(unit, f, bad)
        except error:
            cases.fail(unit, error)
    report.parallel(task, CORE_HELPERS_FIXTURES)


# ---- The device column (was core_helpers_gpu_suite.mojo) ----
# Row-gather fixtures on the device, in order on one DeviceContext.
#
# The fixtures share one context and its stream; each gather is already one
# thread per byte, so ordering the three submissions costs nothing.
def gather_device(ctx: DeviceContext, mut f: GatherFixture) raises -> Bool:
    """The device gather; True when it reports an index out of bounds."""
    var ok = device_gather_rows_bytes(
        ctx, Int(f.source.unsafe_ptr()), Int(f.dst.unsafe_ptr()), Int(f.idx.unsafe_ptr()),
        SOURCE_ROWS, OUT_ROWS, WIDTH,
    )
    return not ok


def _run_gpu(mut report: Report) raises:
    var ctx = DeviceContext()
    var cases = report.cases(CORE_HELPERS_FIXTURES)
    for unit in range(CORE_HELPERS_FIXTURES):
        try:
            var f = gather_fixture(unit)
            var bad = gather_device(ctx, f)
            check_gather(unit, f, bad)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("core_helpers", cpu=True, gpu=True, serial="", run=run_cases)
