# SPDX-License-Identifier: Apache-2.0
"""The one runner of the native checks (checks/native_tests/).

A check is ONE file `<name>.mojo` holding its cases, a `run_cases(column,
report)` entry and a `register() -> Check`. `registry.mojo` imports every
check and lists it; the two mains (`run_cpu.mojo`, built with
`-D MOJOLEARN_COLUMN_CPU=1`, and `run_gpu.mojo`) import only the registry, so
one compile per column covers every check (it was one compile per check and
column: five files and two pixi tasks per check).

    pixi run check-native-cpu [all | <name> ...]     # the host column
    pixi run check-native-gpu [all | <name> ...]     # the device column
    <binary> --list                                   # the names this column runs

THE DRIVER. `run_cases` takes the column and a `Report`. Independent cases ask
the report for failure slots (`report.cases(n)`), run their closure with
`report.parallel(task, n)` (`host_parallelize` across the host cores; serial
when the check registered a `serial` reason, because a global registry or the
one device forces it) and record a failure with `cases.fail(unit, error)`. An
error escaping `run_cases` aborts the check and counts as one failure. A check
that never asks for slots counts as one case.

THE REPORT. One line per check, then the process exits 1 when any check failed:

    CHECK <name> column=<cpu|gpu> cases=<n> failed=<k> skipped=<s> <ms>ms

FIXTURES. Stored inputs live under `checks/native_tests/fixtures/<family>/` as
raw little-endian words (no npz); `fixture_path` names one and the
`read_*_file` readers load it (the per-check readers they replace were copies).

No `main()` here: AGENTS.md "a module containing main() cannot be imported".
"""
from std.memory import bitcast
from std.sys import argv, exit
from std.sys.compile import is_defined
from std.time import perf_counter_ns

from core.host_parallel import host_parallelize

# The column a main was built for: run_cpu.mojo is built with -D MOJOLEARN_COLUMN_CPU=1.
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()

comptime FIXTURE_ROOT = "checks/native_tests/fixtures"

comptime _CASE_FAILED = 1
comptime _CASE_SKIPPED = 2

comptime CheckRun = def(Column, mut Report) thin -> None


@fieldwise_init
struct Column(TrivialRegisterPassable):
    """The column a check runs on: the host (`cpu`) or the device (`gpu`)."""
    var cpu: Bool

    def name(self) -> String:
        if self.cpu:
            return "cpu"
        return "gpu"


struct Check(Copyable, Movable):
    """One registered check: its name, the columns it runs on, its entry and,
    when its cases cannot run concurrently, the reason (empty: parallel)."""
    var name: String
    var cpu: Bool
    var gpu: Bool
    var serial: String
    var run: CheckRun

    def __init__(out self, name: String, *, cpu: Bool, gpu: Bool, serial: String, run: CheckRun):
        self.name = name
        self.cpu = cpu
        self.gpu = gpu
        self.serial = serial
        self.run = run

    def runs_on(self, column: Column) -> Bool:
        if column.cpu:
            return self.cpu
        return self.gpu


@fieldwise_init
struct Cases(TrivialRegisterPassable):
    """Failure slots of one batch of independent cases: one word per case,
    written only by the case's own task, so tasks never share a slot."""
    var address: Int
    var count: Int

    def _store(self, unit: Int, value: Int):
        var p = UnsafePointer[Int, MutUntrackedOrigin](unsafe_from_address=self.address)
        p.unsafe_store(unit, value)

    def fail(self, unit: Int, error: Error):
        print("FAIL case", unit, String(error))
        self._store(unit, _CASE_FAILED)

    def fail_message(self, unit: Int, message: String):
        print("FAIL case", unit, message)
        self._store(unit, _CASE_FAILED)

    def skip(self, unit: Int, reason: String):
        print("SKIP case", unit, reason)
        self._store(unit, _CASE_SKIPPED)


struct Report(Movable):
    """What one check run recorded: its case slots and whether it aborted."""
    var check: String
    var column: Column
    var serial: Bool
    var blocks: List[List[Int]]
    var aborted: Bool

    def __init__(out self, check: String, column: Column, serial: Bool):
        self.check = check
        self.column = column
        self.serial = serial
        self.blocks = List[List[Int]]()
        self.aborted = False

    def cases(mut self, n: Int) -> Cases:
        """`n` fresh failure slots. A block's storage never moves while the
        report lives (a moved List keeps its buffer)."""
        self.blocks.append(List[Int](length=n, fill=0))
        var last = len(self.blocks) - 1
        return Cases(Int(self.blocks[last].unsafe_ptr()), n)

    def parallel[F: def(Int) -> None](self, ref func: F, n: Int):
        """Run `func(0) ... func(n - 1)`: across the host cores, or in order
        when the check registered a serial reason."""
        if self.serial:
            for unit in range(n):  # driver-loop: over the cases of a serial check
                func(unit)
        else:
            host_parallelize(func, n)

    def abort(mut self, error: Error):
        print("FAIL", self.check, "aborted:", String(error))
        self.aborted = True

    def counts(self) -> Tuple[Int, Int, Int]:
        var total = 0
        var failed = 0
        var skipped = 0
        for b in range(len(self.blocks)):
            for i in range(len(self.blocks[b])):
                total += 1
                if self.blocks[b][i] == _CASE_FAILED:
                    failed += 1
                elif self.blocks[b][i] == _CASE_SKIPPED:
                    skipped += 1
        if total == 0:
            total = 1
        if self.aborted:
            failed += 1
        return (total, failed, skipped)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def count_failures(failures: List[Int]) -> Int:
    var failed = 0
    for value in failures:
        failed += value
    return failed


def fixture_path(family: String, name: String) -> String:
    return String(FIXTURE_ROOT) + "/" + family + "/" + name


def read_fixture_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    return bytes^


def read_f32_file(path: String) raises -> List[Float32]:
    """A stored little-endian float32 vector."""
    var bytes = read_fixture_bytes(path)
    require(len(bytes) % 4 == 0, path + ": not a float32 vector")
    var count = len(bytes) // 4
    var values = List[Float32](length=count, fill=0.0)
    var vp = values.unsafe_ptr()
    var bp = bytes.unsafe_ptr()
    def word(i: Int) {imm vp, imm bp}:
        var w = (UInt32(bp[4 * i]) | (UInt32(bp[4 * i + 1]) << 8) | (UInt32(bp[4 * i + 2]) << 16)
                 | (UInt32(bp[4 * i + 3]) << 24))
        vp.unsafe_store(i, bitcast[DType.float32](w))
    host_parallelize(word, count)
    _ = bytes^
    return values^


def read_u32_file(path: String) raises -> List[UInt32]:
    """A stored little-endian uint32 vector (raw words)."""
    var bytes = read_fixture_bytes(path)
    require(len(bytes) % 4 == 0, path + ": not a 32-bit word vector")
    var count = len(bytes) // 4
    var values = List[UInt32](length=count, fill=0)
    var vp = values.unsafe_ptr()
    var bp = bytes.unsafe_ptr()
    def word(i: Int) {imm vp, imm bp}:
        vp.unsafe_store(i, UInt32(bp[4 * i]) | (UInt32(bp[4 * i + 1]) << 8) | (UInt32(bp[4 * i + 2]) << 16)
                        | (UInt32(bp[4 * i + 3]) << 24))
    host_parallelize(word, count)
    _ = bytes^
    return values^


def read_f64_file(path: String) raises -> List[Float64]:
    """A stored little-endian float64 vector."""
    var bytes = read_fixture_bytes(path)
    require(len(bytes) % 8 == 0, path + ": not a float64 vector")
    var count = len(bytes) // 8
    var values = List[Float64](length=count, fill=0.0)
    var vp = values.unsafe_ptr()
    var bp = bytes.unsafe_ptr()
    def word(i: Int) {imm vp, imm bp}:
        var w = UInt64(0)
        for k in range(8):  # small-loop(byte assembly): eight bytes of one word
            w |= UInt64(bp[8 * i + k]) << UInt64(8 * k)
        vp.unsafe_store(i, bitcast[DType.float64](w))
    host_parallelize(word, count)
    _ = bytes^
    return values^


def _wanted(checks: List[Check], column: Column, names: List[String]) raises -> List[Int]:
    var picked = List[Int]()
    var every = len(names) == 0
    for n in range(len(names)):
        if names[n] == "all":
            every = True
    if every:
        for i in range(len(checks)):
            if checks[i].runs_on(column):
                picked.append(i)
        return picked^
    for n in range(len(names)):
        var found = -1
        for i in range(len(checks)):
            if checks[i].name == names[n]:
                found = i
        if found < 0:
            raise Error("no registered check named " + names[n])
        if not checks[found].runs_on(column):
            raise Error("check " + names[n] + " does not run on the " + column.name() + " column")
        picked.append(found)
    return picked^


def run_main(checks: List[Check], column: Column) raises:
    """The body of run_cpu.mojo / run_gpu.mojo: `--list`, `all` or names."""
    var args = argv()
    var names = List[String]()
    for i in range(1, len(args)):
        var arg = String(args[i])
        if arg == "--list":
            for c in range(len(checks)):
                if checks[c].runs_on(column):
                    print(checks[c].name)
            return
        names.append(arg)
    var picked = _wanted(checks, column, names)
    var failed_checks = 0
    for p in range(len(picked)):  # driver-loop: over checks; each check runs its own cases in parallel
        var check = checks[picked[p]].copy()
        print("RUN", check.name, "column=" + column.name())
        var report = Report(check.name, column, check.serial.byte_length() > 0)
        var started = perf_counter_ns()
        var run = check.run
        run(column, report)
        var ms = Int((perf_counter_ns() - started) // 1_000_000)
        var counts = report.counts()
        var failed = counts[1]
        print("CHECK", check.name, "column=" + column.name(), "cases=" + String(counts[0]),
              "failed=" + String(failed), "skipped=" + String(counts[2]), String(ms) + "ms")
        if failed > 0:
            failed_checks += 1
    print("CHECKS", len(picked), "column=" + column.name(), "failed=" + String(failed_checks))
    if failed_checks > 0:
        exit(1)
