# SPDX-License-Identifier: Apache-2.0
"""Fresh optimizer state reads zeros (lane unfilled-storage-audit, 2026-11-15).

Optimizer moments are STATE: a step reads them before it writes them, so a
fresh owner must hold zeros (docs/plans/UNFILLED_STORAGE_AUDIT_2026-11-15.md).
The Python-side array.array state (`new_zeros` in
bindings/optimizer_public_door.mojo) lives in a boundary file a native check
may not import; this check covers the native state owner the step uses, the
resident slot (training/optimizer_public.mojo `resident_open`), on both
columns:

  1. a fresh slot downloads zeros;
  2. a slot reopened after a close that left it dirty downloads zeros (the
     pool reuses the freed handle; its old moments must not leak through);
  3. a larger slot reopened over a smaller dirty one downloads zeros in
     every cell.

Serial: the optimizer pool is one process-wide registry, and case 2 relies on
the freed handle being the one reopened.
"""
from training.optimizer_native_api import OptimizerNativeDriver
from training.host.optimizer_native import optimizer_host_v1
from training.optimizer_native_gpu import optimizer_gpu_v1
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime _GUARD = Float32(913.25)
comptime _DIRT = Float32(-7.5)
comptime _CASES = 3


struct _Cells(Movable):
    """`n` cells between two guard words, prefilled with a nonzero value so a
    download that skips a cell is caught."""
    var data: List[Float32]
    var n: Int

    def __init__(out self, n: Int, value: Float32):
        self.n = n
        self.data = List[Float32](length=n + 2, fill=value)
        self.data[0] = _GUARD
        self.data[n + 1] = _GUARD

    def address(self) -> Int:
        return Int(self.data.unsafe_ptr()) + 4

    def expect_zeros(self, what: String) raises:
        require(self.data[0] == _GUARD and self.data[self.n + 1] == _GUARD,
            what + ": download wrote past the buffer")
        for i in range(self.n):  # small-loop(fixture cells)
            require(self.data[i + 1] == Float32(0), what + ": cell " + String(i) + " is not zero")


def _expect_fresh(driver: OptimizerNativeDriver, handle: Int, n: Int, what: String) raises:
    var m = _Cells(n, Float32(17))
    var v = _Cells(n, Float32(17))
    require(driver.download(handle, m.address(), v.address(), n) == n, what + ": download count differs")
    m.expect_zeros(what + " exp_avg")
    v.expect_zeros(what + " exp_avg_sq")


def _dirty_close(driver: OptimizerNativeDriver, n: Int) raises -> Int:
    var handle = driver.open(n)
    var dirt = _Cells(n, _DIRT)
    require(driver.upload(handle, dirt.address(), dirt.address(), n) == n, "dirty upload count differs")
    _ = driver.close(handle)
    return handle


def _case(driver: OptimizerNativeDriver, case_id: Int) raises:
    if case_id == 0:
        var handle = driver.open(5)
        try:
            _expect_fresh(driver, handle, 5, "fresh slot")
        finally:
            _ = driver.close(handle)
    elif case_id == 1:
        var closed = _dirty_close(driver, 6)
        var handle = driver.open(6)
        try:
            require(handle == closed, "the pool did not reuse the freed handle; the reuse case is not exercised")
            _expect_fresh(driver, handle, 6, "reopened slot")
        finally:
            _ = driver.close(handle)
    else:
        var closed = _dirty_close(driver, 3)
        var handle = driver.open(9)
        try:
            require(handle == closed, "the pool did not reuse the freed handle; the reuse case is not exercised")
            _expect_fresh(driver, handle, 9, "grown slot")
        finally:
            _ = driver.close(handle)


def run_cases(column: Column, mut report: Report):
    var cases = report.cases(_CASES)
    for case_id in range(_CASES):  # serial: one process-wide optimizer pool (see the module docstring)
        try:
            comptime if CPU_COLUMN_BUILD:
                _case(OptimizerNativeDriver(optimizer_host_v1, True), case_id)
            else:
                _case(OptimizerNativeDriver(optimizer_gpu_v1, False), case_id)
        except error:
            cases.fail(case_id, error)


def register() -> Check:
    return Check("fresh_state_zeros", cpu=True, gpu=True,
        serial="one process-wide optimizer pool; the reuse cases need the freed handle", run=run_cases)
