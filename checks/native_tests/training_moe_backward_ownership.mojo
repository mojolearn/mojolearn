# SPDX-License-Identifier: Apache-2.0
"""The MoE backward keeps every output buffer until its writer finishes
(moved from python/mojolearn/tests/test_moe_backward_ownership.py, lane
tmig-training-2, 2026-11-15), through the entry every binding calls,
`autograd/route_ops.mojo::moe_backward_addrs`.

One case per row count (7 and 256): x (rows, 16) and dy hashed normals;
two blocks in turn on the same executor (the second, differently shaped,
exercises allocator reuse): width 24, 6 experts, top 2, renormalised; then
width 12, 5 experts, top 3, not renormalised, with router row 3 set to
router row 1 (a routing tie). For each block: dx, the router, gate_up and
down gradients are finite; unrelated host storage is allocated and consumed
after the native writes; a second backward into fresh outputs gives the same
bytes for all four. The former child process (heap corruption surfaced as
an actionable failure) is the runner's own process: a corrupted heap aborts
the check.

The weights are hashed uniforms in the block initialiser's ranges (the
original used the block's seeded constructor; its assertions never depended
on the values).
"""
from std.math import isfinite, sqrt
from autograd.route_ops import moe_backward_addrs
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import u01, normal
from checks.native_tests.training_check_common import same_words
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime MOE_OWNER_CASES = 2
comptime MOE_OWNER_D = 16


def _rows(unit: Int) -> Int:
    return 7 if unit == 0 else 256


def _normal(count: Int, salt: Int) -> List[Float32]:
    var values = List[Float32](length=count, fill=0.0)
    for i in range(count):  # small-loop(fixture fill): fixture-sized, no product work
        values[i] = Float32(normal(salt, i))
    return values^


def _uniform(count: Int, salt: Int, bound: Float64) -> List[Float32]:
    var values = List[Float32](length=count, fill=0.0)
    for i in range(count):  # small-loop(fixture fill): fixture-sized, no product work
        values[i] = Float32((2.0 * u01(salt, i) - 1.0) * bound)
    return values^


struct _Grads(Movable):
    var dx: List[Float32]
    var drouter: List[Float32]
    var dgate_up: List[Float32]
    var ddown: List[Float32]

    def __init__(out self, T: Int, D: Int, F: Int, En: Int):
        self.dx = List[Float32](length=T * D, fill=Float32(-7.0))
        self.drouter = List[Float32](length=En * D, fill=Float32(-7.0))
        self.dgate_up = List[Float32](length=En * 2 * F * D, fill=Float32(-7.0))
        self.ddown = List[Float32](length=En * D * F, fill=Float32(-7.0))


def _backward[E: Exec](
    mut ex: E, x: List[Float32], dy: List[Float32], router: List[Float32], gate_up: List[Float32],
    down: List[Float32], T: Int, F: Int, En: Int, k: Int, renorm: Int,
) raises -> _Grads:
    var D = MOE_OWNER_D
    var grads = _Grads(T, D, F, En)
    var y = List[Float32](length=T * D, fill=0.0)
    var logits = List[Float32](length=T * En, fill=0.0)
    var selected = List[Float32](length=T * k, fill=-1.0)
    var weights = List[Float32](length=T * k, fill=0.0)
    var addrs: List[Int] = [
        Int(x.unsafe_ptr()), Int(router.unsafe_ptr()), Int(gate_up.unsafe_ptr()), Int(down.unsafe_ptr()),
        Int(dy.unsafe_ptr()), Int(grads.dx.unsafe_ptr()), Int(grads.drouter.unsafe_ptr()),
        Int(grads.dgate_up.unsafe_ptr()), Int(grads.ddown.unsafe_ptr()), Int(y.unsafe_ptr()),
        Int(logits.unsafe_ptr()), Int(selected.unsafe_ptr()), Int(weights.unsafe_ptr()),
    ]
    var ip: List[Int] = [T, D, F, En, k, renorm]
    _ = moe_backward_addrs(ex, addrs, ip, List[Int]())
    # every buffer the entry read or wrote stays alive past the call
    _ = x
    _ = dy
    _ = router
    _ = gate_up
    _ = down
    _ = y^
    _ = logits^
    _ = selected^
    _ = weights^
    return grads^


def _finite(values: List[Float32], what: String) raises:
    for i in range(len(values)):  # small-loop(check comparison): fixture-sized, no product work
        require(isfinite(values[i]), what + " is not finite at " + String(i))


def run_moe_owner_case[E: Exec](mut ex: E, unit: Int) raises:
    var T = _rows(unit)
    var D = MOE_OWNER_D
    var x = _normal(T * D, 0x912 + unit)
    var dy = _normal(T * D, 0x913 + unit)
    for block in range(2):  # driver-loop: over the two block shapes, in order (allocator reuse)
        var F = 24 if block == 0 else 12
        var En = 6 if block == 0 else 5
        var k = 2 if block == 0 else 3
        var renorm = 1 if block == 0 else 0
        var router = _uniform(En * D, 0x9120 + block, 1.0 / sqrt(Float64(D)))
        if renorm == 0:
            for c in range(D):  # small-loop(fixture tie): one router row
                router[3 * D + c] = router[1 * D + c]
        var gate_up = _uniform(En * 2 * F * D, 0x9130 + block, 1.0 / sqrt(Float64(D)))
        var down = _uniform(En * D * F, 0x9140 + block, 1.0 / sqrt(Float64(F)))
        var first = _backward(ex, x, dy, router, gate_up, down, T, F, En, k, renorm)
        var label = "block " + String(block) + " rows " + String(T) + ": "
        _finite(first.dx, label + "dx")
        _finite(first.drouter, label + "router grad")
        _finite(first.dgate_up, label + "gate_up grad")
        _finite(first.ddown, label + "down grad")
        # unrelated host storage allocated and consumed after the native writes
        var scratch = List[List[Float32]]()
        for i in range(6):  # small-loop(6: scratch arrays): fixture-sized, no product work
            scratch.append(List[Float32](length=T * D, fill=Float32(i)))
        var total = Float64(0)
        for i in range(6):  # small-loop(6: scratch arrays): fixture-sized, no product work
            total += Float64(scratch[i][0])
        require(total == 15.0, label + "scratch storage was overwritten")
        var again = _backward(ex, x, dy, router, gate_up, down, T, F, En, k, renorm)
        require(same_words(first.dx, again.dx), label + "dx differs on the repeated backward")
        require(same_words(first.drouter, again.drouter), label + "router grad differs on the repeated backward")
        require(same_words(first.dgate_up, again.dgate_up), label + "gate_up grad differs on the repeated backward")
        require(same_words(first.ddown, again.ddown), label + "down grad differs on the repeated backward")
        _ = scratch^
        _ = router^
        _ = gate_up^
        _ = down^
    _ = x^
    _ = dy^


# ---- The host column ----
# The two row counts run across host cores, each on its own HostExec with a
# private failure slot.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(MOE_OWNER_CASES)
    def task(unit: Int) {imm cases}:
        try:
            var ex = HostExec()
            run_moe_owner_case(ex, unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, MOE_OWNER_CASES)


# ---- The device column ----
# The two row counts run in order on one DeviceExec: they share the one device
# and its queue, and each backward's kernels already spread across the device.
def _run_gpu(mut report: Report) raises:
    var ex = DeviceExec()
    var cases = report.cases(MOE_OWNER_CASES)
    for unit in range(MOE_OWNER_CASES):  # driver-loop: over cases on the one device
        try:
            run_moe_owner_case(ex, unit)
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
    return Check("training_moe_backward_ownership", cpu=True, gpu=True, serial="", run=run_cases)
