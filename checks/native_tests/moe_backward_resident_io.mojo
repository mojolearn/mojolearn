# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""MoEBlock.backward on resident x / dy / dx (lane vdf-seq, 2026-11-24,
addendum 3 of the vd-functions briefs): `autograd/route_ops.mojo::moe_backward_io`
with the three tensors already in the executor's memory (read and written in
place) gives the same bytes as the host-input entry `moe_backward_addrs`
(x and dy uploaded, dx downloaded) for dx and the three weight gradients, the
forward output and the routing outputs, on the column it runs on.

One case per row count (5 and 96): width 16, 6 experts, top 2,
renormalised; x, dy hashed normals, weights hashed uniforms in the block
initialiser's ranges. The resident tensors are made with the executor's own
`alloc` / `upload` (a device buffer on the device column, host memory on the
CPU column), as `seq_tensor_alloc` / `seq_tensor_upload` make them.

On the device column the same case also runs the host-input entry on a
`HostExec` in the same process (the CPU column's executor, the executor of
the CPU identical image): device resident bytes == device host-input bytes
== CPU column bytes, for every output.
"""
from std.math import sqrt
from autograd.route_ops import moe_backward_addrs, moe_backward_io
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP
from checks.native_tests.sequence_port_common import u01, normal
from checks.native_tests.training_check_common import same_words
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, check_cases_serial, require


comptime MOE_IO_CASES = 2
comptime MOE_IO_D = 16
comptime MOE_IO_F = 24
comptime MOE_IO_E = 6
comptime MOE_IO_K = 2


def _rows(unit: Int) -> Int:
    return 5 if unit == 0 else 96


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


struct _Out(Movable):
    var dx: List[Float32]
    var drouter: List[Float32]
    var dgate_up: List[Float32]
    var ddown: List[Float32]
    var y: List[Float32]
    var logits: List[Float32]
    var selected: List[Float32]
    var weights: List[Float32]

    def __init__(out self, T: Int):
        var D = MOE_IO_D
        var F = MOE_IO_F
        var En = MOE_IO_E
        var k = MOE_IO_K
        self.dx = List[Float32](length=T * D, fill=Float32(-7.0))
        self.drouter = List[Float32](length=En * D, fill=Float32(-7.0))
        self.dgate_up = List[Float32](length=En * 2 * F * D, fill=Float32(-7.0))
        self.ddown = List[Float32](length=En * D * F, fill=Float32(-7.0))
        self.y = List[Float32](length=T * D, fill=Float32(-7.0))
        self.logits = List[Float32](length=T * En, fill=Float32(-7.0))
        self.selected = List[Float32](length=T * k, fill=Float32(-7.0))
        self.weights = List[Float32](length=T * k, fill=Float32(-7.0))


def _addrs(x: List[Float32], dy: List[Float32], router: List[Float32], gate_up: List[Float32],
           down: List[Float32], mut o: _Out, resident: Bool) -> List[Int]:
    # resident: aa[0], aa[4], aa[5] are unread (0), as the door passes them
    var addrs: List[Int] = [
        0 if resident else Int(x.unsafe_ptr()), Int(router.unsafe_ptr()), Int(gate_up.unsafe_ptr()),
        Int(down.unsafe_ptr()), 0 if resident else Int(dy.unsafe_ptr()), 0 if resident else Int(o.dx.unsafe_ptr()),
        Int(o.drouter.unsafe_ptr()), Int(o.dgate_up.unsafe_ptr()), Int(o.ddown.unsafe_ptr()), Int(o.y.unsafe_ptr()),
        Int(o.logits.unsafe_ptr()), Int(o.selected.unsafe_ptr()), Int(o.weights.unsafe_ptr()),
    ]
    return addrs^


def _same(a: _Out, b: _Out, label: String) raises:
    # label names the two sides compared
    require(same_words(a.dx, b.dx), label + "dx differs")
    require(same_words(a.drouter, b.drouter), label + "router grad differs")
    require(same_words(a.dgate_up, b.dgate_up), label + "gate_up grad differs")
    require(same_words(a.ddown, b.ddown), label + "down grad differs")
    require(same_words(a.y, b.y), label + "y differs")
    require(same_words(a.logits, b.logits), label + "logits differs")
    require(same_words(a.selected, b.selected), label + "selected differs")
    require(same_words(a.weights, b.weights), label + "routing weights differs")


def run_moe_io_case[E: Exec](mut ex: E, unit: Int) raises -> _Out:
    var T = _rows(unit)
    var D = MOE_IO_D
    var F = MOE_IO_F
    var En = MOE_IO_E
    var k = MOE_IO_K
    var x = _normal(T * D, 0xA10 + unit)
    var dy = _normal(T * D, 0xA11 + unit)
    var router = _uniform(En * D, 0xA120 + unit, 1.0 / sqrt(Float64(D)))
    var gate_up = _uniform(En * 2 * F * D, 0xA130 + unit, 1.0 / sqrt(Float64(D)))
    var down = _uniform(En * D * F, 0xA140 + unit, 1.0 / sqrt(Float64(F)))
    var ip: List[Int] = [T, D, F, En, k, 1]
    # the host-input path
    var host = _Out(T)
    var host_addrs = _addrs(x, dy, router, gate_up, down, host, False)
    _ = moe_backward_addrs(ex, host_addrs, ip, List[Int]())
    # the resident path: x and dy uploaded once beforehand, dx read back after
    var rx = ex.alloc(T * D)
    var rdy = ex.alloc(T * D)
    var rdx = ex.alloc(T * D)
    ex.upload(rx, FP(unsafe_from_address=Int(x.unsafe_ptr())), T * D)
    ex.upload(rdy, FP(unsafe_from_address=Int(dy.unsafe_ptr())), T * D)
    var res = _Out(T)
    var res_addrs = _addrs(x, dy, router, gate_up, down, res, True)
    var io: List[Int] = [Int(rx), Int(rdy), Int(rdx)]
    _ = moe_backward_io(ex, res_addrs, ip, List[Int](), io)
    ex.download(FP(unsafe_from_address=Int(res.dx.unsafe_ptr())), rdx, T * D)
    ex.sync()
    _same(host, res, "rows " + String(T) + " (resident vs host input): ")
    ex.free(rx)
    ex.free(rdy)
    ex.free(rdx)
    _ = x^
    _ = dy^
    _ = router^
    _ = gate_up^
    _ = down^
    _ = res^
    return host^


# The host column: the two row counts across host cores, each on its own HostExec. The device column: in order on one
# DeviceExec (one device and its queue; each backward's kernels already spread across the device).
def test_moe_backward_resident_io() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_case(unit: Int) raises:
            var ex = HostExec()
            _ = run_moe_io_case(ex, unit)
        check_cases("moe_resident_io", MOE_IO_CASES, host_case)
    else:
        var ex = DeviceExec()
        def device_case(unit: Int) raises {mut ex}:
            var device = run_moe_io_case(ex, unit)
            # the CPU column, in this process
            var host_ex = HostExec()
            var cpu = run_moe_io_case(host_ex, unit)
            _same(device, cpu, "rows " + String(_rows(unit)) + " (device column vs CPU column): ")
        check_cases_serial("moe_resident_io", MOE_IO_CASES, device_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
