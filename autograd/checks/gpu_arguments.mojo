# SPDX-License-Identifier: Apache-2.0
# Minimal caller for the retained Mojo/Metal argument-binding failure.
# This deliberately bypasses Tape, graph allocation, and backward traversal.
from sequence.exec_device import DeviceExec
from sequence.exec import HostExec
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from autograd.ops import OP_AUTOGRAD, ADD
from std.sys.compile import is_defined


def probe[E: Exec](mut ex: E) raises:
    var left: List[Float32] = [2, 3]
    var right: List[Float32] = [4, 5]
    var result = List[Float32](length=2, fill=0)
    var x = ex.alloc(2)
    var y = ex.alloc(2)
    var z = ex.alloc(2)
    ex.upload(x, FP(unsafe_from_address=Int(left.unsafe_ptr())), 2)
    ex.upload(y, FP(unsafe_from_address=Int(right.unsafe_ptr())), 2)
    _ = left^
    _ = right^
    var a = Args()
    a.p0 = x
    a.p1 = y
    a.p2 = z
    a.p3 = z
    a.p4 = z
    a.p5 = z
    a.p6 = z
    a.p7 = z
    a.p8 = z
    a.p9 = z
    a.p10 = z
    a.p11 = z
    a.i1 = ADD
    a.i2 = 1
    a.i3 = 2
    a.i4 = 1
    a.i5 = 2
    a.i6 = 1
    a.i7 = 2
    ex.launch[OP_AUTOGRAD](a, 2)
    ex.download(FP(unsafe_from_address=Int(result.unsafe_ptr())), z, 2)
    print("argument_probe", result[0], result[1], "expected", 6, 8)
    if result[0] != 6 or result[1] != 8:
        raise Error("native autograd argument probe failed")


def main() raises:
    comptime if is_defined["MOJOLEARN_AUTOGRAD_PROBE_CPU"]():
        var ex = HostExec()
        probe(ex)
    else:
        var ex = DeviceExec()
        probe(ex)
