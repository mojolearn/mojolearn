# SPDX-License-Identifier: Apache-2.0
# Minimal caller of the binary-forward body (OP_AG_BIN_FWD), the first
# operation of the split autograd family; it reproduced the Metal argument
# binding failure of the former combined body. It deliberately bypasses
# Tape, graph allocation, and backward traversal, and binds the slots the
# way the tape does: p0/p1/p2 live and distinct, the rest placeholders.
from sequence.exec_device import DeviceExec
from sequence.exec import HostExec
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from autograd.ops import OP_AG_BIN_FWD, ADD
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
    a.i1 = ADD
    a.i2 = 1
    a.i3 = 2
    a.i4 = 1
    a.i5 = 2
    a.i6 = 1
    a.i7 = 2
    ex.launch[OP_AG_BIN_FWD](a, 2)
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
