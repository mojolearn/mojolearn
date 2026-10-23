# SPDX-License-Identifier: Apache-2.0
# External test executable: compare the frozen adapter library against direct
# Tape calls on the SAME AMD device. This is not a cross-vendor campaign.
from std.ffi import OwnedDLHandle
from std.memory import bitcast
from std.sys import argv
from sequence.exec_device import DeviceExec
from sequence.ops import FP
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN


def desc(pointer: FP, shape: ShapeN) -> List[Int]:
    var words = List[Int](length=11, fill=1)
    words[0] = Int(pointer)
    words[1] = 0
    words[2] = shape.ndim
    for k in range(shape.ndim):
        words[k + 3] = shape.d[k]
    return words^


def values(n: Int, salt: Int) -> List[Float32]:
    var result = List[Float32]()
    for k in range(n):
        result.append(Float32(1 + (k * 43 + salt * 71) % 113) / Float32(37))
    return result^


def check(lib: OwnedDLHandle, code: Int, rows: Int, features: Int, columns: Int) raises:
    var ex = DeviceExec()
    var tape = Tape()
    var a = values(rows * features, 3)
    var b = values(features * columns if code == 1 else rows * features, 7)
    var ap = ex.alloc(len(a))
    var bp = ex.alloc(len(b))
    ex.upload(ap, FP(unsafe_from_address=Int(a.unsafe_ptr())), len(a))
    ex.upload(bp, FP(unsafe_from_address=Int(b.unsafe_ptr())), len(b))
    var x = tape.variable(ex, ap, rows, features, False)
    var y = tape.variable(ex, bp, features if code == 1 else rows, columns if code == 1 else features, False)
    var reference: Tensor
    var inputs = desc(ap, ShapeN.matrix(rows, features))
    var count = 1
    var ints = List[Int]()
    if code <= 5:
        count = 2
        var other = desc(bp, ShapeN.matrix(features if code == 1 else rows, columns if code == 1 else features))
        for word in other:
            inputs.append(word)
    if code == 1:
        reference = tape.matmul(ex, x, y)
    elif code == 2:
        reference = tape.add(ex, x, y)
    elif code == 3:
        reference = tape.subtract(ex, x, y)
    elif code == 4:
        reference = tape.multiply(ex, x, y)
    elif code == 5:
        reference = tape.divide(ex, x, y)
    elif code == 6:
        reference = tape.relu(ex, x)
    elif code == 7:
        reference = tape.sigmoid(ex, x)
    elif code == 8:
        reference = tape.tanh(ex, x)
    elif code == 9:
        reference = tape.exp(ex, x)
    elif code == 10:
        reference = tape.log(ex, x)
    elif code == 11:
        reference = tape.sqrt(ex, x)
    elif code == 13:
        reference = tape.transpose(ex, x)
    elif code == 15:
        ints.append(-1)
        reference = tape.sum(ex, x)
    elif code == 16:
        ints.append(-1)
        reference = tape.mean(ex, x)
    elif code == 17:
        reference = tape.softmax(ex, x)
    elif code == 18:
        reference = tape.log_softmax(ex, x)
    else:
        raise Error("unknown test operation")
    var shape = tape.shape_of(reference)
    var n = shape.count()
    var expected = List[Float32](length=n, fill=0)
    ex.download(FP(unsafe_from_address=Int(expected.unsafe_ptr())), tape.value_of(ex, reference), n)
    var destination = ex.alloc(n)
    var output = desc(destination, shape)
    var error = List[UInt8](length=2048, fill=0)
    ex.sync()
    var status = lib.get_function[Int]("mojolearn_adapter_operation")(
        Int(0), code, Int(inputs.unsafe_ptr()), count, Int(output.unsafe_ptr()),
        Int(ints.unsafe_ptr()), len(ints), Int(0), Int(0), Int(error.unsafe_ptr()), len(error))
    if status != 0:
        raise Error(String(unsafe_from_utf8_ptr=error.unsafe_ptr()))
    var actual = List[Float32](length=n, fill=0)
    ex.download(FP(unsafe_from_address=Int(actual.unsafe_ptr())), destination, n)
    ex.sync()
    for k in range(n):
        if bitcast[DType.uint32](actual[k]) != bitcast[DType.uint32](expected[k]):
            raise Error("adapter differs from direct Tape: code=" + String(code) + " word=" + String(k))
    _ = a
    _ = b
    _ = inputs
    _ = output
    _ = ints
    tape.release(ex)
    ex.free(ap)
    ex.free(bp)
    ex.free(destination)
    ex.sync()


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: direct_equivalence_check /absolute/native-library.so")
    var lib = OwnedDLHandle(args[1])
    var count = 0
    for shape in range(3):
        var rows = 2 if shape == 0 else (17 if shape == 1 else 3)
        var features = 3 if shape == 0 else (19 if shape == 1 else 65)
        var columns = 4 if shape == 0 else (7 if shape == 1 else 5)
        for code in range(1, 19):
            if code == 12 or code == 14:
                continue
            check(lib, code, rows, features, columns)
            count += 1
    print("PASS direct Tape versus native adapter cases=", count, "device=AMD words=exact")
