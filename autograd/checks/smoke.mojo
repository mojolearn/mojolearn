# SPDX-License-Identifier: Apache-2.0
from autograd import Tape
from sequence.exec import HostExec


def main() raises:
    var ex = HostExec()
    var tape = Tape()
    var x = tape.tensor(ex, [Float32(2.0), Float32(-3.0)], 1, 2, True)
    var xx = tape.multiply(ex, x, x)
    var branch = tape.add(ex, xx, x)
    var loss = tape.sum(ex, branch)
    tape.backward(ex, loss)
    var g = tape.gradients(ex, x)
    if g[0] != 5 or g[1] != -5:
        raise Error("branched gradient incorrect")
    print("PASS autograd branched graph")
