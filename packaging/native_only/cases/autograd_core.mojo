# SPDX-License-Identifier: Apache-2.0
# Native tape construction, branching, backward, and consumed gradient oracle.
from autograd import Tape
from sequence.exec_trait import Exec


def check_tape[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var x = tape.tensor(ex, [Float32(2), Float32(-3)], 1, 2, True)
    var xx = tape.multiply(ex, x, x)
    var branch = tape.add(ex, xx, x)
    var loss = tape.sum(ex, branch)
    tape.backward(ex, loss)
    var g = tape.gradients(ex, x)
    if len(g) != 2 or g[0] != 5 or g[1] != -5:
        raise Error("native-only tape: derivative of x*x+x must be 2*x+1")
