# SPDX-License-Identifier: Apache-2.0
# A user-defined model and loss trained entirely in native Mojo.
from autograd import Tape
from sequence.exec import HostExec
from sequence.ops import FP, Args, OP_OPT, OPT_SGD


def main() raises:
    var ex = HostExec()
    var tape = Tape()
    var x_host: List[Float32] = [-1, -1, -1, 1, 1, -1, 1, 1, 0, 0, 0.5, -0.5]
    var y_host: List[Float32] = [-0.5, -2.5, 3.5, 1.5, 0.5, 2]
    var initial: List[Float32] = [0.1, -0.2, 0]
    var x_data = ex.alloc(12)
    var y_data = ex.alloc(6)
    var parameters = ex.alloc(3)
    ex.upload(x_data, FP(unsafe_from_address=Int(x_host.unsafe_ptr())), 12)
    ex.upload(y_data, FP(unsafe_from_address=Int(y_host.unsafe_ptr())), 6)
    ex.upload(parameters, FP(unsafe_from_address=Int(initial.unsafe_ptr())), 3)
    # The untracked pointer conversion does not retain its List owner.
    # Keep each source alive through its upload call.
    _ = x_host^
    _ = y_host^
    _ = initial^
    var first_loss = Float32(0)
    var last_loss = Float32(0)
    for step in range(80):
        tape.reset()
        var x = tape.variable(ex, x_data, 6, 2, False)
        var y = tape.variable(ex, y_data, 6, 1, False)
        var w = tape.variable(ex, parameters, 2, 1)
        var b = tape.variable(ex, parameters + 2, 1, 1)
        var projection = tape.matmul(ex, x, w)
        var prediction = tape.add(ex, projection, b)
        var loss = tape.mse_loss(ex, prediction, y)
        tape.backward(ex, loss)
        if step == 0 or step == 79:
            var scalar = tape.values(ex, loss)
            if step == 0:
                first_loss = scalar[0]
            last_loss = scalar[0]
        var update = Args()
        update.i0 = OPT_SGD
        update.f0 = 0.1
        update.p0 = parameters
        update.p1 = tape.gradient(w)
        ex.launch[OP_OPT](update, 2)
        update.p0 = parameters + 2
        update.p1 = tape.gradient(b)
        ex.launch[OP_OPT](update, 1)
    print("regression_loss", "initial", first_loss, "final", last_loss)
    if not (last_loss < first_loss * Float32(0.001)):
        raise Error("user-defined model did not learn")
    print("PASS custom native regression", "initial_loss", first_loss, "final_loss", last_loss)
