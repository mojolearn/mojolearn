# SPDX-License-Identifier: Apache-2.0
from sequence.exec_trait import Exec
from sequence.mlp_fit import MLPNet, mlp_fit, mlp_predict
from sequence.mlp import LOSS_SQUARED, LOSS_BINARY_LOG, LOSS_LOG, ACT_SOFTMAX
from sequence.ops import FP


def _ptr(mut values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def run_mlp_cases[E: Exec](mut ex: E) raises:
    var cases = 0
    for activation in range(4):
        for solver in range(2):
            for loss in range(3):
                for batches in range(2):
                    var outputs = 1 if loss == LOSS_BINARY_LOG else 2
                    var out_activation = 0 if loss == LOSS_SQUARED else (1 if loss == LOSS_BINARY_LOG else ACT_SOFTMAX)
                    var sizes: List[Int] = [3, 5, 4, outputs]
                    var net = MLPNet(sizes^, activation, out_activation)
                    var x = List[Float32](length=33, fill=0)
                    var y = List[Float32](length=11 * outputs, fill=0)
                    var params = List[Float32](length=net.n_params(), fill=0)
                    var curve = List[Float32](length=5, fill=0)
                    for i in range(len(x)):
                        x[i] = Float32((i * 7) % 13 - 6) / Float32(8)
                    for r in range(11):
                        if loss == LOSS_SQUARED:
                            for c in range(outputs):
                                y[r * outputs + c] = Float32((r + c * 3) % 7 - 3) / Float32(4)
                        elif loss == LOSS_BINARY_LOG:
                            y[r] = Float32(r % 2)
                        else:
                            y[r * outputs + r % outputs] = 1
                    for i in range(len(params)):
                        params[i] = Float32((i * 3) % 17 - 8) / Float32(32)
                    var n = mlp_fit(
                        ex, net, _ptr(x), _ptr(y), 11, _ptr(params), _ptr(curve), loss, solver,
                        batches, True, 4 if batches == 0 else 11, 5, True, UInt64(731), 8,
                        Float32(0.01), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.8),
                        Float64(0.5), Float32(0.02), Float64(1.0e-8),
                    )
                    _ = x^
                    _ = y^
                    print("CASE", cases, "epochs", n)
                    for i in range(len(params)):
                        print("PARAM", cases, i, params[i])
                    for i in range(n):
                        print("LOSS", cases, i, curve[i])
                    cases += 1
    print("PASS mlp_cases", cases)
