# SPDX-License-Identifier: Apache-2.0
from sequence.exec_trait import Exec
from sequence.mlp_fit import MLPNet, mlp_fit, mlp_predict
from sequence.mlp_progress import MLPProgress
from sequence.mlp import LOSS_SQUARED, LOSS_BINARY_LOG, LOSS_LOG, ACT_SOFTMAX
from sequence.ops import FP
from std.memory import bitcast


def _ptr(mut values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def check_mlp_continuation[E: Exec](mut ex: E, solver: Int) raises:
    var net = MLPNet([2, 2, 1], 2, 0)
    var x: List[Float32] = [1, 0.5, -1, 0.25, 0.75, -0.5]
    var y: List[Float32] = [0.25, 0.75, -1]
    var full = List[Float32](length=net.n_params(), fill=0.25)
    var partial = full.copy()
    var losses = List[Float32](length=2, fill=0)
    var moments = List[Float32](length=2 * net.n_params(), fill=0)
    var order = List[Float32](length=3, fill=0)
    var progress = MLPProgress(0, 0, 0, 731, 731, 0, 1, 1, 0)
    # The first run is the uninterrupted reference; the next two share only
    # the state exported to the public boundary between calls.
    for call in range(3):
        var split = call != 0
        var n = mlp_fit(ex, net, _ptr(x), _ptr(y), 3, _ptr(partial) if split else _ptr(full), _ptr(losses),
                        LOSS_SQUARED, solver, 1, True, 2, 1 if split else 2, True, UInt64(731), 8,
                        0.02, 0.9, 0.999, 1e-8, 0.8, 0.5, 0.02, 1e-8, 0, 0, False,
                        Int(MutPointer(to=progress)) if split else 0,
                        Int(_ptr(moments)) if call == 2 else 0, Int(_ptr(moments)) if split else 0,
                        Int(_ptr(order)) if call == 2 else 0, Int(_ptr(order)) if split else 0)
        if n != (1 if split else 2):
            raise Error("unexpected MLP epoch count")
    if progress.epochs != 2 or progress.samples != 6:
        raise Error("MLP lost its continuation counters")
    for i in range(len(full)):
        if bitcast[DType.uint32](full[i]) != bitcast[DType.uint32](partial[i]):
            raise Error("split MLP training disagrees with uninterrupted weights")


def run_mlp_cases[E: Exec](mut ex: E) raises:
    check_mlp_continuation(ex, 0)
    check_mlp_continuation(ex, 1)
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
                    # Bitwise digest of the fitted parameters and the loss
                    # curve: tools/check_autograd.py compares it with the stored
                    # reference (mlp_cases_digests.tsv) for every build (default
                    # fused tape, AUTOGRAD_MLP_UNFUSED) and executor.
                    var digest = UInt64(1469598103934665603)
                    for i in range(len(params)):
                        digest = (digest ^ UInt64(bitcast[DType.uint32](params[i]))) * UInt64(1099511628211)
                    for i in range(n):
                        digest = (digest ^ UInt64(bitcast[DType.uint32](curve[i]))) * UInt64(1099511628211)
                    print("DIGEST", cases, digest)
                    cases += 1
    print("PASS mlp_cases", cases)
