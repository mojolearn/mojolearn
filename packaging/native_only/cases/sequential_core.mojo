# SPDX-License-Identifier: Apache-2.0
# Whole native model fit and predict, with an independent one-step SGD oracle.
from autograd.model import FitConfig, LOSS_MSE, TARGET_DENSE, SCHED_CONSTANT, OUT_RAW, parse_model, model_fit, model_predict
from sequence.exec_trait import Exec
from sequence.ops import FP, OPT_SGD
from sequence.recurrent import OptConfig
from core.cast_in import CastIn


def check_model[E: Exec](mut ex: E) raises:
    var program: List[Int] = [1, 1, 1, 1, 1, 1, 0]
    var table: List[Float32] = [0]
    var spec = parse_model(program, table)
    var x: List[Float32] = [1, 2, 3, 4]
    var y: List[Float32] = [3, 5, 7, 9]
    var params: List[Float32] = [0, 0]
    var curve: List[Float32] = [0]
    var output = List[Float32](length=4, fill=0)
    var opt = OptConfig(OPT_SGD, 0, 0, 0, Float32(1e-8), 0, 0)
    var config = FitConfig(LOSS_MSE, TARGET_DENSE, opt, Float32(0.01), SCHED_CONSTANT, 1, 1, 4, 1, False, UInt64(5), 0, 0, 1, False)
    var xin = CastIn.f32_c(Int(x.unsafe_ptr()), 4, 1)
    var yin = CastIn.f32_c(Int(y.unsafe_ptr()), 4, 1)
    var pp = FP(unsafe_from_address=Int(params.unsafe_ptr()))
    var count = model_fit(ex, spec, config, xin, yin, 4, pp, FP(unsafe_from_address=Int(curve.unsafe_ptr())))
    if count != 1 or abs(curve[0] - Float32(41)) > Float32(1e-5):
        raise Error("native-only model: mean squared loss must be 41")
    # dW = -2*mean(x*y) = -35; db = -2*mean(y) = -12.
    if abs(params[0] - Float32(0.35)) > Float32(1e-6) or abs(params[1] - Float32(0.12)) > Float32(1e-6):
        raise Error("native-only model: SGD parameters differ from closed form")
    model_predict(ex, spec, xin, 4, pp, FP(unsafe_from_address=Int(output.unsafe_ptr())), 2, OUT_RAW)
    for i in range(4):
        var want = Float32(0.35) * Float32(i + 1) + Float32(0.12)
        if abs(output[i] - want) > Float32(1e-6):
            raise Error("native-only model: consumed prediction differs from fitted affine function")
    ex.sync()
    _ = x^
    _ = y^
    _ = params^
    _ = curve^
    _ = output^
