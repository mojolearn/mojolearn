# SPDX-License-Identifier: Apache-2.0
from std.memory import bitcast
from std.sys.compile import is_defined
from mojolearn import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from mojolearn.core.host_gemm_simd import host_gemm_identical
from mojolearn.gemm.contract import OP_NN
from mojolearn.training.byte_lm_config import ByteConfig
from mojolearn.training.checks.optimizer_contract import OptimizerConfig, OPT_ADAMW
from mojolearn.training.optimizer_host_rows import adam_host_rows


def main() raises:
    comptime expected = NUMERIC_FAST if is_defined["MOJOLEARN_NUMERIC_FAST"]() else NUMERIC_IDENTICAL
    comptime assert GLOBAL_NUMERIC_MODE == expected, "incorrect package default"
    var a: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var b: List[Float32] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    var want: List[Float32] = [58.0, 64.0, 139.0, 154.0]
    var c = host_gemm_identical(a, b, OP_NN, 2, 2, 3)
    for i in range(4):
        if bitcast[DType.uint32](c[i]) != bitcast[DType.uint32](want[i]):
            raise Error("incorrect GEMM result")
    var cfg = OptimizerConfig(OPT_ADAMW, 0.01, 0.9, 0.999, 1e-8, 0.01, 0.0, 0.0, False, 0.0)
    var params: List[Float32] = [1.0, -2.0, 0.5]
    var gradients: List[Float32] = [0.25, -0.5, 0.125]
    var m = List[Float32](length=3, fill=0.0)
    var v = List[Float32](length=3, fill=0.0)
    adam_host_rows(params, gradients, m, v, cfg, 1)
    if not (params[0] < 1.0 and params[1] > -2.0 and params[2] < 0.5):
        raise Error("optimizer did not update parameters")
    for i in range(3):
        print("adam", i, bitcast[DType.uint32](params[i]), bitcast[DType.uint32](m[i]), bitcast[DType.uint32](v[i]))
    var model = ByteConfig()
    model.validate()
    if model.n_total() != 34944:
        raise Error("incorrect model configuration")
    print("PASS native CPU: mode, GEMM, Adam, language-model configuration")
