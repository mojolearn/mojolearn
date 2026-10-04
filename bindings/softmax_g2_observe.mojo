# SPDX-License-Identifier: Apache-2.0
from std.python import PythonObject
from experiments.apple_fast.gemm.softmax_narrow import SOFTMAX_G2, softmax_count, softmax_last, softmax_reset

def softmax_g2_state_binding() raises -> PythonObject:
    return PythonObject(Int(SOFTMAX_G2))

def softmax_g2_count_binding(index: PythonObject) raises -> PythonObject:
    return PythonObject(softmax_count(Int(py=index)))

def softmax_g2_last_binding(index: PythonObject) raises -> PythonObject:
    return PythonObject(softmax_last(Int(py=index)))

def softmax_g2_reset_binding() raises -> PythonObject:
    softmax_reset()
    return PythonObject(1)
