# SPDX-License-Identifier: Apache-2.0
# SVC's scalar adapters all use the existing shared descriptor provider.
# Legacy compiled entries remain selectable for a bounded same-build A/B.
from std.python import PythonObject
from bindings.py_balanced import py_call
from providers.python.neural_descriptor_pyapi import neural_descriptor_execute, neural_descriptor_entry


def svc_epilogue_execute(binding: PythonObject, native: PythonObject,
                         dec: PythonObject, pairs: PythonObject, ab: PythonObject,
                         out: PythonObject, params: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_pair_epilogue"), dec, pairs, ab, out, params)
    if len(params) != 5:
        raise Error("svc_pair_epilogue: params must contain 5 values")
    var mode = Int(py=params[0])
    var p = List[Int](length=5, fill=0)
    var f = List[Float64](length=5, fill=0.0)
    p[0] = mode
    for i in range(1, 5):  # small-loop(four stable scalar words in the selected epilogue schema)
        if (mode == 8 and i == 3) or (mode <= 5 and i == 4):
            f[i] = Float64(py=params[i])
        else:
            p[i] = Int(py=params[i])
    return PythonObject(neural_descriptor_execute(native, 2,
        [Int(py=dec), Int(py=pairs), Int(py=ab), Int(py=out)], p, f))


def svc_platt_execute(binding: PythonObject, native: PythonObject, dec: PythonObject,
                       labels: PythonObject, out: PythonObject, n: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_platt_train"), dec, labels, out, n)
    return PythonObject(neural_descriptor_execute(native, 3,
        [Int(py=dec), Int(py=labels), Int(py=out)], [Int(py=n)], []))


def svc_splitmix_execute(binding: PythonObject, native: PythonObject, out: PythonObject,
                          n: PythonObject, lo: PythonObject, hi: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_splitmix_perm"), out, n, lo, hi)
    return PythonObject(neural_descriptor_execute(native, 4, [Int(py=out)], [Int(py=n), Int(py=lo), Int(py=hi)], []))
