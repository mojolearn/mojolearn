# SPDX-License-Identifier: Apache-2.0
# SVC's scalar adapters all use the existing shared descriptor provider.
# Legacy compiled entries remain selectable for a bounded same-build A/B.
from std.python import PythonObject
from bindings.py_balanced import py_call
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from svm.operation_metadata import SVC_ABI_NAME
from providers.python.neural_descriptor_pyapi import neural_descriptor_execute, neural_descriptor_entry



def svc_descriptor_execute(snapshot: PythonObject, op: Int, a: List[Int], p: List[Int],
                           f: List[Float64]) raises -> Int:
    var owner = snapshot.downcast_value_ptr[CompiledBindingSnapshot]()
    if owner[].abi != SVC_ABI_NAME:
        raise Error("SVC requires its operation-local binding snapshot")
    return neural_descriptor_execute(owner[].provider, op, a, p, f)


def svc_descriptor_flags(snapshot: PythonObject) raises -> Int:
    return snapshot.downcast_value_ptr[CompiledBindingSnapshot]()[].capabilities[0]


def svc_ovo_execute(binding: PythonObject, native: PythonObject, supports: PythonObject,
                    duals: PythonObject, metadata: PythonObject, outputs: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_ovo_layout"), supports, duals, metadata, outputs)
    if len(metadata) < 3 or len(outputs) != 4:
        raise Error("svc_ovo_layout: malformed pair or output descriptor")
    var count = Int(py=metadata[2])
    if len(supports) != count or len(duals) != count or len(metadata) != 3 + 3 * count:
        raise Error("svc_ovo_layout: one support and one dual address per pair, meta [k, n, P, (i, j, m) * P]")
    var a: List[Int] = [Int(py=outputs[0]), Int(py=outputs[1]), Int(py=outputs[2])]
    var p: List[Int] = [Int(py=metadata[0]), Int(py=metadata[1]), count, Int(py=outputs[3])]
    for i in range(count):  # driver-loop: one native descriptor per class pair
        a.append(Int(py=supports[i]))
        a.append(Int(py=duals[i]))
        p.append(Int(py=metadata[3 + i * 3]))
        p.append(Int(py=metadata[4 + i * 3]))
        p.append(Int(py=metadata[5 + i * 3]))
    return PythonObject(svc_descriptor_execute(native, 10, a, p, []))

def svc_epilogue_execute(binding: PythonObject, native: PythonObject,
                         dec: PythonObject, pairs: PythonObject, ab: PythonObject,
                         output: PythonObject, params: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_pair_epilogue"), dec, pairs, ab, output, params)
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
    return PythonObject(svc_descriptor_execute(native, 2,
        [Int(py=dec), Int(py=pairs), Int(py=ab), Int(py=output)], p, f))


def svc_platt_execute(binding: PythonObject, native: PythonObject, dec: PythonObject,
                       labels: PythonObject, output: PythonObject, n: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_platt_train"), dec, labels, output, n)
    return PythonObject(svc_descriptor_execute(native, 3,
        [Int(py=dec), Int(py=labels), Int(py=output)], [Int(py=n)], []))


def svc_splitmix_execute(binding: PythonObject, native: PythonObject, output: PythonObject,
                          n: PythonObject, lo: PythonObject, hi: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_splitmix_perm"), output, n, lo, hi)
    return PythonObject(svc_descriptor_execute(native, 4, [Int(py=output)], [Int(py=n), Int(py=lo), Int(py=hi)], []))
