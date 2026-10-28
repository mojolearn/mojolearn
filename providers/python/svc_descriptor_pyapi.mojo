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
    var a = List[Int](Int(py=outputs[0]), Int(py=outputs[1]), Int(py=outputs[2]))
    var p = List[Int](Int(py=metadata[0]), Int(py=metadata[1]), count, Int(py=outputs[3]))
    for i in range(count):  # driver-loop: one native descriptor per class pair
        a.append(Int(py=supports[i]))
        a.append(Int(py=duals[i]))
        p.append(Int(py=metadata[3 + i * 3]))
        p.append(Int(py=metadata[4 + i * 3]))
        p.append(Int(py=metadata[5 + i * 3]))
    return PythonObject(svc_descriptor_execute(native, 10, a, p, []))

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
    return PythonObject(svc_descriptor_execute(native, 2,
        [Int(py=dec), Int(py=pairs), Int(py=ab), Int(py=out)], p, f))


def svc_platt_execute(binding: PythonObject, native: PythonObject, dec: PythonObject,
                       labels: PythonObject, out: PythonObject, n: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_platt_train"), dec, labels, out, n)
    return PythonObject(svc_descriptor_execute(native, 3,
        [Int(py=dec), Int(py=labels), Int(py=out)], [Int(py=n)], []))


def svc_splitmix_execute(binding: PythonObject, native: PythonObject, out: PythonObject,
                          n: PythonObject, lo: PythonObject, hi: PythonObject) raises -> PythonObject:
    if native is None:
        return py_call(neural_descriptor_entry(binding, "svc_splitmix_perm"), out, n, lo, hi)
    return PythonObject(svc_descriptor_execute(native, 4, [Int(py=out)], [Int(py=n), Int(py=lo), Int(py=hi)], []))


def svr_execute(binding: PythonObject, native: PythonObject, fit: Bool,
                addresses: PythonObject, params: PythonObject) raises -> PythonObject:
    if native is None:
        if fit:
            return py_call(neural_descriptor_entry(binding, "svr_fit"), addresses[0], addresses[1],
                addresses[2], addresses[3], addresses[4], addresses[5], params)
        return py_call(neural_descriptor_entry(binding, "svr_predict"), addresses[0], addresses[1],
            addresses[2], addresses[3], params)
    var n = len(params)
    if fit and n != 9 and n != 10 and n != 12:
        raise Error("svr_fit: params must contain 9 values (10 with the per-row bounds, 12 with degree and coef0), got " + String(n))
    if not fit and n != 7 and n != 9:
        raise Error("svr_predict: params must contain 7 values (9 with degree and coef0), got " + String(n))
    if len(addresses) != (6 if fit else 4):
        raise Error("SVR native ABI: wrong address count")
    var a = List[Int]()
    for i in range(len(addresses)):  # small-loop(six fit or four predict buffer addresses)
        a.append(Int(py=addresses[i]))
    var p = List[Int](length=n, fill=0)
    var f = List[Float64](length=n, fill=0.0)
    for i in range(n):  # small-loop(up to twelve stable scalar words)
        var floating = ((i >= 3 and i <= 6) or i == 11) if fit else (i == 3 or i == 5 or i == 6 or i == 8)
        if floating:
            f[i] = Float64(py=params[i])
        else:
            p[i] = Int(py=params[i])
    return PythonObject(svc_descriptor_execute(native, 6 if fit else 7, a, p, f))
