# SPDX-License-Identifier: Apache-2.0
# Python door for distributed captured programs (distributed_entries.mojo):
# Python forwards arrays/options in one call; this file converts them and
# builds results. Topology, ownership, graph walks and checkpoint bytes stay
# in the typed entries.
from bindings.native_json import native_path
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.distributed_entries import (
    check_device_ids, distributed, distributed_open_native, distributed_load_native, distributed_run_native,
    distributed_state_native, distributed_residency_native, distributed_save_native, distributed_ready,
    distributed_close_native,
)
from providers.python.autograd_program_pyapi import array_input, output_array
from providers.python.autograd_program_arrays_pyapi import initialize_program_arrays, host_array_py, host_arrays_py
from autograd.rankn import ShapeN
from core.cast_in import CastIn


def distributed_object(cls: PythonObject, binding: PythonObject, handle: PythonObject) raises -> PythonObject:
    # Allocate the Python-facing shell through C builtins, without executing
    # its Python __init__. The native registry owns all execution state.
    var b = Python.import_module("builtins")
    var out = py_call(b.object.__new__, cls)
    _ = py_call(b.setattr, out, "_b", binding)
    _ = py_call(b.setattr, out, "_id", handle)
    return out


def distributed_devices(devices: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for k in range(len(devices)):  # small-loop(devices: physical device IDs): argument parsing, no data
        out.append(Int(py=devices[k]))
    check_device_ids(out)
    return out^


def distributed_open[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, device_ids: List[Int], logical_shards: PythonObject,
    optimizer: PythonObject, parameters: PythonObject, seed: PythonObject, var executors: List[E],
) raises -> PythonObject:
    var selected = List[Int]()
    for k in range(len(parameters)):  # small-loop(parameters: the selected trainable handles): argument parsing, no data
        selected.append(Int(py=parameters[k]))
    return PythonObject(distributed_open_native[E, KIND](
        Int(py=sid), Int(py=pid), device_ids, Int(py=logical_shards), Int(py=optimizer), selected,
        Int(py=seed), executors^,
    ))


def distributed_load[E: Exec & Movable & Deinitable, KIND: Int](
    path: PythonObject, device_ids: List[Int], var executors: List[E],
) raises -> PythonObject:
    initialize_program_arrays()
    return PythonObject(distributed_load_native[E, KIND](String(py=native_path(path)), device_ids, executors^))


def distributed_fit_py[E: Exec & Movable & Deinitable, KIND: Int](
    id: PythonObject, arrays: PythonObject, targets: PythonObject, loss: PythonObject,
    epochs: PythonObject, batch: PythonObject, shuffle: PythonObject, delta: PythonObject,
) raises -> PythonObject:
    var name = String(py=loss)
    var kind = -1
    if name == "output":
        kind = 0
    elif name == "mse":
        kind = 1
    elif name == "mae":
        kind = 2
    elif name == "huber":
        kind = 3
    elif name == "bce_with_logits":
        kind = 4
    elif name == "cross_entropy":
        kind = 5
    else:
        raise Error("distributed fit: unknown loss")
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # small-loop(arrays: one descriptor per program input): argument parsing, no data
        var parsed = array_input(arrays[k], "distributed input")
        srcs.append(parsed[0])
        shapes.append(parsed[1])
    var target = CastIn.f32_c(0, 1, 1)
    if kind != 0:
        target = array_input(targets, "distributed target")[0]
    var ne = Int(py=epochs)
    if ne < 1:
        raise Error("distributed fit: epochs must be positive")
    var curve = output_array(ShapeN.of([ne]))
    var p = distributed[E, KIND](Int(py=id))
    p[].fit(srcs, shapes, target, kind, Float32(Float64(py=delta)), ne, Int(py=batch), Bool(py=shuffle),
            FP(unsafe_from_address=Int(py=curve.__array_interface__["data"][0])))
    return curve


def distributed_run_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, arrays: PythonObject) raises -> PythonObject:
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # small-loop(arrays: one descriptor per program input): argument parsing, no data
        var parsed = array_input(arrays[k], "distributed inference input")
        srcs.append(parsed[0])
        shapes.append(parsed[1])
    return host_array_py(distributed_run_native[E, KIND](Int(py=id), srcs, shapes))


def distributed_save_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, path: PythonObject) raises -> PythonObject:
    return PythonObject(distributed_save_native[E, KIND](Int(py=id), String(py=native_path(path))))


def distributed_state_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var r = distributed_state_native[E, KIND](Int(py=id))
    var result = Python.dict()
    py_setitem(result, "parameters", host_arrays_py(r.parameters))
    py_setitem(result, "first_moments", host_arrays_py(r.first_moments))
    py_setitem(result, "second_moments", host_arrays_py(r.second_moments))
    py_setitem(result, "epoch", PythonObject(r.epoch))
    py_setitem(result, "step", PythonObject(r.step))
    py_setitem(result, "seed", PythonObject(r.seed))
    py_setitem(result, "logical_shards", PythonObject(r.logical_shards))
    return result


def distributed_residency_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    var r = distributed_residency_native[E, KIND](Int(py=id))
    var result = Python.dict()
    var devices = Python.list()
    var owned = Python.list()
    var ranges = Python.list()
    for rank in range(len(r.devices)):  # small-loop(devices: one record per physical device): result packing, no data
        _ = py_call(devices.append, PythonObject(r.devices[rank]))
        var row = Python.list()
        for j in range(r.n_params):  # small-loop(params: one range per program parameter): result packing, no data
            var pair = Python.list()
            _ = py_call(pair.append, PythonObject(r.first[rank * r.n_params + j]))
            _ = py_call(pair.append, PythonObject(r.width[rank * r.n_params + j]))
            _ = py_call(row.append, pair)
        _ = py_call(owned.append, PythonObject(r.owned[rank]))
        _ = py_call(ranges.append, row)
    py_setitem(result, "devices", devices)
    py_setitem(result, "optimizer_elements_per_rank", owned)
    py_setitem(result, "optimizer_ranges", ranges)
    py_setitem(result, "parameter_residency", PythonObject("replicated"))
    py_setitem(result, "optimizer_residency", PythonObject("disjoint contiguous ranges"))
    py_setitem(result, "logical_shards", PythonObject(r.logical_shards))
    py_setitem(result, "transport", PythonObject("native host staging"))
    return result


def distributed_enter_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, owner: PythonObject) raises -> PythonObject:
    distributed_ready[E, KIND](Int(py=id))
    return owner


def distributed_close_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    distributed_close_native[E, KIND](Int(py=id))
    return PythonObject(0)
