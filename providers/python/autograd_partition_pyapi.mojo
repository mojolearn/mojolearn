# SPDX-License-Identifier: Apache-2.0
# Python door for model/tensor/pipeline partitioned captured graphs
# (partition_entries.mojo): argument unpacking and results only.
from bindings.native_json import native_path
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.partition_plan import GraphBytes
from autograd.partition_entries import (
    partition, partition_open_native, partition_load_native, partition_run_native, partition_save_native,
    partition_state_native, partition_residency_native, partition_events, partition_ready, partition_close_native,
)
from providers.python.autograd_program_pyapi import program_input, output_array, _integer
from providers.python.autograd_program_arrays_pyapi import initialize_program_arrays, host_arrays_py, program_admit_value, program_native_input
from providers.python.autograd_distributed_pyapi import owned_program_id_admission, owned_program_array_admission, owned_program_fit_admission
from autograd.rankn import ShapeN
from autograd.program_io import ProgramInput
from core.cast_in import CastIn


def partition_open[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, devices: List[Int], stages: PythonObject,
    tensor_shards: PythonObject, microbatches: PythonObject, optimizer: PythonObject,
    parameters: PythonObject, seed: PythonObject, var executors: List[E],
) raises -> PythonObject:
    program_admit_value(Python.tuple(sid, pid, stages, tensor_shards, microbatches, optimizer, parameters, seed))
    var selected = List[Int]()
    var parameter_values = program_native_input(parameters)
    for k in range(len(parameter_values)):  # small-loop(parameters: the selected trainable handles): argument parsing, no data
        selected.append(_integer(parameter_values[k], "parameter", 1))
    return PythonObject(partition_open_native[E, KIND](
        _integer(sid, "session", 1), _integer(pid, "program", 1), devices, _integer(stages, "stages", 1), _integer(tensor_shards, "tensor_shards", 1), _integer(microbatches, "microbatches", 1),
        _integer(optimizer, "optimizer", 1), selected, _integer(seed, "seed", 0), executors^,
    ))


def partition_load[E: Exec & Movable & Deinitable, KIND: Int](
    path: PythonObject, devices: List[Int], var executors: List[E],
) raises -> PythonObject:
    return PythonObject(partition_load_native[E, KIND](String(py=native_path(path)), devices, executors^))


def partition_fit_py[E: Exec & Movable & Deinitable, KIND: Int](
    id: PythonObject, arrays: PythonObject, targets: PythonObject, loss: PythonObject,
    epochs: PythonObject, batch: PythonObject, shuffle: PythonObject, delta: PythonObject,
) raises -> PythonObject:
    owned_program_fit_admission(id, arrays, loss, epochs, batch, shuffle, delta)
    var retained = List[ProgramInput]()
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
        raise Error("partitioned fit: unknown loss")
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # small-loop(arrays: one descriptor per program input): argument parsing, no data
        var prepared = program_input(arrays[k], False)
        srcs.append(prepared.src)
        shapes.append(prepared.shape)
        retained.append(prepared^)
    var target = CastIn.f32_c(0, 1, 1)
    if kind != 0:
        var prepared_target = program_input(targets, False)
        target = prepared_target.src
        retained.append(prepared_target^)
    var ne = _integer(epochs, "epochs", 1)
    if ne < 1:
        raise Error("partitioned fit: epochs must be positive")
    var curve = output_array(ShapeN.of([ne]))
    var d = partition[E, KIND](Int(py=id))
    d[].fit(srcs, shapes, target, kind, Float32(Float64(py=delta)), ne, _integer(batch, "batch_size", 1), Bool(py=shuffle),
            FP(unsafe_from_address=Int(py=curve.__array_interface__["data"][0])))
    _ = retained  # CastIn borrows sequence backing stores through the complete fit
    _ = arrays
    _ = targets
    return curve


def graph_array_output(a: GraphBytes) raises -> PythonObject:
    var out = output_array(a.shape)
    memcpy(dest=FP(unsafe_from_address=Int(py=out.__array_interface__["data"][0])),
           src=a.words.unsafe_ptr(), count=a.shape.count())
    return out


def partition_run_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, arrays: PythonObject) raises -> PythonObject:
    owned_program_array_admission(id, arrays)
    var retained = List[ProgramInput]()
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # small-loop(arrays: one descriptor per program input): argument parsing, no data
        var prepared = program_input(arrays[k], False)
        srcs.append(prepared.src)
        shapes.append(prepared.shape)
        retained.append(prepared^)
    var result = graph_array_output(partition_run_native[E, KIND](Int(py=id), srcs, shapes))
    _ = retained
    _ = arrays
    return result


def partition_save_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, path: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    return PythonObject(partition_save_native[E, KIND](Int(py=id), String(py=native_path(path))))


def partition_state_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    var r = partition_state_native[E, KIND](Int(py=id))
    var result = Python.dict()
    py_setitem(result, "parameters", host_arrays_py(r.arrays[0]))
    py_setitem(result, "first_moments", host_arrays_py(r.arrays[1]))
    py_setitem(result, "second_moments", host_arrays_py(r.arrays[2]))
    py_setitem(result, "epoch", PythonObject(r.epoch))
    py_setitem(result, "step", PythonObject(r.step))
    py_setitem(result, "seed", PythonObject(r.seed))
    py_setitem(result, "logical_stages", PythonObject(r.stages))
    py_setitem(result, "tensor_shards", PythonObject(r.tensor_shards))
    py_setitem(result, "microbatches", PythonObject(r.microbatches))
    return result


def _int_list(xs: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for k in range(len(xs)):  # small-loop(xs: one word per device, parameter or node): result packing, no data
        _ = py_call(out.append, PythonObject(xs[k]))
    return out


def partition_residency_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    var r = partition_residency_native[E, KIND](Int(py=id))
    var result = Python.dict()
    py_setitem(result, "devices", _int_list(r.devices))
    py_setitem(result, "parameter_elements_per_rank", _int_list(r.counts))
    py_setitem(result, "parameter_owner_devices", _int_list(r.owners))
    py_setitem(result, "node_devices", _int_list(r.placements))
    py_setitem(result, "parameter_residency", PythonObject("one authoritative device owner per tensor shard"))
    py_setitem(result, "optimizer_residency", PythonObject("same unique owner as its parameter shard"))
    py_setitem(result, "activation_residency", PythonObject("native host-staged bytes with local recomputation"))
    py_setitem(result, "logical_stages", PythonObject(r.stages))
    py_setitem(result, "tensor_shards", PythonObject(r.tensor_shards))
    py_setitem(result, "microbatches", PythonObject(r.microbatches))
    py_setitem(result, "partitioned_products", PythonObject(r.partitioned_products))
    return result


def partition_profile_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    var events = partition_events[E, KIND](Int(py=id))
    var result = Python.list()
    for k in range(len(events)):  # small-loop(events: one row per recorded operation interval): result packing, no data
        var e = events[k]
        var row = Python.dict()
        py_setitem(row, "node", PythonObject(e.node))
        py_setitem(row, "microbatch", PythonObject(e.microbatch))
        py_setitem(row, "stage", PythonObject(e.stage))
        py_setitem(row, "device", PythonObject(e.device))
        py_setitem(row, "backward", PythonObject(e.backward))
        py_setitem(row, "start_ns", PythonObject(e.start_ns))
        py_setitem(row, "end_ns", PythonObject(e.end_ns))
        _ = py_call(result.append, row)
    return result


def partition_enter_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, owner: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    partition_ready[E, KIND](Int(py=id))
    return owner


def partition_close_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    partition_close_native[E, KIND](Int(py=id))
    return PythonObject(0)
