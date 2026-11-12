# SPDX-License-Identifier: Apache-2.0
# Python door for distributed captured programs (distributed_entries.mojo):
# Python forwards arrays/options in one call; this file converts them and
# builds results. Topology, ownership, graph walks and checkpoint bytes stay
# in the typed entries.
# The PACK half of the former `providers/python/autograd_distributed_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/autograd_distributed_pyapi_unpack.mojo`.
from bindings.native_json_door import native_path
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem, py_type
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.distributed_entries import (
    check_device_ids, distributed, distributed_open_native, distributed_load_native, distributed_run_native,
    distributed_state_native, distributed_residency_native, distributed_save_native, distributed_ready,
    distributed_close_native,
)
from providers.python.autograd_program_pyapi_door import program_input, output_array, _integer
from providers.python.autograd_program_arrays_pyapi_door import initialize_program_arrays, host_array_py, host_arrays_py, program_admit_value, program_native_input, program_numeric_scalar
from providers.python.autograd_owned_program_public_pyapi_door import owned_program_object_py
from autograd.rankn import ShapeN
from autograd.program_io import ProgramInput
from core.cast_in import CastIn
from autograd.autograd_distributed_run import distributed_loss_kind


from providers.python.autograd_distributed_pyapi_unpack import (
    owned_program_array_admission,
    owned_program_id_admission,
)


def distributed_open[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, device_ids: List[Int], logical_shards: PythonObject,
    optimizer: PythonObject, parameters: PythonObject, seed: PythonObject, var executors: List[E],
) raises -> PythonObject:
    program_admit_value(Python.tuple(sid, pid, logical_shards, optimizer, parameters, seed))
    var selected = List[Int]()
    var parameter_values = program_native_input(parameters)
    for k in range(len(parameter_values)):  # small-loop(parameters: the selected trainable handles): argument parsing, no data
        selected.append(_integer(parameter_values[k], "parameter", 1))
    return PythonObject(distributed_open_native[E, KIND](
        _integer(sid, "session", 1), _integer(pid, "program", 1), device_ids, _integer(logical_shards, "logical_shards", 1), _integer(optimizer, "optimizer", 1), selected,
        _integer(seed, "seed", 0), executors^,
    ))


def distributed_load[E: Exec & Movable & Deinitable, KIND: Int](
    path: PythonObject, device_ids: List[Int], var executors: List[E],
) raises -> PythonObject:
    return PythonObject(distributed_load_native[E, KIND](String(py=native_path(path)), device_ids, executors^))


def distributed_run_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, arrays: PythonObject) raises -> PythonObject:
    owned_program_array_admission(id, arrays)
    var retained = List[ProgramInput]()
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # small-loop(arrays: one descriptor per program input): argument parsing, no data
        var prepared = program_input(arrays[k], False)
        srcs.append(prepared.src)
        shapes.append(prepared.shape)
        retained.append(prepared^)
    var result = host_array_py(distributed_run_native[E, KIND](Int(py=id), srcs, shapes))
    _ = retained
    _ = arrays
    return result


def distributed_save_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, path: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    return PythonObject(distributed_save_native[E, KIND](Int(py=id), String(py=native_path(path))))


def distributed_state_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
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
    owned_program_id_admission(id)
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


def distributed_close_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    distributed_close_native[E, KIND](Int(py=id))
    return PythonObject(0)
