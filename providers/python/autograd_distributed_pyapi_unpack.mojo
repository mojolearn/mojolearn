# SPDX-License-Identifier: Apache-2.0
# Python door for distributed captured programs (distributed_entries.mojo):
# Python forwards arrays/options in one call; this file converts them and
# builds results. Topology, ownership, graph walks and checkpoint bytes stay
# in the typed entries.
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


def distributed_object(cls: PythonObject, binding: PythonObject, handle: PythonObject) raises -> PythonObject:
    return owned_program_object_py(cls, binding, handle)


def distributed_devices(devices: PythonObject) raises -> List[Int]:
    program_admit_value(devices)
    if py_type(devices) is not py_type(Python.tuple()) and py_type(devices) is not py_type(Python.list()):
        raise Error("TypeError: devices require an exact tuple or list of native integers")
    var out = List[Int]()
    for k in range(len(devices)):  # small-loop(devices: physical device IDs): argument parsing, no data
        out.append(_integer(devices[k], "device", 0))
    check_device_ids(out)
    return out^


def distributed_fit_py[E: Exec & Movable & Deinitable, KIND: Int](
    id: PythonObject, arrays: PythonObject, targets: PythonObject, loss: PythonObject,
    epochs: PythonObject, batch: PythonObject, shuffle: PythonObject, delta: PythonObject,
) raises -> PythonObject:
    owned_program_fit_admission(id, arrays, loss, epochs, batch, shuffle, delta)
    var retained = List[ProgramInput]()
    var name = String(py=loss)
    var kind = distributed_loss_kind(name)
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
        raise Error("distributed fit: epochs must be positive")
    var curve = output_array(ShapeN.of([ne]))
    var p = distributed[E, KIND](Int(py=id))
    p[].fit(srcs, shapes, target, kind, Float32(Float64(py=delta)), ne, _integer(batch, "batch_size", 1), Bool(py=shuffle),
            FP(unsafe_from_address=Int(py=curve.__array_interface__["data"][0])))
    _ = retained  # CastIn borrows sequence backing stores through the complete fit
    _ = arrays
    _ = targets
    return curve


def distributed_enter_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, owner: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    distributed_ready[E, KIND](Int(py=id))
    return owner


def owned_program_id_admission(id: PythonObject) raises:
    if py_type(id) is not py_type(PythonObject(0)):
        raise Error("TypeError: captured program handle requires an exact native integer")


def owned_program_array_admission(id: PythonObject, arrays: PythonObject) raises:
    owned_program_id_admission(id)
    if py_type(arrays) is not py_type(Python.tuple()):
        raise Error("TypeError: captured program input envelope requires an exact tuple")


def owned_program_fit_admission(id: PythonObject, arrays: PythonObject, loss: PythonObject,
                                epochs: PythonObject, batch: PythonObject, shuffle: PythonObject, delta: PythonObject) raises:
    owned_program_array_admission(id, arrays)
    if py_type(loss) is not py_type(PythonObject("")):
        raise Error("TypeError: captured program loss requires an exact native string")
    if not program_numeric_scalar(epochs) or not program_numeric_scalar(batch) or not program_numeric_scalar(shuffle) or not program_numeric_scalar(delta):
        raise Error("TypeError: captured program fit settings require native numeric scalars")
