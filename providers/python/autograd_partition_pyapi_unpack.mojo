# SPDX-License-Identifier: Apache-2.0
# Python door for model/tensor/pipeline partitioned captured graphs
# (partition_entries.mojo): argument unpacking and results only.
from bindings.native_json_door import native_path
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.partition_plan import GraphBytes
from autograd.partition_entries import (
    partition, partition_open_native, partition_load_native, partition_run_native, partition_save_native,
    partition_state_native, partition_residency_native, partition_events, partition_ready, partition_close_native,
)
from providers.python.autograd_program_pyapi_door import program_input, output_array, _integer
from providers.python.autograd_program_arrays_pyapi_door import initialize_program_arrays, host_arrays_py, program_admit_value, program_native_input
from providers.python.autograd_distributed_pyapi_unpack import owned_program_id_admission, owned_program_array_admission, owned_program_fit_admission
from autograd.rankn import ShapeN
from autograd.program_io import ProgramInput
from core.cast_in import CastIn
from autograd.autograd_partition_run import partition_loss_kind


def partition_fit_py[E: Exec & Movable & Deinitable, KIND: Int](
    id: PythonObject, arrays: PythonObject, targets: PythonObject, loss: PythonObject,
    epochs: PythonObject, batch: PythonObject, shuffle: PythonObject, delta: PythonObject,
) raises -> PythonObject:
    owned_program_fit_admission(id, arrays, loss, epochs, batch, shuffle, delta)
    var retained = List[ProgramInput]()
    var name = String(py=loss)
    var kind = partition_loss_kind(name)
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
    unsafe_memcpy(dest=FP(unsafe_from_address=Int(py=out.__array_interface__["data"][0])),
           src=a.words.unsafe_ptr(), count=a.shape.count())
    return out


def partition_enter_py[E: Exec & Movable & Deinitable, KIND: Int](id: PythonObject, owner: PythonObject) raises -> PythonObject:
    owned_program_id_admission(id)
    partition_ready[E, KIND](Int(py=id))
    return owner
