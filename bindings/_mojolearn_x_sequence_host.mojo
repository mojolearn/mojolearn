# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU host binding of the sequence expansion lane. HOST ONLY: no
DeviceContext, no kernel launch. Every entry is `providers/python/sequence_pyapi.mojo`'s over a
`HostExec`, the element bodies of `sequence/ops.mojo` looped in ascending
order, under the GPU binding's names and address contract, so the lane's
classes run unchanged on a CPU-only install (`_backend._HOST_MODULES`).

The sabotage arm (`x_sequence_host_sabotage`, -D MOJOLEARN_HOST_SABOTAGE=1)
runs the GEMM reduction k descending (`sequence/ops.mojo`)."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr
from sequence.model_host_dispatch import NATIVE_MODEL_API
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from providers.python.autograd_python_errors_pyapi import graph_python_error
from providers.python.autograd_ops_frontend_pyapi import ops_session_call_py
from providers.python.autograd_session_frontend_pyapi import ops_session_initialize_py
from autograd.session_lifetime import SessionLifetime
from providers.python.autograd_session_lifetime_pyapi import ops_session_create_py

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.exec import HostExec
from providers.python.sequence_opt_public_host_pyapi import (
    seq_optimizer_init_host_py,
    seq_optimizer_step_host_py,
    seq_lamb_step_host_py,
    seq_adafactor_step_host_py,
    seq_optimizer_state_host_py,
    seq_optimizer_export_host_py,
    seq_optimizer_import_host_py,
    seq_optimizer_close_host_py,
)
from sequence.ops import SEQUENCE_HOST_SABOTAGE
from providers.python.sequence_prophet_prep_pyapi import prophet_changepoints_py, prophet_days_py, prophet_features_py
from providers.python.autograd_program_pyapi import ops_program_execute_py, ops_program_inspect_py, ops_program_fit_native_py, ops_imported_run_py
from providers.python.autograd_program_control_pyapi import ops_program_control_py
from providers.python.autograd_torch_program_pyapi import ops_torch_program_run_py
from providers.python.autograd_distributed_pyapi import (
    distributed_object,
    distributed_devices, distributed_open, distributed_load, distributed_fit_py,
    distributed_run_py, distributed_state_py, distributed_residency_py,
    distributed_save_py, distributed_close_py, distributed_enter_py,
)
from providers.python.autograd_partition_pyapi import (
    partition_open, partition_load, partition_fit_py, partition_run_py,
    partition_state_py, partition_residency_py, partition_save_py,
    partition_close_py, partition_enter_py, partition_profile_py,
)
from providers.python.autograd_program_control_grad_pyapi import ops_program_control_vjp_py
from providers.python.autograd_pyapi import model_fit_py, model_labels_check, model_labels_result, model_n_params_py, model_predict_py
from providers.python.autograd_pyapi import model_gather_labels_py
from providers.python.autograd_pyapi import model_fit_labels_check, model_fit_labels_py, model_predict_labels_py
from providers.python.autograd_pyapi import model_import_params_py
from core.label_cast import host_unique_inverse_cast
from providers.python.autograd_pyapi import ops_shape_py
from autograd.ops_api import OPS_HOST
# the ONNX model reader and shape resolution (autograd/import_reader.mojo,
# import_resolve.mojo): host only, the same entries in both bindings
from providers.python.autograd_import_reader_pyapi import onnx_read_py, onnx_info_py, onnx_buffers_py, onnx_close_py, onnx_native_api_py
from providers.python.autograd_import_resolve_pyapi import onnx_resolve_py
# the lowering onto session operations (autograd/import_lower.mojo)
from providers.python.autograd_import_frontend_pyapi import ops_import_model_c
from providers.python.autograd_graph_metadata_pyapi import graph_record_c
from providers.python.autograd_import_lower_pyapi import onnx_import_py, onnx_import_file_py
from providers.python.autograd_model_verify_pyapi import verify_digest_py, verify_file_sha256_py, verify_fill_py, verify_model_info_py
from providers.python.autograd_model_checkpoint_pyapi import model_checkpoint_info_py, model_load_checkpoint_py, model_save_checkpoint_py
from providers.python.sequence_pyapi import opt_step_py, rnn_fit_py, rnn_predict_py, stl_py, var_fit_py, var_forecast_py, mlp_fit_py, mlp_predict_py, layer_norm_py, theta_py, croston_py, croston_forecast_py, ets_py, garch_py, prophet_fit_py, prophet_predict_py, moe_forward_py
from providers.python.sequence_pyapi import moe_backward_py


def host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "x_sequence host: IDENTICAL only; bindings/build_host_family.sh passes"
        " -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "x_sequence host: compiles the CPU column only (-D MOJOLEARN_COLUMN_CPU)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def host_sabotage_binding() raises -> PythonObject:
    return PythonObject(SEQUENCE_HOST_SABOTAGE)


def rnn_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return rnn_fit_py(ex, addrs, ip, fp)


def rnn_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return rnn_predict_py(ex, addrs, ip)


def optimizer_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return opt_step_py(ex, addrs, ip, fp)


def stl_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return stl_py(ex, addrs, ip)


def var_fit_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return var_fit_py(ex, addrs, ip)


def var_forecast_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return var_forecast_py(ex, addrs, ip)


def mlp_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return mlp_fit_py(ex, addrs, ip, fp)


def mlp_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return mlp_predict_py(ex, addrs, ip)


def model_native_api_binding() raises -> PythonObject:
    var api = NATIVE_MODEL_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def model_fit_binding(addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return model_fit_py(ex, addrs, prog, ip, fp)


def model_predict_binding(addrs: PythonObject, prog: PythonObject, ptab: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return model_predict_py(ex, addrs, prog, ptab, ip)


def model_labels_binding(src: PythonObject, classes_addr: PythonObject, codes_addr: PythonObject) raises -> PythonObject:
    """Sorted distinct labels and one int32 code per row for Sequential's
    classification losses, by the host twin (`core/label_cast.mojo`, the
    same statements as the device path): `src` the user's labels as they
    are (providers/python/autograd_pyapi.mojo cast_in_of descriptor, one per row), the
    classes in the labels' own dtype into classes_addr (n slots), codes
    into codes_addr. Returns the class count."""
    var ck = model_labels_check(src, classes_addr, codes_addr)
    var k = host_unique_inverse_cast(ck[0], ck[1], Int(py=classes_addr), Int(py=codes_addr))
    return model_labels_result(k)


def model_predict_labels_binding(addrs: PythonObject, prog: PythonObject, ptab: PythonObject, ip: PythonObject) raises -> PythonObject:
    """Sequential.predict of a classifier: the codes and the label gather
    stay on the executor (providers/python/autograd_pyapi.mojo model_predict_labels_py)."""
    var ex = HostExec()
    return model_predict_labels_py(ex, addrs, prog, ptab, ip)


def model_fit_labels_binding(
    addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject, lab: PythonObject,
) raises -> PythonObject:
    """Sequential.fit of a classifier in one call: the label sort writes the
    classes to the caller's buffer and the codes into an executor buffer,
    and the fit reads them there (providers/python/autograd_pyapi.mojo model_fit_labels_py).
    Returns [class count, epochs run]."""
    var la = model_fit_labels_check(lab)
    var ex = HostExec()
    var codes = ex.alloc(la[0].rows)
    var k = host_unique_inverse_cast(la[0], la[1], la[2], Int(codes))
    return model_fit_labels_py(ex, k, codes, la[3], la[4], addrs, prog, ip, fp)


def model_gather_labels_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return model_gather_labels_py(ex, addrs, ip)


def model_n_params_binding(prog: PythonObject, ptab: PythonObject) raises -> PythonObject:
    return model_n_params_py(prog, ptab)


def model_import_params_binding(
    prog: PythonObject, ptab: PythonObject, recs: PythonObject, names: PythonObject, params_addr: PythonObject,
) raises -> PythonObject:
    """Sequential.load_state_dict (autograd/weight_import.mojo)."""
    var ex = HostExec()
    return model_import_params_py(ex, prog, ptab, recs, names, params_addr)


# TOMBSTONE: ops_* per-op session entries + lamb_step (dead native entries: no live caller; ops.Session runs ops_session_call) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_sequence.patch


# The operation session (autograd/ops_api.mojo): one HostExec and one tape kept
# between calls, tensors by handle.
def ops_session_create_binding(record: PythonObject) raises -> PythonObject:
    return ops_session_create_py[HostExec, OPS_HOST](HostExec(), record)


def ops_partitioned_open_binding(sid: PythonObject, pid: PythonObject, devices: PythonObject,
    options: PythonObject, cls: PythonObject, binding: PythonObject,
) raises -> PythonObject:
    var ids = distributed_devices(devices)
    var executors = List[HostExec]()
    if len(ids) != 1 or ids[0] != 0:
        raise Error("CPU replay uses devices=(0,); logical shards retain the distributed reduction order")
    executors.append(HostExec())
    var handle = partition_open[HostExec, OPS_HOST](sid, pid, ids, options[0], options[1], options[2], options[3], options[4], options[5], executors^)
    return distributed_object(cls, binding, handle)


def ops_partitioned_load_binding(path: PythonObject, devices: PythonObject, cls: PythonObject, binding: PythonObject) raises -> PythonObject:
    var ids = distributed_devices(devices)
    var executors = List[HostExec]()
    if len(ids) != 1 or ids[0] != 0:
        raise Error("CPU replay uses devices=(0,); logical shards retain the distributed reduction order")
    executors.append(HostExec())
    var handle = partition_load[HostExec, OPS_HOST](path, ids, executors^)
    return distributed_object(cls, binding, handle)


def ops_partitioned_fit_binding(id: PythonObject, arrays: PythonObject, targets: PythonObject,
    loss: PythonObject, epochs: PythonObject, batch: PythonObject, shuffle: PythonObject, delta: PythonObject,
) raises -> PythonObject:
    return partition_fit_py[HostExec, OPS_HOST](id, arrays, targets, loss, epochs, batch, shuffle, delta)


def ops_partitioned_run_binding(id: PythonObject, arrays: PythonObject) raises -> PythonObject:
    return partition_run_py[HostExec, OPS_HOST](id, arrays)


def ops_partitioned_state_binding(id: PythonObject) raises -> PythonObject:
    return partition_state_py[HostExec, OPS_HOST](id)


def ops_partitioned_residency_binding(id: PythonObject) raises -> PythonObject:
    return partition_residency_py[HostExec, OPS_HOST](id)


def ops_partitioned_save_binding(id: PythonObject, path: PythonObject) raises -> PythonObject:
    return partition_save_py[HostExec, OPS_HOST](id, path)


def ops_partitioned_enter_binding(id: PythonObject, owner: PythonObject) raises -> PythonObject:
    return partition_enter_py[HostExec, OPS_HOST](id, owner)


def ops_partitioned_exit_binding(id: PythonObject) raises -> PythonObject:
    # __exit__ historically discards close's integer sentinel and returns None.
    _ = partition_close_py[HostExec, OPS_HOST](id)
    return PythonObject(None)


def ops_partitioned_close_binding(id: PythonObject) raises -> PythonObject:
    return partition_close_py[HostExec, OPS_HOST](id)


def ops_partitioned_profile_binding(id: PythonObject) raises -> PythonObject:
    return partition_profile_py[HostExec, OPS_HOST](id)


def ops_distributed_open_binding(sid: PythonObject, pid: PythonObject, devices: PythonObject,
    options: PythonObject, cls: PythonObject, binding: PythonObject,
) raises -> PythonObject:
    var ids = distributed_devices(devices)
    var executors = List[HostExec]()
    if len(ids) != 1 or ids[0] != 0:
        raise Error("CPU replay uses devices=(0,); logical shards retain the distributed reduction order")
    executors.append(HostExec())
    var handle = distributed_open[HostExec, OPS_HOST](sid, pid, ids, options[0], options[1], options[2], options[3], executors^)
    return distributed_object(cls, binding, handle)


def ops_distributed_load_binding(path: PythonObject, devices: PythonObject, cls: PythonObject, binding: PythonObject) raises -> PythonObject:
    var ids = distributed_devices(devices)
    var executors = List[HostExec]()
    if len(ids) != 1 or ids[0] != 0:
        raise Error("CPU replay uses devices=(0,); logical shards retain the distributed reduction order")
    executors.append(HostExec())
    var handle = distributed_load[HostExec, OPS_HOST](path, ids, executors^)
    return distributed_object(cls, binding, handle)


def ops_distributed_fit_binding(id: PythonObject, arrays: PythonObject, targets: PythonObject,
    loss: PythonObject, epochs: PythonObject, batch: PythonObject, shuffle: PythonObject, delta: PythonObject,
) raises -> PythonObject:
    return distributed_fit_py[HostExec, OPS_HOST](id, arrays, targets, loss, epochs, batch, shuffle, delta)


def ops_distributed_run_binding(id: PythonObject, arrays: PythonObject) raises -> PythonObject:
    return distributed_run_py[HostExec, OPS_HOST](id, arrays)


def ops_distributed_state_binding(id: PythonObject) raises -> PythonObject:
    return distributed_state_py[HostExec, OPS_HOST](id)


def ops_distributed_residency_binding(id: PythonObject) raises -> PythonObject:
    return distributed_residency_py[HostExec, OPS_HOST](id)


def ops_distributed_save_binding(id: PythonObject, path: PythonObject) raises -> PythonObject:
    return distributed_save_py[HostExec, OPS_HOST](id, path)


def ops_distributed_enter_binding(id: PythonObject, owner: PythonObject) raises -> PythonObject:
    return distributed_enter_py[HostExec, OPS_HOST](id, owner)


def ops_distributed_exit_binding(id: PythonObject) raises -> PythonObject:
    # __exit__ historically discards close's integer sentinel and returns None.
    _ = distributed_close_py[HostExec, OPS_HOST](id)
    return PythonObject(None)


def ops_distributed_close_binding(id: PythonObject) raises -> PythonObject:
    return distributed_close_py[HostExec, OPS_HOST](id)


def ops_session_call_c_binding(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 3:
            raise Error("TypeError: ops_session_call expects three arguments")
        var result = ops_session_call_py[HostExec, OPS_HOST](args[0], args[1], args[2])
        return result.steal_data()
    except error:
        return graph_python_error(error)


def ops_program_execute_c_binding(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 7:
            raise Error("TypeError: ops_program_execute expects 7 arguments")
        var result = ops_program_execute_binding(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        return result.steal_data()
    except error:
        return graph_python_error(error)


def ops_torch_program_run_c_binding(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 4:
            raise Error("TypeError: ops_torch_program_run expects 4 arguments")
        var result = ops_torch_program_run_py(args[0], args[1], args[2], Bool(py=args[3]))
        return result.steal_data()
    except error:
        return graph_python_error(error)


def ops_imported_run_c_binding(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    # mojolearn.importers.ImportedProgram.run: (program, arrays, named).
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 3:
            raise Error("TypeError: ops_imported_run expects 3 arguments")
        var result = ops_imported_run_py(args[0], args[1], args[2])
        return result.steal_data()
    except error:
        return graph_python_error(error)


def ops_program_inspect_c_binding(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 3:
            raise Error("TypeError: ops_program_inspect expects 3 arguments")
        var result = ops_program_inspect_binding(args[0], args[1], args[2])
        return result.steal_data()
    except error:
        return graph_python_error(error)


def ops_session_initialize_c_binding(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 6:
            raise Error("TypeError: ops_session_initialize expects 6 arguments")
        return ops_session_initialize_py(args[0], args[1], args[2], args[3], args[4], args[5]).steal_data()
    except error:
        return graph_python_error(error)


def ops_program_control_c_binding(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 6:
            raise Error("TypeError: ops_program_control expects 6 arguments")
        var result = ops_program_control_binding(args[0], args[1], args[2], args[3], args[4], args[5])
        return result.steal_data()
    except error:
        return graph_python_error(error)


def ops_program_fit_native_c_binding(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 5:
            raise Error("TypeError: ops_program_fit_native expects 5 arguments")
        var result = ops_program_fit_native_binding(args[0], args[1], args[2], args[3], args[4])
        return result.steal_data()
    except error:
        return graph_python_error(error)


def ops_program_execute_binding(sid: PythonObject, pid: PythonObject, owner_sid: PythonObject, arrays: PythonObject,
                                dynamic_shapes: PythonObject, optimize: PythonObject, profile: PythonObject) raises -> PythonObject:
    return ops_program_execute_py[HostExec, OPS_HOST](sid, pid, owner_sid, arrays, dynamic_shapes, optimize, profile)


def ops_program_inspect_binding(sid: PythonObject, pid: PythonObject, optimize: PythonObject) raises -> PythonObject:
    return ops_program_inspect_py[HostExec, OPS_HOST](sid, pid, optimize)


def ops_program_control_binding(sid: PythonObject, pid: PythonObject, programs: PythonObject,
                                arrays: PythonObject, limit: PythonObject, optimize: PythonObject) raises -> PythonObject:
    if len(programs) == 6:
        return ops_program_control_vjp_py[HostExec, OPS_HOST](sid, pid, programs, arrays, limit)
    return ops_program_control_py[HostExec, OPS_HOST](sid, pid, programs[1], programs[3],
                                               programs[0], programs[2], arrays, limit, optimize)


def ops_program_fit_native_binding(sid: PythonObject, pid: PythonObject, arrays: PythonObject,
                                   targets: PythonObject, options: PythonObject) raises -> PythonObject:
    return ops_program_fit_native_py[HostExec, OPS_HOST](sid, pid, arrays, targets, options[0],
                                                  options[1], options[2], options[3], options[4],
                                                  options[5], options[6], options[7])


def ops_shape_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_shape_py[HostExec, OPS_HOST](sid, h)


# Checkpoints (autograd/model_checkpoint.mojo, training/checkpoint.mojo's
# container): host files, the same bytes from every binding.
def model_save_checkpoint_binding(
    path: PythonObject, addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject,
    classes: PythonObject,
) raises -> PythonObject:
    return model_save_checkpoint_py(path, addrs, prog, ip, fp, classes)


def model_checkpoint_info_binding(path: PythonObject) raises -> PythonObject:
    return model_checkpoint_info_py(path)


# Model verification reports (autograd/model_verify.mojo): host digests over
# files and returned arrays, the same numbers from every binding.
def verify_digest_binding(addr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    return verify_digest_py(addr, nbytes)


def verify_file_sha256_binding(path: PythonObject) raises -> PythonObject:
    return verify_file_sha256_py(path)


def verify_model_info_binding(path: PythonObject) raises -> PythonObject:
    return verify_model_info_py(path)


def verify_fill_binding(addr: PythonObject, n: PythonObject, seed: PythonObject, stream: PythonObject) raises -> PythonObject:
    return verify_fill_py(addr, n, seed, stream)


def model_load_checkpoint_binding(path: PythonObject, addrs: PythonObject) raises -> PythonObject:
    return model_load_checkpoint_py(path, addrs)


def layer_norm_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return layer_norm_py(ex, addrs, ip, fp)


def theta_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return theta_py(ex, addrs, ip, fp)


def croston_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return croston_py(ex, addrs, ip)


def croston_forecast_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return croston_forecast_py(ex, addrs, ip)


def ets_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return ets_py(ex, addrs, ip, fp)


def garch_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return garch_py(ex, addrs, ip)


def prophet_days_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    return prophet_days_py(addrs, ip)


def prophet_features_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return prophet_features_py(ex, addrs, ip, fp)


def prophet_changepoints_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    return prophet_changepoints_py(addrs, ip, fp)


def prophet_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return prophet_fit_py(ex, addrs, ip, fp)


def prophet_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return prophet_predict_py(ex, addrs, ip)


def moe_forward_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return moe_forward_py(ex, addrs, ip)


def moe_backward_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return moe_backward_py(ex, addrs, ip)


def onnx_read_binding(path: PythonObject) raises -> PythonObject:
    """importers.onnx.read: read and check the model file; returns its handle."""
    return onnx_read_py(path)


def onnx_info_binding(h: PythonObject) raises -> PythonObject:
    return onnx_info_py(h)


def onnx_buffers_binding(h: PythonObject) raises -> PythonObject:
    """The address of each buffer graph h holds (the model file, its own bytes, each external data file)."""
    return onnx_buffers_py(h)


def onnx_resolve_binding(h: PythonObject, names: PythonObject, flat: PythonObject) raises -> PythonObject:
    """importers._graph.resolve of a graph read by onnx_read."""
    return onnx_resolve_py(h, names, flat)


def onnx_close_binding(h: PythonObject) raises -> PythonObject:
    return onnx_close_py(h)


def onnx_import_binding(
    h: PythonObject, sid: PythonObject, names: PythonObject, flat: PythonObject, addrs: PythonObject,
    source: PythonObject,
) raises -> PythonObject:
    """importers.import_model: resolve, refuse, lower and capture graph h in session sid."""
    return onnx_import_py[HostExec, OPS_HOST](h, sid, names, flat, addrs, source)


def onnx_import_file_binding(
    path: PythonObject, sid: PythonObject, names: PythonObject, flat: PythonObject, source: PythonObject,
) raises -> PythonObject:
    """importers.import_model of an ONNX file: read, resolve, refuse, lower and capture in session sid."""
    return onnx_import_file_py[HostExec, OPS_HOST](path, sid, names, flat, source)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_sequence_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_sequence_host() -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_sequence_host")
        m.def_function[host_numeric_mode_binding]("x_sequence_host_numeric_mode")
        m.def_function[host_vendor_binding]("x_sequence_host_vendor")
        m.def_function[host_column_binding]("x_sequence_host_column")
        m.def_function[host_sabotage_binding]("x_sequence_host_sabotage")
        m.def_function[host_numeric_mode_binding]("x_sequence_numeric_mode")
        m.def_function[host_vendor_binding]("x_sequence_vendor")
        m.def_function[rnn_fit_binding]("rnn_fit")
        m.def_function[rnn_predict_binding]("rnn_predict")
        m.def_function[optimizer_step_binding]("optimizer_step")
        # the public optimizers' one native call per method (sequence/opt_public*.mojo)
        m.def_function[seq_optimizer_init_host_py]("seq_optimizer_init")
        m.def_function[seq_optimizer_step_host_py]("seq_optimizer_step")
        m.def_function[seq_lamb_step_host_py]("seq_lamb_step")
        m.def_function[seq_adafactor_step_host_py]("seq_adafactor_step")
        m.def_function[seq_optimizer_state_host_py]("seq_optimizer_state")
        m.def_function[seq_optimizer_export_host_py]("seq_optimizer_export")
        m.def_function[seq_optimizer_import_host_py]("seq_optimizer_import")
        m.def_function[seq_optimizer_close_host_py]("seq_optimizer_close")
        m.def_function[stl_binding]("stl")
        m.def_function[var_fit_binding]("var_fit")
        m.def_function[var_forecast_binding]("var_forecast")
        m.def_function[mlp_fit_binding]("mlp_fit")
        m.def_function[mlp_predict_binding]("mlp_predict")
        # the model-description trainer (autograd/model.mojo)
        m.def_function[model_native_api_binding]("model_native_api_v1")
        m.def_function[model_fit_binding]("model_fit")
        m.def_function[model_predict_binding]("model_predict")
        m.def_function[model_n_params_binding]("model_n_params")
        m.def_function[model_labels_binding]("model_labels")
        m.def_function[model_gather_labels_binding]("model_gather_labels")
        m.def_function[model_predict_labels_binding]("model_predict_labels")
        m.def_function[model_fit_labels_binding]("model_fit_labels")
        m.def_function[model_import_params_binding]("model_import_params")
        # TOMBSTONE: ops_* per-op session entries + lamb_step (dead native entries: no live caller; ops.Session runs ops_session_call) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_sequence.patch
        # the operation session (autograd/ops_api.mojo)
        _ = m.add_type[SessionLifetime[HostExec, OPS_HOST]]("_OpsSessionLifetime")
        m.def_function[ops_session_create_binding]("ops_session_create")
        # TOMBSTONE: ops_* per-op session entries + lamb_step (dead native entries: no live caller; ops.Session runs ops_session_call) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_sequence.patch
        m.def_function[ops_shape_binding]("ops_shape")
        # TOMBSTONE: ops_* per-op session entries + lamb_step (dead native entries: no live caller; ops.Session runs ops_session_call) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_sequence.patch
        m.def_py_c_function(ops_session_call_c_binding, "ops_session_call")
        m.def_py_c_function(ops_session_initialize_c_binding, "ops_session_initialize")
        m.def_py_c_function(ops_program_execute_c_binding, "ops_program_execute")
        m.def_py_c_function(ops_program_inspect_c_binding, "ops_program_inspect")
        m.def_py_c_function(ops_imported_run_c_binding, "ops_imported_run")
        m.def_py_c_function(ops_torch_program_run_c_binding, "ops_torch_program_run")
        m.def_py_c_function(ops_program_control_c_binding, "ops_program_control")
        # TOMBSTONE: ops_* per-op session entries + lamb_step (dead native entries: no live caller; ops.Session runs ops_session_call) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_sequence.patch
        m.def_function[ops_partitioned_open_binding]("ops_partitioned_open")
        m.def_function[ops_partitioned_load_binding]("ops_partitioned_load")
        m.def_function[ops_partitioned_fit_binding]("ops_partitioned_fit")
        m.def_function[ops_partitioned_run_binding]("ops_partitioned_run")
        m.def_function[ops_partitioned_state_binding]("ops_partitioned_state")
        m.def_function[ops_partitioned_residency_binding]("ops_partitioned_residency")
        m.def_function[ops_partitioned_save_binding]("ops_partitioned_save")
        m.def_function[ops_partitioned_close_binding]("ops_partitioned_close")
        m.def_function[ops_partitioned_enter_binding]("ops_partitioned_enter")
        m.def_function[ops_partitioned_exit_binding]("ops_partitioned_exit")
        m.def_function[ops_partitioned_profile_binding]("ops_partitioned_profile")
        m.def_function[ops_distributed_open_binding]("ops_distributed_open")
        m.def_function[ops_distributed_load_binding]("ops_distributed_load")
        m.def_function[ops_distributed_fit_binding]("ops_distributed_fit")
        m.def_function[ops_distributed_run_binding]("ops_distributed_run")
        m.def_function[ops_distributed_state_binding]("ops_distributed_state")
        m.def_function[ops_distributed_residency_binding]("ops_distributed_residency")
        m.def_function[ops_distributed_save_binding]("ops_distributed_save")
        m.def_function[ops_distributed_close_binding]("ops_distributed_close")
        m.def_function[ops_distributed_enter_binding]("ops_distributed_enter")
        m.def_function[ops_distributed_exit_binding]("ops_distributed_exit")
        m.def_py_c_function(ops_program_fit_native_c_binding, "ops_program_fit_native")
        # TOMBSTONE: ops_* per-op session entries + lamb_step (dead native entries: no live caller; ops.Session runs ops_session_call) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_sequence.patch
        m.def_function[model_save_checkpoint_binding]("model_save_checkpoint")
        m.def_function[model_checkpoint_info_binding]("model_checkpoint_info")
        m.def_function[model_load_checkpoint_binding]("model_load_checkpoint")
        m.def_function[verify_digest_binding]("verify_digest")
        m.def_function[verify_file_sha256_binding]("verify_file_sha256")
        m.def_function[verify_model_info_binding]("verify_model_info")
        m.def_function[verify_fill_binding]("verify_fill")
        # TOMBSTONE: ops_* per-op session entries + lamb_step (dead native entries: no live caller; ops.Session runs ops_session_call) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_sequence.patch
        m.def_function[layer_norm_binding]("layer_norm")
        m.def_function[theta_binding]("theta")
        m.def_function[croston_binding]("croston")
        m.def_function[croston_forecast_binding]("croston_forecast")
        m.def_function[ets_binding]("ets")
        m.def_function[garch_binding]("garch")
        m.def_function[prophet_days_binding]("prophet_days")
        m.def_function[prophet_features_binding]("prophet_features")
        m.def_function[prophet_changepoints_binding]("prophet_changepoints")
        m.def_function[prophet_fit_binding]("prophet_fit")
        m.def_function[prophet_predict_binding]("prophet_predict")
        m.def_function[moe_forward_binding]("moe_forward")
        m.def_function[moe_backward_binding]("moe_backward")
        # the ONNX model reader (autograd/import_reader.mojo, import_resolve.mojo)
        m.def_function[onnx_native_api_py]("onnx_native_api_v1")
        m.def_function[onnx_read_binding]("onnx_read")
        m.def_function[onnx_info_binding]("onnx_info")
        m.def_function[onnx_buffers_binding]("onnx_buffers")
        m.def_function[onnx_resolve_binding]("onnx_resolve")
        m.def_function[onnx_close_binding]("onnx_close")
        m.def_function[onnx_import_binding]("onnx_import")
        m.def_function[onnx_import_file_binding]("onnx_import_file")
        m.def_py_c_function(ops_import_model_c, "ops_import_model")
        m.def_py_c_function(graph_record_c, "ops_graph_record")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_sequence_host: ", e))
