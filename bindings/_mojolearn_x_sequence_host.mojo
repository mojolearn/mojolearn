# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU host binding of the sequence expansion lane. HOST ONLY: no
DeviceContext, no kernel launch. Every entry is `sequence/pyapi.mojo`'s over a
`HostExec`, the element bodies of `sequence/ops.mojo` looped in ascending
order, under the GPU binding's names and address contract, so the lane's
classes run unchanged on a CPU-only install (`_backend._HOST_MODULES`).

The sabotage arm (`x_sequence_host_sabotage`, -D MOJOLEARN_HOST_SABOTAGE=1)
runs the GEMM reduction k descending (`sequence/ops.mojo`)."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.exec import HostExec
from sequence.ops import SEQUENCE_HOST_SABOTAGE
from sequence.schedule import epoch_schedule_py
from sequence.prophet_prep import prophet_changepoints_py, prophet_days_py, prophet_features_py
from autograd.program_pyapi import ops_program_dynamic_py, ops_program_fit_py
from autograd.distributed_pyapi import (
    distributed_devices, distributed_open, distributed_load, distributed_fit_py,
    distributed_run_py, distributed_state_py, distributed_residency_py,
    distributed_save_py, distributed_close_py,
)
from autograd.partition_pyapi import (
    partition_open, partition_load, partition_fit_py, partition_run_py,
    partition_state_py, partition_residency_py, partition_save_py,
    partition_close_py, partition_profile_py,
)
from autograd.pyapi import model_fit_py, model_labels_check, model_labels_result, model_n_params_py, model_predict_py
from autograd.pyapi import model_gather_labels_py
from autograd.pyapi import model_import_params_py, ops_load_weights_py
from core.label_cast import host_unique_inverse_cast
from autograd.pyapi import (
    ops_open_py, ops_close_py, ops_upload_py, ops_download_py, ops_op_py, ops_scalar_py,
    ops_shape_py, ops_free_py, ops_reset_py, ops_metric_py, ops_backward_py, ops_grad_py,
    ops_profile_reset_py, ops_profile_report_py, ops_op_packed_py,
    ops_optimizer_py, ops_step_py, ops_scheduler_step_py, ops_learning_rate_py, ops_zero_grad_py,
    ops_optimizer_state_py, ops_load_optimizer_state_py,
    ops_backward_graph_py, ops_backward_seed_py, ops_grad_handle_py, ops_vjp_py, ops_hvp_py, ops_jvp_py,
    ops_capture_py, ops_program_info_py, ops_program_run_py, ops_program_free_py,
)
from autograd.ops_api import OPS_HOST
# the ONNX model reader and shape resolution (autograd/import_reader.mojo,
# import_resolve.mojo): host only, the same entries in both bindings
from autograd.import_reader import onnx_read_py, onnx_info_py, onnx_copy_blob_py, onnx_close_py
from autograd.import_resolve import onnx_resolve_py
# the lowering onto session operations and the GraphIR-from-words handle (autograd/import_lower.mojo)
from autograd.import_lower import onnx_import_py, onnx_from_words_py
from autograd.model_verify import verify_digest_py, verify_file_sha256_py, verify_fill_py, verify_model_info_py
from autograd.model_checkpoint import (
    model_checkpoint_info_py, model_load_checkpoint_py, model_save_checkpoint_py, ops_load_state_py,
    ops_save_state_py, ops_load_program_py, ops_save_program_py,
)
from sequence.pyapi import opt_step_py, rnn_fit_py, rnn_n_params_py, rnn_predict_py, stl_py, var_fit_py, var_forecast_py, mlp_fit_py, mlp_predict_py, adafactor_step_py, lamb_step_py, layer_norm_py, theta_py, croston_py, croston_forecast_py, ets_py, garch_py, prophet_fit_py, prophet_predict_py, moe_forward_py
from sequence.pyapi import moe_backward_py


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


def rnn_n_params_binding(ip: PythonObject) raises -> PythonObject:
    return rnn_n_params_py(ip)


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
    are (autograd/pyapi.mojo cast_in_of descriptor, one per row), the
    classes in the labels' own dtype into classes_addr (n slots), codes
    into codes_addr. Returns the class count."""
    var ck = model_labels_check(src, classes_addr, codes_addr)
    var k = host_unique_inverse_cast(ck[0], ck[1], Int(py=classes_addr), Int(py=codes_addr))
    return model_labels_result(k)


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


def ops_load_weights_binding(sid: PythonObject, recs: PythonObject, names: PythonObject) raises -> PythonObject:
    """Session.load_weights (autograd/weight_import.mojo)."""
    return ops_load_weights_py[HostExec, OPS_HOST](sid, recs, names)


# The operation session (autograd/ops_api.mojo): one HostExec and one tape kept
# between calls, tensors by handle.
def ops_open_binding(record: PythonObject) raises -> PythonObject:
    return ops_open_py[HostExec, OPS_HOST](HostExec(), record)


def ops_close_binding(sid: PythonObject) raises -> PythonObject:
    return ops_close_py[HostExec, OPS_HOST](sid)


def ops_upload_binding(sid: PythonObject, addr: PythonObject, ip: PythonObject) raises -> PythonObject:
    return ops_upload_py[HostExec, OPS_HOST](sid, addr, ip)


def ops_download_binding(sid: PythonObject, h: PythonObject, addr: PythonObject) raises -> PythonObject:
    return ops_download_py[HostExec, OPS_HOST](sid, h, addr)


def ops_op_binding(sid: PythonObject, code: PythonObject, hs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    return ops_op_py[HostExec, OPS_HOST](sid, code, hs, ip, fp)


def ops_op_packed_binding(sid: PythonObject, code: PythonObject, block: PythonObject) raises -> PythonObject:
    return ops_op_packed_py[HostExec, OPS_HOST](sid, code, block)



# captured programs (autograd/program.mojo; OpsSession.capture / run_program)
def ops_capture_binding(sid: PythonObject, inputs: PythonObject, outputs: PythonObject) raises -> PythonObject:
    return ops_capture_py[HostExec, OPS_HOST](sid, inputs, outputs)


def ops_program_info_binding(sid: PythonObject, pid: PythonObject) raises -> PythonObject:
    return ops_program_info_py[HostExec, OPS_HOST](sid, pid)



def ops_partitioned_open_binding(sid: PythonObject, pid: PythonObject, devices: PythonObject,
    stages: PythonObject, tensor_shards: PythonObject, microbatches: PythonObject, optimizer: PythonObject, parameters: PythonObject, seed: PythonObject,
) raises -> PythonObject:
    var ids = distributed_devices(devices)
    var executors = List[HostExec]()
    if len(ids) != 1 or ids[0] != 0:
        raise Error("CPU replay uses devices=(0,); logical shards retain the distributed reduction order")
    executors.append(HostExec())
    return partition_open[HostExec, OPS_HOST](sid, pid, ids, stages, tensor_shards, microbatches, optimizer, parameters, seed, executors^)


def ops_partitioned_load_binding(path: PythonObject, devices: PythonObject) raises -> PythonObject:
    var ids = distributed_devices(devices)
    var executors = List[HostExec]()
    if len(ids) != 1 or ids[0] != 0:
        raise Error("CPU replay uses devices=(0,); logical shards retain the distributed reduction order")
    executors.append(HostExec())
    return partition_load[HostExec, OPS_HOST](path, ids, executors^)


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


def ops_partitioned_close_binding(id: PythonObject) raises -> PythonObject:
    return partition_close_py[HostExec, OPS_HOST](id)


def ops_partitioned_profile_binding(id: PythonObject) raises -> PythonObject:
    return partition_profile_py[HostExec, OPS_HOST](id)


def ops_distributed_open_binding(sid: PythonObject, pid: PythonObject, devices: PythonObject,
    logical_shards: PythonObject, optimizer: PythonObject, parameters: PythonObject, seed: PythonObject,
) raises -> PythonObject:
    var ids = distributed_devices(devices)
    var executors = List[HostExec]()
    if len(ids) != 1 or ids[0] != 0:
        raise Error("CPU replay uses devices=(0,); logical shards retain the distributed reduction order")
    executors.append(HostExec())
    return distributed_open[HostExec, OPS_HOST](sid, pid, ids, logical_shards, optimizer, parameters, seed, executors^)


def ops_distributed_load_binding(path: PythonObject, devices: PythonObject) raises -> PythonObject:
    var ids = distributed_devices(devices)
    var executors = List[HostExec]()
    if len(ids) != 1 or ids[0] != 0:
        raise Error("CPU replay uses devices=(0,); logical shards retain the distributed reduction order")
    executors.append(HostExec())
    return distributed_load[HostExec, OPS_HOST](path, ids, executors^)


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


def ops_distributed_close_binding(id: PythonObject) raises -> PythonObject:
    return distributed_close_py[HostExec, OPS_HOST](id)


def ops_program_dynamic_binding(sid: PythonObject, pid: PythonObject, arrays: PythonObject) raises -> PythonObject:
    return ops_program_dynamic_py[HostExec, OPS_HOST](sid, pid, arrays)


def ops_program_fit_binding(sid: PythonObject, pid: PythonObject, arrays: PythonObject, targets: PythonObject,
                            parameters: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    return ops_program_fit_py[HostExec, OPS_HOST](sid, pid, arrays, targets, parameters, ip, fp)


def ops_program_run_binding(sid: PythonObject, pid: PythonObject, descs: PythonObject, addrs: PythonObject) raises -> PythonObject:
    return ops_program_run_py[HostExec, OPS_HOST](sid, pid, descs, addrs)


def ops_program_free_binding(sid: PythonObject, pid: PythonObject) raises -> PythonObject:
    return ops_program_free_py[HostExec, OPS_HOST](sid, pid)


def ops_save_program_binding(sid: PythonObject, pid: PythonObject, path: PythonObject) raises -> PythonObject:
    return ops_save_program_py[HostExec, OPS_HOST](sid, pid, path)


def ops_load_program_binding(sid: PythonObject, path: PythonObject) raises -> PythonObject:
    return ops_load_program_py[HostExec, OPS_HOST](sid, path)


def ops_scalar_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_scalar_py[HostExec, OPS_HOST](sid, h)


def ops_shape_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_shape_py[HostExec, OPS_HOST](sid, h)


def ops_free_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_free_py[HostExec, OPS_HOST](sid, h)


def ops_reset_binding(sid: PythonObject) raises -> PythonObject:
    return ops_reset_py[HostExec, OPS_HOST](sid)


def ops_metric_binding(sid: PythonObject, kind: PythonObject, hs: PythonObject) raises -> PythonObject:
    return ops_metric_py[HostExec, OPS_HOST](sid, kind, hs)


def ops_backward_binding(sid: PythonObject, h: PythonObject, accumulate: PythonObject) raises -> PythonObject:
    return ops_backward_py[HostExec, OPS_HOST](sid, h, accumulate)


def ops_grad_binding(sid: PythonObject, h: PythonObject, addr: PythonObject) raises -> PythonObject:
    return ops_grad_py[HostExec, OPS_HOST](sid, h, addr)


def ops_optimizer_binding(sid: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    return ops_optimizer_py[HostExec, OPS_HOST](sid, ip, fp)


def ops_step_binding(sid: PythonObject, opt: PythonObject, hs: PythonObject) raises -> PythonObject:
    return ops_step_py[HostExec, OPS_HOST](sid, opt, hs)


def ops_scheduler_step_binding(sid: PythonObject, opt: PythonObject) raises -> PythonObject:
    return ops_scheduler_step_py[HostExec, OPS_HOST](sid, opt)


def ops_learning_rate_binding(sid: PythonObject, opt: PythonObject) raises -> PythonObject:
    return ops_learning_rate_py[HostExec, OPS_HOST](sid, opt)


def ops_zero_grad_binding(sid: PythonObject) raises -> PythonObject:
    return ops_zero_grad_py[HostExec, OPS_HOST](sid)


def ops_optimizer_state_binding(sid: PythonObject, opt: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_optimizer_state_py[HostExec, OPS_HOST](sid, opt, h)


def ops_load_optimizer_state_binding(
    sid: PythonObject, opt: PythonObject, h: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    return ops_load_optimizer_state_py[HostExec, OPS_HOST](sid, opt, h, ip, fp)


# Higher-order gradients (lane ag-ho-session; autograd/ops_api.mojo).
def ops_backward_graph_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_backward_graph_py[HostExec, OPS_HOST](sid, h)


def ops_backward_seed_binding(
    sid: PythonObject, h: PythonObject, seed: PythonObject, create_graph: PythonObject,
) raises -> PythonObject:
    return ops_backward_seed_py[HostExec, OPS_HOST](sid, h, seed, create_graph)


def ops_grad_handle_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_grad_handle_py[HostExec, OPS_HOST](sid, h)


def ops_vjp_binding(
    sid: PythonObject, out_h: PythonObject, inputs: PythonObject, seed: PythonObject, create_graph: PythonObject,
) raises -> PythonObject:
    return ops_vjp_py[HostExec, OPS_HOST](sid, out_h, inputs, seed, create_graph)


def ops_hvp_binding(sid: PythonObject, loss: PythonObject, params: PythonObject, vs: PythonObject) raises -> PythonObject:
    return ops_hvp_py[HostExec, OPS_HOST](sid, loss, params, vs)


def ops_jvp_binding(sid: PythonObject, out_h: PythonObject, inputs: PythonObject, us: PythonObject) raises -> PythonObject:
    return ops_jvp_py[HostExec, OPS_HOST](sid, out_h, inputs, us)


def ops_profile_reset_binding() raises -> PythonObject:
    return ops_profile_reset_py()


def ops_profile_report_binding() raises -> PythonObject:
    return ops_profile_report_py()


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


def ops_save_state_binding(
    sid: PythonObject, path: PythonObject, params: PythonObject, optimizer: PythonObject, counters: PythonObject,
) raises -> PythonObject:
    return ops_save_state_py[HostExec, OPS_HOST](sid, path, params, optimizer, counters)


def ops_load_state_binding(
    sid: PythonObject, path: PythonObject, requires_grad: PythonObject, optimizer: PythonObject,
) raises -> PythonObject:
    return ops_load_state_py[HostExec, OPS_HOST](sid, path, requires_grad, optimizer)


def adafactor_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return adafactor_step_py(ex, addrs, ip, fp)


def lamb_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return lamb_step_py(ex, addrs, ip, fp)


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


def epoch_schedule_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    return epoch_schedule_py(addrs, ip)


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


def onnx_copy_blob_binding(h: PythonObject, addr: PythonObject) raises -> PythonObject:
    return onnx_copy_blob_py(h, addr)


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


def onnx_from_words_binding(
    words: PythonObject, strings: PythonObject, floats: PythonObject, addrs: PythonObject,
) raises -> PythonObject:
    """A graph handle from onnx_info's word layout (a torch.export GraphIR)."""
    return onnx_from_words_py(words, strings, floats, addrs)


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
        m.def_function[rnn_n_params_binding]("rnn_n_params")
        m.def_function[optimizer_step_binding]("optimizer_step")
        m.def_function[stl_binding]("stl")
        m.def_function[var_fit_binding]("var_fit")
        m.def_function[var_forecast_binding]("var_forecast")
        m.def_function[mlp_fit_binding]("mlp_fit")
        m.def_function[mlp_predict_binding]("mlp_predict")
        # the model-description trainer (autograd/model.mojo)
        m.def_function[model_fit_binding]("model_fit")
        m.def_function[model_predict_binding]("model_predict")
        m.def_function[model_n_params_binding]("model_n_params")
        m.def_function[model_labels_binding]("model_labels")
        m.def_function[model_gather_labels_binding]("model_gather_labels")
        m.def_function[model_import_params_binding]("model_import_params")
        m.def_function[ops_load_weights_binding]("ops_load_weights")
        # the operation session (autograd/ops_api.mojo)
        m.def_function[ops_open_binding]("ops_open")
        m.def_function[ops_close_binding]("ops_close")
        m.def_function[ops_upload_binding]("ops_upload")
        m.def_function[ops_download_binding]("ops_download")
        m.def_function[ops_op_binding]("ops_op")
        m.def_function[ops_op_packed_binding]("ops_op_packed")
        m.def_function[ops_scalar_binding]("ops_scalar")
        m.def_function[ops_shape_binding]("ops_shape")
        m.def_function[ops_free_binding]("ops_free")
        m.def_function[ops_reset_binding]("ops_reset")
        m.def_function[ops_metric_binding]("ops_metric")
        m.def_function[ops_backward_binding]("ops_backward")
        m.def_function[ops_grad_binding]("ops_grad")
        m.def_function[ops_optimizer_binding]("ops_optimizer")
        m.def_function[ops_step_binding]("ops_step")
        m.def_function[ops_scheduler_step_binding]("ops_scheduler_step")
        m.def_function[ops_learning_rate_binding]("ops_learning_rate")
        m.def_function[ops_zero_grad_binding]("ops_zero_grad")
        m.def_function[ops_optimizer_state_binding]("ops_optimizer_state")
        m.def_function[ops_load_optimizer_state_binding]("ops_load_optimizer_state")
        m.def_function[ops_backward_graph_binding]("ops_backward_graph")
        m.def_function[ops_backward_seed_binding]("ops_backward_seed")
        m.def_function[ops_grad_handle_binding]("ops_grad_handle")
        m.def_function[ops_vjp_binding]("ops_vjp")
        m.def_function[ops_hvp_binding]("ops_hvp")
        m.def_function[ops_jvp_binding]("ops_jvp")
        m.def_function[ops_capture_binding]("ops_capture")
        m.def_function[ops_program_info_binding]("ops_program_info")
        m.def_function[ops_program_run_binding]("ops_program_run")
        m.def_function[ops_program_dynamic_binding]("ops_program_dynamic")
        m.def_function[ops_program_fit_binding]("ops_program_fit")
        m.def_function[ops_partitioned_open_binding]("ops_partitioned_open")
        m.def_function[ops_partitioned_load_binding]("ops_partitioned_load")
        m.def_function[ops_partitioned_fit_binding]("ops_partitioned_fit")
        m.def_function[ops_partitioned_run_binding]("ops_partitioned_run")
        m.def_function[ops_partitioned_state_binding]("ops_partitioned_state")
        m.def_function[ops_partitioned_residency_binding]("ops_partitioned_residency")
        m.def_function[ops_partitioned_save_binding]("ops_partitioned_save")
        m.def_function[ops_partitioned_close_binding]("ops_partitioned_close")
        m.def_function[ops_partitioned_profile_binding]("ops_partitioned_profile")
        m.def_function[ops_distributed_open_binding]("ops_distributed_open")
        m.def_function[ops_distributed_load_binding]("ops_distributed_load")
        m.def_function[ops_distributed_fit_binding]("ops_distributed_fit")
        m.def_function[ops_distributed_run_binding]("ops_distributed_run")
        m.def_function[ops_distributed_state_binding]("ops_distributed_state")
        m.def_function[ops_distributed_residency_binding]("ops_distributed_residency")
        m.def_function[ops_distributed_save_binding]("ops_distributed_save")
        m.def_function[ops_distributed_close_binding]("ops_distributed_close")
        m.def_function[ops_program_free_binding]("ops_program_free")
        m.def_function[ops_save_program_binding]("ops_save_program")
        m.def_function[ops_load_program_binding]("ops_load_program")
        m.def_function[ops_profile_reset_binding]("ops_profile_reset")
        m.def_function[ops_profile_report_binding]("ops_profile_report")
        m.def_function[model_save_checkpoint_binding]("model_save_checkpoint")
        m.def_function[model_checkpoint_info_binding]("model_checkpoint_info")
        m.def_function[model_load_checkpoint_binding]("model_load_checkpoint")
        m.def_function[verify_digest_binding]("verify_digest")
        m.def_function[verify_file_sha256_binding]("verify_file_sha256")
        m.def_function[verify_model_info_binding]("verify_model_info")
        m.def_function[verify_fill_binding]("verify_fill")
        m.def_function[ops_save_state_binding]("ops_save_state")
        m.def_function[ops_load_state_binding]("ops_load_state")
        m.def_function[adafactor_step_binding]("adafactor_step")
        m.def_function[lamb_step_binding]("lamb_step")
        m.def_function[layer_norm_binding]("layer_norm")
        m.def_function[theta_binding]("theta")
        m.def_function[croston_binding]("croston")
        m.def_function[croston_forecast_binding]("croston_forecast")
        m.def_function[ets_binding]("ets")
        m.def_function[garch_binding]("garch")
        m.def_function[epoch_schedule_binding]("epoch_schedule")
        m.def_function[prophet_days_binding]("prophet_days")
        m.def_function[prophet_features_binding]("prophet_features")
        m.def_function[prophet_changepoints_binding]("prophet_changepoints")
        m.def_function[prophet_fit_binding]("prophet_fit")
        m.def_function[prophet_predict_binding]("prophet_predict")
        m.def_function[moe_forward_binding]("moe_forward")
        m.def_function[moe_backward_binding]("moe_backward")
        # the ONNX model reader (autograd/import_reader.mojo, import_resolve.mojo)
        m.def_function[onnx_read_binding]("onnx_read")
        m.def_function[onnx_info_binding]("onnx_info")
        m.def_function[onnx_copy_blob_binding]("onnx_copy_blob")
        m.def_function[onnx_resolve_binding]("onnx_resolve")
        m.def_function[onnx_close_binding]("onnx_close")
        m.def_function[onnx_import_binding]("onnx_import")
        m.def_function[onnx_from_words_binding]("onnx_from_words")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_sequence_host: ", e))
