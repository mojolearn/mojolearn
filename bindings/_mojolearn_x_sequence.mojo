# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU binding of the sequence expansion lane (algorithm
expansion lane 6). Every entry is `sequence/pyapi.mojo`'s, run on a `DeviceExec`; the
CPU host binding `bindings/_mojolearn_x_sequence_host.mojo` exports the same
names over a `HostExec`, the same element bodies (`sequence/ops.mojo`)."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from sequence.exec_device import DeviceExec
from sequence.fit_team_py import garch_team_py, prophet_fit_team_py
from sequence.ets_team import ETS_TEAM, IDN_ETS_TEAM
from sequence.ets_team_py import ets_team_applies, ets_team_py
from sequence.pyapi import opt_step_py, rnn_fit_py, rnn_n_params_py, rnn_predict_py, stl_py, var_fit_py, var_forecast_py, mlp_fit_py, mlp_predict_py, adafactor_step_py, lamb_step_py, layer_norm_py, theta_py, croston_py, croston_forecast_py, ets_py, prophet_predict_py, moe_forward_py
from sequence.opt_resident import AF_RESIDENT, adafactor_resident_open_py, adafactor_resident_step_py, lamb_resident_open_py, lamb_resident_step_py, opt_resident_close_py, opt_resident_move_py, opt_resident_open_py, opt_resident_step_py
from autograd.pyapi import model_fit_py, model_labels_check, model_labels_result, model_n_params_py, model_predict_py
from autograd.pyapi import model_gather_labels_py
from autograd.pyapi import model_import_params_py, ops_load_weights_py
from core.label_cast_device import device_unique_inverse_cast
from sequence.exec_device import sequence_ctx
from autograd.pyapi import (
    ops_open_py, ops_close_py, ops_upload_py, ops_download_py, ops_op_py, ops_scalar_py,
    ops_shape_py, ops_free_py, ops_reset_py, ops_metric_py, ops_backward_py, ops_grad_py,
    ops_profile_reset_py, ops_profile_report_py, ops_op_packed_py,
    ops_optimizer_py, ops_step_py, ops_scheduler_step_py, ops_learning_rate_py, ops_zero_grad_py,
    ops_optimizer_state_py, ops_load_optimizer_state_py,
    ops_backward_graph_py, ops_backward_seed_py, ops_grad_handle_py, ops_vjp_py, ops_hvp_py, ops_jvp_py,
)
from autograd.ops_api import OPS_DEVICE
from autograd.model_checkpoint import (
    model_checkpoint_info_py, model_load_checkpoint_py, model_save_checkpoint_py, ops_load_state_py,
    ops_save_state_py,
)
from sequence.pyapi import ival, _getenv_seq, moe_forward_check, moe_forward_run, fptr
from sequence.pyapi import moe_backward_py, moe_backward_run
from sequence.opt_resident import adafactor_resident_step_dev_py, lamb_resident_step_dev_py, opt_resident_step_dev_py
from sequence.seq_tensor import (
    seq_tensor_alloc_py, seq_tensor_copy_py, seq_tensor_download_py, seq_tensor_free_py, seq_tensor_upload_py,
)
from sequence.layernorm_dev import layer_norm_dev_py
from sequence.sched_table import sched_exp_block_py
from checks.numerics import NUMERIC_FAST
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

#: MOJOLEARN_SCHED_FAST_TABLE (FAST + Apple; sequence/sched_table.mojo):
#: registers `sched_exp_block`, which ExponentialLR uses when present.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-05,
#: rab13-schedtable): lr-exponential 231.8 -> 19.2 ms, digest same. KEEP: the
#: FAST + Apple default; rollback -D MOJOLEARN_SCHED_FAST_TABLE_OFF.
comptime SCHED_FAST_TABLE = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                             and not is_defined["MOJOLEARN_SCHED_FAST_TABLE_OFF"]())
from sequence.prophet_prep import prophet_changepoints_py, prophet_days_py, prophet_features_py
from sequence.moe_weights import moe_weights_put, moe_weights_ptrs, moe_weights_free
from sequence.moe_dev import moe_forward_dev_py

#: lane neural-io-2: MoEBlock's forward on resident tensors (sequence/moe_dev.mojo):
#: x, y and the routing outputs stay on the device (2 x 4 T D bytes over PCIe and four
#: host zero-filled outputs per call removed; same launches, no bit moves). Default on;
#: -D MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF leaves the entry out, so MoEBlock takes the
#: host-array entry again (the grid's before arm).
comptime SEQ_MOE_DEVICE_IO = not is_defined["MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF"]()


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


def rnn_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return rnn_fit_py(ex, addrs, ip, fp)


def rnn_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return rnn_predict_py(ex, addrs, ip)


def rnn_n_params_binding(ip: PythonObject) raises -> PythonObject:
    return rnn_n_params_py(ip)


def optimizer_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return opt_step_py(ex, addrs, ip, fp)


def optimizer_resident_open_binding(ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    return opt_resident_open_py(ip, fp)


def lamb_resident_open_binding(ip: PythonObject) raises -> PythonObject:
    return lamb_resident_open_py(ip)


def optimizer_resident_close_binding(handle: PythonObject) raises -> PythonObject:
    return opt_resident_close_py(handle)


def optimizer_resident_move_binding(handle: PythonObject, slot: PythonObject, addr: PythonObject,
                                    up: PythonObject) raises -> PythonObject:
    return opt_resident_move_py(handle, slot, addr, up)


def optimizer_resident_step_binding(handle: PythonObject, addrs: PythonObject, ip: PythonObject,
                                    fp: PythonObject) raises -> PythonObject:
    return opt_resident_step_py(handle, addrs, ip, fp)


def lamb_resident_step_binding(handle: PythonObject, addrs: PythonObject, ip: PythonObject,
                               fp: PythonObject) raises -> PythonObject:
    return lamb_resident_step_py(handle, addrs, ip, fp)


def adafactor_resident_open_binding(ip: PythonObject) raises -> PythonObject:
    return adafactor_resident_open_py(ip)


def adafactor_resident_step_binding(handle: PythonObject, addrs: PythonObject, ip: PythonObject,
                                    fp: PythonObject) raises -> PythonObject:
    return adafactor_resident_step_py(handle, addrs, ip, fp)


# lane gap-neural-io (2026-10-08): resident sequence tensors and the device
# forms of the optimizer steps and LayerNorm (sequence/seq_tensor.mojo)
def seq_tensor_alloc_binding(n: PythonObject, zero: PythonObject) raises -> PythonObject:
    return seq_tensor_alloc_py(n, zero)
def seq_tensor_free_binding(h: PythonObject) raises -> PythonObject:
    return seq_tensor_free_py(h)
def seq_tensor_upload_binding(h: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    return seq_tensor_upload_py(h, addr, n)
def seq_tensor_download_binding(h: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    return seq_tensor_download_py(h, addr, n)
def seq_tensor_copy_binding(dst: PythonObject, src: PythonObject, n: PythonObject) raises -> PythonObject:
    return seq_tensor_copy_py(dst, src, n)
def optimizer_resident_step_dev_binding(handle: PythonObject, handles: PythonObject, ip: PythonObject,
                                        fp: PythonObject) raises -> PythonObject:
    return opt_resident_step_dev_py(handle, handles, ip, fp)
def lamb_resident_step_dev_binding(handle: PythonObject, handles: PythonObject, ip: PythonObject,
                                   fp: PythonObject) raises -> PythonObject:
    return lamb_resident_step_dev_py(handle, handles, ip, fp)
def adafactor_resident_step_dev_binding(handle: PythonObject, handles: PythonObject, ip: PythonObject,
                                        fp: PythonObject) raises -> PythonObject:
    return adafactor_resident_step_dev_py(handle, handles, ip, fp)
def layer_norm_dev_binding(handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                           fp: PythonObject) raises -> PythonObject:
    return layer_norm_dev_py(handles, addrs, ip, fp)
def stl_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return stl_py(ex, addrs, ip)


def sched_exp_block_binding(addrs: PythonObject, fp: PythonObject, ip: PythonObject) raises -> PythonObject:
    return sched_exp_block_py(addrs, fp, ip)


def var_fit_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_VAR_NODRAIN (DROPPED-slower, var taxi-hourly +10.6%) deleted 2026-10-09 on
    # lane/owed-deletions-D2; code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_VAR_NODRAIN.patch
    var r = var_fit_py(ex, addrs, ip)
    return r


def var_forecast_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    var r = var_forecast_py(ex, addrs, ip)
    return r


def mlp_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return mlp_fit_py(ex, addrs, ip, fp)


def mlp_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return mlp_predict_py(ex, addrs, ip)


def model_fit_binding(addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return model_fit_py(ex, addrs, prog, ip, fp)


def model_predict_binding(addrs: PythonObject, prog: PythonObject, ptab: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return model_predict_py(ex, addrs, prog, ptab, ip)


def model_labels_binding(src: PythonObject, classes_addr: PythonObject, codes_addr: PythonObject) raises -> PythonObject:
    """Sorted distinct labels and one int32 code per row for Sequential's
    classification losses, on the device (`core/label_cast_device.mojo`
    around the base binding's `unique_inverse` sort): `src` the user's
    labels as they are (autograd/pyapi.mojo cast_in_of descriptor, one per
    row), the classes in the labels' own dtype into classes_addr (n slots),
    codes into codes_addr. Returns the class count."""
    var ck = model_labels_check(src, classes_addr, codes_addr)
    var ctx = sequence_ctx()
    var k = device_unique_inverse_cast(ctx, ck[0], ck[1], Int(py=classes_addr), Int(py=codes_addr))
    return model_labels_result(k)


def model_gather_labels_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return model_gather_labels_py(ex, addrs, ip)


def model_n_params_binding(prog: PythonObject, ptab: PythonObject) raises -> PythonObject:
    return model_n_params_py(prog, ptab)


def model_import_params_binding(
    prog: PythonObject, ptab: PythonObject, recs: PythonObject, names: PythonObject, params_addr: PythonObject,
) raises -> PythonObject:
    """Sequential.load_state_dict (autograd/weight_import.mojo)."""
    var ex = DeviceExec()
    return model_import_params_py(ex, prog, ptab, recs, names, params_addr)


def ops_load_weights_binding(sid: PythonObject, recs: PythonObject, names: PythonObject) raises -> PythonObject:
    """Session.load_weights (autograd/weight_import.mojo)."""
    return ops_load_weights_py[DeviceExec, OPS_DEVICE](sid, recs, names)


# The operation session (autograd/ops_api.mojo): one DeviceExec and one tape kept
# between calls, tensors by handle.
def ops_open_binding(record: PythonObject) raises -> PythonObject:
    return ops_open_py[DeviceExec, OPS_DEVICE](DeviceExec(), record)


def ops_close_binding(sid: PythonObject) raises -> PythonObject:
    return ops_close_py[DeviceExec, OPS_DEVICE](sid)


def ops_upload_binding(sid: PythonObject, addr: PythonObject, ip: PythonObject) raises -> PythonObject:
    return ops_upload_py[DeviceExec, OPS_DEVICE](sid, addr, ip)


def ops_download_binding(sid: PythonObject, h: PythonObject, addr: PythonObject) raises -> PythonObject:
    return ops_download_py[DeviceExec, OPS_DEVICE](sid, h, addr)


def ops_op_binding(sid: PythonObject, code: PythonObject, hs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    return ops_op_py[DeviceExec, OPS_DEVICE](sid, code, hs, ip, fp)


def ops_op_packed_binding(sid: PythonObject, code: PythonObject, block: PythonObject) raises -> PythonObject:
    return ops_op_packed_py[DeviceExec, OPS_DEVICE](sid, code, block)


def ops_scalar_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_scalar_py[DeviceExec, OPS_DEVICE](sid, h)


def ops_shape_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_shape_py[DeviceExec, OPS_DEVICE](sid, h)


def ops_free_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_free_py[DeviceExec, OPS_DEVICE](sid, h)


def ops_reset_binding(sid: PythonObject) raises -> PythonObject:
    return ops_reset_py[DeviceExec, OPS_DEVICE](sid)


def ops_metric_binding(sid: PythonObject, kind: PythonObject, hs: PythonObject) raises -> PythonObject:
    return ops_metric_py[DeviceExec, OPS_DEVICE](sid, kind, hs)


def ops_backward_binding(sid: PythonObject, h: PythonObject, accumulate: PythonObject) raises -> PythonObject:
    return ops_backward_py[DeviceExec, OPS_DEVICE](sid, h, accumulate)


def ops_grad_binding(sid: PythonObject, h: PythonObject, addr: PythonObject) raises -> PythonObject:
    return ops_grad_py[DeviceExec, OPS_DEVICE](sid, h, addr)


def ops_optimizer_binding(sid: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    return ops_optimizer_py[DeviceExec, OPS_DEVICE](sid, ip, fp)


def ops_step_binding(sid: PythonObject, opt: PythonObject, hs: PythonObject) raises -> PythonObject:
    return ops_step_py[DeviceExec, OPS_DEVICE](sid, opt, hs)


def ops_scheduler_step_binding(sid: PythonObject, opt: PythonObject) raises -> PythonObject:
    return ops_scheduler_step_py[DeviceExec, OPS_DEVICE](sid, opt)


def ops_learning_rate_binding(sid: PythonObject, opt: PythonObject) raises -> PythonObject:
    return ops_learning_rate_py[DeviceExec, OPS_DEVICE](sid, opt)


def ops_zero_grad_binding(sid: PythonObject) raises -> PythonObject:
    return ops_zero_grad_py[DeviceExec, OPS_DEVICE](sid)


def ops_optimizer_state_binding(sid: PythonObject, opt: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_optimizer_state_py[DeviceExec, OPS_DEVICE](sid, opt, h)


def ops_load_optimizer_state_binding(
    sid: PythonObject, opt: PythonObject, h: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    return ops_load_optimizer_state_py[DeviceExec, OPS_DEVICE](sid, opt, h, ip, fp)


# Higher-order gradients (lane ag-ho-session; autograd/ops_api.mojo).
def ops_backward_graph_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_backward_graph_py[DeviceExec, OPS_DEVICE](sid, h)


def ops_backward_seed_binding(
    sid: PythonObject, h: PythonObject, seed: PythonObject, create_graph: PythonObject,
) raises -> PythonObject:
    return ops_backward_seed_py[DeviceExec, OPS_DEVICE](sid, h, seed, create_graph)


def ops_grad_handle_binding(sid: PythonObject, h: PythonObject) raises -> PythonObject:
    return ops_grad_handle_py[DeviceExec, OPS_DEVICE](sid, h)


def ops_vjp_binding(
    sid: PythonObject, out_h: PythonObject, inputs: PythonObject, seed: PythonObject, create_graph: PythonObject,
) raises -> PythonObject:
    return ops_vjp_py[DeviceExec, OPS_DEVICE](sid, out_h, inputs, seed, create_graph)


def ops_hvp_binding(sid: PythonObject, loss: PythonObject, params: PythonObject, vs: PythonObject) raises -> PythonObject:
    return ops_hvp_py[DeviceExec, OPS_DEVICE](sid, loss, params, vs)


def ops_jvp_binding(sid: PythonObject, out_h: PythonObject, inputs: PythonObject, us: PythonObject) raises -> PythonObject:
    return ops_jvp_py[DeviceExec, OPS_DEVICE](sid, out_h, inputs, us)


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


def model_load_checkpoint_binding(path: PythonObject, addrs: PythonObject) raises -> PythonObject:
    return model_load_checkpoint_py(path, addrs)


def ops_save_state_binding(
    sid: PythonObject, path: PythonObject, params: PythonObject, optimizer: PythonObject, counters: PythonObject,
) raises -> PythonObject:
    return ops_save_state_py[DeviceExec, OPS_DEVICE](sid, path, params, optimizer, counters)


def ops_load_state_binding(
    sid: PythonObject, path: PythonObject, requires_grad: PythonObject, optimizer: PythonObject,
) raises -> PythonObject:
    return ops_load_state_py[DeviceExec, OPS_DEVICE](sid, path, requires_grad, optimizer)


def adafactor_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return adafactor_step_py(ex, addrs, ip, fp)


def lamb_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return lamb_step_py(ex, addrs, ip, fp)


def layer_norm_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return layer_norm_py(ex, addrs, ip, fp)


def theta_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return theta_py(ex, addrs, ip, fp)


def croston_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return croston_py(ex, addrs, ip)


def croston_forecast_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return croston_forecast_py(ex, addrs, ip)


def ets_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    comptime if ETS_TEAM or IDN_ETS_TEAM:
        # Apple FAST default (off: -D MOJOLEARN_ETS_TEAM_OFF), and IDENTICAL behind
        # IDENTICAL by default (off: -D MOJOLEARN_IDN_ETS_TEAM_OFF; the host binding replays it in
        # sequence/exec.mojo): ETS(A, A|Ad, N) one series per block
        # (sequence/ets_team.mojo); every other model keeps ets_py
        if ets_team_applies(ip):
            return ets_team_py(ex, addrs, ip, fp)
    return ets_py(ex, addrs, ip, fp)


def garch_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """GARCH on the device, one series per block (`sequence/fit_team.mojo`),
    every batch size: the same statements per series as the host column's
    `op_garch`."""
    var ex = DeviceExec()
    return garch_team_py(ex, addrs, ip)


def prophet_days_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    return prophet_days_py(addrs, ip)


def prophet_features_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return prophet_features_py(ex, addrs, ip, fp)


def prophet_changepoints_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    return prophet_changepoints_py(addrs, ip, fp)


def prophet_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Prophet fit on the device, one series per block
    (`sequence/fit_team.mojo`), every batch size: the same statements per
    series as the host column's `op_prophet_fit`."""
    var ex = DeviceExec()
    return prophet_fit_team_py(ex, addrs, ip, fp)


def prophet_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return prophet_predict_py(ex, addrs, ip)


def moe_forward_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """ip = [T, D, F, E, k, renormalise] uploads the weights at addrs[1..3];
    a seventh entry > 0 is a `moe_weights_put` handle whose device copies
    are read instead (addrs[1..3] are then unread). Lane
    gap-neural-overhead2: the same launches on the same words."""
    if len(ip) == 7 and ival(ip, 6) > 0:
        var t = moe_forward_check(addrs, ip, 7)
        var w = moe_weights_ptrs(ival(ip, 6))
        var ex = DeviceExec()
        return moe_forward_run(ex, addrs, t[0], t[1], t[2], t[3], t[4], ival(ip, 5), w[0], w[1], w[2])
    var ex = DeviceExec()
    return moe_forward_py(ex, addrs, ip)


def moe_backward_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """`moe_backward` (sequence/pyapi.mojo moe_backward_py): ip = [T, D, F,
    E, k, renormalise]; a seventh entry > 0 is a `moe_weights_put` handle
    whose device copies are read instead of addrs[1..3]."""
    if len(ip) == 7 and ival(ip, 6) > 0:
        if len(addrs) != 13:
            raise Error("moe_backward: requires 13 addresses and 7 integer parameters")
        var T = ival(ip, 0)
        var D = ival(ip, 1)
        var F = ival(ip, 2)
        var En = ival(ip, 3)
        var k = ival(ip, 4)
        if T < 1 or D < 1 or F < 1 or En < 1 or k < 1 or k > En:
            raise Error("moe_backward: T, D, F, E >= 1 and 1 <= k <= E")
        var w = moe_weights_ptrs(ival(ip, 6))
        var ex = DeviceExec()
        return moe_backward_run(ex, addrs, T, D, F, En, k, ival(ip, 5), w[0], w[1], w[2])
    var ex = DeviceExec()
    return moe_backward_py(ex, addrs, ip)


def moe_weights_put_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [router (E, D), gate_up (E, 2F, D), down (E, D, F)]; ip = [E, D, F]."""
    if len(addrs) != 3 or len(ip) != 3:
        raise Error("moe_weights_put: requires 3 addresses and [E, D, F]")
    return PythonObject(moe_weights_put(
        fptr(addrs[0], "router"), fptr(addrs[1], "gate_up_proj"), fptr(addrs[2], "down_proj"),
        ival(ip, 0), ival(ip, 1), ival(ip, 2),
    ))


def moe_forward_dev_binding(handles: PythonObject, ip: PythonObject) raises -> PythonObject:
    return moe_forward_dev_py(handles, ip)


def moe_weights_free_binding(h: PythonObject) raises -> PythonObject:
    moe_weights_free(Int(py=h))
    return PythonObject(0)


@export
def PyInit__mojolearn_x_sequence() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_sequence")
        m.def_function[numeric_mode_binding]("x_sequence_numeric_mode")
        m.def_function[vendor_binding]("x_sequence_vendor")
        m.def_function[rnn_fit_binding]("rnn_fit")
        m.def_function[rnn_predict_binding]("rnn_predict")
        m.def_function[rnn_n_params_binding]("rnn_n_params")
        m.def_function[optimizer_step_binding]("optimizer_step")
        m.def_function[stl_binding]("stl")
        m.def_function[var_fit_binding]("var_fit")
        comptime if SCHED_FAST_TABLE:
            m.def_function[sched_exp_block_binding]("sched_exp_block")
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
        m.def_function[ops_profile_reset_binding]("ops_profile_reset")
        m.def_function[ops_profile_report_binding]("ops_profile_report")
        m.def_function[model_save_checkpoint_binding]("model_save_checkpoint")
        m.def_function[model_checkpoint_info_binding]("model_checkpoint_info")
        m.def_function[model_load_checkpoint_binding]("model_load_checkpoint")
        m.def_function[ops_save_state_binding]("ops_save_state")
        m.def_function[ops_load_state_binding]("ops_load_state")
        m.def_function[adafactor_step_binding]("adafactor_step")
        m.def_function[lamb_step_binding]("lamb_step")
        # the resident optimizer state (sequence/opt_resident.mojo)
        m.def_function[optimizer_resident_open_binding]("optimizer_resident_open")
        m.def_function[lamb_resident_open_binding]("lamb_resident_open")
        m.def_function[optimizer_resident_close_binding]("optimizer_resident_close")
        m.def_function[optimizer_resident_move_binding]("optimizer_resident_move")
        m.def_function[optimizer_resident_step_binding]("optimizer_resident_step")
        m.def_function[lamb_resident_step_binding]("lamb_resident_step")
        # Adafactor's second moment on the device (FAST + Apple default, off
        # under -D MOJOLEARN_AF_FAST_RESIDENT_OFF; the Python side uses it
        # only when these exist)
        comptime if AF_RESIDENT:
            m.def_function[adafactor_resident_open_binding]("adafactor_resident_open")
            m.def_function[adafactor_resident_step_binding]("adafactor_resident_step")
        m.def_function[layer_norm_binding]("layer_norm")
        # lane gap-neural-io: resident tensors and the device-I/O entries
        # (the Python side uses them only when they exist; the host binding
        # has none)
        m.def_function[seq_tensor_alloc_binding]("seq_tensor_alloc")
        m.def_function[seq_tensor_free_binding]("seq_tensor_free")
        m.def_function[seq_tensor_upload_binding]("seq_tensor_upload")
        m.def_function[seq_tensor_download_binding]("seq_tensor_download")
        m.def_function[seq_tensor_copy_binding]("seq_tensor_copy")
        m.def_function[optimizer_resident_step_dev_binding]("optimizer_resident_step_dev")
        m.def_function[lamb_resident_step_dev_binding]("lamb_resident_step_dev")
        comptime if AF_RESIDENT:
            m.def_function[adafactor_resident_step_dev_binding]("adafactor_resident_step_dev")
        m.def_function[layer_norm_dev_binding]("layer_norm_dev")
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
        m.def_function[moe_weights_put_binding]("moe_weights_put")
        m.def_function[moe_weights_free_binding]("moe_weights_free")
        comptime if SEQ_MOE_DEVICE_IO:
            m.def_function[moe_forward_dev_binding]("moe_forward_dev")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_sequence: ", e))
