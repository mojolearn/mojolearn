# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ADDRESS CONTRACT of the sequence lane's bindings, written once over
`Exec`: `bindings/_mojolearn_x_sequence.mojo` calls these with a
`DeviceExec`, `bindings/_mojolearn_x_sequence_host.mojo` with a `HostExec`.
Every buffer is a caller-owned C-contiguous array handed over by address;
nothing is retained after the call."""
# The PACK half of the former `providers/python/sequence_pyapi.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/sequence_pyapi_unpack.mojo`.
from std.python import PythonObject

from std.os import getenv as _getenv_seq
from sequence.exec_trait import Exec
from sequence.ops import FP
from sequence.ops_impl import (
    ETS_FAST_STALL_ITERS,
    ETS_FAST_STALL_REL,
    GARCH_FAST_STALL_ITERS,
    GARCH_FAST_STALL_REL,
    PROPHET_FAST_MIN_N,
    adafactor_core,
    adafactor_step_impl,
    lamb_bias,
    lamb_core,
    lamb_table,
    layer_norm_impl,
    mlp_fit_impl,
    mlp_predict_impl,
    moe_forward_core,
    moe_forward_impl,
    moe_forward_run_impl,
    opt_slots,
    opt_step_impl,
    rnn_fit_impl,
    rnn_predict_impl,
)
from autograd.route_ops import moe_backward_addrs


from providers.python.sequence_pyapi_unpack import (
    py_f64s,
    py_ints,
)


def rnn_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `rnn_fit_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(rnn_fit_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def rnn_predict_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `rnn_predict_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(rnn_predict_impl(ex, py_ints(addrs), py_ints(ip)))


def opt_step_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `opt_step_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(opt_step_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def mlp_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `mlp_fit_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(mlp_fit_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def mlp_predict_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `mlp_predict_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(mlp_predict_impl(ex, py_ints(addrs), py_ints(ip)))


# TOMBSTONE: the ops_* per-op session doors and lamb_step_py (dead native entries: reached only through their deleted exports) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_sequence.patch


def layer_norm_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `layer_norm_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(layer_norm_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def moe_forward_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `moe_forward_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(moe_forward_impl(ex, py_ints(addrs), py_ints(ip)))


def moe_backward_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `moe_backward_addrs` (autograd/route_ops.mojo), weights uploaded from addrs[1..3]."""
    return PythonObject(moe_backward_addrs(ex, py_ints(addrs), py_ints(ip), List[Int]()))


def moe_forward_run[E: Exec](
    mut ex: E, addrs: PythonObject, T: Int, D: Int, F: Int, En: Int, k: Int, renorm: Int, Wg: FP, Gu: FP, Dn: FP,
) raises -> PythonObject:
    """The Python door of `moe_forward_run_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(moe_forward_run_impl(ex, py_ints(addrs), T, D, F, En, k, renorm, Wg, Gu, Dn))


def moe_backward_resident[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, resident: List[Int]) raises -> PythonObject:
    """The Python door of `moe_backward_addrs` (autograd/route_ops.mojo) with the weights' device copies at `resident`."""
    return PythonObject(moe_backward_addrs(ex, py_ints(addrs), py_ints(ip), resident))
