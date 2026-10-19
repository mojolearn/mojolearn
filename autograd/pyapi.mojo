# SPDX-License-Identifier: Apache-2.0
"""The address contract of the model-description entries, written once over
`Exec`: `bindings/_mojolearn_x_sequence.mojo` calls these with a
`DeviceExec`, `bindings/_mojolearn_x_sequence_host.mojo` with a `HostExec`.
Python passes the description and the options as integer and float lists
plus caller-owned C-contiguous float32 buffers by address; Mojo validates
them and runs the whole fit or prediction. Nothing is retained after a call.

Description (both entries): prog = [1, input features, layer count, then per
layer (code, arg0, arg1, 0)], one float per layer in ptab
(autograd/model.mojo LAYER_*; docs/plans/AUTOGRAD_API.md)."""
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from sequence.pyapi import fptr, fval, ival
from sequence.recurrent import OptConfig
from autograd.model import FitConfig, ModelSpec, model_fit, model_predict, parse_model

# Training options ahead of the per-layer float table in model_fit's fp.
comptime FIT_INTS = 14
comptime FIT_FLOATS = 9


def _ints(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # driver-loop: over description words
        out.append(ival(p, i))
    return out^


def _floats(p: PythonObject, start: Int) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(start, len(p)):  # driver-loop: over layers' float arguments
        out.append(fval(p, i))
    return out^


def _seed(ip: PythonObject, at: Int) raises -> UInt64:
    var lo = ival(ip, at)
    var hi = ival(ip, at + 1)
    if lo < 0 or hi < 0 or lo > 0xFFFFFFFF or hi > 0xFFFFFFFF:
        raise Error("model: the seed words must be 32-bit non-negative values")
    return (UInt64(hi) << 32) | UInt64(lo)


def model_fit_py[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    """addrs = [X (N, D), Y ((N, out) float32, or N int32 codes), params
    (n_params, in/out), loss_curve (epochs, out)];
    ip = [N, loss, target, optimizer kind, optimizer flags, schedule,
    schedule step, batch, epochs, shuffle, seed_lo, seed_hi,
    n_iter_no_change, init];
    fp = [lr, f1 (momentum / beta1), f2 (beta2), eps, weight_decay,
    f7 (dampening), tol, schedule gamma, huber delta] + one float per layer.
    Returns the number of epochs run."""
    if len(addrs) != 4 or len(ip) != FIT_INTS or len(fp) < FIT_FLOATS + 1:
        raise Error("model_fit: requires 4 addresses, 14 integer parameters and 9 floats plus one per layer")
    var spec = parse_model(_ints(prog), _floats(fp, FIT_FLOATS))
    var N = ival(ip, 0)
    var opt = OptConfig(ival(ip, 3), ival(ip, 4), fval(fp, 1), fval(fp, 2), fval(fp, 3), fval(fp, 4), fval(fp, 5))
    var cfg = FitConfig(
        ival(ip, 1), ival(ip, 2), opt, fval(fp, 0), ival(ip, 5), ival(ip, 6), fval(fp, 7),
        ival(ip, 7), ival(ip, 8), ival(ip, 9) != 0, _seed(ip, 10), ival(ip, 12),
        Float64(py=fp[6]), fval(fp, 8), ival(ip, 13) != 0,
    )
    var n_iter = model_fit(ex, spec, cfg, fptr(addrs[0], "X"), fptr(addrs[1], "Y"), N,
                           fptr(addrs[2], "params"), fptr(addrs[3], "loss_curve"))
    return PythonObject(n_iter)


def model_predict_py[E: Exec](
    mut ex: E, addrs: PythonObject, prog: PythonObject, ptab: PythonObject, ip: PythonObject,
) raises -> PythonObject:
    """addrs = [X (N, D), params, out (N, out)]; ptab = one float per layer;
    ip = [N, chunk, transform (0 none, 1 softmax, 2 sigmoid, 3 the (N, 2)
    [1 - p, p] of a one-output sigmoid)]. Returns the number of values
    written."""
    if len(addrs) != 3 or len(ip) != 3:
        raise Error("model_predict: requires 3 addresses and 3 integer parameters")
    var spec = parse_model(_ints(prog), _floats(ptab, 0))
    var N = ival(ip, 0)
    model_predict(ex, spec, fptr(addrs[0], "X"), N, fptr(addrs[1], "params"), fptr(addrs[2], "out"),
                  ival(ip, 1), ival(ip, 2))
    if ival(ip, 2) == 3:
        return PythonObject(N * 2)
    return PythonObject(N * spec.out_features())


def model_n_params_py(prog: PythonObject, ptab: PythonObject) raises -> PythonObject:
    """[parameter count, output width] of a description (validates it)."""
    var spec = parse_model(_ints(prog), _floats(ptab, 0))
    var out = Python.list()
    out.append(PythonObject(spec.n_params))
    out.append(PythonObject(spec.out_features()))
    return out
