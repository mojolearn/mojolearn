# SPDX-License-Identifier: Apache-2.0
"""Private GPU probe; no product dispatch and no host numerical arithmetic."""
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.hostptr import f32_ptr, i32_ptr, copy_f32


from arima.k3_df_probe_impl import _probe_df
from arima.impl.fast_scalar_df import DF_MAX_OBS


def arima_k3_df_probe_binding(y: PythonObject, state: PythonObject,
        stages: PythonObject, stats: PythonObject, info: PythonObject,
        gradients: PythonObject, config: PythonObject) raises -> PythonObject:
    """ABI3 config=[batch,nobs,intercept,gradient]. Private borrowed pointers.

    state=(5,batch): T,Q,P0,alpha0,mu; y=(batch,nobs).
    stages=(3,batch,nobs): prediction,residual,variance.
    stats=(4,batch): rounded LL,final P,LL hi,LL lo; info=(batch,) i32.
    Enabled gradient=(batch//4,3), members model-major [base,p0+h,p1+h,p2+h].
    All other arrays contiguous f32; no model initialization on this route.
    """
    if len(config) != 4:
        raise Error("K3 DF probe: expected [batch,nobs,intercept,gradient]")
    var batch = Int(py=config[0])
    var nobs = Int(py=config[1])
    var intercept = Int(py=config[2])
    var gradient = Int(py=config[3])
    if batch < 1 or batch > 4096 or nobs < 1 or nobs > DF_MAX_OBS:
        raise Error("K3 DF probe: unsupported dimensions")
    if (intercept != 0 and intercept != 1) or (gradient != 0 and gradient != 1):
        raise Error("K3 DF probe: invalid flags")
    if gradient != 0 and (batch % 4 != 0 or nobs < 2):
        raise Error("K3 DF probe: gradient needs four members/model and nobs>=2")
    var yp = Int(py=y)
    var sp = Int(py=state)
    var op = Int(py=stages)
    var lp = Int(py=stats)
    var ip = Int(py=info)
    var gp = Int(py=gradients)
    if yp == 0 or sp == 0 or op == 0 or lp == 0 or ip == 0 or (gradient != 0 and gp == 0):
        raise Error("K3 DF probe: null input/output")
    with GILReleased(Python()):
        _probe_df(yp, sp, op, lp, ip, gp, batch, nobs, intercept, gradient)
    return PythonObject(3)
