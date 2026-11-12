# SPDX-License-Identifier: Apache-2.0
"""Shared host/device residual layer boundary admission; no GPU imports."""
from std.python import PythonObject
from training.residual_dropout_contract import residual_dropout_admit
from training.residual_dropout_common_run import ResidualDropoutArgs


def residual_dropout_args(params: PythonObject) raises -> ResidualDropoutArgs:
    if len(params)!=6:
        raise Error("residual_dropout params must be [n,offset,seed_lo,seed_hi,stream,p]")
    var args = ResidualDropoutArgs(Int(py=params[0]),Int(py=params[1]),Int(py=params[2]),
        Int(py=params[3]),Int(py=params[4]),Float32(Float64(py=params[5])))
    _ = residual_dropout_admit(args.n,args.offset,args.p,args.seed_lo,args.seed_hi,args.stream)
    return args^
