# SPDX-License-Identifier: Apache-2.0
"""Shared host/device residual layer boundary admission; no GPU imports."""
from std.python import PythonObject
from training.residual_dropout_contract import residual_dropout_admit

@fieldwise_init
struct ResidualDropoutArgs(Copyable,Movable):
    var n: Int
    var offset: Int
    var seed_lo: Int
    var seed_hi: Int
    var stream: Int
    var p: Float32


def residual_dropout_args(params: PythonObject) raises -> ResidualDropoutArgs:
    if len(params)!=6:
        raise Error("residual_dropout params must be [n,offset,seed_lo,seed_hi,stream,p]")
    var args = ResidualDropoutArgs(Int(py=params[0]),Int(py=params[1]),Int(py=params[2]),
        Int(py=params[3]),Int(py=params[4]),Float32(Float64(py=params[5])))
    _ = residual_dropout_admit(args.n,args.offset,args.p,args.seed_lo,args.seed_hi,args.stream)
    return args^


def _residual_overlap(a: Int,b: Int,n: Int) -> Bool:
    return a<b+4*n and b<a+4*n


def residual_dropout_addresses[BACKWARD: Bool](x: Int,residual: Int,first: Int,second: Int,n: Int) raises:
    if n==0:
        return
    if x==0 or first==0:
        raise Error("residual_dropout refuses null borrowed buffers")
    if _residual_overlap(x,first,n):
        raise Error("residual_dropout outputs must be disjoint from inputs")
    comptime if BACKWARD:
        if second==0 or _residual_overlap(second,x,n) or _residual_overlap(first,second,n):
            raise Error("residual_dropout backward outputs must be nonnull and disjoint")
    else:
        if residual==0 or _residual_overlap(first,residual,n):
            raise Error("residual_dropout residual/output must be nonnull and disjoint")
