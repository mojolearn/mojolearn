# SPDX-License-Identifier: Apache-2.0
"""Public training residual-dropout layer; CPU-only forward and backward."""
from std.python import Python,PythonObject
from std.python._cpython import GILReleased
from bindings.hostptr import f32_ptr
from bindings.residual_dropout_common import residual_dropout_args,residual_dropout_addresses
from training.residual_dropout_host import residual_dropout_host


def residual_dropout_binding(x_addr: PythonObject,residual_addr: PythonObject,
    output_addr: PythonObject,params: PythonObject) raises -> PythonObject:
    var args = residual_dropout_args(params)
    var x = Int(py=x_addr)
    var r = Int(py=residual_addr)
    var y = Int(py=output_addr)
    residual_dropout_addresses[False](x,r,y,y,args.n)
    if args.n==0:
        return PythonObject(0)
    var xp = f32_ptr(x)
    var rp = f32_ptr(r)
    var yp = f32_ptr(y)
    var count = 0
    with GILReleased(Python()):
        count = residual_dropout_host[False](yp,yp,xp,rp,args.n,args.offset,args.p,args.seed_lo,args.seed_hi,args.stream)
    return PythonObject(count)


def residual_dropout_backward_binding(dy_addr: PythonObject,dx_addr: PythonObject,
    dresidual_addr: PythonObject,params: PythonObject) raises -> PythonObject:
    var args = residual_dropout_args(params)
    var dy = Int(py=dy_addr)
    var dx = Int(py=dx_addr)
    var dr = Int(py=dresidual_addr)
    residual_dropout_addresses[True](dy,dy,dx,dr,args.n)
    if args.n==0:
        return PythonObject(0)
    var yp = f32_ptr(dy)
    var xp = f32_ptr(dx)
    var rp = f32_ptr(dr)
    var count = 0
    with GILReleased(Python()):
        count = residual_dropout_host[True](xp,rp,yp,yp,args.n,args.offset,args.p,args.seed_lo,args.seed_hi,args.stream)
    return PythonObject(count)
