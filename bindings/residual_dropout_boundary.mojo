# SPDX-License-Identifier: Apache-2.0
"""Public training residual-dropout layer; GPU forward and paired backward."""
from std.python import Python,PythonObject
from std.python._cpython import GILReleased
from core.neural_context import neural_ctx
from core.hostptr import f32_ptr
from bindings.residual_dropout_common import residual_dropout_args,residual_dropout_addresses
from training.residual_dropout import residual_dropout_device,residual_dropout_into
from training.residual_dropout_contract import residual_dropout_admit
from training.dev_tensors import train_dev_view
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# The training binding's context (bindings/_mojolearn_training.mojo
# _NEURAL_CTX): the train_dev handles live on it.
comptime _RD_DEV_CTX = "MojoNeuralTrainingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingContextFast"


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
        var ctx = neural_ctx["MojoNeuralTrainingContextIdentical"]()
        count = residual_dropout_device[False](ctx,yp,yp,xp,rp,args.n,args.offset,args.p,args.seed_lo,args.seed_hi,args.stream)
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
        var ctx = neural_ctx["MojoNeuralTrainingContextIdentical"]()
        count = residual_dropout_device[True](ctx,xp,rp,yp,yp,args.n,args.offset,args.p,args.seed_lo,args.seed_hi,args.stream)
    return PythonObject(count)


# Lane ne3-c-blocks-embedding (row N117): the same layer on training device
# arrays (train_dev handles, training/dev_tensors.mojo). Nothing crosses the
# bus but the refusal scans' words: the inputs are read and the outputs
# written in place on the device, so a training step that keeps its
# activations resident pays no upload of x / residual and no download of the
# output. The launches are `residual_dropout_into`, the host entry's own:
# no bit moves. The handles must be distinct (the backward reads its upstream
# gradient after writing dx).
def _rd_dev_handles(handles: PythonObject) raises -> List[Int]:
    if len(handles)!=3:
        raise Error("residual_dropout_dev requires three device array handles")
    var out = List[Int]()
    for i in range(3):  # small-loop(three handles of one layer call)
        out.append(Int(py=handles[i]))
    if out[0]==out[1] or out[0]==out[2] or out[1]==out[2]:
        raise Error("residual_dropout outputs must be disjoint from inputs")
    return out^


def residual_dropout_dev_binding(handles: PythonObject,params: PythonObject) raises -> PythonObject:
    """handles = [output, x, residual] (n floats each); params as
    `residual_dropout`. Returns n."""
    var args = residual_dropout_args(params)
    var hs = _rd_dev_handles(handles)
    if args.n==0:
        return PythonObject(0)
    with GILReleased(Python()):
        var ctx = neural_ctx[_RD_DEV_CTX]()
        var scale = residual_dropout_admit(args.n,args.offset,args.p,args.seed_lo,args.seed_hi,args.stream)
        var output = train_dev_view(ctx,hs[0],args.n,"residual_dropout output")
        var x = train_dev_view(ctx,hs[1],args.n,"residual_dropout values")
        var residual = train_dev_view(ctx,hs[2],args.n,"residual_dropout residual")
        var unused = ctx.enqueue_create_buffer[DType.float32](1)
        try:
            residual_dropout_into[False](ctx,output,unused,x,residual,args.n,args.offset,args.p,scale,
                args.seed_lo,args.seed_hi,args.stream)
            ctx.synchronize()
        except error:
            ctx.synchronize()
            raise error^
        _ = unused^
        _ = output^
        _ = x^
        _ = residual^
    return PythonObject(args.n)


def residual_dropout_backward_dev_binding(handles: PythonObject,params: PythonObject) raises -> PythonObject:
    """handles = [dx, dresidual, dy] (n floats each); params as
    `residual_dropout_backward`. Returns n."""
    var args = residual_dropout_args(params)
    var hs = _rd_dev_handles(handles)
    if args.n==0:
        return PythonObject(0)
    with GILReleased(Python()):
        var ctx = neural_ctx[_RD_DEV_CTX]()
        var scale = residual_dropout_admit(args.n,args.offset,args.p,args.seed_lo,args.seed_hi,args.stream)
        var dx = train_dev_view(ctx,hs[0],args.n,"residual_dropout dx")
        var dresidual = train_dev_view(ctx,hs[1],args.n,"residual_dropout dresidual")
        var dy = train_dev_view(ctx,hs[2],args.n,"residual_dropout gradient")
        var unused = ctx.enqueue_create_buffer[DType.float32](1)
        try:
            residual_dropout_into[True](ctx,dx,dresidual,dy,unused,args.n,args.offset,args.p,scale,
                args.seed_lo,args.seed_hi,args.stream)
            ctx.synchronize()
        except error:
            ctx.synchronize()
            raise error^
        _ = unused^
        _ = dx^
        _ = dresidual^
        _ = dy^
    return PythonObject(args.n)
