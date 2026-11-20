# SPDX-License-Identifier: Apache-2.0
"""RUN half of `bindings/residual_dropout_boundary.mojo` (wave 8 run split):
the residual-dropout layer on training device arrays (train_dev handles),
forward and backward. The door unpacks the handles and parameters and releases
the GIL around these calls. Typed Mojo only: no Python import."""
from core.neural_context import neural_ctx
from training.residual_dropout import residual_dropout_into
from training.residual_dropout_contract import residual_dropout_admit
from training.residual_dropout_common_run import ResidualDropoutArgs
from training.dev_tensors import train_dev_view
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# The training binding's context (bindings/_mojolearn_training.mojo
# _NEURAL_CTX): the train_dev handles live on it.
comptime _RD_DEV_CTX = "MojolearnNeuralTrainingContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralTrainingContextFastV3"


def residual_dropout_dev_run(hs: List[Int],args: ResidualDropoutArgs) raises:
    """handles = [output, x, residual] (n > 0 floats each)."""
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


def residual_dropout_backward_dev_run(hs: List[Int],args: ResidualDropoutArgs) raises:
    """handles = [dx, dresidual, dy] (n > 0 floats each)."""
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
