# SPDX-License-Identifier: Apache-2.0
"""NN06: rounded bias/scale/residual/ReLU stages with shared host mirrors.

Generic component API, uncompiled/unverified. The selected public NN06
placement is training.mlp_ops, with its existing activation/backward rules;
this generic component does not redefine another caller's activation. Order is product -> bias -> scale -> residual -> optional ReLU.
Each enabled stage owns an explicit FP32/FTZ boundary in both A and B.
"""
from gemm.experiments.neural_profile_device import neural_profile_device
from std.sys.compile import is_defined, get_defined_int
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz,identical_mul,identical_fmax
from gemm.contract import CONTRACT_K_LEAF_MIN
from gemm.experiments.neural_profile import (
    NEURAL_EXPERIMENTS_ALLOWED, NEURAL_LEAF, NEURAL_CHAINS,neural_partition,neural_strides,neural_validate,
    neural_cell,neural_profile_host,
)

# OFF: no compile/identity/quality/timing evidence in this worktree. This is
# separate from I05's previously losing bias-only scalar component candidate.
# ONE epilogue-fusion switch with per-caller arms (NN06 + NI09 merged,
# 2026-10-07): -D MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE=<mask>, 1 = MLP caller
# (this rounded epilogue, training/mlp_ops.mojo), 2 = CNN caller (NI09 tiled
# bias, x_cnn/device.mojo), 3 = both. One idea, two kernels; the mask lets
# the grid time each caller family separately.
comptime NEURAL_GEMM_EPILOGUE = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE", 0]()
comptime NN06 = NEURAL_EXPERIMENTS_ALLOWED and (NEURAL_GEMM_EPILOGUE & 1) != 0
comptime EP_BIAS = 1
comptime EP_SCALE = 2
comptime EP_RESIDUAL = 4
comptime EP_RELU = 8


@always_inline
def neural_epilogue_stage[STAGE: Int](value: Float32, operand: Float32) -> Float32:
    comptime if STAGE == EP_SCALE:
        # Pinned multiplication cannot fuse with the following residual add.
        return ftz(identical_mul(ftz(value),ftz(operand)))
    elif STAGE == EP_RELU:
        return ftz(identical_fmax(ftz(value),Float32(0)))
    else:
        return ftz(ftz(value)+ftz(operand))


@always_inline
def neural_epilogue_apply[KIND: Int](
    value: Float32, bias: MutPointer[Float32,MutAnyOrigin],
    residual: MutPointer[Float32,MutAnyOrigin],preactivation: MutPointer[Float32,MutAnyOrigin],
    cell: Int,n: Int,scale: Float32,
) -> Float32:
    var result = ftz(value)
    comptime if (KIND & EP_BIAS) != 0:
        result = neural_epilogue_stage[EP_BIAS](result,bias.unsafe_load(cell%n))
    comptime if (KIND & EP_SCALE) != 0:
        result = neural_epilogue_stage[EP_SCALE](result,scale)
    comptime if (KIND & EP_RESIDUAL) != 0:
        result = neural_epilogue_stage[EP_RESIDUAL](result,residual.unsafe_load(cell))
    comptime if (KIND & EP_RELU) != 0:
        preactivation.unsafe_store(cell,result)
        result = neural_epilogue_stage[EP_RELU](result,Float32(0))
    return result


def _neural_epilogue_fused_kernel[KIND: Int](
    c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],bias: MutPointer[Float32,MutAnyOrigin],
    residual: MutPointer[Float32,MutAnyOrigin],preactivation: MutPointer[Float32,MutAnyOrigin],
    m: Int32,n: Int32,k: Int32,leaf: Int32,leaves: Int32,
    asi: Int32,asp: Int32,bsp: Int32,bsj: Int32,scale: Float32,
):
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell < Int(m)*Int(n):
        var product = neural_cell[NEURAL_CHAINS](a,b,cell//Int(n),cell%Int(n),Int(k),Int(leaf),
            Int(leaves),Int(asi),Int(asp),Int(bsp),Int(bsj))
        c.unsafe_store(cell,neural_epilogue_apply[KIND](product,bias,residual,preactivation,cell,Int(n),scale))


def _neural_epilogue_stage_kernel[STAGE: Int](
    c: MutPointer[Float32,MutAnyOrigin],operand: MutPointer[Float32,MutAnyOrigin],
    preactivation: MutPointer[Float32,MutAnyOrigin],cells: Int32,n: Int32,scale: Float32,
):
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell < Int(cells):
        var value = ftz(c.unsafe_load(cell))
        var other = scale
        comptime if STAGE == EP_BIAS:
            other = operand.unsafe_load(cell%Int(n))
        elif STAGE == EP_RESIDUAL:
            other = operand.unsafe_load(cell)
        elif STAGE == EP_RELU:
            preactivation.unsafe_store(cell,value)
        c.unsafe_store(cell,neural_epilogue_stage[STAGE](value,other))


def neural_epilogue_ab[KIND: Int = EP_BIAS, CANDIDATE: Bool = False](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],mut bias: DeviceBuffer[DType.float32],
    mut residual: DeviceBuffer[DType.float32],mut preactivation: DeviceBuffer[DType.float32],
    m: Int,n: Int,k: Int,op: Int,scale: Float32 = 1.0,
) raises:
    comptime assert KIND >= 0 and KIND < 16, "epilogue stage mask"
    neural_validate(m,n,k,op)
    if len(c)<m*n or len(a)<m*k or len(b)<n*k:
        raise Error("neural epilogue matrix buffer too short")
    comptime if (KIND & EP_BIAS) != 0:
        if len(bias)<n or bias.unsafe_ptr() == c.unsafe_ptr():
            raise Error("neural epilogue bias buffer short or aliases output")
    comptime if (KIND & EP_RESIDUAL) != 0:
        if len(residual)<m*n or residual.unsafe_ptr() == c.unsafe_ptr():
            raise Error("neural epilogue residual must be a distinct full buffer")
    comptime if (KIND & EP_RELU) != 0:
        if len(preactivation)<m*n or preactivation.unsafe_ptr() == c.unsafe_ptr():
            raise Error("neural epilogue preactivation must be a distinct full buffer")
    if m == 0 or n == 0:
        return
    comptime if CANDIDATE and NN06:
        var part = neural_partition[NEURAL_LEAF](k)
        var st = neural_strides(op,m,n,k)
        ctx.enqueue_function[_neural_epilogue_fused_kernel[KIND]](c,a,b,bias,residual,preactivation,
            Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),
            Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),scale,
            grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
    else:
        neural_profile_device[NEURAL_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)
        comptime if (KIND & EP_BIAS) != 0:
            ctx.enqueue_function[_neural_epilogue_stage_kernel[EP_BIAS]](c,bias,preactivation,
                Int32(m*n),Int32(n),scale,grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
        comptime if (KIND & EP_SCALE) != 0:
            ctx.enqueue_function[_neural_epilogue_stage_kernel[EP_SCALE]](c,bias,preactivation,
                Int32(m*n),Int32(n),scale,grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
        comptime if (KIND & EP_RESIDUAL) != 0:
            ctx.enqueue_function[_neural_epilogue_stage_kernel[EP_RESIDUAL]](c,residual,preactivation,
                Int32(m*n),Int32(n),scale,grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
        comptime if (KIND & EP_RELU) != 0:
            ctx.enqueue_function[_neural_epilogue_stage_kernel[EP_RELU]](c,bias,preactivation,
                Int32(m*n),Int32(n),scale,grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))


def neural_epilogue_host[KIND: Int = EP_BIAS](
    c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],bias: MutPointer[Float32,MutAnyOrigin],
    residual: MutPointer[Float32,MutAnyOrigin],preactivation: MutPointer[Float32,MutAnyOrigin],
    m: Int,n: Int,k: Int,op: Int,scale: Float32 = 1.0,
) raises:
    neural_profile_host[NEURAL_LEAF,NEURAL_CHAINS](c,a,b,m,n,k,op)
    for cell in range(m*n):
        c.unsafe_store(cell,neural_epilogue_apply[KIND](c.unsafe_load(cell),bias,residual,preactivation,cell,n,scale))


@always_inline
def neural_relu_backward_value(gradient: Float32,preactivation: Float32) -> Float32:
    # The explicit derivative convention at zero is zero. Full caller mapping
    # must establish it matches the model's convention before wiring this API.
    return ftz(gradient) if ftz(preactivation) > Float32(0) else Float32(0)


def neural_relu_backward_kernel(
    output: MutPointer[Float32,MutAnyOrigin],gradient: MutPointer[Float32,MutAnyOrigin],
    preactivation: MutPointer[Float32,MutAnyOrigin],count: Int32,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(count):
        output.unsafe_store(i,neural_relu_backward_value(gradient.unsafe_load(i),preactivation.unsafe_load(i)))


def neural_relu_backward_host(
    output: MutPointer[Float32,MutAnyOrigin],gradient: MutPointer[Float32,MutAnyOrigin],
    preactivation: MutPointer[Float32,MutAnyOrigin],count: Int,
):
    for i in range(count):
        output.unsafe_store(i,neural_relu_backward_value(gradient.unsafe_load(i),preactivation.unsafe_load(i)))
