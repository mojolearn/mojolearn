# SPDX-License-Identifier: Apache-2.0
"""Fuse a rounded bias add with the production FLAT contract body.

Both arms explicitly materialize the product's Float32/FTZ seam. Each thread
owns one output; bias arithmetic follows that rounded product. This is an
opt-in research adapter and never changes the production GEMM dispatcher.
"""
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from gemm.checks.gemm_identical import identical_gemm_flat_kernel,contract_partition,gemm_operand_strides


def bias_kernel(c: MutPointer[Float32,MutAnyOrigin],bias: MutPointer[Float32,MutAnyOrigin],m: Int32,n: Int32):
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell < Int(m)*Int(n):
        c.unsafe_store(cell,ftz(ftz(c.unsafe_load(cell))+ftz(bias.unsafe_load(cell%Int(n)))))


def fused_bias_kernel(c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],bias: MutPointer[Float32,MutAnyOrigin],
    m: Int32,n: Int32,k: Int32,leaf: Int32,leaves: Int32,
    asi: Int32,asp: Int32,bsp: Int32,bsj: Int32):
    identical_gemm_flat_kernel(c,a,b,m,n,k,leaf,leaves,asi,asp,bsp,bsj)
    bias_kernel(c,bias,m,n)


def gemm_bias[FUSED: Bool](ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut bias: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
    if m < 1 or n < 1 or k < 1 or len(c)<m*n or len(a)<m*k or len(b)<n*k or len(bias)<n:
        raise Error("invalid GEMM bias buffer geometry")
    var part = contract_partition(k)
    var st = gemm_operand_strides(op,m,n,k)
    comptime if FUSED:
        ctx.enqueue_function[fused_bias_kernel](c,a,b,bias,Int32(m),Int32(n),Int32(k),
            Int32(part[0]),Int32(part[1]),Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
            grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
    else:
        ctx.enqueue_function[identical_gemm_flat_kernel](c,a,b,Int32(m),Int32(n),Int32(k),
            Int32(part[0]),Int32(part[1]),Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
            grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
        ctx.enqueue_function[bias_kernel](c,bias,Int32(m),Int32(n),
            grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
