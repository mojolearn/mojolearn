# SPDX-License-Identifier: Apache-2.0
"""Device launchers for the pure neural_profile contract; no host binding imports this module."""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.contract import CONTRACT_K_LEAF_MIN
from gemm.experiments.neural_profile import (
    NN03,NN04,neural_validate,neural_partition,neural_strides,neural_cell,
)

def neural_profile_kernel[CHAINS: Int, SLOTS: Int](
    c: MutPointer[Float32, MutAnyOrigin], a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin], m: Int32, n: Int32, k: Int32,
    leaf: Int32, leaves: Int32, asi: Int32, asp: Int32, bsp: Int32, bsj: Int32,
):
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell < Int(m)*Int(n):
        c.unsafe_store(cell,neural_cell[CHAINS,SLOTS](a,b,cell//Int(n),cell%Int(n),
            Int(k),Int(leaf),Int(leaves),Int(asi),Int(asp),Int(bsp),Int(bsj)))


def neural_profile_device[MIN_LEAF: Int = 128, CHAINS: Int = 1, SLOTS: Int = 16](
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int, op: Int,
) raises:
    """Low-level explicit profile launch; NN03/04 selection lives below."""
    neural_validate(m,n,k,op)
    if len(c)<m*n or len(a)<m*k or len(b)<n*k:
        raise Error("neural profile buffer too short")
    if m == 0 or n == 0:
        return
    var part = neural_partition[MIN_LEAF](k)
    if part[1] >= (1 << SLOTS):
        raise Error("neural profile fold stack too small")
    var st = neural_strides(op,m,n,k)
    ctx.enqueue_function[neural_profile_kernel[CHAINS,SLOTS]](
        c,a,b,Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),
        Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
        grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))


def neural_profile_ab_device[LEAF: Int = 128, CHAINS: Int = 1, CANDIDATE: Bool = False](
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int, op: Int,
) raises:
    """A and B share scheduling; leaf and chain changes can be isolated.

    NN03 permits LEAF; NN04 permits CHAINS. Missing defines and ALL_OFF
    resolve that dimension to the incumbent profile, including its I04 build
    constant. Default CANDIDATE=False cannot select a new profile.
    """
    comptime L = LEAF if CANDIDATE and NN03 else CONTRACT_K_LEAF_MIN
    comptime C = CHAINS if CANDIDATE and NN04 else 1
    neural_profile_device[L,C](ctx,c,a,b,m,n,k,op)


