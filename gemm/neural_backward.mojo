# SPDX-License-Identifier: Apache-2.0
"""Existing linear-gradient routing under the selected neural GEMM profile.

The table and its negative controls are reused unchanged. Forward, dInput,
dWeight and GEMM bias reductions all reach gemm.neural_dispatch, so NN03/04
never update only half of a layer. No compilation or verification performed.
"""
from max.gpu.host import DeviceBuffer,DeviceContext
from gemm.contract import OP_NN,OP_NT,OP_TN
from gemm.checks.gemm_backward import (
    BWD_DC_LEFT,BWD_DC_RIGHT,ANY_BWD_SABOTAGE,SAB_BWD_BIAS_AXIS,
    gemm_backward_a_call,gemm_backward_b_call,gemm_backward_call_name,
    gemm_backward_sabotage_name,identical_gemm_backward_bias_ones_floats,
)
from gemm.neural_dispatch import identical_gemm_into,identical_gemm_workspace_max_floats


def identical_gemm_backward_a_workspace_max_floats(op: Int,m: Int,n: Int,k: Int) -> Int:
    var call = gemm_backward_a_call(op,m,n,k)
    return identical_gemm_workspace_max_floats(call[1],call[2],call[3])


def identical_gemm_backward_b_workspace_max_floats(op: Int,m: Int,n: Int,k: Int) -> Int:
    var call = gemm_backward_b_call(op,m,n,k)
    return identical_gemm_workspace_max_floats(call[1],call[2],call[3])


def identical_gemm_backward_bias_workspace_max_floats(m: Int,n: Int) -> Int:
    return identical_gemm_workspace_max_floats(1,n,m)


def identical_gemm_backward_workspace_max_floats(op: Int,m: Int,n: Int,k: Int,with_bias: Bool) -> Int:
    var count = max(identical_gemm_backward_a_workspace_max_floats(op,m,n,k),
                    identical_gemm_backward_b_workspace_max_floats(op,m,n,k))
    if with_bias:
        count = max(count,identical_gemm_backward_bias_workspace_max_floats(m,n))
    return max(1,count)


def identical_gemm_backward_a_into(ctx: DeviceContext,mut da: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
    var call = gemm_backward_a_call(op,m,n,k)
    if call[4]==BWD_DC_LEFT:
        identical_gemm_into(ctx,da,dc,b,ws,call[1],call[2],call[3],call[0])
    else:
        identical_gemm_into(ctx,da,b,dc,ws,call[1],call[2],call[3],call[0])


def identical_gemm_backward_b_into(ctx: DeviceContext,mut db: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
    var call = gemm_backward_b_call(op,m,n,k)
    if call[4]==BWD_DC_LEFT:
        identical_gemm_into(ctx,db,dc,a,ws,call[1],call[2],call[3],call[0])
    else:
        identical_gemm_into(ctx,db,a,dc,ws,call[1],call[2],call[3],call[0])


def identical_gemm_backward_bias_into(ctx: DeviceContext,mut dbias: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],mut ones: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int) raises:
    comptime if SAB_BWD_BIAS_AXIS:
        identical_gemm_into(ctx,dbias,dc,ones,ws,m,1,n,OP_NT)
    else:
        identical_gemm_into(ctx,dbias,ones,dc,ws,1,n,m,OP_NN)
