# SPDX-License-Identifier: Apache-2.0
"""G03 independent NT-linear dA/dW jobs. NOT COMPILED OR TESTED.

Each job uses its original contraction length and the existing FLAT arithmetic
body. Two physical grids share dC; no gradient reductions are concatenated.
The caller guarantees disjoint outputs and lifetime through completion.
"""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import _flat_cell_body, contract_partition


def neural_linear_backward_kernel(
    da: MutPointer[Float32, MutAnyOrigin], dw: MutPointer[Float32, MutAnyOrigin],
    dc: MutPointer[Float32, MutAnyOrigin], a: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    m: Int32, n: Int32, k: Int32, a_leaf: Int32, a_parts: Int32,
    w_leaf: Int32, w_parts: Int32, first_job: Int32,
):
    var cell = Int32(Int(block_idx.x)*Int(block_dim.x) + Int(thread_idx.x))
    var job = Int(block_idx.y) + Int(first_job)
    if job == 0:
        # dA[m,k] = dC[m,n] W[n,k], contract K=n (OP_NN).
        _flat_cell_body(da, dc, w, m, k, n, a_leaf, a_parts,
                        n, Int32(1), k, Int32(1), cell)
    else:
        # dW[n,k] = dC[m,n]^T A[m,k], contract K=m (OP_TN).
        _flat_cell_body(dw, dc, a, n, k, m, w_leaf, w_parts,
                        Int32(1), n, k, Int32(1), cell)


def neural_linear_backward_into[GROUPED: Bool](
    ctx: DeviceContext, mut da: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32], mut dc: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32], mut w: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int,
) raises:
    var ap = contract_partition(n)
    var wp = contract_partition(m)
    comptime if GROUPED:  # NOT TESTED — NOT COMPILED — NOT MEASURED; selected only by the caller's OFF toggle.
        ctx.enqueue_function[neural_linear_backward_kernel](
            da, dw, dc, a, w, Int32(m), Int32(n), Int32(k),
            Int32(ap[0]), Int32(ap[1]), Int32(wp[0]), Int32(wp[1]), Int32(0),
            grid_dim=((max(m*k, n*k)+127)//128, 2, 1), block_dim=(128, 1, 1))
    else:
        ctx.enqueue_function[neural_linear_backward_kernel](
            da, dw, dc, a, w, Int32(m), Int32(n), Int32(k),
            Int32(ap[0]), Int32(ap[1]), Int32(wp[0]), Int32(wp[1]), Int32(0),
            grid_dim=((m*k+127)//128, 1, 1), block_dim=(128, 1, 1))
        ctx.enqueue_function[neural_linear_backward_kernel](
            da, dw, dc, a, w, Int32(m), Int32(n), Int32(k),
            Int32(ap[0]), Int32(ap[1]), Int32(wp[0]), Int32(wp[1]), Int32(1),
            grid_dim=((n*k+127)//128, 1, 1), block_dim=(128, 1, 1))
