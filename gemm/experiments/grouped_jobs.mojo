# SPDX-License-Identifier: Apache-2.0
"""Separate grid jobs for independent products sharing one left operand.

The job axis changes operand/output offsets only. The production FLAT body's
leaf boundaries and fold are called unchanged; contractions are never joined.
This tests launch batching separately from tiled shared-operand staging.
"""
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import (
    _flat_cell_body, contract_partition, gemm_operand_strides,
)


def grouped_flat_kernel(c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    m: Int32,n: Int32,k: Int32,leaf: Int32,leaves: Int32,
    asi: Int32,asp: Int32,bsp: Int32,bsj: Int32):
    var job = Int(block_idx.y)
    _flat_cell_body(c+job*Int(m)*Int(n),a,b+job*Int(n)*Int(k),
                              m,n,k,leaf,leaves,asi,asp,bsp,bsj,
                              Int32(Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)))


# I03/N05 2026-10-06 L40S grouped component mixed WIN/LOSS: small shapes
# improve two layouts (~1.24x-1.31x), but middle layout loses (~0.25x);
# large shapes near parity (0.988x-1.029x), including rebound N05 versions.
# One same-process warmup/score; no full-workload default promotion. Evidence:
# overnight-ab-20261006/nvidia/{default,specific}-repair-normalized-measurements.json.
# Explicit grouped adapter only; the shared FLAT arithmetic body is unchanged.
def grouped_gemm(mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    ctx: DeviceContext,m: Int,n: Int,k: Int,op: Int,jobs: Int) raises:
    if m < 1 or n < 1 or k < 1 or jobs < 1 or jobs > 65535:
        raise Error("invalid grouped product geometry")
    if len(c) < jobs*m*n or len(a) < m*k or len(b) < jobs*n*k:
        raise Error("grouped product storage too small")
    var part = contract_partition(k)
    var st = gemm_operand_strides(op,m,n,k)
    ctx.enqueue_function[grouped_flat_kernel](c,a,b,Int32(m),Int32(n),Int32(k),
        Int32(part[0]),Int32(part[1]),Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
        grid_dim=((m*n+127)//128,jobs,1),block_dim=(128,1,1))
