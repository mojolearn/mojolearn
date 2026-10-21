# SPDX-License-Identifier: Apache-2.0
"""Typed checked GEMM body moved word for word from bindings/neural_gemm_boundary_host.mojo."""
from std.math import isfinite
from gemm.host.neural_gemm import neural_gemm_host_into




def neural_gemm_checked_host_into(
    ap: MutPointer[Float32, MutUntrackedOrigin], bp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin], m: Int, n: Int, k: Int, op: Int,
) raises:
    for i in range(m*k):
        if not isfinite(ap.unsafe_load(i)):
            raise Error("SmallMLPTrainer GEMM refuses nonfinite inputs")
    for i in range(n*k):
        if not isfinite(bp.unsafe_load(i)):
            raise Error("SmallMLPTrainer GEMM refuses nonfinite inputs")
    neural_gemm_host_into(cp,ap,bp,m,n,k,op)
    for i in range(m*n):
        if not isfinite(cp.unsafe_load(i)):
            raise Error("SmallMLPTrainer GEMM returned a nonfinite result")
