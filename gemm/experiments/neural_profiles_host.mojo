# SPDX-License-Identifier: Apache-2.0
"""Independent host spelling for NEURAL V01. NOT COMPILED OR TESTED.

Not imported by the GPU entry. This source is an intended identity witness,
not evidence of identity. It deliberately spells the chains as separate loops
and uses the host level-by-level tree rather than the device fold stack.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from gemm.contract import contract_leaf_size, leaf_count
from gemm.host.gemm_oracle import _a_at, _b_at, fold_balanced_tree, gemm_oracle

comptime NI_HOST_TWO_CHAIN = (  # NOT TESTED — NOT COMPILED — NOT MEASURED; V01 host mirrors explicit new version, OFF.
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI_V01_TWO_CHAIN"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def neural_gemm_host(a: List[Float32], b: List[Float32],
                     op: Int, m: Int, n: Int, k: Int) raises -> List[Float32]:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "NEURAL experiment requires IDENTICAL"
    if m < 0 or n < 0 or k < 0 or op < 0 or op > 2:
        raise Error("invalid neural GEMM geometry/orientation")
    if len(a) < m*k or len(b) < n*k:
        raise Error("neural GEMM storage too small")
    comptime if not NI_HOST_TWO_CHAIN:
        return gemm_oracle(a, b, op, m, n, k)
    var result = List[Float32]()
    var leaf = contract_leaf_size(k)
    for row in range(m):
        for col in range(n):
            var partials = List[Float32]()
            for part in range(leaf_count(k, leaf)):
                var start = part*leaf
                var end = min(start+leaf, k)
                var even = Float32(0.0)
                var odd = Float32(0.0)
                for p in range(start, end, 2):
                    even = ftz(identical_mul_add(
                        ftz(_a_at(a, op, row, p, m, k)),
                        ftz(_b_at(b, op, p, col, n, k)), even))
                for p in range(start+1, end, 2):
                    odd = ftz(identical_mul_add(
                        ftz(_a_at(a, op, row, p, m, k)),
                        ftz(_b_at(b, op, p, col, n, k)), odd))
                if end-start > 1:
                    partials.append(ftz(ftz(even)+ftz(odd)))
                else:
                    partials.append(ftz(even))
            result.append(fold_balanced_tree(partials))
    return result^
