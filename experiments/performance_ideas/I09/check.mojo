# SPDX-License-Identifier: Apache-2.0
"""Actual prefill and public decode entry share the same weights and
absolute token positions. Every recurrence stage, including final state,
is compared by the existing contract oracle on neighboring prefix tails.
New chunk/affine profiles remain conditional on complete replay contracts."""
from max.gpu.host import DeviceContext
from mamba.checks.mamba_check import planted_weights, clause_d
from mamba.checks.mamba_fixture import corpus_case_seed, corpus_x
from mamba.impl.modules.mamba import MambaDims

def main() raises:
    var ctx = DeviceContext()
    var widths: List[Int] = [8, 17]
    var prefixes: List[Int] = [15, 17, 33]
    for wi in range(len(widths)):
        var dims = MambaDims.of(widths[wi])
        var weights = planted_weights(dims)
        for li in range(len(prefixes)):
            var n = prefixes[li]
            var x = corpus_x(corpus_case_seed(1), 1, n, dims.d_model)
            clause_d(ctx, weights, x, n, dims)
    print("I09 PASS token_parallel_prefill_decode_cases=6")
