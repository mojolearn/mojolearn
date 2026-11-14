# SPDX-License-Identifier: Apache-2.0
"""The device chunked LM head (training/chunked_lm_head_v2.mojo, the
`_mojolearn_training` entries) on the training binding's process context, fixtures in order:
they share the context's stream, and each call keeps its own kernel
parallelism."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from core.hostptr import f32_ptr, i32_ptr
from training.chunked_lm_head_v2 import chunked_lm_head_v2_loss_host, chunked_lm_head_v2_train_host
from checks.native_tests.core_chunked_lm_head_denormals_cases import (
    CHUNKED_DENORMAL_FIXTURES, ChunkedHeadCalls, require, run_chunked_denormal_fixture,
)

# The training binding's process context (bindings/_mojolearn_training.mojo).
comptime _NEURAL_CTX = "MojolearnNeuralTrainingContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralTrainingContextFastV3"


struct DeviceChunkedHead(ChunkedHeadCalls):
    @staticmethod
    def loss(loss: Int, row_max: Int, denom: Int, hidden: Int, weight: Int, targets: Int,
             rows: Int, vocab: Int, width: Int) raises:
        var ctx = neural_ctx[_NEURAL_CTX]()
        _ = chunked_lm_head_v2_loss_host(ctx, f32_ptr(loss), f32_ptr(row_max), f32_ptr(denom),
                                         f32_ptr(hidden), f32_ptr(weight), i32_ptr(targets), rows, vocab, width)

    @staticmethod
    def train(loss: Int, row_max: Int, denom: Int, d_hidden: Int, d_weight: Int, hidden: Int,
              weight: Int, targets: Int, rows: Int, vocab: Int, width: Int) raises:
        var ctx = neural_ctx[_NEURAL_CTX]()
        _ = chunked_lm_head_v2_train_host(ctx, f32_ptr(loss), f32_ptr(row_max), f32_ptr(denom),
                                          f32_ptr(d_hidden), f32_ptr(d_weight), f32_ptr(hidden),
                                          f32_ptr(weight), i32_ptr(targets), rows, vocab, width)


def run_checks() raises:
    var failed = 0
    for unit in range(CHUNKED_DENORMAL_FIXTURES):
        try:
            run_chunked_denormal_fixture[DeviceChunkedHead](unit)
        except error:
            failed += 1
            print("FAIL core_chunked_lm_head_denormals_gpu fixture", unit, String(error))
    require(failed == 0, "core_chunked_lm_head_denormals_gpu failed " + String(failed) + " fixtures")
    print("PASS core_chunked_lm_head_denormals_gpu:", CHUNKED_DENORMAL_FIXTURES, "shapes, ordered on the device")
