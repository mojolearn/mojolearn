# SPDX-License-Identifier: Apache-2.0
"""The host column's chunked LM head (core/mojolearn_training_host_impl.mojo,
the `_mojolearn_training_host` entries), fixtures run across host cores."""
from core.host_parallel import host_parallelize
from core.mojolearn_training_host_impl import chunked_lm_head_v2_loss_host_into, chunked_lm_head_v2_train_host_into
from checks.native_tests.core_chunked_lm_head_denormals_cases import (
    CHUNKED_DENORMAL_FIXTURES, ChunkedHeadCalls, require, run_chunked_denormal_fixture,
)


struct HostChunkedHead(ChunkedHeadCalls):
    @staticmethod
    def loss(loss: Int, row_max: Int, denom: Int, hidden: Int, weight: Int, targets: Int,
             rows: Int, vocab: Int, width: Int) raises:
        var a: List[Int] = [loss, row_max, denom, hidden, weight, targets]
        _ = chunked_lm_head_v2_loss_host_into(a, rows, vocab, width)

    @staticmethod
    def train(loss: Int, row_max: Int, denom: Int, d_hidden: Int, d_weight: Int, hidden: Int,
              weight: Int, targets: Int, rows: Int, vocab: Int, width: Int) raises:
        var a: List[Int] = [loss, row_max, denom, d_hidden, d_weight, hidden, weight, targets]
        _ = chunked_lm_head_v2_train_host_into(a, rows, vocab, width)


def run_checks() raises:
    var failures = List[Int](length=CHUNKED_DENORMAL_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            run_chunked_denormal_fixture[HostChunkedHead](unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL core_chunked_lm_head_denormals_cpu fixture", unit, String(error))
    host_parallelize(task, CHUNKED_DENORMAL_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_chunked_lm_head_denormals_cpu failed " + String(failed) + " fixtures")
    print("PASS core_chunked_lm_head_denormals_cpu:", CHUNKED_DENORMAL_FIXTURES, "shapes, parallel")
