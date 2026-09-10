# SPDX-License-Identifier: Apache-2.0
"""Focused PLAN_SORT edge cases and actual device negative-control gate."""
from max.gpu.host import DeviceContext
from embedding.checks.embedding_check import _upload_i32, _upload_f32, _download_i32, _download_f32, _zeros_i32_list, compare_i32, compare_f32
from embedding.checks.embedding_identical import identical_embedding_backward_into
from embedding.checks.embedding_sort import PLAN_SCAN, PLAN_SORT
from embedding.checks.embedding_oracle import EmbConfig, emb_counts, emb_run_begin, emb_perm_by_scan


def check_case(ctx: DeviceContext, n: Int, padding: Int, all_padding: Bool, accumulate: Bool) raises:
    var vocab = 7
    var width = 3
    var cfg = EmbConfig(vocab, width, padding, accumulate)
    var ids = List[Int32]()
    var dy = List[Float32]()
    for t in range(n):
        ids.append(Int32(padding if all_padding else ((t * 13 + t // 5) % vocab)))
        for j in range(width):
            dy.append(Float32((t + j) % 11 - 5) * Float32(0.125))
    var expected_counts = emb_counts(ids, cfg)
    var expected_begin = emb_run_begin(expected_counts)
    var expected_perm = emb_perm_by_scan(ids, cfg)
    var previous = List[Float32]()
    for i in range(vocab * width):
        previous.append(Float32(i - 10) * Float32(0.25))
    var reference = List[Float32]()
    var geometries: List[Int] = [32, 96, 160]
    for plan in range(2):
        for g in range(len(geometries)):
            var d_ids = _upload_i32(ctx, ids)
            var d_dy = _upload_f32(ctx, dy)
            var d_dw = _upload_f32(ctx, previous)
            var counts = _upload_i32(ctx, _zeros_i32_list(vocab))
            var begin = _upload_i32(ctx, _zeros_i32_list(vocab + 1))
            var poison = List[Int32]()
            for i in range(n + 3):
                poison.append(Int32(-12345))
            var perm = _upload_i32(ctx, poison)
            identical_embedding_backward_into(ctx, d_dw, d_dy, d_ids, counts, begin, perm, n, cfg, plan, geometries[g])
            ctx.synchronize()
            var got_counts = _download_i32(ctx, counts, vocab)
            var got_begin = _download_i32(ctx, begin, vocab + 1)
            var got_perm = _download_i32(ctx, perm, len(expected_perm))
            var got_tail = _download_i32(ctx, perm, n + 3)
            var got_dw = _download_f32(ctx, d_dw, vocab * width)
            if n > 0:
                if compare_i32("counts", expected_counts, got_counts, True).n_diff != 0 or compare_i32("begin", expected_begin, got_begin, True).n_diff != 0 or compare_i32("perm", expected_perm, got_perm, True).n_diff != 0:
                    raise Error("PLAN_SORT edge metadata mismatch")
            for i in range(len(expected_perm), n + 3):
                if got_tail[i] != Int32(-12345):
                    raise Error("PLAN_SORT wrote unused permutation tail")
            if plan == PLAN_SCAN and g == 0:
                reference = got_dw.copy()
            elif compare_f32("dw", reference, got_dw, True).n_diff != 0:
                raise Error("PLAN_SORT edge gradient mismatch")
            _ = d_ids^
            _ = d_dy^
            _ = d_dw^
            _ = counts^
            _ = begin^
            _ = perm^


def main() raises:
    var ctx = DeviceContext()
    var sizes: List[Int] = [0, 1, 3, 31, 33, 129, 257]
    for i in range(len(sizes)):
        for accumulate in range(2):
            check_case(ctx, sizes[i], -1, False, accumulate != 0)
            check_case(ctx, sizes[i], 0, False, accumulate != 0)
            check_case(ctx, sizes[i], 6, False, accumulate != 0)
            check_case(ctx, sizes[i], 3, True, accumulate != 0)
    print("PLAN_SORT edge gate PASS: 56 cases x 2 plans x 3 geometries, exact metadata/dw and untouched tail")
