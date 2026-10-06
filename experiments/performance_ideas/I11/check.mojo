# SPDX-License-Identifier: Apache-2.0
"""Radix grouping versus scan on all-identical, mostly-unique, and skewed
IDs, two vocabularies, two widths, padding and dirty dense output.
Exact counts, run boundaries, stable original positions, dense gradients,
and unwritten permutation tails are independently verified."""
from max.gpu.host import DeviceContext
from embedding.checks.embedding_check import _upload_i32, _upload_f32, _download_i32, _download_f32, _zeros_i32_list, compare_i32, compare_f32
from embedding.checks.embedding_identical import identical_embedding_backward_into, emb_sabotage_name
from embedding.checks.embedding_sort import PLAN_SCAN, PLAN_SORT
from embedding.checks.embedding_oracle import EmbConfig, emb_counts, emb_run_begin, emb_perm_by_scan


def check_case(ctx: DeviceContext, n: Int, vocab: Int, width: Int, skew: Int, padding: Int, all_padding: Bool, accumulate: Bool) raises:
    var cfg = EmbConfig(vocab, width, padding, accumulate)
    var ids = List[Int32]()
    var dy = List[Float32]()
    for t in range(n):
        ids.append(Int32(padding if all_padding else (0 if skew == 0 else (t if skew == 1 else t * 13 + t // 5) % vocab)))
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
    for vocab in [7, 131, 7]:
        for width in [3, 17]:
            for skew in range(3):
                check_case(ctx, 263, vocab, width, skew, vocab-1, False, False)
                check_case(ctx, 263, vocab, width, skew, vocab-1, False, True)
    print("I11 PASS alternating_vocab=3 widths=2 skews=3 accumulation=2 plans=2 geometries=3")
