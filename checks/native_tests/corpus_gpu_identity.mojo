# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The GPU side of `mojolearn.lm_corpus.prepare` (tokenizer/gpu/corpus_gpu.mojo) against the host
column (tokenizer/impl/tokenizer_host_shell_impl.mojo `bpe_train_corpus_fill`,
`bpe_encode_corpus_into`), lane vdf2-tok, 2026-11-24: the trained vocabulary (its rank text, merge
and tie counts) and the encoded corpus (every id, every split point's token offset, the document
count, the largest id and the ids above 255) must be the same.

The fixture: a corpus with split points inside it, documents limited to a few bytes so the cutter
pulls documents back to their last 0x0A, windows with no 0x0A at all, a document limit larger than
the corpus, multi-byte and invalid UTF-8, and a corpus long enough that the cutter's pointer
jumping runs many levels."""
from std.testing import TestSuite, assert_equal, assert_true

from tokenizer.encoding import BpeTokenizer, string_bytes
from tokenizer.impl.ranks import parse_rank_table
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.impl.tokenizer_host_shell_impl import (
    BpeHandle, BpeTrainedHandle, bpe_encode_corpus_into, bpe_train_corpus_fill,
)
from tokenizer.train.bpe_train import trained_vocab_from_pairs
from tokenizer.train.emit import render_ranks
from tokenizer.gpu.corpus_gpu import corpus_gpu_encode, corpus_gpu_train


def _corpus(repeats: Int) -> List[UInt8]:
    var parts: List[String] = [
        "the cat's hat sat on the mat.\nthe dog's log\n",
        "naïve café 日本語 12 345 1234567\n\n",
        "averyveryverylongwordwithnonewlineinsideitatall ",
        "Hello!!\n world...\r\n\r\n  \n  x   \n",
    ]
    var out = List[UInt8]()
    for _ in range(repeats):
        for i in range(len(parts)):
            var b = string_bytes(parts[i])
            for k in range(len(b)):
                out.append(b[k])
        out.append(0xFF)
        out.append(0x80)
    return out^


def _points(n: Int) -> List[Int]:
    var points: List[Int] = [0, n // 3, (2 * n) // 3 + 1, n]
    return points^


def _check(data: List[UInt8], document_bytes: Int, vocab_size: Int) raises:
    var n = len(data)
    var points = _points(n)
    var lo = points[1]
    var hi = points[3]
    var width = min(document_bytes, max(hi - lo, 1))
    # Training: host column, then the device.
    var host = BpeTrainedHandle()
    bpe_train_corpus_fill(host, Int(data.unsafe_ptr()), points, lo, hi, width, vocab_size, 1, False)
    var cuts = List[Int64]()
    cuts.append(Int64(0))
    for k in range(len(points)):
        if points[k] > lo and points[k] < hi:
            cuts.append(Int64(points[k] - lo))
    cuts.append(Int64(hi - lo))
    var cap = max(2 * (vocab_size - 256), 2)
    var pairs = List[Int32](length=cap, fill=Int32(0))
    var words = List[Int64](length=3, fill=Int64(0))
    var n_docs = corpus_gpu_train(Int(data.unsafe_ptr()) + lo, hi - lo, cuts, width, vocab_size, 1,
                                  Int(pairs.unsafe_ptr()), cap, Int(words.unsafe_ptr()))
    assert_true(n_docs >= 1, "the device cut found no document")
    var device = trained_vocab_from_pairs(
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(pairs.unsafe_ptr())), Int(words[0]),
        Int(words[1]), Int(words[2]))
    ref expected = host.vocab.value()
    assert_equal(device.n_merges(), expected.n_merges(), "merge count, width " + String(width))
    assert_equal(device.n_ties_broken, expected.n_ties_broken, "tie count, width " + String(width))
    var text = render_ranks(expected)
    assert_equal(render_ranks(device), text, "rank text, width " + String(width))
    # Encoding the whole corpus with that vocabulary.
    var handle = BpeHandle()
    handle.tok = BpeTokenizer(parse_rank_table(text, "<rank text>"), builtin_unicode_classes())
    var enc_width = min(document_bytes, max(n, 1))
    var ids_h = List[Int32](length=max(n, 1), fill=Int32(0))
    var bounds_h = List[Int64](length=len(points), fill=Int64(0))
    var stats_h = List[Int64](length=3, fill=Int64(0))
    var total_h = bpe_encode_corpus_into(handle, Int(data.unsafe_ptr()), points, n, len(points), enc_width, n,
                                         Int(ids_h.unsafe_ptr()), Int(bounds_h.unsafe_ptr()), Int(stats_h.unsafe_ptr()))
    var marks = List[Int64]()
    for k in range(len(points)):
        marks.append(Int64(points[k]))
    var ids_d = List[Int32](length=max(n, 1), fill=Int32(-7))
    var bounds_d = List[Int64](length=len(points), fill=Int64(-7))
    var stats_d = List[Int64](length=3, fill=Int64(-7))
    var total_d = corpus_gpu_encode(Int(data.unsafe_ptr()), n, marks, enc_width, Int(text.unsafe_ptr()),
                                    text.byte_length(), Int(ids_d.unsafe_ptr()), Int(bounds_d.unsafe_ptr()),
                                    Int(stats_d.unsafe_ptr()))
    assert_equal(total_d, total_h, "id count, width " + String(enc_width))
    for i in range(total_h):
        assert_equal(Int(ids_d[i]), Int(ids_h[i]), "id " + String(i) + ", width " + String(enc_width))
    for j in range(len(points)):
        assert_equal(Int(bounds_d[j]), Int(bounds_h[j]), "point " + String(j) + " token offset")
    for j in range(3):
        assert_equal(Int(stats_d[j]), Int(stats_h[j]), "statistic " + String(j))
    _ = len(data)
    _ = text.byte_length()


def test_short_documents_match_the_host_column() raises:
    var data = _corpus(6)
    var widths: List[Int] = [1, 7, 23, 64]
    for w in range(len(widths)):
        _check(data, widths[w], 300)


def test_a_limit_beyond_the_corpus_matches_the_host_column() raises:
    _check(_corpus(3), 1 << 40, 290)


def test_only_single_bytes_match_the_host_column() raises:
    _check(_corpus(2), 16, 256)


def test_a_long_corpus_matches_the_host_column() raises:
    _check(_corpus(400), 512, 400)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
