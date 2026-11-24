# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The identical GPU tokenizer (tokenizer/gpu/bpe_gpu.mojo) against the host column
(tokenizer/encoding.mojo), on a small fixture built for the pre-tokenizer's edge cases (lane
vdf-parallel, 2026-11-24):

* encode: every document alone and the documents as one batch, with allow_endoftext False and
  True: the GPU ids equal the host `encode_bytes` ids, id for id, and each document's count;
* decode: the host ids of every document, from a host buffer AND from a device-resident buffer
  (read in place): the GPU bytes equal the host `decode_bytes` bytes, and the two GPU inputs agree;
* an id outside [0, n_vocab) reports the first offending position;
* training: the GPU trainer's vocabulary (token bytes, merges, n_ties_broken, n_groups) equals the
  host `train_bpe` result on the same documents, at vocabulary sizes 256 to 70000 (above the
  old int32 pair-key limit 46340) and min_frequency 1 to 3.

The fixture covers contractions (`'s`, `'ll`, `IT'S`, `''s`, ` 's`), whitespace runs before words,
at the end and before invalid bytes, tabs and newlines, multi-byte letters and numbers, invalid and
truncated UTF-8, stray continuation bytes, `<|endoftext|>` at the start, the end, twice in a row and
split across documents, and empty documents."""
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext
from checks.native_tests.kit import upload_into

from tokenizer.encoding import BpeTokenizer, string_bytes
from tokenizer.impl.vocab_build import table_from_tokens
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.gpu.bpe_gpu import (
    BpeGpuClasses, BpeGpuTable, bpe_gpu_ctx, bpe_gpu_encode_batch_into, bpe_gpu_decode_batch_into, bpe_gpu_train_into,
)
from tokenizer.train.bpe_train import TrainedVocabulary, train_bpe


def _tokenizer() raises -> BpeTokenizer:
    var arena = List[UInt8]()
    var lengths = List[Int]()
    for b in range(256):
        arena.append(UInt8(b))
        lengths.append(1)
    var extra: List[String] = ["th", "the", " t", " the", "in", "er", "'s", "ll", "  ", " c", "at", " cat",
                               "é", "\t\t", "\n\n", "12", " 1", "IT", "''"]
    for w in range(len(extra)):
        var bs = string_bytes(extra[w])
        for x in range(len(bs)):
            arena.append(bs[x])
        lengths.append(len(bs))
    return BpeTokenizer(table_from_tokens(arena, lengths), builtin_unicode_classes())


def _documents() -> List[List[UInt8]]:
    var texts: List[String] = [
        "the cat's hat, they'll go; we've IT'S ''s 's  's",
        "a  b   c\n\n  in   ",
        "tab\t\tend \t x\r\n",
        "naïve café 日本語 １２３ 12 345",
        "<|endoftext|>the<|endoftext|><|endoftext|>cat",
        "",
        "   ",
        "end<|endoftext|>",
        "<|endof",
        "text|>tail",
        "x",
    ]
    var docs = List[List[UInt8]]()
    for i in range(len(texts)):
        docs.append(string_bytes(texts[i]))
    # Invalid UTF-8: a lone lead byte, a stray continuation, a truncated 3-byte form, an overlong,
    # a surrogate, spaces before an invalid byte.
    var raw: List[UInt8] = [0x61, 0x20, 0xFF, 0x62, 0x80, 0x80, 0x20, 0x20, 0xC3, 0x20, 0xE2, 0x82, 0x20, 0xC0, 0xAF,
                            0xED, 0xA0, 0x80, 0x27, 0x73, 0xE2, 0x82, 0xAC, 0x31]
    docs.append(raw^)
    return docs^


def _batch(docs: List[List[UInt8]], lo: Int, hi: Int, mut text: List[UInt8], mut offs: List[Int64]):
    text = List[UInt8]()
    offs = List[Int64]()
    offs.append(Int64(0))
    for d in range(lo, hi):
        for k in range(len(docs[d])):
            text.append(docs[d][k])
        offs.append(Int64(len(text)))


def _check_encode(tok: BpeTokenizer, table: BpeGpuTable, docs: List[List[UInt8]], lo: Int, hi: Int, allow: Bool) raises:
    var text = List[UInt8]()
    var offs = List[Int64]()
    _batch(docs, lo, hi, text, offs)
    var n = len(text)
    var ids = List[Int32](length=max(n, 1), fill=Int32(-1))
    var counts = List[Int64](length=max(hi - lo, 1), fill=Int64(-1))
    var total = bpe_gpu_encode_batch_into(table, Int(text.unsafe_ptr()), n, offs, allow, tok.eot_id(),
                                          Int(ids.unsafe_ptr()), Int(counts.unsafe_ptr()))
    _ = text^
    var at = 0
    for d in range(lo, hi):
        var host = tok.encode_bytes(docs[d], allow)
        assert_equal(Int(counts[d - lo]), len(host), "document " + String(d) + " id count, allow=" + String(allow))
        for k in range(len(host)):
            assert_equal(Int(ids[at + k]), host[k], "document " + String(d) + " id " + String(k))
        at += len(host)
    assert_equal(total, at, "total ids")


def test_encode_matches_the_host_column() raises:
    var tok = _tokenizer()
    var table = BpeGpuTable(bpe_gpu_ctx(), tok)
    var docs = _documents()
    for a in range(2):
        var allow = a == 1
        for d in range(len(docs)):
            _check_encode(tok, table, docs, d, d + 1, allow)
        _check_encode(tok, table, docs, 0, len(docs), allow)


def _decode(table: BpeGpuTable, tok: BpeTokenizer, ids: List[Int64], soffs: List[Int64], device: Bool,
            mut out: List[UInt8], mut lens: List[Int64]) raises -> Int:
    var nids = len(ids)
    var cap = nids * 64
    out = List[UInt8](length=max(cap, 1), fill=UInt8(0))
    lens = List[Int64](length=max(len(soffs) - 1, 1), fill=Int64(0))
    if not device:
        return bpe_gpu_decode_batch_into(table, Optional[MutPointer[Int64, MutAnyOrigin]](None), Int(ids.unsafe_ptr()),
            soffs, 64, tok.eot_id(), Int(out.unsafe_ptr()), cap, Int(lens.unsafe_ptr()))
    var ctx = bpe_gpu_ctx()
    var d_ids = ctx.enqueue_create_buffer[DType.int64](max(nids, 1))
    if nids > 0:
        upload_into(ctx, d_ids, ids)
    ctx.synchronize()
    var p = d_ids.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var total = bpe_gpu_decode_batch_into(table, Optional[MutPointer[Int64, MutAnyOrigin]](p), 0, soffs, 64,
        tok.eot_id(), Int(out.unsafe_ptr()), cap, Int(lens.unsafe_ptr()))
    _ = d_ids^
    return total


def test_decode_matches_the_host_column_from_host_and_device_ids() raises:
    var tok = _tokenizer()
    var table = BpeGpuTable(bpe_gpu_ctx(), tok)
    var docs = _documents()
    var ids = List[Int64]()
    var soffs = List[Int64]()
    soffs.append(Int64(0))
    var host_bytes = List[List[UInt8]]()
    for d in range(len(docs)):
        var host = tok.encode_bytes(docs[d], True)
        for k in range(len(host)):
            ids.append(Int64(host[k]))
        soffs.append(Int64(len(ids)))
        host_bytes.append(tok.decode_bytes(host))
    for which in range(2):
        var out = List[UInt8]()
        var lens = List[Int64]()
        var total = _decode(table, tok, ids, soffs, which == 1, out, lens)
        var at = 0
        for d in range(len(docs)):
            assert_equal(Int(lens[d]), len(host_bytes[d]), "sequence " + String(d) + " byte count")
            for k in range(len(host_bytes[d])):
                assert_equal(Int(out[at + k]), Int(host_bytes[d][k]), "sequence " + String(d) + " byte " + String(k))
            at += len(host_bytes[d])
        assert_equal(total, at, "total bytes, device input " + String(which == 1))


def test_decode_reports_the_first_bad_position() raises:
    var tok = _tokenizer()
    var table = BpeGpuTable(bpe_gpu_ctx(), tok)
    var ids: List[Int64] = [1, 2, 3, Int64(tok.n_vocab()), 5, -1]
    var soffs: List[Int64] = [0, 2, 6]
    for which in range(2):
        var out = List[UInt8]()
        var lens = List[Int64]()
        var result = _decode(table, tok, ids, soffs, which == 1, out, lens)
        assert_equal(result, -1 - 3, "first offending global position")


def _check_train(docs: List[List[UInt8]], vocab_size: Int, min_frequency: Int) raises:
    var host = train_bpe(docs, builtin_unicode_classes(), vocab_size, min_frequency)
    var text = List[UInt8]()
    var offs = List[Int64]()
    _batch(docs, 0, len(docs), text, offs)
    var classes = BpeGpuClasses(bpe_gpu_ctx(), builtin_unicode_classes())
    var gpu = bpe_gpu_train_into(classes, Int(text.unsafe_ptr()), len(text), offs, vocab_size, min_frequency)
    _ = text^
    var label = "vocab " + String(vocab_size) + " minf " + String(min_frequency)
    assert_equal(gpu.n_tokens(), host.n_tokens(), label + ": tokens")
    assert_equal(gpu.n_merges(), host.n_merges(), label + ": merges")
    assert_equal(gpu.n_ties_broken, host.n_ties_broken, label + ": ties broken")
    assert_equal(gpu.n_groups, host.n_groups, label + ": groups")
    for k in range(host.n_merges()):
        assert_equal(gpu.merge_left[k], host.merge_left[k], label + ": merge left " + String(k))
        assert_equal(gpu.merge_right[k], host.merge_right[k], label + ": merge right " + String(k))
    assert_equal(len(gpu.arena), len(host.arena), label + ": arena bytes")
    for k in range(host.n_tokens()):
        assert_equal(gpu.length[k], host.length[k], label + ": token length " + String(k))
    for k in range(len(host.arena)):
        assert_equal(Int(gpu.arena[k]), Int(host.arena[k]), label + ": arena byte " + String(k))


def test_train_matches_the_host_trainer() raises:
    var docs = _documents()
    # More text with repeated words so ties and min_frequency both bite.
    var extra: List[String] = ["the the the cat cat sat on the mat", "that's the cat's hat; the hat's red",
                               "low lower lowest newer wider", "aaaa aaaa bbbb ab ab ab"]
    for i in range(len(extra)):
        docs.append(string_bytes(extra[i]))
    _check_train(docs, 300, 1)
    _check_train(docs, 300, 2)
    _check_train(docs, 270, 3)
    _check_train(docs, 256, 1)
    # Just over the old int32 pair-key limit (46340): the pair is held as two ids at every size.
    _check_train(docs, 46341, 1)
    _check_train(docs, 70000, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
