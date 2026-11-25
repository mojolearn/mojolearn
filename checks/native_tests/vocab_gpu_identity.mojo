# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`mojolearn.models.Tokenizer` on the GPU (tokenizer/gpu/vocab_gpu.mojo, the `device="gpu"` side)
against the host column (tokenizer/vocab.mojo `VocabTokenizer.encode` / `append_decoded`), id for id
and byte for byte, for the GPT-2, Llama 3 and Qwen 2 patterns, with and without the added-token
split (lane vdf2-tokmodel, 2026-11-24).

The vocabulary is the 256 single bytes plus merged tokens (so the per-piece merge runs); the added
tokens overlap on purpose: `<|endoftext|>` and its prefix `<|end`, `<|e`, a self-overlapping `aa`
(`aaaaa`: leftmost first), and a token holding a newline and a space (it cuts whitespace runs and
contractions, so the Llama 3 / Qwen 2 cut must stop at each segment's end). The texts cover the
pre-tokenizer fixture of pretokenize_gpu_identity.mojo plus tokens at the start, the end, adjacent
to each other and inside words; one long text runs many pointer-jumping levels. Decode covers
added ids, rank ids, and the first bad id's position."""
from std.testing import TestSuite, assert_equal, assert_true

from tokenizer.encoding import string_bytes
from tokenizer.impl.pretokenize import PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.impl.vocab_build import table_from_tokens
from tokenizer.vocab import VocabTokenizer
from tokenizer.gpu.vocab_gpu import vocab_gpu_decode, vocab_gpu_encode, vocab_gpu_free, vocab_gpu_load

comptime IntP = MutPointer[Int, MutUntrackedOrigin]


def _tokenizer(pattern: Int) raises -> VocabTokenizer:
    var arena = List[UInt8]()
    var lengths = List[Int]()
    for b in range(256):
        arena.append(UInt8(b))
        lengths.append(1)
    var merged: List[String] = ["th", "he", "the", " t", " the", "in", "ing", " c", "at", " cat", "'s", "ll",
        "er", "aa", "  ", "\n\n", "12", "é", "日本"]
    for i in range(len(merged)):
        var bytes = string_bytes(merged[i])
        for k in range(len(bytes)):
            arena.append(bytes[k])
        lengths.append(len(bytes))
    var ranks = table_from_tokens(arena, lengths)
    var top = ranks.n_tokens()
    var tok = VocabTokenizer(ranks^, builtin_unicode_classes(), pattern)
    var specials: List[String] = ["<|endoftext|>", "<|end", "<|e", "aa", "\n <sep>"]
    var sp_arena = List[UInt8]()
    var sp_lengths = List[Int]()
    var sp_ids = List[Int]()
    for i in range(len(specials)):
        var bytes = string_bytes(specials[i])
        for k in range(len(bytes)):
            sp_arena.append(bytes[k])
        sp_lengths.append(len(bytes))
        sp_ids.append(top + 3 + i)
    tok.set_specials(sp_arena, sp_lengths, sp_ids)
    return tok^


def _load(tok: VocabTokenizer) raises -> Int:
    ref ranks = tok.ranks
    var a: List[Int] = [
        Int(ranks.arena.unsafe_ptr()), Int(ranks.offset.unsafe_ptr()), Int(ranks.length.unsafe_ptr()),
        Int(ranks.buckets.unsafe_ptr()), Int(tok.sp_arena.unsafe_ptr()), Int(tok.sp_offset.unsafe_ptr()),
        Int(tok.sp_length.unsafe_ptr()), Int(tok.sp_id.unsafe_ptr()), Int(tok.added_of.unsafe_ptr()),
    ]
    var p: List[Int] = [
        len(ranks.arena), len(ranks.offset), len(ranks.buckets), ranks.mask, len(tok.sp_arena), len(tok.sp_id),
        len(tok.added_of), tok.pattern,
    ]
    var handle = vocab_gpu_load(IntP(unsafe_from_address=Int(a.unsafe_ptr())), IntP(unsafe_from_address=Int(p.unsafe_ptr())))
    _ = len(a)
    _ = len(p)
    return handle


def _texts() -> List[List[UInt8]]:
    var texts: List[String] = [
        "the cat's hat, they'll go; we've IT'S ''s 's  's I'M you'Re WE'VE",
        "a  b   c\n\n  in   ",
        "tab\t\tend \t x\r\n",
        "naïve café 日本語 １２３ 12 345 1234567 x9y88z",
        "Hello!!\n\n world...\r\n\r\n  \n  x   \n",
        "<|endoftext|>the cat<|endoftext|><|endoftext|> the<|end <|e<|endof",
        "aaaaa baaab aa",
        "words  \n <sep>  more\n <sep>\n <sep>'s x'\n <sep>ll",
        "end with a token<|endoftext|>",
        "<|e",
        "   ",
        "x",
        "\n",
    ]
    var out = List[List[UInt8]]()
    for i in range(len(texts)):
        out.append(string_bytes(texts[i]))
    var bad: List[UInt8] = [0x61, 0xFF, 0x20, 0x80, 0x80, 0x62, 0xC3, 0x20, 0xE6, 0x97, 0x20, 0xF0, 0x9F, 0x98]
    out.append(bad^)
    out.append(List[UInt8]())
    return out^


def _check(tok: VocabTokenizer, handle: Int, data: List[UInt8], allow: Bool) raises:
    var host = tok.encode(data, allow)
    var n = len(data)
    var ids = List[Int32](length=max(n, 1), fill=Int32(-1))
    var count = vocab_gpu_encode(handle, Int(data.unsafe_ptr()), n, allow, Int(ids.unsafe_ptr()))
    var what = "pattern " + String(tok.pattern) + ", allow " + String(allow) + ", " + String(n) + " bytes"
    assert_equal(count, len(host), "id count, " + what)
    for i in range(count):
        assert_equal(Int(ids[i]), host[i], "id " + String(i) + ", " + what)
    # decode: the host's bytes for the same ids
    var wide = List[Int](capacity=count)
    for i in range(count):
        wide.append(host[i])
    var expect = List[UInt8]()
    for k in range(len(wide)):
        tok.append_decoded(expect, wide[k], k)
    var cap = max(len(wide) * tok.max_token_bytes(), 1)
    var out = List[UInt8](length=cap, fill=UInt8(0))
    var total = vocab_gpu_decode(handle, Int(wide.unsafe_ptr()), len(wide), Int(out.unsafe_ptr()), cap)
    assert_equal(total, len(expect), "decoded bytes, " + what)
    for i in range(total):
        assert_equal(Int(out[i]), Int(expect[i]), "decoded byte " + String(i) + ", " + what)
    _ = len(ids)


def test_every_pattern_matches_the_host_column() raises:
    var texts = _texts()
    var patterns: List[Int] = [PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2]
    for p in range(len(patterns)):
        var tok = _tokenizer(patterns[p])
        var handle = _load(tok)
        for t in range(len(texts)):
            _check(tok, handle, texts[t], False)
            _check(tok, handle, texts[t], True)
        vocab_gpu_free(handle)


def test_a_long_text_matches_the_host_column() raises:
    var texts = _texts()
    var long = List[UInt8]()
    for _ in range(200):
        for t in range(len(texts)):
            for b in range(len(texts[t])):
                long.append(texts[t][b])
    var patterns: List[Int] = [PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2]
    for p in range(len(patterns)):
        var tok = _tokenizer(patterns[p])
        var handle = _load(tok)
        _check(tok, handle, long, False)
        _check(tok, handle, long, True)
        vocab_gpu_free(handle)


def test_decode_refuses_the_first_bad_id() raises:
    var tok = _tokenizer(PATTERN_GPT2)
    var handle = _load(tok)
    var bad: List[Int] = [5, tok.n_ranks() + 1, 7, -3, tok.n_vocab() + 9]
    var out = List[UInt8](length=64, fill=UInt8(0))
    var result = vocab_gpu_decode(handle, Int(bad.unsafe_ptr()), len(bad), Int(out.unsafe_ptr()), 64)
    # n_ranks + 1 is neither a rank nor an added id (the added ids start at n_ranks + 3)
    assert_equal(result, -1 - 1, "first bad id position")
    var expect_raises = False
    try:
        var host = List[UInt8]()
        for k in range(len(bad)):
            tok.append_decoded(host, bad[k], k)
    except error:
        expect_raises = String(error).startswith("ValueError: id ")
    assert_true(expect_raises, "the host refuses the same ids")
    vocab_gpu_free(handle)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
