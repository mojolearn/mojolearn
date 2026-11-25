# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`mojolearn.models.tokenizer.pretokenize(device="gpu")` through the GPU tokenizer image's native
entry (TOK_GPU_OP_VOCAB_CUT of core/tokenizer_gpu_entry_table.mojo, dispatched by
tokenizer/gpu/corpus_gpu.mojo `corpus_gpu_entry_dispatch`, the body of
`_mojolearn_x_tokenizer`'s `tokenizer_gpu_entry_v1`) against the host cut
(tokenizer/impl/pretokenize.mojo `pretokenize_pattern`, the device="cpu" side), offset for offset,
for the GPT-2, Llama 3 and Qwen 2 patterns (lane vdl-pretok, 2026-11-24).

The call goes the way the door's RUN goes: `entry_ref_call` over two inline request envelopes,
through a C entry with the binding's own signature that calls the same dispatch body. The fixture
mixes scripts (Latin with diacritics, Cyrillic, Greek, Arabic, CJK, emoji), digit runs of every
length, runs of spaces and tabs, LF, CR and CRLF, and one long text so the pointer jumping runs
many levels. The entry refuses a pattern code outside the three, on empty data too, and an
envelope of the wrong shape."""
from std.testing import TestSuite, assert_equal, assert_true

from core.entry_table import EntryRef, entry_ref_call
from core.native_descriptor_abi import NativeDescriptorError
from core.tokenizer_gpu_entry_table import TOK_GPU_OP_VOCAB_CUT
from tokenizer.encoding import string_bytes
from tokenizer.gpu.corpus_gpu import corpus_gpu_entry_dispatch
from tokenizer.impl.pretokenize import PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2, pretokenize_pattern
from tokenizer.impl.unicode_class import builtin_unicode_classes


def tokenizer_gpu_entry_check(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The image entry's signature over the image entry's body (bindings/_mojolearn_x_tokenizer.mojo
    `tokenizer_gpu_entry_v1`)."""
    return corpus_gpu_entry_dispatch(operation, aa, na, pp, np, ff, nf, result, context, sink)


def _entry() -> EntryRef:
    return EntryRef(tokenizer_gpu_entry_check, 0)


def _texts() -> List[List[UInt8]]:
    var texts: List[String] = [
        "Hello world, it's 2024! They'LL pay $1,234.56 (approx.)\n",
        "naïve café Ölçü — Привет, мир! Γειά σου κόσμε 123",
        "مرحبا بالعالم ٣٤٥ and 日本語のテキスト１２３ 中文 9",
        "emoji 😀😀 x🚀y  \t\t tabs\r\nCRLF\r\n\r\n  \n  end   \n",
        "1 12 123 1234 12345 123456 1234567 x9y88z007",
        "   leading spaces and trailing    ",
        "line one\nline two\n\n\nline five\r\rcr only\r",
        "'s 'S 'Ll ''s IT'S I'M you'Re WE'VE don't",
        "a",
        " ",
        "\n",
    ]
    var out = List[List[UInt8]]()
    for i in range(len(texts)):
        out.append(string_bytes(texts[i]))
    out.append(List[UInt8]())
    return out^


def _entry_cut(data: List[UInt8], pattern: Int, mut bounds: List[Int64]) raises -> Int:
    """The door's RUN: the bytes and the boundary room by address, n and the pattern code."""
    var n = len(data)
    var addresses: Array[Int, 2] = [Int(data.unsafe_ptr()), Int(bounds.unsafe_ptr())]
    var ints: Array[Int, 2] = [n, pattern]
    var floats: Array[Float64, 0] = []
    var count = entry_ref_call(_entry(), TOK_GPU_OP_VOCAB_CUT, addresses, ints, floats)
    _ = len(data)
    _ = len(bounds)
    return count


def _check(data: List[UInt8], pattern: Int) raises:
    var host = pretokenize_pattern(data, builtin_unicode_classes(), pattern)
    var n = len(data)
    var bounds = List[Int64](length=n + 1, fill=Int64(-1))
    var count = _entry_cut(data, pattern, bounds)
    var what = "pattern " + String(pattern) + ", " + String(n) + " bytes"
    assert_equal(count, len(host), "boundary count, " + what)
    for i in range(count):
        assert_equal(Int(bounds[i]), host[i], "boundary " + String(i) + ", " + what)
    _ = len(bounds)


def test_entry_cut_matches_the_host_cut() raises:
    var texts = _texts()
    var patterns: List[Int] = [PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2]
    for p in range(len(patterns)):
        for t in range(len(texts)):
            _check(texts[t], patterns[p])


def test_entry_cut_of_a_long_mixed_text_matches_the_host_cut() raises:
    var texts = _texts()
    var long = List[UInt8]()
    for _ in range(200):
        for t in range(len(texts)):
            for b in range(len(texts[t])):
                long.append(texts[t][b])
    var patterns: List[Int] = [PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2]
    for p in range(len(patterns)):
        _check(long, patterns[p])


def _refused(data: List[UInt8], pattern: Int, words: String) raises -> Bool:
    var bounds = List[Int64](length=len(data) + 1, fill=Int64(-1))
    var refused = False
    try:
        _ = _entry_cut(data, pattern, bounds)
    except problem:
        refused = String(problem).find(words) >= 0
    _ = len(bounds)
    return refused


def test_entry_refuses_a_bad_pattern() raises:
    var data = string_bytes("Hello world")
    var empty = List[UInt8]()
    assert_true(_refused(data, 3, "unknown pattern code 3"), "pattern 3 on data")
    assert_true(_refused(data, -1, "unknown pattern code -1"), "pattern -1 on data")
    assert_true(_refused(empty, 7, "unknown pattern code 7"), "pattern 7 on empty data")
    _ = len(data)
    _ = len(empty)


def test_entry_refuses_a_bad_envelope() raises:
    var data = string_bytes("Hello world")
    var bounds = List[Int64](length=len(data) + 1, fill=Int64(-1))
    var addresses: Array[Int, 1] = [Int(data.unsafe_ptr())]
    var ints: Array[Int, 2] = [len(data), PATTERN_GPT2]
    var floats: Array[Float64, 0] = []
    var refused = False
    try:
        _ = entry_ref_call(_entry(), TOK_GPU_OP_VOCAB_CUT, addresses, ints, floats)
    except problem:
        refused = String(problem).find("vocab cut takes 2 addresses and 2 words") >= 0
    assert_true(refused, "one address")
    _ = len(data)
    _ = len(bounds)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
