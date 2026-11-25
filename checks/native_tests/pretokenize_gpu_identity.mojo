# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The GPU pre-tokenizer cut (tokenizer/gpu/pretokenize_gpu.mojo, the `device="gpu"` side of
`mojolearn.models.tokenizer.pretokenize`) against the host column
(tokenizer/impl/pretokenize.mojo `pretokenize_pattern`), boundary for boundary, for the GPT-2,
Llama 3 and Qwen 2 patterns (lane vdf2-public, 2026-11-24).

The fixture covers every alternative of the three patterns: contractions in both cases (`'s`,
`'LL`, `IT'S`, `''s`, U+017F after an apostrophe), an optional non-letter before a letter run,
digit runs longer than three, punctuation runs with trailing CR/LF, whitespace runs through their
last CR/LF, `\\s+(?!\\S)` before a word and at the end, tabs, multi-byte letters, numbers and
spaces, invalid and truncated UTF-8, stray continuation bytes, empty and one-byte data; and one
long text (the fixture repeated) so the pointer jumping runs many levels."""
from std.testing import TestSuite, assert_equal

from tokenizer.encoding import string_bytes
from tokenizer.impl.pretokenize import PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2, pretokenize_pattern
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.gpu.pretokenize_gpu import pretokenize_gpu_into


def _texts() -> List[List[UInt8]]:
    var texts: List[String] = [
        "the cat's hat, they'll go; we've IT'S ''s 's  's I'M you'Re WE'VE",
        "a  b   c\n\n  in   ",
        "tab\t\tend \t x\r\n",
        "naïve café 日本語 １２３ 12 345 1234567 x9y88z",
        "Hello!!\n\n world...\r\n\r\n  \n  x   \n",
        "'ſ 'ſt ''' ' '  'x 'll'll",
        "   ",
        "x",
        "\n",
        "price: $1,234.56 (approx.)\t\r\n--\n",
    ]
    var out = List[List[UInt8]]()
    for i in range(len(texts)):
        out.append(string_bytes(texts[i]))
    # invalid and truncated UTF-8, stray continuation bytes
    var bad: List[UInt8] = [0x61, 0xFF, 0x20, 0x80, 0x80, 0x62, 0xC3, 0x20, 0xE6, 0x97, 0x20, 0xF0, 0x9F, 0x98]
    out.append(bad^)
    out.append(List[UInt8]())
    return out^


def _check(data: List[UInt8], pattern: Int) raises:
    var host = pretokenize_pattern(data, builtin_unicode_classes(), pattern)
    var n = len(data)
    var bounds = List[Int64](length=n + 1, fill=Int64(-1))
    var count = pretokenize_gpu_into(Int(data.unsafe_ptr()), n, pattern, Int(bounds.unsafe_ptr()))
    assert_equal(count, len(host), "boundary count, pattern " + String(pattern) + ", " + String(n) + " bytes")
    for i in range(count):
        assert_equal(Int(bounds[i]), host[i], "boundary " + String(i) + ", pattern " + String(pattern))


def test_every_pattern_matches_the_host_column() raises:
    var texts = _texts()
    var patterns: List[Int] = [PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2]
    for p in range(len(patterns)):
        for t in range(len(texts)):
            _check(texts[t], patterns[p])


def test_a_long_text_matches_the_host_column() raises:
    var texts = _texts()
    var long = List[UInt8]()
    for _ in range(300):
        for t in range(len(texts)):
            for b in range(len(texts[t])):
                long.append(texts[t][b])
    var patterns: List[Int] = [PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2]
    for p in range(len(patterns)):
        _check(long, patterns[p])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
