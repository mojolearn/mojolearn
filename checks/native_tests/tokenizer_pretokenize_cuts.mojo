# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The pre-token cut of the three built-in patterns on hand-computed cases
(lane tests-prune-remaining-models, 2026-11-15; ported from
`test_models_loader.py::test_pretokenize_cuts_as_the_pattern_says`).

The entry is the one the tokenizer door's `vocab_cut` runs
(bindings/tokenizer_public_native_door.mojo): tokenizer/impl/pretokenize.mojo
`pretokenize_pattern(data, builtin_unicode_classes(), pattern)`, which returns
`m + 1` boundaries, first 0 and last `len(data)`. Each case is one literal byte
string, its pattern code (PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2) and the
boundaries of the pieces the Python test listed:

  0-13  llama3: "Hello world", "IT'S", "a  b", "12345", " 123", "x,\\n\\ny",
        "a \\n b", "-hello", "hi\\n", "a   ", "tab\\tx", "we're",
        b"caf\\xc3\\xa9 2024!", b"\\xff\\xfe"
  14-15 qwen2:  "12345", "Hello world"
  16-18 gpt2:   "IT'S", "a  b", "12345"

The pattern-name mapping of the public `mojolearn.models.tokenizer.pretokenize`
door keeps one case per pattern in the Python test (boundary).

Host column only: the tokenizer is a host component. The cases are
independent and run across host cores.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from tokenizer.encoding import string_bytes
from tokenizer.impl.pretokenize import PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2, pretokenize_pattern
from tokenizer.impl.unicode_class import builtin_unicode_classes


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)

comptime TOKENIZER_PRETOKENIZE_CASES = 19


def _ascii(text: String) -> List[UInt8]:
    return string_bytes(text)


def _cafe() -> List[UInt8]:
    """b"caf\\xc3\\xa9 2024!" (11 bytes)."""
    var out: List[UInt8] = [0x63, 0x61, 0x66, 0xC3, 0xA9, 0x20, 0x32, 0x30, 0x32, 0x34, 0x21]
    return out^


def _invalid() -> List[UInt8]:
    """b"\\xff\\xfe": two invalid UTF-8 lead bytes."""
    var out: List[UInt8] = [0xFF, 0xFE]
    return out^


def _case(index: Int, mut data: List[UInt8], mut pattern: Int) raises -> List[Int]:
    """The case's text and pattern into `data` and `pattern`; returns the
    expected boundaries."""
    pattern = PATTERN_LLAMA3
    if index == 0:
        data = _ascii("Hello world")
        return [0, 5, 11]
    if index == 1:
        data = _ascii("IT'S")
        return [0, 2, 4]
    if index == 2:
        data = _ascii("a  b")
        return [0, 1, 2, 4]
    if index == 3:
        data = _ascii("12345")
        return [0, 3, 5]
    if index == 4:
        data = _ascii(" 123")
        return [0, 1, 4]
    if index == 5:
        data = _ascii("x,\n\ny")
        return [0, 1, 4, 5]
    if index == 6:
        data = _ascii("a \n b")
        return [0, 1, 3, 5]
    if index == 7:
        data = _ascii("-hello")
        return [0, 6]
    if index == 8:
        data = _ascii("hi\n")
        return [0, 2, 3]
    if index == 9:
        data = _ascii("a   ")
        return [0, 1, 4]
    if index == 10:
        data = _ascii("tab\tx")
        return [0, 3, 5]
    if index == 11:
        data = _ascii("we're")
        return [0, 2, 5]
    if index == 12:
        data = _cafe()
        return [0, 5, 6, 9, 10, 11]
    if index == 13:
        data = _invalid()
        return [0, 1, 2]
    if index == 14:
        pattern = PATTERN_QWEN2
        data = _ascii("12345")
        return [0, 1, 2, 3, 4, 5]
    if index == 15:
        pattern = PATTERN_QWEN2
        data = _ascii("Hello world")
        return [0, 5, 11]
    pattern = PATTERN_GPT2
    if index == 16:
        data = _ascii("IT'S")
        return [0, 2, 3, 4]
    if index == 17:
        data = _ascii("a  b")
        return [0, 1, 2, 4]
    if index == 18:
        data = _ascii("12345")
        return [0, 5]
    raise Error("tokenizer_pretokenize_cuts: no case " + String(index))


def run_case(index: Int) raises:
    var data = List[UInt8]()
    var pattern = 0
    var want = _case(index, data, pattern)
    var classes = builtin_unicode_classes()
    var got = pretokenize_pattern(data, classes, pattern)
    var label = "pretokenize case " + String(index) + " (pattern " + String(pattern) + ")"
    require(len(got) == len(want), label + ": " + String(len(got) - 1) + " pieces, want " + String(len(want) - 1))
    for i in range(len(want)):  # small-loop(at most six boundaries)
        require(got[i] == want[i], label + ": boundary " + String(i) + " is " + String(got[i])
                + ", want " + String(want[i]))


def _host_fixture(unit: Int) raises:
    run_case(unit)


def test_tokenizer_pretokenize_cuts() raises:
    check_cases("tokenizer_pretokenize_cuts", TOKENIZER_PRETOKENIZE_CASES, _host_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
