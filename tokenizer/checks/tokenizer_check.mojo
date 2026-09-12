# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE GATE: our GPT-2 encoder against tiktoken 0.14.0, id for id.

    pixi run check-tokenizer

`checks/fixtures/gpt2_reference.json` holds 43 cases recorded from tiktoken
0.14.0's `gpt2` encoding. For every case this asserts:

    * EXACT id-sequence equality. Same length, same ids, same order. There is
      no tolerance to loosen and no case to skip -- a tokenizer that is
      nearly right is wrong, because the ids are indices into someone's
      embedding table. A failure PRINTS the case, the first differing
      position, both id sequences and the pre-token split, which is what
      tells you whether the pattern or the merge loop is at fault.
    * decode(encode(text)) == text, BYTE FOR BYTE, including the case whose
      text holds NUL.

plus the table and pattern preconditions that make the comparison meaningful:

    check_pattern_matches_fixture   the fixture's own `pat_str` equals the
                                    pattern `impl/pretokenize.mojo` claims to
                                    implement. A fixture regenerated from a
                                    different pattern fails HERE rather than
                                    silently redefining the target
    check_tables                    50256 ranks, ascending, all 256 single
                                    bytes present, n_vocab 50257 with
                                    `<|endoftext|>`
    check_byte_unicode_bijection    256 distinct codepoints, both directions,
                                    the three fixed runs and the 68 remapped
                                    bytes at U+0100..U+0143
    check_pattern_reach             the alternatives that are easy to write
                                    and never exercise: `\\s++$` vs
                                    `\\s+(?!\\S)` on the same run, the
                                    lowercase-only contraction set, and the
                                    special token under both flag values

NOT a claim about vendors. Nothing in `tokenizer/` touches a device or a
float, so there is nothing to be identical about across GPUs; this gate is
about agreeing with the reference implementation, which is the property that
can actually be wrong.
"""

from tokenizer.checks.json_lite import JsonFixture, load_fixture
from tokenizer.encoding import (
    GPT2_ENDOFTEXT,
    GPT2_ENDOFTEXT_ID,
    GPT2_N_VOCAB,
    GPT2_PAT_STR,
    Gpt2Tokenizer,
    load_gpt2_tokenizer,
    string_bytes,
)
from tokenizer.impl.byte_unicode import (
    byte_to_codepoint,
    codepoint_to_byte,
    spell_bytes,
)
from tokenizer.impl.pretokenize import pretokenize

comptime FIXTURE = "tokenizer/checks/fixtures/gpt2_reference.json"


def _ids_string(ids: List[Int]) -> String:
    var s = String("[")
    for k in range(len(ids)):
        if k > 0:
            s += ", "
        s += String(ids[k])
    s += "]"
    return s^


def _bytes_hex(data: List[UInt8]) -> String:
    comptime DIGITS = "0123456789abcdef"
    var d = String(DIGITS)
    var s = String("")
    for i in range(len(data)):
        s += String(d[byte = Int(data[i]) >> 4])
        s += String(d[byte = Int(data[i]) & 15])
    return s^


def _split_string(
    tok: Gpt2Tokenizer, data: List[UInt8]
) raises -> String:
    """The pre-token split, each piece in GPT-2's printable spelling between
    pipes. This is the first thing to read when ids disagree."""
    var bounds = pretokenize(data, tok.classes)
    var s = String("")
    for k in range(len(bounds) - 1):
        s += "|"
        s += spell_bytes(data, bounds[k], bounds[k + 1] - bounds[k])
    s += "|"
    return s^


def check_pattern_matches_fixture(fx: JsonFixture) raises -> Int:
    if fx.pat_str != String(GPT2_PAT_STR):
        print("FAIL pattern: the fixture was produced from a DIFFERENT pattern")
        print("  fixture: " + fx.pat_str)
        print("  ours   : " + String(GPT2_PAT_STR))
        return 1
    if fx.n_vocab != GPT2_N_VOCAB:
        print(
            "FAIL n_vocab: fixture says",
            fx.n_vocab,
            "and encoding.mojo says",
            GPT2_N_VOCAB,
        )
        return 1
    print(
        "  pattern and n_vocab agree with the fixture ("
        + fx.encoding
        + ", tiktoken "
        + fx.tiktoken_version
        + ")"
    )
    return 0


def check_tables(tok: Gpt2Tokenizer) raises -> Int:
    var bad = 0
    if tok.ranks.n_tokens() != GPT2_N_VOCAB - 1:
        print(
            "FAIL ranks: table holds",
            tok.ranks.n_tokens(),
            "tokens, expected",
            GPT2_N_VOCAB - 1,
        )
        bad += 1
    if tok.n_vocab() != GPT2_N_VOCAB:
        print("FAIL n_vocab:", tok.n_vocab())
        bad += 1

    # Every single byte has to be a token or the merge loop can dead-end.
    var one = List[UInt8]()
    one.append(UInt8(0))
    var missing = 0
    for b in range(256):
        one[0] = UInt8(b)
        if tok.ranks.rank(one, 0, 1) < 0:
            missing += 1
    if missing != 0:
        print("FAIL ranks:", missing, "of the 256 single-byte tokens missing")
        bad += 1

    # A rank probe has to answer -1 rather than 0 for an absent key: rank 0 is
    # a real token ('!'), so a table that returned 0 for "not found" would
    # merge everything into it.
    var absent = string_bytes(String("\xff\xfe\xfd\xfc"))
    if tok.ranks.rank(absent, 0, 4) >= 0:
        print("FAIL ranks: a key that is not in the table was found")
        bad += 1
    if bad == 0:
        print(
            "  tables: "
            + String(tok.ranks.n_tokens())
            + " ranks, all 256 byte tokens present, n_vocab "
            + String(tok.n_vocab())
            + " with "
            + String(GPT2_ENDOFTEXT)
            + " = "
            + String(GPT2_ENDOFTEXT_ID)
        )
        print("  " + tok.classes.source_header)
    return bad


def check_byte_unicode_bijection() raises -> Int:
    var bad = 0
    var fwd = byte_to_codepoint()
    var rev = codepoint_to_byte()
    if len(fwd) != 256:
        print("FAIL bijection: forward table is not 256 entries")
        return 1

    var seen = 0
    for b in range(256):
        var cp = fwd[b]
        if cp < 0 or cp >= len(rev) or rev[cp] != b:
            print("FAIL bijection: byte", b, "-> cp", cp, "does not come back")
            bad += 1
        else:
            seen += 1
    if seen != 256:
        bad += 1

    # The three fixed runs are fixed points, and nothing else is.
    var fixed = 0
    for b in range(256):
        if fwd[b] == b:
            fixed += 1
    var expect_fixed = (0x7F - 0x21) + (0xAD - 0xA1) + (0x100 - 0xAE)
    if fixed != expect_fixed:
        print(
            "FAIL bijection:",
            fixed,
            "fixed points, expected",
            expect_fixed,
        )
        bad += 1

    # The 68 remapped bytes land on U+0100..U+0143, ascending by byte value.
    var next_expected = 256
    for b in range(256):
        if fwd[b] != b:
            if fwd[b] != next_expected:
                print(
                    "FAIL bijection: byte",
                    b,
                    "maps to",
                    fwd[b],
                    "not",
                    next_expected,
                )
                bad += 1
            next_expected += 1
    if next_expected != 256 + 68:
        print("FAIL bijection: remapped", next_expected - 256, "bytes, not 68")
        bad += 1

    # Space is 0x20, so it is remapped, and its spelling is U+0120 -- the
    # 'Ġ' that every published GPT-2 vocabulary file shows.
    var sp = string_bytes(String(" a"))
    var spelled = spell_bytes(sp, 0, 2)
    if spelled != "Ġa":
        print("FAIL bijection: ' a' spells as '" + spelled + "', not 'Ġa'")
        bad += 1
    if bad == 0:
        print(
            "  bijection: 256 distinct codepoints, "
            + String(expect_fixed)
            + " fixed points, 68 remapped to U+0100..U+0143, ' ' spells Ġ"
        )
    return bad


def check_pattern_reach(tok: Gpt2Tokenizer) raises -> Int:
    """The alternatives whose absence would still pass a careless test.

    Each pair below differs ONLY in the alternative named, so a
    pre-tokenizer that dropped it would fail here rather than in a case
    nobody reads.
    """
    var bad = 0

    # `\s++$` (whole trailing run, one pre-token) against `\s+(?!\S)` (run
    # minus its last codepoint) on the SAME run of three spaces.
    var tail = _split_string(tok, string_bytes(String("a   ")))
    var interior = _split_string(tok, string_bytes(String("a   b")))
    if tail != "|a|ĠĠĠ|":
        print("FAIL reach: trailing `\\s++$` split is " + tail)
        bad += 1
    if interior != "|a|ĠĠ|Ġb|":
        print("FAIL reach: interior `\\s+(?!\\S)` split is " + interior)
        bad += 1

    # A single whitespace codepoint followed by a non-space takes
    # alternative 7 and nothing else can.
    var single = _split_string(tok, string_bytes(String("a b")))
    if single != "|a|Ġb|":
        print("FAIL reach: single-space split is " + single)
        bad += 1

    # The contraction alternative is LOWERCASE ONLY.
    var lower = _split_string(tok, string_bytes(String("it's")))
    var upper = _split_string(tok, string_bytes(String("IT'S")))
    if lower != "|it|'s|":
        print("FAIL reach: lowercase contraction split is " + lower)
        bad += 1
    if upper != "|IT|'|S|":
        print("FAIL reach: uppercase contraction split is " + upper)
        bad += 1

    # `\p{N}` is not `[0-9]`, `\p{L}` is not `[A-Za-z]`, and a combining
    # mark is in NEITHER -- the three ways a range table can be wrong.
    if not tok.classes.is_number(0x0661):  # ARABIC-INDIC DIGIT ONE
        print("FAIL reach: U+0661 is not in \\p{N}")
        bad += 1
    if not tok.classes.is_letter(0x4E2D):  # CJK ideograph
        print("FAIL reach: U+4E2D is not in \\p{L}")
        bad += 1
    if tok.classes.is_letter(0x0301):  # COMBINING ACUTE ACCENT
        print("FAIL reach: U+0301 (Mn) counts as a letter")
        bad += 1
    if tok.classes.is_space(0x001C):  # Cc, White_Space=No
        print("FAIL reach: U+001C counts as whitespace (Python isspace bug)")
        bad += 1
    if not tok.classes.is_space(0x00A0):  # NBSP, White_Space=Yes
        print("FAIL reach: U+00A0 is not whitespace")
        bad += 1

    # The special token under both flag values, from the same input.
    var as_text = tok.encode(String(GPT2_ENDOFTEXT), False)
    var as_special = tok.encode(String(GPT2_ENDOFTEXT), True)
    if len(as_special) != 1 or as_special[0] != GPT2_ENDOFTEXT_ID:
        print("FAIL reach: allowed special did not give one id 50256")
        bad += 1
    if len(as_text) < 2:
        print("FAIL reach: disallowed special collapsed to one token anyway")
        bad += 1

    # A special token in the MIDDLE, with text on both sides, and the
    # segments encoded independently.
    var mixed = tok.encode(
        String("hi ") + String(GPT2_ENDOFTEXT) + String(" there"), True
    )
    var seen_eot = 0
    for k in range(len(mixed)):
        if mixed[k] == GPT2_ENDOFTEXT_ID:
            seen_eot += 1
    if seen_eot != 1 or len(mixed) < 3:
        print("FAIL reach: interior special token gave " + _ids_string(mixed))
        bad += 1
    if bad == 0:
        print(
            "  reach: both whitespace alternatives, the single-space arm, the"
            " lowercase-only contraction set, three category-table traps, and"
            " the special token in all three positions"
        )
    return bad


def main() raises:
    print("tokenizer_check: GPT-2 byte-level BPE against " + String(FIXTURE))
    var tok = load_gpt2_tokenizer()
    var fx = load_fixture(String(FIXTURE))

    var bad = 0
    bad += check_pattern_matches_fixture(fx)
    bad += check_tables(tok)
    bad += check_byte_unicode_bijection()
    bad += check_pattern_reach(tok)

    var n_ids_wrong = 0
    var n_roundtrip_wrong = 0
    for ci in range(len(fx.cases)):
        var fxcase = fx.cases[ci].copy()
        # `endoftext_as_special` is the one case recorded with the special
        # token ALLOWED; every other case, `<|endoftext|>` included, is
        # recorded as ordinary text.
        var allow = fxcase.name == "endoftext_as_special"
        var got = tok.encode_bytes(fxcase.text, allow)

        var same = len(got) == len(fxcase.ids)
        var first_diff = -1
        if same:
            for k in range(len(got)):
                if got[k] != fxcase.ids[k]:
                    same = False
                    first_diff = k
                    break
        elif len(got) > 0 and len(fxcase.ids) > 0:
            var m = len(got) if len(got) < len(fxcase.ids) else len(fxcase.ids)
            for k in range(m):
                if got[k] != fxcase.ids[k]:
                    first_diff = k
                    break
            if first_diff < 0:
                first_diff = m

        if not same:
            n_ids_wrong += 1
            print("FAIL ids " + fxcase.name)
            print("    text hex   " + _bytes_hex(fxcase.text))
            print("    split      " + _split_string(tok, fxcase.text))
            print("    want       " + _ids_string(fxcase.ids))
            print("    got        " + _ids_string(got))
            print(
                "    lengths    want "
                + String(len(fxcase.ids))
                + ", got "
                + String(len(got))
                + "; first difference at position "
                + String(first_diff)
            )
            if first_diff >= 0:
                if first_diff < len(fxcase.ids):
                    print(
                        "      want id "
                        + String(fxcase.ids[first_diff])
                        + " = '"
                        + tok.token_spelling(fxcase.ids[first_diff])
                        + "'"
                    )
                if first_diff < len(got):
                    print(
                        "      got  id "
                        + String(got[first_diff])
                        + " = '"
                        + tok.token_spelling(got[first_diff])
                        + "'"
                    )

        # The round trip is over the ids WE produced when they match, and
        # over the fixture's ids when they do not, so a decode bug cannot
        # hide behind an encode bug.
        var ids_for_decode: List[Int]
        if same:
            ids_for_decode = got.copy()
        else:
            ids_for_decode = fxcase.ids.copy()
        var back = tok.decode_bytes(ids_for_decode)
        var rt_ok = len(back) == len(fxcase.text)
        if rt_ok:
            for k in range(len(back)):
                if back[k] != fxcase.text[k]:
                    rt_ok = False
                    break
        if not rt_ok:
            n_roundtrip_wrong += 1
            print("FAIL roundtrip " + fxcase.name)
            print("    want hex " + _bytes_hex(fxcase.text))
            print("    got  hex " + _bytes_hex(back))
        if not fxcase.roundtrip_ok and rt_ok:
            print(
                "NOTE "
                + fxcase.name
                + ": the fixture records roundtrip_ok false and ours round"
                " trips; the fixture is the reference, so this is a fixture"
                " question, not a pass"
            )

    print(
        "cases: "
        + String(len(fx.cases) - n_ids_wrong)
        + "/"
        + String(len(fx.cases))
        + " exact id sequences, "
        + String(len(fx.cases) - n_roundtrip_wrong)
        + "/"
        + String(len(fx.cases))
        + " byte-exact decode round trips"
    )
    bad += n_ids_wrong + n_roundtrip_wrong
    if bad != 0:
        raise Error(
            "tokenizer_check: "
            + String(bad)
            + " failures. Exact equality with tiktoken 0.14.0 is the whole"
            " point of this gate: fix the tokenizer, do not loosen the"
            " assertion."
        )
    print("tokenizer_check: PASS")
