# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tokenizer's Python door.

`GPT2Tokenizer` through the built `_mojolearn_tokenizer_host` binding. mojolearn
ships no vocabulary (2026-09-15), so the vocabulary here is the synthetic one
`mojolearn/_tokenizer_synthetic.py` trains itself, and every expected id comes
from that module's second, pure Python encoder of the same algorithm: exact id
sequences, byte-exact round trips, both readings of `<|endoftext|>`, byte
fallback, Unicode classes, batch encoding, and every refusal by name. A build
with `-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1` (ids written in reverse) must
fail the exact-id tests, which is how this file is known to read the binary.

A user-supplied GPT-2 vocabulary is tested when MOJOLEARN_GPT2_ENCODER_JSON
and MOJOLEARN_GPT2_VOCAB_BPE name the two files; without them that test
skips (under pytest) or is reported as skipped (module run).

    cd python && python3 -m mojolearn.tests.test_tokenizer_surface
    cd python && python3 -m pytest -q mojolearn/tests/test_tokenizer_surface.py

Without the binding built (`bindings/build_tokenizer_host.sh`) the module run
exits 2 and says so; pytest skips with the same sentence. Neither is a pass.
"""
import json
import os
import sys
import tempfile

try:
    import pytest
except ImportError:  # the module run needs no pytest
    pytest = None

from mojolearn import GPT2Tokenizer
from mojolearn import _tokenizer_synthetic as syn

ENCODER_ENV = "MOJOLEARN_GPT2_ENCODER_JSON"
MERGES_ENV = "MOJOLEARN_GPT2_VOCAB_BPE"


class Skipped(Exception):
    pass


#: True inside `main` (the module run), where a skip is reported by this file
#: rather than by pytest, even when pytest is importable.
_MODULE_RUN = False


def _skip(why):
    if pytest is not None and not _MODULE_RUN:
        pytest.skip(why)
    raise Skipped(why)


def _tokenizer():
    try:
        return GPT2Tokenizer._synthetic()
    except ImportError as exc:
        if pytest is not None:
            pytest.skip(f"tokenizer host binding not built: {exc}")
        raise


if pytest is not None:
    @pytest.fixture(scope="module")
    def tok():
        return _tokenizer()


def _vocab():
    return syn.vocabulary()


def _ref(raw, allow=False):
    return syn.reference_encode(_vocab(), raw, allow)


def test_synthetic_vocabulary_shape(tok):
    v = _vocab()
    assert v[:256] == [bytes([b]) for b in range(256)]
    assert len(v) == syn.TARGET_TOKENS and len(set(v)) == len(v)
    assert sum(1 for t in v if len(t) > 1) >= 200, "the vocabulary must have real merges"
    assert tok.n_vocab == len(v) + 1
    assert tok.eot_token == len(v)
    assert syn.vocabulary() == v, "the synthetic vocabulary is deterministic"


def test_cases_exact_ids(tok):
    wrong = []
    for name, text, allow in syn.cases():
        raw = text.encode("utf-8")
        want = _ref(raw, allow)
        got = tok.encode_bytes(raw, allow_endoftext=allow)
        if got != want:
            wrong.append((name, want, got))
    assert wrong == [], f"{len(wrong)} of {len(syn.cases())} id sequences differ: {wrong}"


def test_cases_exercise_merges_and_fallback(tok):
    """The cases reach what they claim: a merged token, a byte-fallback run,
    the special token and a multi-byte codepoint split to bytes."""
    ids = [i for name, text, allow in syn.cases() for i in tok.encode(text, allow_endoftext=allow)]
    assert any(256 <= i < tok.eot_token for i in ids), "no merged token in the cases"
    assert tok.eot_token in ids
    assert tok.encode("xyzw QXJ") == list(b"xyzw QXJ"), "unseen bytes fall back to byte tokens"
    assert tok.encode(" kalo")[0] >= 256


def test_cases_round_trip(tok):
    wrong = []
    for name, text, allow in syn.cases():
        raw = text.encode("utf-8")
        back = tok.decode_bytes(tok.encode_bytes(raw, allow_endoftext=allow))
        if back != raw:
            wrong.append((name, raw, back))
    assert wrong == [], f"{len(wrong)} round trips differ: {wrong}"


def test_unicode_classes(tok):
    """The pre-token boundaries the classes decide, through the ids: a
    precomposed letter joins its word, a combining mark does not, Arabic-Indic
    digits are numbers, NBSP is whitespace and U+001C is not."""
    for text in ("e\u0301", "caf\u00e9", "\u0661\u0662 3", "a\u00a0b", "a\u001cb", "\u4e2d\u6587 x",
                 "it's IT'S", "a   b", "a   ", "\U0001F642!"):
        raw = text.encode("utf-8")
        assert syn.pretokenize(raw) and tok.encode_bytes(raw) == _ref(raw), text


def test_invalid_utf8_round_trips(tok):
    """A byte that begins no well-formed sequence is its own one-byte
    pre-token (every single byte is a token), so it encodes and comes back."""
    raw = b"\xff\xfe abc \x00 \xc3 \xed\xa0\x80"
    ids = tok.encode_bytes(raw)
    assert ids == _ref(raw)
    assert all(0 <= i < tok.n_vocab for i in ids)
    assert tok.decode_bytes(ids) == raw


def test_endoftext_both_readings(tok):
    text = "<|endoftext|>"
    eot = tok.eot_token
    assert tok.encode(text, allow_endoftext=True) == [eot]
    plain = tok.encode(text, allow_endoftext=False)
    assert len(plain) >= 2 and eot not in plain
    assert tok.encode(text) == plain
    both = tok.encode("a<|endoftext|>b", allow_endoftext=True)
    assert both == tok.encode("a") + [eot] + tok.encode("b")
    assert tok.decode([eot]) == text
    assert tok.decode_bytes([eot]) == text.encode()


def test_decode_text_and_errors(tok):
    assert tok.decode(tok.encode("h\u00e9llo w\u00f6rld")) == "h\u00e9llo w\u00f6rld"
    half = tok.encode_bytes(b"\xc3")  # the first byte of a two-byte character alone
    assert half == [0xC3]
    assert tok.decode(half) == "\ufffd"
    try:
        tok.decode(half, errors="strict")
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("errors='strict' did not raise on half a character")
    assert tok.decode([]) == "" and tok.decode_bytes([]) == b""


def test_binding_reads_back_cpu_identical(tok):
    m = tok._m
    assert str(m.tokenizer_host_vendor()) == "cpu"
    assert int(m.tokenizer_host_numeric_mode()) == 1
    assert str(m.tokenizer_host_column()) == "cpu"
    assert bool(m.tokenizer_host_sabotage()) is False or _sabotage_allowed()


def _sabotage_allowed():
    return os.environ.get("MOJOLEARN_HOST_ALLOW_SABOTAGE") == "1"


def _raises(fn, exc_type, needle):
    try:
        fn()
    except exc_type as exc:
        assert needle in str(exc), f"{exc_type.__name__} said {exc!r}, not {needle!r}"
        return
    raise AssertionError(f"{exc_type.__name__} containing {needle!r} was not raised")


def test_refuses_text_type(tok):
    _raises(lambda: tok.encode(123), TypeError, "text must be str or bytes-like, got int")
    _raises(lambda: tok.encode(None), TypeError, "text must be str or bytes-like, got NoneType")
    _raises(lambda: tok.encode_bytes("str"), TypeError, "encode_bytes takes bytes; use encode for str")


def test_refuses_allow_endoftext_type(tok):
    _raises(lambda: tok.encode("x", allow_endoftext=1), TypeError, "allow_endoftext must be a bool, got int")
    _raises(lambda: tok.encode_bytes(b"x", allow_endoftext=None), TypeError, "allow_endoftext must be a bool, got NoneType")


def test_refuses_ids_out_of_range(tok):
    n = tok.n_vocab
    _raises(lambda: tok.decode_bytes([1, n]), ValueError, f"id {n} at position 1 is outside [0, {n})")
    _raises(lambda: tok.decode([-1]), ValueError, f"id -1 at position 0 is outside [0, {n})")
    assert tok.decode_bytes([n - 1]) == b"<|endoftext|>"  # the last valid id


def test_refuses_ids_of_the_wrong_type(tok):
    _raises(lambda: tok.decode_bytes([True]), TypeError, "ids must be int, not bool, at position 0")
    _raises(lambda: tok.decode_bytes([1, 1.5]), TypeError, "ids must be int, got float at position 1")
    _raises(lambda: tok.decode_bytes("abc"), TypeError, "ids must be a sequence of int, got str")
    _raises(lambda: tok.decode_bytes(5), TypeError, "ids must be a sequence of int, got int")


def _batch_documents():
    """Every case text, empty documents between them and invalid UTF-8
    documents: adjacent documents whose concatenation would merge."""
    docs = [text.encode("utf-8") for _, text, _ in syn.cases()]
    docs += [b"", b"kalo", b"", b" mine", b"\xff\xfe<|endoftext|>\xc3", b"it", b"'s", b"ka", b"lo"]
    return docs


def test_encode_batch_each_document_as_alone(tok):
    docs = _batch_documents()
    for allow in (False, True):
        got = tok.encode_batch(docs, allow_endoftext=allow)
        assert len(got) == len(docs)
        for k, (d, ids) in enumerate(zip(docs, got)):
            assert ids == tok.encode_bytes(d, allow_endoftext=allow) == _ref(d, allow), (k, d, allow)


def test_encode_batch_split_invariant(tok):
    """The batch boundary is invisible: every split of the batch, every
    single document and the reversed order read the same ids per document.
    A binding built with MOJOLEARN_TOKENIZER_BATCH_SABOTAGE fails here."""
    docs = _batch_documents()
    whole = tok.encode_batch(docs, allow_endoftext=True)
    for a in (1, 7, len(docs) // 2):
        assert tok.encode_batch(docs[:a], allow_endoftext=True) + tok.encode_batch(docs[a:], allow_endoftext=True) == whole
    assert [tok.encode_batch([d], allow_endoftext=True)[0] for d in docs] == whole
    assert tok.encode_batch(docs[::-1], allow_endoftext=True) == whole[::-1]
    assert tok.encode_batch([b"ka", b"lo"]) == [_ref(b"ka"), _ref(b"lo")]


def test_encode_batch_accepts_str_and_empty(tok):
    assert tok.encode_batch([]) == []
    assert tok.encode_batch(["", b""]) == [[], []]
    assert tok.encode_batch([" kalo mine", bytearray(b"kalo")]) == [_ref(b" kalo mine"), _ref(b"kalo")]
    assert tok.encode_batch(("<|endoftext|>",), allow_endoftext=True) == [[tok.eot_token]]
    assert tok.encode_batch(iter(["\u00e9"])) == [_ref("\u00e9".encode())]


def test_decode_batch_round_trip(tok):
    docs = _batch_documents()
    ids = tok.encode_batch(docs, allow_endoftext=True)
    assert tok.decode_bytes_batch(ids) == docs
    assert tok.decode_batch(ids) == [d.decode("utf-8", "replace") for d in docs]
    assert tok.decode_batch([]) == []


def test_batch_refusals(tok):
    n = tok.n_vocab
    _raises(lambda: tok.encode_batch("abc"), TypeError, "encode_batch takes a sequence of documents, got str")
    _raises(lambda: tok.encode_batch(b"abc"), TypeError, "encode_batch takes a sequence of documents, got bytes")
    _raises(lambda: tok.encode_batch(5), TypeError, "encode_batch takes a sequence of documents, got int")
    _raises(lambda: tok.encode_batch(["a", 3]), TypeError, "document 1 must be str or bytes-like, got int")
    _raises(lambda: tok.encode_batch(["a"], allow_endoftext=1), TypeError, "allow_endoftext must be a bool, got int")
    _raises(lambda: tok.decode_batch([[1], [n]]), ValueError, f"id {n} at position 0 is outside [0, {n})")
    _raises(lambda: tok.decode_bytes_batch(b"ab"), TypeError, "decode_bytes_batch takes a sequence of id sequences")


def test_ranks_file_matches_token_bytes(tok):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "ranks.tsv")
    syn.write_ranks(_vocab(), path)
    other = GPT2Tokenizer.from_ranks_file(path)
    text = "".join(t for _, t, _ in syn.cases()).encode("utf-8")
    assert other.n_vocab == tok.n_vocab
    assert other.encode_bytes(text, allow_endoftext=True) == tok.encode_bytes(text, allow_endoftext=True)


def test_from_files_spelled_vocabulary(tok):
    """The synthetic vocabulary written in the GPT-2 file format (spelled
    encoder JSON plus a merge list) loads to the same tokenizer."""
    from mojolearn.tokenizer import _byte_to_char
    spell = _byte_to_char()
    v = _vocab()
    d = tempfile.mkdtemp()
    enc = {"".join(spell[b] for b in t): i for i, t in enumerate(v)}
    enc["<|endoftext|>"] = len(v)
    with open(os.path.join(d, "encoder.json"), "w", encoding="utf-8") as fh:
        json.dump(enc, fh)
    index = {t: i for i, t in enumerate(v)}
    lines = ["#version: synthetic"]
    for i, t in enumerate(v):
        if i < 256:
            continue
        # the earliest split into two lower tokens
        for k in range(1, len(t)):
            a, b = t[:k], t[k:]
            if a in index and b in index and index[a] < i and index[b] < i:
                lines.append("".join(spell[x] for x in a) + " " + "".join(spell[x] for x in b))
                break
    with open(os.path.join(d, "vocab.bpe"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    other = GPT2Tokenizer.from_files(os.path.join(d, "encoder.json"), os.path.join(d, "vocab.bpe"))
    text = "".join(t for _, t, _ in syn.cases()).encode("utf-8")
    assert other.n_vocab == tok.n_vocab
    assert other.encode_bytes(text, allow_endoftext=True) == tok.encode_bytes(text, allow_endoftext=True)


def test_user_supplied_gpt2_files():
    """The GPT-2 encoder.json and vocab.bpe a user downloaded themselves.
    Skips when the two environment variables do not name existing files."""
    enc, merges = os.environ.get(ENCODER_ENV, ""), os.environ.get(MERGES_ENV, "")
    if not (enc and merges and os.path.isfile(enc) and os.path.isfile(merges)):
        _skip(f"{ENCODER_ENV} and {MERGES_ENV} do not name the GPT-2 files")
    tok = GPT2Tokenizer.from_files(enc, merges)
    with open(enc, "r", encoding="utf-8") as fh:
        encoder = json.load(fh)
    assert tok.n_vocab == len(encoder)
    assert tok.eot_token == encoder["<|endoftext|>"]
    assert tok.encode("<|endoftext|>", allow_endoftext=True) == [encoder["<|endoftext|>"]]
    # a word the vocabulary holds whole is one id, with and without its space
    for spelled in ("hello", "\u0120hello", "\u0120world"):
        if spelled in encoder:
            text = spelled.replace("\u0120", " ")
            assert tok.encode(text) == [encoder[spelled]], text
    text = "It's 2026: h\u00e9llo, w\u00f6rld!  \u4e2d\u6587\n<|endoftext|>"
    assert tok.decode(tok.encode(text, allow_endoftext=True)) == text


def test_refuses_no_vocabulary_by_name():
    """Resolved before the binding is loaded, so this needs no build."""
    _raises(lambda: GPT2Tokenizer(), ValueError, "GPT2Tokenizer needs a vocabulary, and mojolearn ships none")
    _raises(lambda: GPT2Tokenizer(), ValueError, "GPT2Tokenizer.from_files(encoder_json, vocab_bpe)")
    _raises(lambda: GPT2Tokenizer(), ValueError, "OpenAI publishes with its GPT-2 release")
    missing = os.path.join(tempfile.mkdtemp(), "ranks.tsv")
    _raises(lambda: GPT2Tokenizer.from_ranks_file(missing), FileNotFoundError, "does not exist")


def test_refuses_bad_token_bytes():
    """Refused in Python before any binding is loaded."""
    full = [bytes([b]) for b in range(256)]
    _raises(lambda: GPT2Tokenizer.from_token_bytes(full[:255]), ValueError, "lacks 1 of the 256 single-byte tokens")
    _raises(lambda: GPT2Tokenizer.from_token_bytes(full + [b"ab", b"ab"]), ValueError, "token 257 repeats token 256")
    _raises(lambda: GPT2Tokenizer.from_token_bytes(full + [b""]), ValueError, "token 256 is empty")
    _raises(lambda: GPT2Tokenizer.from_token_bytes(b"abc"), TypeError, "tokens must be a sequence of bytes")


TESTS = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
_NO_TOK = ("test_user_supplied_gpt2_files", "test_refuses_no_vocabulary_by_name", "test_refuses_bad_token_bytes")


def main(argv=None):
    global _MODULE_RUN
    _MODULE_RUN = True
    out = sys.stdout
    try:
        tok = GPT2Tokenizer._synthetic()
    except ImportError as exc:
        out.write(f"test_tokenizer_surface: NOT RUN. {exc}\n")
        return 2
    failures, skipped = [], []
    for name, fn in TESTS:
        try:
            if name in _NO_TOK:
                fn()
            else:
                fn(tok)
        except Skipped as why:
            skipped.append((name, str(why)))
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    for name, why in failures:
        out.write(f"FAIL {name}: {why}\n")
    for name, why in skipped:
        out.write(f"SKIP {name}: {why}\n")
    n = len(TESTS)
    if failures:
        out.write(f"test_tokenizer_surface: RED. {len(failures)} of {n} checks failed.\n")
        return 1
    out.write(
        f"test_tokenizer_surface: GREEN. {n - len(skipped)} of {n} checks on this host ({len(skipped)} skipped): "
        f"{len(syn.cases())} exact id sequences and round trips against the synthetic vocabulary's\n"
        "Python encoder through the Python door, merges, byte fallback, Unicode classes, both\n"
        "readings of <|endoftext|>, batch encoding, the rank-file and spelled-file loaders, and\n"
        "every refusal by name. It says nothing about another box.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
