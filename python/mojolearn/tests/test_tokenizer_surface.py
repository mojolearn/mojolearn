# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPT-2 tokenizer's Python door (the expose-tokenizer lane, 2026-09-14).

`GPT2Tokenizer` through the built `_mojolearn_tokenizer_host` binding,
against the SAME 43 cases `pixi run check-tokenizer` holds the Mojo encoder
to (`tokenizer/checks/fixtures/gpt2_reference.json`, recorded from tiktoken
0.14.0): exact id sequences, byte-exact round trips, the two readings of
`<|endoftext|>`, fixed byte strings with their ids spelled out here, and
every refusal by name. A build with `-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1`
(ids written in reverse) must fail the exact-id tests, which is how this
file is known to read the binary and not a Python stand-in.

Runs two ways:

    cd python && python3 -m mojolearn.tests.test_tokenizer_surface
    cd python && python3 -m pytest -q mojolearn/tests/test_tokenizer_surface.py

Without the binding built (`bindings/build_tokenizer_host.sh`) the module
run exits 2 and says so; pytest skips with the same sentence. Neither is a
pass. This file measures the host it runs on and says nothing about another
box: the tokenizer has no float arithmetic, so what it can get wrong is
agreement with the reference, and that is what is asserted.
"""
import json
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # the module run needs no pytest
    pytest = None

from mojolearn import GPT2Tokenizer

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tokenizer" / "checks" / "fixtures" / "gpt2_reference.json"

#: Fixed byte strings and the ids tiktoken 0.14.0 assigns them, spelled out
#: so a reader needs no fixture to see what "exact" means here.
FIXED = (
    (b"hello world", False, [31373, 995]),
    (b"hello", False, [31373]),
    (b" hello", False, [23748]),
    (b"it's", False, [270, 338]),
    (b"IT'S", False, [2043, 6, 50]),
    (b"\x00\x01\x7f", False, [188, 189, 221]),
    (b"\xc3", False, [127]),  # a lone continuation-lead byte, its own token
    ("é".encode("utf-8"), False, [2634]),  # one token, not two bytes
    (b"<|endoftext|>", True, [50256]),
    (b"", False, []),
)


def _tokenizer():
    try:
        return GPT2Tokenizer()
    except ImportError as exc:
        if pytest is not None:
            pytest.skip(f"tokenizer host binding not built: {exc}")
        raise


if pytest is not None:
    @pytest.fixture(scope="module")
    def tok():
        return _tokenizer()


def _cases():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        fx = json.load(fh)
    assert fx["encoding"] == "gpt2" and fx["n_vocab"] == 50257
    assert fx["tiktoken_version"] == "0.14.0"
    return fx["cases"]


def test_fixed_bytes_exact_ids(tok):
    for raw, allow, want in FIXED:
        assert tok.encode_bytes(raw, allow_endoftext=allow) == want, raw
        assert tok.encode(raw, allow_endoftext=allow) == want, raw
    assert tok.encode("hello world") == [31373, 995]
    assert tok.encode("") == []


def test_reference_cases_exact_ids(tok):
    """All 43 recorded cases, id for id; `endoftext_as_special` is the one
    case recorded with the special token allowed, as the Mojo gate reads
    it."""
    cases = _cases()
    assert len(cases) == 43
    wrong = []
    for case in cases:
        text = case["text"].encode("utf-8")
        allow = case["name"] == "endoftext_as_special"
        got = tok.encode_bytes(text, allow_endoftext=allow)
        if got != case["ids"]:
            wrong.append((case["name"], case["ids"], got))
    assert wrong == [], f"{len(wrong)} of 43 id sequences differ: {wrong}"


def test_reference_cases_round_trip(tok):
    cases = _cases()
    wrong = []
    for case in cases:
        text = case["text"].encode("utf-8")
        allow = case["name"] == "endoftext_as_special"
        back = tok.decode_bytes(tok.encode_bytes(text, allow_endoftext=allow))
        if back != text:
            wrong.append((case["name"], text, back))
    assert wrong == [], f"{len(wrong)} of 43 round trips differ: {wrong}"


def test_invalid_utf8_round_trips(tok):
    """A byte that begins no well-formed sequence is its own one-byte
    pre-token (every single byte is a token), so it encodes and comes back;
    tiktoken's `&str` input cannot hold it, so this is ours to state."""
    raw = b"\xff\xfe abc \x00 \xc3"
    ids = tok.encode_bytes(raw)
    assert all(0 <= i < 50257 for i in ids)
    assert tok.decode_bytes(ids) == raw


def test_endoftext_both_readings(tok):
    text = "<|endoftext|>"
    assert tok.encode(text, allow_endoftext=True) == [50256]
    plain = tok.encode(text, allow_endoftext=False)
    assert len(plain) == 7 and 50256 not in plain
    assert tok.encode(text) == plain
    both = tok.encode("a<|endoftext|>b", allow_endoftext=True)
    assert both == tok.encode("a") + [50256] + tok.encode("b")
    assert tok.decode([50256]) == text
    assert tok.decode_bytes([50256]) == text.encode()


def test_decode_text_and_errors(tok):
    assert tok.decode(tok.encode("héllo wörld")) == "héllo wörld"
    assert tok.encode("é") == [2634]  # one token for the two bytes
    half = tok.encode_bytes(b"\xc3")  # the first byte of "é" alone
    assert half == [127]
    assert tok.decode(half) == "�"  # tiktoken's errors="replace"
    try:
        tok.decode(half, errors="strict")
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("errors='strict' did not raise on half a character")
    assert tok.decode([]) == "" and tok.decode_bytes([]) == b""


def test_vocabulary_constants(tok):
    assert tok.n_vocab == 50257 == GPT2Tokenizer.N_VOCAB
    assert tok.eot_token == 50256 == GPT2Tokenizer.ENDOFTEXT_ID
    assert GPT2Tokenizer.ENDOFTEXT == "<|endoftext|>"
    assert (Path(tok.data_directory) / "gpt2_ranks.tsv").is_file()


def test_binding_reads_back_cpu_identical(tok):
    m = tok._m
    assert str(m.tokenizer_host_vendor()) == "cpu"
    assert int(m.tokenizer_host_numeric_mode()) == 1
    assert str(m.tokenizer_host_column()) == "cpu"
    assert bool(m.tokenizer_host_sabotage()) is False or _sabotage_allowed()


def _sabotage_allowed():
    import os
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
    _raises(lambda: tok.decode_bytes([1, 50257]), ValueError, "id 50257 at position 1 is outside [0, 50257)")
    _raises(lambda: tok.decode([-1]), ValueError, "id -1 at position 0 is outside [0, 50257)")
    assert tok.decode_bytes([50256]) == b"<|endoftext|>"  # the last valid id


def test_refuses_ids_of_the_wrong_type(tok):
    _raises(lambda: tok.decode_bytes([True]), TypeError, "ids must be int, not bool, at position 0")
    _raises(lambda: tok.decode_bytes([1, 1.5]), TypeError, "ids must be int, got float at position 1")
    _raises(lambda: tok.decode_bytes("abc"), TypeError, "ids must be a sequence of int, got str")
    _raises(lambda: tok.decode_bytes(5), TypeError, "ids must be a sequence of int, got int")


def _batch_documents():
    """Every fixture case (both readings are asked separately below), the
    fixed byte strings, empty documents between them and one invalid UTF-8
    document: adjacent documents whose concatenation would merge."""
    docs = [c["text"].encode("utf-8") for c in _cases()]
    docs += [raw for raw, _allow, _want in FIXED]
    docs += [b"", b"hello", b"", b" world", b"\xff\xfe<|endoftext|>\xc3", b"it", b"'s"]
    return docs


def test_encode_batch_each_document_as_alone(tok):
    docs = _batch_documents()
    for allow in (False, True):
        got = tok.encode_batch(docs, allow_endoftext=allow)
        assert len(got) == len(docs)
        for k, (d, ids) in enumerate(zip(docs, got)):
            assert ids == tok.encode_bytes(d, allow_endoftext=allow), (k, d, allow)


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
    assert tok.encode_batch([b"hello", b" world"]) == [[31373], [995]]


def test_encode_batch_accepts_str_and_empty(tok):
    assert tok.encode_batch([]) == []
    assert tok.encode_batch(["", b""]) == [[], []]
    assert tok.encode_batch(["hello world", bytearray(b"hello")]) == [[31373, 995], [31373]]
    assert tok.encode_batch(("<|endoftext|>",), allow_endoftext=True) == [[50256]]
    assert tok.encode_batch(iter(["é"])) == [[2634]]


def test_decode_batch_round_trip(tok):
    docs = _batch_documents()
    ids = tok.encode_batch(docs, allow_endoftext=True)
    assert tok.decode_bytes_batch(ids) == docs
    assert tok.decode_batch(ids) == [d.decode("utf-8", "replace") for d in docs]
    assert tok.decode_batch([]) == []


def test_batch_refusals(tok):
    _raises(lambda: tok.encode_batch("abc"), TypeError, "encode_batch takes a sequence of documents, got str")
    _raises(lambda: tok.encode_batch(b"abc"), TypeError, "encode_batch takes a sequence of documents, got bytes")
    _raises(lambda: tok.encode_batch(5), TypeError, "encode_batch takes a sequence of documents, got int")
    _raises(lambda: tok.encode_batch(["a", 3]), TypeError, "document 1 must be str or bytes-like, got int")
    _raises(lambda: tok.encode_batch(["a"], allow_endoftext=1), TypeError, "allow_endoftext must be a bool, got int")
    _raises(lambda: tok.decode_batch([[1], [50257]]), ValueError, "id 50257 at position 0 is outside [0, 50257)")
    _raises(lambda: tok.decode_bytes_batch(b"ab"), TypeError, "decode_bytes_batch takes a sequence of id sequences")


def test_refuses_a_missing_table_by_name(tmp_path=None):
    """Resolved before the binding is loaded, so this needs no build."""
    import tempfile
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    _raises(lambda: GPT2Tokenizer(data_directory=d), FileNotFoundError, "gpt2_ranks.tsv does not exist")


TESTS = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main(argv=None):
    out = sys.stdout
    try:
        tok = GPT2Tokenizer()
    except ImportError as exc:
        out.write(f"test_tokenizer_surface: NOT RUN. {exc}\n")
        return 2
    failures = []
    for name, fn in TESTS:
        try:
            if name == "test_refuses_a_missing_table_by_name":
                fn()
            else:
                fn(tok)
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    for name, why in failures:
        out.write(f"FAIL {name}: {why}\n")
    n = len(TESTS)
    if failures:
        out.write(f"test_tokenizer_surface: RED. {len(failures)} of {n} checks failed.\n")
        return 1
    out.write(
        f"test_tokenizer_surface: GREEN. {n} checks on this host: 43/43 exact id\n"
        "sequences and 43/43 byte-exact round trips against tiktoken 0.14.0's\n"
        "recorded fixture through the Python door, the fixed byte strings, both\n"
        "readings of <|endoftext|>, invalid UTF-8 round-tripping, and every\n"
        "refusal by name. It says nothing about another box.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
