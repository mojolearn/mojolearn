"""Focused public boundary controls. Run only against the integrated rebuild."""
import array
import _codecs
import sys

import pytest

from mojolearn.tokenizer import BpeTokenizer, BpeVocabularyTrainer
from mojolearn.models import Tokenizer
from mojolearn.models import tokenizer as model_tokenizer


@pytest.fixture
def bpe():
    return BpeTokenizer.from_token_bytes([bytes([i]) for i in range(256)])


def test_native_buffers_iterators_batch_and_decoding_keep_results(bpe):
    data = memoryview(bytearray(b"axbxc"))[::2]
    assert bpe.encode(data) == list(b"abc")
    assert bpe.encode_batch(iter(["abc", b"", data])) == [list(b"abc"), [], list(b"abc")]
    assert bpe.decode_bytes(array.array("i", [97, 98, 99])) == b"abc"
    assert bpe.decode_bytes(memoryview(array.array("i", [97, 0, 98]))[::2]) == b"ab"
    assert bpe.decode_batch([[], [97]]) == ["", "a"]
    with pytest.raises(UnicodeDecodeError):
        bpe.decode([0xC3], errors="strict")


@pytest.mark.parametrize("operation", ["encode", "encode_batch", "decode_bytes"])
def test_input_protocols_are_refused_before_invocation(bpe, operation):
    seen = []

    class Hook:
        def __iter__(self):
            seen.append("iter")
            return iter([])

        def __len__(self):
            seen.append("len")
            return 0

        def __buffer__(self, flags):
            seen.append("buffer")
            return memoryview(b"a")

    with pytest.raises(TypeError):
        getattr(bpe, operation)(Hook())
    assert seen == []


def test_nested_scalar_and_list_subclass_hooks_are_not_read(bpe):
    seen = []

    class Integer(int):
        def __index__(self):
            seen.append("index")
            return 97

        def __int__(self):
            seen.append("int")
            return 97

    class Documents(list):
        def __iter__(self):
            seen.append("iter")
            return super().__iter__()

    with pytest.raises(TypeError):
        bpe.decode([Integer(97)])
    with pytest.raises(TypeError):
        bpe.encode_batch(Documents(["a"]))
    assert seen == []


def test_replaced_dict_descriptor_does_not_run(bpe):
    seen = []

    class HookLayout(BpeTokenizer):
        __slots__ = ()

        @property
        def __dict__(self):
            seen.append("dict")
            return {}

    bpe.__class__ = HookLayout
    with pytest.raises(TypeError):
        bpe.encode("a")
    assert seen == []


def test_mutated_state_descriptor_and_scalar_refuse_before_hooks(bpe, monkeypatch):
    seen = []

    class Count(int):
        def __int__(self):
            seen.append("int")
            return 257

    bpe._n_vocab = Count(257)
    with pytest.raises(TypeError):
        _ = bpe.n_vocab
    monkeypatch.setattr(BpeTokenizer, "_handle", property(lambda self: seen.append("handle")), raising=False)
    with pytest.raises(TypeError):
        bpe.encode("a")
    assert seen == []


def test_model_tokenizer_has_no_mutable_provider_lookup(monkeypatch):
    tokenizer = Tokenizer([bytes([i]) for i in range(256)], [], pattern="gpt2")
    fingerprint = model_tokenizer._NATIVE.vocab_fingerprint(tokenizer)
    seen = []

    class Provider:
        def __getattr__(self, name):
            seen.append(name)
            raise AssertionError(name)

    tokenizer._m = Provider()
    assert tokenizer.encode("abc") == list(b"abc")
    assert tokenizer.decode(list(b"abc")) == "abc"
    assert model_tokenizer._NATIVE.vocab_fingerprint(tokenizer) == fingerprint
    assert seen == []


def test_training_rendering_files_and_identity_round_trip(tmp_path):
    trained = BpeVocabularyTrainer(vocab_size=258, min_frequency=1).train([b"ab ab", "abc"])
    ranks = tmp_path / "ranks.tsv"
    tokenizer_json = tmp_path / "tokenizer.json"
    assert trained.write_ranks(ranks) is ranks
    assert trained.write_tokenizer_json(tokenizer_json) is tokenizer_json
    from_ranks = BpeTokenizer.from_ranks_file(ranks)
    assert from_ranks.identity == trained.identity == trained.tokenizer().identity
    loaded = Tokenizer.from_pretrained(tokenizer_json)
    assert loaded.decode(loaded.encode("abc")) == "abc"


@pytest.mark.parametrize("policy", ["strict", "ignore", "replace", "backslashreplace", "surrogateescape", "surrogatepass"])
@pytest.mark.parametrize("surface", ["bpe", "bpe_batch", "model"])
def test_builtin_shaped_error_handler_replacement_is_refused_before_output_hooks(bpe, monkeypatch, policy, surface):
    tokenizer = Tokenizer([bytes([i]) for i in range(256)], [], pattern="gpt2") if surface == "model" else bpe
    original = _codecs.lookup_error(policy)
    seen = []

    class Output:
        def write(self, text):
            seen.append(text)
            return len(text)

        def flush(self):
            seen.append("flush")

    # print has the same builtin_function_or_method type as the canonical
    # handlers, but it can invoke the application's Python stdout hooks.
    assert type(print) is type(original)
    try:
        _codecs.register_error(policy, print)
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", Output())
            with pytest.raises(ValueError, match="canonical builtin UTF-8 error policy"):
                if surface == "bpe_batch":
                    tokenizer.decode_batch([[0xC3]], errors=policy)
                else:
                    tokenizer.decode([0xC3], errors=policy)
    finally:
        _codecs.register_error(policy, original)
    assert seen == []


@pytest.mark.parametrize("policy", ["strict", "ignore", "replace", "backslashreplace", "surrogateescape", "surrogatepass"])
def test_canonical_error_policies_preserve_the_utf8_decoder_result(bpe, policy):
    raw = b"a\xc3(\xed\xa0\x80"
    # Independent CPython oracle is test preparation, outside product work.
    try:
        expected = raw.decode("utf-8", errors=policy)
    except UnicodeDecodeError:
        with pytest.raises(UnicodeDecodeError):
            bpe.decode(list(raw), errors=policy)
    else:
        assert bpe.decode(list(raw), errors=policy) == expected


def test_decode_uses_the_captured_native_registry_lookup(bpe, monkeypatch):
    seen = []

    def lookup(policy):
        seen.append(policy)
        raise AssertionError("replaced codec lookup must not execute")

    monkeypatch.setattr(_codecs, "lookup_error", lookup)
    assert bpe.decode([0xC3], errors="replace") == "\ufffd"
    assert seen == []
