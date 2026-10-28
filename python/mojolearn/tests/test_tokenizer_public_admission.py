"""Focused public boundary controls. Run only against the integrated rebuild."""
import array

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
