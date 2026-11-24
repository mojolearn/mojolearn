# SPDX-License-Identifier: Apache-2.0
"""Boundary words of `device=` on the BPE tokenizers (lane vdf-parallel):
BpeTokenizer, BpeVocabularyTrainer and TrainedBpeVocabulary. Words and
exception types only; the numbers (GPU path against the host column) are the
native check checks/native_tests/bpe_gpu_identity.mojo and verify's."""
import inspect

import pytest

import mojolearn

MISSING = "vdf-parallel-no-such-checkpoint.ckpt"


# The tokenizers (BpeTokenizer, BpeVocabularyTrainer, TrainedBpeVocabulary).
# numeric_mode and device: None follows numeric_mode; 'cpu' is the host code
# and is refused while numeric_mode is 'fast'; 'gpu' is the Apple fast arm,
# refused wherever that arm would not run. These cases pin numeric_mode to
# 'identical', where 'gpu' is refused on every installation.
from mojolearn.tokenizer import BpeTokenizer, BpeVocabularyTrainer, TrainedBpeVocabulary  # noqa: E402

TOKENS = [bytes([i]) for i in range(256)]


@pytest.fixture
def identical_mode():
    before = mojolearn.numeric_mode()
    mojolearn.set_numeric_mode("identical")
    try:
        yield
    finally:
        mojolearn.set_numeric_mode(before)


TOKENIZER_CONSTRUCTORS = [
    ("BpeTokenizer", lambda **kw: BpeTokenizer(MISSING, **kw)),
    ("BpeTokenizer", lambda **kw: BpeTokenizer.from_ranks_file(MISSING, **kw)),
    ("BpeTokenizer", lambda **kw: BpeTokenizer.from_token_bytes(TOKENS, **kw)),
    ("BpeTokenizer", lambda **kw: BpeTokenizer.from_files(MISSING, MISSING, **kw)),
    ("BpeTokenizer", lambda **kw: TrainedBpeVocabulary(TOKENS, [], {}).tokenizer(**kw)),
    ("BpeVocabularyTrainer", lambda **kw: BpeVocabularyTrainer(vocab_size=300, **kw)),
]


def test_tokenizer_constructors_take_keyword_only_device_default_none():
    for method in (BpeTokenizer.__init__, BpeTokenizer.from_ranks_file, BpeTokenizer.from_token_bytes,
                   BpeTokenizer.from_files, BpeVocabularyTrainer.__init__, TrainedBpeVocabulary.tokenizer):
        parameter = inspect.signature(method).parameters["device"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, method
        assert parameter.default is None, method
    # A trained vocabulary holds data and computes nothing: no device parameter.
    assert "device" not in inspect.signature(TrainedBpeVocabulary.__init__).parameters


@pytest.mark.parametrize("name,build", TOKENIZER_CONSTRUCTORS)
@pytest.mark.parametrize("bad", ["tpu", 0])
def test_tokenizer_unknown_device_is_refused_in_the_shared_words(identical_mode, name, build, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got "):
        build(device=bad)


def test_tokenizer_gpu_on_a_cpu_only_installation_is_refused(identical_mode):
    # Holds on a CPU-only installation (mojolearn.vendor() == "cpu"); on a GPU installation
    # device='gpu' runs the GPU tokenizer (test_tokenizer_gpu_round_trip).
    if mojolearn.vendor() != "cpu":
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn"):
        BpeTokenizer.from_token_bytes(TOKENS, device="gpu")


def test_tokenizer_gpu_round_trip(identical_mode):
    # Holds on a GPU installation: words and types only (the ids against the host column are the
    # native check checks/native_tests/bpe_gpu_identity.mojo).
    if mojolearn.vendor() == "cpu":
        return
    tok = BpeTokenizer.from_token_bytes(TOKENS, device="gpu")
    ids = tok.encode("ab cd")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == "ab cd"
    assert tok.decode_batch(tok.encode_batch(["a", "", "b<|endoftext|>"], allow_endoftext=True)) == ["a", "", "b<|endoftext|>"]
    with pytest.raises(ValueError, match="id 999 at position 1 is outside"):
        tok.decode([1, 999])


def test_trainer_gpu(identical_mode):
    # On a CPU-only installation (mojolearn.vendor() == "cpu") 'gpu' is refused at construction; on a
    # GPU installation it trains on the GPU (words and types only; the vocabulary against the host
    # trainer is the native check checks/native_tests/bpe_gpu_identity.mojo).
    if mojolearn.vendor() == "cpu":
        with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn"):
            BpeVocabularyTrainer(vocab_size=300, device="gpu")
        return
    for size in (260, 50000):
        vocab = BpeVocabularyTrainer(vocab_size=size, device="gpu").train(["abab abab", "ab"])
        assert isinstance(vocab, TrainedBpeVocabulary)


def test_decode_batch_reports_the_earliest_sequence_error_first(identical_mode):
    # The host route's order on both devices: an out-of-range id in sequence 0 wins over a
    # non-integer in sequence 1, and the non-integer wins when sequence 0 is valid.
    for device in ("cpu",) if mojolearn.vendor() == "cpu" else ("cpu", "gpu"):
        tok = BpeTokenizer.from_token_bytes(TOKENS, device=device)
        with pytest.raises(ValueError, match="id 999 at position 0 is outside"):
            tok.decode_batch([[999], ["x"]])
        with pytest.raises(TypeError):
            tok.decode_batch([[1], ["x"]])


def test_tokenizer_cpu_runs_the_host_code_in_identical_mode(identical_mode):
    tok = BpeTokenizer.from_token_bytes(TOKENS, device="cpu")
    assert tok.decode(tok.encode("ab")) == "ab"
    vocab = BpeVocabularyTrainer(vocab_size=257, device="cpu").train(["abab"])
    assert isinstance(vocab, TrainedBpeVocabulary)


def test_tokenizer_cpu_in_fast_mode_is_refused():
    # Holds where numeric_mode 'fast' exists (Apple, mojolearn.vendor() == "metal");
    # on other installations set_numeric_mode('fast') is not the fast arm's mode.
    if mojolearn.vendor() != "metal":
        return
    before = mojolearn.numeric_mode()
    mojolearn.set_numeric_mode("fast")
    try:
        with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
            BpeTokenizer.from_token_bytes(TOKENS, device="cpu")
        with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
            BpeVocabularyTrainer(vocab_size=300, device="cpu")
    finally:
        mojolearn.set_numeric_mode(before)
