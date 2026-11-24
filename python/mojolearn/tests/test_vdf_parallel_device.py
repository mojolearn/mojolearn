# SPDX-License-Identifier: Apache-2.0
"""Boundary words of `device=` on the host-only inference classes (lane
vdf-parallel): LanguageModelInference, MLPInference and SambaInference run
compiled CPU code on every installation, so None and 'cpu' keep today's
route and 'gpu' is refused naming the class. Words and exception types only;
the numbers are verify's. The word is admitted before any file or weight is
read, so a missing checkpoint path never reaches its own error here."""
import inspect

import pytest

import mojolearn

CLASSES = [
    ("LanguageModelInference", lambda: mojolearn.LanguageModelInference),
    ("MLPInference", lambda: mojolearn.MLPInference),
    ("SambaInference", lambda: mojolearn.SambaInference),
]
MISSING = "vdf-parallel-no-such-checkpoint.ckpt"


@pytest.mark.parametrize("name,cls", CLASSES)
def test_constructors_take_keyword_only_device_default_none(name, cls):
    for method in (cls().__init__, cls().from_checkpoint):
        parameter = inspect.signature(method).parameters["device"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is None, name


@pytest.mark.parametrize("name,cls", CLASSES)
@pytest.mark.parametrize("bad", ["tpu", 0])
def test_unknown_device_is_refused_in_the_shared_words(name, cls, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got "):
        cls().from_checkpoint(MISSING, device=bad)


@pytest.mark.parametrize("name,cls", CLASSES)
def test_gpu_is_refused_naming_the_class(name, cls):
    with pytest.raises(ValueError, match="device='gpu' is not available for " + name + ": it has no GPU implementation"):
        cls().from_checkpoint(MISSING, device="gpu")


def test_gpu_is_refused_by_the_constructors_before_the_weights():
    with pytest.raises(ValueError, match="device='gpu' is not available for MLPInference"):
        mojolearn.MLPInference(None, None, None, None, device="gpu")
    with pytest.raises(ValueError, match="device='gpu' is not available for LanguageModelInference"):
        mojolearn.LanguageModelInference(None, device="gpu")


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
    ("TrainedBpeVocabulary", lambda **kw: TrainedBpeVocabulary(TOKENS, [], {}, **kw)),
]


def test_tokenizer_constructors_take_keyword_only_device_default_none():
    for method in (BpeTokenizer.__init__, BpeTokenizer.from_ranks_file, BpeTokenizer.from_token_bytes,
                   BpeTokenizer.from_files, BpeVocabularyTrainer.__init__, TrainedBpeVocabulary.__init__,
                   TrainedBpeVocabulary.tokenizer):
        parameter = inspect.signature(method).parameters["device"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, method
        assert parameter.default is None, method


@pytest.mark.parametrize("name,build", TOKENIZER_CONSTRUCTORS)
@pytest.mark.parametrize("bad", ["tpu", 0])
def test_tokenizer_unknown_device_is_refused_in_the_shared_words(identical_mode, name, build, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got "):
        build(device=bad)


@pytest.mark.parametrize("name,build", TOKENIZER_CONSTRUCTORS)
def test_tokenizer_gpu_outside_the_fast_arm_is_refused_naming_the_class(identical_mode, name, build):
    with pytest.raises(ValueError, match="device='gpu' is not available for " + name + ": it has no GPU implementation"):
        build(device="gpu")


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
