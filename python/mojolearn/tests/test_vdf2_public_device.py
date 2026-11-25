# SPDX-License-Identifier: Apache-2.0
"""Boundary of `device=` on `mojolearn.models.tokenizer.pretokenize` and `mojolearn.lm_corpus.prepare`,
and of `import mojolearn.linalg_fast` on every installation (lane vdf2-public): the keyword is keyword-only
with default None, a bad word is refused in the shared words before any work, 'gpu' is refused in the rule's
words where it cannot run. No numbers: the digests are verify's, the GPU cut's offsets are the native check
checks/native_tests/pretokenize_gpu_identity.mojo."""
import importlib
import inspect

import pytest

import mojolearn
from mojolearn import lm_corpus
from mojolearn.models import tokenizer as hf_tokenizer

BAD = "device must be 'gpu', 'cpu' or None"
GPU = mojolearn.vendor() != "cpu"


@pytest.mark.parametrize("function", [hf_tokenizer.pretokenize, lm_corpus.prepare])
def test_device_is_keyword_only_with_default_none(function):
    parameter = inspect.signature(function).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("bad", ["tpu", 0, "GPU"])
def test_pretokenize_refuses_a_bad_word_before_reading_the_data(bad):
    # the data is not bytes: a bad device word is refused first
    with pytest.raises(ValueError, match=BAD):
        hf_tokenizer.pretokenize(object(), "gpt2", device=bad)


@pytest.mark.parametrize("bad", ["tpu", 0])
def test_prepare_refuses_a_bad_word_before_reading_any_file(bad, tmp_path):
    with pytest.raises(ValueError, match=BAD):
        lm_corpus.prepare(tmp_path / "absent.txt", cache_dir=tmp_path, device=bad)


@pytest.mark.skipif(GPU, reason="a CPU-only installation refuses device='gpu' in the estimators' words")
def test_pretokenize_gpu_on_cpu_only_is_refused():
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn"):
        hf_tokenizer.pretokenize(b"a b", "gpt2", device="gpu")


# STALE TEST UPDATED (lane vdf2-tok): prepare has its GPU side now, so 'gpu' is refused only where it
# cannot run (a CPU-only installation), in the estimators' words, before any file is read. The
# device == host check of its tokens is checks/native_tests/corpus_gpu_identity.mojo.
@pytest.mark.skipif(GPU, reason="a CPU-only installation refuses device='gpu' in the estimators' words")
def test_prepare_gpu_on_cpu_only_is_refused_before_reading_any_file(tmp_path):
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn"):
        lm_corpus.prepare(tmp_path / "absent.txt", cache_dir=tmp_path, device="gpu")


@pytest.mark.skipif(not GPU, reason="the GPU side runs only on a GPU installation")
def test_prepare_gpu_reads_the_corpus_like_the_host_side(tmp_path):
    # past the device rule the GPU side reads the same files: an absent corpus is the host's refusal,
    # never the former "not available" words
    with pytest.raises(Exception) as refused:
        lm_corpus.prepare(tmp_path / "absent.txt", cache_dir=tmp_path, device="gpu")
    assert "is not available for mojolearn.lm_corpus.prepare" not in str(refused.value)


def test_pretokenize_cpu_returns_a_list_of_ints():
    cut = hf_tokenizer.pretokenize(b"Hello world", "llama3", device="cpu")
    assert type(cut) is list and all(type(offset) is int for offset in cut)


def test_linalg_fast_imports_on_every_installation():
    module = importlib.import_module("mojolearn.linalg_fast")
    # has_bf16 answers from the loaded column; on a CPU-only installation it is False
    if not GPU:
        assert module.has_bf16() is False
