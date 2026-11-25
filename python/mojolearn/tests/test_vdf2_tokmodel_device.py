# SPDX-License-Identifier: Apache-2.0
"""Boundary of `device=` on `mojolearn.models.Tokenizer` (the constructor and `from_pretrained`):
keyword-only, default None, stored as given, refused before any file or table is read, and on a
GPU installation the same ids and bytes as the CPU (lane vdf2-tokmodel). The device == host check
of every id is checks/native_tests/vocab_gpu_identity.mojo."""
import inspect
import json

import pytest

import mojolearn
from mojolearn.models import Tokenizer

BAD = "device must be 'gpu', 'cpu' or None"
GPU = mojolearn.vendor() != "cpu"


def _tokens():
    tokens = [bytes([b]) for b in range(256)]
    tokens += [b"th", b"he", b"the", b" t", b" the", b"in", b"ing", b"ll", b"'s"]
    merges = [(b"t", b"h"), (b"h", b"e"), (b"th", b"e"), (b" ", b"t"), (b" t", b"he"), (b"i", b"n"),
              (b"in", b"g"), (b"l", b"l"), (b"'", b"s")]
    return tokens, merges


def _make(pattern, device=None):
    tokens, merges = _tokens()
    added = {"<|endoftext|>": 300, "<|end": 301, "aa": 302}
    return Tokenizer(tokens, merges, pattern=pattern, added=added, device=device)


@pytest.mark.parametrize("function", [Tokenizer.__init__, Tokenizer.from_pretrained])
def test_device_is_keyword_only_with_default_none(function):
    parameter = inspect.signature(function).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("bad", ["tpu", 0, "GPU"])
def test_from_pretrained_refuses_a_bad_word_before_reading_any_file(bad, tmp_path):
    with pytest.raises(ValueError, match=BAD):
        Tokenizer.from_pretrained(tmp_path / "absent", device=bad)


@pytest.mark.parametrize("bad", ["tpu", 0])
def test_constructor_refuses_a_bad_word(bad):
    with pytest.raises(ValueError, match=BAD):
        _make("gpt2", device=bad)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_device_is_stored_as_given(device):
    assert _make("gpt2", device=device).device == device


@pytest.mark.skipif(GPU, reason="a CPU-only installation refuses device='gpu' in the estimators' words")
def test_gpu_on_cpu_only_is_refused_before_reading_any_file(tmp_path):
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn"):
        Tokenizer.from_pretrained(tmp_path / "absent", device="gpu")


TEXTS = ["the cat's hat<|endoftext|>the thing", "aaaaa <|end<|endoftext|>", "  in\n\n the  ", "", "x"]


@pytest.mark.skipif(not GPU, reason="the GPU side runs only on a GPU installation")
@pytest.mark.parametrize("pattern", ["gpt2", "llama3", "qwen2"])
def test_gpu_gives_the_cpu_ids_and_bytes(pattern):
    cpu = _make(pattern, device="cpu")
    gpu = _make(pattern, device="gpu")
    assert gpu.device == "gpu"
    for text in TEXTS:
        for allow in (False, True):
            ids = cpu.encode(text, allow_special=allow)
            assert gpu.encode(text, allow_special=allow) == ids
            assert gpu.decode_bytes(ids) == cpu.decode_bytes(ids)
            assert gpu.decode(ids) == cpu.decode(ids)


@pytest.mark.skipif(not GPU, reason="the GPU side runs only on a GPU installation")
def test_gpu_decode_refuses_a_bad_id_in_the_cpu_words():
    cpu = _make("gpt2", device="cpu")
    gpu = _make("gpt2", device="gpu")
    with pytest.raises(ValueError) as host:
        cpu.decode_bytes([5, 299, 7])
    with pytest.raises(ValueError) as device:
        gpu.decode_bytes([5, 299, 7])
    assert str(device.value) == str(host.value)


@pytest.mark.skipif(not GPU, reason="the GPU side runs only on a GPU installation")
def test_from_pretrained_gpu_gives_the_cpu_ids(tmp_path):
    tokens, merges = _tokens()
    table = {}
    for b in list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)):
        table[b] = chr(b)
    extra = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + extra)
            extra += 1
    spell = lambda raw: "".join(table[b] for b in raw)  # noqa: E731
    data = {
        "model": {"type": "BPE", "vocab": {spell(t): i for i, t in enumerate(tokens)},
                  "merges": [spell(a) + " " + spell(b) for a, b in merges]},
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [{"content": "<|endoftext|>", "id": 300}],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(data))
    cpu = Tokenizer.from_pretrained(tmp_path, device="cpu")
    gpu = Tokenizer.from_pretrained(tmp_path, device="gpu")
    assert gpu.device == "gpu"
    for text in TEXTS:
        assert gpu.encode(text, allow_special=True) == cpu.encode(text, allow_special=True)
