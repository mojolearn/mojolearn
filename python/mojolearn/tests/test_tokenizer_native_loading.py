# SPDX-License-Identifier: Apache-2.0
"""Native tokenizer load/construct boundaries and independent JSON oracle."""
import json

import pytest

from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.models import Tokenizer
from mojolearn.models import tokenizer as module
from mojolearn.tests.test_models_loader import _small_vocab, _tokenizer_json


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 2**63,
    [0.25, -0.0, 1.0e300, 1.0e-300], {"one": [True, None, {"two": "é\x00€😀"}]},
    "\\\"/\b\f\n\r\t", "\ud800", "\udc00"])
def test_native_json_matches_independent_standard_parser(value):
    raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    assert module._NATIVE.native_json_parse(raw) == json.loads(raw)


@pytest.mark.parametrize("raw", ["", "[1,]", '{"a":1,}', "01", "1.", "1e", "--1",
    '"\x01"', '"\\q"', '"\\uQ000"', '"unclosed', "null true", "[", '{"a" 1}'])
def test_native_json_rejects_invalid_input(raw):
    with pytest.raises(ValueError):
        module._NATIVE.native_json_parse(raw)


@pytest.mark.parametrize("pattern", ["gpt2", "llama3", "qwen2"])
def test_constructor_and_file_load_have_one_native_boundary(tmp_path, pattern, monkeypatch):
    tokens, merges = _small_vocab()
    data = _tokenizer_json(tokens, merges, module.PATTERNS[pattern][0],
                           [("<end>", 300)], template=True, normalizer={"type": "NFC"})
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<end>"}), encoding="utf-8")
    tok = Tokenizer.__new__(Tokenizer)
    audit_native_call(Tokenizer.__init__, tok, tokens, merges, pattern=pattern).assert_native()
    loaded = audit_native_call(Tokenizer.from_pretrained, path).assert_native()
    from_dir = audit_native_call(Tokenizer.from_pretrained, str(tmp_path)).assert_native()
    direct = audit_native_call(Tokenizer._from_json, data, "fixture.json").assert_native()
    assert loaded.eos_token_id == 300 and direct.bos_ids == (300,)
    expected = tok.encode("hello world 1234", add_special_tokens=False)
    for instance in (loaded, from_dir, direct):
        assert audit_native_call(instance.encode, "hello world 1234", add_special_tokens=False).assert_native() == expected
        assert audit_native_call(instance.decode, expected).assert_native() == "hello world 1234"
        assert audit_native_call(instance.pretokenize, "hello").assert_native() == [b"hello"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Python JSON/config callback executed")

    with monkeypatch.context() as patch:
        patch.setattr(json, "load", forbidden)
        patch.setattr(json, "loads", forbidden)
        patch.setattr(type(path), "__fspath__", forbidden)
        assert audit_native_call(Tokenizer.from_pretrained, path).assert_native().encode("hello")


def test_native_loader_preserves_public_refusal_classes(tmp_path):
    with pytest.raises(FileNotFoundError):
        Tokenizer.from_pretrained(tmp_path)
    with pytest.raises(ValueError, match="unknown pattern"):
        Tokenizer([bytes([i]) for i in range(256)], [], pattern="unknown")
    with pytest.raises(TypeError):
        module._NATIVE.vocab_init_public()
