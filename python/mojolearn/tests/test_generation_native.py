# SPDX-License-Identifier: Apache-2.0
"""Whole native generation against independent block/API and integer oracles.

Fixtures are testing workloads, not performance evidence. These tests require
the installed native entries and fail if they are absent; no feature skip can
be mistaken for generation qualification. Select CPU/GPU with the environment.
"""
import os
import pickle

import numpy as np
import pytest

from mojolearn import Array
from mojolearn._causal_lm_fixtures import (
    family_fixture, _llama_config, _llama_tensors, _write_checkpoint,
)
from mojolearn.models import CausalLM, Tokenizer


DEVICE = os.environ.get("MOJOLEARN_GENERATION_DEVICE", "cpu")


def load_fixture(tmp_path, family="llama", *, uniform=False, profile="fp32_v1"):
    if uniform:
        cfg = _llama_config(vocab_size=256)
        tensors = _llama_tensors(cfg)
        dtype, shape, raw = tensors["lm_head.weight"]
        tensors["lm_head.weight"] = (dtype, shape, bytes(len(raw)))
    else:
        cfg, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), cfg, tensors)
    return CausalLM.load(tmp_path, device=DEVICE, numeric_profile=profile)


def block_reference(model, ids, state=None):
    """Independent test arm through the established public block interfaces."""
    b, l = ids.shape
    x = model._prims.embedding(model._embed, ids.reshape((b * l,)))
    x = x.reshape((b, l, model.d_model))
    for i, blk in enumerate(model.blocks):
        x = blk.forward(x, None if state is None else state.layers[i])
    x = x.reshape((b * l, model.d_model))
    if model._norm is not None:
        x = model._prims.rms_norm(x, model._norm, model.norm_eps)
    logits = model._prims.linear(x, model._head)
    return np.asarray(logits).reshape(b, l, model.vocab_size)


@pytest.mark.parametrize("family", ["llama", "mamba", "mamba2"])
def test_forward_state_and_greedy_match_existing_blocks(tmp_path, family):
    model = load_fixture(tmp_path, family)
    ids = Array.from_list([[1, 4, 7], [2, 5, 8]], "<i4")
    expected = block_reference(model, ids)
    assert np.array_equal(np.asarray(model.forward(ids)).view("u4"), expected.view("u4"))
    reference_state = model.allocate_state(2, 9)
    logits = block_reference(model, ids, reference_state)
    continuation = []
    for i in range(5):
        next_ids = logits[:, -1].argmax(axis=1).astype("i4")
        continuation.append(next_ids)
        if i != 4:
            logits = block_reference(model, Array.from_buffer(next_ids[:, None]), reference_state)
    want = np.concatenate((np.asarray(ids), np.stack(continuation, axis=1)), axis=1)
    assert np.array_equal(np.asarray(model.generate(ids, 5)), want)
    state = model.allocate_state(2, 9)
    model.forward(ids, state)
    assert state.positions == 3
    model.step(np.array([3, 9], dtype="i4"), state)
    assert state.positions == 4
    assert model.reset_state(state) is state
    assert state.positions == 0


def test_uniform_sampling_has_independent_integer_oracle_and_chunk_invariance(tmp_path):
    model = load_fixture(tmp_path, uniform=True)
    prompt = np.array([[33], [34]], dtype="i4")
    options = dict(greedy=False, seed=0, top_k=4)
    expected = [[3, 1, 0, 3, 0, 1, 0, 3], [1, 0, 2, 2, 0, 0, 3, 0]]
    result = np.asarray(model.generate(prompt, 8, **options))
    assert result[:, 1:].tolist() == expected
    stream = model.stream(prompt, 8, **options)
    chunks = [np.asarray(stream.read(n)) for n in (1, 3, 2, 9)]
    assert np.concatenate(chunks, axis=1).tolist() == expected
    assert stream.info() == dict(done=True, generated_tokens=8, lengths=[8, 8], usable=True)
    assert stream.read(1).shape == (2, 0)


def test_eos_stops_rows_and_reports_lengths(tmp_path):
    model = load_fixture(tmp_path, uniform=True)
    stream = model.stream(np.array([[33], [34]], dtype="i4"), 8,
                          greedy=False, seed=0, top_k=4, eos_token_id=3, pad_token_id=9)
    ids = np.asarray(stream.read(20))
    assert ids.tolist() == [[3, 9, 9, 9, 9, 9, 9], [1, 0, 2, 2, 0, 0, 3]]
    assert stream.info()["lengths"] == [1, 7]
    assert stream.info()["done"]
    stopped = np.asarray(model.generate(np.array([[33]], dtype="i4"), 100,
                                       eos_token_id=[0, 1, 2, 3], greedy=False, seed=0, top_k=4))
    assert stopped.tolist() == [[33, 3]]
    assert model.generate(np.array([[33]], dtype="i4"), 0).shape == (1, 1)


@pytest.mark.parametrize("options", [
    {"greedy": False}, {"seed": -1}, {"seed": 1 << 64}, {"top_p": 0},
    {"temperature": float("nan")}, {"temperature": 0}, {"top_k": 257},
    {"eos_token_id": 256}, {"pad_token_id": -1},
])
def test_invalid_options_refused_before_execution(tmp_path, options):
    model = load_fixture(tmp_path, uniform=True)
    with pytest.raises(Exception, match="generation:"):
        model.generate(np.array([[33]], dtype="i4"), 1, **options)


@pytest.mark.parametrize("family", ["llama", "mamba", "mamba2"])
def test_checkpoint_resume_roundtrip_and_corruption_refusal(tmp_path, family):
    model = load_fixture(tmp_path, family)
    prompt = np.array([[1, 3, 5]], dtype="i4")
    options = dict(greedy=False, seed=(1 << 64) - 1, top_k=7, top_p=0.83, temperature=0.8)
    stream = model.stream(prompt, 6, **options)
    prefix = np.asarray(stream.read(2)).copy()
    saved = pickle.loads(pickle.dumps(stream.state_dict()))
    tail = np.asarray(stream.read(10)).copy()
    resumed = model.resume_generation(saved)
    assert np.array_equal(np.asarray(resumed.read(20)), tail)
    whole = np.asarray(model.generate(prompt, 6, **options))[:, prompt.shape[1]:]
    assert np.array_equal(np.concatenate((prefix, tail), axis=1), whole)
    saved["produced"] += 1
    with pytest.raises(Exception, match="checksum"):
        model.resume_generation(saved)


def test_model_and_array_python_callbacks_are_not_used(tmp_path, monkeypatch):
    model = load_fixture(tmp_path, uniform=True)
    prompt = np.array([[33, 34]], dtype="i4")
    state = model.allocate_state(1, 8)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python runtime callback executed")

    with monkeypatch.context() as patch:
        for name in ("__init__", "reshape", "copy", "from_buffer", "min", "max", "__buffer__"):
            patch.setattr(Array, name, forbidden)
        for blk in model.blocks:
            for name in ("forward", "step", "allocate_state", "_extension"):
                patch.setattr(blk, name, forbidden)
        for name in ("embedding", "rms_norm", "linear", "linear_int15"):
            patch.setattr(model._prims, name, forbidden)
        for name in ("empty", "zeros", "array", "asarray", "ascontiguousarray"):
            patch.setattr(np, name, forbidden)
        from mojolearn._mode import PublicOutputsMixin
        patch.setattr(PublicOutputsMixin, "__getattribute__", forbidden)
        out = model.forward(prompt, state)
        generated = model.generate(prompt, 3, greedy=False, seed=7, top_k=1)
    assert out.shape == (1, 2, 256)
    assert np.asarray(generated).tolist() == [[33, 34, 0, 0, 0]]


def test_transitive_runtime_audit_covers_metadata_and_state(tmp_path):
    from mojolearn._verify_native_runtime import audit_native_call
    model = load_fixture(tmp_path, uniform=True)
    parameters = audit_native_call(model.parameters).assert_native()
    assert parameters["lm_head.weight"] is model._head
    prompt = np.array([[33, 34]], dtype="i4")
    state = audit_native_call(model.allocate_state, 1, 8).assert_native()
    audit_native_call(model.forward, prompt, state).assert_native()
    audit_native_call(model.reset_state, state).assert_native()
    stream = audit_native_call(model.stream, prompt, 3, greedy=False, seed=7).assert_native()
    audit_native_call(stream.read, 1).assert_native()
    saved = audit_native_call(stream.state_dict).assert_native()
    audit_native_call(model.resume_generation, saved).assert_native()


def test_native_generation_packs_strided_and_fortran_token_inputs(tmp_path):
    model = load_fixture(tmp_path)
    ids = np.array([[1, 3, 5], [2, 4, 6]], dtype="i4")
    want = np.asarray(model.generate(ids, 2)).copy()
    padded = np.zeros((2, 6), dtype="i8")
    padded[:, ::2] = ids
    for value in (padded[:, ::2], np.asfortranarray(ids), Array.from_buffer(np.asfortranarray(ids))):
        assert np.array_equal(np.asarray(model.generate(value, 2)), want)


def test_tokenizer_native_unicode_specials_and_text_generation(tmp_path, monkeypatch):
    model = load_fixture(tmp_path, uniform=True)
    tokenizer = Tokenizer([bytes([i]) for i in range(256)], [], pattern="gpt2", normalizer="NFC")
    assert tokenizer.decode(tokenizer.encode("e\u0301")) == "é"
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(b"\xff")) == b"\xff"
    with pytest.raises(Exception):
        tokenizer.encode_bytes("not bytes")

    def forbidden(*args, **kwargs):
        raise AssertionError("Python tokenizer composition executed")

    with monkeypatch.context() as patch:
        patch.setattr(tokenizer, "encode", forbidden)
        patch.setattr(tokenizer, "decode", forbidden)
        patch.setattr(tokenizer, "decode_bytes", forbidden)
        assert model.generate_text("!", 3, tokenizer=tokenizer, eos_token_id=0) == ""
        stream = model.stream_text("!", 3, tokenizer=tokenizer)
        assert stream.read(1) == "\x00"
        saved = stream.state_dict()
        assert model.resume_generation(saved, tokenizer=tokenizer).read(9) == "\x00\x00"


def test_text_stream_retains_incomplete_utf8_across_reads_and_resume(tmp_path):
    model = load_fixture(tmp_path, uniform=True)
    # The independent seed0/top-k4 oracle emits 3,1,0. Assign those base-byte
    # ranks to the three UTF-8 bytes of U+20AC. No tokenizer merge is involved.
    first = [0xAC, 0x82, 0x41, 0xE2]
    order = first + [i for i in range(256) if i not in first]
    tok = Tokenizer([bytes([i]) for i in order], [], pattern="gpt2")
    stream = model.stream_text("!", 3, tokenizer=tok, greedy=False, seed=0, top_k=4)
    assert stream.read(1) == ""
    assert stream.read(1) == ""
    saved = stream.state_dict()
    assert stream.read(1) == "€"
    assert model.resume_generation(saved, tokenizer=tok).read(1) == "€"
    other = Tokenizer([bytes([i]) for i in range(256)], [], pattern="gpt2")
    with pytest.raises(Exception, match="tokenizer differs"):
        model.resume_generation(saved, tokenizer=other)
    assert model.generate_text("!", 3, tokenizer=tok, greedy=False, seed=0, top_k=4) == "€"
    short = model.stream_text("!", 2, tokenizer=tok, greedy=False, seed=0, top_k=4)
    assert short.read(1) == ""
    assert short.read(1) == "�"


def test_public_execution_methods_are_one_native_call():
    import ast
    from pathlib import Path
    from mojolearn.models import causal_lm
    tree = ast.parse(Path(causal_lm.__file__).read_text())
    methods = {"forward", "step", "allocate_state", "reset_state", "generate", "stream",
               "generate_text", "stream_text", "resume_generation", "read", "info", "state_dict", "parameters"}
    checked = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in methods:
            body = node.body[1:] if ast.get_docstring(node) else node.body
            assert len(body) == 1 and isinstance(body[0], ast.Return), node.name
            assert isinstance(body[0].value, ast.Call), node.name
            checked.append(node.name)
    assert set(checked) == methods
