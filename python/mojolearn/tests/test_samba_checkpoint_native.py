# SPDX-License-Identifier: Apache-2.0
"""Public checkpoint byte compatibility, complete resume and corruption audits."""
import copy
import hashlib
import json
import struct

import numpy as np
import pytest

from mojolearn import SambaConfig, SambaStack
from mojolearn import _samba_checkpoint as reference_format
from mojolearn import _training_impl as training
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison
from mojolearn.tests.test_samba_state_native import ARRAY_FIELDS, _reference_state, _assert_state_equal


def _model():
    config = SambaConfig(13, 8, ["attention"], n_heads=2, n_kv_heads=1,
                         intermediate=11, dropout=0.125, tie_embeddings=False)
    model = SambaStack(config, generator=training.Generator(0xFEDCBA9876543210),
                       lr=0.003, lr_schedule=training.WarmupCosineLR(0.003, 3, 17),
                       max_norm=0.4, betas=(0.8, 0.95), weight_decay=0.02)
    inputs = np.array([[1, 3, 5], [2, 4, 6]], dtype="i4")
    targets = np.array([[3, 5, 7], [4, -100, 8]], dtype="i4")
    model.train_step(inputs, targets)
    return model, inputs, targets


def _assert_refusal(result, error):
    assert isinstance(result.exception, error), result.report()
    for name in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads"):
        assert not result.report()[name], result.report()


def _legacy_bytes(state):
    payload = dict(state)
    for name in ARRAY_FIELDS:
        array = state[name]
        payload[name] = {"dtype": "<i4" if name == "buf_initialized" else "<f4",
                         "shape": list(array.shape), "hex": array.tobytes().hex()}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("ascii")
    envelope = {"schema": "mojolearn.samba-stack-json-checkpoint.v1", "payload": payload,
                "payload_sha256": hashlib.sha256(canonical).hexdigest()}
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


@pytest.mark.parametrize("path_kind", ["pathlib", "str", "bytes"])
def test_public_streamed_checkpoint_matches_independent_format_and_resume(tmp_path, monkeypatch, path_kind):
    model, inputs, targets = _model()
    state = _reference_state(model)
    oracle_path = tmp_path / "reference.bin"
    expected_digest = reference_format.save(oracle_path, state)
    path = tmp_path / "native.bin"
    argument = path if path_kind == "pathlib" else str(path)
    if path_kind == "bytes":
        argument = bytes(argument, "utf-8")
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(reference_format, "save", lambda *a: pytest.fail("Python checkpoint writer"))
        patch.setattr(reference_format, "load", lambda *a: pytest.fail("Python checkpoint reader"))
        digest = audit_native_call(model.save_checkpoint, argument).assert_native()
        restored = audit_native_call(SambaStack.from_checkpoint, argument).assert_native()
    assert path.read_bytes() == oracle_path.read_bytes()
    assert digest == expected_digest == hashlib.sha256(path.read_bytes()).hexdigest()
    _assert_state_equal(_reference_state(restored), state)
    expected_metrics = model.train_step(inputs, targets)
    actual_metrics = audit_native_call(restored.train_step, inputs, targets).assert_native()
    assert actual_metrics == expected_metrics
    _assert_state_equal(_reference_state(restored), _reference_state(model))
    assert not list(tmp_path.glob("native.bin.tmp-*"))


def test_public_legacy_json_checkpoint_preserves_complete_state(tmp_path, monkeypatch):
    model, _, _ = _model()
    state = _reference_state(model)
    state["rng"]["counter"] = 1 << 32
    path = tmp_path / "legacy.json"
    path.write_bytes(_legacy_bytes(state))
    with monkeypatch.context() as patch:
        _poison(patch)
        restored = audit_native_call(SambaStack.from_checkpoint, path).assert_native()
    _assert_state_equal(_reference_state(restored), state)
    result = audit_native_call(restored.train_step, np.array([[1, 2]], dtype="i4"),
                               np.array([[2, 3]], dtype="i4"))
    _assert_refusal(result, RuntimeError)
    _assert_state_equal(_reference_state(restored), state)


def _change_header(raw, operation):
    start = len(reference_format.MAGIC)
    length = struct.unpack("<Q", raw[start:start+8])[0]
    header = json.loads(raw[start+8:start+8+length])
    operation(header)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return (raw[:start] + struct.pack("<Q", len(encoded)) + encoded
            + hashlib.sha256(encoded).digest() + raw[start+8+length+32:])


@pytest.mark.parametrize("corruption", ["header_hash", "array_hash", "truncated", "trailing", "descriptor", "registry"])
def test_public_checkpoint_rejects_corruption_before_model_publication(tmp_path, corruption):
    model, _, _ = _model()
    path = tmp_path / "corrupt.bin"
    reference_format.save(path, _reference_state(model))
    raw = path.read_bytes()
    if corruption == "header_hash":
        data = bytearray(raw)
        data[len(reference_format.MAGIC) + 8] ^= 1
        raw = bytes(data)
    elif corruption == "array_hash":
        raw = raw[:-1] + bytes([raw[-1] ^ 1])
    elif corruption == "truncated":
        raw = raw[:-7]
    elif corruption == "trailing":
        raw += b"unexpected"
    elif corruption == "descriptor":
        raw = _change_header(raw, lambda h: h["arrays"][0].update(shape=[1 << 40], nbytes=1 << 42))
    else:
        raw = _change_header(raw, lambda h: h["state"]["registry"][0].update(offset=1))
    path.write_bytes(raw)
    before = _reference_state(model)
    _assert_refusal(audit_native_call(SambaStack.from_checkpoint, path), ValueError)
    _assert_state_equal(_reference_state(model), before)


def test_failed_checkpoint_save_preserves_existing_file_and_leaves_no_temporary(tmp_path):
    model, _, _ = _model()
    path = tmp_path / "existing.bin"
    path.write_bytes(b"previous complete checkpoint")
    original = model.flat.shape
    model.flat.shape = (original[0] - 1,)
    try:
        _assert_refusal(audit_native_call(model.save_checkpoint, path), ValueError)
    finally:
        model.flat.shape = original
    assert path.read_bytes() == b"previous complete checkpoint"
    assert not list(tmp_path.glob("existing.bin.tmp-*"))


def test_checkpoint_missing_path_and_python_path_callback_refusals(tmp_path):
    _assert_refusal(audit_native_call(SambaStack.from_checkpoint, tmp_path / "missing"), FileNotFoundError)

    class CallbackPath:
        def __fspath__(self):
            pytest.fail("Python path callback executed")

    _assert_refusal(audit_native_call(SambaStack.from_checkpoint, CallbackPath()), TypeError)
