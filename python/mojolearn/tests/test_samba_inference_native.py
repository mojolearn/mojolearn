# SPDX-License-Identifier: Apache-2.0
"""Public Samba inference against an independent primitive/block sequence."""
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import SambaConfig, SambaStack
from mojolearn._samba_impl import SambaState
from mojolearn import _native_samba_bridge as bridge
from mojolearn import _training_impl as training
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison


def _models(layers, dropout=0.0, tied=True):
    dm = 32 if "mamba3" in layers else 8
    config = SambaConfig(19, dm, layers, n_heads=2, n_kv_heads=1,
                         intermediate=48 if dm == 32 else 11,
                         tie_embeddings=tied, dropout=dropout)
    rng = np.random.default_rng(428)
    weights = {name: rng.uniform(-0.04, 0.04, shape).astype("f4")
               for name, shape in config.registry()}
    for name in weights:
        if "norm" in name or name.endswith(".D"):
            weights[name].fill(1)
    return [SambaStack(config, weights=weights) for _ in range(2)]


def _reference(model, ids, layers=None, step=False):
    c = model.config
    if ids.ndim == 1:
        ids = ids[:, None]
    b, length = ids.shape
    x = training.embedding_forward(model.arrays["embed.weight"], ids.reshape(-1)).reshape((b, length, c.d_model))
    for i, block in enumerate(model._blocks):
        if layers is None:
            x = block.forward(x)
        elif step:
            x = block.step(x, layers[i])
        else:
            x = block.forward(x, layers[i])
    normalized = training.rms_norm_forward(x, model.arrays["norm_f.weight"], c.norm_eps)
    head = model.arrays["embed.weight" if c.tie_embeddings else "lm_head.weight"]
    return training.linear_forward(normalized.reshape((-1, c.d_model)), head).reshape((b, length, c.vocab))


def _layer_state(model, batch, capacity):
    return [block.allocate_state(batch) if kind == "mamba3" else block.allocate_state(batch, capacity)
            for kind, block in zip(model.config.layers, model._blocks)]


def _assert_layers(actual, expected):
    for a, b in zip(actual, expected):
        assert type(a) is type(b)
        # Full resumable state bytes, including the Mamba working chunk,
        # pending state and Transformer positions; no output-only assertion.
        assert set(vars(a)) == set(vars(b))
        for name, reference in vars(b).items():
            value = vars(a)[name]
            if hasattr(reference, "tobytes"):
                assert value.shape == reference.shape, name
                assert value.tobytes() == reference.tobytes(), name
            else:
                assert value == reference, name


@pytest.mark.parametrize("layers,dropout,tied", [
    (["attention"], 0.0, True),
    (["attention"], 0.125, False),
    (["mamba3"], 0.0, False),
    (["mamba3", "attention"], 0.125, True),
])
def test_public_forward_loss_match_primitive_recipe(monkeypatch, layers, dropout, tied):
    reference, model = _models(layers, dropout, tied)
    inputs = np.array([[1, 3, 5], [2, 4, 6]], dtype="i8")
    expected = _reference(reference, inputs)
    targets = np.array([[3, -100, 7], [4, 6, 8]], dtype="i4")
    expected_loss = training.cross_entropy(expected.reshape((-1, 19)), targets.reshape(-1))
    before_rng = model.generator.state_dict()
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(SambaStack, "_forward", lambda *a, **k: pytest.fail("Python model orchestration"))
        patch.setattr(SambaStack, "_forward_resident", lambda *a, **k: pytest.fail("Python resident orchestration"))
        actual = audit_native_call(model.forward, inputs).assert_native()
        loss = audit_native_call(model.loss, inputs, targets).assert_native()
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert np.float32(loss).tobytes() == np.float32(expected_loss).tobytes()
    assert model.generator.state_dict() == before_rng
    if dropout == 0.0 and hasattr(bridge.CONTEXT["entries"], "samba_resident_enabled"):
        if bridge.CONTEXT["entries"].samba_resident_enabled():
            assert model._resident_handle is not None
            handle = model._resident_handle
            audit_native_call(model.forward, inputs).assert_native()
            assert model._resident_handle is handle


@pytest.mark.parametrize("layers", [["attention"], ["mamba3"], ["mamba3", "attention"]])
def test_public_stateful_forward_and_dedicated_step_preserve_complete_state(monkeypatch, layers):
    reference, model = _models(layers)
    expected_states = _layer_state(reference, 2, 8)
    actual = audit_native_call(model.allocate_state, 2, 8).assert_native()
    assert isinstance(actual, SambaState)
    assert actual.batch_size == 2 and actual.max_tokens == 8
    _assert_layers(actual.layers, expected_states)
    for ids, step in [(np.array([[1, 2, 3], [4, 5, 6]], dtype="i4"), False),
                      (np.array([7, 8], dtype="i4"), True),
                      (np.array([[9, 10], [11, 12]], dtype="i4"), False)]:
        expected = _reference(reference, ids, expected_states, step)
        with monkeypatch.context() as patch:
            _poison(patch)
            patch.setattr(SambaStack, "_forward_state", lambda *a, **k: pytest.fail("Python state traversal"))
            value = audit_native_call(model.step if step else model.forward, ids, actual).assert_native()
        assert value.shape == ((2, 19) if step else expected.shape)
        assert value.tobytes() == expected.tobytes()
        _assert_layers(actual.layers, expected_states)
        for native, old, kind in zip(model._blocks, reference._blocks, layers):
            if kind == "mamba3":
                for name in ("h_last_", "k_last_", "v_last_", "theta_last_"):
                    assert getattr(native, name).tobytes() == getattr(old, name).tobytes()


@pytest.mark.parametrize("layers", [["attention"], ["mamba3", "attention"]])
def test_public_ragged_forward_ignores_invalid_padding_and_zeros_output(layers):
    reference, model = _models(layers, dropout=0.125)
    ids = np.array([[1, 2, -900], [3, -200, 10000]], dtype="i8")
    lengths = np.array([2, 1], dtype="i8")
    expected = np.zeros((2, 3, 19), dtype="f4")
    for i, n in enumerate(lengths):
        expected[i:i+1, :n] = np.asarray(_reference(reference, ids[i:i+1, :n]))
    actual = audit_native_call(model.forward, ids, lengths=lengths).assert_native()
    assert actual.tobytes() == expected.tobytes()
    assert not np.asarray(actual).view("u4")[0, 2].any()
    assert not np.asarray(actual).view("u4")[1, 1:].any()


@pytest.mark.parametrize("operation,args", [("step", (np.array([1], dtype="i4"), None)),
                                           ("allocate_state", (0, 2)),
                                           ("forward", (np.array([[999]], dtype="i4"),))])
def test_public_inference_refusals_are_native(operation, args):
    _, model = _models(["attention"])
    result = audit_native_call(getattr(model, operation), *args)
    assert isinstance(result.exception, ValueError), result.report()
    report = result.report()
    for name in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads"):
        assert not report[name], report
