# SPDX-License-Identifier: Apache-2.0
"""CPU SambaInference public boundaries against independent host primitives."""
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import SambaConfig, SambaInference, lowbit
from mojolearn import neural_inference as inference
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison
from mojolearn.tests.test_samba_inference_native import _assert_layers
from mojolearn.tests.test_samba_checkpoint_native import _model, _legacy_bytes
from mojolearn.tests.test_samba_state_native import _reference_state
from mojolearn import _samba_checkpoint as reference_format


def _case(layers, format="float32"):
    dm = 32 if "mamba3" in layers else 8
    config = SambaConfig(19, dm, layers, n_heads=2, n_kv_heads=1,
                         intermediate=48 if dm == 32 else 11, tie_embeddings=False)
    rng = np.random.default_rng(412)
    weights = {name: rng.uniform(-0.04, 0.04, shape).astype("f4")
               for name, shape in config.registry()}
    for name in weights:
        if "norm" in name or name.endswith(".D"):
            weights[name].fill(1)
    packed = lowbit.pack(weights, format)
    unpacked, _ = lowbit.unpack(packed)
    blocks = []
    for i, kind in enumerate(layers):
        layer = {name: unpacked[f"layers.{i}.{name}"] for name, _ in config.block_shapes(kind)}
        blocks.append(inference.Mamba3BlockInference(layer) if kind == "mamba3" else
                      inference.TransformerBlockInference(layer, n_heads=2, n_kv_heads=1))
    return config, packed, unpacked, blocks


def _reference(config, weights, blocks, ids, states=None, step=False):
    # Independent testing recipe explicitly selects the original CPU host ABI,
    # even when this test runs inside a GPU-equipped package installation.
    entry = inference._binding()
    if ids.ndim == 1:
        ids = ids[:, None]
    ids = np.asarray(ids, dtype="i4", order="C")
    b, length = ids.shape
    d, v, n = config.d_model, config.vocab, ids.size
    x = np.empty((b, length, d), dtype="f4")
    embedding = np.asarray(weights["embed.weight"], dtype="f4", order="C")
    entry.embedding_forward([x.ctypes.data, embedding.ctypes.data, ids.ctypes.data], [n, v, d])
    for i, block in enumerate(blocks):
        x = block.forward(x) if states is None else (block.step(x, states[i]) if step else block.forward(x, states[i]))
    x = np.asarray(x)
    normalized = np.empty_like(x)
    norm = np.asarray(weights["norm_f.weight"])
    entry.rms_norm_forward([normalized.ctypes.data, x.ctypes.data, norm.ctypes.data], [n, d, config.norm_eps])
    head = np.asarray(weights["embed.weight" if config.tie_embeddings else "lm_head.weight"])
    output = np.empty((b, length, v), dtype="f4")
    entry.linear_forward([output.ctypes.data, normalized.ctypes.data, head.ctypes.data], [n, v, d])
    return output


@pytest.mark.parametrize("layers,format", [(["attention"], "float32"),
                                          (["mamba3"], "float32"),
                                          (["mamba3", "attention"], "bfloat16"),
                                          (["attention", "mamba3"], "int8")])
def test_cpu_samba_constructor_forward_and_parameter_copies_are_native(monkeypatch, layers, format):
    config, packed, weights, blocks = _case(layers, format)
    ids = np.array([[1, 3, 5], [2, 4, 6]], dtype="i4")
    expected = _reference(config, weights, blocks, ids)
    model = SambaInference.__new__(SambaInference)
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(lowbit, "unpack", lambda *a: pytest.fail("Python unpack"))
        patch.setattr(inference, "_binding", lambda: pytest.fail("Python binding selection"))
        audit_native_call(model.__init__, config, packed).assert_native()
        actual = audit_native_call(model.forward, ids).assert_native()
        copied = audit_native_call(model.parameters).assert_native()
        description = audit_native_call(model.__repr__).assert_native()
    assert actual.tobytes() == expected.tobytes()
    assert model.weight_format == format
    assert not hasattr(model, "optimizer")
    assert "SambaInference(layers=" in description
    for name in weights:
        assert copied[name].tobytes() == weights[name].tobytes()
        assert copied[name]._addr != model._w[name]._addr
    assert all(type(block) is (inference.Mamba3BlockInference if kind == "mamba3" else inference.TransformerBlockInference)
               for block, kind in zip(model._blocks, layers))
    # Public aliases execute the same complete native boundary.
    assert audit_native_call(model.logits, ids).assert_native().tobytes() == expected.tobytes()
    assert audit_native_call(model.__call__, ids).assert_native().tobytes() == expected.tobytes()


@pytest.mark.parametrize("foreign", [False, True])
def test_cpu_samba_stateful_prefill_decode_preserve_full_cache_and_foreign_aliases(foreign):
    config, packed, weights, blocks = _case(["mamba3", "attention"])
    model = SambaInference(config, packed)
    states = [block.allocate_state(2) if kind == "mamba3" else block.allocate_state(2, 8)
              for block, kind in zip(blocks, config.layers)]
    state = audit_native_call(model.allocate_state, 2, 8).assert_native()
    aliases = []
    if foreign:
        for layer in state.layers:
            for name, value in list(vars(layer).items()):
                if hasattr(value, "tobytes"):
                    array = np.asarray(value).copy()
                    setattr(layer, name, array)
                    aliases.append((layer, name, array))
    for ids, step in [(np.array([[1, 3, 5], [2, 4, 6]], dtype="i4"), False),
                      (np.array([7, 8], dtype="i4"), True)]:
        expected = _reference(config, weights, blocks, ids, states, step)
        output = audit_native_call(model.step if step else model.forward, ids, state).assert_native()
        assert output.tobytes() == expected.tobytes()
        _assert_layers(state.layers, states)
        for layer, name, array in aliases:
            assert getattr(layer, name) is array


@pytest.mark.parametrize("legacy", [False, True])
def test_cpu_samba_checkpoint_load_matches_host_weight_recipe(tmp_path, monkeypatch, legacy):
    training_model, inputs, _ = _model()
    state = _reference_state(training_model)
    path = tmp_path / "model.ckpt"
    if legacy:
        path.write_bytes(_legacy_bytes(state))
    else:
        reference_format.save(path, state)
    config = training_model.config
    weights = training_model.parameters()
    layer = {name: weights[f"layers.0.{name}"] for name, _ in config.block_shapes("attention")}
    blocks = [inference.TransformerBlockInference(layer, n_heads=2, n_kv_heads=1)]
    expected = _reference(config, weights, blocks, inputs)
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(reference_format, "load", lambda *a: pytest.fail("Python checkpoint reader"))
        model = audit_native_call(SambaInference.from_checkpoint, path).assert_native()
        output = audit_native_call(model.forward, inputs).assert_native()
    assert output.tobytes() == expected.tobytes()
    assert not hasattr(model, "optimizer")


@pytest.mark.parametrize("method", ["loss", "train_step"])
def test_cpu_samba_training_refusal_is_native(method):
    config, packed, _, _ = _case(["attention"])
    model = SambaInference(config, packed)
    ids = np.array([[1, 2]], dtype="i4")
    result = audit_native_call(getattr(model, method), ids, ids)
    assert isinstance(result.exception, NotImplementedError), result.report()
    for name in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads"):
        assert not result.report()[name], result.report()
