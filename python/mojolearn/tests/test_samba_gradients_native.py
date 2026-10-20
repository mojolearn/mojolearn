# SPDX-License-Identifier: Apache-2.0
"""Full native model gradients against an independent explicit operation chain."""
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import SambaConfig, SambaStack, TransformerBlock
from mojolearn import _native_samba_bridge as bridge
from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison


def _model(layers, tied):
    config = SambaConfig(19, 32, layers, n_heads=2, n_kv_heads=1,
                         intermediate=48, tie_embeddings=tied, dropout=0.125)
    rng = np.random.default_rng(143)
    weights = {name: rng.uniform(-0.04, 0.04, shape).astype("f4")
               for name, shape in config.registry()}
    for name in weights:
        if "norm" in name or name.endswith(".D"):
            weights[name].fill(1)
    model = SambaStack(config, weights=weights)
    # Pin the training profile independently of the process inference default.
    for i, kind in enumerate(layers):
        if kind == "attention":
            block_weights = {name: model.arrays[f"layers.{i}.{name}"]
                             for name, shape in config.block_shapes(kind)}
            model._blocks[i] = TransformerBlock(block_weights, n_heads=2, n_kv_heads=1,
                                                numeric_profile="fp32_v1")
    return model


def _reference(model, inputs, targets, items, stream, offset):
    # Testing-only independent operation sequence. No call to the Samba forward,
    # gradient coordinator, registry walk or state reconstruction under test.
    c = model.config
    b, length = inputs.shape
    ids = inputs.reshape(-1)
    x = T.embedding_forward(model.arrays["embed.weight"], ids).reshape((b, length, c.d_model))
    key = None
    if stream is not None:
        x, key = model.generator.dropout(x, c.dropout, stream=stream, offset=offset * c.d_model)
    layer_inputs = []
    for block in model._blocks:
        layer_inputs.append(x)
        x = block.forward(x)
    normalized = T.rms_norm_forward(x, model.arrays["norm_f.weight"], c.norm_eps)
    head = model.arrays["embed.weight" if c.tie_embeddings else "lm_head.weight"]
    logits = T.linear_forward(normalized.reshape((-1, c.d_model)), head)
    count = int(np.count_nonzero(targets != -100)) if items is None else items
    loss, dlogits = T.cross_entropy(logits, targets.reshape(-1), reduction="sum",
                                    num_items=count, return_grad=True)
    dnormalized, dhead = T.linear_backward(dlogits, normalized.reshape((-1, c.d_model)), head)
    dx, dnorm = T.rms_norm_backward(dnormalized.reshape(x.shape), x,
                                    model.arrays["norm_f.weight"], c.norm_eps)
    gradients = {"norm_f.weight": dnorm}
    for i in range(len(model._blocks) - 1, -1, -1):
        result = model._blocks[i].backward(layer_inputs[i], dx)
        dx = result.pop("x")
        gradients.update({f"layers.{i}.{name}": value for name, value in result.items()})
    if key is not None:
        dx = model.generator.dropout_backward(dx, key)
    dembed = T.embedding_backward(dx.reshape((-1, c.d_model)), ids, c.vocab)
    if c.tie_embeddings:
        dembed = T.accumulate_grads([dembed, dhead], tokens=None)
    else:
        gradients["lm_head.weight"] = dhead
    gradients["embed.weight"] = dembed
    return float(loss), [gradients[name].tobytes() for name in model.names]


@pytest.mark.parametrize("layers,tied,stream,items", [
    (["attention"], True, None, None),
    (["mamba3"], False, None, None),
    (["mamba3", "attention"], True, 19, None),
    (["attention", "mamba3"], False, 19, 11),
])
@pytest.mark.parametrize("force_per_op", [False, True])
def test_native_samba_complete_gradient_matches_explicit_chain(monkeypatch, layers, tied, stream, items, force_per_op):
    model = _model(layers, tied)
    inputs = np.array([[1, 3, 5], [2, 4, 6]], dtype="i4")
    targets = np.array([[3, 5, 7], [4, -100, 8]], dtype="i4")
    expected_loss, expected_gradients = _reference(model, inputs, targets, items, stream, 7)
    before_rng = model.generator.state_dict()
    if force_per_op:
        entries = dict(vars(bridge.CONTEXT["entries"]))
        for name in ("samba_head_loss", "training_chunked_lm_head_enabled",
                     "samba_afn_norm_head_forward", "samba_afn_tail_train",
                     "samba_afn_embedding_backward_tied"):
            entries.pop(name, None)
        monkeypatch.setitem(bridge.CONTEXT, "entries", SimpleNamespace(**entries))
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(SambaStack, "_forward", lambda *a, **k: pytest.fail("Python model walk"))
        patch.setattr(SambaStack, "_block", lambda *a: pytest.fail("Python layer selection"))
        patch.setattr(T, "embedding_forward", lambda *a: pytest.fail("Python primitive callback"))
        actual = audit_native_call(model.loss_and_grads, inputs, targets, num_items=items,
                                   dropout_stream=stream, token_offset=7).assert_native()
    assert np.float32(actual[0]).tobytes() == np.float32(expected_loss).tobytes()
    assert len(actual[1]) == len(expected_gradients)
    for name, got, expected in zip(model.names, actual[1], expected_gradients):
        assert got.tobytes() == expected, name
    assert model.generator.state_dict() == before_rng


@pytest.mark.parametrize("length", [2, 3, 5])
def test_native_samba_worker_state_gradient_matches_live_model(monkeypatch, length):
    model = _model(["attention", "mamba3"], True)
    inputs = np.arange(1, length + 1, dtype="i4").reshape(1, length)
    targets = np.arange(2, length + 2, dtype="i4").reshape(1, length)
    targets[0, 1] = -100
    state = model.state_dict()
    expected = _reference(model, inputs, targets, None, 27, 9)
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(SambaStack, "__init__", lambda *a, **k: pytest.fail("Python model construction"))
        actual = audit_native_call(bridge.RUNTIME.samba_native_call, "gradients",
                                   (state, inputs, targets, 27, 9), bridge.CONTEXT).assert_native()
    assert np.float32(actual[0]).tobytes() == np.float32(expected[0]).tobytes()
    assert [value.tobytes() for value in actual[1]] == expected[1]
    assert state["rng"] == model.generator.state_dict()
