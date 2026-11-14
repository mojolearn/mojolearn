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
        actual = audit_native_call(model.forward, inputs).assert_native()
        loss = audit_native_call(model.loss, inputs, targets).assert_native()
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert np.float32(loss).tobytes() == np.float32(expected_loss).tobytes()
    assert model.generator.state_dict() == before_rng
    program = getattr(model, "_samba_program", None)
    if program is not None:
        # A device build runs the stack program, opened once per shape and
        # kept as its native handle owner (no Python program object).
        assert type(program).__name__ == "_SambaProgramHandle"
        audit_native_call(model.forward, inputs).assert_native()
        assert model._samba_program is program


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


def test_stateless_forward_leaves_block_reports_untouched():
    # lane perf-neural-sessions (2026-11-13): a stack's Mamba-3 blocks discard
    # their reports, so the stateless forward requests none (as the 0.8.55
    # resident forward did): no report arrays are allocated or published.
    _, model = _models(["mamba3", "attention"])
    block = model._blocks[0]
    before = {name: vars(block).get(name) for name in ("h_last_", "k_last_", "v_last_", "theta_last_")}
    audit_native_call(model.forward, np.array([[1, 2, 3], [4, 5, 6]], dtype="i4")).assert_native()
    for name, value in before.items():
        assert vars(block).get(name) is value, name


def test_device_stateless_forward_takes_the_resident_route():
    # lane restore-samba-resident-forward (2026-11-13): on a device build the
    # stateless forward runs the resident session's forward (descriptor op
    # 16), the measured samba-forward winner; the stack program is its B arm
    # (-D MOJOLEARN_SAMBA_PROGRAM_FORWARD) and serves a dropout stack. The
    # door records the arm it ran; the host route (CPU builds) records none.
    _, model = _models(["mamba3", "attention"])
    ids = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4")
    audit_native_call(model.forward, ids).assert_native()
    route = vars(model).get("_samba_forward_route")
    if route is None:
        pytest.skip("host route: no device forward route on this build")
    if model.numeric_mode != "identical":
        assert route == "program"
        return
    assert route == "resident"
    assert vars(model).get("_samba_program") is None
    # The session is the one the train step uses (op 12 owner).
    assert vars(model).get("_resident_handle") is not None
    _, dropped = _models(["attention"], dropout=0.125)
    audit_native_call(dropped.forward, ids).assert_native()
    assert vars(dropped).get("_samba_forward_route") == "program"


def test_resident_forward_uploads_unchanged_weights_once():
    # lane fix-samba-weight-upload (2026-11-13): the resident forward keys its
    # device registry by the registry address and that registry's epoch
    # (core/weight_epoch.mojo RegistryKey). Two forwards with unchanged weights make one
    # registry upload between them; every registry writer (train_step,
    # load_state_dict) stamps the registry, so the next forward uploads again and
    # its logits are the new weights' bits, never stale ones.
    model, twin = _models(["mamba3", "attention"])
    ids = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4")
    targets = np.array([[2, 3, 4], [5, 6, 7]], dtype="i4")
    first = np.asarray(audit_native_call(model.forward, ids).assert_native()).copy()
    if vars(model).get("_samba_forward_route") != "resident":
        pytest.skip("no resident forward on this build")
    uploads = vars(model)["_samba_resident_uploads"]
    second = np.asarray(audit_native_call(model.forward, ids).assert_native()).copy()
    assert vars(model)["_samba_resident_uploads"] == uploads, "unchanged weights were uploaded again"
    assert second.tobytes() == first.tobytes()
    saved = model.state_dict()
    model.train_step(ids, targets)
    twin.train_step(ids, targets)
    trained = np.asarray(model.forward(ids)).copy()
    assert vars(model)["_samba_resident_uploads"] > uploads, "train_step did not invalidate the resident registry"
    # the twin's fresh session holds the trained weights: the same bits
    assert trained.tobytes() == np.asarray(twin.forward(ids)).tobytes()
    assert trained.tobytes() != first.tobytes()
    uploads = vars(model)["_samba_resident_uploads"]
    model.load_state_dict(saved)
    restored = np.asarray(model.forward(ids)).copy()
    assert vars(model)["_samba_resident_uploads"] > uploads, "load_state_dict did not invalidate the resident registry"
    assert restored.tobytes() == first.tobytes()


def test_resident_forward_two_models_alternating_upload_once_each():
    # lane fix-samba-weight-epoch-per-model (2026-11-13): each stack's resident
    # registry is keyed by its own registry's epoch (core/weight_epoch.mojo
    # RegistryKey), not the process epoch. Two models alternating forwards
    # upload once each at their first call and never after; a train_step on A
    # re-uploads A once and leaves B's copy current.
    a, b = _models(["mamba3", "attention"])
    ids = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4")
    targets = np.array([[2, 3, 4], [5, 6, 7]], dtype="i4")
    first_a = np.asarray(a.forward(ids)).copy()
    if vars(a).get("_samba_forward_route") != "resident":
        pytest.skip("no resident forward on this build")
    first_b = np.asarray(b.forward(ids)).copy()
    ua = vars(a)["_samba_resident_uploads"]
    ub = vars(b)["_samba_resident_uploads"]
    for _ in range(3):
        assert np.asarray(a.forward(ids)).tobytes() == first_a.tobytes()
        assert np.asarray(b.forward(ids)).tobytes() == first_b.tobytes()
    assert vars(a)["_samba_resident_uploads"] == ua, "A re-uploaded unchanged weights while B ran"
    assert vars(b)["_samba_resident_uploads"] == ub, "B re-uploaded unchanged weights while A ran"
    a.train_step(ids, targets)
    ua = vars(a)["_samba_resident_uploads"]
    trained = np.asarray(a.forward(ids)).copy()
    assert vars(a)["_samba_resident_uploads"] == ua + 1, "A's write did not re-upload A exactly once"
    assert trained.tobytes() != first_a.tobytes()
    assert np.asarray(b.forward(ids)).tobytes() == first_b.tobytes()
    assert vars(b)["_samba_resident_uploads"] == ub, "a write to A re-uploaded B"


class _HookedInt(int):
    def __index__(self):  # pragma: no cover - must never run
        raise AssertionError("Python hook ran inside a native Samba forward")


def test_read_only_forward_still_refuses_hooks_in_fields_it_reads():
    # The read-only admission walks only the fields forward reads; each of them
    # keeps the full walk's refusal.
    _, model = _models(["attention"])
    ids = np.array([[1, 2, 3]], dtype="i4")
    audit_native_call(model.forward, ids).assert_native()
    vars(model)["n_total"] = _HookedInt(model.n_total)
    with pytest.raises(TypeError):
        model.forward(ids)
