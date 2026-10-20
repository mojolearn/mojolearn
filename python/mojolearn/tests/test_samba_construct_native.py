# SPDX-License-Identifier: Apache-2.0
"""Constructor behavior against independent existing initializer kernels."""
import numpy as np
import pytest

from mojolearn import SambaConfig, SambaStack
from mojolearn import _samba_impl as samba
from mojolearn import _training_impl as training
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison


@pytest.mark.parametrize("layers,dm,intermediate,tied", [
    (["attention"], 8, 7, True),
    (["attention"], 16, 13, False),
    (["mamba3", "attention"], 32, 48, False),
])
@pytest.mark.parametrize("lazy", [True, False])
def test_native_constructor_preserves_seeded_initialization_and_views(
        monkeypatch, layers, dm, intermediate, tied, lazy):
    monkeypatch.setenv("MOJOLEARN_OPT_LAZY_MOMENTS", "1" if lazy else "0")
    config = SambaConfig(19, dm, layers, n_heads=2, n_kv_heads=1,
                         intermediate=intermediate, tie_embeddings=tied)
    reference_generator = training.Generator(0xFEDCBA9876543210)
    reference_generator.counter = 13
    expected = {}
    # Testing-only legacy initializer recipe. Its public RNG operations still
    # use the independently established scalar math and RNG kernel ABI.
    for name, shape in config.registry():
        if name in ("embed.weight", "lm_head.weight"):
            value = reference_generator.normal(shape, 0.0, 0.02)
        elif name.endswith(".weight") and len(shape) == 2:
            value = reference_generator.kaiming_uniform(shape, fan_in=shape[1])
        elif name.endswith(".dt_bias"):
            value = reference_generator.uniform(shape, -4.0, -2.0)
        else:
            value = np.ones(shape, dtype="f4")
        expected[name] = value.tobytes()
    generator = training.Generator(reference_generator.seed)
    generator.counter = 13
    model = SambaStack.__new__(SambaStack)
    with monkeypatch.context() as patch:
        _poison(patch)
        patch.setattr(samba, "_init_tensor", lambda *a: pytest.fail("Python initializer"))
        patch.setattr(training.AdamW, "__init__", lambda *a, **k: pytest.fail("Python optimizer construction"))
        patch.setattr(training.Generator, "normal", lambda *a: pytest.fail("Python RNG"))
        patch.setattr(training.Generator, "uniform", lambda *a: pytest.fail("Python RNG"))
        patch.setattr(training.Generator, "kaiming_uniform", lambda *a, **k: pytest.fail("Python RNG"))
        audit_native_call(model.__init__, config, generator=generator, accumulation_steps=2).assert_native()
        parameters = audit_native_call(model.parameters).assert_native()
    assert model.generator is generator
    assert generator.state_dict() == reference_generator.state_dict()
    assert list(parameters) == list(expected)
    assert list(model.arrays) == list(expected)
    assert model.optimizer.accumulation_steps == 2
    assert model.optimizer._pristine is lazy
    assert (model.optimizer._m is None) is lazy
    assert (model.optimizer._v is None) is lazy
    for i, (name, shape) in enumerate(config.registry()):
        value = parameters[name]
        assert value.tobytes() == expected[name], name
        assert value is model.arrays[name] is model.optimizer.params[i]
        assert value._base is model.flat
        assert value._addr == model.flat._addr + 4 * model.offsets[i]
        assert value.shape == shape
    for block in model._blocks:
        assert block.numeric_profile == "fp32_v1"


def test_native_explicit_weight_constructor_and_load_preserve_aliases(monkeypatch):
    config = SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11,
                         tie_embeddings=False)
    rng = np.random.default_rng(17)
    weights = {name: np.asfortranarray(rng.normal(0, 0.02, shape).astype("f4"))
               for name, shape in config.registry()}
    expected = {name: value.tobytes(order="C") for name, value in weights.items()}
    model = SambaStack.__new__(SambaStack)
    with monkeypatch.context() as patch:
        _poison(patch)
        audit_native_call(model.__init__, config, weights=weights).assert_native()
    assert model.generator.state_dict() == {"seed": 0, "counter": 0}
    assert {name: value.tobytes() for name, value in model.parameters().items()} == expected
    before = model.parameters()
    replacement = {name: np.full(shape, 0.125, dtype="f4") for name, shape in config.registry()}
    with monkeypatch.context() as patch:
        _poison(patch)
        audit_native_call(model.load_weights, replacement).assert_native()
    for name, value in model.parameters().items():
        assert value is before[name]
        assert value.tobytes() == replacement[name].tobytes()


def test_native_constructor_preserves_opaque_custom_schedule(monkeypatch):
    class CustomSchedule:
        def lr_at(self, step):
            raise AssertionError("constructor executed schedule")

        def config(self):
            raise AssertionError("constructor serialized schedule")

    config = SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11)
    schedule = CustomSchedule()
    weights = {name: np.ones(shape, dtype="f4") for name, shape in config.registry()}
    model = SambaStack.__new__(SambaStack)
    audit_native_call(model.__init__, config, weights=weights,
                      lr_schedule=schedule).assert_native()
    assert model.optimizer.lr_schedule is schedule


@pytest.mark.parametrize("corruption", ["keys", "shape", "dtype", "finite"])
def test_native_weight_admission_reports_invalid_inputs(corruption):
    config = SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11)
    weights = {name: np.ones(shape, dtype="f4") for name, shape in config.registry()}
    if corruption == "keys":
        weights.pop("embed.weight")
    elif corruption == "shape":
        weights["embed.weight"] = np.ones((13, 9), dtype="f4")
    elif corruption == "dtype":
        weights["embed.weight"] = weights["embed.weight"].astype("f8")
    else:
        weights["embed.weight"][0, 0] = np.nan
    model = SambaStack.__new__(SambaStack)
    result = audit_native_call(model.__init__, config, weights=weights)
    assert isinstance(result.exception, TypeError if corruption == "dtype" else ValueError), result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work",
                "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
