# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `"gpu"` and `"cpu"` on `models.CausalLM`, at the Python boundary.

The rule is the one every class with `device=` follows: None is this installation's device, an explicit
`device="gpu"` on a CPU-only installation is refused in words that name `device="cpu"`, `device="cpu"` always runs.
`"auto"` is CausalLM's older spelling of None and stays accepted. The value is read once, at construction or load,
into the model's route (`model.device` is the route word, `"cpu"` or `"gpu"`). On a GPU installation the same
checkpoint on each device, in one process, returns the same logits bits in the identical profile. What needs two
devices is skipped by name on a CPU-only installation.
"""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
from mojolearn.models import CausalLM

FAMILIES = ["llama", "mamba", "mamba2"]
TWO_DEVICES = mojolearn.vendor() != "cpu"
INSTALLED = "gpu" if TWO_DEVICES else "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")
one_device = pytest.mark.skipif(TWO_DEVICES, reason="the refusal is the CPU-only installation's")


def checkpoint(tmp_path, family):
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)
    return tmp_path


def test_device_defaults_to_none_on_both_constructors():
    assert inspect.signature(CausalLM.__init__).parameters["device"].default is None
    assert inspect.signature(CausalLM.load).parameters["device"].default is None


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("given", [None, "auto"])
def test_none_and_auto_are_this_installations_device(tmp_path, family, given):
    model = CausalLM.load(checkpoint(tmp_path, family), device=given)
    assert model.device == INSTALLED
    assert CausalLM.load(tmp_path).device == INSTALLED


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_always_runs(tmp_path, family):
    assert CausalLM.load(checkpoint(tmp_path, family), device="cpu").device == "cpu"


@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None"])
def test_unknown_device_is_refused_at_load(tmp_path, bad):
    with pytest.raises(ValueError, match="device must be 'auto', 'cpu', 'gpu' or None"):
        CausalLM.load(checkpoint(tmp_path, "llama"), device=bad)


@pytest.mark.parametrize("bad", [0, False, b"cpu"])
def test_a_device_that_is_not_a_name_is_refused_at_load(tmp_path, bad):
    with pytest.raises(TypeError):
        CausalLM.load(checkpoint(tmp_path, "llama"), device=bad)


@one_device
def test_explicit_gpu_on_a_cpu_only_installation_is_refused_in_the_shared_words(tmp_path):
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn") as refused:
        CausalLM.load(checkpoint(tmp_path, "llama"), device="gpu")
    assert "device='cpu'" in str(refused.value)


@two_devices
@pytest.mark.parametrize("family", FAMILIES)
def test_the_same_checkpoint_returns_the_same_logits_bits_on_each_device(tmp_path, family):
    path = checkpoint(tmp_path, family)
    gpu = CausalLM.load(path, device="gpu", numeric_profile="fp32_v1")
    cpu = CausalLM.load(path, device="cpu", numeric_profile="fp32_v1")
    assert (gpu.device, cpu.device) == ("gpu", "cpu")
    ids = np.array([[1, 7, 3, 11, 5], [2, 9, 2, 4, 8]], dtype="<i4")     # the verify fixture's ids
    left = np.asarray(gpu.forward(ids))
    right = np.asarray(cpu.forward(ids))
    assert left.shape == right.shape and left.tobytes() == right.tobytes()


@pytest.mark.parametrize("family", FAMILIES)
def test_vendor_used_names_the_device_that_answers(tmp_path, family):
    cpu = CausalLM.load(checkpoint(tmp_path, family), device="cpu")
    assert cpu.vendor_used() == "cpu"
    assert CausalLM.load(tmp_path).vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("weight_format", ["float32", "bfloat16", "int8"])
def test_the_device_twin_is_the_same_model_from_the_same_tensors(tmp_path, family, weight_format):
    # The hook `model.check` uses (`_check_both.on_device`): on the model's own device the answer is the model
    # itself. The compiled twin door is called directly here, so it is exercised on a CPU-only installation too:
    # a second model on "cpu" built from the first one's tensors, in every weight format.
    from mojolearn import _check_both
    model = CausalLM.load(checkpoint(tmp_path, family), device="cpu", weight_format=weight_format,
                          numeric_profile="fp32_v1")
    assert _check_both.takes_device(model)
    assert _check_both.on_device(model, "cpu") is model
    twin = model._mojolearn_device_twin("cpu")
    assert type(twin) is CausalLM and twin is not model
    assert (twin.device, twin.weight_format, twin.numeric_profile) == ("cpu", weight_format, "fp32_v1")
    assert twin.max_positions == model.max_positions and len(twin.blocks) == len(model.blocks)
    ids = np.array([[1, 7, 3, 11, 5], [2, 9, 2, 4, 8]], dtype="<i4")
    assert np.asarray(twin.forward(ids)).tobytes() == np.asarray(model.forward(ids)).tobytes()


@two_devices
@pytest.mark.parametrize("family", FAMILIES)
def test_check_runs_the_same_model_on_both_devices_in_this_process(tmp_path, family, capsys):
    model = CausalLM.load(checkpoint(tmp_path, family), numeric_profile="fp32_v1")
    ids = np.array([[1, 7, 3, 11, 5], [2, 9, 2, 4, 8]], dtype="<i4")
    result = model.check(ids)
    assert result["ran"], result.get("reason")
    assert result["transfer"] == "device" and result["method"] == "forward"
    assert result["sides"]["gpu"]["route"]["vendor_used"] == mojolearn.vendor()
    assert result["sides"]["cpu"]["route"]["vendor_used"] == "cpu"
    assert result["verdict"]["verdict"] == "MATCH"
    assert "MATCH" in capsys.readouterr().out


@one_device
def test_check_on_a_cpu_only_installation_says_why_nothing_was_compared(tmp_path, capsys):
    model = CausalLM.load(checkpoint(tmp_path, "llama"))
    result = model.check(np.array([[1, 7, 3]], dtype="<i4"))
    assert result["ran"] is False and result["reason"]
