# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the language-model trainer, at the Python
boundary. Same rule and same words as Ridge and LinearRegression (test_linear_device_boundary.py).

The device is a constructor parameter (and a keyword of the three checkpoint restores), checked where it is
given and read once into the trainer's one native route owner; no operation reads it again. On a GPU
installation both sets of compiled entries exist in one process and `device="cpu"` runs the CPU binary. What
needs two devices is skipped by name on a CPU-only installation.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn._byte_lm_impl import SmallByteLanguageModelTrainer

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")

SCHEDULE = {'corpus_sha256': '1' * 64, 'token_schedule_sha256': '2' * 64,
            'batch_offsets': [0, 32], 'planned_steps': 4}


def initial():
    return (np.random.default_rng(23).standard_normal(34944) * .02).astype(np.float32)


def ids():
    return (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)


def trainer(**options):
    return SmallByteLanguageModelTrainer(initial(), data_schedule=dict(SCHEDULE), **options)


def test_default_device_is_none_and_trains():
    model = trainer()
    assert model.device is None
    assert np.isfinite(model.train_step(ids())["loss"])


@pytest.mark.parametrize("value", ["CPU", "cuda", 0, True, b"cpu"])
def test_unknown_device_is_refused_at_construction(value):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        trainer(device=value)


def test_unknown_device_is_refused_at_restore(tmp_path):
    model = trainer()
    path = tmp_path / "state.json"
    model.export_checkpoint(path)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        SmallByteLanguageModelTrainer.from_checkpoint(path, device="metal")
    assert SmallByteLanguageModelTrainer.from_checkpoint(path).device is None


def test_explicit_gpu_follows_the_installation():
    if TWO_DEVICES:
        assert trainer(device="gpu").device == "gpu"
    else:
        with pytest.raises(ValueError, match="device='gpu' needs a GPU build"):
            trainer(device="gpu")


def test_cpu_device_trains_and_reports_the_cpu_image():
    model = trainer(device="cpu")
    assert model.device == "cpu"
    assert np.isfinite(model.train_step(ids())["loss"])
    assert model.run_metadata()["native_vendor"] == "cpu"


@two_devices
def test_default_device_reports_the_gpu_image():
    model = trainer()
    model.train_step(ids())
    assert model.run_metadata()["native_vendor"] == mojolearn.vendor()


@two_devices
def test_cpu_state_restores_on_the_cpu_device(tmp_path):
    model = trainer(device="cpu")
    model.train_step(ids())
    path = tmp_path / "state.bin"
    model.export_checkpoint_binary(path)
    restored = SmallByteLanguageModelTrainer.from_checkpoint_binary(path, device="cpu")
    assert restored.device == "cpu" and restored.step_ == 1
    assert restored.train_step(ids())["step"] == 2
