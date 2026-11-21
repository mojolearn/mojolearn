# SPDX-License-Identifier: Apache-2.0
"""`device=` on `SambaStack`, at the Python boundary. Same rule and same words as every other class.

The device is a constructor parameter and a keyword of `from_checkpoint`, checked where it is given, before
any field is written or any file is read, and resolved once into the stack's one native route (training
entries, primitives, blocks, optimizer and decode program together); no operation reads it again. A stack
runs on the device of its installation: on a GPU installation `device="cpu"` is refused in words, because
the blocks' backward has no CPU images loaded beside the GPU set.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import SambaConfig, SambaStack, training

TWO_DEVICES = mojolearn.vendor() != "cpu"


def config():
    return SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11)


def stack(**options):
    return SambaStack(config(), generator=training.Generator(7), **options)


def ids():
    return (np.arange(12, dtype=np.int32) % 13).reshape(2, 6)


def test_default_device_is_none_and_runs():
    model = stack()
    assert model.device is None
    assert np.asarray(model.forward(ids())).shape == (2, 6, 13)
    assert np.isfinite(model.train_step(ids(), ids())["loss"])


@pytest.mark.parametrize("value", ["CPU", "cuda", 0, True, b"cpu"])
def test_unknown_device_is_refused_at_construction(value):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        stack(device=value)


def test_installation_device_by_name():
    name = "gpu" if TWO_DEVICES else "cpu"
    model = stack(device=name)
    assert model.device == name
    reference = stack()
    assert np.array_equal(np.asarray(model.forward(ids())), np.asarray(reference.forward(ids())))
    assert model.train_step(ids(), ids())["loss"] == reference.train_step(ids(), ids())["loss"]


def test_other_device_is_refused_at_construction():
    if TWO_DEVICES:
        with pytest.raises(ValueError, match="device='cpu' is not available for SambaStack"):
            stack(device="cpu")
    else:
        with pytest.raises(ValueError, match="device='gpu' needs a GPU build"):
            stack(device="gpu")


def test_refused_constructor_leaves_a_live_stack_untouched():
    model = stack()
    before = np.asarray(model.forward(ids())).copy()
    with pytest.raises(ValueError):
        model.__init__(config(), generator=training.Generator(9), device="metal")
    assert model.device is None
    assert np.array_equal(np.asarray(model.forward(ids())), before)


def test_checkpoint_restore_takes_and_refuses_device(tmp_path):
    model = stack()
    model.train_step(ids(), ids())
    path = tmp_path / "samba.ckpt"
    model.save_checkpoint(path)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        SambaStack.from_checkpoint(path, device="metal")
    other = "cpu" if TWO_DEVICES else "gpu"
    with pytest.raises(ValueError):
        SambaStack.from_checkpoint(path, device=other)
    name = "gpu" if TWO_DEVICES else "cpu"
    restored = SambaStack.from_checkpoint(path, device=name)
    assert restored.device == name
    assert SambaStack.from_checkpoint(path).device is None
    assert np.array_equal(np.asarray(restored.forward(ids())), np.asarray(model.forward(ids())))


def test_changed_numeric_mode_is_admitted_again():
    model = stack()
    model.forward(ids())
    model.numeric_mode = "no-such-mode"
    with pytest.raises(ValueError):
        model.forward(ids())


def test_decode_session_and_state_dict_use_the_stack_route():
    model = stack()
    state = model.allocate_state(2, 8)
    session = model.decode_session(state)
    assert session.is_open
    session.close()
    saved = model.state_dict()
    model.load_state_dict(saved)
    assert np.array_equal(np.asarray(model.state_dict()["parameters"]), np.asarray(saved["parameters"]))
