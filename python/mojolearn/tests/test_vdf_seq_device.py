# SPDX-License-Identifier: Apache-2.0
"""`device=` on layer_norm_forward / layer_norm_backward, LayerNorm, MoEBlock,
the sequence optimizers and SGD / Adam / AdamW: the boundary words only (lane
vdf-seq). The digests of both sides are verify's; nothing numerical is
asserted here."""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn import _x_sequence_moe, _x_sequence_norm, _x_sequence_optim, training

BAD_WORDS = "device must be 'gpu', 'cpu' or None"


def _x():
    return np.ones((2, 4), dtype=np.float32)


def _w():
    return np.zeros(4, dtype=np.float32)


# (public name, a call with tiny inputs that forwards `device`)
CALLS = [
    ("layer_norm_forward", lambda d: _x_sequence_norm.layer_norm_forward(_x(), device=d)),
    ("layer_norm_backward", lambda d: _x_sequence_norm.layer_norm_backward(_x(), _x(), device=d)),
    ("LayerNorm", lambda d: _x_sequence_norm.LayerNorm(4, device=d)),
    ("MoEBlock", lambda d: _x_sequence_moe.MoEBlock(hidden_size=4, intermediate_size=4, num_experts=2, top_k=1,
                                                    device=d)),
    ("RMSprop", lambda d: _x_sequence_optim.RMSprop([_w()], device=d)),
    ("Adagrad", lambda d: _x_sequence_optim.Adagrad([_w()], device=d)),
    ("Lion", lambda d: _x_sequence_optim.Lion([_w()], device=d)),
    ("Adamax", lambda d: _x_sequence_optim.Adamax([_w()], device=d)),
    ("NAdam", lambda d: _x_sequence_optim.NAdam([_w()], device=d)),
    ("Adafactor", lambda d: _x_sequence_optim.Adafactor([_w()], device=d)),
    ("LAMB", lambda d: _x_sequence_optim.LAMB([_w()], device=d)),
    ("SGD", lambda d: training.SGD([_w()], device=d)),
    ("Adam", lambda d: training.Adam([_w()], device=d)),
    ("AdamW", lambda d: training.AdamW([_w()], device=d)),
]

SIGNATURES = [
    _x_sequence_norm.layer_norm_forward, _x_sequence_norm.layer_norm_backward,
    _x_sequence_norm.LayerNorm.__init__, _x_sequence_moe.MoEBlock.__init__,
    _x_sequence_optim.RMSprop.__init__, _x_sequence_optim.Adagrad.__init__, _x_sequence_optim.Lion.__init__,
    _x_sequence_optim.Adamax.__init__, _x_sequence_optim.NAdam.__init__, _x_sequence_optim.Adafactor.__init__,
    _x_sequence_optim.LAMB.__init__, training.SGD.__init__, training.Adam.__init__, training.AdamW.__init__,
]


@pytest.mark.parametrize("function", SIGNATURES, ids=lambda f: f.__qualname__)
def test_signature_has_keyword_only_device(function):
    parameter = inspect.signature(function).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("bad", ["tpu", 0, "GPU"])
@pytest.mark.parametrize("call", [c for _, c in CALLS], ids=[n for n, _ in CALLS])
def test_bad_word_is_refused(call, bad):
    with pytest.raises(ValueError, match=BAD_WORDS):
        call(bad)


@pytest.mark.parametrize("call", [c for _, c in CALLS], ids=[n for n, _ in CALLS])
def test_gpu_on_cpu_only_installation_is_refused(call):
    # Holds only on a CPU-only installation (vendor 'cpu'); on a GPU one the
    # same call runs today's route, which this boundary file does not judge.
    if mojolearn.vendor() != "cpu":
        pytest.skip("CPU-only installation check; this installation has a GPU set")
    with pytest.raises(ValueError, match="needs a GPU build of mojolearn"):
        call("gpu")


@pytest.mark.parametrize("call", [c for _, c in CALLS], ids=[n for n, _ in CALLS])
def test_none_and_cpu_construct_or_run(call):
    # device=None is today's route; "cpu" runs the CPU identical image (beside a
    # GPU set) or today's route (CPU-only). Words and types only.
    call(None)
    if mojolearn.vendor() == "cpu" or mojolearn._backend.default_mode() == "identical":
        call("cpu")


def test_device_is_a_get_params_word_of_the_optimizers():
    for cls in (training.SGD, training.Adam, training.AdamW):
        assert cls([_w()]).get_params()["device"] is None


def test_resident_input_with_cpu_is_refused():
    # Holds only on a GPU installation: a SequenceDeviceTensor lives on the GPU binding.
    if mojolearn.vendor() == "cpu" or mojolearn._backend.default_mode() != "identical":
        pytest.skip("GPU identical installation check")
    from mojolearn import _x_sequence_device
    x = _x_sequence_device.to_device(_x())
    with pytest.raises(ValueError, match="was given device='cpu' but x is a SequenceDeviceTensor"):
        _x_sequence_norm.layer_norm_forward(x, device="cpu")
    with pytest.raises(ValueError, match="was given device='cpu' but x is a SequenceDeviceTensor"):
        _x_sequence_norm.LayerNorm(4, device="cpu")(x)
    with pytest.raises(ValueError, match="was given device='cpu' but parameter 0 is a SequenceDeviceTensor"):
        _x_sequence_optim.RMSprop([_x_sequence_device.to_device(_w())], device="cpu")


def test_fast_mode_cpu_is_refused():
    # The CPU images beside a GPU set are identical only (the estimators' words).
    if mojolearn.vendor() == "cpu":
        pytest.skip("GPU installation check")
    with pytest.raises(ValueError):
        _x_sequence_norm.layer_norm_forward(_x(), numeric_mode="fast", device="cpu")
