# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the CNN layers and CNNClassifier, at the Python
boundary. Same rule and same words as Ridge and LinearRegression (test_linear_device_boundary.py).

The device is a constructor parameter, checked where it is given and read once, by the layer's first operation,
into its one native route owner; no operation reads it again. A layer built by another layer (BasicBlock's
convolutions and norms, SAGEConv's projections, the classifier's stack, an adaptive pool's inner pool) runs on that
layer's device. On a GPU installation both sets of compiled entries exist in one process and `device="cpu"` runs
the CPU binaries; in `identical` mode it returns the bits the GPU returns. What needs two devices is skipped by
name on a CPU-only installation.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import (
    AdaptiveAvgPool2d, AdaptiveMaxPool2d, AvgPool1d, AvgPool2d, BasicBlock, BatchNorm1d, BatchNorm2d, CNNClassifier,
    Conv1d, Conv2d, Dropout2d, GCNConv, MaxPool1d, MaxPool2d, SAGEConv,
)

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def images(rows=4, channels=3, side=8, seed=3):
    rng = np.random.default_rng(seed)
    return rng.random((rows, channels, side, side), dtype=np.float32)


def signals(rows=4, channels=3, length=12, seed=3):
    rng = np.random.default_rng(seed)
    return rng.random((rows, channels, length), dtype=np.float32)


def graph(nodes=10, features=3, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.random((nodes, features), dtype=np.float32)
    source = np.arange(nodes, dtype=np.int64)
    edges = np.stack([np.concatenate([source, source]), np.concatenate([(source + 1) % nodes, (source + 3) % nodes])])
    return x, edges


def labeled(rows=48, channels=1, side=8, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, channels, side, side), dtype=np.float32)
    return x, (x.mean(axis=(1, 2, 3)) > np.float32(0.5)).astype(np.int64)


# (id, class, positional arguments, keywords, the forward's arguments)
LAYERS = [
    ("conv2d", Conv2d, (3, 4, 3), dict(padding=1, random_state=1), lambda: (images(),)),
    ("conv1d", Conv1d, (3, 4, 3), dict(padding=1, random_state=1), lambda: (signals(),)),
    ("maxpool2d", MaxPool2d, (2,), {}, lambda: (images(),)),
    ("avgpool2d", AvgPool2d, (2,), {}, lambda: (images(),)),
    ("maxpool1d", MaxPool1d, (2,), {}, lambda: (signals(),)),
    ("avgpool1d", AvgPool1d, (2,), {}, lambda: (signals(),)),
    ("batchnorm2d", BatchNorm2d, (3,), {}, lambda: (images(),)),
    ("batchnorm1d", BatchNorm1d, (3,), {}, lambda: (signals(),)),
    ("dropout2d", Dropout2d, (), dict(p=0.25, random_state=2), lambda: (images(),)),
    ("adaptive-avg", AdaptiveAvgPool2d, (2,), {}, lambda: (images(),)),
    ("adaptive-max", AdaptiveMaxPool2d, (2,), {}, lambda: (images(),)),
    ("basic-block", BasicBlock, (3, 4), dict(stride=2, random_state=1), lambda: (images(),)),
    ("gcn", GCNConv, (3, 4), dict(random_state=1), graph),
    ("sage", SAGEConv, (3, 4), dict(random_state=1), graph),
    ("sage-project", SAGEConv, (3, 4), dict(project=True, random_state=1), graph),
]
layer = pytest.mark.parametrize("spec", LAYERS, ids=[spec[0] for spec in LAYERS])
CLASSIFIER = dict(conv_channels=(4,), kernel_size=3, pool_size=2, batch_size=16, max_iter=2, random_state=3)


def build(spec, **extra):
    return spec[1](*spec[2], **dict(spec[3], **extra))


def children(model):
    # glue: the layers a layer built (one level of its instance dictionary: fields, lists and pairs)
    found = []
    for value in vars(model).values():
        for item in (value if isinstance(value, (list, tuple)) else (value,)):
            for member in (item if isinstance(item, (list, tuple)) else (item,)):
                if hasattr(type(member), "_mojolearn_device") and type(member).__module__ == type(model).__module__:
                    found.append(member)
    return found


@layer
def test_device_is_a_constructor_parameter_with_default_none(spec):
    assert build(spec).device is None
    named = "cpu" if not TWO_DEVICES else "gpu"
    assert build(spec, device=named).device == named


@layer
@pytest.mark.parametrize("value", ["GPU", "cuda", "", 0, True, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(spec, value):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        build(spec, device=value)


@layer
def test_explicit_gpu_is_refused_in_words_on_a_cpu_only_installation(spec):
    if TWO_DEVICES:
        assert build(spec, device="gpu").device == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        build(spec, device="gpu")


@layer
def test_a_layer_built_by_a_layer_takes_its_device(spec):
    model = build(spec, device="cpu")
    np.asarray(model.forward(*spec[4]()))                      # an adaptive pool builds its inner pool here
    for member in children(model):
        assert member.device == "cpu", type(member).__name__


@layer
def test_cpu_device_runs_and_answers(spec):
    model = build(spec, device="cpu")
    out = np.asarray(model.forward(*spec[4]()))
    assert out.dtype == np.float32 and out.size > 0


@two_devices
def test_device_cpu_in_fast_mode_is_refused_in_words():
    model = Conv2d(3, 4, 3, numeric_mode="fast", device="cpu", random_state=1)
    # Off Apple with no fast set the identical set answers `fast` for this family (core/mode_answer.mojo,
    # MODE_FAMILIES_ANSWERED; lane gs-fix: the test predated it and failed on the L40S GPU install).
    try:
        answered = np.asarray(model.forward(images()))
    except (ValueError, ImportError, RuntimeError) as refused:
        # a build without the fast image says that first; with it, the shared words
        assert "identical mode only" in str(refused) or "fast" in str(refused)
    else:
        plain = np.asarray(Conv2d(3, 4, 3, device="cpu", random_state=1).forward(images()))
        assert answered.tobytes() == plain.tobytes()


def test_a_constructor_run_again_reads_the_device_again():
    model = Conv2d(3, 4, 3, padding=1, random_state=1)
    first = np.asarray(model.forward(images()))
    model.__init__(3, 4, 3, padding=1, random_state=1, device="cpu")
    assert model.device == "cpu"
    assert np.asarray(model.forward(images())).shape == first.shape
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        model.__init__(3, 4, 3, device="tpu")


def test_classifier_takes_device_and_its_layers_follow_it():
    x, y = labeled()
    model = CNNClassifier((1, 8, 8), device="cpu", **CLASSIFIER).fit(x, y)
    assert model.device == "cpu" and model.head_.device == "cpu"
    assert all(member.device == "cpu" for member in model.layers_)
    assert np.asarray(model.predict(x)).shape == (len(x),)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        CNNClassifier((1, 8, 8), device="cuda", **CLASSIFIER)


@two_devices
def test_classifier_check_compares_the_same_model_on_both_devices_in_this_process(capsys):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = labeled()
    model = CNNClassifier((1, 8, 8), numeric_mode="identical", **CLASSIFIER).fit(x, y)
    result = model.check(x)
    capsys.readouterr()
    assert result["ran"] and result["transfer"] == "device"   # no save, no host_model, no child process
    assert result["verdict"]["match"]
    assert model.device is None and model.head_.device is None  # the checked model is not moved
