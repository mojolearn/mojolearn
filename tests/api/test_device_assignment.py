# SPDX-License-Identifier: Apache-2.0
"""Python boundary tests of a direct `model.device = ...` on a live estimator.

`device` is read once into the estimator's route (providers/python/device_route_boundary_door.mojo). A direct
assignment after the first operation must reach the next operation: it runs on the new device, or it refuses a bad
value in words. One case per route owner kind: the linear entry route, LogisticRegression's own hit test, the
decomposition route, the scaler route, the tree binding route and the shared estimator binding route. The fits are
small and independent; pytest-xdist splits the tests across cores when it is installed.
"""
import numpy as np
import pytest

mojolearn = pytest.importorskip("mojolearn")

RNG = np.random.default_rng(0)
X = RNG.random((300, 5), dtype=np.float32)
Y = X @ np.arange(1, 6, dtype=np.float32)
C = (Y > np.median(Y)).astype(np.int32)

#: (class name, parameters, target, the operation that follows the assignment)
CASES = [
    ("Ridge", dict(alpha=1.0), "r", "predict"),
    ("LinearRegression", {}, "r", "predict"),
    ("LogisticRegression", {}, "c", "predict"),
    ("QNRegressor", {}, "r", "predict"),
    ("KMeans", dict(n_clusters=3, random_state=0), None, "predict"),
    ("PCA", dict(n_components=2), None, "transform"),
    ("TruncatedSVD", dict(n_components=2), None, "transform"),
    ("StandardScaler", {}, None, "transform"),
    ("MinMaxScaler", {}, None, "transform"),
    ("RandomForestRegressor", dict(n_estimators=5, random_state=0), "r", "predict"),
    ("GradientBoosting", dict(n_estimators=5), "r", "predict"),
    ("GaussianNB", {}, "c", "predict"),
]

pytestmark = pytest.mark.skipif(mojolearn.numeric_mode() != "identical", reason="device='cpu' runs in identical mode")


def _build(name, params, **more):
    cls = getattr(mojolearn, name, None)
    if cls is None:
        pytest.skip(name + " is not in this installation")
    return cls(**params, **more)


def _fit(est, target):
    return est.fit(X) if target is None else est.fit(X, Y if target == "r" else C)


def _cpu_twin(name, params, target):
    """The same estimator constructed with device='cpu'; skipped where the class does not run there."""
    try:
        return _fit(_build(name, params, device="cpu"), target)
    except (ValueError, ImportError) as error:
        if "device='cpu'" in str(error):
            pytest.skip(str(error))
        raise


@pytest.mark.parametrize("name,params,target,operation", CASES)
def test_assigned_cpu_runs_the_next_operation_on_the_cpu(name, params, target, operation):
    twin = _cpu_twin(name, params, target)
    expected = np.asarray(getattr(twin, operation)(X))
    model = _fit(_build(name, params), target)
    before = np.asarray(getattr(model, operation)(X))
    model.device = "cpu"
    after = np.asarray(getattr(model, operation)(X))
    assert model.vendor_used() == "cpu"
    assert model.get_params()["device"] == "cpu"
    # identical mode: the same bits on every device, so the fitted state carries over
    assert np.array_equal(before, after) and np.array_equal(after, expected)


@pytest.mark.parametrize("name,params,target,operation", CASES)
def test_assigned_none_returns_to_the_installation_device(name, params, target, operation):
    model = _cpu_twin(name, params, target)
    getattr(model, operation)(X)
    assert model.vendor_used() == "cpu"
    model.device = None
    getattr(model, operation)(X)
    assert model.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("name,params,target,operation", CASES)
def test_a_bad_value_is_refused_in_words_at_the_next_operation(name, params, target, operation):
    model = _fit(_build(name, params), target)
    expected = np.asarray(getattr(model, operation)(X))
    for bad in ("cuda", "CPU", 0, b"cpu"):
        model.device = bad
        with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
            getattr(model, operation)(X)
    # nothing was changed by the refusals: the estimator runs again once the value is one of the three
    model.device = None
    assert np.array_equal(np.asarray(getattr(model, operation)(X)), expected)


def test_explicit_gpu_is_refused_in_words_on_a_cpu_only_installation():
    model = _fit(_build("Ridge", dict(alpha=1.0)), "r")
    model.predict(X)
    model.device = "gpu"
    if mojolearn.vendor() == "cpu":
        with pytest.raises(ValueError, match="needs a GPU build"):
            model.predict(X)
    else:
        model.predict(X)
        assert model.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("write", ["setattr", "vars", "dict_update"])
def test_every_form_of_the_write_reaches_the_route(write):
    _cpu_twin("Ridge", dict(alpha=1.0), "r")
    model = _fit(_build("Ridge", dict(alpha=1.0)), "r")
    model.predict(X)
    if write == "setattr":
        setattr(model, "device", "cpu")
    elif write == "vars":
        vars(model)["device"] = "cpu"
    else:
        vars(model).update(device="cpu")
    model.predict(X)
    assert model.vendor_used() == "cpu"


def test_deleting_the_field_runs_the_installation_device():
    model = _cpu_twin("Ridge", dict(alpha=1.0), "r")
    model.predict(X)
    del model.device
    model.predict(X)
    assert model.vendor_used() == mojolearn.vendor()


def test_an_assignment_on_one_estimator_leaves_the_others_on_their_device():
    _cpu_twin("Ridge", dict(alpha=1.0), "r")
    first = _fit(_build("Ridge", dict(alpha=1.0)), "r")
    second = _fit(_build("Ridge", dict(alpha=2.0)), "r")
    first.predict(X)
    second.predict(X)
    first.device = "cpu"
    second.predict(X)
    first.predict(X)
    assert second.vendor_used() == mojolearn.vendor() and first.vendor_used() == "cpu"
    # the same object assigned again keeps the route
    first.device = first.device
    first.predict(X)
    assert first.vendor_used() == "cpu"


def test_assignment_before_the_first_operation_is_the_constructor_value():
    _cpu_twin("Ridge", dict(alpha=1.0), "r")
    model = _build("Ridge", dict(alpha=1.0))
    model.device = "cpu"
    _fit(model, "r").predict(X)
    assert model.vendor_used() == "cpu"


# ---- neural blocks (lane fx-device-attr round 2) ------------------------------------------------------------------
# A standalone block routes by its `_native_route` word. A direct `block.device = ...` reaches the next forward,
# backward, decode session and pickle through the block's route stamp.

BLOCK_FAMILIES = ["llama", "mamba", "mamba2"]
TWO_DEVICES = mojolearn.vendor() != "cpu"


def _block_inputs(tmp_path, family):
    """The GPU block class of `family`, its weights, its keyword arguments and an input (as test_torch_conventions)."""
    from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
    from mojolearn.models import CausalLM, causal_lm
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    old = model.blocks[0]
    weights = dict(zip(old._W_NAMES, old._w))
    kwargs = {}
    if model.kind == "transformer":
        for name, value in zip(causal_lm._NATIVE_CONTEXT["optional_names"], old._wopt):
            if value is not None:
                weights[name] = value
        kwargs = dict(model.plan.block_options, n_heads=model.plan.n_heads, numeric_profile="fp32_v1")
    elif model.kind == "mamba2":
        kwargs = dict(model.plan.block_kwargs)
    cls = causal_lm._NATIVE_CONTEXT["block_classes"][model.kind]
    x = np.random.default_rng(7).normal(size=(2, 4, model.d_model)).astype("f4") * np.float32(0.05)
    return cls, weights, kwargs, x


@pytest.mark.parametrize("family", BLOCK_FAMILIES)
def test_a_block_assigned_cpu_runs_its_next_forward_on_the_cpu(tmp_path, family):
    cls, weights, kwargs, x = _block_inputs(tmp_path, family)
    expected = np.asarray(cls(weights, device="cpu", **kwargs).forward(x))
    block = cls(weights, **kwargs)
    block.forward(x)
    block.device = "cpu"
    after = np.asarray(block.forward(x))
    assert block._native_route == "cpu" and block.vendor_used() == "cpu"
    assert block.get_params()["device"] == "cpu"
    assert after.tobytes() == expected.tobytes()


@pytest.mark.parametrize("family", BLOCK_FAMILIES)
def test_a_block_assigned_back_to_none_returns_to_the_installation_device(tmp_path, family):
    cls, weights, kwargs, x = _block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    block.forward(x)
    block.device = None
    block.forward(x)
    assert block._native_route == ("gpu" if TWO_DEVICES else "cpu")


@pytest.mark.parametrize("family", BLOCK_FAMILIES)
def test_a_bad_block_device_is_refused_in_words_and_the_block_runs_again(tmp_path, family):
    cls, weights, kwargs, x = _block_inputs(tmp_path, family)
    block = cls(weights, device="cpu", **kwargs)
    expected = np.asarray(block.forward(x))
    for bad in ("cuda", "CPU", 0):
        block.device = bad
        with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
            block.forward(x)
    block.device = "cpu"
    assert np.asarray(block.forward(x)).tobytes() == expected.tobytes()


@pytest.mark.parametrize("family", BLOCK_FAMILIES)
def test_a_block_assignment_reaches_backward(tmp_path, family):
    cls, weights, kwargs, x = _block_inputs(tmp_path, family)
    twin = cls(weights, device="cpu", **kwargs)
    grad = np.ones_like(np.asarray(twin.forward(x)))
    expected = twin.backward(x, grad)
    block = cls(weights, **kwargs)
    block.forward(x)
    block.backward(x, grad)
    block.device = "cpu"
    got = block.backward(x, grad)
    assert block._native_route == "cpu"
    # backward returns the parameter gradients by name (a dict of arrays); compare each array's bytes. The former
    # `np.asarray(dict).tobytes()` compared the bytes of a 0-d object array (an object pointer), never equal.
    first = lambda value: value[0] if isinstance(value, tuple) else value
    got, expected = first(got), first(expected)
    assert sorted(got) == sorted(expected)
    for name in expected:
        assert np.asarray(got[name]).tobytes() == np.asarray(expected[name]).tobytes(), name


def test_a_pickled_block_carries_the_assigned_route(tmp_path):
    import pickle
    cls, weights, kwargs, x = _block_inputs(tmp_path, "llama")
    block = cls(weights, **kwargs)
    block.forward(x)
    block.device = "cpu"
    restored = pickle.loads(pickle.dumps(block))
    assert restored.device == "cpu" and restored._native_route == "cpu"


def test_an_unchanged_same_word_keeps_the_block(tmp_path):
    cls, weights, kwargs, x = _block_inputs(tmp_path, "llama")
    block = cls(weights, device="cpu", **kwargs)
    block.forward(x)
    held = block._w
    block.device = "c" + "pu"  # an equal word, another object: the same route, no rebuild
    block.forward(x)
    assert block._native_route == "cpu" and block._w is held


# ---- CNN containers (lane fx-device-attr round 2) -----------------------------------------------------------------
# A container's children copy its `device` at construction; an assignment on the container reaches them at its next
# operation.

def _images(rows=4, channels=3, side=8, seed=3):
    return np.random.default_rng(seed).random((rows, channels, side, side), dtype=np.float32)


def _labeled(rows=48, side=8, seed=7):
    x = np.random.default_rng(seed).random((rows, 1, side, side), dtype=np.float32)
    return x, (x.mean(axis=(1, 2, 3)) > np.float32(0.5)).astype(np.int64)


CNN_CLASSIFIER = dict(conv_channels=(4,), kernel_size=3, pool_size=2, batch_size=16, max_iter=2, random_state=3)


def test_a_basic_block_assignment_reaches_its_children():
    expected = np.asarray(mojolearn.BasicBlock(3, 4, stride=2, random_state=1, device="cpu").forward(_images()))
    block = mojolearn.BasicBlock(3, 4, stride=2, random_state=1)
    block.forward(_images())
    block.device = "cpu"
    after = np.asarray(block.forward(_images()))
    children = [block.conv1, block.bn1, block.relu1, block.conv2, block.bn2, block.relu2, *block.downsample]
    assert all(child.device == "cpu" for child in children)
    assert after.tobytes() == expected.tobytes()


def test_a_basic_block_assigned_before_its_first_forward_hands_the_word_down():
    block = mojolearn.BasicBlock(3, 4, random_state=1)
    block.device = "cpu"
    block.forward(_images())
    assert all(child.device == "cpu" for child in (block.conv1, block.bn1, block.relu1, block.conv2, block.bn2, block.relu2))


def test_a_classifier_assignment_reaches_its_layers():
    x, y = _labeled()
    expected = np.asarray(mojolearn.CNNClassifier((1, 8, 8), device="cpu", **CNN_CLASSIFIER).fit(x, y).predict(x))
    model = mojolearn.CNNClassifier((1, 8, 8), **CNN_CLASSIFIER).fit(x, y)
    model.predict(x)
    model.device = "cpu"
    after = np.asarray(model.predict(x))
    assert model.head_.device == "cpu" and all(member.device == "cpu" for member in model.layers_)
    assert model.vendor_used() == "cpu"
    assert after.tobytes() == expected.tobytes()


def test_a_bad_classifier_device_is_refused_and_leaves_the_layers():
    x, y = _labeled()
    model = mojolearn.CNNClassifier((1, 8, 8), device="cpu", **CNN_CLASSIFIER).fit(x, y)
    model.device = "tpu"
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        model.predict(x)
    assert model.head_.device == "cpu" and all(member.device == "cpu" for member in model.layers_)


def test_an_adaptive_pool_assignment_reaches_its_cached_inner_pool():
    pool = mojolearn.AdaptiveAvgPool2d(2)
    expected = np.asarray(mojolearn.AdaptiveAvgPool2d(2, device="cpu").forward(_images()))
    pool.forward(_images())
    pool.device = "cpu"
    assert np.asarray(pool.forward(_images())).tobytes() == expected.tobytes()
    assert vars(pool)["_pool_cache"][1].device == "cpu"
