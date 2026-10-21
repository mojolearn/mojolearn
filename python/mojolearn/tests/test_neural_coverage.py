# SPDX-License-Identifier: Apache-2.0
"""Release-gate coverage of neural and sequence endpoints changed since
0.8.55 (docs/plans/TEST_COVERAGE_20261021.md): SambaStack gradients and
checkpoints, SambaInference, TSNE through the one-call PCA, Sequential
layers, and the integer counts of CausalLM and its blocks. Real calls first,
then poisoned inputs that the native boundary refuses by name. Tiny models,
no measurements."""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import SambaConfig, SambaStack
from mojolearn.neural_inference import SambaInference


def _samba():
    config = SambaConfig(19, 32, ["attention", "mamba3"], n_heads=2, n_kv_heads=1, intermediate=48)
    rng = np.random.default_rng(1)
    weights = {name: rng.uniform(-0.04, 0.04, shape).astype("f4") for name, shape in config.registry()}
    return config, weights


IDS = np.array([[1, 2, 3, 4]], np.int32)
TARGETS = np.array([[2, 3, 4, 5]], np.int32)


def test_samba_loss_and_grads_divisor_and_refusals():
    config, weights = _samba()
    model = SambaStack(config, weights=weights)
    loss, grads = model.loss_and_grads(IDS, TARGETS)
    assert len(grads) == len(config.registry())
    assert loss == pytest.approx(model.loss(IDS, TARGETS), rel=1e-6)
    # num_items is the divisor of the summed loss: twice the items, half the loss
    half, _ = model.loss_and_grads(IDS, TARGETS, num_items=8)
    assert half == pytest.approx(loss / 2, rel=1e-6)
    with pytest.raises(ValueError, match="targets must match inputs' shape"):
        model.loss_and_grads(IDS, TARGETS[:, :3])
    with pytest.raises(ValueError, match="token id outside vocabulary"):
        model.loss_and_grads(IDS, np.array([[2, 3, 4, 99]], np.int32))
    with pytest.raises(ValueError, match="token ids must be native-endian integer buffers"):
        model.loss_and_grads(IDS.astype(np.float32), TARGETS)
    with pytest.raises(ValueError, match="num_items must be a positive integer"):
        model.loss_and_grads(IDS, TARGETS, num_items=0)
    for bad in ("4", True, 2.5):
        with pytest.raises(TypeError, match="num_items must be a positive integer"):
            model.loss_and_grads(IDS, TARGETS, num_items=bad)


def test_samba_save_checkpoint_round_trip_and_missing_directory(tmp_path):
    config, weights = _samba()
    model = SambaStack(config, weights=weights)
    path = tmp_path / "samba.ckpt"
    model.save_checkpoint(path)
    assert path.exists()
    with pytest.raises(ValueError, match="No such file or directory"):
        model.save_checkpoint(tmp_path / "absent" / "samba.ckpt")


def test_samba_inference_parameters_and_allocate_state():
    config, weights = _samba()
    inference = SambaInference(config, weights)
    params = inference.parameters()
    assert sorted(params) == sorted(weights)
    for name, value in weights.items():
        np.testing.assert_array_equal(np.asarray(params[name]), value)
    state = inference.allocate_state(1, 8)
    assert type(state).__name__ == "SambaState"
    for bad in ((0, 8), (1, 0)):
        with pytest.raises(ValueError, match="batch_size and max_tokens must be positive"):
            inference.allocate_state(*bad)
    for bad in (("1", 8), (1, True)):
        with pytest.raises(TypeError, match="batch_size and max_tokens must be integers"):
            inference.allocate_state(*bad)


def test_tsne_fit_transform_through_pca_and_refusals():
    X = np.random.default_rng(0).standard_normal((30, 4)).astype(np.float32)
    embedding = ml.TSNE(n_components=2, perplexity=5, init="pca", max_iter=250,
                        random_state=0).fit_transform(X)
    assert embedding.shape == (30, 2) and np.isfinite(embedding).all()
    fitted = ml.TSNE(n_components=2, perplexity=5, init="pca", max_iter=250, random_state=0).fit(X)
    np.testing.assert_array_equal(fitted.embedding_, embedding)


def test_tsne_refuses_poisoned_input_by_name():
    X = np.random.default_rng(0).standard_normal((30, 4)).astype(np.float32)
    X[0, 0] = np.nan
    for method in ("fit", "fit_transform"):
        with pytest.raises(ValueError, match="Input X contains NaN or infinity"):
            getattr(ml.TSNE(n_components=2, perplexity=5, max_iter=250), method)(X)
    with pytest.raises(ValueError, match=r"perplexity \(50\) must be less than n_samples \(30\)"):
        ml.TSNE(n_components=2, perplexity=50, max_iter=250).fit(np.zeros((30, 4), np.float32))


def test_sequential_layers_sigmoid_and_avg_pool2d():
    model = ml.Sequential().linear(2).sigmoid().avg_pool2d(2)
    assert model.layers == [("linear", 2, 1, 0.0), ("sigmoid", 0, 0, 0.0),
                            ("avg_pool2d", (2, 2, 0, 0, 0, 0), 0.0)]
    X = np.random.default_rng(0).standard_normal((8, 3)).astype(np.float32)
    proba = ml.Sequential().linear(1).sigmoid().compile(loss="mse").fit(X, np.zeros((8, 1), np.float32),
                                                                         epochs=1).predict(X)
    assert proba.shape == (8, 1) and ((proba > 0) & (proba < 1)).all()
    images = np.arange(64, dtype=np.float32).reshape(4, 1, 4, 4)
    net = ml.Sequential(input_shape=(1, 4, 4)).avg_pool2d(2).flatten().linear(1)
    assert net.compile().fit(images, np.zeros((4, 1), np.float32), epochs=1).predict(images).shape == (4, 1)


@pytest.mark.parametrize("build,error,match", [
    (lambda: ml.Sequential().avg_pool2d(0), ValueError, "kernel_size must be an integer >= 1"),
    (lambda: ml.Sequential().avg_pool2d("2"), ValueError, "kernel_size must be an integer >= 1"),
    (lambda: ml.Sequential().avg_pool2d(2, stride=0), ValueError, "stride must be an integer >= 1"),
    (lambda: ml.Sequential().add("nope"), ValueError, "kind must name a supported Sequential layer"),
    (lambda: ml.Sequential().add("linear", 3, bogus=1), TypeError, "linear received an unknown keyword"),
    (lambda: ml.Sequential().dropout(1.5), ValueError, r"p must be in \[0, 1\)"),
])
def test_sequential_layer_arguments_refused_by_name(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_sequential_refuses_bad_layer_stacks_at_fit():
    X = np.zeros((8, 3), np.float32)
    y = np.zeros((8, 1), np.float32)
    with pytest.raises(ValueError, match="add at least one Linear layer"):
        ml.Sequential().sigmoid().compile(loss="mse").fit(X, y, epochs=1)
    images = np.zeros((4, 1, 4, 4), np.float32)
    with pytest.raises(ValueError, match="kernel is larger than the"):
        ml.Sequential(input_shape=(1, 4, 4)).avg_pool2d(8).flatten().linear(1).compile().fit(
            images, y[:4], epochs=1)
    with pytest.raises(ValueError, match="must list each of the 3 sample axes once"):
        ml.Sequential(input_shape=(2, 2, 3)).permute((0, 0, 1)).flatten().linear(1).compile().fit(
            np.zeros((8, 2, 2, 3), np.float32), y, epochs=1)


def test_causal_lm_counts_refuse_str_and_bool(tmp_path):
    from mojolearn.tests.test_generation_native import load_fixture
    model = load_fixture(tmp_path, uniform=True)
    prompt = np.array([[33]], dtype="i4")
    assert np.asarray(model.generate(prompt, 2)).shape == (1, 3)
    assert np.asarray(model.generate(prompt, np.int64(2))).shape == (1, 3)
    with pytest.raises(ValueError, match="max_new_tokens must be nonnegative"):
        model.generate(prompt, -1)
    for bad in ("2", True):
        with pytest.raises(TypeError, match="max_new_tokens must be an integer"):
            model.generate(prompt, bad)
        with pytest.raises(TypeError, match="max_new_tokens must be an integer"):
            model.stream(prompt, bad)
    assert model.allocate_state(1, 8).max_tokens == 8
    with pytest.raises(ValueError, match="batch_size and max_tokens must be positive"):
        model.allocate_state(0, 8)
    with pytest.raises(TypeError, match="batch_size and max_tokens must be integers"):
        model.allocate_state("1", 8)
    block = model.blocks[0]
    assert block.allocate_state(1, 8) is not None
    with pytest.raises(TypeError, match="batch_size and max_tokens must be integers"):
        block.allocate_state("1", 8)
