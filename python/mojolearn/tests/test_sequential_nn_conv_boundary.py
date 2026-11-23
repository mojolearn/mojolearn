# SPDX-License-Identifier: Apache-2.0
"""Public boundary of the convolution and shape layers of Sequential (lane nn-conv, 2026-11-23):
conv_transpose1d / 2d, the adaptive and global pools, the pads, upsample and pixel (un)shuffle. The
description each call writes, the refusals (at add, and at fit for what needs the input's shape), and a
fitted model's save / load and state_dict round trip. The numerics (forward against a float64 oracle,
finite-difference gradients, the autoencoder and decoder fits, host versus device words) are the native
check checks/native_tests/nn_conv.mojo."""
import numpy as np
import pytest

import mojolearn


def _kinds(model):
    return [layer[0] for layer in model.layers]


def test_conv_transpose_describes_its_arguments():
    m = (mojolearn.Sequential(input_shape=(4, 8)).conv_transpose1d(6, 3, stride=2, padding=1, output_padding=1,
                                                                   groups=2, bias=False, dilation=2)
         .flatten().linear(1))
    assert _kinds(m)[0] == "conv_transpose1d"
    assert m.layers[0][1] == (6, 3, 2, 1, 1, 4)        # groups * 2 + bias
    assert m.layers[0][2] == 2.0                       # dilation
    m2 = mojolearn.Sequential(input_shape=(2, 4, 4)).conv_transpose2d(3, 4, stride=2, padding=1).flatten().linear(1)
    assert m2.layers[0][1] == (3, 4, 2, 1, 0, 3)


def test_pools_pads_upsample_shuffle_describe_their_arguments():
    m = (mojolearn.Sequential(input_shape=(4, 6, 6)).adaptive_avg_pool2d((3, None)).adaptive_max_pool2d(2)
         .zero_pad2d((1, 2, 3, 4)).constant_pad2d(1, 0.5).reflection_pad2d(1).replication_pad2d((0, 1, 1, 0))
         .upsample(scale_factor=2, mode="bilinear", align_corners=True).upsample(size=(5, 7))
         .pixel_unshuffle(1).pixel_shuffle(1).global_max_pool().linear(1))
    assert _kinds(m) == ["adaptive_avg_pool2d", "adaptive_max_pool2d", "constant_pad", "constant_pad",
                         "reflection_pad", "replication_pad", "upsample", "upsample", "pixel_shuffle",
                         "pixel_shuffle", "global_max_pool", "linear"]
    assert m.layers[0][1][:2] == (3, 0)                # None keeps the input's width
    assert m.layers[1][1][:2] == (2, 2)
    assert m.layers[2][1] == (1, 2, 3, 4, 2, 0)
    assert m.layers[3][1] == (1, 1, 1, 1, 2, 0) and m.layers[3][2] == 0.5
    assert m.layers[6][1][:4] == (0, 0, 2, 1) and m.layers[6][2] == 2.0
    assert m.layers[7][1][:4] == (5, 7, 0, 0)
    assert m.layers[8][1][:2] == (1, 1) and m.layers[9][1][:2] == (1, 0)
    one = (mojolearn.Sequential(input_shape=(2, 9)).adaptive_max_pool1d(3).reflection_pad1d((1, 2))
           .upsample(size=12, mode="linear").flatten().linear(1))
    assert one.layers[1][1] == (1, 2, 0, 0, 1, 0)
    assert one.layers[2][1][:3] == (12, 12, 1)


@pytest.mark.parametrize("call,message", [
    (lambda m: m.conv_transpose2d(4, 3, stride=2, output_padding=2), "output_padding"),
    (lambda m: m.conv_transpose2d(4, 3, groups=3), "groups"),
    (lambda m: m.upsample(size=4, scale_factor=2), "either size or scale_factor"),
    (lambda m: m.upsample(), "either size or scale_factor"),
    (lambda m: m.upsample(scale_factor=2, align_corners=True), "align_corners"),
    (lambda m: m.upsample(scale_factor=2, mode="bicubic"), "mode"),
    (lambda m: m.upsample(scale_factor=-1.0), "scale_factor"),
    (lambda m: m.reflection_pad2d(-1), "non-negative"),
    (lambda m: m.replication_pad2d((1, 2)), "tuple of 4"),
    (lambda m: m.constant_pad1d(2), "missing"),
    (lambda m: m.zero_pad1d(1, value=2.0), "unexpected keyword argument 'value'"),
    (lambda m: m.constant_pad2d(1, "a"), "value must be a number"),
    (lambda m: m.pixel_shuffle(0), "upscale_factor"),
    (lambda m: m.adaptive_avg_pool2d((2, 3, 4)), "pair"),
    (lambda m: m.adaptive_max_pool1d(None), "output_size"),
    (lambda m: m.global_max_pool(1), "takes 1 positional argument"),
])
def test_refusals_at_add(call, message):
    with pytest.raises(Exception, match=message):
        call(mojolearn.Sequential(input_shape=(4, 6, 6)))


@pytest.mark.parametrize("shape,build,message", [
    ((2, 3, 3), lambda m: m.reflection_pad2d(3), "smaller than the padded dimension"),
    ((3, 4, 4), lambda m: m.pixel_shuffle(2), "divisible by upscale_factor"),
    ((2, 3, 5), lambda m: m.pixel_unshuffle(2), "divisible by downscale_factor"),
    ((2, 4, 4), lambda m: m.upsample(scale_factor=2, mode="linear"), "linear"),
    ((2, 8), lambda m: m.upsample(scale_factor=2, mode="bilinear"), "bilinear"),
    ((2, 4, 4), lambda m: m.adaptive_max_pool1d(2), "AdaptiveMaxPool1d"),
    ((2, 8), lambda m: m.conv_transpose2d(2, 3), "ConvTranspose2d"),
    ((3, 8), lambda m: m.conv_transpose1d(2, 3, groups=2), "groups"),
    ((2, 8), lambda m: m.reflection_pad2d(1), "ReflectionPad"),
])
def test_refusals_at_fit(shape, build, message):
    m = build(mojolearn.Sequential(input_shape=shape)).flatten().linear(1).compile()
    x = np.zeros((4,) + shape, np.float32)
    with pytest.raises(Exception, match=message):
        m.fit(x, np.zeros((4, 1), np.float32), epochs=1)


def _fitted(tmp_path=None):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 2, 4, 4)).astype(np.float32)
    y = rng.standard_normal((16, 1)).astype(np.float32)
    m = (mojolearn.Sequential(input_shape=(2, 4, 4))
         .conv_transpose2d(3, 3, stride=2, padding=1, output_padding=1).tanh()
         .reflection_pad2d(1).upsample(scale_factor=0.5, mode="bilinear").pixel_unshuffle(1)
         .adaptive_max_pool2d(3).conv_transpose2d(2, 2, bias=False).global_max_pool().linear(1)
         .compile(loss="mse", optimizer="adam", lr=0.01))
    m.fit(x, y, epochs=2, batch_size=8, shuffle=False, seed=0)
    return m, x


def test_fit_predict_and_state_dict_round_trip(tmp_path):
    m, x = _fitted()
    pred = np.asarray(m.predict(x))
    assert pred.shape == (16, 1) and np.all(np.isfinite(pred))
    state = m.state_dict()
    assert np.asarray(state["0.weight"]).shape == (2, 3, 3, 3)      # (in, out // groups, k, k), PyTorch's
    assert np.asarray(state["0.bias"]).shape == (3,)
    assert np.asarray(state["6.weight"]).shape == (3, 2, 2, 2)
    assert "6.bias" not in state
    fresh, _ = _fitted()
    fresh.load_state_dict(state)
    np.testing.assert_array_equal(np.asarray(fresh.predict(x)), pred)


def test_checkpoint_of_a_shaped_model_is_refused_in_words(tmp_path):
    # save_checkpoint / export take models of 2-D inputs only (bindings/sequential_checkpoint_door.mojo); every
    # layer here needs a (channels, ...) sample, so the refusal must be the clear one
    m, _ = _fitted()
    with pytest.raises(Exception, match="2-D inputs"):
        m.save_checkpoint(str(tmp_path / "nn_conv.ckpt"))
    with pytest.raises(Exception, match="2-D inputs"):
        m.export(str(tmp_path / "nn_conv.mlx"))
