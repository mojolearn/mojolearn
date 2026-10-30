"""Public Conv2d input gradients must equal the explicit pad's adjoint."""
import numpy as np
import pytest

from mojolearn import Conv2d


@pytest.mark.parametrize("shape", [(2, 2, 3, 5), (3, 2, 4, 6)])
@pytest.mark.parametrize("case", ["reflect", "replicate", "circular", "same"])
def test_explicit_padding_backward_matches_independent_scatter(shape, case):
    options = {
        "reflect": dict(out_channels=3, kernel_size=(2, 3), padding=(1, 2), padding_mode="reflect"),
        "replicate": dict(out_channels=4, kernel_size=3, padding=2, padding_mode="replicate", groups=2),
        "circular": dict(out_channels=2, kernel_size=(3, 2), stride=(1, 2), padding=(1, 1), padding_mode="circular"),
        "same": dict(out_channels=2, kernel_size=(2, 4), padding="same", groups=2),
    }
    pads = {"reflect": (1, 1, 2, 2), "replicate": (2, 2, 2, 2),
            "circular": (1, 1, 1, 1), "same": (0, 1, 1, 2)}
    pad_mode = {"reflect": "reflect", "replicate": "edge", "circular": "wrap", "same": "constant"}[case]
    kwargs = options[case]
    layer = Conv2d(2, random_state=51, **kwargs)
    x = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.float32(97) - np.float32(0.4))
    y = layer.forward(x)
    g = np.linspace(-0.5, 0.75, y.size, dtype=np.float32).reshape(y.shape)
    actual = layer.backward(g)

    # A separate zero-padding convolution provides the gradient on the padded
    # domain. NumPy's indexed addition independently applies the pad adjoint.
    top, bottom, left, right = pads[case]
    pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
    xp = np.ascontiguousarray(np.pad(x, pad_width, mode=pad_mode))
    peer_kwargs = dict(kwargs, padding=0, padding_mode="zeros")
    peer = Conv2d(2, random_state=51, **peer_kwargs)
    peer.weight_ = layer.weight_.copy()
    peer.bias_ = layer.bias_.copy()
    np.testing.assert_array_equal(peer.forward(xp), y)
    padded_gradient = peer.backward(g)
    height, width = shape[-2:]
    indices = np.arange(height * width, dtype=np.int64).reshape(height, width)
    extra = {"constant_values": -1} if case == "same" else {}
    mapping = np.pad(indices, ((top, bottom), (left, right)), mode=pad_mode, **extra).ravel()
    valid = mapping >= 0
    expected = np.zeros_like(x)
    for n in range(shape[0]):
        for c in range(shape[1]):
            np.add.at(expected[n, c].reshape(-1), mapping[valid], padded_gradient[n, c].ravel()[valid])
    np.testing.assert_array_equal(actual, expected)
