# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of ops.Session.conv1d, conv3d, conv_transpose1d, conv_transpose2d and conv2d's dilation and
per-axis words (lane session-ops-conv, 2026-11-25): the parameter names and order (PyTorch's), the admission of an
int or a per-axis tuple and of the padding words, the refusals in words, the returned shapes, and a captured
Program saved and loaded again. The values and gradients are checked natively
(checks/native_tests/session_ops_conv.mojo: equal to Sequential's layers word for word, a float64 oracle, central
differences) and by the verify lane ops-conv-layers through device=."""
import inspect

import numpy as np
import pytest

from mojolearn import _backend


def _has_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_session_call")
    except Exception:
        return False


needs_binding = pytest.mark.skipif(not _has_entries(),
                                   reason="the sequence binding has no ops_session_call entry (rebuild it)")

_CONV = ["self", "x", "weight", "bias", "stride", "padding", "dilation", "groups"]
_TRANSPOSE = ["self", "x", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"]


@pytest.mark.parametrize("name, expected", [
    ("conv1d", _CONV), ("conv2d", _CONV), ("conv3d", _CONV),
    ("conv_transpose1d", _TRANSPOSE), ("conv_transpose2d", _TRANSPOSE),
])
def test_parameter_names(name, expected):
    from mojolearn import ops
    assert list(inspect.signature(getattr(ops.Session, name)).parameters) == expected


def _t(s, shape, value=0.1):
    return s.tensor(np.full(shape, value, dtype=np.float32))


@needs_binding
def test_shapes():
    from mojolearn import ops
    with ops.Session() as s:
        assert s.shape(s.conv1d(_t(s, (2, 4, 7)), _t(s, (4, 2, 3)), _t(s, (4,)), stride=2, padding=2,
                                dilation=2, groups=2)) == (2, 4, 4)
        assert s.shape(s.conv1d(_t(s, (2, 3, 8)), _t(s, (4, 3, 3)), padding="same")) == (2, 4, 8)
        assert s.shape(s.conv1d(_t(s, (2, 3, 8)), _t(s, (4, 3, 3)), padding="valid")) == (2, 4, 6)
        assert s.shape(s.conv2d(_t(s, (1, 2, 5, 6)), _t(s, (3, 2, 2, 3)), stride=(1, 2), padding=(2, 0),
                                dilation=(2, 1))) == (1, 3, 7, 2)
        # the default call keeps the old operation; a square call and its tuple spelling agree
        a = s.numpy(s.conv2d(_t(s, (1, 2, 5, 5), 0.25), _t(s, (3, 2, 3, 3)), stride=2, padding=1))
        b = s.numpy(s.conv2d(_t(s, (1, 2, 5, 5), 0.25), _t(s, (3, 2, 3, 3)), stride=(2, 2), padding=(1, 1),
                             dilation=(1, 1)))
        assert a.tobytes() == b.tobytes()
        assert s.shape(s.conv3d(_t(s, (1, 2, 3, 4, 4)), _t(s, (2, 1, 2, 2, 2)), stride=(1, 2, 1),
                                padding=(1, 0, 1), dilation=(1, 1, 2), groups=2)) == (1, 2, 4, 2, 4)
        assert s.shape(s.conv_transpose1d(_t(s, (1, 2, 4)), _t(s, (2, 3, 3)), _t(s, (3,)), stride=2, padding=1,
                                          output_padding=1)) == (1, 3, 8)
        assert s.shape(s.conv_transpose2d(_t(s, (1, 2, 3, 3)), _t(s, (2, 2, 3, 2)), stride=(2, 1),
                                          padding=(1, 0), output_padding=(1, 0), groups=2,
                                          dilation=(1, 2))) == (1, 4, 6, 5)


@needs_binding
def test_refusals_in_words():
    from mojolearn import ops
    with ops.Session() as s:
        x1, w1 = _t(s, (1, 2, 4)), _t(s, (2, 2, 3))
        with pytest.raises(ValueError, match=r"conv1d: the dilated kernel \(5\) is larger than the padded input \(4\)"):
            s.conv1d(x1, w1, dilation=2)
        with pytest.raises(ValueError, match="conv1d: stride must be an integer or a tuple of 1 integers, got 2"):
            s.conv1d(x1, w1, stride=(1, 1))
        with pytest.raises(ValueError, match="conv1d: padding must be an integer, a tuple of integers, 'valid' or 'same'"):
            s.conv1d(x1, w1, padding="full")
        with pytest.raises(ValueError, match="padding='same' needs stride 1"):
            s.conv1d(x1, w1, padding="same", stride=2)
        with pytest.raises(ValueError, match=r"padding='same' needs dilation \* \(kernel_size - 1\) even"):
            s.conv1d(_t(s, (1, 2, 6)), _t(s, (2, 2, 2)), padding="same")
        with pytest.raises(TypeError, match="stride must be an integer"):
            s.conv1d(x1, w1, stride=1.5)
        with pytest.raises(ValueError, match="dilation must be >= 1"):
            s.conv2d(_t(s, (1, 2, 4, 4)), _t(s, (2, 2, 2, 2)), dilation=0)
        with pytest.raises(ValueError, match=r"with groups \(2\) dividing C \(3\)"):
            s.conv3d(_t(s, (1, 3, 2, 2, 2)), _t(s, (2, 1, 1, 1, 1)), groups=2)
        with pytest.raises(ValueError, match=r"conv3d takes x \(N, C, D, H, W\)"):
            s.conv3d(_t(s, (1, 2, 4, 4)), _t(s, (3, 2, 2, 2)))
        with pytest.raises(ValueError, match=r"the bias needs one value per output channel \(2\)"):
            s.conv1d(x1, w1, _t(s, (3,)))
        with pytest.raises(ValueError, match="output_padding must be smaller than stride"):
            s.conv_transpose1d(x1, w1, stride=2, output_padding=2)
        with pytest.raises(ValueError, match="the weight's first dimension must be x's channels"):
            s.conv_transpose2d(_t(s, (1, 2, 3, 3)), _t(s, (3, 2, 2, 2)))
        with pytest.raises(TypeError, match="must be floating-point tensors"):
            s.conv1d(s.ids(np.zeros((2, 4), dtype=np.int64)), w1)


@needs_binding
def test_gradients_reach_every_input():
    from mojolearn import ops
    with ops.Session(record=True) as s:
        x = s.tensor(np.ones((1, 2, 4), np.float32), requires_grad=True)
        w = s.tensor(np.full((2, 3, 3), 0.5, np.float32), requires_grad=True)
        b = s.tensor(np.zeros((3,), np.float32), requires_grad=True)
        s.backward(s.sum(s.conv_transpose1d(x, w, b, stride=2)))
        assert s.grad(x).shape == (1, 2, 4)
        assert s.grad(w).shape == (2, 3, 3)
        assert np.asarray(s.grad(b)).size == 3


@needs_binding
def test_program_save_load_round_trip(tmp_path):
    from mojolearn import ops
    x1 = np.linspace(-1, 1, 2 * 4 * 7, dtype=np.float32).reshape(2, 4, 7)
    w1 = np.linspace(-0.5, 0.5, 4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    x2 = np.linspace(-1, 1, 1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3)
    w2 = np.linspace(-0.5, 0.5, 2 * 2 * 3 * 2, dtype=np.float32).reshape(2, 2, 3, 2)
    with ops.Session() as s:
        h1, hw1, h2, hw2 = s.tensor(x1), s.tensor(w1), s.tensor(x2), s.tensor(w2)
        y1 = s.conv1d(h1, hw1, stride=2, padding=2, dilation=2, groups=2)
        y2 = s.conv_transpose2d(h2, hw2, stride=(2, 1), padding=(1, 0), output_padding=(1, 0), groups=2,
                                dilation=(1, 2))
        y3 = s.conv2d(h2, s.tensor(np.ones((2, 2, 2, 2), np.float32)), padding=(1, 0), dilation=(2, 1))
        expected = [s.numpy(y1), s.numpy(y2), s.numpy(y3)]
        program = s.capture(inputs=[h1, hw1, h2, hw2], outputs=[y1, y2, y3])
        path = str(tmp_path / "convolutions.program")
        program.save(path, verify=False)
        loaded = s.load_program(path)
        again = loaded.run(x1, w1, x2, w2)
        for got, want in zip(again, expected):
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
