# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of ops.Session.interpolate, pixel_shuffle, pixel_unshuffle and pad's mode (lane
session-ops-resize, 2026-11-25): the parameter names, the refusals in words, the returned shapes, pad's existing
call unchanged, and a captured Program saved and loaded again. The values, the gradients and the equality with
Sequential's layers are checked natively (checks/native_tests/session_ops_resize.mojo) and by the verify lane
ops-resize through device=."""
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


@pytest.mark.parametrize("name, expected", [
    ("interpolate", ["self", "x", "size", "scale_factor", "mode", "align_corners", "recompute_scale_factor",
                     "antialias"]),
    ("pixel_shuffle", ["self", "x", "upscale_factor"]),
    ("pixel_unshuffle", ["self", "x", "downscale_factor"]),
    ("pad", ["self", "x", "pads", "value", "mode"]),
])
def test_parameter_names(name, expected):
    from mojolearn import ops
    assert list(inspect.signature(getattr(ops.Session, name)).parameters) == expected


@needs_binding
def test_shapes():
    from mojolearn import ops
    with ops.Session() as s:
        x = s.tensor(np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4))
        z = s.tensor(np.arange(10, dtype=np.float32).reshape(1, 2, 5))
        assert s.numpy(s.interpolate(x, scale_factor=2)).shape == (1, 2, 6, 8)
        assert s.numpy(s.interpolate(x, scale_factor=(2, 1.5), mode="bilinear")).shape == (1, 2, 6, 6)
        assert s.numpy(s.interpolate(x, size=5, mode="bilinear", align_corners=True)).shape == (1, 2, 5, 5)
        assert s.numpy(s.interpolate(z, size=(7,), mode="linear")).shape == (1, 2, 7)
        assert s.numpy(s.pixel_shuffle(s.tensor(np.ones((1, 8, 2, 3), np.float32)), 2)).shape == (1, 2, 4, 6)
        assert s.numpy(s.pixel_unshuffle(x, 1)).shape == (1, 2, 3, 4)
        assert s.numpy(s.pad(x, [0, 0, 1, 2, 0, 0, 2, 1], mode="reflect")).shape == (1, 2, 6, 7)
        assert s.numpy(s.pad(z, [0, 0, 3, 0, 0, 9], mode="replicate")).shape == (1, 2, 17)


@needs_binding
def test_pad_constant_unchanged():
    # the existing call (positional value, ONNX pads over any axes) and mode='constant' give the same bytes
    from mojolearn import ops
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    with ops.Session() as s:
        x = s.tensor(data)
        a = np.asarray(s.numpy(s.pad(x, [1, 0, 0, 2], -1.5)))
        b = np.asarray(s.numpy(s.pad(x, [1, 0, 0, 2], value=-1.5, mode="constant")))
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() == np.pad(data, ((1, 0), (0, 2)), constant_values=-1.5).tobytes()


@needs_binding
def test_reflect_and_replicate_values():
    from mojolearn import ops
    with ops.Session() as s:
        z = s.tensor(np.arange(4, dtype=np.float32).reshape(1, 1, 4))
        np.testing.assert_array_equal(s.numpy(s.pad(z, [0, 0, 2, 0, 0, 1], mode="reflect")).ravel(),
                                      [2, 1, 0, 1, 2, 3, 2])
        np.testing.assert_array_equal(s.numpy(s.pad(z, [0, 0, 2, 0, 0, 1], mode="replicate")).ravel(),
                                      [0, 0, 0, 1, 2, 3, 3])


@needs_binding
def test_refusals_in_words():
    from mojolearn import ops
    with ops.Session() as s:
        x = s.tensor(np.ones((1, 2, 2, 3), dtype=np.float32))
        z = s.tensor(np.ones((1, 2, 3), dtype=np.float32))
        v = s.tensor(np.ones((1, 1, 2, 2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="give exactly one of size or scale_factor"):
            s.interpolate(x)
        with pytest.raises(ValueError, match="give exactly one of size or scale_factor"):
            s.interpolate(x, size=4, scale_factor=2)
        with pytest.raises(ValueError, match="mode 'bicubic' is not available"):
            s.interpolate(x, scale_factor=2, mode="bicubic")
        with pytest.raises(ValueError, match="mode 'trilinear' is not available"):
            s.interpolate(v, scale_factor=2, mode="trilinear")
        with pytest.raises(ValueError, match="unknown mode 'cubic'"):
            s.interpolate(x, scale_factor=2, mode="cubic")
        with pytest.raises(ValueError, match="align_corners can only be set with mode 'linear' or 'bilinear'"):
            s.interpolate(x, scale_factor=2, align_corners=False)
        with pytest.raises(ValueError, match="antialias=True is not available"):
            s.interpolate(x, scale_factor=0.5, mode="bilinear", antialias=True)
        with pytest.raises(ValueError, match="recompute_scale_factor needs scale_factor, not size"):
            s.interpolate(x, size=4, recompute_scale_factor=True)
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            s.interpolate(x, scale_factor=-2)
        with pytest.raises(ValueError, match=r"mode 'linear' needs a 3-D input \(N, C, L\)"):
            s.interpolate(x, scale_factor=2, mode="linear")
        with pytest.raises(ValueError, match=r"mode 'bilinear' needs a 4-D input \(N, C, H, W\)"):
            s.interpolate(z, scale_factor=2, mode="bilinear")
        with pytest.raises(ValueError, match="5-D inputs and mode 'trilinear' have no kernel"):
            s.interpolate(v, scale_factor=2)
        with pytest.raises(ValueError, match="size must have 2 values for a 4-D input, got 1"):
            s.interpolate(x, size=(4,))
        with pytest.raises(ValueError, match="must be divisible by upscale_factor squared"):
            s.pixel_shuffle(x, 2)
        with pytest.raises(ValueError, match="must be divisible by downscale_factor"):
            s.pixel_unshuffle(x, 2)
        with pytest.raises(ValueError, match=r"x must be \(N, C, H, W\)"):
            s.pixel_shuffle(z, 1)
        with pytest.raises(ValueError, match="mode 'circular' is not available"):
            s.pad(z, [0, 0, 1, 0, 0, 1], mode="circular")
        with pytest.raises(ValueError, match="unknown mode 'edge'"):
            s.pad(z, [0, 0, 1, 0, 0, 1], mode="edge")
        with pytest.raises(ValueError, match="mode 'reflect' takes no value"):
            s.pad(z, [0, 0, 1, 0, 0, 1], 2.0, mode="reflect")
        with pytest.raises(ValueError, match="must be smaller than the axis it pads"):
            s.pad(z, [0, 0, 3, 0, 0, 0], mode="reflect")
        with pytest.raises(ValueError, match="the batch and channel counts must be 0"):
            s.pad(z, [0, 1, 0, 0, 0, 0], mode="replicate")
        with pytest.raises(ValueError, match="takes non-negative counts"):
            s.pad(z, [0, 0, -1, 0, 0, 0], mode="replicate")
        with pytest.raises(ValueError, match="5-D reflection and replication have no kernel"):
            s.pad(v, [0] * 10, mode="reflect")
        with pytest.raises(TypeError, match="mode must be a string"):
            s.pad(z, [0, 0, 1, 0, 0, 1], mode=1)


@needs_binding
def test_program_save_load_round_trip(tmp_path):
    from mojolearn import ops
    data = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / np.float32(8)
    with ops.Session() as s:
        x = s.tensor(data)
        y = s.pad(s.interpolate(x, scale_factor=1.5, mode="bilinear"), [0, 0, 1, 2, 0, 0, 1, 0], mode="reflect")
        out = s.pixel_shuffle(s.pixel_unshuffle(y, 2), 2)
        expected = s.numpy(out)
        program = s.capture(inputs=[x], outputs=[out])
        path = str(tmp_path / "resize.program")
        program.save(path, verify=False)
        loaded = s.load_program(path)
        again = loaded.run(data)
        again = again[0] if isinstance(again, (list, tuple)) else again
        assert np.asarray(again).tobytes() == np.asarray(expected).tobytes()


@needs_binding
def test_create_graph_through_a_layer_op_is_a_value_or_a_refusal():
    """lane fable-review-2: the layer rules are chosen on the first-order backward paths only. A backward with
    create_graph=True over a graph that holds a layer node (pixel_shuffle records a Sequential layer tag) must
    either give the first derivative (ones: the sum of a rearrangement) or refuse in words; it must never return
    another value or end the process."""
    from mojolearn import ops
    with ops.Session(record=True) as s:
        x = s.tensor(np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2), requires_grad=True)
        loss = s.sum(s.pixel_shuffle(x, 2))
        try:
            s.backward(loss, create_graph=True)
        except (ValueError, TypeError, RuntimeError, NotImplementedError) as refusal:
            assert str(refusal)
            return
        assert np.asarray(s.numpy(s.grad_tensor(x))).reshape(-1).tolist() == [1.0] * 16
