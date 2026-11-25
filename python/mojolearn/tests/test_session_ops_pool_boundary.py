# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of Sequential's pools as ops.Session operations (sub-lane session-ops-pool, 2026-11-25):
the parameter names (torch.nn.functional's), the admission of int-or-tuple arguments and of None in an adaptive
output size, the refusals in words, the returned shapes, a captured Program saved and loaded again, and a pool
inside a scan cell and inside a custom operation's program. The values and gradients are checked natively
(checks/native_tests/session_ops_pool.mojo) and by the verify lanes ops-pool-window and ops-pool-adaptive through
device=."""
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

_MAX = ["self", "x", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"]
_AVG = ["self", "x", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"]


@pytest.mark.parametrize("name, expected", [
    ("max_pool1d", _MAX),
    ("max_pool2d", _MAX),
    ("max_pool3d", _MAX),
    ("avg_pool1d", _AVG[:-1]),
    ("avg_pool2d", _AVG),
    ("avg_pool3d", _AVG),
    ("adaptive_avg_pool1d", ["self", "x", "output_size"]),
    ("adaptive_avg_pool2d", ["self", "x", "output_size"]),
    ("adaptive_avg_pool3d", ["self", "x", "output_size"]),
    ("adaptive_max_pool1d", ["self", "x", "output_size", "return_indices"]),
    ("adaptive_max_pool2d", ["self", "x", "output_size", "return_indices"]),
])
def test_parameter_names(name, expected):
    from mojolearn import ops
    assert list(inspect.signature(getattr(ops.Session, name)).parameters) == expected


def test_defaults_are_torchs():
    from mojolearn import ops
    p = inspect.signature(ops.Session.avg_pool2d).parameters
    assert (p["stride"].default, p["padding"].default, p["ceil_mode"].default, p["count_include_pad"].default,
            p["divisor_override"].default) == (None, 0, False, True, None)
    p = inspect.signature(ops.Session.max_pool3d).parameters
    assert (p["dilation"].default, p["return_indices"].default) == (1, False)


@needs_binding
def test_shapes_and_admission():
    from mojolearn import ops
    with ops.Session() as s:
        x1 = s.tensor(np.ones((2, 3, 7), dtype=np.float32))
        x2 = s.tensor(np.ones((1, 2, 5, 6), dtype=np.float32))
        x3 = s.tensor(np.ones((1, 2, 3, 4, 4), dtype=np.float32))
        assert s.numpy(s.max_pool1d(x1, 2, stride=2, padding=1, dilation=2, ceil_mode=True)).shape == (2, 3, 4)
        assert s.numpy(s.avg_pool1d(x1, 3, stride=2, padding=1, count_include_pad=False)).shape == (2, 3, 4)
        assert s.numpy(s.max_pool2d(x2, (3, 2), stride=(2, 1), padding=(1, 1), dilation=(1, 2))).shape == (1, 2, 3, 6)
        assert s.numpy(s.max_pool2d(x2, 2)).shape == (1, 2, 2, 3)
        assert s.numpy(s.avg_pool2d(x2, 2, divisor_override=3)).shape == (1, 2, 2, 3)
        assert s.numpy(s.max_pool3d(x3, 2, stride=(1, 2, 2), padding=(1, 1, 0), ceil_mode=True)).shape == \
            (1, 2, 4, 3, 2)
        assert s.numpy(s.avg_pool3d(x3, (1, 2, 2))).shape == (1, 2, 3, 2, 2)
        assert s.numpy(s.adaptive_avg_pool1d(x1, 3)).shape == (2, 3, 3)
        assert s.numpy(s.adaptive_avg_pool1d(x1, 1)).shape == (2, 3, 1)
        assert s.numpy(s.adaptive_max_pool1d(x1, 4)).shape == (2, 3, 4)
        assert s.numpy(s.adaptive_avg_pool2d(x2, (None, 4))).shape == (1, 2, 5, 4)
        assert s.numpy(s.adaptive_max_pool2d(x2, 1)).shape == (1, 2, 1, 1)
        assert s.numpy(s.adaptive_avg_pool3d(x3, (None, 2, 1))).shape == (1, 2, 3, 2, 1)
        ids = s.ids(np.zeros((2, 3, 7), dtype=np.int64))
        with pytest.raises(ValueError, match="max_pool1d: x must be a floating-point tensor"):
            s.max_pool1d(ids, 2)


@needs_binding
def test_refusals_in_words():
    from mojolearn import ops
    with ops.Session() as s:
        x1 = s.tensor(np.ones((1, 1, 7), dtype=np.float32))
        x2 = s.tensor(np.ones((1, 1, 4, 4), dtype=np.float32))
        x3 = s.tensor(np.ones((1, 1, 2, 4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="return_indices=True is not supported: a Session operation returns "
                                             "one tensor"):
            s.max_pool2d(x2, 2, return_indices=True)
        with pytest.raises(ValueError, match="return_indices=True is not supported"):
            s.adaptive_max_pool1d(x1, 2, return_indices=True)
        with pytest.raises(ValueError, match="kernel_size must be an integer or a tuple of 2 integers, got 3 values"):
            s.max_pool2d(x2, (2, 2, 2))
        with pytest.raises(TypeError, match="kernel_size must be an integer"):
            s.avg_pool1d(x1, 2.5)
        with pytest.raises(ValueError, match="stride must be >= 1"):
            s.max_pool3d(x3, 2, stride=0)
        with pytest.raises(ValueError, match=r"padding must be non-negative and at most half of kernel_size "
                                             r"\(padding 2, kernel_size 3\)"):
            s.max_pool1d(x1, 3, padding=2)
        with pytest.raises(ValueError, match="the window does not fit the padded input"):
            s.avg_pool1d(x1, 9)
        with pytest.raises(ValueError, match="divisor_override must be >= 1"):
            s.avg_pool2d(x2, 2, divisor_override=0)
        with pytest.raises(ValueError, match=r"max_pool1d: x must be \(N, C, L\), got a tensor of 4 dimensions"):
            s.max_pool1d(x2, 2)
        with pytest.raises(ValueError, match=r"avg_pool3d: x must be \(N, C, D, H, W\)"):
            s.avg_pool3d(x2, 2)
        with pytest.raises(ValueError, match="output_size must be >= 1"):
            s.adaptive_avg_pool2d(x2, 0)
        with pytest.raises(ValueError, match="output_size must be an integer or a tuple of 3 integers or None"):
            s.adaptive_avg_pool3d(x3, (1, 1))
        with pytest.raises(Exception, match="the output sizes must be 1 or divide the input's"):
            s.adaptive_avg_pool3d(x3, 3)


@needs_binding
def test_program_save_load_round_trip(tmp_path):
    from mojolearn import ops
    a = (np.arange(42, dtype=np.float32).reshape(2, 3, 7) * np.float32(0.25)) - np.float32(4.0)
    b = np.arange(70, dtype=np.float32).reshape(1, 2, 5, 7) * np.float32(-0.125)
    with ops.Session() as s:
        x = s.tensor(a)
        y = s.tensor(b)
        out = [s.max_pool1d(x, 2, padding=1, dilation=2, ceil_mode=True),
               s.avg_pool2d(y, 3, stride=2, padding=1, count_include_pad=False),
               s.max_pool2d(y, 2),
               s.adaptive_avg_pool1d(x, 3),
               s.adaptive_max_pool2d(y, (None, 3))]
        expected = [s.numpy(h) for h in out]
        program = s.capture(inputs=[x, y], outputs=out)
        path = str(tmp_path / "pools.program")
        program.save(path, verify=False)
        loaded = s.load_program(path)
        again = loaded.run(a, b)
        for got, want in zip(again, expected):
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


@needs_binding
def test_pool_in_a_scan_cell_and_a_custom_operation():
    from mojolearn import ops
    xs = np.arange(4 * 2 * 7, dtype=np.float32).reshape(4, 1, 2, 7) * np.float32(0.5)
    with ops.Session() as c:
        state = c.tensor(np.zeros((1, 2, 3), dtype=np.float32))
        x_t = c.tensor(xs[0])
        pooled = c.adaptive_avg_pool1d(x_t, 3)
        cell = c.capture(inputs=[state, x_t], outputs=[c.add(state, pooled), pooled])
        with ops.Session() as s:
            ys, final = s.scan(cell, s.tensor(xs), s.tensor(np.zeros((1, 2, 3), dtype=np.float32)))
            direct = [s.numpy(s.adaptive_avg_pool1d(s.tensor(xs[t]), 3)) for t in range(4)]
            assert np.asarray(s.numpy(ys)).tobytes() == np.stack(direct).tobytes()
    x0 = np.arange(14, dtype=np.float32).reshape(1, 2, 7)
    with ops.Session() as f, ops.Session() as g:
        fx = f.tensor(x0)
        forward = f.capture(inputs=[fx], outputs=[f.max_pool1d(fx, 2, ceil_mode=True)])
        gy = g.tensor(np.zeros((1, 2, 4), dtype=np.float32))
        backward = g.capture(inputs=[gy], outputs=[g.pad(gy, (0, 0, 0, 0, 0, 3))])
        with ops.Session(record=True) as s:
            op = s.custom_op(forward, backward)
            x = s.tensor(x0, requires_grad=True)
            y = s.custom(op, x)
            assert np.asarray(s.numpy(y)).tobytes() == np.asarray(
                [[[1, 3, 5, 6], [8, 10, 12, 13]]], dtype=np.float32).tobytes()
