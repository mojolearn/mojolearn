# SPDX-License-Identifier: Apache-2.0
"""Independent LayerNorm forward and derivative oracles for pending views."""
import numpy as np
import pytest

from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call


def native(call, *args, **kwargs):
    return audit_native_call(call, *args, **kwargs).assert_native()


@pytest.mark.parametrize("views", ["input", "affine", "all"])
def test_layernorm_pending_views_preserve_forward_and_gradients(views):
    x = np.array([[.2, -.7, 1.1], [2., .3, -.4], [-.1, .8, 1.7], [.6, -.2, -.9]], np.float32)
    w = np.array([[.7, -1.2, .4]], np.float32)
    b = np.array([[.2, .1, -.3]], np.float32)
    seed = np.linspace(-.6, .9, x.size, dtype=np.float32).reshape(x.shape)
    input_view, affine_view = views in ("input", "all"), views in ("affine", "all")
    eps = .03
    with ops.Session(record=True) as session:
        hx = session.tensor(np.ascontiguousarray(x.T if input_view else x), requires_grad=True)
        hw = session.tensor(np.ascontiguousarray(w.T if affine_view else w), requires_grad=True)
        hb = session.tensor(np.ascontiguousarray(b.T if affine_view else b), requires_grad=True)
        vx = session.transpose(hx) if input_view else hx
        vw = session.transpose(hw) if affine_view else hw
        vb = session.transpose(hb) if affine_view else hb
        output = native(session.layer_norm, vx, vw, vb, eps=eps)
        native(session.backward, session.sum(session.mul(output, session.tensor(seed))))
        z = x.astype(np.float64)
        centered = z - z.mean(axis=-1, keepdims=True)
        inv = 1 / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + eps)
        normalized = centered * inv
        weighted = seed.astype(np.float64) * w
        dx = inv * (weighted - weighted.mean(axis=-1, keepdims=True)
                    - normalized * np.mean(weighted * normalized, axis=-1, keepdims=True))
        dw = np.sum(seed.astype(np.float64) * normalized, axis=0, keepdims=True)
        db = np.sum(seed.astype(np.float64), axis=0, keepdims=True)
        np.testing.assert_allclose(native(session.numpy, output), normalized*w+b, rtol=4e-6, atol=2e-7)
        for handle, expected in ((hx, dx.T if input_view else dx),
                                 (hw, dw.T if affine_view else dw),
                                 (hb, db.T if affine_view else db)):
            actual = native(session.grad, handle)
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=3e-7)
