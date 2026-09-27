# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The sequence lane's optimizers against float64 NumPy restatements of
torch.optim's update rules, at a tolerance. Parity with torch itself was
checked on the lane's pod (docs/lanes/progress/sequence.md)."""
import numpy as np

import mojolearn as ml


def _run(cls, ref, steps=6, **kw):
    rng = np.random.default_rng(0)
    p = rng.standard_normal((4, 3)).astype(np.float32)
    grads = [rng.standard_normal((4, 3)).astype(np.float32) for _ in range(steps)]
    opt = cls([p], **kw)
    q = p.astype(np.float64)
    state = {}
    for t, g in enumerate(grads, 1):
        opt.step([g])
        q = ref(q, g.astype(np.float64), t, state, **kw)
    np.testing.assert_allclose(p, q, rtol=2e-5, atol=2e-6)


def ref_rmsprop(p, g, t, st, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False):
    g = g + weight_decay * p
    st["v"] = alpha * st.get("v", 0.0) + (1 - alpha) * g * g
    v = st["v"]
    if centered:
        st["ga"] = alpha * st.get("ga", 0.0) + (1 - alpha) * g
        v = v - st["ga"] ** 2
    avg = np.sqrt(v) + eps
    if momentum > 0:
        st["b"] = momentum * st.get("b", 0.0) + g / avg
        return p - lr * st["b"]
    return p - lr * g / avg


def test_rmsprop():
    _run(ml.RMSprop, ref_rmsprop)
    _run(ml.RMSprop, ref_rmsprop, lr=3e-3, centered=True, momentum=0.7, weight_decay=1e-2)


def ref_adagrad(p, g, t, st, lr=1e-2, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0,
                eps=1e-10):
    g = g + weight_decay * p
    clr = lr / (1 + (t - 1) * lr_decay)
    st["s"] = st.get("s", initial_accumulator_value) + g * g
    return p - clr * g / (np.sqrt(st["s"]) + eps)


def test_adagrad():
    _run(ml.Adagrad, ref_adagrad)
    _run(ml.Adagrad, ref_adagrad, lr=5e-2, lr_decay=0.1, weight_decay=1e-2, initial_accumulator_value=0.1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
