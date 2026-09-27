# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MoEBlock against a float64 NumPy restatement of Mixtral's sparse block,
and the routing tie rule (ties go to the lower expert index)."""
import numpy as np

import mojolearn as ml


def _silu(v):
    return v / (1 + np.exp(-v))


def test_forward_and_ties():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 8)).astype(np.float32)
    m = ml.MoEBlock(8, 12, num_experts=4, top_k=2, random_state=1)
    m.router *= 30
    y = m(x)
    xd = x.astype(np.float64)
    logits = xd @ m.router.T.astype(np.float64)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ref = np.zeros_like(xd)
    for t in range(len(x)):
        idx = np.argsort(-p[t], kind="stable")[:2]
        w = p[t, idx] / p[t, idx].sum()
        for j, e in enumerate(idx):
            g, u = np.split(m.gate_up_proj[e].astype(np.float64) @ xd[t], 2)
            ref[t] += w[j] * (m.down_proj[e].astype(np.float64) @ (_silu(g) * u))
    np.testing.assert_allclose(y, ref, atol=2e-5)
    t = ml.MoEBlock(8, 4, num_experts=3, top_k=1, random_state=2)
    t.router[:] = 0.0                      # every probability ties
    t(x)
    assert np.all(t.selected_experts_[:, 0] == 0)


if __name__ == "__main__":
    test_forward_and_ties()
    print("PASS test_forward_and_ties")
