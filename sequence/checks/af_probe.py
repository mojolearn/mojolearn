# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Adafactor through the loaded binding (Metal on a Mac, cuda on the pod):
the sequence-adafactor lane's run on fixed arrays, the params' and state's
bits after every step, to diff against the CPU/NVIDIA column.

    PYTHONPATH=python pixi run python sequence/checks/af_probe.py
"""
import hashlib

import numpy as np

import mojolearn as ml


def h(*a):
    m = hashlib.sha256()
    for x in a:
        m.update(np.ascontiguousarray(x).tobytes())
    return m.hexdigest()[:16]


print("vendor", ml.vendor(), "mode", ml.numeric_mode())
rng = np.random.default_rng(5)
X = rng.standard_normal((400, 16)).astype(np.float32)
for kw in ({}, dict(lr=3e-2, beta2_decay=-0.6, d=2.0, weight_decay=0.1)):
    p1 = np.ascontiguousarray(X[:32, :8]).copy()
    p2 = np.ascontiguousarray(X[32:40, 0]).copy()
    opt = ml.Adafactor([p1, p2], **kw)
    for k in range(6):
        base = 40 + 48 * k
        g1 = np.ascontiguousarray(X[base:base + 32, 8:16]) * np.float32(0.5)
        g2 = np.ascontiguousarray(X[base + 32:base + 40, 1])
        opt.step([g1, g2])
        st = [v for s in opt.state for v in s.values()]
        print(kw, "step", k + 1, "p1", h(p1), "p2", h(p2), "state", h(*st))
        if k == 0:
            print("  p1[0]", p1[0].view(np.uint32).tolist())
            print("  p2", p2.view(np.uint32).tolist())
