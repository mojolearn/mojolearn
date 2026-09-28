# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every `_mojolearn_x_metrics` entry family, called TWICE in one process, on
the GPU binding and on the CPU host binding (CURRENT DIRECTIVES 2026-09-27:
x_cluster and x_neighbors hung on the second GPU call of a process because
each call built a DeviceContext whose buffers outlived it; x_metrics keeps
ONE process-lifetime context, x_metrics/device.mojo `metrics_ctx`). The
binding's one entry point is `x_metrics_run`; the calls below reach every
user op and every planned parallel schedule (sorts, curves, weighted
percentiles, group sums, permutations). The two passes must return the same
bytes, and the GPU the host's.

    .pixi/envs/test/bin/python -m pytest python/mojolearn/tests/test_x_metrics_repeat.py -q
"""
import hashlib

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn import _expansion_metrics as X


def _data(n=3000):
    rng = np.random.default_rng(7)
    yr = (np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32)
    pr = (yr + 0.3 * rng.standard_normal(n).astype(np.float32)).clip(0.01).astype(np.float32)
    ct = rng.integers(0, 5, n)
    cp = np.where(rng.random(n) < 0.6, ct, rng.integers(0, 5, n))
    proba = rng.random((n, 5)).astype(np.float32)
    proba /= proba.sum(axis=1, keepdims=True)
    hy = rng.integers(0, 2, n)
    hs = np.round(rng.random(n), 3).astype(np.float32)  # ties on purpose
    sw = rng.integers(1, 4, n).astype(np.float32)
    Xc = rng.standard_normal((n, 4)).astype(np.float32)
    return dict(yr=yr, pr=pr, ct=ct, cp=cp, proba=proba, hy=hy, hs=hs, sw=sw, X=Xc, n=n)


def _digest(v):
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                walk(e)
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                walk(x[k])
        else:
            a = np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)
            h.update(repr(a.tolist()).encode() if a.dtype == object else np.ascontiguousarray(a).tobytes())
    walk(v)
    return h.hexdigest()


def _calls(d):
    import mojolearn.metrics as M
    import mojolearn.model_selection as S
    n = d["n"]
    y2 = np.stack([d["yr"], d["yr"] * 2], axis=1)
    p2 = np.stack([d["pr"], d["pr"] * 2], axis=1)
    E = np.empty((n, 1))
    cases = [
        ("median_absolute_error", lambda: M.median_absolute_error(d["yr"], d["pr"])),
        ("median_absolute_error_w", lambda: M.median_absolute_error(d["yr"], d["pr"], sample_weight=d["sw"])),
        ("mean_pinball_loss", lambda: M.mean_pinball_loss(d["yr"], d["pr"], alpha=0.3)),
        ("d2_absolute_error_score", lambda: M.d2_absolute_error_score(d["yr"], d["pr"])),
        ("max_error", lambda: M.max_error(d["yr"], d["pr"])),
        ("r2_score_w_mo", lambda: M.r2_score(y2, p2, sample_weight=d["sw"], multioutput="raw_values")),
        ("mean_tweedie_deviance", lambda: M.mean_tweedie_deviance(d["yr"], d["pr"], power=1.5)),
        ("matthews_corrcoef", lambda: M.matthews_corrcoef(d["ct"], d["cp"])),
        ("prfs_w", lambda: M.precision_recall_fscore_support(d["ct"], d["cp"], average=None, sample_weight=d["sw"])),
        ("roc_curve", lambda: M.roc_curve(d["hy"], d["hs"])),
        ("roc_curve_w", lambda: M.roc_curve(d["hy"], d["hs"], sample_weight=d["sw"])),
        ("average_precision_score", lambda: M.average_precision_score(d["hy"], d["hs"])),
        ("roc_auc_ovr", lambda: M.roc_auc_score(d["ct"], d["proba"], multi_class="ovr")),
        ("log_loss_w", lambda: M.log_loss(d["ct"], d["proba"], sample_weight=d["sw"])),
        ("top_k_accuracy_score", lambda: M.top_k_accuracy_score(d["ct"], d["proba"], k=2)),
        ("adjusted_mutual_info_score", lambda: M.adjusted_mutual_info_score(d["ct"], d["cp"])),
        ("davies_bouldin_score", lambda: M.davies_bouldin_score(d["X"], d["ct"])),
        ("kfold_shuffle", lambda: [np.asarray(t[1]) for t in S.KFold(5, shuffle=True, random_state=0).split(E)]),
        ("stratified_kfold_shuffle", lambda: [np.asarray(t[1]) for t in
                                              S.StratifiedKFold(5, shuffle=True, random_state=0).split(E, d["ct"])]),
        ("shuffle_split", lambda: [np.asarray(t[1]) for t in S.ShuffleSplit(3, test_size=0.2, random_state=0).split(E)]),
        ("train_test_split", lambda: [np.asarray(a) for a in S.train_test_split(d["yr"], test_size=0.25, random_state=0)]),
    ]
    return [(name, _digest(fn())) for name, fn in cases]


def _gpu_available():
    try:
        X._binding(None)
        return True
    except Exception:  # a CPU-only install has no GPU binding
        return False


def test_every_entry_twice_host_and_gpu(monkeypatch):
    d = _data()
    g1 = g2 = None
    if _gpu_available():
        g1, g2 = _calls(d), _calls(d)
    host = _backend.load_host_module("_mojolearn_x_metrics_host")
    with monkeypatch.context() as m:
        m.setattr(X, "_binding", lambda numeric_mode: host)
        h1, h2 = _calls(d), _calls(d)
    assert h1 == h2, [n for (n, a), (_, b) in zip(h1, h2) if a != b]
    if g1 is None:
        pytest.skip("no GPU binding in this process")
    assert g1 == g2, [n for (n, a), (_, b) in zip(g1, g2) if a != b]
    assert g1 == h1, [n for (n, a), (_, b) in zip(g1, h1) if a != b]
