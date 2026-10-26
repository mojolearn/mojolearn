# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane py-shared: the shared ranges runner and the resident input store
move no bit.

`core/arena_io.mojo` uploads only an arena program's input words (the rest
is zeroed on the device) and downloads only its outputs; `DeviceCache` /
`resident()` keep a big input on the device between programs. Every x_prep
and x_metrics program below runs three ways on the GPU binding: the
whole-arena entries (`MOJOLEARN_ARENA_RANGES=0`), the ranges entries, and
the ranges entries inside `resident()` (every x_prep input resident, the battery
run twice so the second pass re-uses the device copies; x_metrics programs
have no resident store and copy their inputs). All three must return the
same bytes, and the store must hold nothing after the block.

    .pixi/envs/test/bin/python -m pytest python/mojolearn/tests/test_arena_ranges.py -q
"""
import hashlib

import numpy as np
import pytest

from mojolearn import _arena_io
from mojolearn import _expansion_metrics as XM
from mojolearn import _expansion_prep as XP


def _digest(v):
    h = hashlib.sha256()
    for x in v:
        a = np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)
        h.update(repr(a.tolist()).encode() if a.dtype == object else np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _prep_calls(ml):
    rng = np.random.default_rng(7)
    X = np.ascontiguousarray(rng.normal(size=(4000, 5)), dtype=np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64) + (X[:, 2] > 0.5)
    C = np.floor(np.abs(X) * 2).astype(np.float32)
    Xn = X.copy()
    Xn[::9, 1] = np.nan

    def run():
        g = ml.GaussianNB().fit(X, y)
        return [
            ("RobustScaler", lambda: ml.RobustScaler().fit(X).transform(X)),
            ("MaxAbsScaler", lambda: ml.MaxAbsScaler().fit(X).transform(X)),
            ("QuantileTransformer", lambda: ml.QuantileTransformer(n_quantiles=50).fit(X).transform(X)),
            ("PowerTransformer", lambda: ml.PowerTransformer().fit(X).transform(X)),
            ("OneHotEncoder", lambda: ml.OneHotEncoder().fit(C).transform(C)),
            ("OrdinalEncoder", lambda: ml.OrdinalEncoder().fit(C).transform(C)),
            ("TargetEncoder", lambda: ml.TargetEncoder(random_state=0).fit_transform(C, y)),
            ("SimpleImputer", lambda: ml.SimpleImputer().fit(Xn).transform(Xn)),
            ("IterativeImputer", lambda: ml.IterativeImputer(max_iter=3, random_state=0).fit(Xn).transform(Xn)),
            ("KBinsDiscretizer", lambda: ml.KBinsDiscretizer(n_bins=4, encode="ordinal").fit(X).transform(X)),
            ("SplineTransformer", lambda: ml.SplineTransformer().fit(X).transform(X)),
            ("PolynomialFeatures", lambda: ml.PolynomialFeatures(2).fit(X).transform(X)),
            ("PolynomialFeaturesF", lambda: ml.PolynomialFeatures(2, order="F").fit(X).transform(X)),
            ("GaussianNB", lambda: [g.predict_proba(X), g.predict(X)]),
            ("MultinomialNB", lambda: ml.MultinomialNB().fit(C, y).predict_proba(C)),
            ("CategoricalNB", lambda: ml.CategoricalNB().fit(C, y).predict_proba(C)),
            ("LDA", lambda: ml.LinearDiscriminantAnalysis().fit(X, y).predict_proba(X)),
            ("QDA", lambda: ml.QuadraticDiscriminantAnalysis().fit(X, y).predict_proba(X)),
            ("SelectKBest", lambda: ml.SelectKBest(k=2).fit(X, y).scores_),
            ("mutual_info_classif", lambda: ml.mutual_info_classif(X, y, random_state=0)),
            ("LabelEncoder", lambda: ml.LabelEncoder().fit(y).transform(y)),
        ]
    return run


def _battery():
    import mojolearn as ml
    from mojolearn.tests import test_x_metrics_repeat as R

    d = R._data(20000)
    prep = _prep_calls(ml)

    def run():
        out = []
        for name, fn in prep():
            v = fn()
            out.append((name, _digest(v if isinstance(v, list) else [v])))
        out += [("m:" + n, h) for n, h in R._calls(d)]
        return out
    return run


def _gpu_ranges():
    try:
        mb = XM._binding(None)
        pb = XP._prep_binding(None)
    except Exception:  # a CPU-only install
        return None
    if not (hasattr(mb, "x_metrics_run_ranges") and hasattr(pb, "x_prep_run_ranges")):
        return None
    return mb, pb


def test_ranges_and_resident_move_no_bit(monkeypatch):
    b = _gpu_ranges()
    if b is None:
        pytest.skip("no GPU binding with the ranges entries in this process")
    mb, pb = b
    run = _battery()
    monkeypatch.setenv("MOJOLEARN_ARENA_RANGES", "0")
    whole = run()
    monkeypatch.setenv("MOJOLEARN_ARENA_RANGES", "1")
    ranges = run()
    with _arena_io.resident(min_words=1) as r:
        res1 = run()
        res2 = run()
        stats = r.stats()
        live_inside = int(pb.x_prep_dev_live())
    live_after = int(pb.x_prep_dev_live())
    for arm, got in (("ranges", ranges), ("resident", res1), ("resident second pass", res2)):
        assert got == whole, (arm, [n for (n, a), (_, c) in zip(whole, got) if a != c])
    assert stats and all(u > 0 for u, _ in stats.values()), stats
    assert all(h > 0 for _, h in stats.values()), stats
    assert live_inside > 0, live_inside
    assert live_after == 0, live_after


def test_device_cache_frees_deterministically():
    b = _gpu_ranges()
    if b is None:
        pytest.skip("no GPU binding with the ranges entries in this process")
    _, pb = b
    from mojolearn._array import Array
    from mojolearn._verify_native_runtime import audit_native_call
    a = Array.from_list([float(i) for i in range(1000)], "<f4")
    c = _arena_io.DeviceCache(pb, "x_prep")
    base = c.live()
    i1 = c.id_of(a)
    assert audit_native_call(c.get, a).assert_native() == i1
    assert c.id_of(a) == i1 and c.hits == 1 and c.uploads == 1
    assert c.live() == base + 1
    c.release(a)
    assert c.live() == base and audit_native_call(c.get, a).assert_native() is None
    with _arena_io.DeviceCache(pb, "x_prep") as c2:
        c2.id_of(a)
        assert c2.live() == base + 1
    assert c2.live() == base
