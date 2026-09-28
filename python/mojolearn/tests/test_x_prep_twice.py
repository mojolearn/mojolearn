# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's one device entry point, `x_prep_run` (every estimator in
python/mojolearn/_expansion_prep.py runs its program through it), called many
times in one process and the whole sequence run TWICE, on the GPU binding and
on the CPU host binding, with the second pass's output equal to the first's
byte for byte.

CURRENT DIRECTIVES 2026-09-27: x_cluster and x_neighbors hung on the SECOND
GPU call in a process because each call built a new DeviceContext whose
buffers outlived it. `x_prep/device.mojo` now holds ONE process-lifetime
context (`x_prep_ctx`, the x_cnn `_Global` pattern). Each backend runs in a
child process with a timeout, so a hang is a failure, not a stuck test run."""
import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent('''
    import numpy as np
    import mojolearn as ml

    rng = np.random.default_rng(7)
    X = np.ascontiguousarray(rng.normal(size=(120, 4)), dtype=np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64) + (X[:, 2] > 0.5)
    C = np.floor(np.abs(X) * 2).astype(np.float32)
    Xn = X.copy(); Xn[::9, 1] = np.nan

    def run():
        out = []
        out.append(ml.RobustScaler().fit(X).transform(X))
        out.append(ml.MaxAbsScaler().fit(X).transform(X))
        out.append(ml.QuantileTransformer(n_quantiles=50).fit(X).transform(X))
        out.append(ml.PowerTransformer().fit(X).transform(X))
        out.append(ml.OneHotEncoder().fit(C).transform(C))
        out.append(ml.OrdinalEncoder().fit(C).transform(C))
        out.append(ml.TargetEncoder(random_state=0).fit_transform(C, y))
        out.append(ml.SimpleImputer().fit(Xn).transform(Xn))
        out.append(ml.IterativeImputer(max_iter=3, random_state=0).fit(Xn).transform(Xn))
        out.append(ml.KBinsDiscretizer(n_bins=4, encode="ordinal").fit(X).transform(X))
        out.append(ml.SplineTransformer().fit(X).transform(X))
        out.append(ml.PolynomialFeatures(2).fit(X).transform(X))
        g = ml.GaussianNB().fit(X, y)
        out += [g.predict_proba(X), g.predict(X)]
        out.append(ml.MultinomialNB().fit(C, y).predict_proba(C))
        out.append(ml.CategoricalNB().fit(C, y).predict_proba(C))
        out.append(ml.LinearDiscriminantAnalysis().fit(X, y).predict_proba(X))
        out.append(ml.QuadraticDiscriminantAnalysis().fit(X, y).predict_proba(X))
        out.append(ml.SelectKBest(k=2).fit(X, y).scores_)
        out.append(ml.mutual_info_classif(X, y, random_state=0))
        return [np.asarray(v).tobytes() for v in out]

    first = run()
    second = run()
    assert len(first) == len(second)
    for i, (u, v) in enumerate(zip(first, second)):
        assert u == v, f"output {i} of the second pass differs from the first"
    print("TWICE OK", ml.vendor())
''')


def _run(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=900)


def _binding_present(name):
    try:
        from mojolearn import _backend
        _backend.binding(name, "identical")
        return True
    except Exception:
        return False


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_x_prep_run_many_times_twice_in_one_process(backend):
    if backend == "gpu":
        if not _binding_present("_mojolearn_x_prep"):
            pytest.skip("no GPU x_prep binding in this install")
        r = _run({})
    else:
        r = _run({"MOJOLEARN_VENDOR": "cpu"})
    if r.returncode != 0 and "No module named" in r.stderr and backend == "cpu":
        pytest.skip("no CPU host binding in this install")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert "TWICE OK" in r.stdout
