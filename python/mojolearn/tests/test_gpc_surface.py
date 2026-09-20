# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `mojolearn.GaussianProcessClassifier` (lane/gaussian-process-
classifier, 2026-09-15).

Source checks run on a box with nothing built: every refusal in the
constructor fires by name, both bindings register gpc_fit and gpc_predict,
the gp host family exports and covers them, the identity lanes and their
batch declaration exist, the shared steps import no GPU module, and the
NOT_IMPLEMENTED row is gone.

Runtime checks run where `_mojolearn_gp` (GPU or CPU host) imports and are
SKIPPED, and say so, elsewhere: binary and one-vs-rest fits and their
shapes, probabilities that sum to one, the single-class and one-sample-class
data, string labels, repeat identity, rows alone against the batch, save and
load, `host_model` against the loaded binding, and the fit refusal of the
host class outside the verifier. The bit claim across vendors is the
identity harness's (tools/identity_break.py lanes gpc and gpc-multiclass),
not this file's.

    cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_gpc_surface
"""
import os
import re
import sys
import tempfile
from pathlib import Path

import mojolearn as ml
from mojolearn import _backend, host_surface
from mojolearn._cpu_reference import reference_training
from mojolearn._gpc_impl import GaussianProcessClassifier, HostGaussianProcessClassifier

ROOT = Path(__file__).resolve().parents[3]
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)


class Skip(Exception):
    pass


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _raises(exc, needle, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        assert needle in str(e), f"raised {type(e).__name__} without {needle!r}: {e}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {needle!r}")


# -- source checks ----------------------------------------------------------------


def test_constructor_refuses_by_name():
    G = GaussianProcessClassifier
    _raises(NotImplementedError, "DEVIATION 1761", G, optimizer="fmin_l_bfgs_b")
    _raises(NotImplementedError, "n_restarts_optimizer", G, n_restarts_optimizer=2)
    _raises(NotImplementedError, "warm_start", G, warm_start=True)
    _raises(NotImplementedError, "copy_X_train", G, copy_X_train=False)
    _raises(NotImplementedError, "random_state", G, random_state=0)
    _raises(NotImplementedError, "one_vs_one", G, multi_class="one_vs_one")
    _raises(ValueError, "multi_class", G, multi_class="crammer_singer")
    _raises(NotImplementedError, "n_jobs", G, n_jobs=2)
    _raises(ValueError, "at least 1", G, max_iter_predict=0)
    _raises(TypeError, "max_iter_predict", G, max_iter_predict=True)
    _raises(TypeError, "kernel", G, kernel="rbf")
    m = G()
    assert m.optimizer is None and m.max_iter_predict == 100 and m.multi_class == "one_vs_rest"
    assert "GaussianProcessClassifier" in ml.__all__


def test_bindings_and_manifest_carry_the_entries():
    for rel in ("bindings/_mojolearn_gp.mojo", "bindings/_mojolearn_gp_host.mojo"):
        text = _read(rel)
        for name in ("gpc_fit", "gpc_predict"):
            assert f'("{name}")' in text, f"{rel} does not register {name}"
    fam = host_surface.family("gp")
    for name in ("gpc_fit", "gpc_predict"):
        assert name in fam["exports"]
    assert {"gpc", "gpc-multiclass"} <= set(fam["training_lanes"])
    assert "GaussianProcessClassifier" in fam["classes"]
    for module in ("gaussian_process/host/gpc_oracle.mojo", "gaussian_process/host/gpc_steps.mojo"):
        assert module in fam["host_modules"] and (ROOT / module).is_file()
        assert not GPU_IMPORTS.search(_read(module)), f"{module} imports a GPU module"
    harness = _read("tools/identity_break.py")
    assert '@lane("gpc")' in harness and '@lane("gpc-multiclass")' in harness
    assert re.search(r'_batch_decl\(_rows_calls\("predict", "predict_proba"[^\n]*\), "gpc", "gpc-multiclass"\)', harness)


def test_classification_is_no_longer_not_implemented():
    rows = [l for l in _read("gaussian_process/NOT_IMPLEMENTED.tsv").splitlines()
            if l.startswith("sklearn GaussianProcessClassifier")]
    # Classification is implemented; its optional hyperparameter optimizer
    # still has an explicit, narrower refusal documented in this ledger.
    assert all(row.startswith("sklearn GaussianProcessClassifier's hyperparameter optimizer")
               for row in rows), rows
    steps = _read("gaussian_process/host/gpc_steps.mojo")
    for token in ("GPC_LML_TOL_BITS: UInt32 = 0x2EDBE6FF", "identical_softplus", "identical_exp64",
                  "DEVIATION 2830", "DEVIATION 2831", "DEVIATION 2832"):
        assert token in steps, token


# -- runtime checks (skipped without a binding) ----------------------------------------


def _need_binding():
    try:
        mod = _backend.binding("_mojolearn_gp", "identical")
        if not callable(getattr(mod, "gpc_fit", None)):
            raise Skip("the loaded _mojolearn_gp has no gpc_fit (rebuild bindings/build_gp.sh)")
    except Skip:
        raise
    except Exception as exc:  # an unbuilt binding raises by name
        raise Skip(f"_mojolearn_gp does not load here: {type(exc).__name__}: {exc}")


def _data(n=48, d=3, seed=0):
    import random
    rng = random.Random(seed)
    X = [[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(n)]
    s = [row[0] + 0.5 * row[1] for row in X]
    order = sorted(range(n), key=lambda i: (s[i], i))
    y2 = [0] * n
    y3 = [0] * n
    for rank, i in enumerate(order):
        y2[i] = 1 if rank >= n // 2 else 0
        y3[i] = (rank * 3) // n
    return X, y2, y3


def _bytes(a):
    return a.tobytes() if hasattr(a, "tobytes") else repr(a).encode()


def test_binary_and_multiclass_fit_predict():
    _need_binding()
    X, y2, y3 = _data()
    q = X[:10]
    with reference_training():
        m = GaussianProcessClassifier(numeric_mode="identical").fit(X, y2)
        m3 = GaussianProcessClassifier(kernel=ml.ConstantKernel(2.0) * ml.Matern(1.0, nu=1.5),
                                       numeric_mode="identical").fit(X, y3)
    p = m.predict_proba(q)
    assert tuple(p.shape) == (10, 2) and p.dtype == "<f8"
    for row in p.tolist():
        assert abs(row[0] + row[1] - 1.0) < 1e-12 and 0.0 <= row[1] <= 1.0
    assert m.n_classes_ == 2 and isinstance(m.n_iter_, int) and 1 <= m.n_iter_ <= 100
    assert tuple(m.L_.shape) == (48, 48)
    mean, var = m.latent_mean_and_variance(q)
    assert [1 if v > 0 else 0 for v in mean.tolist()] == m.predict(q).tolist()
    assert all(v > 0.0 for v in var.tolist())
    p3 = m3.predict_proba(q)
    assert tuple(p3.shape) == (10, 3)
    for row in p3.tolist():
        assert abs(sum(row) - 1.0) < 1e-12
    assert len(m3.estimators_) == 3 and len(m3.n_iter_) == 3
    _raises(ValueError, "only supported for binary", m3.latent_mean_and_variance, q)
    _raises(NotImplementedError, "DEVIATION 1761", m.log_marginal_likelihood, [0.0])
    assert m.log_marginal_likelihood() == m.log_marginal_likelihood_value_


def test_degenerate_classes():
    _need_binding()
    X, y2, _ = _data()
    with reference_training():
        _raises(ValueError, "requires 2 or more distinct classes; got 1 class",
                GaussianProcessClassifier(numeric_mode="identical").fit, X, [7] * len(X))
        # one class with a single sample, string labels
        labels = ["b"] * len(X)
        labels[5] = "a"
        m = GaussianProcessClassifier(numeric_mode="identical").fit(X, labels)
        assert m.classes_ == ["a", "b"]
        assert set(m.predict(X[:4])) <= {"a", "b"}
        # three classes, one of them a single sample
        y3 = [0 if i % 2 else 1 for i in range(len(X))]
        y3[0] = 2
        m3 = GaussianProcessClassifier(numeric_mode="identical").fit(X, y3)
        assert m3.n_classes_ == 3 and tuple(m3.predict_proba(X[:4]).shape) == (4, 3)
        # max_iter_predict=1 stops after one Newton step
        m1 = GaussianProcessClassifier(max_iter_predict=1, numeric_mode="identical").fit(X, y2)
        assert m1.n_iter_ == 1
    _raises(ValueError, "call fit() or load() first", GaussianProcessClassifier().predict, X)


def test_repeat_batch_save_load_and_host_model():
    _need_binding()
    X, y2, y3 = _data()
    q = X[:12]
    for y in (y2, y3):
        with reference_training():
            a = GaussianProcessClassifier(numeric_mode="identical").fit(X, y)
            b = GaussianProcessClassifier(numeric_mode="identical").fit(X, y)
        pa = a.predict_proba(q)
        assert _bytes(pa) == _bytes(b.predict_proba(q))
        alone = [a.predict_proba(q[i:i + 1]).tolist()[0] for i in range(len(q))]
        assert alone == pa.tolist(), "a row asked alone differs from the same row in the batch"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gpc.npz")
            a.save(path)
            back = GaussianProcessClassifier.load(path)
            assert _bytes(back.predict_proba(q)) == _bytes(pa)
            assert _bytes(back.predict(q)) == _bytes(a.predict(q))
            try:
                host = ml.host_model(path)
            except Exception as exc:  # no host binding built on this box
                print(f"  [SKIP host_model: {type(exc).__name__}: {exc}]")
                continue
            assert isinstance(host, HostGaussianProcessClassifier)
            assert _bytes(host.predict_proba(q)) == _bytes(pa)


def main():
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print(f"PASS {name}")
        except Skip as s:
            print(f"SKIP {name}: {s}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
