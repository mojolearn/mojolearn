# SPDX-License-Identifier: Apache-2.0
"""`digest_` at the boundary (lanes digest rounds 1 to 3, fx-audit session 3).

Reuses every test of tests/api/test_digest.py (same fit = same digest, changed data changes it, load restores it,
CPU device = GPU digest for the linear cases, ...) and adds the RUN OWED items it does not cover: the GPU digest
equals the CPU digest for every case family, predict / transform leave the digest unchanged, the loadable classes
of the public rule and the HostForest / HostGBDT twins restore the saving model's digest.
"""
import numpy as np
import pytest

from _proving import assert_same, make, reexport

_digest = reexport(globals(), "tests/api/test_digest.py", "proving_reused_test_digest")
mojolearn = _digest.mojolearn
X, Y, C = _digest.X, _digest.Y, _digest.C
pytestmark = pytest.mark.skipif(not _digest._identical(), reason="digest_ exists in identical mode only")


def _fit(name, params, target, device):
    model = make(name, device, **params)
    return model.fit(X) if target is None else model.fit(X, Y if target == "r" else C)


@pytest.mark.gpu
@pytest.mark.parametrize("name,params,target", _digest.CASES)
def test_proving_gpu_digest_equals_cpu_digest_for_every_case(name, params, target):
    assert _fit(name, params, target, "gpu").digest_ == _fit(name, params, target, "cpu").digest_


def _reads(model):
    for word in ("predict", "transform", "predict_proba", "score_samples"):
        if hasattr(model, word):
            yield word, getattr(model, word)


@pytest.mark.parametrize("name,params,target", _digest.CASES)
def test_proving_predict_and_transform_leave_the_digest(name, params, target):
    # digest round 3 item 8b: a read does not change the fitted state the digest hashes.
    model = _fit(name, params, target, "cpu")
    first = model.digest_
    for word, read in _reads(model):
        read(X[:17])
        assert model.digest_ == first, f"{name}.{word} changed the digest"


#: digest round 3: the loadable classes hashed by the public rule; `load` restores exactly the hashed fields.
LOADABLE = [
    ("KNeighborsClassifier", {}, "c"),
    ("KNeighborsRegressor", {}, "r"),
    ("NearestNeighbors", {}, None),
    ("KernelRidge", {}, "r"),
    ("Nystroem", dict(n_components=20, random_state=0), None),
    ("RBFSampler", dict(n_components=20, random_state=0), None),
    ("LinearRegression", {}, "r"),
    ("Ridge", {}, "r"),
    ("StandardScaler", {}, None),
    ("MinMaxScaler", {}, None),
]


@pytest.mark.parametrize("name,params,target", LOADABLE)
def test_proving_loaded_model_has_the_fitted_digest(name, params, target, tmp_path):
    model = _fit(name, params, target, "cpu")
    path = str(tmp_path / f"{name}.npz")
    model.save(path)
    assert type(model).load(path).digest_ == model.digest_


@pytest.mark.parametrize("name,params,target", [
    ("RandomForestRegressor", dict(n_estimators=5, random_state=0), "r"),
    ("ExtraTreesRegressor", dict(n_estimators=5, random_state=0), "r"),
    # HostGBDT loads a `GradientBoosting.save` file (`host_model`'s docstring). The scikit-learn adapters
    # GradientBoostingRegressor / Classifier refuse save by design (python/mojolearn/_gbdt_adapters.py `save`,
    # commit 4a4abf3f01: no adapter archive format, pickle keeps the adapter state).
    ("GradientBoosting", dict(n_estimators=5, max_depth=3, random_state=0), "r"),
])
def test_proving_host_twin_has_the_saving_models_digest(name, params, target, tmp_path):
    # fx-audit session 3: HostForest (rf, et) and HostGBDT digest_ == the saving model's digest_, and the twin
    # predicts as the model.
    model = _fit(name, params, target, "cpu")
    path = str(tmp_path / f"{name}.npz")
    model.save(path)
    twin = mojolearn.host_model(path)
    assert twin.digest_ == model.digest_
    assert_same(np.asarray(twin.predict(X[:17])), np.asarray(model.predict(X[:17])), f"{name} twin predict")


@pytest.mark.gpu
def test_proving_reinit_to_cpu_then_predict_runs_and_refit_keeps_the_digest():
    # fx-audit session 3: model.__init__(device="cpu") on a fitted GPU estimator, then predict runs (CPU route,
    # or a refusal in words, never a crash); a refit on the CPU route gives the GPU fit's digest.
    model = _fit("Ridge", dict(alpha=1.0), "r", "gpu")
    gpu_digest = model.digest_
    gpu_predict = np.asarray(model.predict(X[:17]))
    model.__init__(alpha=1.0, device="cpu")
    try:
        after = np.asarray(model.predict(X[:17]))
    except (AttributeError, ValueError, RuntimeError) as refusal:   # not fitted any more: words, not a crash
        assert str(refusal)
    else:
        assert_same(after, gpu_predict, "predict after __init__(device='cpu')")
    assert model.fit(X, Y).digest_ == gpu_digest
