# SPDX-License-Identifier: Apache-2.0
"""Python boundary tests of `model.digest_` (python/mojolearn/_mode.py `CheckMixin.digest_`,
bindings/fitted_digest_door.mojo).

What the attribute promises at the boundary: its form, when it exists, when it changes, and that nothing of it is
stored on the model. The hash walk itself is tested natively (checks/native_tests/fitted_digest_suite.mojo). The
fits are small and independent; pytest-xdist splits the tests across cores when it is installed.
"""
import numpy as np
import pytest

mojolearn = pytest.importorskip("mojolearn")

RNG = np.random.default_rng(0)
X = RNG.random((400, 6), dtype=np.float32)
Y = X @ np.arange(1, 7, dtype=np.float32)
C = (Y > np.median(Y)).astype(np.int32)
X_OTHER = X * np.float32(1.5) + np.float32(0.25)

#: (class name, parameters, target): one per attribute owner and field source. Ridge: the classical layout;
#: LogisticRegression, KMeans, PCA, RandomForestRegressor: a save schema; GaussianNB, BayesianRidge: the public
#: fitted attributes read through a public attribute plan or the generic slot.
CASES = [
    ("Ridge", dict(alpha=1.0), "r"),
    ("LinearRegression", {}, "r"),
    ("LogisticRegression", {}, "c"),
    ("KMeans", dict(n_clusters=3, random_state=0), None),
    ("PCA", dict(n_components=2), None),
    ("StandardScaler", {}, None),
    ("RandomForestRegressor", dict(n_estimators=5, random_state=0), "r"),
    ("GaussianNB", {}, "c"),
    ("BayesianRidge", {}, "r"),
]


def _fit(name, params, target, data=X, **more):
    est = getattr(mojolearn, name)(**params, **more)
    return est.fit(data) if target is None else est.fit(data, Y if target == "r" else C)


def _identical():
    return mojolearn.numeric_mode() == "identical"


pytestmark = pytest.mark.skipif(not _identical(), reason="digest_ exists in identical mode only")


@pytest.mark.parametrize("name,params,target", CASES)
def test_digest_is_one_lower_case_hex_string(name, params, target):
    digest = _fit(name, params, target).digest_
    assert type(digest) is str and len(digest) == 64
    assert digest == digest.lower() and set(digest) <= set("0123456789abcdef")


@pytest.mark.parametrize("name,params,target", CASES)
def test_equal_fits_have_equal_digests_and_changed_data_changes_it(name, params, target):
    first = _fit(name, params, target).digest_
    assert _fit(name, params, target).digest_ == first
    assert _fit(name, params, target, X_OTHER).digest_ != first


@pytest.mark.parametrize("name,params,target", CASES)
def test_unfitted_model_has_no_digest(name, params, target):
    est = getattr(mojolearn, name)(**params)
    assert not hasattr(est, "digest_")
    with pytest.raises(AttributeError):
        est.digest_


@pytest.mark.parametrize("name,params,target", CASES)
def test_digest_is_hashed_at_the_read_and_never_stored_on_the_model(name, params, target):
    model = _fit(name, params, target)
    assert "digest_" not in vars(model)             # nothing was hashed or stored by fit
    before = dict(vars(model))
    digest = model.digest_
    assert model.digest_ == digest                  # the unchanged model hashes to the same string again
    assert "digest_" not in vars(model) and vars(model).keys() == before.keys()
    assert "digest_" not in model.get_params()      # not a constructor parameter


def test_an_in_place_write_into_a_fitted_array_changes_the_next_read():
    # Nothing is kept between reads: a write into the elements of a stored fitted array (which changes no
    # attribute of the model) is in the next digest.
    model = _fit("Ridge", dict(alpha=1.0), "r")
    first = model.digest_
    stored = model.coef_
    if not (isinstance(stored, np.ndarray) and stored.flags.writeable and np.shares_memory(stored, model.coef_)):
        pytest.skip("coef_ is not handed out as the stored writeable array on this build")
    stored[0] += np.float32(1.0)
    second = model.digest_
    assert second != first
    stored[0] -= np.float32(1.0)
    assert model.digest_ == first


@pytest.mark.parametrize("name,params,target", CASES)
def test_refit_changes_the_digest(name, params, target):
    model = _fit(name, params, target)
    first = model.digest_
    model.fit(X_OTHER) if target is None else model.fit(X_OTHER, Y if target == "r" else C)
    second = model.digest_
    assert second != first
    assert second == _fit(name, params, target, X_OTHER).digest_


def test_set_params_drops_the_fitted_state_and_the_digest():
    model = _fit("Ridge", dict(alpha=1.0), "r")
    first = model.digest_
    model.set_params(alpha=2.0)
    assert not hasattr(model, "digest_")
    assert model.fit(X, Y).digest_ != first


def test_a_replaced_fitted_attribute_changes_the_digest():
    model = _fit("BayesianRidge", {}, "r")
    first = model.digest_
    model.coef_ = np.zeros_like(np.asarray(model.coef_))
    assert model.digest_ != first


def test_the_class_name_is_part_of_the_digest():
    assert _fit("Ridge", dict(alpha=1.0), "r").digest_ != _fit("BayesianRidge", {}, "r").digest_


@pytest.mark.parametrize("name,params,target", [case for case in CASES if case[0] in ("Ridge", "LinearRegression")])
def test_cpu_device_gives_the_gpu_digest(name, params, target):
    if mojolearn.vendor() == "cpu":
        pytest.skip("a CPU-only installation has one device")
    assert _fit(name, params, target, device="cpu").digest_ == _fit(name, params, target).digest_


def test_meta_estimator_digest_includes_its_members():
    voting = getattr(mojolearn, "VotingRegressor", None)
    if voting is None:
        pytest.skip("no VotingRegressor in this build")

    def build(alpha):
        return voting(estimators=[("a", mojolearn.Ridge(alpha=alpha)), ("b", mojolearn.LinearRegression())]).fit(X, Y)

    try:
        first = build(1.0).digest_
    except AttributeError as refusal:                # a family without a digest yet says so in words
        assert "digest_" in str(refusal)
        pytest.skip(str(refusal))
    assert build(1.0).digest_ == first
    assert build(50.0).digest_ != first              # a member's fitted state changed


#: Classes whose answer-deciding state is (partly) in private fields, layer objects or one member forest
#: (python/mojolearn/_digest_fields.py EXTRA / LAYERS): the digest must see it.
PRIVATE_STATE = [
    ("KNNImputer", dict(n_neighbors=3), None),            # all of its state is private
    ("LocalOutlierFactor", dict(n_neighbors=5), None),
    ("GaussianRandomProjection", dict(n_components=3, random_state=0), None),
    ("PowerTransformer", {}, None),
    ("GradientBoostingRegressor", dict(n_estimators=5, random_state=0), "r"),   # the inner learner
    ("BaggingRegressor", dict(n_estimators=3, random_state=0), "r"),            # one forest or a member list
    ("LinearSVR", {}, "r"),
]


@pytest.mark.parametrize("name,params,target", PRIVATE_STATE)
def test_private_fitted_state_is_in_the_digest(name, params, target):
    if getattr(mojolearn, name, None) is None:
        pytest.skip(f"no {name} in this build")
    first = _fit(name, params, target).digest_
    assert _fit(name, params, target).digest_ == first
    assert _fit(name, params, target, X_OTHER).digest_ != first


def test_how_the_input_arrived_is_not_in_the_digest():
    # `input_copied_` records whether fit copied its input; the fitted model is the same either way.
    from_array = mojolearn.Ridge(alpha=1.0).fit(X, Y).digest_
    from_fortran = mojolearn.Ridge(alpha=1.0).fit(np.asfortranarray(X), Y).digest_
    assert from_array == from_fortran


def test_fast_mode_has_no_digest():
    if mojolearn.vendor() != "metal":
        pytest.skip("fast is Apple only")
    try:
        model = mojolearn.Ridge(alpha=1.0, numeric_mode="fast").fit(X, Y)
    except Exception as unavailable:                  # no fast binaries in this installation
        pytest.skip(str(unavailable))
    with pytest.raises(AttributeError, match="identical mode only"):
        model.digest_


def test_receipt_and_reproduce_are_gone():
    assert not hasattr(mojolearn, "reproduce")
    assert "reproduce" not in mojolearn.__all__
    assert not hasattr(mojolearn.Ridge, "receipt")
    import importlib.util
    assert importlib.util.find_spec("mojolearn._reproduce") is None


def test_field_tables_name_real_classes():
    from mojolearn import _digest_fields
    for table in (_digest_fields.EXTRA, _digest_fields.REFUSED):
        assert all(isinstance(cls, type) for cls in table)
    assert all(type(reason) is str and reason for reason in _digest_fields.REFUSED.values())
