# SPDX-License-Identifier: Apache-2.0
"""Model selection and meta-estimators over members (lanes ma-core, ma-callers).

Every case of _arm_cases.py (cross_val_score / cross_validate / cross_val_predict / GridSearchCV values, named
scorers, error_score, Bagging / Voting / Stacking / MultiOutput / OneVsRest / Calibrated / RFE / explainers over
members, the NaN refusal words) runs in subprocesses with this interpreter's wheel and its results are checked.
The cases are split across every core (one subprocess per shard; MOJOLEARN_PROVING_WORKERS caps it); under
pytest-xdist the workers share one run through a lock file.
"""
import fcntl
import os
import subprocess
import sys

import numpy as np
import pytest

from _proving import DEVICES, HERE, assert_same, cls, make, member, workers_for

import _arm_cases

CASES = sorted(_arm_cases.CASES)
# TOMBSTONE: MOJOLEARN_MSEL_MEMBER_UNITS_OFF and MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_MSEL_MEMBER_UNITS_OFF.patch (and MOJOLEARN_MEMBER_SET_OFF.patch).
SCRIPT = os.path.join(HERE, "_arm_cases.py")


@pytest.fixture(scope="session")
def arm_root(tmp_path_factory):
    base = tmp_path_factory.getbasetemp()
    root = (base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base) / "proving-arms"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_arms(root, device):
    """Run every case for `device` once per pytest run; returns {"default": outdir}."""
    done = root / f"{device}.done"
    with open(root / f"{device}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        labels = ["default"]
        outs = {label: root / device / label for label in labels}
        if not done.exists():
            shards = max(1, workers_for(len(CASES) * len(labels)) // len(labels))
            running = []
            env = dict(os.environ, MOJOLEARN_PROVING_DEVICE=device)
            for label in labels:
                outs[label].mkdir(parents=True, exist_ok=True)
                for shard in range(shards):
                    chunk = CASES[shard::shards]
                    if chunk:
                        log = open(outs[label] / f"shard{shard}.log", "w")
                        running.append((label, log, subprocess.Popen(
                            [sys.executable, SCRIPT, str(outs[label]), *chunk], env=env, cwd=str(outs[label]),
                            stdout=log, stderr=subprocess.STDOUT)))
            for label, log, process in running:
                process.wait(timeout=3600)
                log.close()
                (outs[label] / f"rc-{process.pid}").write_text(str(process.returncode))
            done.write_text("done")
    return outs


def _load(outdir, name):
    path = outdir / f"{name}.npz"
    assert path.exists(), f"{path} missing: the case process died (see {outdir}/shard*.log)"
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", CASES)
def test_case_runs_on_the_default_arm(arm_root, device, name):
    result = _load(_run_arms(arm_root, device)["default"], name)
    assert "__error__" not in result, str(result.get("__error__"))
    if name == "cvs_error_score":
        assert np.all(result["scores"] == 0.5), result["scores"]
    if name == "cvs_linear_regression_n_jobs":
        assert_same(result["n_jobs_-1"], result["n_jobs_1"], "n_jobs=-1 vs 1")
    if name == "voting_stacking_multioutput_ovr":
        for n in (-1, 1, 2):
            assert_same(result[f"voting_n_jobs_{n}"], result["voting_n_jobs_None"], f"voting n_jobs={n}")
    if name == "nan_refusals":
        for key, words in result.items():
            assert str(words) != "no error", f"{name}.{key}: the invalid input was admitted"
    if name == "missing_class_folds":
        # Both admit a fold whose training rows lack a class, as scikit-learn does: Stacking places the fold's
        # outputs by the member's classes (lane bm-meta-3, 2026-11-23: `_enforce_prediction_order`; it refused
        # before), CalibratedClassifierCV maps each fold's scores by the member's class codes (lane ma-callers R2
        # unit 4, `_cal_scores`).
        assert str(result["stacking"]) == "no error", f"{name}.stacking: {result['stacking']}"
        assert str(result["calibrated"]) == "no error", f"{name}.calibrated: {result['calibrated']}"
    if name == "grid_ridge_and_logistic":
        assert float(result["ridge_alpha"]) in (0.1, 1.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("build", [
    lambda d: cls("VotingClassifier")([("a", make("LogisticRegression", d))], n_jobs=0),
    lambda d: cls("StackingClassifier")([("a", make("LogisticRegression", d))], make("LogisticRegression", d),
                                        n_jobs=0),
    lambda d: cls("MultiOutputClassifier")(make("LogisticRegression", d), n_jobs=0),
    lambda d: cls("OneVsRestClassifier")(make("LogisticRegression", d), n_jobs=0),
    lambda d: make("BaggingClassifier", d, estimator=member("DecisionTreeClassifier"), n_jobs=0),
], ids=["voting", "stacking", "multioutput", "ovr", "bagging"])
def test_n_jobs_zero_is_refused(device, build):
    # ma-callers round 2: n_jobs=0 -> ValueError.
    X, C = _arm_cases.X, _arm_cases.C
    y = np.stack([C, C], axis=1) if "MultiOutput" in type(build(device)).__name__ else C
    with pytest.raises(ValueError):
        build(device).fit(X, y)


@pytest.mark.parametrize("device", DEVICES)
def test_failing_candidate_raises_the_estimators_own_words(device):
    # ma-core rounds 1-2: a failing candidate (alpha < 0) raises the estimator's own exception type and words.
    X, Y = _arm_cases.X, _arm_cases.Y
    with pytest.raises(Exception) as direct:
        make("Ridge", device, alpha=-1.0).fit(X, Y)
    with pytest.raises(type(direct.value)) as through:
        cls("cross_val_score")(make("Ridge", device, alpha=-1.0), X, Y)
    assert str(direct.value) in str(through.value)


def _import_every_family():
    import mojolearn

    for name in dir(mojolearn):
        try:
            getattr(mojolearn, name)
        except Exception:
            pass
    from mojolearn import model_selection

    return model_selection._NATIVE_CONTEXT     # python/mojolearn/_families.py connects the CV dispatch into it


#: The marked classes whose constructor is not a field-only schema, so no member-only plan can clone them
#: (python/mojolearn/_native_msel_more.py `_derive_member`): their Python __init__ calls a native constructor
#: operation (the Theta / ETS forecasters and CNNClassifier through `_XSEQ` / `_XCNN`). Every other marked class is
#: registered; GradientBoosting (16, 1) is registered from its constructor's `_INIT_NAMES` and `trees_gbdt_init`
#: (python/mojolearn/_native_msel_more.py `_TEXT_MODEL_MEMBERS`, lane w3-fix-a: RFE over the booster). The ma-core claim (rounds 5-6: "CNNClassifier and DBSCAN
#: only") was never run; the 2026-11-22 proving read listed 48 classes, because `_derive_member` refused every
#: native (`_native_full_reset_params` / `_native_declared_constructors`) constructor.
UNREGISTERED = ["AutoTheta", "CNNClassifier", "DampedETS", "DynamicOptimizedTheta", "DynamicTheta", "ETS",
                "OptimizedTheta", "Theta"]


def test_every_family_is_registered_for_the_member_route():
    context = _import_every_family()
    if "family_msel_member_unregistered" not in context:
        pytest.fail("the model-selection context exposes no 'family_msel_member_unregistered' entry")
    assert sorted(c.__name__ for c in context["family_msel_member_unregistered"]) == UNREGISTERED


#: Predictors whose member serves the meta-estimators once registered (Voting, one-vs-rest, multi-output fit every
#: member through bindings/member_set_door.mojo and predict through bindings/trees_public_meta_door.mojo): each
#: meta-estimator's answer is the member's own fit + predict. Before the registration every one of them was refused
#: ("Sequential requires native input types"). GaussianNB and LinearDiscriminantAnalysis (hand-written schemas)
#: ride the same doors.
MEMBER_PREDICTORS = ["NearestCentroid", "BernoulliNB", "MultinomialNB", "ComplementNB",
                     "QuadraticDiscriminantAnalysis", "GaussianNB", "LinearDiscriminantAnalysis", "SVGP"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", MEMBER_PREDICTORS)
def test_member_route_serves_the_marked_predictor(device, name):
    _import_every_family()
    X, C, Y = _arm_cases.X, _arm_cases.C, _arm_cases.Y
    classifier = getattr(cls(name), "_estimator_type", None) == "classifier"
    y = C if classifier else Y
    own = np.asarray(make(name, device).fit(X, y).predict(X)).astype(np.float64)
    if classifier:
        answers = {"voting": cls("VotingClassifier")([("a", make(name, device)), ("b", make(name, device))]).fit(X, C)}
        both = cls("MultiOutputClassifier")(make(name, device)).fit(X, np.stack([C, C], axis=1))
    else:
        answers = {"voting": cls("VotingRegressor")([("a", make(name, device)), ("b", make(name, device))]).fit(X, Y)}
        both = cls("MultiOutputRegressor")(make(name, device)).fit(X, np.stack([Y, Y], axis=1))
    got = {key: np.asarray(model.predict(X)).astype(np.float64) for key, model in answers.items()}
    columns = np.asarray(both.predict(X)).astype(np.float64)
    got["multioutput_0"], got["multioutput_1"] = columns[:, 0], columns[:, 1]
    for key, value in got.items():
        assert np.array_equal(value, own), f"{name} {key} over its member differs from its own predict"
    if classifier:
        # One-vs-rest over two classes fits ONE member on the labels (scikit-learn's OneVsRestClassifier); over
        # more it fits one binary member per class, which is not the member's own multiclass model.
        two = (C > 0).astype(C.dtype)
        own_two = np.asarray(make(name, device).fit(X, two).predict(X)).astype(np.float64)
        ovr = np.asarray(cls("OneVsRestClassifier")(make(name, device)).fit(X, two).predict(X)).astype(np.float64)
        assert np.array_equal(ovr, own_two), f"{name} one-vs-rest over its member differs from its own predict"


@pytest.mark.parametrize("device", DEVICES)
def test_member_only_class_is_refused_by_cross_validation_in_its_words(device):
    # A member-only plan (a registered mark without method recipes) its member cannot serve keeps the words it had
    # before its mark, also under cross_validate's default error_score=nan (bindings/msel_member_units_door.mojo
    # `_admission_refusal`, lane mf-r9b; it was the deleted units' check): the 2026-11-22 registration first turned
    # the refusal into silent nan scores.
    _import_every_family()
    X, Y = _arm_cases.X, _arm_cases.Y
    with pytest.raises(RuntimeError, match="pending migration"):
        cls("cross_validate")(make("Binarizer", device), X, Y, cv=3, scoring="r2")
