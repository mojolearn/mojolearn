# SPDX-License-Identifier: Apache-2.0
"""Model selection and meta-estimators over members, against their `_OFF` build arms (lanes ma-core, ma-callers).

Every case of _arm_cases.py (cross_val_score / cross_validate / cross_val_predict / GridSearchCV values, named
scorers, error_score, Bagging / Voting / Stacking / MultiOutput / OneVsRest / Calibrated / RFE / explainers over
members, the NaN refusal words) runs in subprocesses, once with this interpreter's wheel and once with each arm
wheel named by an environment variable:

    MOJOLEARN_PROVING_ARM_MSEL_MEMBER_UNITS_OFF=/path/to/python   (wheel built with -D MOJOLEARN_MSEL_MEMBER_UNITS_OFF)
    MOJOLEARN_PROVING_ARM_MEMBER_SET_OFF=/path/to/python          (wheel built with -D MOJOLEARN_MEMBER_SET_OFF)

The default arm's results are checked on their own always; an arm comparison skips when its variable is unset.
The cases are split across every core (one subprocess per shard, all arms at once; MOJOLEARN_PROVING_WORKERS caps
it); under pytest-xdist the workers share one run through a lock file.
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
ARM_DEFINES = ["MSEL_MEMBER_UNITS_OFF", "MEMBER_SET_OFF"]
ARMS = [pytest.param(define, id=define) for define in ARM_DEFINES]
SCRIPT = os.path.join(HERE, "_arm_cases.py")


def _arm_python(label):
    return sys.executable if label == "default" else os.environ.get(f"MOJOLEARN_PROVING_ARM_{label}")


@pytest.fixture(scope="session")
def arm_root(tmp_path_factory):
    base = tmp_path_factory.getbasetemp()
    root = (base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base) / "proving-arms"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_arms(root, device):
    """Run every case on every configured arm for `device` once per pytest run; returns {label: outdir}."""
    done = root / f"{device}.done"
    with open(root / f"{device}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        labels = ["default"] + [d for d in ARM_DEFINES if _arm_python(d)]
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
                            [_arm_python(label), SCRIPT, str(outs[label]), *chunk], env=env, cwd=str(outs[label]),
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
        # Stacking refuses a fold whose training rows lack a class, in words. CalibratedClassifierCV admits it: each
        # fold's member is fitted on the classes present and its scores are mapped by the member's class codes (lane
        # ma-callers R2 unit 4, `_cal_scores`), as scikit-learn's CalibratedClassifierCV does.
        assert str(result["stacking"]) != "no error", f"{name}.stacking: the invalid input was admitted"
        assert str(result["calibrated"]) == "no error", f"{name}.calibrated: {result['calibrated']}"
    if name == "grid_ridge_and_logistic":
        assert float(result["ridge_alpha"]) in (0.1, 1.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("name", CASES)
def test_case_equals_the_off_arm(arm_root, device, arm, name):
    if not _arm_python(arm):
        pytest.skip(f"MOJOLEARN_PROVING_ARM_{arm} is not set (no {arm} arm wheel)")
    outs = _run_arms(arm_root, device)
    default, off = _load(outs["default"], name), _load(outs[arm], name)
    assert sorted(default) == sorted(off), f"{name}: keys differ"
    for key in default:
        assert_same(off[key], default[key], f"{name}.{key} ({arm} arm vs default)")


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


def test_every_family_is_registered_for_the_member_route():
    # ma-core rounds 5-6: after importing every family the unregistered list is CNNClassifier and DBSCAN only.
    import mojolearn

    for name in dir(mojolearn):
        try:
            getattr(mojolearn, name)
        except Exception:
            pass
    from mojolearn import model_selection

    context = model_selection._NATIVE_CONTEXT     # python/mojolearn/_families.py connects the CV dispatch into it
    if "family_msel_member_unregistered" not in context:
        pytest.fail("the model-selection context exposes no 'family_msel_member_unregistered' entry")
    assert sorted(c.__name__ for c in context["family_msel_member_unregistered"]) == ["CNNClassifier", "DBSCAN"]
