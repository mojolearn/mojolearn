# SPDX-License-Identifier: Apache-2.0
"""`digest_` means "this model gives these answers" (python/mojolearn/_digest_fields.py): the invariants at the
Python boundary, where save / load / export / pickle / the CPU twins live.

- equal after save + load, export + import and the CPU twin (`host_model`); equal after pickle, for two fits with the
  same seed and data and on either device is the generic-digest and generic-pickle verify lanes;
- changed by any changed fitted value, class label or answer-deciding hyperparameter, and a model whose answers
  differ has a different digest (random perturbation of every fitted array);
- unchanged by the answer methods, by the training record (`TRAINING`), by run settings and by the thread count;
- `training_digest_` is a separate value: it moves with the training record and never equals `digest_`.

The fits are small and independent; pytest-xdist splits the tests across cores when it is installed.
"""
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

mojolearn = pytest.importorskip("mojolearn")
from mojolearn import _digest_fields  # noqa: E402

pytestmark = pytest.mark.skipif(mojolearn.numeric_mode() != "identical", reason="digest_ exists in identical mode only")

RNG = np.random.default_rng(0)
X = RNG.random((160, 5), dtype=np.float32)
Y = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
C = np.digitize(Y, np.quantile(Y, [1 / 3, 2 / 3])).astype(np.int64)
XQ = RNG.random((23, 5), dtype=np.float32)
ANSWERS = ("predict", "predict_proba", "decision_function", "transform", "score_samples", "predict_log_proba")

#: (class, parameters, target): one per field source (classical layout, save schema, public fitted attributes,
#: schema with a training record, a trained network with optimizer state, a member list, parameters read at
#: predict). Target "r" regression, "c" classes, None unsupervised.
CASES = [
    ("Ridge", dict(alpha=1.0), "r"),
    ("LogisticRegression", {}, "c"),
    ("KMeans", dict(n_clusters=3, random_state=0), None),
    ("GaussianMixture", dict(n_components=2, random_state=0), None),
    ("PCA", dict(n_components=2), None),
    ("StandardScaler", {}, None),
    ("GaussianNB", {}, "c"),
    ("SGDRegressor", dict(random_state=0), "r"),
    ("MLPRegressor", dict(hidden_layer_sizes=(8,), max_iter=30, random_state=0), "r"),
    ("MLPClassifier", dict(hidden_layer_sizes=(8,), max_iter=30, random_state=0), "c"),
    ("KNeighborsClassifier", dict(n_neighbors=4), "c"),
    ("RandomForestRegressor", dict(n_estimators=4, random_state=0), "r"),
    ("BaggingRegressor", dict(n_estimators=3, random_state=0), "r"),
]


def _fit(name, params, target, data=X, labels=None, **more):
    est = getattr(mojolearn, name)(**params, **more)
    if target is None:
        return est.fit(data)
    return est.fit(data, (C if labels is None else labels) if target == "c" else Y)


def _answers(model, data=XQ):
    out = {}
    for method in ANSWERS:
        f = getattr(model, method, None)
        if f is None:
            continue
        try:
            value = np.asarray(f(data))
        except (AttributeError, NotImplementedError, ValueError, TypeError):
            continue
        out[method] = (value.dtype.str, value.shape, value.tobytes())
    return out


def _clone(model):
    try:
        return pickle.loads(pickle.dumps(model))
    except (TypeError, pickle.PicklingError, AttributeError, NotImplementedError) as refusal:
        pytest.skip(f"{type(model).__name__} does not pickle: {refusal}")


@pytest.mark.parametrize("name,params,target", CASES)
def test_answer_methods_and_rereads_leave_the_digest(name, params, target):
    model = _fit(name, params, target)
    first, training = model.digest_, model.training_digest_
    assert _answers(model)
    assert model.digest_ == first and model.training_digest_ == training
    if hasattr(model, "score") and target is not None:
        model.score(X, C if target == "c" else Y)
        assert model.digest_ == first


@pytest.mark.parametrize("name,params,target", CASES)
def test_the_training_digest_is_a_separate_value(name, params, target):
    model = _fit(name, params, target)
    assert len(model.training_digest_) == 64 and model.training_digest_ != model.digest_
    unfitted = getattr(mojolearn, name)(**params)
    assert not hasattr(unfitted, "training_digest_") and not hasattr(unfitted, "digest_")


@pytest.mark.parametrize("name,params,target", CASES)
def test_save_load_and_the_cpu_twin_keep_the_digest(name, params, target, tmp_path):
    model = _fit(name, params, target)
    path = str(tmp_path / (name + ".npz"))
    try:
        model.save(path)
    except (AttributeError, NotImplementedError) as refusal:
        pytest.skip(f"{name} has no save: {refusal}")
    loaded = type(model).load(path)
    assert loaded.digest_ == model.digest_
    twin = mojolearn.host_model(path)
    if hasattr(type(twin), "digest_"):
        assert twin.digest_ == model.digest_


@pytest.mark.parametrize("name,params,target", CASES)
def test_the_training_record_is_not_in_the_digest(name, params, target):
    # A copy whose training record is rewritten answers exactly as the model does: same digest_, and the
    # training_digest_ sees the change.
    model = _fit(name, params, target)
    present = [field for field in _digest_fields.TRAINING if vars(model).get(field) is not None]
    if not present:
        pytest.skip(f"{name} keeps no training record")
    edited = _clone(model)
    for field in present:
        value = vars(edited)[field]
        if isinstance(value, (bool, np.bool_)):
            vars(edited)[field] = not value
        elif isinstance(value, (int, float, np.number)):
            vars(edited)[field] = value + 7
        else:
            vars(edited)[field] = [987.0]
    assert _answers(edited) == _answers(model)
    assert edited.digest_ == model.digest_
    assert edited.training_digest_ != model.training_digest_


def _answer_arrays(model):
    """(name, array) of every fitted float array the digest walks at the top level."""
    skip = set(_digest_fields.TRAINING) | set(_digest_fields.NOT_STATE)
    found = []
    for name, value in vars(model).items():
        if name in skip or not name.endswith("_") or name.startswith("_"):
            continue
        array = np.asarray(value) if hasattr(value, "__array__") else None
        if array is not None and array.dtype.kind == "f" and array.size:
            found.append(name)
    return found


@pytest.mark.parametrize("name,params,target", CASES)
def test_a_model_whose_answers_differ_has_a_different_digest(name, params, target):
    # Random perturbation: one element of each fitted float array of a copy, one at a time. Whenever the copy's
    # answers differ from the model's, its digest must differ too.
    model = _fit(name, params, target)
    first, answers = model.digest_, _answers(model)
    rng = np.random.default_rng(1)
    moved = 0
    for field in _answer_arrays(model):
        edited = _clone(model)
        array = np.array(np.asarray(vars(edited)[field]), copy=True)
        flat = array.reshape(-1)
        flat[rng.integers(flat.size)] += np.asarray(1.0 + rng.random(), dtype=array.dtype)
        try:
            setattr(edited, field, array)
        except (AttributeError, TypeError):
            continue
        try:
            changed = _answers(edited) != answers
        except Exception:            # a perturbed model the answer methods refuse is not an answer
            continue
        if changed:
            moved += 1
            assert edited.digest_ != first, f"{name}.{field}: the answers moved and the digest did not"
    if not moved:
        pytest.skip(f"{name}: no single-element change of a fitted array moved its answers")


def test_a_changed_class_label_changes_the_digest():
    first = _fit("LogisticRegression", {}, "c")
    relabeled = _fit("LogisticRegression", {}, "c", labels=C + 10)
    assert _answers(first)["predict"] != _answers(relabeled)["predict"]
    assert first.digest_ != relabeled.digest_


#: (class, parameters, target, the parameter, another value): parameters an answer method reads at the call
#: (PARAMS_BY_NAME). Each pair fits the same fitted arrays and gives different answers.
ANSWER_PARAMETERS = [
    ("Binarizer", {}, None, "threshold", 0.5),
    ("AdditiveChi2Sampler", {}, None, "sample_steps", 3),
    ("IncrementalPCA", dict(n_components=2), None, "whiten", True),
    ("LocalOutlierFactor", dict(n_neighbors=5), None, "novelty", True),
    ("MLPRegressor", dict(hidden_layer_sizes=(8,), max_iter=30, random_state=0), "r", "activation", "tanh"),
    ("KNeighborsRegressor", dict(n_neighbors=4), "r", "n_neighbors", 7),
]


@pytest.mark.parametrize("name,params,target,parameter,value", ANSWER_PARAMETERS)
def test_an_answer_deciding_parameter_changes_the_digest(name, params, target, parameter, value):
    if getattr(mojolearn, name, None) is None:
        pytest.skip(f"no {name} in this build")
    a = _fit(name, params, target)
    b = _fit(name, {**params, parameter: value}, target)
    assert a.digest_ != b.digest_


#: Settings that change how a call runs, never its answers.
RUN_SETTINGS = [
    ("KNeighborsClassifier", dict(n_neighbors=4), "c", "query_tile", 1),
    ("SVC", {}, "c", "cache_size", 1000.0),
]


@pytest.mark.parametrize("name,params,target,parameter,value", RUN_SETTINGS)
def test_a_run_setting_leaves_the_digest(name, params, target, parameter, value):
    a = _fit(name, params, target)
    b = _fit(name, {**params, parameter: value}, target)
    assert _answers(a) == _answers(b)
    assert a.digest_ == b.digest_


def test_a_model_the_file_format_does_not_carry_still_has_a_digest(tmp_path):
    # String class labels: k-NN's save refuses them; the digest is the field walk's.
    labels = np.array(["a", "b", "c"], dtype=object)[C]
    model = mojolearn.KNeighborsClassifier(n_neighbors=4).fit(X, labels)
    with pytest.raises(ValueError):
        model.save(str(tmp_path / "knn.npz"))
    first = model.digest_
    assert mojolearn.KNeighborsClassifier(n_neighbors=4).fit(X, labels).digest_ == first
    assert mojolearn.KNeighborsClassifier(n_neighbors=5).fit(X, labels).digest_ != first
    other = np.array(["a", "b", "z"], dtype=object)[C]
    assert mojolearn.KNeighborsClassifier(n_neighbors=4).fit(X, other).digest_ != first


_THREADS_SCRIPT = r"""
import os, sys
allowed = os.sched_getaffinity(0)
os.sched_setaffinity(0, {min(allowed)} if sys.argv[1] == "one" else allowed)
import numpy as np, mojolearn
rng = np.random.default_rng(0)
X = rng.random((400, 6), dtype=np.float32)
y = X @ np.arange(1, 7, dtype=np.float32)
models = [mojolearn.Ridge(alpha=1.0).fit(X, y), mojolearn.KMeans(n_clusters=3, random_state=0).fit(X),
          mojolearn.RandomForestRegressor(n_estimators=4, random_state=0).fit(X, y)]
print(" ".join(m.digest_ for m in models))
"""


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="core affinity is Linux only")
def test_the_thread_count_does_not_change_the_digest():
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("one core: nothing to compare")
    done = [subprocess.run([sys.executable, "-c", _THREADS_SCRIPT, which], capture_output=True, text=True,
                           timeout=600) for which in ("one", "all")]
    for run in done:
        assert run.returncode == 0, run.stderr[-2000:]
    runs = [run.stdout.split() for run in done]
    assert runs[0] == runs[1] and len(runs[0]) == 3


# Composed networks: Sequential keeps its optimizer state and history (save_checkpoint keeps them, export drops
# them), so these are the cases where the training record and the answers part.

def _sequential(seed=0, epochs=2, eps=None, loss="mse"):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((32, 8)).astype(np.float32)
    target = rng.standard_normal((32, 1)).astype(np.float32) if loss == "mse" else (data[:, 0] > 0).astype(np.int64)
    model = mojolearn.Sequential(numeric_mode="identical").linear(16).tanh()
    model = (model.layer_norm(eps=eps) if eps is not None else model.layer_norm()).linear(8).relu()
    model = model.linear(1 if loss == "mse" else 2).compile(loss=loss, optimizer="adam", lr=0.01)
    return model.fit(data, target, epochs=epochs, batch_size=16, shuffle=False, seed=seed), data


@pytest.mark.parametrize("loss", ["mse", "cross_entropy"])
def test_sequential_export_checkpoint_and_pickle_keep_the_digest(loss, tmp_path):
    model, data = _sequential(loss=loss)
    first, training = model.digest_, model.training_digest_
    model.export(str(tmp_path / "m.export"))
    exported = mojolearn.Sequential.load(str(tmp_path / "m.export"))
    assert exported.digest_ == first
    assert exported.training_digest_ != training             # the export drops the optimizer state
    model.save_checkpoint(str(tmp_path / "m.ckpt"))
    resumed = mojolearn.Sequential.from_checkpoint(str(tmp_path / "m.ckpt"))
    assert resumed.digest_ == first
    assert resumed.training_digest_ == training
    assert np.array_equal(np.asarray(exported.predict(data)), np.asarray(model.predict(data)))
    model.predict(data)
    assert model.digest_ == first and model.training_digest_ == training


def test_sequential_float_layer_argument_survives_the_file(tmp_path):
    model, _ = _sequential(eps=1e-5)
    model.export(str(tmp_path / "m.export"))
    assert mojolearn.Sequential.load(str(tmp_path / "m.export")).digest_ == model.digest_


def test_sequential_more_training_changes_both_digests_and_the_same_fit_neither():
    a, _ = _sequential()
    b, _ = _sequential()
    assert (a.digest_, a.training_digest_) == (b.digest_, b.training_digest_)
    c, _ = _sequential(epochs=3)
    assert c.digest_ != a.digest_ and c.training_digest_ != a.training_digest_
    d, _ = _sequential(seed=1)
    assert d.training_digest_ != a.training_digest_


def test_sequential_loss_kind_is_in_the_digest():
    # The same layers and weights answer differently under a classification loss (labels and probabilities):
    # `_fit_loss` decides predict.
    assert "_fit_loss" in _digest_fields.EXTRA_BY_NAME["Sequential"]
    assert "_y_1d" in _digest_fields.EXTRA_BY_NAME["Sequential"]


def test_the_tables_are_disjoint_and_name_real_classes():
    assert not set(_digest_fields.TRAINING) & set(_digest_fields.NOT_STATE)
    for table in (_digest_fields.EXTRA, _digest_fields.TRAINING_EXTRA, _digest_fields.SCHEMAS):
        assert all(isinstance(cls, type) for cls in table)
    for name, fields in _digest_fields.TRAINING_EXTRA_BY_NAME.items():
        assert not set(fields) & set(_digest_fields.EXTRA_BY_NAME.get(name, ()))
