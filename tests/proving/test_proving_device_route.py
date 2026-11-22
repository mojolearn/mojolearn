# SPDX-License-Identifier: Apache-2.0
"""device="cpu" beside a GPU set, and the CPU host images (lanes fx-host-import, fx-audit session 3, label-steps,
fx-device-attr).

On a GPU install, `import mojolearn` loads no CPU-device host image (`_mojolearn_*_host`) until a device="cpu"
model needs one; a device="cpu" fit then runs, and a pickled device="cpu" model runs in a fresh process. On every
install the classifier families fit, predict and score on device="cpu". The device-assignment tests of
tests/api/test_device_assignment.py are reused here.
"""
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from _proving import assert_same, binary, data, make, reexport

reexport(globals(), "tests/api/test_device_assignment.py", "proving_reused_test_device_assignment")

X, Y, C = data()
X_NEW = data(rows=40, seed=7)[0]
X3 = np.random.default_rng(1).random((80, 5, 3), dtype=np.float32)

HOST_PROBE = r"""
import json, sys
import mojolearn
loaded = lambda: sorted(m for m in sys.modules if m.split('.')[-1].startswith('_mojolearn') and m.endswith('_host'))
before = loaded()
step = {}
if len(sys.argv) > 1:
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.random((64, 4), dtype=np.float32); y = X.sum(axis=1).astype(np.float32)
    getattr(mojolearn, sys.argv[1])(device="cpu").fit(X, y)
    step = {"after": loaded()}
print(json.dumps(dict(before=before, **step)))
"""


def _probe(*args):
    done = subprocess.run([sys.executable, "-c", HOST_PROBE, *args], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.gpu
def test_gpu_import_loads_no_cpu_host_image():
    assert _probe()["before"] == [], "a GPU process imported a CPU host image at import"


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["LinearRegression", "Ridge"])
def test_cpu_device_fit_loads_its_host_image_on_demand(name):
    probe = _probe(name)
    assert probe["before"] == []
    assert probe["after"], f"device='cpu' {name} fit loaded no host image (did it run on the GPU?)"


#: fx-host-import: a device="cpu" fit of each family beside a GPU set, and the same after a pickle round trip in a
#: fresh process.
HOST_FAMILIES = [
    ("StandardScaler", {}, None, "transform"),
    ("PCA", dict(n_components=2), None, "transform"),
    ("NMF", dict(n_components=2, random_state=0), None, "transform"),
    ("KernelPCA", dict(n_components=2), None, "transform"),
    ("Ridge", {}, "r", "predict"),
    ("LinearRegression", {}, "r", "predict"),
    ("Lasso", dict(alpha=0.01), "r", "predict"),
    ("LogisticRegression", {}, "c", "predict_proba"),
    ("QNRegressor", {}, "r", "predict"),
    ("RandomForestRegressor", dict(n_estimators=5, random_state=0), "r", "predict"),
    ("MLPRegressor", dict(max_iter=5, random_state=0), "r", "predict"),
]

UNPICKLE = r"""
import pickle, sys
import numpy as np
model = pickle.loads(open(sys.argv[1], 'rb').read())
X = np.load(sys.argv[2])
np.save(sys.argv[3], np.asarray(getattr(model, sys.argv[4])(X)))
"""


@pytest.mark.parametrize("name,params,target,read", HOST_FAMILIES, ids=[f[0] for f in HOST_FAMILIES])
def test_cpu_device_model_runs_after_pickle_in_a_fresh_process(name, params, target, read, tmp_path):
    model = make(name, "cpu", **params)
    model.fit(X) if target is None else model.fit(X, Y if target == "r" else C)
    expected = np.asarray(getattr(model, read)(X_NEW))
    (tmp_path / "m.pkl").write_bytes(pickle.dumps(model))
    np.save(tmp_path / "x.npy", X_NEW)
    done = subprocess.run([sys.executable, "-c", UNPICKLE, str(tmp_path / "m.pkl"), str(tmp_path / "x.npy"),
                           str(tmp_path / "out.npy"), read], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert_same(np.load(tmp_path / "out.npy"), expected, f"{name} after unpickling")


#: label-steps: device="cpu" classifiers fit + predict + score (labels decoded by the native label steps).
CLASSIFIERS = [
    ("LogisticRegression", {}),
    ("SVC", {}),
    ("RandomForestClassifier", dict(n_estimators=5, random_state=0)),
    ("KNeighborsClassifier", {}),
    ("LinearSVC", {}),
]


@pytest.mark.parametrize("labels", ["int32", "str", "sparse_int"])
@pytest.mark.parametrize("name,params", CLASSIFIERS, ids=[c[0] for c in CLASSIFIERS])
def test_cpu_device_classifier_labels_round_trip(name, params, labels):
    y = {"int32": C, "str": np.array(["a", "b", "c"])[C], "sparse_int": np.array([3, 7, 9], np.int64)[C]}[labels]
    model = make(name, "cpu", **params).fit(X, y)
    predicted = np.asarray(model.predict(X))
    assert set(predicted.tolist()) <= set(np.asarray(y).tolist())
    assert float(model.score(X, y)) == pytest.approx(float(np.mean(predicted == np.asarray(y))), abs=1e-6)


def test_cpu_device_lstm_classifier_fit_predict_score():
    y = binary(C[:80])
    model = make("LSTMClassifier", "cpu", hidden_size=4, max_epochs=2, random_state=0).fit(X3, y)
    predicted = np.asarray(model.predict(X3))
    assert set(predicted.tolist()) <= {0, 1}
    assert float(model.score(X3, y)) == pytest.approx(float(np.mean(predicted == y)), abs=1e-6)


@pytest.mark.parametrize("name,target", [("SVC", "binary"), ("SVR", "r")])
def test_kernel_svm_save_load_round_trip(name, target, tmp_path):
    # fx-audit session 2: kernel SVM binary / SVR export + import round trip.
    y = binary(C) if target == "binary" else Y
    model = make(name, "cpu").fit(X, y)
    path = str(tmp_path / f"{name}.npz")
    model.save(path)
    assert_same(type(model).load(path).predict(X_NEW), model.predict(X_NEW), f"{name} loaded")


def test_pls_repeated_predicts_are_equal():
    # fx-audit session 2: PLS member repeated predicts (one upload).
    model = make("PLSRegression", "cpu", n_components=2).fit(X, Y)
    first = np.asarray(model.predict(X_NEW))
    for _ in range(3):
        assert_same(model.predict(X_NEW), first, "PLS predict again")
