# SPDX-License-Identifier: Apache-2.0
"""The check of an estimator that takes `device=` (python/mojolearn/_check_both.py), at the Python boundary.

No binding and no GPU: the module is loaded by path and `mojolearn` is a stub. The estimator is a stub that takes
`device=` the way Ridge, LinearRegression and LogisticRegression do (`_mojolearn_device`, `get_params`, `vendor_used`;
the classifier stub adds a label table and probabilities). What is held here:
both sides are the same class constructed on each device in THIS process; no file is saved, `host_model` is never
called (there is no child process at all); the report names both routes; the check can still fail (one element, one ULP);
and a check whose two sides ran on one device is refused instead of comparing a thing to itself.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python/mojolearn"

_spec = importlib.util.spec_from_file_location("check_both_device_under_test", PKG / "_check_both.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

LABELS = dict(describe_device=lambda: "NVIDIA Test GPU, 550.1", cpu_model=lambda: "Test CPU 16-Core")


class DeviceModel:
    """Answers X @ coef_ on the device it was constructed for."""
    _mojolearn_device = True
    numeric_mode = "identical"
    built = []          # every construction, in order: (device,)
    nudge_cpu = None    # an element index the CPU answer moves by one ULP
    one_device = False  # both devices report the GPU binary

    def __init__(self, *, alpha=1.0, device=None):
        if device not in (None, "gpu", "cpu"):
            raise ValueError(f"device must be 'gpu', 'cpu' or None, got {device!r}")
        self.alpha, self.device, self.fits = alpha, device, 0
        DeviceModel.built.append(device)

    def get_params(self, deep=True):
        return dict(alpha=self.alpha, device=self.device)

    def vendor_used(self):
        return "cuda" if self.device != "cpu" or DeviceModel.one_device else "cpu"

    def fit(self, X, y):
        self.fits += 1
        self.coef_ = np.linalg.lstsq(X.astype(np.float64), y.astype(np.float64), rcond=None)[0].astype(np.float32)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        y = (X @ self.coef_).astype(np.float32)
        if self.device == "cpu" and DeviceModel.nudge_cpu is not None:
            y[DeviceModel.nudge_cpu] = np.nextafter(y[DeviceModel.nudge_cpu], np.float32(np.inf))
        return y

    def save(self, path):
        raise AssertionError("an estimator that takes device= must not be saved to reach the CPU")


class HostDeviceModel(DeviceModel):
    _HOST_INFERENCE_ONLY = True


def install(tmp_path, vendor="cuda", mode="identical", host_loaded=True):
    pkg = str(tmp_path / "site/mojolearn")
    bindings = {
        "mojolearn._mojolearn_estimators": dict(path=f"{pkg}/cuda/sm_89/identical/_mojolearn_estimators.so",
                                                loaded=True),
        "mojolearn._host._mojolearn_estimators_host": dict(path=f"{pkg}/host/_mojolearn_estimators_host.so",
                                                           loaded=host_loaded),
    }
    report = dict(version="9.9.9", numeric_mode=mode, vendor=vendor, gpu_arch="sm_89", code_kind="native",
                  package_dir=pkg, bindings=bindings)

    def host_model(path):
        raise AssertionError("an estimator that takes device= must not go through host_model")

    return SimpleNamespace(vendor=lambda: vendor, numeric_mode=lambda: mode, runtime_report=lambda: report,
                           host_model=host_model)


def setup(nudge=None, one_device=False):
    DeviceModel.built = []
    DeviceModel.nudge_cpu = nudge
    DeviceModel.one_device = one_device
    rng = np.random.default_rng(11)
    X = rng.random((48, 3), dtype=np.float32)
    y = (X @ np.array([0.5, -1.25, 2.0], dtype=np.float32)).astype(np.float32)
    return X, y


def test_takes_device_is_the_class_marker_and_not_the_explicit_host_class():
    setup()
    assert cb.takes_device(DeviceModel())
    assert not cb.takes_device(HostDeviceModel())
    assert not cb.takes_device(SimpleNamespace(device=None))


def test_on_device_returns_the_estimator_itself_on_its_own_device_and_a_twin_with_its_fitted_state_on_the_other():
    X, y = setup()
    est = DeviceModel(alpha=0.25).fit(X, y)
    assert cb.on_device(est, "gpu") is est                           # None is the installation's GPU here
    explicit = DeviceModel(alpha=0.25, device="gpu")
    assert cb.on_device(explicit, "gpu") is explicit
    twin = cb.on_device(est, "cpu")
    assert type(twin) is DeviceModel and twin is not est
    assert twin.get_params() == dict(alpha=0.25, device="cpu") and est.get_params()["device"] is None
    assert twin.coef_ is est.coef_ and twin.n_features_in_ == 3      # the same fitted objects, in memory
    bare = cb.on_device(est, "cpu", fitted=False)
    assert not hasattr(bare, "coef_")


def test_fitted_model_form_runs_the_same_class_on_both_devices_in_this_process(tmp_path):
    X, y = setup()
    est = DeviceModel().fit(X, y)
    result = cb.check_estimator(install(tmp_path), est, X, **LABELS)
    assert result["ran"] and result["transfer"] == "device" and result["form"] == "fitted-model"
    assert DeviceModel.built == [None, "cpu"]                        # the given estimator, then its CPU twin
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"] == dict(device=None, vendor_used="cuda")
    assert cpu["route"] == dict(device="cpu", vendor_used="cpu")
    assert result["verdict"]["match"] and gpu["outputs"][0]["sha256"] == cpu["outputs"][0]["sha256"]
    text = cb.format_check(result)
    assert "route     device=None (this installation's device) in this process, binary vendor cuda" in text
    assert "route     device='cpu' in this process, binary vendor cpu" in text
    assert "constructed on each device" in text and "saved and loaded" not in text
    assert text.splitlines()[-1].startswith("MATCH")


def test_the_device_check_can_fail_one_element_one_ulp(tmp_path):
    X, y = setup(nudge=5)
    est = DeviceModel().fit(X, y)
    result = cb.check_estimator(install(tmp_path), est, X, **LABELS)
    assert result["ran"] and not result["verdict"]["match"]
    assert list(result["verdict"]["first"]["index"]) == [5]
    assert cb.format_check(result).splitlines()[-1].startswith("MISMATCH")


def test_fit_argument_fits_the_given_estimator_once_and_both_devices_answer_from_it(tmp_path):
    X, y = setup()
    est = DeviceModel()
    result = cb.check_estimator(install(tmp_path), est, X, fit=(X, y), **LABELS)
    assert result["ran"] and result["fitted_here"] and est.fits == 1
    assert DeviceModel.built == [None, "cpu"] and result["verdict"]["match"]
    assert "fit_wall_s" in result["sides"]["gpu"]


def test_fit_both_fits_one_estimator_per_device_in_this_process_and_starts_no_child(tmp_path):
    X, y = setup()
    est = DeviceModel()
    result = cb.check_estimator(install(tmp_path), est, X, fit=(X, y), fit_both=True, **LABELS)
    assert result["ran"] and result["form"] == "fit-both" and result["transfer"] == "device"
    assert est.fits == 1 and DeviceModel.built == [None, "cpu"]
    for side in ("gpu", "cpu"):
        assert "fit_wall_s" in result["sides"][side] and "process_wall_s" not in result["sides"][side]
    assert result["verdict"]["match"]
    text = cb.format_check(result)
    assert "one estimator per device (this installation's GPU, device='cpu'), both in this process" in text
    assert "child process" not in text


def test_an_estimator_given_on_the_cpu_gets_its_gpu_side_constructed(tmp_path):
    X, y = setup()
    est = DeviceModel(device="cpu").fit(X, y)
    result = cb.check_estimator(install(tmp_path), est, X, **LABELS)
    assert result["ran"] and DeviceModel.built == ["cpu", "gpu"]
    assert result["sides"]["gpu"]["route"]["vendor_used"] == "cuda"
    assert result["sides"]["cpu"]["route"]["vendor_used"] == "cpu"


def test_two_sides_on_one_device_are_refused_not_compared(tmp_path):
    X, y = setup(one_device=True)
    est = DeviceModel().fit(X, y)
    result = cb.check_estimator(install(tmp_path), est, X, **LABELS)
    assert not result["ran"] and "did not run on two devices" in result["reason"]


def test_no_loaded_cpu_binary_is_refused(tmp_path):
    X, y = setup()
    est = DeviceModel().fit(X, y)
    result = cb.check_estimator(install(tmp_path, host_loaded=False), est, X, **LABELS)
    assert not result["ran"] and "no CPU binary is loaded" in result["reason"]


def test_fast_mode_is_refused_before_anything_is_constructed(tmp_path):
    X, y = setup()
    est = DeviceModel().fit(X, y)
    DeviceModel.built = []
    fast = cb.check_estimator(install(tmp_path, mode="fast"), est, X, **LABELS)
    assert not fast["ran"] and fast.get("refused") == "fast"
    assert DeviceModel.built == []


def test_cpu_only_install_runs_the_cpu_side_and_the_gpu_side_is_not_applicable(tmp_path):
    X, y = setup()
    est = DeviceModel().fit(X, y)
    DeviceModel.built = []
    cpu_only = cb.check_estimator(install(tmp_path, vendor="cpu"), est, X, **LABELS)
    assert not cpu_only["ran"] and cpu_only["reason"] == cb.NO_GPU and "verdict" not in cpu_only
    assert cpu_only["gpu_side"] == cb.NOT_APPLICABLE and set(cpu_only["sides"]) == {"cpu"}
    assert cpu_only["sides"]["cpu"]["route"]["vendor_used"] == "cpu" and DeviceModel.built == ["cpu"]
    text = cb.format_check(cpu_only)
    assert "NOT RUN" in text and cb.NOT_APPLICABLE in text and "side      cpu" in text
    both = cb.check_estimator(install(tmp_path, vendor="cpu"), DeviceModel(), X, fit=(X, y), fit_both=True,
                              **LABELS)
    assert not both["ran"] and set(both["sides"]) == {"cpu"} and both["sides"]["cpu"]["fit_wall_s"] >= 0


def test_to_cpu_is_the_one_state_transfer_verify_cross_check_calls(tmp_path):
    X, y = setup()
    est = DeviceModel().fit(X, y)
    host, path = cb.to_cpu(install(tmp_path), est, str(tmp_path), name="ridge", form=("save", ".npz"))
    assert path is None and host.get_params()["device"] == "cpu" and host.coef_ is est.coef_
    assert not list(tmp_path.glob("*.npz"))


class DeviceClassifier:
    """A two-device classifier stub in LogisticRegression's shape: a sorted label table (`classes_`, a list), a
    weight block, `predict` returning labels and `predict_proba` returning float64 probabilities."""
    _mojolearn_device = True
    numeric_mode = "identical"
    built = []
    nudge_cpu = None    # (row, column) of one probability the CPU answer moves by one ULP
    flip_cpu = None     # a row whose CPU label is the other class

    def __init__(self, *, C=1.0, class_weight=None, device=None):
        if device not in (None, "gpu", "cpu"):
            raise ValueError(f"device must be 'gpu', 'cpu' or None, got {device!r}")
        self.C, self.class_weight, self.device = C, class_weight, device
        DeviceClassifier.built.append(device)

    def get_params(self, deep=True):
        return dict(C=self.C, class_weight=self.class_weight, device=self.device)

    def vendor_used(self):
        return "cpu" if self.device == "cpu" else "cuda"

    def fit(self, X, y, sample_weight=None):
        self.classes_ = sorted(set(y.tolist()))
        weight = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        means = [np.average(X[y == label], axis=0, weights=weight[y == label]) for label in self.classes_]
        self._w = np.asarray(means, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        scores = -((X[:, None, :].astype(np.float64) - self._w[None, :, :]) ** 2).sum(axis=2)
        p = np.exp(scores - scores.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        if self.device == "cpu" and DeviceClassifier.nudge_cpu is not None:
            p[DeviceClassifier.nudge_cpu] = np.nextafter(p[DeviceClassifier.nudge_cpu], np.inf)
        return p

    def predict(self, X):
        codes = self.predict_proba(X).argmax(axis=1)
        labels = np.asarray(self.classes_, dtype=np.int64)[codes]
        if self.device == "cpu" and DeviceClassifier.flip_cpu is not None:
            row = DeviceClassifier.flip_cpu
            labels[row] = next(label for label in self.classes_ if label != labels[row])
        return labels

    def save(self, path):
        raise AssertionError("an estimator that takes device= must not be saved to reach the CPU")


def classifier_setup(classes=2, nudge=None, flip=None):
    DeviceClassifier.built = []
    DeviceClassifier.nudge_cpu = nudge
    DeviceClassifier.flip_cpu = flip
    rng = np.random.default_rng(23)
    X = rng.random((60, 4), dtype=np.float32)
    y = (np.arange(60) % classes).astype(np.int64) * 3 + 1       # labels 1, 4, 7, ...: not their own codes
    X += (y[:, None] / 3.0).astype(np.float32)
    return X, y


def test_classifier_labels_and_probabilities_are_checked_on_both_devices_in_this_process(tmp_path):
    for classes in (2, 4):
        X, y = classifier_setup(classes)
        est = DeviceClassifier(class_weight="balanced").fit(X, y, np.linspace(0.5, 2.0, len(X)))
        for method in (None, "predict_proba"):
            DeviceClassifier.built = []
            result = cb.check_estimator(install(tmp_path), est, X, method=method, **LABELS)
            assert result["ran"] and result["transfer"] == "device" and result["method"] == (method or "predict")
            assert DeviceClassifier.built == ["cpu"]                 # the CPU twin; the given estimator is the GPU side
            assert result["sides"]["cpu"]["route"] == dict(device="cpu", vendor_used="cpu")
            assert result["verdict"]["match"]
            shape = [len(X)] if method is None else [len(X), classes]
            assert list(result["sides"]["gpu"]["outputs"][0]["shape"]) == shape


def test_the_cpu_twin_of_a_classifier_answers_from_the_same_label_table_and_weights(tmp_path):
    X, y = classifier_setup(4)
    est = DeviceClassifier(C=0.5, class_weight={1: 2.0}).fit(X, y)
    twin = cb.on_device(est, "cpu")
    assert twin.get_params() == dict(C=0.5, class_weight={1: 2.0}, device="cpu")
    assert twin.classes_ is est.classes_ and twin._w is est._w
    assert twin.classes_ == [1, 4, 7, 10]


def test_the_classifier_check_can_fail_on_one_probability_and_on_one_label(tmp_path):
    X, y = classifier_setup(4, nudge=(7, 2))
    est = DeviceClassifier().fit(X, y)
    result = cb.check_estimator(install(tmp_path), est, X, method="predict_proba", **LABELS)
    assert result["ran"] and not result["verdict"]["match"]
    assert list(result["verdict"]["first"]["index"]) == [7, 2]
    X, y = classifier_setup(2, flip=11)
    est = DeviceClassifier().fit(X, y)
    result = cb.check_estimator(install(tmp_path), est, X, **LABELS)
    assert result["ran"] and not result["verdict"]["match"]
    assert list(result["verdict"]["first"]["index"]) == [11]


def test_classifier_fit_both_fits_one_estimator_per_device_with_the_same_weights(tmp_path):
    X, y = classifier_setup(4)
    est = DeviceClassifier(class_weight="balanced")
    weight = np.linspace(0.5, 2.0, len(X))
    result = cb.check_estimator(install(tmp_path), est, X, method="predict_proba", fit=(X, y, weight),
                                fit_both=True, **LABELS)
    assert result["ran"] and result["form"] == "fit-both" and result["transfer"] == "device"
    assert DeviceClassifier.built == [None, "cpu"] and result["verdict"]["match"]     # the given estimator, its twin


# ---- the hooks of `on_device` (lane dw-core round 2): one design, two steps, each with a default and a class hook


class _Leaf:
    _mojolearn_device = True

    def __init__(self, a=1, device=None):
        self.a, self.device = a, device

    def get_params(self, deep=True):
        return dict(a=self.a, device=self.device)


class _Meta:
    _mojolearn_device = True

    def __init__(self, estimator=None, estimators=(), device=None):
        self.estimator, self.estimators, self.device = estimator, estimators, device

    def get_params(self, deep=True):
        return dict(estimator=self.estimator, estimators=self.estimators, device=self.device)


class _NoParams:
    """No get_params (Sequential), constructor data (ExponentialSmoothing), no state to share (the neural blocks)."""
    _mojolearn_device = True
    _mojolearn_device_state = None

    def __init__(self, data, device=None):
        self.data, self.device = data, device

    def _mojolearn_device_twin(self, device):
        return _NoParams(self.data, device=device)


class _NativeState(_Leaf):
    """Fitted state that is not the instance dictionary (trees)."""

    def _mojolearn_device_state(self, twin, device):
        twin.model_ = ("moved", self.model_, device)


def test_a_meta_estimator_puts_its_members_and_its_fitted_members_on_the_other_device():
    meta = _Meta(estimator=_Leaf(), estimators=[("x", _Leaf(2)), ("y", 3)])
    meta.best_estimator_ = _Leaf(5)
    meta.best_estimator_.coef_ = [9]
    meta.n_ = 4
    twin = cb.on_device(meta, "cpu")
    assert twin.device == "cpu" and twin.estimator.device == "cpu" and twin.estimators[0][1].device == "cpu"
    assert twin.estimators[1] == ("y", 3) and twin.n_ == 4
    assert twin.best_estimator_.device == "cpu" and twin.best_estimator_.coef_ is meta.best_estimator_.coef_
    # the given estimator and its members are left where they were
    assert meta.estimator.device is None and meta.estimators[0][1].device is None and meta.best_estimator_.device is None


def test_a_class_twin_builder_serves_a_model_without_get_params_and_the_state_copy_can_be_declined():
    model = _NoParams([1, 2])
    model.weights_ = object()
    twin = cb.on_device(model, "cpu")
    assert type(twin) is _NoParams and twin.device == "cpu" and twin.data is model.data
    assert not hasattr(twin, "weights_")
    assert cb.on_device(twin, "cpu") is twin and cb.on_device(model, "gpu") is model


def test_a_class_state_hook_hands_over_fitted_state_that_is_not_the_instance_dictionary():
    tree = _NativeState()
    tree.model_ = "native owner"
    assert cb.on_device(tree, "cpu").model_ == ("moved", "native owner", "cpu")
    assert not hasattr(cb.on_device(tree, "cpu", fitted=False), "model_")


def test_a_failing_hook_is_a_refusal_in_words_not_a_comparison():
    class Broken(_Leaf):
        def _mojolearn_device_state(self, twin, device):
            raise RuntimeError("no owner")

    try:
        cb.on_device(Broken(), "cpu")
    except cb.NotChecked as refused:
        assert "fitted state of Broken" in str(refused) and "no owner" in str(refused)
    else:
        raise AssertionError("a failing state hook was not refused")


def test_forward_is_a_default_method():
    assert cb.DEFAULT_METHODS == ("predict", "transform", "forward")
