# SPDX-License-Identifier: Apache-2.0
"""The check of an estimator that takes `device=` (python/mojolearn/_check_both.py), at the Python boundary.

No binding and no GPU: the module is loaded by path and `mojolearn` is a stub. The estimator is a stub that takes
`device=` the way Ridge and LinearRegression do (`_mojolearn_device`, `get_params`, `vendor_used`). What is held here:
both sides are the same class constructed on each device in THIS process; no file is saved, `host_model` is never
called and no child process is started; the report names both routes; the check can still fail (one element, one ULP);
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


def no_child(directory):
    raise AssertionError("an estimator that takes device= must not start a child process")


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
    result = cb.check_estimator(install(tmp_path), est, X, spawn=no_child, **LABELS)
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
    result = cb.check_estimator(install(tmp_path), est, X, spawn=no_child, **LABELS)
    assert result["ran"] and not result["verdict"]["match"]
    assert list(result["verdict"]["first"]["index"]) == [5]
    assert cb.format_check(result).splitlines()[-1].startswith("MISMATCH")


def test_fit_argument_fits_the_given_estimator_once_and_both_devices_answer_from_it(tmp_path):
    X, y = setup()
    est = DeviceModel()
    result = cb.check_estimator(install(tmp_path), est, X, fit=(X, y), spawn=no_child, **LABELS)
    assert result["ran"] and result["fitted_here"] and est.fits == 1
    assert DeviceModel.built == [None, "cpu"] and result["verdict"]["match"]
    assert "fit_wall_s" in result["sides"]["gpu"]


def test_fit_both_fits_one_estimator_per_device_in_this_process_and_starts_no_child(tmp_path):
    X, y = setup()
    est = DeviceModel()
    result = cb.check_estimator(install(tmp_path), est, X, fit=(X, y), fit_both=True, spawn=no_child, **LABELS)
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
    result = cb.check_estimator(install(tmp_path), est, X, spawn=no_child, **LABELS)
    assert result["ran"] and DeviceModel.built == ["cpu", "gpu"]
    assert result["sides"]["gpu"]["route"]["vendor_used"] == "cuda"
    assert result["sides"]["cpu"]["route"]["vendor_used"] == "cpu"


def test_two_sides_on_one_device_are_refused_not_compared(tmp_path):
    X, y = setup(one_device=True)
    est = DeviceModel().fit(X, y)
    result = cb.check_estimator(install(tmp_path), est, X, spawn=no_child, **LABELS)
    assert not result["ran"] and "did not run on two devices" in result["reason"]


def test_no_loaded_cpu_binary_is_refused(tmp_path):
    X, y = setup()
    est = DeviceModel().fit(X, y)
    result = cb.check_estimator(install(tmp_path, host_loaded=False), est, X, spawn=no_child, **LABELS)
    assert not result["ran"] and "no CPU binary is loaded" in result["reason"]


def test_cpu_only_install_and_fast_mode_are_refused_before_anything_is_constructed(tmp_path):
    X, y = setup()
    est = DeviceModel().fit(X, y)
    DeviceModel.built = []
    cpu_only = cb.check_estimator(install(tmp_path, vendor="cpu"), est, X, spawn=no_child, **LABELS)
    fast = cb.check_estimator(install(tmp_path, mode="fast"), est, X, spawn=no_child, **LABELS)
    assert not cpu_only["ran"] and cpu_only["reason"] == cb.NO_GPU
    assert not fast["ran"] and fast.get("refused") == "fast"
    assert DeviceModel.built == []


def test_to_cpu_is_the_one_state_transfer_verify_cross_check_calls(tmp_path):
    X, y = setup()
    est = DeviceModel().fit(X, y)
    host, path = cb.to_cpu(install(tmp_path), est, str(tmp_path), name="ridge", form=("save", ".npz"))
    assert path is None and host.get_params()["device"] == "cpu" and host.coef_ is est.coef_
    assert not list(tmp_path.glob("*.npz"))
