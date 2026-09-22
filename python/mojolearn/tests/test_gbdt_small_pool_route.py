# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GradientBoosting's small-pool host route (perf/gbdt-small-round2,
2026-09-22): `ensemble._small_pool_host`.

What is checked: the route switch accepts auto, device and host and refuses
anything else by name; `device` never routes; the configurations outside the
covered lanes (weights, categorical columns, groups, an RMSE or multiclass
fit with an eval set, another grow policy or loss) never route; every lane
`_HOST_ROUTE_LANES` names has all four recorded columns; the identity harness pins the device
route so the GPU columns keep hashing the GPU. Where a GPU set AND the gbdt
host binding are built, a small Ordered and a small Plain fit train on the
host under `auto` and return the device fit's model text, loss curve and
predictions bit for bit, and so do Logloss fits with an eval set (Plain and
Ordered, with the detector and use_best_model) and Plain multiclass fits at
320 and 3,200 rows (perf/gbdt-route-eval-multiclass); the refused cases train
on the device.

    cd python && python3 -m pytest -q mojolearn/tests/test_gbdt_small_pool_route.py
"""
import hashlib
import os
import pathlib

import pytest

from mojolearn import _backend
from mojolearn import ensemble
from mojolearn.ensemble import GradientBoosting

ENV = ensemble._HOST_ROUTE_ENV


@pytest.fixture
def route(monkeypatch):
    def set_route(value):
        monkeypatch.setenv(ENV, value)
    return set_route


def test_route_switch_refuses_unknown_values(route):
    route("gpu")
    with pytest.raises(ValueError, match=ENV):
        GradientBoosting()._small_pool_host(320, 10, 0, 0, 0, False)


def test_device_never_routes(route):
    route("device")
    assert GradientBoosting()._small_pool_host(320, 10, 0, 0, 0, False) is None


@pytest.mark.parametrize("kw, args", [
    (dict(), (320, 10, 320, 0, 0, False)),           # sample_weight
    (dict(), (320, 10, 0, 10, 0, False)),            # categorical flags
    (dict(), (320, 10, 0, 0, 50, False)),            # eval set, Ordered RMSE
    (dict(boosting_type="Plain"), (320, 10, 0, 0, 50, False)),   # eval set, Plain RMSE
    (dict(boosting_type="Ordered"), (320, 10, 0, 0, 50, False)),
    (dict(loss="MultiClass"), (320, 10, 0, 0, 50, False)),       # eval set, multiclass
    (dict(loss="MultiClassOneVsAll"), (320, 10, 0, 0, 50, False)),
    (dict(), (320, 10, 0, 0, 0, True)),              # group_id / pairs
    (dict(grow_policy="Depthwise"), (320, 10, 0, 0, 0, False)),
    (dict(loss="MAE"), (320, 10, 0, 0, 0, False)),
])
def test_uncovered_configurations_never_route(route, kw, args):
    route("host")
    assert GradientBoosting(**kw)._small_pool_host(*args) is None


def test_auto_route_is_bounded_by_pool_size(route):
    route("auto")
    cells = ensemble._HOST_ROUTE_MAX_CELLS
    assert GradientBoosting()._small_pool_host(cells // 10 + 1, 10, 0, 0, 0, False) is None


def test_identity_harness_pins_the_device_route():
    here = pathlib.Path(__file__).resolve()
    candidates = [here.parents[3] / "tools" / "identity_break.py",
                  here.parents[1] / "_identity_break.py"]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        pytest.skip("neither tools/identity_break.py nor its wheel copy is present")
    assert 'os.environ["MOJOLEARN_GBDT_ROUTE"] = "device"' in source.read_text()


def _fit(bt, loss, n):
    np = pytest.importorskip("numpy")
    rng = np.random.RandomState(0)
    X = rng.randn(n, 10).astype(np.float32)
    y = (X[:, 0] * 2 + np.sin(X[:, 1]) + 0.3 * rng.randn(n)).astype(np.float32)
    if loss == "Logloss":
        y = (y > 0).astype(np.float32)
    m = GradientBoosting(loss=loss, max_depth=6, n_estimators=20,
                         boosting_type=bt, numeric_mode="identical").fit(X, y)
    digest = hashlib.sha256(
        str(m.model_).encode() + repr(m.loss_curve_).encode()
        + np.asarray(m.predict(X)).tobytes()
    ).hexdigest()
    return m.fit_route_, digest


@pytest.mark.parametrize("bt, loss", [("Ordered", "RMSE"), ("Plain", "Logloss")])
def test_host_route_returns_the_device_bits(route, bt, loss):
    if _backend._CPU_ONLY is not None:
        pytest.skip("CPU-only install: every fit is the host fit")
    if not os.path.exists(_backend.host_module_path("_mojolearn_gbdt_host")):
        pytest.skip("the gbdt host binding is not built")
    route("auto")
    ran_auto, auto = _fit(bt, loss, 320)
    route("device")
    ran_device, device = _fit(bt, loss, 320)
    assert (ran_auto, ran_device) == ("host", "device")
    assert auto == device


# ---- one-border columns (perf/gbdt-host-one-border, 2026-09-22) ----

_ONE_BORDER_TEXT = "format mojolearn-model 2\nfeature 0 folds 7 one_hot 0\nfeature 1 folds 1 one_hot 0\n"
_NO_ONE_BORDER_TEXT = "format mojolearn-model 2\nfeature 0 folds 7 one_hot 0\nfeature 1 folds 12 one_hot 0\n"


@pytest.mark.parametrize("vendor, route_value, text, admitted", [
    ("metal", "auto", _ONE_BORDER_TEXT, True),     # witnessed: Apple == CPU
    ("cuda", "auto", _ONE_BORDER_TEXT, False),     # NVIDIA column owed
    ("hip", "auto", _ONE_BORDER_TEXT, False),      # AMD column owed
    ("cuda", "host", _ONE_BORDER_TEXT, True),      # forced host keeps it
    ("cuda", "auto", _NO_ONE_BORDER_TEXT, True),   # no binary column
])
def test_one_border_pools_route_only_where_witnessed(monkeypatch, route, vendor,
                                                     route_value, text, admitted):
    route(route_value)
    monkeypatch.setattr(_backend, "vendor", lambda: vendor)
    assert GradientBoosting()._host_one_border_admitted(text) is admitted


def test_one_border_host_route_returns_the_device_bits(route):
    """Where it routes (Metal), a pool with binary columns trains on the host
    and returns the device fit's bits."""
    if _backend._CPU_ONLY is not None:
        pytest.skip("CPU-only install: every fit is the host fit")
    if not os.path.exists(_backend.host_module_path("_mojolearn_gbdt_host")):
        pytest.skip("the gbdt host binding is not built")
    if _backend.vendor() not in ensemble._HOST_ONE_BORDER_VENDORS:
        pytest.skip("one-border pools do not auto-route on this vendor")
    np = pytest.importorskip("numpy")
    rng = np.random.RandomState(0)
    X = rng.randn(320, 10).astype(np.float32)
    X[:, 2] = (rng.rand(320) < 0.3).astype(np.float32)
    X[:, 7] = (rng.rand(320) < 0.6).astype(np.float32)
    y = (X[:, 0] + 1.5 * X[:, 2] - X[:, 7]).astype(np.float32)
    out = {}
    for value in ("auto", "device"):
        route(value)
        for bt in ("Ordered", "Plain"):
            m = GradientBoosting(max_depth=6, n_estimators=20, boosting_type=bt,
                                 numeric_mode="identical").fit(X, y)
            out[value, bt] = (m.fit_route_, str(m.model_),
                              np.asarray(m.loss_curve_).tobytes(),
                              np.asarray(m.predict(X)).tobytes())
    for bt in ("Ordered", "Plain"):
        assert out["auto", bt][0] == "host" and out["device", bt][0] == "device"
        assert " folds 1 " in out["auto", bt][1]
        assert out["auto", bt][1:] == out["device", bt][1:]


# ---- eval sets and multiclass (perf/gbdt-route-eval-multiclass, 2026-09-22) ----

_ALL_FOUR = {"nvidia", "amd", "apple", "cpu"}


def test_every_routed_configuration_names_lanes_with_all_four_columns():
    """The rule of the route: a configuration trains on the host only where a
    verifier lane with all four recorded columns covers it. Every lane
    `_HOST_ROUTE_LANES` names must have the NVIDIA, AMD, Apple and CPU
    columns on the train part of every fixture of the shipped table."""
    import json
    table = pathlib.Path(ensemble.__file__).resolve().parent / "verify_reference" / "table.json"
    cells = json.loads(table.read_text())["cells"]
    for config, lanes in ensemble._HOST_ROUTE_LANES.items():
        assert lanes, config
        for lane in lanes:
            fixtures = [k for k in cells if k.split("/")[0] == lane]
            assert fixtures, (config, lane)
            for key in fixtures:
                cols = set(cells[key]["train"]["cols"])
                assert _ALL_FOUR <= cols, (config, lane, key, sorted(cols))
    for config in ensemble._HOST_ONE_BORDER_CONFIGS:
        assert config in ensemble._HOST_ROUTE_LANES, config


@pytest.mark.parametrize("config", [
    ("Plain", "RMSE", True),          # refused by the host binding by name
    ("Ordered", "RMSE", True),        # no lane records it
    ("Plain", "MultiClass", True),
    ("Plain", "MultiClassOneVsAll", True),
    ("Ordered", "MultiClass", False),
])
def test_uncovered_eval_and_multiclass_configurations_are_absent(config):
    assert config not in ensemble._HOST_ROUTE_LANES


@pytest.mark.parametrize("kw, n_eval", [
    (dict(loss="Logloss", boosting_type="Plain"), 50),
    (dict(loss="Logloss", boosting_type="Ordered"), 50),
    (dict(loss="Logloss"), 50),                        # the defaults, Ordered
    (dict(loss="MultiClass"), 0),
    (dict(loss="MultiClassOneVsAll"), 0),
])
def test_opened_configurations_route(route, kw, n_eval):
    if _backend._CPU_ONLY is not None:
        pytest.skip("CPU-only install: every fit is the host fit")
    if not os.path.exists(_backend.host_module_path("_mojolearn_gbdt_host")):
        pytest.skip("the gbdt host binding is not built")
    route("auto")
    assert GradientBoosting(**kw)._small_pool_host(320, 10, 0, 0, n_eval, False) is not None


@pytest.mark.parametrize("vendor, config, admitted", [
    ("metal", ("Plain", "RMSE", False), True),
    ("metal", ("Ordered", "Logloss", False), True),
    ("metal", ("Plain", "Logloss", True), False),     # no one-border lane with an eval set
    ("metal", ("Ordered", "Logloss", True), False),
    ("metal", ("Plain", "MultiClass", False), False),
])
def test_one_border_routes_only_for_the_binary_lane_configurations(
        monkeypatch, route, vendor, config, admitted):
    route("auto")
    monkeypatch.setattr(_backend, "vendor", lambda: vendor)
    got = GradientBoosting()._host_one_border_admitted(_ONE_BORDER_TEXT, config)
    assert got is admitted


def _needs_host_and_device():
    if _backend._CPU_ONLY is not None:
        pytest.skip("CPU-only install: every fit is the host fit")
    if not os.path.exists(_backend.host_module_path("_mojolearn_gbdt_host")):
        pytest.skip("the gbdt host binding is not built")


def _pool(n, kind, seed=0):
    np = pytest.importorskip("numpy")
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10).astype(np.float32)
    z = X[:, 0] * 2 + np.sin(X[:, 1]) + 0.8 * rng.randn(n)
    if kind == "Logloss":
        y = (z > 0).astype(np.float32)
    elif kind == "multi":
        y = np.digitize(z, [-1.0, 1.0]).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


def _read_back(m, X):
    np = pytest.importorskip("numpy")
    parts = [str(m.model_), repr(m.loss_curve_), repr(m.test_loss_curve_),
             int(m.best_iteration_), bool(m.stopped_early_),
             np.asarray(m.predict(X)).tobytes()]
    if m.loss != "RMSE":
        parts.append(np.asarray(m.predict_proba(X)).tobytes())
    return m.fit_route_, parts


def _auto_and_device(route, make, X, y, **fit_kw):
    route("auto")
    auto = _read_back(make().fit(X, y, **fit_kw), X)
    route("device")
    device = _read_back(make().fit(X, y, **fit_kw), X)
    return auto, device


_EVAL_CASES = {
    "plain-logloss-eval": dict(loss="Logloss", boosting_type="Plain", n_estimators=20),
    "plain-logloss-eval-od-shrink": dict(loss="Logloss", boosting_type="Plain",
                                         n_estimators=30, max_depth=7, learning_rate=1.8,
                                         od_type="Iter", od_wait=2),
    "ordered-logloss-eval-od": dict(loss="Logloss", boosting_type="Ordered",
                                    n_estimators=20, od_type="Iter", od_wait=5),
    "ordered-logloss-eval-od-shrink": dict(loss="Logloss", boosting_type="Ordered",
                                           n_estimators=30, max_depth=7, learning_rate=1.8,
                                           od_type="Iter", od_wait=2),
    "ordered-logloss-eval-defaults": dict(loss="Logloss", boosting_type="Ordered",
                                          n_estimators=20, bootstrap_type="Bayesian",
                                          random_strength=1.0),
}


@pytest.mark.parametrize("n", [320, 3200])
@pytest.mark.parametrize("case", sorted(_EVAL_CASES))
def test_eval_set_host_route_returns_the_device_bits(route, case, n):
    _needs_host_and_device()
    X, y = _pool(n, "Logloss")
    Xe, ye = _pool(n // 4, "Logloss", seed=1)
    kw = dict(max_depth=6, numeric_mode="identical")
    kw.update(_EVAL_CASES[case])
    auto, device = _auto_and_device(
        route, lambda: GradientBoosting(**kw), X, y, eval_set=(Xe, ye))
    assert (auto[0], device[0]) == ("host", "device")
    assert auto[1] == device[1]


_MULTI_CASES = {
    "multiclass-class-weights": dict(loss="MultiClass", class_weights=[1.0, 2.0, 0.5],
                                     bootstrap_type="No", random_strength=0.0),
    "multiclass-defaults": dict(loss="MultiClass"),
    "onevsall": dict(loss="MultiClassOneVsAll", bootstrap_type="No", random_strength=0.0),
    "onevsall-defaults": dict(loss="MultiClassOneVsAll", class_weights=[1.0, 2.0, 0.5]),
}


@pytest.mark.parametrize("n", [320, 3200])
@pytest.mark.parametrize("case", sorted(_MULTI_CASES))
def test_multiclass_host_route_returns_the_device_bits(route, case, n):
    _needs_host_and_device()
    X, y = _pool(n, "multi")
    kw = dict(max_depth=6, n_estimators=20, numeric_mode="identical")
    kw.update(_MULTI_CASES[case])
    auto, device = _auto_and_device(route, lambda: GradientBoosting(**kw), X, y)
    assert (auto[0], device[0]) == ("host", "device")
    assert auto[1] == device[1]


@pytest.mark.parametrize("case", ["plain-rmse-eval", "ordered-rmse-eval",
                                  "multiclass-one-border", "logloss-eval-one-border",
                                  "multiclass-eval"])
def test_refused_configurations_fall_back_to_the_device(route, case):
    """Each trains on the device under auto, silently, and returns exactly
    the device-pinned fit."""
    _needs_host_and_device()
    np = pytest.importorskip("numpy")
    fit_kw = {}
    kw = dict(max_depth=6, n_estimators=20, numeric_mode="identical")
    if case in ("plain-rmse-eval", "ordered-rmse-eval"):
        X, y = _pool(320, "RMSE")
        Xe, ye = _pool(80, "RMSE", seed=1)
        kw.update(loss="RMSE", boosting_type=case.split("-")[0].title())
        fit_kw = dict(eval_set=(Xe, ye))
    elif case == "multiclass-eval":
        X, y = _pool(320, "multi")
        Xe, ye = _pool(80, "multi", seed=1)
        kw.update(loss="MultiClass")
        fit_kw = dict(eval_set=(Xe, ye))
    else:
        kind = "multi" if case.startswith("multiclass") else "Logloss"
        X, y = _pool(320, kind)
        X[:, 2] = (np.random.RandomState(3).rand(320) < 0.4).astype(np.float32)
        kw.update(loss="MultiClass" if kind == "multi" else "Logloss")
        if kind == "Logloss":
            Xe, ye = _pool(80, "Logloss", seed=1)
            Xe[:, 2] = (np.random.RandomState(4).rand(80) < 0.4).astype(np.float32)
            kw.update(boosting_type="Plain")
            fit_kw = dict(eval_set=(Xe, ye))
    auto, device = _auto_and_device(
        route, lambda: GradientBoosting(**kw), X, y, **fit_kw)
    assert (auto[0], device[0]) == ("device", "device")
    assert auto[1] == device[1]
