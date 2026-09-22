# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GradientBoosting's small-pool host route (perf/gbdt-small-round2,
2026-09-22): `ensemble._small_pool_host`.

What is checked: the route switch accepts auto, device and host and refuses
anything else by name; `device` never routes; the configurations outside the
covered lanes (weights, categorical columns, groups, an eval set, another
grow policy or loss) never route; the identity harness pins the device
route so the GPU columns keep hashing the GPU. Where a GPU set AND the gbdt
host binding are built, a small Ordered and a small Plain fit train on the
host under `auto` and return the device fit's model text, loss curve and
predictions bit for bit.

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
    (dict(), (320, 10, 0, 0, 50, False)),            # eval set
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
