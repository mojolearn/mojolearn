# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public CPU inference for the newest gradient boosting modes through
HostGBDT (lane/inference-gbdt-modes, 2026-09-15): saved OrderedRMSE,
ExperimentalTwoLevelFeatureFreq, pointwise Bayesian eval and one-hot
categorical models, loaded by `mojolearn.host_model` and predicted by the
shipped forest host binding.

What the boundary checks hold: HostGBDT refuses an archive saved by any
other estimator name and a subclass archive whose loss that class cannot
save, by name (a CTR or tensor CTR record whose tables are missing is the
native check `gbdt_host_archive_text`); the forest gate's coded fixture transform is the rule it documents.

The runtime check (skipped, and SAID to be skipped, unless the gbdt host
binding and the forest host binary are both built and the package took the
CPU-only path): small fits of the four modes on the host, saved, loaded
through `host_model`, whose `predict` (and `predict_proba` on the Logloss
mode) bytes equal the fitted estimator's. The bit claim against the GPU
columns is identity_break's `MOJOLEARN_IDENTITY_HOST_INFER` diff, not this
file's.

    cd python && python3 -m mojolearn.tests.test_gbdt_host_modes
"""
import os
import struct
import sys
import tempfile
from pathlib import Path

import mojolearn
from mojolearn import _backend, _serialize
from mojolearn._array import Array
from mojolearn._buffer import frombytes
from mojolearn._gbdt_host import GBDT_FORMAT, HostGBDT

ROOT = Path(__file__).resolve().parents[3]

#: one float feature with one border, no trees
TEXT = ("format mojolearn-model 2\nfeatures 1 0\ntrees 0\nlosses 0\n"
        "feature 0 folds 1 one_hot 0 type float nan as_is borders 1 0.5/3f000000\n")


def _archive(path, estimator, loss, text=TEXT):
    body = text.encode("utf-8")
    _serialize.write_npz(path, {
        "format": GBDT_FORMAT, "numeric_mode": "identical", "estimator": estimator, "loss": loss,
        "model": frombytes(body, "<u1", (len(body),)),
        "meta": Array.from_list([1, 1, -1, -1, 0], "<i8"),
        "bias": Array.from_list([0.0], "<f8"),
    })


def _raises(fn, needle):
    try:
        fn()
    except (ValueError, ImportError, RuntimeError) as exc:
        assert needle in str(exc), f"expected {needle!r} in {exc}"
        return
    raise AssertionError(f"no refusal naming {needle!r}")


def test_archive_names_and_losses_refuse_before_any_binding_loads():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.npz")
        _archive(path, "CatBoostRegressor", "RMSE")
        _raises(lambda: HostGBDT.from_file(path), "was saved by CatBoostRegressor")
        path2 = os.path.join(d, "m2.npz")
        _archive(path2, "OrderedRMSE", "Logloss")
        _raises(lambda: HostGBDT.from_file(path2), "saves RMSE only")


def test_gate_coded_transform_is_the_documented_rule():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import forest_host_gate as gate
    finally:
        sys.path.pop(0)
    row = [0.9, 0.1, 0.7, 0.2, 0.6, 0.3]
    out = struct.unpack("<6f", gate.apply_transform(struct.pack("<6f", *row), 1, 6, gate.CODED))
    assert out[0] == 1.0 and out[1] == 1.0 and out[2:] == struct.unpack("<4f", struct.pack("<4f", *row[2:]))
    assert gate.apply_transform(b"ab", 1, 1, None) == b"ab"


def _forest_binary_built():
    from mojolearn._forest_host import binary_path
    return os.path.exists(binary_path())


def test_four_modes_predict_through_host_model_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_gbdt_host" not in _backend.host_families_built() or not _forest_binary_built():
        print("SKIP: the gbdt host binding or the forest host binary is not built")
        return
    import numpy as np
    from mojolearn import host_model
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1200, 6)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1]).astype(np.float32)
    yc = (x[:, 0] > 0).astype(np.int32)
    coded = np.column_stack([
        (x[:, 2] > 0).astype(np.float32),
        ((x[:, 3] > 0).astype(np.int32) * 2 + (x[:, 4] > 0)).astype(np.float32),
        x[:, :4],
    ]).astype(np.float32)
    xe = rng.standard_normal((300, 6)).astype(np.float32)
    ye = (xe[:, 0] > 0).astype(np.int32)
    fits = [
        ("OrderedRMSE", mojolearn.OrderedRMSE(n_estimators=3, max_depth=3).fit(
            x, y, permutation=rng.permutation(1200)), x, False),
        ("ExperimentalTwoLevelFeatureFreq",
         mojolearn.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(coded, y), coded, False),
        ("GradientBoosting", mojolearn.GradientBoosting(
            n_estimators=4, max_depth=3, loss="Logloss", score_function="L2", use_pointwise_searcher=True,
            bootstrap_type="Bayesian", bagging_temperature=0.5, boost_from_average=True, od_type="Iter",
            od_wait=2, use_best_model=True).fit(
                x, yc, sample_weight=rng.uniform(0.5, 1.5, 1200).astype(np.float32), eval_set=(xe, ye)), x, True),
        ("GradientBoosting", mojolearn.GradientBoosting(
            n_estimators=3, max_depth=3, loss="Logloss", cat_features=[0], one_hot_features=[1],
            # the deterministic configuration the host's one-hot arm restates
            # (identity_break._gbdt_legacy_kw); the stochastic defaults refuse
            bootstrap_type="No", random_strength=0.0).fit(coded, yc),
         coded, True),
    ]
    with tempfile.TemporaryDirectory() as d:
        for i, (name, est, xs, proba) in enumerate(fits):
            path = os.path.join(d, f"m{i}.npz")
            est.save(path)
            host = host_model(path)
            assert isinstance(host, HostGBDT) and host.estimator == name, (type(host), host.estimator)
            assert np.asarray(host.predict(xs)).tobytes() == np.asarray(est.predict(xs)).tobytes(), name
            if proba:
                assert np.asarray(host.predict_proba(xs)).tobytes() == np.asarray(est.predict_proba(xs)).tobytes()


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
