# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The 0.8.47 smoke refusals fixed by lane smoke-mappings (2026-10-15).

    .pixi/envs/test/bin/python -m pytest tools/test_smoke_mappings.py

1. algos/{rnn,lstm,gru}-{clf,reg}: ours' recurrent models have get_params (ParamsMixin), and
   their read-back must still carry the effective optimizer settings (betas, eps, weight_decay).
2. classical2/spectral: degree/coef0/kernel_params, absent from cuML, map only while ours holds
   None and both arms run the nearest_neighbors affinity.
"""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BP = _load("bench_board_params")

_MOD = "mojolearn._smoke_mappings_fake_rnn"


def _fake_recurrent(options):
    mod = types.ModuleType(_MOD)
    mod._OPT_DEFAULTS = {"adam": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)}
    sys.modules[_MOD] = mod

    class LSTMClassifier:
        def get_params(self, deep=True):
            return dict(hidden_size=16, optimizer="adam", learning_rate=1e-3, optimizer_options=options,
                        random_state=7)

    LSTMClassifier.__module__ = _MOD
    return LSTMClassifier()


def test_recurrent_get_params_read_back_carries_effective_optimizer_options():
    lib, source, got = BP.read_params(_fake_recurrent({"eps": 1e-7}))
    assert lib == "mojolearn"
    assert "effective optimizer options" in source
    assert got["betas"] == (0.9, 0.999) and got["eps"] == 1e-7 and got["weight_decay"] == 0.0


def test_recurrent_unknown_option_adds_nothing():
    _, source, got = BP.read_params(_fake_recurrent({"bogus": 1}))
    assert "betas" not in got and source == "get_params"


def _spectral(param, ours_value, ours_aff="nearest_neighbors", theirs_aff="nearest_neighbors"):
    ours = {param: ours_value, "affinity": ours_aff}
    theirs = {"affinity": theirs_aff}
    return BP.semantic_exception("spectral", param, "cuml-gpu", ours, theirs, "cuml")


def test_spectral_kernel_parameters_map_only_when_inactive():
    for param in ("degree", "coef0", "kernel_params"):
        assert _spectral(param, None)
        assert _spectral(param, 3) is None
        assert _spectral(param, None, theirs_aff="precomputed") is None


def test_spectral_every_ours_only_setting_is_mapped():
    """Every setting ours' SpectralClustering reads back and cuML's lacks maps at the raced values."""
    ours = dict(affinity="nearest_neighbors", assign_labels="kmeans", eigen_solver=None, gamma=None,
                degree=None, coef0=None, kernel_params=None, prediction_data=False, eigen_tol=1e-5,
                n_clusters=8, n_components=8, n_init=1, n_neighbors=10, random_state=42, n_jobs=None,
                verbose=False)
    theirs = dict(affinity="nearest_neighbors", eigen_tol=1e-5, n_clusters=8, n_components=8, n_init=1,
                  n_neighbors=10, output_type=None, random_state=42, verbose=False)
    report = BP.check("spectral", {"ours": dict(ours, __library__="mojolearn"),
                                   "cuml-gpu": dict(theirs, __library__="cuml")}, seed=42)
    assert report["verdict"] == "MATCHED", report["problems"]
    report = BP.check("spectral", {"ours": dict(ours, eigen_solver="arpack", __library__="mojolearn"),
                                   "cuml-gpu": dict(theirs, __library__="cuml")}, seed=42)
    assert any("eigen_solver" in p for p in report["problems"])
