# SPDX-License-Identifier: Apache-2.0
"""The `device=` boundary of the remaining public functions and explainers (lane vdf-misc).

Words and exception types only (no numbers: the digests are verify's). For every converted function or
constructor: the signature has a keyword-only `device` defaulting to None; an unknown word (`'tpu'`, `0`) raises
ValueError in the shared words before any work; a host-only function refuses `device='gpu'` naming itself, on every
installation. Inputs are tiny.
"""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn import ensemble, hdbscan, manifold
from mojolearn import resample as rs

_BAD = "device must be 'gpu', 'cpu' or None"
_X = np.arange(24, dtype=np.float32).reshape(8, 3) % 5
_Y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)


def _hdbscan():
    return mojolearn.HDBSCAN()


def _forest():
    return mojolearn.RandomForestRegressor(n_estimators=2)


# name -> (callable taking device=..., signature owner)
CALLS = {
    "chi2": (lambda d: mojolearn.chi2(_X, _Y, device=d), mojolearn.chi2),
    "f_classif": (lambda d: mojolearn.f_classif(_X, _Y, device=d), mojolearn.f_classif),
    "f_regression": (lambda d: mojolearn.f_regression(_X, _Y.astype(np.float32), device=d), mojolearn.f_regression),
    "r_regression": (lambda d: mojolearn.r_regression(_X, _Y.astype(np.float32), device=d), mojolearn.r_regression),
    "mutual_info_classif": (lambda d: mojolearn.mutual_info_classif(_X, _Y, device=d), mojolearn.mutual_info_classif),
    "mutual_info_regression": (lambda d: mojolearn.mutual_info_regression(_X, _Y.astype(np.float32), device=d),
                               mojolearn.mutual_info_regression),
    "resample.bootstrap": (lambda d: rs.bootstrap(_X[:, 0], n_resamples=9, device=d), rs.bootstrap),
    "resample.permutation_test": (lambda d: rs.permutation_test(_X[:4, 0], _X[4:, 0], n_resamples=9, device=d),
                                  rs.permutation_test),
    "resample.monte_carlo_integrate": (lambda d: rs.monte_carlo_integrate("sum", [0, 0], [1, 1], 16, device=d),
                                       rs.monte_carlo_integrate),
    "resample.resample_indices": (lambda d: rs.resample_indices(8, device=d), rs.resample_indices),
    "resample.resample": (lambda d: rs.resample(_X, device=d), rs.resample),
    "manifold.spectral_embedding": (lambda d: manifold.spectral_embedding(_X, n_components=2, device=d),
                                    manifold.spectral_embedding),
    "hdbscan.approximate_predict": (lambda d: hdbscan.approximate_predict(_hdbscan(), _X, device=d),
                                    hdbscan.approximate_predict),
    "hdbscan.membership_vector": (lambda d: hdbscan.membership_vector(_hdbscan(), _X, device=d),
                                  hdbscan.membership_vector),
    "hdbscan.all_points_membership_vectors": (lambda d: hdbscan.all_points_membership_vectors(_hdbscan(), device=d),
                                              hdbscan.all_points_membership_vectors),
    "kpss_test": (lambda d: mojolearn.kpss_test(_X[:, 0], device=d), mojolearn.kpss_test),
    "select_d": (lambda d: mojolearn.select_d(_X[:, 0], device=d), mojolearn.select_d),
    "connected_components": (lambda d: mojolearn.connected_components(np.eye(4, dtype=np.float32), device=d),
                             mojolearn.connected_components),
    "refine": (lambda d: mojolearn.refine(_X, _X[:2], np.zeros((2, 3), dtype=np.int32), 2, device=d),
               mojolearn.refine),
    "ensemble.catboost_auto_learning_rate": (
        lambda d: ensemble.catboost_auto_learning_rate("Logloss", 1000, 100, False, False, device=d),
        ensemble.catboost_auto_learning_rate),
    "host_model": (lambda d: mojolearn.host_model("vdf-misc-absent.npz", device=d), mojolearn.host_model),
    "host_predict": (lambda d: mojolearn.host_predict("vdf-misc-absent.npz", _X, device=d), mojolearn.host_predict),
    "host_predict_proba": (lambda d: mojolearn.host_predict_proba("vdf-misc-absent.npz", _X, device=d),
                           mojolearn.host_predict_proba),
    "TreeExplainer": (lambda d: mojolearn.TreeExplainer(_forest(), _X, device=d), mojolearn.TreeExplainer.__init__),
    "KernelExplainer": (lambda d: mojolearn.KernelExplainer(_forest(), _X, device=d),
                        mojolearn.KernelExplainer.__init__),
    "PermutationExplainer": (lambda d: mojolearn.PermutationExplainer(_forest(), _X, device=d),
                             mojolearn.PermutationExplainer.__init__),
    "HostGBDT": (lambda d: mojolearn.HostGBDT(loss="RMSE", text="", n_features_in=3, approx_dim=1, device=d),
                 mojolearn.HostGBDT.__init__),
}

#: Host by name: `device='gpu'` is refused naming the function on every installation.
HOST_ONLY = ("ensemble.catboost_auto_learning_rate", "host_model", "host_predict", "host_predict_proba", "HostGBDT")


@pytest.mark.parametrize("name", sorted(CALLS))
def test_signature_has_keyword_only_device(name):
    parameter = inspect.signature(CALLS[name][1]).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("word", ["tpu", 0])
@pytest.mark.parametrize("name", sorted(CALLS))
def test_unknown_word_refused(name, word):
    with pytest.raises(ValueError, match=_BAD):
        CALLS[name][0](word)


@pytest.mark.parametrize("name", HOST_ONLY)
def test_host_only_refuses_gpu(name):
    short = name.rsplit(".", 1)[-1]
    with pytest.raises(ValueError, match="device='gpu' is not available for " + short
                       + ": it has no GPU implementation"):
        CALLS[name][0]("gpu")


def test_spectral_embedding_cpu_beside_gpu_refused():
    # Holds on a GPU installation only: there `_mojolearn_x_decomp` has no CPU image connected beside the GPU
    # set, so device='cpu' is refused in the estimators' words. A CPU-only installation runs it.
    if mojolearn.vendor() == "cpu":
        # CPU-only installation: device='cpu' is this installation's own route, never refused for its device.
        try:
            manifold.spectral_embedding(_X, n_components=2, device="cpu")
        except Exception as error:  # the tiny input may be refused for its own reasons, never for the device
            assert "device=" not in str(error)
        return
    with pytest.raises(ValueError, match="device='cpu' is not available for spectral_embedding"):
        manifold.spectral_embedding(_X, n_components=2, device="cpu")
