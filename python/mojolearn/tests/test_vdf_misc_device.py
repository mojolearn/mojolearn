# SPDX-License-Identifier: Apache-2.0
"""The `device=` boundary of the remaining public functions and explainers (lane vdf-misc).

Words and exception types only (no numbers: the digests are verify's). For every converted function or
constructor: the signature has a keyword-only `device` defaulting to None; an unknown word (`'tpu'`, `0`) raises
ValueError in the shared words before any work. Inputs are tiny.
"""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn import hdbscan, manifold
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
    "TreeExplainer": (lambda d: mojolearn.TreeExplainer(_forest(), _X, device=d), mojolearn.TreeExplainer.__init__),
    "KernelExplainer": (lambda d: mojolearn.KernelExplainer(_forest(), _X, device=d),
                        mojolearn.KernelExplainer.__init__),
    "PermutationExplainer": (lambda d: mojolearn.PermutationExplainer(_forest(), _X, device=d),
                             mojolearn.PermutationExplainer.__init__),
    "RFE": (lambda d: mojolearn.RFE(mojolearn.Ridge(), n_features_to_select=2, device=d).fit(_X, _Y.astype(np.float32)),
            mojolearn.RFE.__init__),
}

def test_spectral_embedding_cpu_is_never_refused_for_the_device():
    # Every installation: "cpu" runs the CPU images (beside a GPU set: the x_decomp, metrics and base CPU images;
    # CPU-only: its own set). The tiny input may be refused for its own reasons, never for the device.
    try:
        manifold.spectral_embedding(_X, n_components=2, device="cpu")
    except Exception as error:
        assert "device=" not in str(error)
