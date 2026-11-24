# SPDX-License-Identifier: Apache-2.0
"""Kernel methods public boundary (KernelRidge, Nystroem, RBFSampler): refusals in words.

The numerical answers of these estimators on each side are verify lanes (kernel-ridge*, nystroem*, rbf-sampler*,
x-neighbors-krr-options); what stays here is what a digest cannot see: which inputs are refused, and with what words.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend
from mojolearn.kernel_methods import KernelRidge


# ---------------------------------------------------------------- moved from test_cpu_kernel_variants.py (vdv-pytest-a g2)

@pytest.mark.parametrize("degree", [-1, 33])
def test_host_rejects_out_of_range_polynomial_degree(degree):
    if _backend._CPU_ONLY is None:
        pytest.skip("requires a CPU-only package")
    X = np.eye(3, dtype="float32")
    with pytest.raises(Exception, match="degree"):
        ml.KernelRidge(kernel="poly", degree=degree).fit(X, np.ones(3, dtype="float32"))


# ---------------------------------------------------------------- moved from test_krr_options.py (vdv-pytest-a g2)
# KernelRidge sample_weight and kernel='precomputed' (lane x-neighbors-krr-options): a precomputed kernel that is
# not square, a negative weight and a weight of the wrong length are refused.

def _krr_options_data(n=96, d=5, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, size=(n, d)).astype(np.float32)
    y = (a @ rng.normal(size=d) + rng.normal(size=n)).astype(np.float32)
    return a, y


def test_kernel_ridge_option_refusals():
    a, y = _krr_options_data()
    with pytest.raises(ValueError, match="square"):
        KernelRidge(kernel="precomputed").fit(a, y)
    with pytest.raises(ValueError, match="sample_weight"):
        KernelRidge().fit(a, y, sample_weight=-np.ones(len(y)))
    with pytest.raises(ValueError, match=r"sample_weight.shape == \(3,\), expected"):
        KernelRidge().fit(a, y, sample_weight=np.ones(3))
