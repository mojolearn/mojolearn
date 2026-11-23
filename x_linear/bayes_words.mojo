# SPDX-License-Identifier: Apache-2.0
"""The result layout of BayesianRidge's and ARDRegression's fits (the
XL_OP_FIT words of ALGO_BAYES / ALGO_ARD) with their posterior, in one place
for the fit kernels (x_linear/bayes.mojo, x_linear/device.mojo,
x_linear/ard_grid.mojo, x_linear/x_linear_host_impl.mojo), the door's UNPACK
and PACK (bindings/x_linear_regressors_*.mojo) and the member
(core/xlinear_member.mojo). Plain integer arithmetic, no import.

BayesianRidge (`bayes_out_words(d)` words):
  coef d | intercept | alpha_ | lambda_ | n_iter | V d*d | ev d | X_offset_ d
ARDRegression (`ard_out_words(d)` words):
  coef d | intercept | alpha_ | lambda_ d | n_iter | dk | X_offset_ d | sigma_ dk*dk (room for d*d)

BayesianRidge keeps the FACTOR its fit already holds, an O(d^2) copy: V the
eigenvectors of the centered Gram (eigenvector m in column m) and ev the
clamped eigenvalues the iterations used. scikit-learn's posterior covariance
(`_bayes.py`: `Vh_full.T @ (Vh_full / (alpha_ * eigen_vals_full +
lambda_)[:, None])`) is V diag(1 / (alpha_ ev + lambda_)) V'; predict
(return_std=True) reads the factor directly and the public sigma_ is formed
from it only when it is read (lane mf-iterimp2: no d^3 step on a fit).
ARDRegression's sigma_ is the last `_update_sigma` over the kept features
(dk of them), which its fit already holds: copied as it is. X_offset_ is the
fit's (weighted) column means (zeros without an intercept).
A caller that sends fewer words (an older request: d + 4, 2d + 4) gets no
posterior words: the fits write them only when the request holds them.
"""


def bayes_out_words(d: Int) -> Int:
    """BayesianRidge's result words with its posterior factor."""
    return d * d + 3 * d + 4


def bayes_v_at(d: Int) -> Int:
    return d + 4


def bayes_ev_at(d: Int) -> Int:
    return d * d + d + 4


def bayes_offset_at(d: Int) -> Int:
    return d * d + 2 * d + 4


def ard_out_words(d: Int) -> Int:
    """ARDRegression's result words with its posterior."""
    return d * d + 3 * d + 4


def ard_dk_at(d: Int) -> Int:
    return 2 * d + 3


def ard_offset_at(d: Int) -> Int:
    return 2 * d + 4


def ard_sigma_at(d: Int) -> Int:
    return 3 * d + 4
