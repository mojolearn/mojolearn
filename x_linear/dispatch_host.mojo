# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column's fit entry (the CPU-only install and the host binding,
bindings/_mojolearn_x_linear_host.mojo).

lane debt-host-call (2026-10-21): `fit_dispatch` moved here verbatim out of
x_linear/dispatch.mojo, which the device binding imports. The fits it runs
keep their row passes on the host thread pool (`par_rows`,
x_linear/rows_host.mojo) and live in the `*_host.mojo` modules of x_linear/;
the device binding never called `fit_dispatch` (every device fit runs on its
grid driver, x_linear/device.mojo `fit_device`), so no GPU-reachable module
imports host threads any more.
"""
from x_linear.ops import FP, IP
from x_linear.team import Team
from x_linear.bayes_host import ard_fit, bayes_ridge_fit
from x_linear.cd import enetcv_fit
from x_linear.glm_host import glm_fit
from x_linear.huber_host import huber_fit
from std.sys.info import is_gpu
from x_linear.isotonic import isotonic_predict
from x_linear.lars import lars_fit
from x_linear.logcv_host import logcv_fit
from x_linear.quantile_host import quantile_fit
from x_linear.ridge_host import ridge_fit
from x_linear.ridgecv import ridge_kfold_fit
from x_linear.sgd_host import sgd_fit
from x_linear.dispatch import (
    ALGO_ARD,
    ALGO_BAYES,
    ALGO_ENETCV,
    ALGO_GLM,
    ALGO_HUBER,
    ALGO_ISOTONIC,
    ALGO_ISOTONIC_PREDICT,
    ALGO_LARS,
    ALGO_LOGCV,
    ALGO_QUANTILE,
    ALGO_RIDGE,
    ALGO_RIDGE_KFOLD,
    ALGO_SGD,
    team_fit,
)


def fit_dispatch(t: Team, algo: Int, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """`t` is the team the fit runs on (a team of one for a fit that
    `team_fit` does not name)."""
    if algo == ALGO_SGD:
        # the device binding runs SGD, GLM and isotonic on the grid
        # (x_linear/device.mojo); no device thread reaches these
        comptime if not is_gpu():
            sgd_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_GLM:
        comptime if not is_gpu():
            glm_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_HUBER:
        huber_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_BAYES:
        bayes_ridge_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_ARD:
        ard_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_LARS:
        lars_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_QUANTILE:
        # the device binding runs QuantileRegressor on the grid
        # (x_linear/quantile_grid.mojo); no device thread reaches this
        comptime if not is_gpu():
            quantile_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_RIDGE:
        ridge_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_ENETCV:
        enetcv_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_LOGCV:
        logcv_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_ISOTONIC:
        # the host binding calls `isotonic_fit_host` (x_linear/isotonic_host.mojo)
        # and the device binding `_iso_fit_grid`; nothing reaches this arm
        pass
    elif algo == ALGO_ISOTONIC_PREDICT:
        comptime if not is_gpu():
            isotonic_predict(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_RIDGE_KFOLD:
        # the device binding runs this on the grid (x_linear/device.mojo)
        comptime if not is_gpu():
            ridge_kfold_fit(x, y, n, d, ip, fp, res, fw)
