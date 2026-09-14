# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Re-export. The KDE host oracle SHIPS in the estimators CPU host binding
(python/mojolearn/host_surface.py names it), so it lives in
`kde/host/kde_oracle.mojo` since the host surface manifest lane, 2026-09-14.
kde/checks/kde_check.mojo keeps resolving through this file; new code
imports `kde.host.kde_oracle` directly."""

from kde.host.kde_oracle import (
    KDE_ORACLE_HOST_SABOTAGE,
    KdeOracleStages,
    ORACLE_FLOAT32_MIN_BITS,
    ORACLE_LOG_FLOOR,
    _host_row_norm_halving,
    _host_row_norm_halving_sqrt,
    oracle_distance,
    oracle_log_kernel,
    oracle_logsumexp_row,
    oracle_naive_log_sum_row,
    oracle_score_samples,
    reference_distance_f64,
    reference_log_kernel_f64,
    reference_log_kernel_norm_f64,
    reference_score_samples_f64,
)
