# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Re-export. The coordinate descent host oracle SHIPS in the solver CPU host
binding (python/mojolearn/host_surface.py names it), so it lives in
`solver/host/cd_oracle.mojo` since the host surface manifest lane,
2026-09-14. solver/checks/cd_check.mojo, solver/cd_main.mojo and the bench
mains keep resolving through this file; new code imports
`solver.host.cd_oracle` directly."""

from solver.host.cd_oracle import (
    CdOracleResult,
    ORACLE_SQUARED_GUARD,
    SAB_NO_FTZ_RESID,
    cd_oracle_fit,
    cd_reference_f64,
    fixture_denormal_residual,
    fixture_nonfinite_labels,
    fixture_planted_sparse,
    fixture_signed_zero,
    oracle_sabotage_name,
    planted_w,
)
