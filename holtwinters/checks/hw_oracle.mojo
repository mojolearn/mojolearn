# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Re-export. The Holt-Winters host oracle SHIPS in the tsa CPU host binding
(python/mojolearn/host_surface.py names it), so it lives in
`holtwinters/host/hw_oracle.mojo` since the host surface manifest lane,
2026-09-14. holtwinters/checks/hw_check.mojo keeps resolving through this
file; new code imports `holtwinters.host.hw_oracle` directly."""

from holtwinters.host.hw_oracle import (
    HW_ORACLE_HOST_SABOTAGE,
    HWOracleFit,
    oracle_eval,
    oracle_fit,
    oracle_forecast,
    oracle_sse_at,
)
