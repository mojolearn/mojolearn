# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tsa: cuML `cpp/src/tsa/` (the stationarity test and the differencing-order
choice of auto_arima) and the `src_prims/timeSeries/` primitives they call.

`tsa/impl/` has one file per reference file; `tsa/checks/` is what the
references never needed (the host oracles, the hashed fixtures, the checks). See
`arima/README.md` (one README covers both lanes) and `tsa/README.md`.
"""
