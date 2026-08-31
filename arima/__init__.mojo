# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""arima: cuML `cpp/src/arima/` -- the batched Kalman filter log-likelihood,
prediction/forecast, and the finite-difference gradient. `arima/derived/`
mirrors their files one for one; `arima/original/` is what they never
needed (the host oracles, the hashed fixtures, the checks). DEVIATION 670
(their double is our Float32 on the device) is stated once in
`arima/derived/tsa/arima_common.mojo` and carried by every file here. See
`arima/README.md`.
"""
