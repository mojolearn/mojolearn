"""arima: cuML `cpp/src/arima/` -- the batched Kalman filter log-likelihood,
prediction/forecast, and the finite-difference gradient. `arima/ported/`
mirrors their files one for one; `arima/mojo_only/` is what they never
needed (the host oracles, the hashed fixtures, the checks). DEVIATION 670
(their double is our Float32 on the device) is stated once in
`arima/ported/tsa/arima_common.mojo` and carried by every file here. See
`arima/README.md`.
"""
