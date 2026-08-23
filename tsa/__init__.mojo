"""tsa: cuML `cpp/src/tsa/` (the stationarity test and the differencing-order
choice of auto_arima) and the `src_prims/timeSeries/` primitives they call.

`tsa/ported/` mirrors their files one for one; `tsa/mojo_only/` is what they
never needed (the host oracles, the hashed fixtures, the checks). See
`arima/README.md` (one README covers both lanes) and `tsa/README.md`.
"""
