# Spectral native resident ownership

CPU 6 and GPU 2 are authored and unrun. The pure entry is
`spectral_state_suite.run_checks`; the actual device entry is
`spectral_resident_gpu_suite.run_checks`. Both use the identical flag.
Dependencies include the exact core state API from `c325e6e68f`, actual backend
`764e2ad158`, and the preceding native prediction helpers `911dcc3964`.
Do not collect local adoption `22185b2c8c` separately from the pilot's producer.

Pure cases exercise single/copied/table-held Arcs and quiet cleanup, including
preserving an original exception while release fails. Real owned counters
outlive every native callback. Actual device cases retain the original 180-row
resident workload and 120-row precomputed workload. Their native prepare call
must own copied storage after every caller fit/state buffer goes out of scope.
An active query survives cache retirement; final release restores registry
count, repeated release stays quiet and later query refuses the retired handle.
Resident labels match the original direct route; both embeddings must be finite,
but their bits are not asserted equal.

CPU ownership cases are parallel. GPU graphs are ordered with quiescent registry
observations on the physical GPU, retaining kernel parallelism and the separate
open shared-context concurrency failure. Nothing removed: original Python
coverage, readonly/rebound arrays, watcher/provider admission and public cache
telemetry remain. No product/kernel, algorithm, defaults, tolerances or timing
changes. Source-only evidence is `E/pca-native-tests/spectral-state-static.json`.
