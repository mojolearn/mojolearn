# Spectral prediction and retained numerical state

CPU 6 and GPU 6 are authored and unrun. Import
`spectral_prediction_{cpu,gpu}_suite.run_checks`; dependencies are the same
actual backend/API and saved inputs as `07dd5500e4`, whose native helpers this
slice reuses. Both targets use the identical flag; CPU also uses the CPU column.

The original 120-row nearest-neighbor and precomputed graphs and duplicated
240-row graph retain every input, option and selected query index. Oracles
retain the original fit-with-state relation, label range, finite output,
eigenvalue threshold, row-order/duplicate consistency and three native state/
affinity refusals. No training-label or training-embedding report-only result
has become an assertion. No new cross-device or repeated-release identity run.

CPU graphs and independent single-query cases use host parallelism. GPU graphs
run in order on the physical GPU, with existing kernels still parallel. Raw
address owners are pinned through calls; outputs start with NaNs and have
guards so incomplete writes or refusal publication cannot masquerade as success.

Nothing removed: Python numerical checks remain until acceptance. Public
dtype/shape/conversion, serialization, installation/host-archive, model-field
refusal, readonly/rebinding and registry telemetry seams remain Python. Native
resident ownership follows the pilot's new state API separately. No product,
kernel, algorithm, defaults, tolerances, benchmark or timing changes.
