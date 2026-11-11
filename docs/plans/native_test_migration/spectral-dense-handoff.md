# Spectral dense native fixtures

CPU14 and GPU8 are authored and unrun. The CPU suite covers the six original
COO scans plus six original dense nearest-neighbor affinity graphs and two
negative/NaN refusals. The GPU suite runs the same eight affinity graphs in
order; it does not recount shared host COO scans as GPU work. Existing kernels,
dimensions, seeds, exact oracles and public defaults are unchanged.

Use the two `check-native-spectral-dense-*` tasks or import the corresponding
`spectral_dense_{cpu,gpu}_suite.run_checks` in the integration bundle. Both need
`MOJOLEARN_NUMERIC_IDENTICAL=1`; CPU additionally needs `MOJOLEARN_COLUMN_CPU=1`.
Run from the repository root so saved fixtures resolve. No new product API or
binding group is needed. The separate shared-context concurrency failure stays open.

Reference preparation used 17 independent process jobs with all 10 host cores.
It saved49 artifacts/334316 bytes, including full original150-row embedding,
120-row prediction, duplicated240-row, resident180-row and assignment inputs
for the forthcoming typed API. Byte checks against source-extracted test data
and independent oracle functions passed for48 artifacts; all49 hashes match.
Host120 data was source-reviewed; no fitted output was generated. Complete
logs and static checks are under `E/pca-native-tests/spectral-*`.

Nothing removed: old Python numerical tests remain until root accepts these
native replacements. Their sparse refusal, public packing and validation seams
stay in Python. No product/kernel, algorithm, benchmark or timing changes.
Full Spectral fit/query/state coverage awaits the pilot's concrete backends;
this handoff covers only the existing dense preparation primitives.
