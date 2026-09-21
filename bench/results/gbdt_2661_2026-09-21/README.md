# GBDT DEVIATION 2661 H100 receipts

These compact receipts cover the 2026-09-21 NVIDIA H100 trial of
`MOJOLEARN_2661_NONSYM_GROUP_WIDTH=1` against the current default. The source
was commit `2e6c8bbf949ff1add71e235225f69505e96f6bec`; the box reported NVIDIA H100
80GB HBM3 and driver 580.126.09. The run used IDENTICAL mode and exactly the
R2-staged Taxi and Istella-S objects named in `timings_quality_compact.json`.

`timings_quality_compact.json` retains all five outer-pass medians, inclusive
quartiles, raw 25-fit ranges, five paired process-level ratios, fixed-output
hashes, quality observations, dataset object provenance, and the build/run
protocol. Each outer pass alternated process order and each process excluded
one harness warmup before five timed fits. Its headline ratios use the median
of all 25 recorded fits per arm, matching `flip_verdict.py`.

`identity_compact.json` retains the hashes for all 18 two-repeat train cells,
all 18 infer, model, and batch columns, and both eight-cell sub-byte matrices.
The small original verdict, binary hash, GPU, and status receipts accompany
those deterministic projections. Source logs remain outside the repository at
`/Users/andrewhendel/mojolearn-evidence/2026-09-21_gbdt_2661/results-2661/`.

The result rejects default promotion: depthwise regressed on Istella-S and its
two-dataset geomean was 1.002014. Lossguide's 0.990590 geomean and the
four-cell aggregate of 0.996286 are too small and noisy to justify enabling a
shared switch whose other affected policy regressed.
