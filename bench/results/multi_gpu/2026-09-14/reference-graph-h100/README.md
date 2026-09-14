# Reference capacity and graph rows: two H100s

RunPod `4ra98lfm0pqum0`, two H100 80GB HBM3 GPUs, IDENTICAL, NVIDIA
`sm_90a`. All compilation and model execution occurred on the pod. Inputs came
from the existing R2 enwik8 corpus (SHA recorded in each report).

Results:

- 57 reference-sharded neighbor cases pass: five distances, duplicate references,
  ragged shards, k exceeding shard length, single/multi-output classification and
  regression, original weighting/voting, one-GPU logical replay, atomic refusal,
  and overflow/NaN selection and host ordering.
- The previous 64 whole-query cases pass with the exact previous report/output
  hashes after extracting the native vote half (`golden-comparison.log`).
- A 96 GiB logical index (100,663,296 rows, 256 features) passes the exact
  257-row GPU oracle. Sixteen 6 GiB shards visit both GPUs; the nearest reference
  is the final global row. Each device has 85,520,809,984 bytes of VRAM. The
  full host matrix remains allocated; this is host-staged index capacity, not
  all-resident VRAM pooling. A 2 GiB case also passes before and after the final
  ordering correction. Timings are diagnostic, not scaling measurements.
- 15 graph cases pass: full hierarchy children/labels and diagnostics;
  spectral embedding/labels for neighbor and precomputed affinity; UMAP fitted
  embedding/retained transform state and transform output, with and without
  negative samples; failed-fit publication checks.
- Six raw-array cases compare every pairwise-distance bit and KNN distance/index
  bit at 3/17/65 rows and 3/129 features, with and without square root.

Provenance:

The pod started at `f8b5f68a23073fe5cc047cfe06052f7c51b5b570`. Native reference
vote sources are in `reference-out/reference-source.tgz`; the base binary SHA
is in `binary.sha256`. The initial 56-case receipts are preserved separately.
The 96 GiB process imported the driver before commit `e7f51c1cc` added the
native post-selection float insertion sort (needed for signed-zero/NaN order).
Its exact Python sources were frozen in `capacity-executed-source.tgz` before
later uploads. That fixture has finite positive distances, for which the added
sort retains the existing order. The final 57-case and 2 GiB receipts exercise
the corrected driver, with source hashes in `final-source.sha256`.

`graph-out/graph-source.tgz` records the graph implementation at `224e6c194`;
metrics/solver binaries and build logs are included. The final source overlay
through `d6854778c` is `reference-out/reference-graph-final-source.tgz` (apply to
the initial commit). Graph gate corrections removed a check for an attribute
that the original precomputed estimator never sets, and corrected the gate's
Mojo bitcast spelling. The initial raw-gate compiler error is retained. No
model arithmetic assertions were removed. Both final jobs returned zero.

Later main changes merged at `801aea993` concern Mamba, identity tooling and
existing lane evidence; they do not change these graph/reference implementations.
This receipt does not requalify those independent changes.

Scope excludes new cross-vendor claims, speedup, eight-GPU runs, RBC/radius
reference partitioning, distributed target tables, root graph/solver memory
pooling, and neural model/optimizer memory pooling.
