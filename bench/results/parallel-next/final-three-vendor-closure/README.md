# Parallel reference closure after 0.8.10 publication

The current reference table covers all 59 parallel lanes across all nine fixtures. All 1,935 required numeric part-cells agree bitwise across AMD, Apple, and NVIDIA. The strict audit has zero missing vendor values, zero gaps, and no conflicts. The other 2,844 part-cells are declared not applicable and are not numerical evidence.

Of the 1,935 numeric part-cells, 771 also have a numeric CPU reference and match it exactly. The remaining 1,164 agree across the three GPU vendors but have no numeric CPU reference in this table. `cpu-reference-split.json` lists those missing CPU references by lane and part; this report does not claim CPU completeness for them. These are one-device cross-vendor records, not physical two-GPU qualification.

This final scoped admission adds the seven NVIDIA lanes listed in `summary.json`, closing the last 126 missing vendor values. The complete NVIDIA record contains 63 cells and 234 exact numeric reference comparisons. Earlier fresh AMD collection contains 126 cells and 459 exact comparisons; fresh Apple collection contains 225 cells and 837 exact comparisons. The comparison totals include values already present before admission.

## Evidence and source limitations

- NVIDIA: `bench/results/identity_break/2026-09-20_parallel-next-nvidia/`, Python source `14944378ff9d7a733cd39779c6bdb24b7d041e76`.
- AMD: `bench/results/identity_break/2026-09-20_parallel-next-amd935/`, Python/harness source `935b6f9046ac59b12d467a13c2ef4700e635bd5c`; the fresh complete run includes causal-batch cleanup and passed the unchanged 12 GiB memory guard. The earlier incomplete memory-stop record remains diagnostic evidence.
- Apple: `bench/results/parallel-next/apple-4459379a4/` and `bench/results/identity_break/2026-09-20_parallel-next-causal-cpu/` retain their original source and native provenance.

These measurements reuse native source `819a47ae48166e91951f54f54e64ee173658e32a`. AMD and NVIDIA used the published 0.8.9 Linux native wheel, SHA256 `b2f7856e5959a518ce0ebc23af0240e9092a9f3a5256faf77cc389565915638f`; Apple records retain their documented native provenance. Existing older reference columns remain identified by their original records. This is reference evidence, not qualification of a newly built wheel. The published 0.8.10 wheels, frozen release reports, and DOI are unchanged.

## Validation

`coverage.json` was generated with `tools/audit_parallel_coverage.py --fail-on-incomplete` (exit 0). The independent CPU/vendor split found zero disagreements. Admission and cross-vendor audit regression suites passed all 80 tests. All nonselected cells retain identical decoded values and record provenance; the count is recorded in `summary.json`.

Reference table SHA256: `e95fbbb863933bbd611fcb8a2e4722efdd3bf33f7d7cc78f179882c3abd0b7c9`.
