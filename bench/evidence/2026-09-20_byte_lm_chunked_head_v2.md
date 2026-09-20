# ByteTrainer chunked LM-head v2 integration

- Date/host: 2026-09-20, macOS 26.5.2 arm64, local Metal
- Numeric mode: IDENTICAL
- Build concurrency: `MAX_JOBS=2`
- Gate: `training/checks/byte_lm_head_v2_integration_check.mojo`
- Small shape: B=1, L=8, DM=16, one layer, V=513
  - Representative V1/V2 chunk-GEMM step: 15.570/22.309 ms
  - V1 retained head-path cells: 9,402; V2: 2,053
- Larger shape: B=1, L=64, DM=64, one layer, V=8,192
  - Scalar V2 baseline before this change: 232.226--241.529 ms
  - Chunk-GEMM V2 raw repeats: 128.988, 149.995, 122.362, 157.341 ms
  - Same-process V1 raw repeats: 17.755, 44.048, 25.086, 21.812 ms
  - V1 retained head-path cells: 1,323,009; V2: 16,389
  - Persistent head-path saving: 5,226,480 bytes
- Whole gate maximum RSS: 94,961,664 bytes; reported peak footprint:
  151,421,648 bytes. Both trainers coexist during this gate, so this is not a
  per-trainer device-memory claim.

Two independently constructed V2 trainers produce identical loss and updated
parameter bits. V2 and V1 loss differ by at most 2e-5 on both shapes. The V2
configuration asserts that only one rows-by-256 logits chunk is retained and
CE exponentials/dlogits remain one-cell placeholders. The direct device gate
also matches the CPU oracle bit-for-bit for all loss/statistic/gradient cells.

The tuned chunk path improves scalar V2 by 32--49% across the raw large-shape
samples. V2 remains deliberately opt-in because it is still slower than V1;
this qualifies a bounded-memory speedup, not a new throughput default.
