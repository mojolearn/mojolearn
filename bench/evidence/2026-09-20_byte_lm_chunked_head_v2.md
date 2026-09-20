# ByteTrainer chunked LM-head v2 integration

- Date/host: 2026-09-20, macOS 26.5.2 arm64, local Metal
- Numeric mode: IDENTICAL
- Build concurrency: `MAX_JOBS=2`
- Gate: `training/checks/byte_lm_head_v2_integration_check.mojo`
- Small shape: B=1, L=8, DM=16, one layer, V=513
  - V1 step: 13.224 ms; V2 step: 30.901 ms
  - V1 retained head-path cells: 9,402; V2: 6
- Larger shape: B=1, L=64, DM=64, one layer, V=8,192
  - V1 step: 37.217 ms; V2 step: 241.529 ms
  - V1 retained head-path cells: 1,323,009; V2: 6
  - Persistent head-path saving: 5,292,012 bytes
- Whole gate maximum RSS: 90,963,968 bytes; reported peak footprint:
  138,330,856 bytes. Both trainers coexist during this gate, so this is not a
  per-trainer device-memory claim.

Two independently constructed V2 trainers produce identical loss and updated
parameter bits. V2 and V1 loss differ by at most 2e-5 on both shapes. The V2
configuration asserts that logits, CE exponentials/dlogits, and the head/CE
workspaces are one-cell placeholders before executing the step.

V2 is deliberately opt-in. On this Apple GPU it is 6.49x slower at the larger
shape because the exact bounded-memory kernels recompute logits. This result
qualifies the memory path, not a throughput win; the default remains V1.
