# Broad verifier execution, 2026-09-17

Installed development wheel source 79eda7622; all 32 native families reused
byte-for-byte from the fresh 7f5b786ae host build. This batch changes selection,
reporting and Python routing declarations; it does not change native arithmetic.
The wheel runs outside the checkout with no Python/host-path overrides. Full
reports are losslessly gzip-compressed. The receipt identifies the exact wheel
and native binaries; the binary is archived externally by SHA-256.

- 17 supported CPU logical-shard drivers, base fixture twice: 49 IDENTICAL,
  17 N/A, 19 OWED; no DIVERGENT or REFUSED. INCOMPLETE is the expected outcome
  because CPU replay does not qualify physical multi-GPU execution.
- All 17 withheld CPU lanes, base twice: 49 IDENTICAL, 19 N/A, 17 OWED;
  no DIVERGENT or REFUSED. Qualification holds remain, including the 20 mapped
  appendix entries. Missing/stale references are not turned into passes.
- Pending Transformer with extended checks, base twice: eight OWED, one N/A;
  batch, step/full, gradient, batch-size, ragged and sampler/replay local
  comparisons pass while independent references remain owed.
- Four bundled GPU-trained models, loaded through the CPU saved-model door
  twice: eight IDENTICAL model/batch parts. Includes HostForest and HostGBDT.
  The training self-test is explicitly not run in models-only mode.
- 246 focused tests pass, including refusal/mismatch behavior with absent
  references, stale-lane accounting, CPU shard admission and repeated model
  instability. Initial test collection without native paths failed and its
  log is retained; the corrected test setup uses the frozen host build.

Coverage still reports 162 default-available, 17 withheld, 50 excluded from
default CPU qualification. Execution scope is now distinct: 196 declared CPU
routes, including 17 logical-shard drivers; 33 parallel drivers still need
GPU execution. Both saved-model appendix entries name their installed command.
No all-nine qualification, native fault rerun, physical GPU test, release
qualification or PyPI upload is claimed by this bounded integration check.
