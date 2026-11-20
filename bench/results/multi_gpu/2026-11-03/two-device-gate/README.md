# Two-device gate, 2026-11-03 (after the multi-GPU encapsulation, main 8ce0218bde)

`tools/two_device_checks.sh` on the shared RunPod pod nvc1 (2x NVIDIA L40S, sm_89, driver 595.91.07),
jobs nvc1-0002..0007 (each resumed the phases the previous one had not passed; status.tsv keeps every row,
the last row per phase counts). R2 tags: nvc-two-device-20261103T234255Z-7ffb8d (the Mojo checks) and
nvc-two-device-20261104T013504Z-308077 (the final run, this directory's files).

- The 12 `training/checks/*_parallel_check.mojo`: one device vs two, same bits: ALL PASS
  (cholesky, dbscan, dot, gmm, gp, gram_outputs, gram, iforest, pointwise, qr, resample, svm).
- `python -m mojolearn verify-distributed --devices 0,1` (`distributed.json`): 15 cells, every case
  (arima, holtwinters, gpc, gpc_fit, ivf) matches across one-device, two-device and reversed-device runs;
  10 transport-fault controls (drop_result, reverse_results) all detected. Status
  NUMERICAL_MATCH_EXECUTION_TRACE_OWED: the physical execution trace and native fault controls are the
  suite's own owed items, unchanged by this run.

Multi-GPU verification as a reference is retired (docs/TOMBSTONES.md, MOJOLEARN_PAR_VERIFY); this gate
is the comparison that remains. Run it after any change under a family's `multi_gpu.mojo`,
`core/device_plan.mojo` or a shard owner.
