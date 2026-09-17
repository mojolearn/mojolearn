# Public verification development evidence

Development diagnostics, not final release qualification. The `probe` path
excludes these columns from reference admission. Source diagnostics used
modified Python and prebuilt native libraries; their recorded Git HEAD is
the parent, not a claim that it contains those modifications.

The installed reports are complete gzip-compressed JSON. They came from an
external CPU-only development wheel with current Python and the packaged
harness, plus prebuilt host libraries from the earlier macOS candidate.
`wheel-receipt.json` identifies that wheel and checks its Python and harness
bytes against the source tested here. The wheel was not published.

- `select-d-cpu.json`: all nine fixtures twice, computation and batch stable.
- `select-d-metal.json`: base twice, computation and batch match CPU.
- `select-d-batch-sabotage.json`: deliberately perturbed batch probe catches
  a mismatch; exit 1. This is a harness control, not a native sabotage build.
- Installed coverage: all 246 appendix entries and additional lanes visible.
- Installed portable models: four models plus OLS verified, exit 0.
- Installed `select-d`, BPE/folds and optional batch checks: successful local
  computations with missing references, correctly nonzero (exit 5).
- Installed Mamba-1: standard batch and step/full references match; all four
  optional local checks pass, but missing hashes prevent an overall pass.

No CUDA, HIP or multi-GPU execution is claimed by these records.
