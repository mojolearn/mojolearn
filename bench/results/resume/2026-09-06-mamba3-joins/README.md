# Mamba3 L65 join attribution and strict inventory

This checkpoint analyzes already retained native arrays. It does not issue a
new native certificate, change tolerances, waive tensors, or measure speed.
All local work used one process at a time, BLAS threads=1 and `nice -n 10`.
No new AMD/NVIDIA rental or local model execution was needed.

## Evidence

The inputs are the corrected `eebd7c9206cef7a5fbc32f7f0ca9f71bdc68ec63`
AMD and NVIDIA runs:

- `bench/results/e1/2026-09-05_234154-mojolearn-e2-amd/diag/followup/mamba-long-cert/mamba3-l65`
- `bench/results/e1g/2026-09-05_195622-nvidia-mamba/remote/followup/mamba-long-cert/mamba3-l65`

`tools/mamba3_join_diagnostics.py` validates matching IDENTICAL native source
digests, case/objective metadata, oracle-file digests, finite values and
shapes. The diagnostic public tensors must also match the captured public
gradient digests. It reports all native tensor digests and both reference
comparisons, then evaluates 17 copies or left-associated additions with the
repository's signed-zero-preserving float32 FTZ convention.

Results in [retained-join-diagnostics.json](retained-join-diagnostics.json):

- All **86/86 native tensors** match between AMD and NVIDIA.
- All **17/17 join/copy outputs** reproduce bitwise from native inputs on
  each GPU, including eight tensors that fail the direct reference gate.
- `partial.join.kscale`'s S16 and S17 inputs each pass both independent
  float64 and staged float32 comparisons. Its two failing output cells
  therefore cannot be attributed to an incorrect final addition. Cancellation
  makes the inherited operand differences significant at the output scale.
- The five remaining failing outputs outside this simple-join trace are
  `partial.qkdot.gamma`, `partial.s15.scale`, `partial.join.s15.scale`,
  `partial.dt.current_total` and `partial.join.dt.current_total`.
  Exact forward operands were not retained, so the report does not claim to
  establish their arithmetic or rule out additional numerical bugs.

Exact arithmetic on native inputs does not establish those inputs' semantics.
Several upstream tensors still fail independent comparisons. The diagnostic
tool's successful exit refers only to its byte/trace assertions.

## Gate change and validation

The public Mamba3 gate previously allowed missing intermediate files or
unlisted diagnostic names. It now pins all **76 diagnostics** independently
of both manifests, in addition to the ten canonical public leaves. The two
oracle-only isolated beta decompositions remain outside the native inventory;
their complete contributions flow through the existing dt/trap joins.

`tools/test_mamba_gradient_oracle.py` passed **16 tests** in 0.495 seconds.
The new controls remove each required diagnostic from both manifests, delete
a declared diagnostic file, corrupt a diagnostic, and make a wrong public
gradient agree with the staged reference while disagreeing with the
independent whole-forward float64 reference.

Both retained arrays were replayed against the tighter gate at unchanged
`rtol=1e-5, atol=1e-6`. Each still fails exactly the original **13 numerical
comparisons**; neither has an inventory failure. See
[verification.json](verification.json), [AMD log](amd-strict-replay.log) and
[NVIDIA log](nvidia-strict-replay.log). These are read-only replays, not new
device executions. The long certificate remains **RED**.

## Next numerical step

Retain exact forward `rot.k`, biased Q/K, dt and sigma operands, together
with independent forward references. Use them to distinguish input error
from pinned FMA reduction error at S14/S15 and to trace the two current-dt
joins. Preserve all existing failing tensors and unchanged tolerances until
the numerical implementation or a fully justified arithmetic contract is
validated on AMD and NVIDIA.
