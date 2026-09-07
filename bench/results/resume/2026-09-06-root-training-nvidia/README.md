# Root NVIDIA training qualification — 2026-09-06

## Admitted sub-result: independent single-block gradients

At frozen source `201e5ebcf58b388573426a2df77b1d5161fd156a`, root built and
ran the fixed B2/L8/DM32/V64, 13,376-parameter capture on a RunPod RTX 4090.
The independent PyTorch FP64 gate passed loss and every cell of all eleven
gradient tensors at the predeclared tolerances. Both sign reversal and a
forward-preserving SiLU-derivative error were detected. Parameters moved.

Native mean loss: 4.781259536743164. FP64 reference: 4.781259779946554.
See [oracle result](run2/remote/training-validation/gradient-oracle.json),
[native capture](run2/remote/training-validation/gradient-capture/capture.json),
[guarded capture log](run2/remote/training-validation/gradient-capture.log),
[guarded oracle log](run2/remote/training-validation/gradient-oracle.log),
[job statuses](run2/remote/training-validation/results.tsv), and
[frozen source](run2/source.tar.gz).

This is independent tolerance correctness for the existing single-block
fixture. It is not a two-block language-model, learning-quality, optimizer
reference, cross-vendor bitwise, or complete-campaign certificate.

## Retained failures and cleanup

- First run, `f30670da6f91c58347ec445c6ef358fb37e45fd1`: the new capture
  harness failed Mojo ownership checking before any model execution. Root
  replaced an implicit-copy tuple with explicit post-synchronization moves.
  [Build error](run/remote/training-validation/gradient-build.log).
- Second run, `201e5ebcf58b388573426a2df77b1d5161fd156a`: the gradient
  sub-result above passed. Compilation of the new MLP helper then failed
  because GPU kernel arguments used host `Int` rather than fixed-width
  integers. Root changed those arguments to `Int32`; validation of that
  repair remains open. [Build error](run2/remote/training-validation/training-build.log).

Both campaigns remain **incomplete/failed**, preserving their original exit
statuses. Both rentals were deleted with HTTP 204 and absence verified with
HTTP 404: first with 48 minutes remaining, second with 55. See
[first controller](run/controller.log) and [second controller](run2/controller.log).

## Resumed investigation

Third run, frozen source `9898d3e`, compiled both bindings and passed all
20 MLP surface/reference tests. Four numerical-edge tests passed; the two
overflow-refusal tests failed because they expected `RuntimeError` or
`ValueError`, while the native Mojo binding raised plain `Exception` with
`small MLP operation has nonfinite input or output`. The expected refusal
occurred; the test's exception-type assumption was wrong. See
[job statuses](run3/remote/training-validation/results.tsv) and
[edge failures](run3/remote/training-validation/mlp-numerical-edges.log).
Checkpoint captures were not reached. The run remains failed, and its
[teardown receipt](run3/teardown.txt) records termination.

The assertions now require that exact native error message. A fourth
serial NVIDIA campaign uses the prior isolated checkout plus this test
repair, frozen at `8f6ed41`; it does not certify unrelated working-tree edits.
All 18 jobs and fetched-evidence admission passed. The six numerical-edge
checks passed, including both exact overflow refusals and the independent
AdamW checks. The fixed MLP completed 16 steps, with loss falling from
1.1047444343566895 to 0.04506960138678551; the complete-state same-device
checkpoint continuation comparison passed. See
[job statuses](run4/remote/training-validation/results.tsv),
[controller and admission](run4/controller.log), and
[exact frozen source](run4/source.tar.gz).

This qualifies only the bounded NVIDIA integration and reference checks.
AMD/Metal equality and the new two-block language model remain separate.
Pod `w3w104ixxhhgpc` was deleted with HTTP 204 and absence verified with HTTP
404, with 39 minutes left on its 60-minute lease.

Root alone executed jobs, serially, under two-core/two-thread guards with
memory limits, deadlines and rental watchdogs. No Apple model/GPU execution.
Implementation agents did not execute tests, builds, models or measurements.
