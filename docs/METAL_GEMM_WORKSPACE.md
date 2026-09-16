# Transformer GEMM workspace on Metal

Small transformer tests repeatedly allocated GEMM scratch, launched work,
then synchronized to release that scratch. The allocation owner forced a
host round trip even when the next operation only needed to enqueue work
on the same stream. The earlier wait-removal change correctly retained these
lifetime fences; deleting them without changing ownership would be unsafe.

Forward and backward transformer stages now own a `GemmWorkspace`. Its `run`
method calls the existing `identical_gemm_into` with the same operands,
orientation, numeric-mode policy and shape. Before each call it asks
`identical_gemm_workspace_max_floats` for the current dispatcher's requirement.
This is important for growing decode shapes and split-K plans. When capacity
must grow it synchronizes before replacing the old allocation. Otherwise,
reuse adds neither an allocation nor a completion wait. Plan-internal scratch
and waits are unchanged. The synchronous public `identical_gemm` remains for
callers that do not retain workspace, including the RMSNorm gradient helper.

The workspace and operands must stay alive through the caller's completion
wait, and must use one in-order context/stream. Existing stage owners already
retain the other asynchronous buffers this way. Scratch is retained at its
largest observed size until the stage owner is destroyed; this trades some
longer-lived device memory for fewer allocations and waits. No arithmetic,
reduction order, fixture size, or input-validation check was removed.

## Validation

Run the new growth/reuse gate with:

```
bash tools/mac_slot.sh metal pixi run check-gemm-workspace
```

It queues two increasing split-K shapes followed by reuse of the larger
workspace, checks all three operand orientations against the host oracle
(9 products, 4,608 cells), and requires no added allocation or wait on reuse.
The check enables counters without enabling synchronized phase timing. A
scratch negative-control build replaces the last reused call with the old
synchronous `identical_gemm` entry; it fails with "reused workspace must not
allocate or wait", proving that the regression assertion detects the old cost.

The transformer backward gate passed 17 cases, all 37 stages and 412,172
cells against the bitwise host oracle. Its optional clauses B through F also
passed: repeatability, batch and length behavior, query chunking, refusal,
signed-zero checks and the existing exact-arithmetic controls.

The forward gate passed 17 oracle cases, 30 stages and 349,206 cells,
and optional clauses B, C and D,
including decode/prefill equivalence and all three sliding-window cases.
The optional forward refusal clause E fails on BOTH the unchanged baseline
and candidate: a planted weight is rejected by `LlamaDeviceWeights` before
the test reaches its exception handler around the forward call. That is a
pre-existing test-harness failure, not a passing gate. Both failure logs are
retained with the successful logs.

Validation here is Apple M4 / Metal / IDENTICAL. CUDA and HIP are
**cross-vendor-pending** for this change. Existing arithmetic kernels and plan
selection are unchanged; that alone is not a cross-vendor test result.

## Timing protocol

Baseline source: `c3d0cd422`. The same Mojo environment built both backward
check binaries with `-j 2` and `MOJOLEARN_NUMERIC_IDENTICAL=1`. Timed binaries
do not enable step-phase instrumentation, and the runtime timing variable is
unset. Each run includes the same 17-case oracle gate and its existing first
case trace. All 37-stage trace hashes match between arms.

The first six-run comparison overlapped a two-worker CPU compilation and is
retained as `timings.json`, not used for the final timing claim. The final
comparison runs without a concurrent build from this task, under one exclusive
Metal lease, in baseline/candidate/candidate/baseline/baseline/candidate order.
These are test-process wall times, not model throughput or a prediction of
full certification-suite speed. Source dispatch and workspace reuse are the
optimization; no timing threshold is used as a correctness gate.

Raw binaries, build logs and runner scripts are in
`~/mojolearn-evidence/metal-gemm-workspace/`. Committed measurements and checks:
`bench/results/metal_gemm_workspace/2026-09-16-apple-m4/`.


## Measured result, 2026-09-16

| Arm | Process seconds, three runs | Median seconds |
| --- | --- | ---: |
| Baseline | 9.720, 9.183, 9.847 | 9.720 |
| Retained workspace | 4.854, 7.891, 4.759 | 4.854 |

The median was 50.1% lower in this bounded backward test. Every candidate
sample was below every baseline sample, but the candidate spread is large;
this is a shared-machine measurement, not a stable throughput guarantee.
The earlier comparison during compilation had medians 12.641 and 10.416
seconds (17.6% lower), demonstrating why the absolute result depends on the
machine's state. Neither run establishes the cost of the full Apple column.

The unchanged 37-stage card SHA-256 in all twelve timed runs is
`a4166b19a0af95ecb9f4978ac8476f61035aa98474c2a449e4ffe97679ec68c8`.
`quiet-timings.json` includes each measured binary's SHA-256. The runner
`compare-quiet.py` expects the two binaries beside it and exits on a failed
oracle gate or mismatched card. Build each source tree's
`transformer/checks/transformer_backward_check.mojo` with `mojo build -j 2
-D MOJOLEARN_NUMERIC_IDENTICAL=1 -I TREE` into `baseline-backward` and
`candidate-backward`, then run the script under `tools/mac_slot.sh metal`.
