# Handoff: fixed15 integration tests on existing resources

## Start here after compaction

Work in `/Users/andrewhendel/mojolearn-wt/lowbit-e2e`, branch
`lane/lowbit-e2e`. Do NOT work in the original cwd
`/Users/andrewhendel/CascadeProjects/mojolearn`: another session owns it and
it was on `lane/lm-attention-fallback`, not main.

Read this document and `docs/lanes/progress/lowbit-e2e.md`. Check git status
before any edit. Preserve other sessions' changes. The branch is pushed to
origin. Before this handoff, its latest commit was f5970e2d2; the submitted
experiment source is 4c875f5ba and must remain that snapshot remotely.

The user authorized creating this worktree and starting/queueing tests on
EXISTING resources. They accept an Apple slowdown as a documented tradeoff.
They want useful speed, quality within the agreed 1% metric threshold, and
bitwise identity across vendors, NOT equality to the old FP32 profile.
This turn only prepares the handoff; no additional jobs were submitted.

## Different kernels: yes. Different arithmetic: only by profile

- Select different kernels per operation, shape and vendor when they produce
  the SAME bits under the selected profile. Small decode can use a direct
  kernel, prefill a tiled/fused kernel; Apple can use exact floating chunks
  and NVIDIA/AMD integer matrix units. Prove bounds and compare each path.
- A kernel implemented with floating-point instructions is permissible if
  it computes the prescribed quantized product exactly and uses the same
  reconstruction/rounding. Its instruction type does not define the profile.
- Do NOT dispatch to the old, unquantized FP32 computation on a slow shape or
  vendor under the name fixed15_v1. That changes arithmetic and generally
  changes output bits. FP32 accumulation of widened 15-bit codes is not
  automatically equivalent to an exact integer dot product either.
- A mixed arithmetic profile can deliberately choose different arithmetic
  for different operation families. That mapping must be versioned, shared
  across vendors, reflected in checkpoints and evaluated as a complete
  configuration. Do not silently change the current mapping to chase timing.
- Current intended F1-pv32 mapping: projections, output head and QK^T use
  fixed15; PV, normalization, softmax, RoPE, activation functions, residuals
  and the other specified seams remain controlled FP32. PV stays FP32 so
  changing the key span does not change its quantization and break decode
  versus prefill equality.
- numeric_profile is separate from numeric_mode (fast/identical). Do not
  change defaults, merge main, or release as part of this test task.

## What exists and what does not

The branch started from `lane/lowbit-merged` at b9c5a2673. It includes the
15-bit kernels, linalg bindings and numeric-profile selector. In this
snapshot the profile registry has inference=False and training=False.

`lane/lowbit-blocks` owns model integration in
`/Users/andrewhendel/mojolearn-wt/lowbit-blocks`. Last inspected commit was
d48b2d4d6 (the integration plan), with active uncommitted edits in the
transformer oracle/model and a new int15_block.mojo. We neither copied nor
modified them. Inspect for a new committed revision on restart; do not
overwrite this lane or independently reimplement its work.

Known integration tradeoffs in that lane's plan: force eager attention
under fixed15, bypass the resident generation path initially, and upload
head planes on each call. Measure their impact; GEMM wins can be lost here.

Quality numbers in the conversation are OTHER lanes' reference results,
not measurements from this branch: inference -0.0020% / +0.0034% perplexity
change on English/code for F1-pv32, and encouraging five-seed training.
Do not interpret the seed spread as a confidence bound or claim better
quality. Native model training remains unsupported, including the known
contracted-length limit at large-vocabulary backward products.

## Jobs and actual results

### H100 NVL, existing nvc3

- Job `nvc3-0031`, cap 45 minutes, lane `lowbit-e2e`.
- Still QUEUED with empty log at handoff status check on 2026-09-29.
- Remote tree `/root/mojolearn-lowbit-e2e`, source marker
  `.lowbit_e2e_commit` = submitted commit 4c875f5ba.
- Do not sync this remote tree while the job is queued or running. New local
  commits after submission are docs/model-runner work, not changes to its job.

From this worktree:

```
sh tools/nvidia_central.sh status nvc3-0031
sh tools/nvidia_central.sh log nvc3-0031 60
```

When done, obtain RESULT_DIRECTORY from its log and fetch that exact path
with `sh tools/nvidia_central.sh fetch lowbit-e2e <remote-path> <local-dir>`.
If it fails, inspect the failure; do not blindly rerun or cancel other jobs.

### AMD MI325X, existing do-amd

Request `1790658242761-speed-lowbit-e2e-4c875f5ba3` PASSED, completed
2026-09-29T05:04:32Z. Raw evidence is committed at
`bench/results/lowbit_e2e/2026-09-29/amd-public-api/`.

fixed15/FP32 median call times: attention projection 1 token 0.7535;
8 tokens 0.7843; 512 tokens 0.5987; feed-forward down 512 tokens 1.3563;
output head 1 token 0.6655. Smaller than 1 means less time.

These are SYNTHETIC PUBLIC API diagnostics with dispatch, allocation,
transfers and activation conversion included. Weight planes are packed
once on host, uploaded each call; not persistent GPU weights. Five timed
samples per arm, alternating order, warmed operations. No whole-model,
quality, corpus-based speed or training-step claim follows from these rows.
Cold first weight packing includes initialization; do not compare its time
to later warm pack rows as equivalent measurements.

## Added tools and their validation limits

- `tools/lowbit_e2e/job.sh`: isolated queue entry; build linalg, run diagnostic,
  preserve unique output directory. Does not allocate machines or enable a
  profile. Source used by both submitted jobs: 4c875f5ba.
- `public_probe.py`: checks packed/raw equivalence, sampled independent
  integer reference, row/batch invariance, repeated bits, and comparator
  sensitivity to a flipped bit. The comparator test is NOT a kernel sabotage
  test. Existing kernel sabotage evidence belongs to its source revision.
  AMD executed all five shapes successfully. H100 result pending.
- `model_probe.py`: PREPARED, not executed on a model. Remote syntax parse
  passed. Explicitly refuses a disabled profile (exit 3 with BLOCKED record),
  no registry monkey-patch. Loads one profile at a time, alternates order,
  warms same-shape calls, times stateful forward and actual step calls,
  saves full raw logits and prompt-token bytes, checks repeated hashes.
  Uses supplied continuation tokens for identical contexts across profiles;
  this is not free-running generation quality. Fresh-state allocation is
  excluded from latency. Does not yet compare prefill versus incremental
  decode or checkpoint resume; add those gates before calling it end-to-end
  certified. Add binary/profile readbacks and expected-logit comparisons as
  integration becomes available. Do not label same-device repeats as
  cross-vendor identity.

## Next actions, in order

1. Collect H100 job; compare input hashes and fixed15 output hashes with AMD
   by shape. Compare within-profile outputs only. Record raw timing samples,
   hardware and source. Preserve failures and report any discrepancy before
   further speed claims. No need to repeat established AMD work.
2. Inspect committed model-integration progress. Bring a reviewed committed
   revision into this branch only once available. Keep each queued snapshot
   immutable. Do not reuse old GEMM evidence as whole-model evidence.
3. Queue the model probe on existing NVIDIA/AMD resources after the profile
   is supported and bindings are built. Stage model/data from existing R2
   store only. English enwik8 and code Pile GitHub; SmolLM2-360M first.
   An unavailable binding/profile is BLOCKED, never silently FP32 or PASS.
4. Full logits across vendors, incremental/prefill equivalence, integration
   versus the quality reference, repeat stability, and named negative controls.
   Use available M2 Pro for Apple identity if still alive. Slowness is allowed.
5. End-to-end latency: prefill, actual stateful decoding, then user-facing
   generation. Separate allocation/load/cold-start from warm execution and
   report it. Complete mixed-profile quality on both texts and a task check.
6. Training is a separate phase: unsupported-profile refusal until backward
   integration, vocabulary-bound handling, full optimizer/state identity,
   resume tests, actual step timing and quality gates exist. Each backward
   product quantizes its own FP32 operands along its own contracted axis;
   never transpose/reuse forward codes as a shortcut.
7. If optimizing dispatch, prioritize operations that regress (currently AMD
   down projection in this public-call test), retaining identical fixed15
   arithmetic. No default flip or numerical-profile revision is authorized
   by this testing request.

## Operational constraints

Existing resources only. No rental, extension, termination or queue
cancellation. No new M3 Ultra jobs: it is scheduled for release. Do not run
builds/tests/benchmarks on the local laptop. NVIDIA FIFO through
`tools/nvidia_central.sh`, AMD/Apple steward through `tools/apple_steward.py`.
Read current resource availability; do not assume a pod survives restart.
Record missing capacity instead of provisioning it.

Read `~/mojolearn-evidence/lane_common_rules.md` and
`~/mojolearn-evidence/lowbit-units/brief_current.md` for operational details.
That external brief mentions a future default flip; it does not authorize
this test branch to perform one. User session scope is testing in isolation.
Do not spawn agents unless the user or applicable instructions request it.
Commit explicit paths, push this branch, preserve existing work. No claims
of novelty: integer decomposition and the relevant optimizations are known.
