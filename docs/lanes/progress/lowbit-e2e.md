# lowbit-e2e: integration evidence on existing resources

Restart entry point: [handoff](../../HANDOFF_LOWBIT_E2E_2026-09-29.md).

User authorization, 2026-09-29: create an isolated worktree and start/queue
tests on existing resources. Apple slowdown is an acceptable documented
tradeoff. No new rental, no default flip, no release or merge requested.

Worktree: `~/mojolearn-wt/lowbit-e2e`, branch `lane/lowbit-e2e`, initially
forked from `lane/lowbit-merged` at b9c5a2673. Kernel/flag changes are in this
snapshot. The other worktree `lowbit-blocks` contains active uncommitted
model integration; it is deliberately neither copied nor modified.

## First queued experiment

`tools/lowbit_e2e/job.sh` builds this snapshot's linalg binding, then runs
`public_probe.py`: actual public Python calls at decode/prefill projection
shapes with weight planes packed once. Allocation, transfers and activation
conversion are included. These are SYNTHETIC diagnostic rows, not model
speed or quality claims. Host-packed weights are not resident GPU weights.
It checks packed/raw equivalence, sampled exact integer outputs, row/batch
invariance, repeated output bits, and that its comparator catches a flipped
bit. That comparator check is not a replacement for the kernel lane's
existing source sabotage gates. All samples and input/output hashes persist.

The runner reports model inference/training availability separately. A
refused or unintegrated model is BLOCKED, never a successful FP32 fallback.
No experimental registry monkey-patch is used.

## End-to-end queue dependencies

1. Wait for a committed integration revision from `lane/lowbit-blocks`.
   Inspect/merge it into this test branch only; keep the source pinned for
   each submitted job. Do not resync a lane tree while a job uses it.
2. Build the changed model bindings. Compare full logits for the SAME
   fixed15 profile on NVIDIA, AMD and available Apple (M2 Pro); also compare
   prefill versus incremental decode and repeated calls. Compare against
   the agreed F1-pv32 reference: projections/head/QK use fixed15, PV stays
   FP32. Kernel equality alone is not this gate.
3. Stage SmolLM2-360M and the two corpora from the existing R2 store. Run
   explicit fp32_v1 and fixed15_v1 arms on the same prompts. Report time to
   first token and actual stateful per-token decode separately, with warm-up,
   alternating arm order, raw samples, input/model/source hashes and full
   output hashes. Never infer decode time by subtracting separate timings.
4. Confirm integrated quality on enwik8 and Pile GitHub against the same
   reference configuration; one task evaluation before public promotion.
5. Training remains separately BLOCKED: the current profile refuses it.
   After backward integration and vocabulary-limit handling, check full
   step state, optimizer state and cross-vendor checkpoint resume, then time
   real steps and evaluate matched-seed quality. Three separately timed
   products are not a training step.

## Resource policy

Use existing NVIDIA FIFO and AMD steward; no new Macs or pods. No new M3
Ultra jobs: the current brief schedules its release. Do not cancel anyone's
jobs, rerun established kernel evidence, or compete with running timings.
One speed job per physical box. No builds/tests/benchmarks on the laptop.

## Status

Submitted public API probe at 4c875f5ba:

- H100 NVL existing pod nvc3: job `nvc3-0031`, cap 45 minutes, exclusive
  one-GPU FIFO behind the existing kernel jobs.
- Existing MI325X steward: `1790658242761-speed-lowbit-e2e-4c875f5ba3`.
- Remote syntax compilation of the Python runner and shell parsing passed
  on the H100 host before submission. No local tests were run.

Prepared `tools/lowbit_e2e/model_probe.py` for the committed integration
revision. It records all prefill/decode logits on fixed token contexts,
checks repeat identity, alternates profiles, and times the actual stateful
step API. A disabled profile writes BLOCKED_NOT_INTEGRATED and exits 3.
Its model execution remains untested/unsubmitted until integration lands.
It deliberately avoids the older harness's generate-minus-prefill estimate.
Example after integration and staging, on a queued box:

```
PYTHONPATH=python pixi run -e test python tools/lowbit_e2e/model_probe.py \
  --model /root/models/SmolLM2-360M --prompts bench/model/prompts.txt \
  --out-dir /root/ev-lowbit-e2e/model-run-1
```

No profile default changed; no machine rented, extended, or released.

## First result: AMD public API diagnostic

The submitted AMD request finished PASS at 2026-09-29T05:04:32Z on the
MI325X. Source 4c875f5ba. Raw request verdict, output and stderr are retained
in `bench/results/lowbit_e2e/2026-09-29/amd-public-api/`.

Five alternating timed samples per arm, one run, warmed operations. Ratios
are fixed15 over fp32 medians on the same box and input, including public
binding transfers and allocation. Input generation is synthetic; these are
diagnostics, not publishable corpus/model speed claims.

| Public call | fp32 ms | fixed15 ms | fixed15 / fp32 |
|---|---:|---:|---:|
| attention projection, 1 token | 0.481098 | 0.362490 | 0.7535 |
| attention projection, 8 tokens | 0.476779 | 0.373940 | 0.7843 |
| attention projection, 512 tokens | 1.011759 | 0.605749 | 0.5987 |
| feed-forward down, 512 tokens | 1.430937 | 1.940777 | 1.3563 |
| output head, 1 token | 3.861262 | 2.569705 | 0.6655 |

Every diagnostic correctness assertion passed. The first one-time weight
pack includes cold initialization (251.831432 ms); it must not be compared
with later warm pack samples as if they used the same protocol. Warm call
ratios exclude this one-time packing but include per-call activation
conversion and uploading both weight planes.

Readiness: inference BLOCKED_NOT_INTEGRATED, training BLOCKED_UNSUPPORTED.
The H100 job was still queued at the final status read; its live log is
available with `sh tools/nvidia_central.sh log nvc3-0031`.
The model runner's syntax was also parsed successfully on the H100 host;
actual model execution remains untested pending integration.

## Resume result: H100 public API diagnostic

Job nvc3-0031 completed exit 0 at 2026-09-29T05:13:25Z, source
4c875f5ba, NVIDIA H100 NVL. Raw JSON and build/job log are retained in
`bench/results/lowbit_e2e/2026-09-29/h100-public-api/lowbit-e2e-Z4b64zXd/`.
Every diagnostic assertion passed. Comparison by named shape against the
AMD run found identical input hashes and identical complete output hashes
for BOTH profiles separately, all five cases. This is not equality between
fixed15 and fp32, nor a whole-model cross-vendor certificate.

| Public call | H100 fixed15 / fp32 | AMD fixed15 / fp32 |
|---|---:|---:|
| attention projection, 1 token | 1.0098 | 0.7535 |
| attention projection, 8 tokens | 0.9915 | 0.7843 |
| attention projection, 512 tokens | 1.2695 | 0.5987 |
| feed-forward down, 512 tokens | 1.3011 | 1.3563 |
| output head, 1 token | 0.5414 | 0.6655 |

These are one-run, five-sample public-call diagnostics, not model speed.
The H100 head's fixed15 samples span 11.095 to 35.901 ms; retain that
variability rather than treating its median as a stable deployment claim.
The two large H100 projection calls regress in this snapshot despite the
other lane's kernel-only gains. Transfers, allocations and dispatch are
included here; their individual contributions have NOT been isolated.

Integration inspection on resume: lane/lowbit-blocks is at 84b6479e8, with
committed Python/binding/model integration and recorded passing 4090 block
gates. Whole-model validation scripts exist there; the public registry
still refuses inference and training. That lane explicitly enables the
experimental profile inside its development-only validation process. Our
public-API model probe still fails closed; it was not queued with an
override or reported as passing. Do not duplicate that lane's active model
gates. No integration merge into this branch, default change or new job was
needed to collect this result.

Next: collect the owning lane's whole-model verdict and supported profile
revision, review/import that committed integration, then queue the paired
stateful model probe. Kernel timing and this public-call timing cannot
substitute for that result. Training remains a separate unsupported phase.
