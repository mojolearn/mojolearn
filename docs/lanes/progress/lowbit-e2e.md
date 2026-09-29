# lowbit-e2e: integration evidence on existing resources

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
