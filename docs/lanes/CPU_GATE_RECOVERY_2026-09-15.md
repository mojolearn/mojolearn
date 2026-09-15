# Routine CPU gate recovery

The routine gate uses three hosted CPU environments: Linux x86-64, Linux
ARM64, and Apple silicon macOS. Five random x86 draws have been reduced to
one. This is CPU coverage; release GPU evidence remains one AMD, one NVIDIA,
and one Apple column under FANOUT_RULES_2026-09-15.md section 00. Extra GPU
models and multi-device records are not routine release requirements.
Existing evidence and exposed APIs remain intact. A three-vendor record does
not newly certify every architecture or multi-device behavior.

## Verification frequency

Routine pushes run CPU inference against committed expected bytes, metrics
regression tests, binding readback, and two small plumbing lanes. They do not
run the broad CPU training suite or rent GPUs. The full CPU reference suite
runs weekly on Sunday, through workflow_dispatch, and as a reusable workflow
required by both publication jobs in release-provenance.yml. Production GPU
certification remains release-only with one Apple, one AMD and one NVIDIA.
The byte-LM and forest inference workflows now also use three CPU environments
and cache their Pixi environments. The already-published byte-LM trainer keeps
its existing focused compatibility checks.

## Work removed and work retained

- Full certification production: every covered lane, all nine fixtures, two repeats, including
  the existing training, inference, model and batch comparisons.
- Full certification sabotage: every covered lane and all nine fixtures, one repeat. This checks
  rejection of altered arithmetic; production still checks repeat stability.
- Independent lane shards run on at most four workers per CI runner. Each
  worker uses one host thread. Local development remains one worker/core.
- Missing lane/fixture cells, failed workers and incomplete columns fail.
  Merging shards requires matching machine, commit, fixtures and binding bytes.
- Pixi caches the locked environment. Production host binaries use an exact
  cache key covering Mojo sources, binding build scripts, manifest, lockfile,
  target, runner image and workflow flags. Cache hits still undergo readback
  and all correctness checks. Sabotage binaries never enter this cache.
- A newer push cancels only that branch's superseded gate. Main runs are
  unique and never cancelled by this workflow. A branch pass does not skip
  main's integration gate.

The lane/fixture execution count changes from 117 x 9 x 2 x 2 x 7 = 29,484
at the starting surface to 117 x 9 x (2 + 1) x 3 = 9,477, about 68% fewer.
That excludes compilation and other tests and is not a wall-time claim.
Parallelism further reduces elapsed time subject to CPU/memory contention.

## OWED cells: CPU cells no GPU record hashes yet

GPU records are taken only at PyPI releases, so a CPU lane can merge a cell
part (train, infer, model, batch) that the committed GPU columns do not
hash. The full gate's diff (`identity_break.py --diff ... --require-columns 4
--owed-json owed_cells.json`) reads such a part `OWED xK` instead of failing
the four-column requirement. OWED is derived per column, never listed by
hand. A part is OWED only when all of the following hold:

- Every column without a hash for it has no cell (a lane not in that
  record), no such part, or an n/a value.
- No column reads REFUSED, MOVED or BATCH_MOVED on it. A GPU column that
  refuses a cell the CPU hashes still fails.
- The CPU column hashes it STABLE over two or more repeats.
- Every column that has a hash agrees. A recorded hash that differs is still
  DIVERGENT and fails.

In the sabotage step, `cpu_identity_gate_check.py owed` requires every owed
part to move between the production and `MOJOLEARN_HOST_SABOTAGE` CPU
columns. A part that does not move, or is refused or absent under sabotage,
fails. The summary prints `OWED=N` beside `IDENTICAL`, and `owed_cells.json`
(lane, fixture, part, missing columns) is the exact list the next release
record must take. The pinned greps over the committed GPU columns alone are
unchanged, because OWED applies only with `--owed-json`.

## Validation

`python3 -m unittest discover -s tools -p test_cpu_identity_gate.py -v`
checks sharding and negative controls without compiling bindings. The host
surface tests, docs facts, packaging pins and package inventory also pass
locally. CI must verify real production and sabotage runs on all three hosts
before merging. Additional local validation at 08b50887a used the gate-budget
branch's host binaries (Mojo sources, build scripts and lockfile unchanged),
copied into an isolated evidence directory. All 117 lanes on the base fixture
with two repeats passed in 181 seconds, one core, M4, shared machine. OLS and
ridge serial versus sharded runs read IDENTICAL x2 for training, inference,
model and batch; corrupting OLS produced DIVERGENT=1 and exit 1. The full
117-lane base-fixture serial/sharded comparison also passed: train
IDENTICAL=117; inference/model IDENTICAL=145, N/A=89; batch IDENTICAL=93,
N/A=24, with require-columns 2 satisfied.

Compare queue, build, production and sabotage step durations
against run 34970403640; cold and warm cache times must be distinguished.

## Follow-up work

Compact per-family fixture sizes need their own versioned GPU reference
records; no fixture dimensions or edge cases change here. Dependency-aware
branch selection needs a complete dependency map and fail-closed fallback.
Neither is required to land this reduction in repeated work. The separate R2
binding-cache lane remains responsible for portable remote GPU build caching.

## Hosted evidence

CPU identity run [34978769155](https://github.com/mojolearn/mojolearn/actions/runs/34978769155)
at 08b50887a: ARM64 passed the entire cold-cache job in 21m 23s, including
production (7m 31s) and sabotage execution (3m 46s). The earlier main run
34970403640 took 28m 05s for ARM64 production alone. Packaging run
34978769232 and community-health run 34978769199 passed. macOS and x86
were still running when these notes were committed; main promotion must wait
for both. The merge from 590c11c86 changes documentation only.

## CPU product boundary

The inference boundary is implemented in `lane/cpu-training-inference-boundary`.
Public CPU estimator fitting refuses; the source identity harness and its
runtime tests explicitly enter a private reference-training context. The
full 117-lane reference surface remains available. Future wheels and routine
CI build eight inference/helper families, while full CI builds all 21.
Training-only families cannot enter a wheel through stale build outputs.
Mixed inference bindings retain private native helpers used by the verifier;
the numeric implementation and GPU training API are unchanged.

The published 0.8.5 macOS and Linux wheels were downloaded and their SHA256
verified against PyPI metadata. Both contain only the byte-LM host binary and
publicly export LanguageModelInference and LanguageModelHostTrainer. The
published trainer remains compatible. The broad CPU training expansion was
never in those published wheels. This change does not publish a new release.

See [boundary validation](CPU_INFERENCE_BOUNDARY_2026-09-15.md). Further
selection by affected inference family is optional follow-up if measured
routine wall time requires it; no additional broad training gate is added.

## Immediate promotion requested

Andrew explicitly requested immediate main promotion after the local checks
and ARM64 hosted gate passed, while the old full macOS/x86 jobs were still
running. The cadence changes were checked by parsing every workflow and shell
step, docs facts, packaging pins/inventory and the gate regression tests.
Their first hosted routine-inference run remains verification to observe,
not a result claimed by this document.

## Completed full hosted run

Run [34978769155](https://github.com/mojolearn/mojolearn/actions/runs/34978769155)
subsequently passed on all three hosts: ARM64 21m23s, macOS 24m29s, x86-64
31m14s, excluding queues. Production full-fixture execution took 7m31s,
10m11s and 13m17s respectively. These measurements precede the latest routine
inference-only cadence and package boundary; they are not routine push times.
