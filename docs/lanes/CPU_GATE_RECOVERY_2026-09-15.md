# Routine CPU gate recovery

The routine gate uses three hosted CPU environments: Linux x86-64, Linux
ARM64, and Apple silicon macOS. Five random x86 draws have been reduced to
one. This is CPU coverage; release GPU evidence remains one AMD, one NVIDIA,
and one Apple column under FANOUT_RULES_2026-09-15.md section 00. Extra GPU
models and multi-device records are not routine release requirements.
Existing evidence and exposed APIs remain intact. A three-vendor record does
not newly certify every architecture or multi-device behavior.

## Work removed and work retained

- Production: every covered lane, all nine fixtures, two repeats, including
  the existing training, inference, model and batch comparisons.
- Sabotage: every covered lane and all nine fixtures, one repeat. This checks
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

## Proposed CPU product boundary

Andrew's intended direction, discussed September 15, is small development
checks, occasional broad CPU bitwise verification, and public CPU inference.
This speedup does not yet change test frequency or public API boundaries.
The follow-up work is:

1. Separate internal CPU training verification builds from public CPU
   inference operations and wheel exports. Preserve the CPU training code as
   an oracle; define inference support by actual load/predict/transform/forward
   capabilities rather than by the existence of a CPU fit implementation.
2. Run small affected-code and public inference checks on relevant pushes.
   Move broad training identity certification to a schedule when numerical
   source changed and to release qualification. Shared numerical changes need
   appropriately broad checks before acceptance.
3. Test wheel loading, inference identity, batching and clear unsupported
   operation errors; align documentation and the support manifest with the
   public boundary.

The actual PyPI 0.8.5 macOS and Linux wheels were downloaded and their SHA256
verified against PyPI metadata. Both contain only the byte-LM host binary;
`_HOST_MODULES` is empty. Both publicly export LanguageModelInference and
LanguageModelHostTrainer. The broader 117-lane CPU training expansion is on
main and is included by its future wheel configuration, but is not in those
published wheels. Preserve the already-shipped trainer's compatibility or
explicitly deprecate it when changing the product boundary.
