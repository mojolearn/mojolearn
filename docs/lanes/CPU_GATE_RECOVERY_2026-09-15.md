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
before merging. Compare queue, build, production and sabotage step durations
against run 34970403640; cold and warm cache times must be distinguished.

## Follow-up work

Compact per-family fixture sizes need their own versioned GPU reference
records; no fixture dimensions or edge cases change here. Dependency-aware
branch selection needs a complete dependency map and fail-closed fallback.
Neither is required to land this reduction in repeated work. The separate R2
binding-cache lane remains responsible for portable remote GPU build caching.
