# Performance experiment implementations

This directory implements the cards in
[`PERFORMANCE_EXPERIMENT_IDEAS_2026-10-05.md`](../../docs/plans/PERFORMANCE_EXPERIMENT_IDEAS_2026-10-05.md).
Each card has its own source changes and manifest. Candidates remain opt-in;
source availability and a successful build do not establish a performance win.

See the [implementation ledger](IMPLEMENTATION_STATUS.md) for individual source
commits and remaining sub-arms, and [Apple FAST coverage](apple_fast/coverage.md)
for the public callers and transport dependency attestations.

| Cards | Numeric mode | Qualification targets |
| --- | --- | --- |
| I01–I24 | IDENTICAL | NVIDIA and AMD performance; NVIDIA, AMD, Apple and host identity |
| A01–A08 | IDENTICAL | AMD experiments with the complete identity contract |
| N01–N08 | IDENTICAL | NVIDIA experiments with the complete identity contract |
| F01–F20 | FAST | Apple M3 Ultra task quality and completion timing |

IDENTICAL compares all columns within a compiled arithmetic version. A candidate
may change that version's arithmetic contract only when all columns change
coherently and the declared quality gates pass. FAST candidates require their
real caller's quality rule; a dispatch hit or kernel-only error check is
insufficient.

## Inspect recipes

Run from the repository root:

```sh
python3 tools/performance_ideas.py list
python3 tools/performance_ideas.py list --mode fast
python3 tools/performance_ideas.py check --require-all
python3 tools/performance_ideas.py check --require-all --require-ready
python3 tools/performance_ideas.py plan I01 --stage build --vendor amd --output /tmp/I01-candidate
```

`check` verifies manifests, source references and dependency cycles. It does not
compile, run, or qualify candidates. `--require-all` requires all 60 cards;
`--require-ready` also rejects recorded toolchain or prerequisite blockers.

Implementation status is deliberately distinct from qualification:

- `source_ready`: executable candidate or harness exists; build or device
  qualification remains outstanding.
- `build_passed`: the recorded build passed; task quality and speed still need
  their declared evidence.
- `blocked_toolchain`: the recipe records a specific missing compiler/runtime
  capability or compiler failure.
- `blocked_prerequisite`: a specific prerequisite prevents the full candidate
  or its execution. Its blocker describes the remaining work.

## Manifest contract

Each `<ID>/manifest.json` declares `schema: 1`, the idea ID, title, numeric mode,
vendors, implementation status, executable `implementation_paths`,
`validation_paths`, bare compiler `candidate_defines` and `baseline_defines`,
`build_argv`, `run_argv`, `depends_on`, `quality_gates`, `timing_contract`, and a
`blocker` (null for ready recipes). Source paths must exist inside the repository.

Optional `baseline_build_argv`, `validation_argv`, and `timing_argv` provide the
separate control build and stages. The runner refuses a baseline build without
its explicit recipe. Arguments are arrays executed without a shell. Supported
argument substitutions are `{repo}`, `{python}`, `{mojo}`, `{compile_slot}`,
`{vendor}`, `{output}`, `{source_sha}`, `{mode}`, and `{arm}`.

`paired_build: true` declares a builder that prepares both attested arms in
one invocation. Use the default build arm once; a separate baseline build is
refused. Its own artifact manifest must retain both hashes and exact defines.

`output_kind: "directory"` is available for paired campaigns; the default is a
file. Builds use the existing compile-slot semaphore unless the recipe sets
`build_uses_compile_slot: true` and manages it itself. Recipes must actually
apply their declared defines; listing a define does not enable an experiment.
Both existing binding-builder flag inputs receive the arm's defines. Build-only
execution suppresses builder device smoke gates; run those as queued validation.

## Execute and preserve evidence

Execution requires a clean, committed source tree. Artifacts and evidence must
be outside that tree, and every run uses fresh receipt/log paths. For example:

```sh
python3 tools/performance_ideas.py execute I01 --stage build --vendor amd \
  --output /tmp/I01-candidate \
  --evidence /tmp/I01-build-evidence
```

The runner records the source SHA, manifest hash, exact command, arm, full log,
return code, and elapsed process duration. `COMPLETED` means the process exited
zero. It is never interpreted as a quality pass or GPU performance measurement.
Source changes during execution invalidate the receipt.

Device validation and execution use the established queues. Queue jobs set
`MOJOLEARN_PERFORMANCE_QUEUE_JOB=1`; that variable is an orchestration marker,
not permission to run on the laptop. Apple FAST execution requires the M3 Ultra;
Apple IDENTICAL validation uses the M2 peer and is not admitted as IDENTICAL
performance timing. Host verification is also excluded from performance timing.

Before `--stage time` or `--stage run`, pass `--quality-receipt PATH`. The JSON
receipt must have the same `id`, `mode`, `vendor`, `source_sha`, and
`manifest_sha256`, `status: "PASS"`, and a `gates` object mapping every exact
manifest quality-gate name to `"PASS"`. The relevant harness must establish
those gates with retained evidence. Do not manufacture this receipt from a
build, process exit, or isolated dispatch counter.

The runner does not promote defaults or update performance boards. Each recipe
still requires matched fixtures, warmup, a frozen source pair, one scored run
per arm, full caller completion timing, and the quality and identity evidence
specified by its card.
