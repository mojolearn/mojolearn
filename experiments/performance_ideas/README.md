# Performance experiment implementations

The newer neural-only NN01–NN64 source arms are indexed alongside this registry
in [the experiment/file inventory](../../docs/plans/NEURAL_AB_EXPERIMENT_INVENTORY_2026-10-06.md).
Their [selector](../../tools/neural_identical_ab.py) writes exact native-builder
and neural-board configurations; the [neural README](../neural_identical_ab/README.md)
describes that interface. These new drafts have no compilation, identity,
quality or timing evidence and do not inherit acceptance from the experiments here.

The [consolidated A/B experiment index](../AB_EXPERIMENT_INDEX.md) lists all
114 registered cards with their files, plus existing IDENTICAL recipes,
historical Apple FAST records and other A/B drivers.

The shared catalog also registers the 54 source-only
[Apple FAST classical candidates](../apple_fast_classical_20261006/README.md)
as `AFCL-L01..L14`, `AFCL-G01..G14`, `AFCL-T01..T12` and `AFCL-P01..P14`.
They are uncompiled, unverified and unmeasured, with every new switch OFF.
Their source-ready status means implementation source exists, not acceptance.
See the [integration workflow](../apple_fast_classical_20261006/INTEGRATION.md)
for common catalog selection, paired packages and full-workload recipe inputs.

This directory implements the cards in
[`PERFORMANCE_EXPERIMENT_IDEAS_2026-10-05.md`](../../docs/plans/PERFORMANCE_EXPERIMENT_IDEAS_2026-10-05.md).
Each card has its own source changes and manifest. Candidates remain opt-in;
source availability and a successful build do not establish a performance win.

The [experiment file index](../apple_fast_trees/EXPERIMENT_INDEX.md) maps all
60 existing cards and 48 new Apple FAST tree cards to their files, alongside
the new interaction plans and older tree A/B records.

The same runner also exposes the 48 namespaced `AFT_F01`–`AFT_P12`
[Apple FAST tree candidates](../apple_fast_trees/README.md). Their manifests
are derived directly from the idea records, with paired-binding and full-workload
integration described [here](../apple_fast_trees/INTEGRATION.md). They remain
uncompiled, unverified and unmeasured; no new defaults are enabled. Legacy F/N
experiment IDs keep their original meanings. The original `EXPECTED` roster is
retained; the complete CLI roster additionally includes `TREE_IDS`.

The later [Apple FAST neural source-only campaign](../apple_fast_neural_20261006/README.md)
adds 44 mechanism cards and eight interaction groups. It is discoverable in
`tools/performance_ideas.py list --mode fast` under the separate `AFN26-`
namespace, with named variants selectable by `plan ... --variant NAME`.
The same plans are exposed by `tools/neural_experiments.py --apple-fast-plan`
and `tools/afn_ab.sh --experiment-plan`. The shared adapter reads the authored
JSON; it does not duplicate defines. All entries remain `not_tested`.
No integration entry point has been run. These are source/plan integrations,
not executable full-workload manifests: AFN26 execution and qualification
remain pending. The original manifest/check contract below remains scoped to
the 60 legacy cards, and legacy AMD `A01` is distinct from `AFN26-A01`.

See the [implementation ledger](IMPLEMENTATION_STATUS.md) for individual source
commits and remaining sub-arms, and [Apple FAST coverage](apple_fast/coverage.md)
for the public callers and transport dependency attestations.

| Cards | Numeric mode | Qualification targets |
| --- | --- | --- |
| I01–I24 | IDENTICAL | NVIDIA and AMD performance; NVIDIA, AMD, Apple and host identity |
| A01–A08 | IDENTICAL | AMD experiments with the complete identity contract |
| N01–N08 | IDENTICAL | NVIDIA experiments with the complete identity contract |
| F01–F20 | FAST | Apple M3 Ultra task quality and completion timing |
| NI01–NI60 | IDENTICAL | [Neural source catalog](../neural_identical_20261006/README.md); default-OFF new arms, qualification pending |

The neural ideas share their lane records with the existing tools through
`python3 tools/performance_ideas.py neural list` (also
`python3 tools/neural_experiments.py ideas list`). `neural plan` selects source
controls, `neural build-plan` targets the frozen native binding builder, and
`neural queue-template` maps those arms into the full-operation A/B queue.
These commands describe work; they do not compile, execute or establish quality.
The original `check --require-all` continues to cover the original 60 manifests.
NI cards use their own source ledger, including partial and rejected ideas.

| AFCL-L/G/T/P (54 cards) | FAST | Apple classical full-workload A/B and task quality; all evidence pending |

IDENTICAL compares all columns within a compiled arithmetic version. A candidate
may change that version's arithmetic contract only when all columns change
coherently and the declared quality gates pass. FAST candidates require their
real caller's quality rule; a dispatch hit or kernel-only error check is
insufficient.

## Required candidate measurements: full datasets

Owner clarification, 2026-10-06: **ALWAYS run candidate A/B measurements on
the full dataset for each affected estimator, through its complete operation.**
This covers AMD, both NVIDIA routes, and applicable Apple FAST candidates.
Compilation, bitwise identity, component timings, tiny caller fixtures, and
opponent-only results do not replace those measurements. Reuse already accepted
compilation and identity evidence; repair actual measurement failures only.

Map every candidate to affected estimators and their full-workload recipes
before timing. Record the recipe/source version, dataset hash and split, actual
dimensions, settings, numeric mode, baseline/candidate defines, and complete
timed boundary. These existing files save the per-estimator workload definitions:

| Workloads | Saved definitions |
| --- | --- |
| Board roster, datasets and tree tasks | [`tools/bench_board.py`](../../tools/bench_board.py): `plan_races`, `TREE_LANES`, `TREE_TASK_DATASETS`; tree driver [`bench/speed/forest_speed_arm.py`](../../bench/speed/forest_speed_arm.py) |
| Classical estimators | [`tools/classical_two_datasets.py`](../../tools/classical_two_datasets.py): `LANES`, `BLOCK_OF`, dataset preparation and lane row constants |
| Additional classical estimators | [`tools/bench_board_more.py`](../../tools/bench_board_more.py): `LANES`, `LANE_CONFIG`, preparation and workload constants |
| Expanded algorithms | [`tools/bench_board_algos.py`](../../tools/bench_board_algos.py): `LANES`, preparation, task settings and timed spans |
| Neural workloads | [`tools/bench_board_neural.py`](../../tools/bench_board_neural.py): full LM/GEMM/block/Samba/MLP shapes and corpus definitions |
| Nearest-neighbor dataset blocks | [`tools/knn_datasets.py`](../../tools/knn_datasets.py): index/query rows and dataset preparation |
| Candidate controls and intended timing contract | Each `<ID>/manifest.json`; these recipes may still contain diagnostic-sized drivers and do not themselves prove full-dataset coverage |

Audit the actual inputs: some existing lane recipes contain fixed dataset
blocks or row caps. `--rows full` and `--neural-shape full` are selectors, not
proof that the intended full workload was executed. Record all such limits;
if the full-dataset mapping is missing or ambiguous, leave qualification pending
and resolve the recipe instead of silently using a reduced substitute. Established
synthetic-only lanes must use their full declared workload, not a smaller fixture.

Time the complete declared operation, including required preparation,
synchronization and output consumption. Keep fit/training, inference, cold use
and repeated use distinguishable. Test relevant interacting toggle combinations
and the proposed complete default configuration: independent component gains do
not imply a combined gain. IDENTICAL promotion still requires the joint NVIDIA
and AMD acceptance rule; Apple timings do not vote on IDENTICAL defaults.

Keep every winner, loser, neutral result and failure, with source/hardware
provenance and sample counts, both in the boards and beside the applicable
switch. Component screening may guide diagnosis but cannot qualify defaults or
close the full measurement board. Report its scope explicitly and keep missing
full-workload measurements pending.

### Audit of the 2026-10-06 campaign

The retained candidate A/B records are component/public-caller screening;
the complete candidate-to-full-dataset estimator campaign remains unfinished.
The separately captured opponent runs do not fill that gap. No AMD or NVIDIA
defaults were flipped in this campaign. Three Apple FAST source defaults **were**
changed from OFF to ON using small caller workloads:

| Default | Source | Promotion commit |
| --- | --- | --- |
| Byte-LM backward no-sync | [`training/byte_lm_afn.mojo`](../../training/byte_lm_afn.mojo), `AFN_LM_BWD_NOSYNC` | `b953bc9a2` |
| HDBSCAN device linkage | [`hdbscan/impl/detail/fast_apple.mojo`](../../hdbscan/impl/detail/fast_apple.mojo), `HDB_LINKAGE_DEVICE` | `429622bef` |
| KDE direct preparation | [`kde/resident_fit.mojo`](../../kde/resident_fit.mojo), `KDE_FAST_DIRECT_PREP` | `709f43abb` |

Those promotions lack the full-dataset and interaction qualification required
above. As of this documentation update they remain ON in source, with explicit
OFF switches; this update does not revert them. Existing enabled defaults that
were merely retained must not be described as newly promoted or newly qualified.

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
compile, run, or qualify candidates. `--require-all` requires all 114 cards (60 original plus 54 AFCL);
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
The standard binding-builder flag input receives the arm's defines; the legacy
extra input is cleared to prevent duplicate Mojo definitions. Build-only
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

## Apple FAST classical integration

The AFCL registration manifests resolve `baseline_defines`, `candidate_defines`,
arm environments and implementation paths from their authoritative lane JSON at
read time. They do not copy a second set of controls. `recipe_digest` covers the
resolved registration and its linked source recipes. Existing F/I/A/N cards keep
their existing manifest-only digest contract.

AFCL build recipes create paired packages via `apple_fast_classical_20261006/build_pair.py`.
Device-stage recipes take `--artifacts` (the retained paired build directory) and
`--workloads` (an explicitly audited full-workload configuration). These inputs
become `{artifacts}` and `{workloads}` in the registered command. `paired_run`
means one invocation processes A and B; asking separately for the baseline arm
is refused. Per-arm runtime environment belongs to the paired runner.

Task-quality admission is tied to the exact workload-configuration hash and
paired-build-manifest hash as well as source, recipe and vendor. A successful
worker exit cannot establish any quality gate. Compilation and validation are
future explicitly selected stages: none was executed during source integration.
