# Classical ML IDENTICAL source lane

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.

The [A/B inventory](AB_EXPERIMENT_INVENTORY.md) lists every new independently selectable arm, its exact A compiler defines, incumbent B, source callers, workload keys, interactions and older experiment references. [The implementation ledger](implementation_ledger.json) retains per-ID scope and gaps. C01–C44 and C52–C60 have production source arms; this does not claim every broader estimator extension in each idea is finished. C45–C51 belong to TREES IDENTICAL and are cross-referenced only.

The original `catalog.json`, `catalog_source.py` and plan remain unchanged. No board results or passing evidence were generated. All C switches are OFF in ordinary builds and require IDENTICAL mode. Numerical changes intend one common host/NVIDIA/AMD/Apple profile per selected source version; every identity and quality claim remains unverified.

## Selection programmed for a future authorized campaign

`tools/performance_ideas.py` discovers the C manifests alongside existing I/A/N/F experiments. The new `--configuration` argument selects a named independent arm or interaction. It propagates to the full-workload harness, build defines, artifact provenance, quality-receipt match and output records. `--arm candidate` is A; `--arm baseline` is B. Absence of new C defines preserves the incumbent, including its existing enabled I controls.

For example, the following are **future command templates, not commands run for this handoff**:

```sh
python tools/performance_ideas.py list --mode identical
python tools/performance_ideas.py plan C52 --configuration pair128 --stage build --vendor nvidia --arm candidate --output /absolute/evidence/C52/A --workload-recipe /absolute/recipes/C52-pair128.json
```

Use the same ID/configuration and separate output directories for A and B. `validate` is programmed to capture full outputs without timing or claiming a comparison passed. `time`/`run` requires an admitted source/configuration-matched quality receipt and the existing owned queue. Host and Apple are identity witnesses; NVIDIA and AMD vote on performance. No default is automatically promoted and no board is automatically written.

Existing builder scripts are invoked only by an explicitly selected future build stage, in a fresh archive outside the worktree. The recipe must name the frozen source commit and the complete required family/host builders. Existing compilation can be reused through a matching artifact record; this lane does not imply permission to compile now. Compiler/environment provisioning, native binding dependencies and full recipes are pending. There is no unsupported AMD portability or toolchain workaround.

## Full-workload recipe facts

[workload_map.json](workload_map.json) points each candidate to saved dataset-preparation and estimator factories in `tools/classical_two_datasets.py`, `tools/bench_board_more.py` or `tools/bench_board_algos.py`. Null facts remain pending. The saved factories contain intrinsic row/subsample limits: finding a recipe is not proof that it uses the full intended dataset.

A resolved campaign recipe supplies:

- Candidate ID, named configuration, `mode: identical`, frozen `source_sha`, and exact acknowledgement of the selected configuration's `source_gaps`.
- Required `build_commands` as existing `bash bindings/build*.sh` argv lists, complete binding dependencies and worker allocation/provisioning facts.
- Every required workload key. One active workload runs per fresh worker; other incomplete workloads stay explicit `pending` records. A partial worker receipt cannot complete the campaign.
- Dataset name/version/hash/split, every input file/hash, actual prepared shapes, full-data coverage basis and reviewed intrinsic-cap audit.
- Saved estimator settings and exact `estimator_settings_record` in the existing board parameter-record format. Runtime numeric-mode/vendor readback and package location must match the frozen arm.
- Required fitted/output attributes, inference operations, requested output schema, full sample-weight files where applicable, and the declared whole-operation boundary.

[recipe.pending.json](recipe.pending.json) is an intentionally unresolved shape template. It cannot run as a completed recipe. Do not fill missing facts with a guessed board dimension, silently change settings, synthesize missing weights, truncate data or use another estimator route to claim reach.

`full_workload.py` includes input decoding/preparation, runner/estimator setup, native fit/training, required synchronization and consumed output bytes in the whole-operation boundary. It records separate preparation, fit, inference and consumption spans; cold, excluded warmup and repeated invocation are distinct. A separate repeated-inference cell retains the same fitted estimator. Existing factories can construct a fresh estimator for a repeated fit; that lifetime is reported explicitly rather than represented as fitted-state reuse. Effective thread pools are recorded after the workload together with worker affinity/cgroup facts; no one-thread cap is introduced.

Report and route adapters in `metric_workloads.py`, `estimator_workloads.py` and `more_workloads.py` call the actual public Mojo-backed estimators. They request full train/query outputs needed to reach report, LDA transform, weighted CV and linkage sub-arms. Additional report/output variants are explicit A/B operations over the saved full dataset, with their own unresolved recipe facts. Ordinary classifier prediction is not evidence for a report-only or LDA-transform arm. C42's x_cluster linkage/output variant is not the incumbent hierarchy single-linkage workload.

## Status and source limits

Every manifest and switch retains the full unqualified status above. The detailed ledgers separate implemented source arms, incomplete extensions, unavailable public representations/APIs and missing workload/compilation prerequisites. Conditional scratch-budget arms may not activate on a saved full workload; future receipts must establish actual route admission. Historical evidence is not reused merely because an old experiment ID is referenced.

No candidate was imported or invoked. No compilation, testing, verification, linting, manifest validation, smoke execution, measurement, benchmark, remote job or build/test hook was run. All runtime calculations added by this lane are Mojo; Python additions are API transport or experiment orchestration.
