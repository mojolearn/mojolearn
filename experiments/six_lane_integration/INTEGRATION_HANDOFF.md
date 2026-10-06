# Six-lane integration handoff

The seven frozen source deliveries are consolidated in worktree `/Users/andrewhendel/CascadeProjects/mojolearn-six-lane-integration-20261006`, branch `integration/six-lane-ab-20261006`. The two neural IDENTICAL deliveries form one reconciled lane. Source worktrees remain intact. The owner subsequently authorized merging this delivery into main; see `main_merge.json` for both parents, preserved results and source evidence.

## Source and scope

`inputs.json` records all seven exact input commits, the initial equal local/remote main `0f779ed5d3f0a2ab2f418e054d3af950a766e5f1`, source completeness and the stable untracked trees planning document preserved externally. No historical neural v1 source was added. `neural_conflict_resolutions.json` and `DECISIONS.md` record semantic decisions, shared controls and numerical alternatives. Main advanced independently during integration. The authorized main merge retains its newer repairs and results; the original integration benchmark freeze is preserved as `benchmark.integration-freeze.json`, and `benchmark.json` records the reviewed merged reference.

`catalog.json` contains 375 source records: 353 new candidates, 20 incumbent dependencies and two rejected source hypotheses. Namespaced aliases include the scoped neural equivalences and seven C45–C51 links to trees. Original handoffs and IDs remain available. Source reach is conservative and is not runtime reach; partial implementations, absent callers, ignored controls, unsupported routes and absent saved recipes remain explicit gaps.

`matrix.json` contains 744 configurations and 46,310 future workload/vendor/arm cells. `build_plan.json` contains 4,880 deduplicated exact binding/vendor/mode/define jobs. These are future coverage plans, not evidence that all configurations compiled. Individual sub-arms, declared interactions, compatible combined proposals, aliases, incompatible selections and missing coverage are represented. A uses candidate controls; B uses the shipped incumbent defaults.

## Harness and frozen workloads

`tools/six_lane_ab.py` is the master planner/compiler and can also be reached through `tools/performance_ideas.py master`. It reuses the existing `performance_full_ab_queue.py` contract and existing workload runners. `six_lane_ab_worker.py`, `six_lane_forest_adapter.py` and `six_lane_evidence.py` implement the future worker, unchanged forest recipe adapter and evidence contract. No produced binding was imported or run during integration. Compilation does not establish that this harness executes correctly.

`benchmark.json` pins the reviewed main recipe sources and integrated capture/selector additions. Existing roster, datasets, splits, seeds, sizes, estimator settings, numeric modes, quality gates, opponents and timing definitions are preserved. Removing tree caps is refused. Concrete dataset hashes, dimensions, actual full-data coverage and some candidate-specific saved recipes remain pending; missing coverage blocks execution rather than changing a race.

## Compilation and retained evidence

Numerical source freeze: `76165b54ec6a31b71854f9e2a7b2952934783b3a`. Local hardware: Apple M4, 10 CPUs, 16 GiB; Mojo 1.0.0 (ed45d567), supported native Apple/host builder flags. NVIDIA: native H100 SM90 on Xeon Platinum 8470, cgroup allocation 22.1 CPUs; Linux compiler/argv/hardware are recorded per receipt.

| Selected build coverage | Compiled | Remaining |
| --- | ---: | --- |
| Apple baseline/combined | 127 | One IDENTICAL GBDT combined superset not attempted after the narrower Metal compiler crash |
| Host baseline/combined | 68 | None in this selected set |
| NVIDIA IDENTICAL baseline/combined | 65 | One GBDT combined configuration failed in Mojo offload compilation |

The full future plan is larger: **260 exact jobs COMPILED, one FAILED, 4,619 NOT_COMPILED**. This selected campaign is not complete compile coverage of every independently selectable alternative or declared interaction. A combined build covers only its actual recorded defines and instantiated code; it does not prove runtime reach or omitted alternatives. `compile_selection.json` lists the selected keys and explicit deferred Metal configuration. `build_coverage.json` retains 740 receipts, including reuse records and original failed freezes; receipt count is not unique compiled coverage.

Both H100 rentals are terminated and verified absent. The second rebuilt only the two NVIDIA x_decomp configurations affected by the final shared-file repair. Their artifacts were downloaded and checked against their build hashes before teardown.

Complete logs, source manifests, compiler/target/define records, artifacts and hashes are retained outside Git at `/Users/andrewhendel/CascadeProjects/mojolearn-integration-evidence-20261006`. `build_coverage.json` maps exact configurations and every implementation ID to evidence, including failures and NOT_COMPILED cells. Reuse requires matching conservative source closure, compiler, target/flags and artifact hash; original build commits remain attached. Changed-source failures are retained, never rewritten as passes. Builds used the supported compiler directly with `--emit shared-lib`, not wrappers that import or smoke-test modules.

`modular_blockers.json` records GBDT compiler failures for Metal and native NVIDIA. No toolchain patch, rewritten IR or unsupported cross-compilation was attempted. AMD remains NOT_COMPILED. The H100 rental was terminated and verified absent; downloaded artifact hashes were checked without loading them. This is artifact transport verification, not estimator output or model-state identity verification.

A source-only host-route scan reported 56 findings; it is not a passing runtime check. Source-context classification and two repaired issues (NN56 device arithmetic and NN64 host-module import) are retained externally. No checker baseline exemptions were added.

## Later phase

The owner will run the measurement/verification phase. This merge launches no estimators, runtime checks, measurements or remote jobs. Resolve and select concrete recipes before executing the queued cells. All emitted execution queues and worker recipes currently have `execution_authorized: false`. No defaults were promoted and no result was admitted to a board.

1. Inspect `build_coverage.json` and `modular_blockers.json`; compile missing selected sub-arms/targets with exact defines. Keep each new numerical source freeze separate. Reuse valid artifacts rather than rebuilding unaffected configurations.
2. Select configurations with `python3 tools/six_lane_ab.py show <namespaced-id>` or `python3 tools/performance_ideas.py master list --lane I.N`. Inspect the linked source gaps before scheduling.
3. Supply concrete saved full-workload recipes to `queue --vendor <vendor> --recipes <recipes.json> --select <configuration-id> --output <queue.json>`. Recipes are keyed by matrix cell key and must include source, benchmark/data hashes, dimensions/settings, intrinsic caps, artifact provenance, workload/package bindings, resource policy and hash-backed resolutions for every coverage gap. Queue generation itself runs no estimator; it leaves execution unauthorized.
4. After authorization, use the existing queue runner with one excluded warmup and one scored sample, serial cells per GPU. Freeze inputs and artifact provenance. Preserve every failure, retry, actual sample count and complete log. Preparation, fit/training, required synchronization and consumed outputs belong to the declared operation; report fit, inference, cold and repeated use separately. Hashing, quality inspection and reporting stay outside timing.
5. IDENTICAL compares A across NVIDIA/AMD/Apple/host and B across those separately; A and B need not match each other. Output hashes alone do not establish model-state identity. Missing state, dtype, shape, encoding and capture scope remain explicit. Apple timing does not vote. Apple FAST follows its existing task-quality policy, not cross-vendor bitwise equality.
6. Prepare future board inputs with `board-plan --output <directory>` and use only the existing board tool recorded there. Keep own-only A/B results separate from incumbent/opponent race results. Do not invent opponent ratios or overwrite historical evidence.

The owner-authorized main merge is complete in the commit carrying `main_merge.json`. No estimator execution, runtime tests, timing measurements, quality assessment, bitwise identity verification or promotion was performed by this integration/merge task.

## Main merge

The master harness is retained alongside main's existing runner interfaces and newer LARS/GMM repairs. The more-runner forwards the saved recipe to preserve main's explicit full-GMM selection. The compiler admits clean committed main as well as integration branches. No Mojo source changed during this merge, so the existing exact compile evidence is retained with its original source hashes. The new merge commit is the harness/recipe freeze for your next run; the old benchmark specification remains available for provenance.

The integration produced compilation evidence, not competing timing measurements. Main's historical and newer measured results are preserved. Measured execution can also collect quality and typed output/model-state identity evidence, but timing alone does not assess those gates.
