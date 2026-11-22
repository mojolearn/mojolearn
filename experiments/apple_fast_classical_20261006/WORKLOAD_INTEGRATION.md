# Full workload integration: source only

**NEVER RUN — PENDING MEASUREMENT.** This delivery programs the catalog and adapters. No adapter, harness, checker, compiler, test, device query, or measurement was executed. Every candidate stays off. No quality, timing, compilation, or route success is claimed.

`full_workloads.json` maps all 54 AFCL cards to affected callers, existing full-workload harness entry points, and intrinsic dataset caps. `run_pair.py` implements future paired execution using those builders and their existing independent quality functions. Additional declarative public-operation recipes cover operations that saved presets do not consume: multioutput regression metrics, ranking curves, GPR uncertainty, full resampling, column statistics, selector scores, GMM outputs, and full TreeSHAP. The generic `public/full-classical-operation` recipe can express other classical transitive callers with a named complete task and accepted scope evidence. These new adapters do not pretend that an audited dataset or a saved measurement already exists.

## Existing workers and boundaries

| Adapter | Saved source reused | Whole operation |
| --- | --- | --- |
| `classical` | `tools/classical_two_datasets.py::BUILDERS[(lane, 'ours')]` | Fresh complete inputs, construction, `.call()`, synchronization, `.outputs()`, final synchronization. Constructor-time fit is included. |
| `classical2` | `tools/bench_board_more.py::build(lane, 'ours-fast', arrays, record)` | Same whole boundary, including queries/forecasts performed by saved `.outputs()`. |
| `algos` | `tools/bench_board_algos.py::build(lane, 'ours-fast', arrays)` | Construction, `.fit()`, `.sync_for_receipt()`, `.infer()`, consumed `.outputs()`, final synchronization. Fits hidden in builders are included. |
| `trees` | `bench/speed/forest_speed_arm.py`, using `tools/speed_gbdt_arm.py` | Existing input conversion and task preparation, model construction, full fit, synchronization, full scoring/prediction consumption, final synchronization. Saved forest scoring includes its independent metric calculation; that scope is recorded. |
| `public` | Declared public calls through `PublicProgram` | Complete construction/fit/transform/predict/metric program and every declared output, using the same full arrays for A and B. No estimator arithmetic is reimplemented. |

The saved loaders and `lane_arrays` functions are deliberately not evidence that a workload is complete. The catalog records, among others, kNN 400k/4k prefixes, KDE 100k/2k, SVC 10k, HDBSCAN constructor slicing, the classical2 per-lane caps, algos feature/row restrictions, text/zone derivation, and shipped forest loader limits. The future run requires audited full lane inputs with source-population lineage. If a saved builder still slices them, the public-call trace rejects the smaller or changed actual input. A public-operation recipe can express the uncapped supported operation without altering model settings to improve timing.

Established synthetic ARIMA/ETS tasks are legitimate workloads. Their generator source/version/seed and entire declared series/time/holdout populations must be frozen. They are labeled `dataset_kind: synthetic`; they never stand in for an existing real dataset. No synthetic replacement is accepted by real-task recipes.

All arms execute serially as separate processes. Each has one excluded first/cold operation and one scored repeated operation. Model construction and fresh full input copies occur inside each operation. Loading source files and parsing the audited cache are excluded, explicitly recorded CPU input steps; their preprocessing and lineage must be declared. Phase fields are preparation, fit or saved primary call, and inference/output consumption. A saved primary call that also queries is not mislabeled as pure fit. Public programs additionally retain per-call scopes and times. Separate fit/inference variants should be separate audited workload IDs when an affected API has materially different operation boundaries.

## Future invocation and build artifacts

The common workflow invokes:

```text
run_pair.py --idea AFCL-ID --arms PAIRED_BUILD_DIR --output FRESH_RESULT_DIR \
  --source-sha FROZEN_SHA --workloads AUDITED_CONFIG --stage validate|time|run
```

This is documentation for later authorized execution, not an instruction to execute during this delivery.

The runner consumes `paired-build.json` from `build_pair.py`, resolves transferable `package_relative` paths, and checks the final staged `.so`, dylib, and Python-source hashes before future execution. Paths inside the artifact tables are relative to each `python/mojolearn` package. Baseline/candidate flags must exactly equal the selected lane manifests, including the same prerequisite flags in both arms. Runtime prerequisites must exactly match each arm. All per-card environment controls are removed from inherited environments before applying the arm's controls; CPU thread caps are removed on dedicated Apple hosts. An inherited candidate flag cannot leak into baseline.

The shared queue policy also guards direct invocation and internal workers: Apple FAST device work belongs to the owned Apple M3 Ultra queue, not an authoring laptop. Machine/hostname/chip/model and effective thread-pool information are captured only during that future run. Validation uses the same complete operation and outputs, but its timing is not admitted as a timing stage. `time`/`run` additionally require previously accepted same-freeze quality receipts, provided by the common CLI through `MOJOLEARN_AFCL_QUALITY_RECEIPT`, or by `prior_quality_receipts` in the unchanged audit config.

`AFCL-G01` and `AFCL-G02` are mutually exclusive: G02 requires MMA off, while G01 changes MMA. Other baseline/candidate prerequisite toggles are taken from each lane JSON, never guessed by the runner. Factorial builder artifacts must be explicitly selected into a baseline/candidate pair; this runner does not silently interpret `configuration_NNN` as an arm.

## Required audited configuration

The catalog is **not** an execution config. A separate JSON config must contain:

- `schema: 1`, `status: "audited_full_workloads"`, and the frozen `source_sha`.
- `workloads`: uniquely named entries, each with `id`, `ideas`, a mapped `recipe`, and the fields below.
- Optional `shared_environment`, excluding per-card controls and CPU/smoke caps.
- Optional `evidence_bundle_path`: a stable path to subsequently accepted proof receipts. It is intentionally a path rather than an embedded digest: attaching later evidence must not change the workload hash. Each actual bundle and every underlying artifact is hashed in retained results.
- Optional `prior_quality_receipts`: card IDs mapped to stable paths, for direct future timing after validation.
- For a combined run, `recipe_digests`: each selected card mapped to its **hydrated** recipe digest from the common workflow. For the usual single-card invocation the runner uses `MOJOLEARN_AFCL_RECIPE_SHA256`. No raw registration hash substitutes for that digest. Missing digests leave quality receipts pending.

Each workload requires:

1. `dataset_kind: "real"` or an allowed established `"synthetic"`, `input_mode: "audited_lane_inputs"`, and `dataset_identity` containing name, version, and nonempty source file receipts. Every file receipt is `{ "path": ..., "sha256": ... }`, relative to the audit config unless absolute.
2. `arrays_npz`, a file receipt for full prepared arrays. `expected_arrays` maps each input name to exact `shape`, `dtype`, and SHA-256 of contiguous array bytes. No pickle/object arrays are admitted. Optional `scalars` hold non-array builder metadata without overwriting arrays.
3. `full_dataset_audit` with `complete: true`, `row_subsampling: false`, explicit `lineage` and `preprocessing`, `source_populations` and `array_populations`. Each source population has `full_count` and `source_split`. Each array population names its `population`, `axis` (default 0), and `usage` (`runtime` by default, or `quality` for heldout labels consumed only by independent scoring). Every full population must map to a complete input axis. Array files alone are not population evidence.
4. Exact `expected_params` matching the constructed saved estimator's parameter record; for `public`, this is the exact `public_program` list. Keep fixed seeds, tolerances, epochs/iterations, tree counts, feature sets, queries, and quality settings matched. Optional `record_json` provides the hashed saved prepared record. A smoke record is refused.
5. For saved workers, `trace_calls`: rules such as `{ "target": "NearestNeighbors.fit", "arguments": { "0": "index" } }` and `{ "target": "NearestNeighbors.kneighbors", "arguments": { "0": "queries" } }`. Positional indices exclude `self`; nonnumeric keys name keyword arguments. Constructor inputs can be traced through `Class.__init__`. Each declared trace must execute, every runtime population must be observed, and actual full argument shapes and data hashes must match the audited inputs. Trace receipt hashing occurs outside the operation clock. Existing API paths that intentionally mutate or transform an argument require a correctly declared actual boundary/array, not a false original-array claim.
6. `quality_gates`: existing independently justified task metrics, each with `metric`, `direction` (`higher` or `lower`), `floor`, nonnegative finite `max_degradation_abs`, and hashed `policy_evidence`. Both arms must satisfy the independent floor and candidate must meet the permitted A/B degradation. No floor is invented by this delivery. Missing/nonfinite metrics remain pending; failed metrics remain failures.
7. `operation_boundary: "prepare-fit-synchronize-infer-consume"` and all applicable quality references. Optional `reference_outputs` are NPZ file receipts with an `arm` key expected by the saved quality function (`sklearn-cpu`, `scipy-cpu`, or `independent-reference` for public recipes) and a `provenance` JSON receipt. Provenance must identify the same `arrays_npz_sha256`, library version, and settings.
8. Established synthetic workloads additionally need `generator` with hashed `source`, version, seed, and full declared dimensions. Their materialized arrays also have ordinary data hashes. Real-derived series/corpus tasks retain the actual source lineage.

CSR input is supported without object arrays: `sparse_inputs` maps a logical input name to `{ "format": "csr", "shape": [...], "data": "npz_data_key", "indices": "npz_indices_key", "indptr": "npz_indptr_key" }`. The three NPZ arrays reconstruct the CSR matrix. Its `expected_arrays` entry records shape/dtype/format plus a `components` mapping of data/indices/indptr byte receipts. Full row and feature populations still refer to the logical matrix. Reconstruction uses ordinary benchmark input glue; no new Python estimator runtime is introduced.

## Public-operation programs

A program is a list of `construct`, `method`, or `function` operations. Each has an `id`. Constructors/functions name an allowed classical callable in `target`; methods name an earlier object ID and `method`. `args` and `kwargs` forward arguments unchanged. Arguments can be literal settings, `{ "array": "X" }`, `{ "ref": "model" }`, or `{ "constructor": "mojolearn.RBF", "kwargs": { "length_scale": 1.0 } }`. No eval, shell command, runtime sampling, or arbitrary estimator implementation is supported.

For example, the GPR uncertainty operation shape is:

```json
[
  {"id":"model","kind":"construct","target":"mojolearn.GaussianProcessRegressor","kwargs":{"optimizer":null}},
  {"id":"fit","kind":"method","object":"model","method":"fit","args":[{"array":"X"},{"array":"y"}],"consume":false},
  {"id":"uncertainty","kind":"method","object":"model","method":"predict","args":[{"array":"Xq"}],"kwargs":{"return_std":true},"consume":true}
]
```

This illustrates syntax, not approved settings, dimensions, a dataset, or a quality policy. Use the exact existing task settings and an independent full uncertainty reference. The consumed tuple has output keys `uncertainty.0` and `uncertainty.1`.

A full multioutput metric program calls e.g. `mojolearn.metrics.mean_squared_error` with complete `y_true`, `y_pred`, optional sample/output weight arrays, and explicit `multioutput`; consume every returned result. Ranking recipes similarly preserve full labels, scores, weights and ties. A resampling recipe calls `mojolearn.resample.resample` with all full input arrays and explicit existing sampling settings; it does not replace that call with a NumPy gather. Missingness, category codes, sparse structures and source labels must be declared and hashed in the inputs.

Numerical tuple/list/dict outputs are recursively consumed. Public result objects can use `consume_fields`, a mapping of output label to public attribute path, such as confidence-interval endpoints. CSR results are consumed as their data/indices/indptr/shape. Fit's estimator return is not a numeric output and should not be consumed. The runner refuses a program that consumes no numeric output.

Independent public quality uses either retained `independent-reference` output files or an explicit `reference_program` targeting the declared CPU libraries NumPy, SciPy, scikit-learn, statsmodels or statsforecast. Reference execution is outside the timed operation, and its outputs/recipe are retained. Actual versus reference outputs must have identical names/shapes and matching finite/infinite sentinel structure; metrics are `<output-key>_max_abs_error` and `<output-key>_relative_l2_error`. Configure the existing task's accepted bounds, including exact-zero bounds where required. Do not use an unrelated library's random ordering as a supposedly exact resampling oracle. Saved resampling workers retain their independent population-quality checks; separate contract evidence establishes seeded API semantics. Independent references and API receipts must cover task-specific quality beyond a generic array-distance metric when that is the established requirement.

## External route/API evidence and complete coverage

The runner never turns process exit zero, author metadata, a selected compile flag, or an output checksum into proof that a candidate kernel executed or that the entire API contract passed. Those gates require an **already accepted external evidence bundle**. This can reuse accepted frozen evidence; it does not require recompiling or rerunning identity.

The bundle is `{ "schema": 1, "receipts": [...] }`. Each accepted receipt contains:

- `id`, `status: "PASS"`, `mode: "fast"`, `vendor: "apple"`, `source_sha`, `workloads_sha256`, `paired_build_sha256`, and `manifest_sha256` equal to the hydrated per-card recipe digest.
- Exact `artifact_hashes` for both arms, matching the result/build provenance.
- `gate`, one of `actual_candidate_route`, `api_contract`, `full_dataset_coverage`.
- `accepted_by`, a hashed `acceptance_evidence` file, and nonempty `execution_evidence` file receipts. These refer to actual accepted execution/check artifacts, not newly authored assertions. The runner verifies provenance and retained bytes; the existing acceptance process is responsible for the substantive truth of its proof.
- For a route receipt: `workload_id`, `executed_candidate_symbols`, `baseline_control_witness`, and `machine.chip` matching the timed Apple chip.
- For an API receipt: `workload_id` and nonempty `checks` identifying the retained executed contract checks.
- For a scope receipt: `covered_callers`, `covered_workload_ids`, plus either `pending_note` exactly matching a catalog note, or `resolved_recipe` naming a catalog recipe whose complete coverage is established by the mapped workloads. The underlying evidence must establish the actual caller/settings coverage or equivalence. Every covered workload ID must have a retained completed matched operation.

Each workload's route/API gate requires a receipt for every selected card affecting it. Static catalog notes and uncovered recipes remain visible until accepted scope receipts resolve them. This avoids both silently ignoring missing transitive callers and permanently hardcoding a note as pending after genuine evidence is available. `public/full-classical-operation` can provide the named operation used by such a resolution. Nothing is resolved in this delivery.

Attach accepted proofs to existing results without another device operation:

```text
run_pair.py --attach-evidence-to RETAINED_RESULT_DIR --workloads UNCHANGED_AUDIT_CONFIG \
  --output FRESH_ATTACHMENT_DIR
```

This checks the unchanged audit/catalog/build/package/worker/output artifacts, reuses the original task-quality results and operation receipts, and writes a new result/quality receipt. Original evidence is retained. The bundle path was frozen in the audit beforehand; missing or mismatched proofs remain pending or are refused. This attachment route does not query hardware, import the runtime, execute a workload, or change the original run's validation/timing stage.

## Receipts and remaining evidence

Every attempt writes `result.json`, including early failures and selected IDs. Complete worker stdout/stderr stays in `<workload>/<arm>/worker.log`; structured `worker.json`, consumed `.outputs.npz`, task JSON, source/build hashes, machine, full consumed dimensions, exact settings, sample scopes, and failures are retained. Failed/incomplete attempts never synthesize quality PASS. Output directories must be fresh; no successful earlier cell is overwritten.

`<AFCL-ID>/quality-receipt.json` contains exact gate names mapped to PASS/FAIL/PENDING, source/recipe/workload/build/artifact/machine provenance, and overall status. A single-card run also writes top-level `quality-receipt.json` for the common CLI. All five gates must actually pass, and a hydrated recipe digest must exist, before a PASS receipt is emitted. A qualified receipt still does not promote any default. The existing exporter/board tools can retain failed and unqualified attempts separately; only qualified timing-stage evidence is eligible for timing admission.

The source adapters are now present. Remaining work is evidence: accepted frozen build artifacts; full dataset/generator/feature/series lineage and hashes; exact workload settings; independent task floors/references; actual candidate route/API witnesses; and complete transitive/settings coverage resolutions. Saved caps or unavailable datasets stay pending. No performance or quality result, default change, complete campaign claim, or fabricated fixture is included.
