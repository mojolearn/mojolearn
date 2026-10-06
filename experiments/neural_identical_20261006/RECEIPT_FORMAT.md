# Future offline receipt comparison

**NOT TESTED — NOT COMPILED — NOT MEASURED.** The source in
[`tools/neural_identical_compare.py`](../../tools/neural_identical_compare.py)
has not been executed, including its help entry. It only reads retained JSON
metadata and checks whether referenced evidence paths exist. It never opens
tensor/log contents, launches a process, compiles, accesses a device, changes
a board or promotes a default. Evidence content authenticity and scientific
qualification still require independent review.

The future command accepts `--bundle receipt-set.json --output new-report.json`.
The output must be a new path. No command was run in this campaign.

## Bundle and predeclared comparison recipe

The bundle is an object with `plan` (the frozen plan produced by
`neural_identical_ab.py`), `comparison_recipe`, and `receipts`. Each receipt
entry supplies `case_id`, `arm` (`A` or `B`), `column` (`nvidia`, `amd`, `apple`
or `host`), `driver_receipt`, and `execution_receipt`. Paths in the bundle
are relative to its directory. Eight cells are required for each case;
duplicate, missing, unknown and skipped cells remain visible.

The comparison recipe has `schema: 1`, `mode: identical`, `source_sha`, a
timezone-aware `declared_at`, nonempty `declaration_evidence`, `sampling`
equal to `{"excluded_warmups": 1, "scored_samples": 1}`, and nonempty `cases`.
Each evidence reference is `{"path": "retained-file", "sha256": "64 lowercase hex digits"}`.
Declaration references resolve relative to the recipe; driver references
resolve relative to the driver receipt. The comparator records missing files
as pending and does not read or hash their contents.

Every case requires:

- `id`, `operation`, dataset name/version/SHA256/split/intended extent/actual
  extent, `actual_dimensions`, explicit `intrinsic_caps`, and common logical
  `estimator_settings`. Intended and actual dataset extents must agree.
- `input_manifest_sha256`, `initial_state_sha256`, and `settings_sha256`.
  These identify the exact logical inputs, restored model/optimizer/RNG state
  and model settings. Different vendor storage layouts do not change them.
- `timed_boundaries`, keyed by applicable `cold`, `fit`, `inference` and
  `repeated` phases. Each boundary has an `includes` list containing
  `preparation`, `operation`, `synchronization` and `consumed_outputs`, plus
  its complete operation-specific description. Every untimed phase needs a
  reason in `non_applicable_phases`. Process wall time is never a score.
- `coverage_requirements`, a nonempty array naming distinct required work
  such as forward, backward, optimizer state, checkpoint/resume and refusals.
  Recipes must actually include every affected full neural caller, relevant
  neighboring shapes, combinations and required non-board workload; the
  comparator cannot discover omitted workloads from a claim of completeness.
- `witness_requirements`, an object mapping group to output name to logical
  descriptor. `outputs`, `model_state` and `gradients` must each have a
  nonempty complete set. Add separate optimizer/checkpoint/refusal groups
  where required. A descriptor records the complete logical extent, dtype,
  layout/encoding and meaning, for example
  `{"dtype":"float32","shape":[2,512,256],"encoding":"little-endian-f32"}`.
  Empty witness collections never establish identity.
- `profiles`, with `A` and `B` objects containing `id`, `contract_sha256`
  and `arithmetic_sha256`, and `profile_relation` of `same_arithmetic` or
  `new_version`. Same-arithmetic cases require equal profile declarations
  and equal A/B witnesses. A new version also needs `arithmetic_revision`
  with `name`, `contract_sha256` and nonempty `changed_seams`, plus an
  explicitly selected V-class design in the frozen plan. A C-class
  combination by itself does not authorize changed bits.
- `timing_policy` with a predeclared nonnegative
  `material_slowdown_fraction`, `combined_method: sum_seconds`, and
  `sample_aggregation: arithmetic_mean`. No slowdown threshold is invented.
- Nonempty `quality_gates`. Each gate has `id`, `metric`,
  `metric_definition_sha256`, `dataset_sha256`, `split`, nonempty `seeds`,
  `training_budget`, a finite `limit`, and a `rule`. Supported rules are
  `absolute_max`, `absolute_min`, `candidate_minus_baseline_max`,
  `candidate_minus_baseline_min`, `candidate_over_baseline_max` and
  `candidate_over_baseline_min`. Absolute rules apply to both arms; relative
  rules compare A against B on every column and declared seed. Ratio gates
  need a positive baseline. Select complete training/held-out quality gates
  before measuring; hashes or finite outputs do not substitute for quality.

## Driver and execution receipts

Use the runner's retained `A.execution.json` or `B.execution.json` as each
`execution_receipt`. The offline comparator requires its `arm`, `exit_code`,
`receipt_present`, `driver_receipt` path and `artifact`. Failed process exits
are failures even if a driver JSON claims PASS.

Each driver receipt has `schema: 1`, `case_id`, `column`, `arm`, `mode`,
`source_sha`, `profile`, `measurement_started_at`, `status`, explicit
`failures` and `skipped_coverage` lists, `coverage_status:
full_workload_attested`, and copies of the case's common dataset/dimensions/
caps/settings/input/state/operation/boundary/coverage fields above. Its
measurement timestamp must follow the declaration timestamp.

The driver records `artifact` exactly as retained by the execution receipt:
source SHA, mode, vendor, exact defines, compiler, target, accepted build
receipt and artifact file paths/hashes. It also records
`experiment_environment` and `experiment_settings` matching the frozen
plan's arm, actual `reached_routes`, nonempty `evidence` references and
`worker_resources`. Resources contain actual `hardware`, `allocation`,
`thread_environment` and `effective_pools`; they come from the worker, not
the controller. NVIDIA/AMD A/B resources, compiler and target must agree
within each vendor before their timing is comparable.

`sampling` contains actual excluded-warmup/scored-sample counts and must
match the recipe. `warmups` and `samples` contain complete indexed records:

- Each warmup has `index`, `excluded: true`, `status: PASS`, and the actual
  `initial_state_sha256`.
- Each scored sample has `index`, `excluded: false`, `status: PASS`, the
  restored `initial_state_sha256`, `timings`, `witnesses` and `coverage`.
- `timings[phase]` has positive finite `seconds`, `synchronized: true`,
  `outputs_consumed: true`, and `boundary` exactly matching the recipe.
- `witnesses[group][name]` contains a nonempty `sha256` and `descriptor`
  exactly matching the declared logical output. The complete sets must
  agree; missing gradients or optimizer state cannot be hidden by a single
  model-output digest.
- `coverage[name]` has `status: PASS` and nonempty evidence references for
  every declared coverage item. Failure/skip evidence remains explicit.

`quality_results[gate_id]` includes `status`, copies of the gate's metric,
definition hash, dataset hash, split, seeds and training budget,
`values_by_seed` (string seed keys to finite measured values), and nonempty
evidence references. A self-reported PASS is insufficient: the comparator
also applies the predeclared rule and limit to the reported values.

## Output interpretation

Within A, NVIDIA/AMD/Apple/host witness sets must match exactly; the complete
B set is compared separately. A/B differences are allowed only for the
explicit new-version declaration above. Profiles and input/settings/state/
coverage must agree across all columns of each arm. The host board's capped
case cannot witness an uncapped GPU workload.

Missing or ambiguous evidence produces `PENDING`; explicit failures and
witness/provenance disagreements are retained separately under `failures`.
`COMPLETE_RECEIPT_MATCH` means this offline metadata comparison completed,
not that source correctness, quality evidence authenticity or a campaign
has been independently qualified.

Timing reports A/B ratios for NVIDIA and AMD separately and the declared
sum-of-seconds ratio, per case and phase. A combined ratio below one with
neither vendor beyond the declared slowdown margin is reported descriptively.
Apple and host never vote. There are no opponent ratios, aggregate campaign
claims, board mutations or automatic promotions. A source default remains
unchanged regardless of the report. Exit codes are 0 for complete receipt
comparison, 1 for recorded failures and 2 for pending evidence/report errors.
