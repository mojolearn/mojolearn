# Verify coverage audit, 2026-11-15

Andrew's four questions: does verify still exercise every exposed API endpoint? Does each endpoint have a fault
injector, so that verify is shown to be able to fail on it? Is the evidence kept inside Mojolearn as one record
that is overwritten each time, with history in git, or does it pile up? Where does verify reach into product
internals instead of going through the public API?

Source: `origin/preinteg/wave2-20261113` at `27e2c72224`. This is static analysis only. Nothing was compiled, and no
verify, test or lane was run. The one runtime fact quoted is the existing wheel smoke of `91d5fdea4e`
(`~/mojolearn-evidence/briefs-20261115/verify-smoke-91d5/`). Its results: 510 lanes ran, 28 failed, the self-test
exited with rc 245, and 3 of 76 imports failed.

Files:

- Machine-readable list: [`VERIFY_COVERAGE_AUDIT_2026-11-15.tsv`](VERIFY_COVERAGE_AUDIT_2026-11-15.tsv). It has one
  row for every line of `tools/public_api_surface.py python/mojolearn`, 3,586 rows in all. Filter `verdict ==
  UNMAPPED` to get the unmapped list.
- Scripts and raw output: `~/mojolearn-evidence/briefs-20261115/verify-coverage-audit/`. The scripts are
  `vca_map.py`, `vca_report.py`, `coupling_scan.py`, `coupling_refine.py`, `lane_names.py` and `r2_list.py`. Every
  per-file and per-member stage runs in a process pool sized to `os.cpu_count()` (10 here). The fresh typed lane
  scan forks one worker per core over the 513 lanes.

## 1. Coverage

### How a mapping was established

Each row's `how_established` column names one or more of the layers below. The layers are kept apart on purpose:
`tools/verification_endpoints.py` merges them into a single `lanes` list, and that list cannot tell a call from a
name match.

| layer | what establishes it | members |
|---|---|---:|
| call: `_verify_api_runner` route | `registry()` maps the endpoint to a `_verify_api_<family>` fixture, which calls the method with `getattr(instance, op)(...)` on an instance of the resolved class. This runs in every `verify`. The map was recomputed at this head. | 1,016 |
| call: runner availability contract | The same route, but the endpoint is in `_verify_api_availability.ABSENT \| CONDITIONAL`. Only presence or refusal is checked, not a numerical result. | 27 |
| call: typed receiver in a lane | `tools/verification_endpoint_calls.Calls` over each lane's own source. The receiver type must be known, so seeing `fit` and `KMeans` in the same file does not count. Recomputed at this head, including for members that are not in the census. | 927 |
| call: supplemental verify module | `source_calls` from the `EXPECTED_CASES` roots of `_verification_sources.SUPPLEMENTAL_CHECKS`. Recomputed at this head. | 136 |
| name: census lane reach | The 2026-10-27 census `lanes` that no typed call confirms: the lane's closure names both the class and the op. | 820 |
| name: lane closure (this audit) | A member missing from the census whose class and member names both appear in a lane's name closure (`verify_coverage_audit.lane_names`). | 85 |

One member can carry several layers. **MAPPED (call)** means at least one call layer holds. **MAPPED (name only)**
means only name layers hold. **UNMAPPED** means no layer holds.

### Numbers

The surface has 3,586 lines: 2,850 `method:` lines, 660 `export:` lines and 76 `module:` lines. Of the exports, 150
are constants, submodules or aliases and cannot be called. That leaves **3,360 members**: 2,850 class members plus
510 callable exports.

| | members | MAPPED (call) | MAPPED (name only) | UNMAPPED |
|---|---:|---:|---:|---:|
| class members (`method:`) | 2,850 | 1,704 | 385 | 761 |
| public functions and classes (`export:`) | 510 | 280 | 115 | 115 |
| **total** | **3,360** | **1,984** | **500** | **876** |

How the 876 unmapped members divide:

| class | count | meaning |
|---|---:|---|
| internal class | 520 | Not exported from a public module, or one of the `Host*` CPU twins in `_classical_host.py` and its siblings. `public_api_surface.py` lists every class whose name has no leading underscore, wherever it lives. These are not public API. |
| documented public API (a gap) | **242** | Exported from a public module and not in `docs-site/api_exclude.txt`: 208 methods and 34 exports. |
| infrastructure or adapter | 72 | `adapters.*` (covered by the installed-wheel smoke under the AGENTS.md adapter rule), `gpu_plugins.*`, `host_surface.*`. |
| in `api_exclude.txt` (fine) | 42 | |

The 242 documented gaps have two causes:

- **138 are declared out of scope by the census.** It marks them "metadata only; no numerical identity output":
  67 `numeric_mode_used`, 66 `vendor_used`, 3 `close`, 1 `sample` and 1 `free`. One metadata route per class would
  close them all.
- **104 are not in the census at all**, and no typed or name reach finds them. The real gaps:
  - `TransformerDecodeSession.decode`, `.load_state`, `.discard` and `.state`.
  - `Mamba1DecodeSession.state`, and `.state`, `.close` and `.is_open` on Mamba2 and Mamba3 decode sessions. The
    `mamba*-decode-session` lanes do reach these classes, but through a helper that takes the class as an
    argument, and no static scan can type that.
  - `Coordinator.run`, `Coordinator.drain_events` and `Worker.run`/`.drain_events` (cross-vendor).
  - `TokenBatches.describe` and `.prefetch`.
  - `linalg_fast.has_fused`.
  - `lowbit.BF16Weight` and `Int8Weight` with `.format`.
  - ONNX inspection: `GraphIR.*`, `Node.*`, `Resolved.*` and `importers.supported_ops`.
  - Scheduler `.kind` on `ConstantLR`, `WarmupCosineLR` and `WarmupLinearLR`.
  - `LogisticRegression.lbfgs_memory`; `penalty_normalized` on `LogisticRegression`, `LinearSVC` and `LinearSVR`.
  - Config fields: `ByteLanguageModelConfig.*`, `Tokenizer.bos_token_id` and `.eos_token_id`,
    `TokenizedCorpus.n_vocab`, `BpeTokenizer.ENDOFTEXT`.

  The rest are helpers that public modules re-export by accident: `extratrees` and `randomforest` re-export
  `ForestArchive`, `ForestProtocol`, `ForestClassifierPredict`, `argmax_rows`, `decode_labels` and
  `forest_estimator`. `NumericModeMixin` leaks from five modules, `ClassifierScoreMixin` and
  `RegressorScoreMixin` leak from `neighbors`, and `linear_model.addr` and `addr_ro` leak too. These belong in
  `__all__` trimming or in `api_exclude`, not in verify.

The 500 name-only members are the next tier of doubt: 351 are documented public, 54 are in `api_exclude` and 95 are
internal. A name match is not proof that a lane calls them.

Per family (methods by the file that defines the class, exports by module):

| family | members | mapped (call) | mapped (name only) | unmapped | documented public | api_exclude | internal | infra/adapter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural/sequence | 556 | 375 | 124 | 57 | 36 | 12 | 9 | 0 |
| host twins (internal) | 407 | 20 | 12 | 375 | 0 | 0 | 375 | 0 |
| linear/kernel/gp/svm | 367 | 234 | 49 | 84 | 36 | 0 | 48 | 0 |
| trees | 333 | 232 | 21 | 80 | 65 | 4 | 11 | 0 |
| neighbors/ann/cluster/density | 319 | 188 | 60 | 71 | 38 | 0 | 33 | 0 |
| ops/graph/programs | 255 | 171 | 26 | 58 | 30 | 4 | 24 | 0 |
| other (cross_vendor, exceptions, plugins, adapters, ...) | 237 | 114 | 28 | 95 | 13 | 4 | 6 | 72 |
| prep | 218 | 203 | 15 | 0 | 0 | 0 | 0 | 0 |
| decomp/manifold | 198 | 168 | 23 | 7 | 7 | 0 | 0 | 0 |
| lm/tokenizer/models | 170 | 98 | 57 | 15 | 14 | 1 | 0 | 0 |
| model_selection/metrics | 120 | 65 | 53 | 2 | 1 | 1 | 0 | 0 |
| cnn | 110 | 65 | 24 | 21 | 0 | 16 | 5 | 0 |
| forecasting | 61 | 51 | 8 | 2 | 2 | 0 | 0 | 0 |
| verify internal | 9 | 0 | 0 | 9 | 0 | 0 | 9 | 0 |
| **total** | **3,360** | **1,984** | **500** | **876** | **242** | **42** | **520** | **72** |

### Verify's own inventory is stale and cannot be regenerated at this head

- **The shipped census is three weeks old.** `python/mojolearn/_verification_endpoint_data.py` was last regenerated
  on 2026-10-27 (`bb6f86a935`). Of the 257 source files in its snapshot, 219 have changed since then, 13 files are
  new and 8 are gone. `verify --coverage` reports it as stale (`endpoint_snapshot_matches_sources` is false).
- **The regenerator crashes.** `python3 tools/verification_endpoints.py --check --json ...` runs `lane_reach` for
  10 min 19 s, on a single core at about 65% CPU, and then fails inside the census
  (`tools/public_runtime_inventory.py:1224`, `_call`: `AttributeError: 'str' object has no attribute 'mod'`). The
  log is `endpoints_fresh.log`.
- **The obligation list is equally old.** `_verification_api_targets.py` (1,275 endpoint obligations) dates from the
  2026-10-27 gap audit. Members added since then have no route. This audit recomputed the call layers at this head
  and synthesized census rows for 113 members of exported classes so that the typed scan covers them. The name
  layer from the census is still the October one.

## 2. Fault injection

### What runs inside `verify`, every time

`_verify_public.run` always calls both the tape checks and the endpoint checks with `self_test=True`.

| arm | where | what it perturbs | scope |
|---|---|---|---|
| comparator self-test | `_verify_all.self_test` (`verify --self-test`) | Moves every value of column 0 of the `ols`/`base` input up by one ULP, then requires the cell to read DIVERGENT, and requires the clean run to read IDENTICAL | **1 lane of 513** |
| public endpoint faults | `_verify_api_faults.FAULTS` | Parameter or state faults through real public calls: Ridge alpha, ElasticNet alpha, Adam lr, a TransformerBlock weight ×2, a trainer checkpoint parameter +0.25, the Generator seed, and an ImportedProgram product | **7 endpoints** (9 member rows once aliases are counted) |
| tape faults | `_verify_light_faults.FAULTS` | Monkeypatches `ops.Session` methods: mul becomes add, the backward seed is doubled, the optimizer lr is doubled, a checkpoint parameter, matmul, transpose, conv2d, copy, and mse +0.25 | **9 tape cases** |

### Separate commands, run only when asked

- `verify-cross-validation`: 4 faults (dropped fold, changed model, changed prediction, aliased device).
- `verify-distributed`: 10 transport controls (drop or reverse a native shard result batch).
- `verify-causal-lm`: 1 composition fault, plus `neural_host_sabotage()` when the build is a sabotage build.

### Release and gate tooling, outside `verify`

None of these was re-run at this head.

| arm | lanes | evidence |
|---|---:|---|
| Seam patch arms, `tools/identity_lanes/coverage_sabotage.tsv` (115 arms, each a `core/sabotage/*.patch`, rebuilt and run by `tools/coverage_sabotage.py`) | **128** | 114 arms CAUGHT, run on the Mac against `51bf10f4f` on 2026-10-21 (`coverage_sabotage_proven.tsv`). 1 arm was never run: `coverage-sequence-cnn_classifier_program`, which covers 3 lanes. |
| Historical build negative controls (`_verification_evidence_data`, `negative_controls[kind=build]`) | 219 | From the evidence snapshot at `65d10a9cb2`, with a dirty tree. Historical only. |
| Historical harness controls | 41 | Same snapshot. |
| `-D MOJOLEARN_HOST_SABOTAGE=1` family rebuild (`.github/workflows/cpu-identity-gate.yml`) | 510 (declared, via `host_surface.covered_lanes()`) | CPU column only, a heavy manual workflow (`RUN-HEAVY`). It shows the host binding of a whole family can move. It shows nothing about a GPU route. |

Per lane: 284 of 513 have a seam arm or a historical build control, 228 have only the host gate, and 1 has nothing
at all (`training-device-cross-entropy`). Per member: of the 1,984 members mapped by a call, 9 have their own fault
arm (the 7 API faults), 486 inherit one from a lane that has a seam arm or the self-test, and **1,489 have none**. No
`NATIVE_*_BAD_*` control exists in `core/experiment_guards_*.mojo` on this branch. The brief named that pattern, but
the guards hold only `removed` refusals.

### What `--self-test` proves, and what a per-endpoint injector would need

`--self-test` proves the comparison machinery can discriminate, on one lane and one part (`ols` train). It proves
nothing about the other 512 lanes. A lane that hashes a constant, a part that is never compared, or a reference
entry that is wrong for one lane would all still read VERIFIED.

A per-endpoint injector would need three things for every endpoint:

1. a perturbation that is known to reach that endpoint's output;
2. a clean run and a faulty run;
3. a requirement that the faulty run reads DIVERGENT and that a restored run matches again.

This is what `_verify_api_faults.check_faults` does for its 7 endpoints. Scaling it to 1,984 members means a hand-made
fault per family plus a second execution of every endpoint.

### The cheapest general mechanism

Put one perturbation hook at the comparison boundary.

**Where:** the hook goes in `tools/identity_break.py:1349` `_h(*arrays)`, which every lane's train and infer parts
already go through (it is shipped as `_identity_break.py`), and in the endpoint recorder
(`_verify_api.Recorder.check`, which digests every endpoint output).

**What:** when the hook is armed, it computes a second digest of the same arrays with one bit flipped in the last
byte of the first array. `judge_rows` and `_verify_api_runner.differences` must read that digest as DIVERGENT from
the reference. This needs no second lane execution, so every lane and every endpoint inherits it in every run at
almost no cost.

**It catches:**

- a comparator stuck on IDENTICAL;
- a part or endpoint whose digest is not actually compared;
- a reference row that matches anything;
- a lane whose part is OWED but reads as passed.

A second, sampled arm moves the input fixture by one ULP at `harness.fixture()` for one lane per family per run. It
catches a lane whose output does not depend on its input: a hashed constant or seed, cached state, or the wrong
array hashed.

**Neither arm catches:**

- a kernel bug that is wrong identically on every vendor (only seam patches and independent oracles catch that);
- a fault on a path the bounded fixtures never reach (size thresholds, tile switches, multi-GPU);
- a fault that only shows on another vendor;
- an endpoint verify does not call at all (§1);
- a lane whose output is insensitive to input by design (tokenizer training on fixed bytes, initializer lanes).

The input arm needs a per-kind fixture hook, because sequence and LM lanes build their own data.

## 3. Evidence

### Where verify and the release tools write, and whether each is overwritten or accumulates

| where | what | overwritten or accumulating | size |
|---|---|---|---|
| wheel | `verify_reference/table.json` and `models/` (59 `.npz` plus `models.json`) | overwritten (built by `_verify_reference.py` from committed records) | 5.40 MB, 60 files |
| wheel | `_verification_evidence_data.py` (snapshot of lane evidence) | overwritten (generated) | 360 KB |
| wheel | `_verification_api_reference.py` and `_verification_light_reference.py` (endpoint and tape references) | overwritten | 173 KB and 152 KB |
| wheel | `_verification_endpoint_data.py`, `_verification_api_targets.py`, `_verification_catalog.py` | overwritten (all stale, see §1; the catalog's PROVENANCE is from 2026-09-16) | 123 KB, 29 KB, 50 KB |
| user's machine | `verify --json`/`--out` reports, worker temp directories | written only to the path the user passes, or to temp | none by default |
| repo | `bench/results/identity_break/` (87 dated run directories plus `retained/<wheel sha>/`) | **accumulating** | 391.6 MB, 2,795 files |
| repo | `bench/results/verify_repairs/` | accumulating | 178.8 MB, 373 files |
| repo | `bench/results/pip_verify/` | accumulating | 89.3 MB, 29 files |
| repo | `bench/results/multi_gpu/` | accumulating | 14.7 MB, 2,238 files |
| repo | `bench/results/release_evidence/`, `release_verification/`, `reference_admissions/`, `verify_external/` | accumulating | 2.8, 1.5, 1.1 and 0.8 MB |
| repo | `tools/release_runtime_coverage.json` | overwritten | 3.6 KB |
| repo, all of `bench/results/` | everything above plus the boards | accumulating, and trimmed daily by `evidence-trim.yml` | 1.35 GB, 15,804 files in the tree (the pack is 3.52 GiB) |
| Mac, `~/mojolearn-evidence/` | `release-candidates/` (53 per-sha measurement builds) | accumulating | 126.3 GB |
| Mac | `release/` (`0.8.66/` 4.7 GB, `published/` 0.34 GB, `identities/` 31 MB, `ledger/` mirror 0.6 MB) | accumulating, per version | 5.1 GB |
| Mac | `release-dependencies/` 5.3 GB, `verify-20261112/` 72 MB | accumulating | |
| R2, `mojolearn-data` (1.23 TB, 30,545 objects) | `release-ledger/v1/` (write-once keys) | accumulating | 1.6 MB, 98 objects |
| R2 | `verification/0.8.63/` | accumulating, per version and sha | 37.9 GB, 6,211 objects |
| R2 | `release-readiness/v1/` | accumulating | 19.6 GB, 52 objects |
| R2 | `release-evidence/<version>.tar.gz` | accumulating | 16.7 GB, 31 objects |
| R2 | `release-identity/0.8.66/` and `verify-smoke/<sha>/` | accumulating | 82 MB and 107 MB |
| R2 | `measurements/` (box-jobs 315 GB, records 255 GB, box-archive 121 GB) | accumulating | 789 GB |

The one store that already behaves the way Andrew asks is the wheel's reference. `table.json` is a single file,
overwritten, and its history is in git. Everything else is per-run or per-sha.

### Proposal: one current record per release

1. **One record in the repo.** The release's single cross-vendor verification writes
   `bench/results/verification/current/` and overwrites it each release: `columns/<cpu|apple|nvidia|amd>.json` (the
   per-cell digests and verdicts), `api.json` (endpoint captures), `faults.json` (every arm's CAUGHT or FAILED) and
   `SUMMARY.txt`. `_verify_reference.py` builds `table.json` from this directory only. Older releases are
   `git show v0.8.NN:bench/results/verification/current/...`.
2. **The record producers stop writing dated directories.** That means `tools/record_identity_column.sh`,
   `tools/release_identity_*.py`, `tools/verify_reference_evidence.py` and `tools/coverage_sabotage.py`, which
   between them write `bench/results/identity_break/<date>_*`, `retained/<sha>`, `release_verification/`,
   `reference_admissions/` and `verify_repairs/`. Raw logs go to R2 under one prefix per release,
   `verification/<version>/`, and the previous release's prefix is deleted after publish. The ledger keys stay,
   because they are an index of bytes, not a record.
3. **What `evidence-trim` then stops doing.** It no longer needs to trim the verify groups
   (`identity_break/<date>`, `verify_repairs`, `pip_verify`, `release_verification`, `reference_admissions`,
   `multi_gpu`), about 0.68 GB and 5,500 files of today's 1.35 GB. It also no longer needs the reference-protection
   walk for them, because `current/` is undated, and therefore never a group member and never trimmed. What is left
   for trim is the measurement boards. The old dated directories are removed once with a tombstone line in
   `ARCHIVE_INDEX.md`. That is history in git, as AGENTS.md "Preserve historical tables, raw failures and
   provenance" requires.

## 4. Coupling into product internals

The scan (`coupling_scan.py`, then `coupling_refine.py`) covers imports of underscore modules, `from` imports of
underscore names, attribute access `x._name` with `x` not `self`/`cls`, and `getattr`/`hasattr`/`setattr` with an
underscore string. It runs over the 42 `_verif*`/`_verify*` wheel modules, the harness, the lane fragments and the
verify tools: 105 files in all.

There are 402 raw sites. Removed from that count: 90 verify-to-verify accesses (for example `harness._h`), 7
`__version__` reads, and 3 verify-module imports that look private (`_verify_model._install`, `_verify._git_commit`).
That leaves **160 sites into product internals**: 75 in 19 wheel verify modules and 85 in 10 harness, lane-fragment
and tool files. The harness is byte-copied into the wheel as `_identity_break.py`, so its 85 ship as well. The full
list is `coupling_product.tsv`.

| target | sites | where (examples) | replacement |
|---|---:|---|---|
| `_backend` (`vendor`, `numeric_mode`, `default_mode`, `binding`, `read_vendor`, `select`, `gpu_arch`, `_CODE_MODE`, `_HOST_MODULES`, `host_families_built`, `load_host_module`) | 21 wheel, 2 harness | `_verify.py:406`, `_verify_all.py:579`, `_verify_api_runner.py:57`, `_verify_light.py:209` | One small typed native "what am I running" report: `mojolearn.runtime_report()` gives vendor, arch, numeric mode, route per family, the binding identity (path and sha256) of every loaded owner, and the host families built. Verify reads only that, so it never needs to touch `_backend`. |
| `_backend.binding("_mojolearn_x_sequence")` plus `hasattr(b, "ops_distributed_*"/"ops_partitioned_*")` | 4 | `_verify_distributed_program.py:58`, `_verify_partitioned_program.py:55`, `_verify_model.py`, `_verify_graph.py:1776` | Delete the probe. The public `Program.distribute()`/`.partition()` call that follows already refuses when an entry is missing. |
| **duplicate `native` and `opened`** | 2 copies | `_verify_distributed_program.py:40-55`, `_verify_partitioned_program.py:37-53` (identical bodies) | Move both into one function in `_verify_native_runtime.py` (`audited_context(report, name, operation, ...)`). Both modules import it. |
| `_identity` (`harness_path`, `commit_witness`, `_checkout_root`) | 3 | `_verify_all.py:136,580,4371` | Fields of the runtime report (`harness_source`, `source_commit`). |
| `_families.load_all`, `_buffer`, `_bufcheck`, `_array.Array` | 5 | `_verify.py:97-100`, `_verify_worker.py:26` | `import mojolearn` (public) plus `mojolearn.Array`. Families load lazily through public names. |
| `_cpu_reference.reference_training` | 3 wheel, 1 harness | `_verify_all.py:1018,3366`, `_verify_worker.py:319` | A public device and route argument, `numeric_mode="identical", device="cpu"`, on the fit, or a `runtime_report().route` assertion. |
| `_forest_host.host_model` and `binary_path`, `host._binding` | 3 wheel, 9 harness | `_verify_all.py:1268,1315`, `identity_break.py:4639,10981` | The public saved-model loader plus the runtime report's binding identity. |
| `._extension`, `._blocks[0]._extension` | 2 | `_verify_api_neural.py:64`, `_verify_generation.py:43` | The report's binding identity, or the block's public `vendor_used`/`numeric_mode_used`. |
| `._b`, `._id` (Session and handle internals), `s._transformer_linear`/`_softmax`, `model._forward` | 11 | `_verify_graph.py:50,75,1367-1383`, `_verify_model.py:178-194`, `_verify_api_programs.py:20` | Public `Session` ops and `ImportedProgram.run`. A handle-identity check becomes `Session.state()`. |
| `_causal_lm_fixtures`, `_tokenizer_synthetic`, `_parallel_cv_witness`, `_gpu_witness` | 13 wheel, 7 harness | `_verify_api_models.py:15`, `_verify_causal_lm.py:19`, `_verify_parallel_cv.py:19`, `_verify_distributed.py:47` | These are testing-only modules that sit inside the product package. Move them under verify ownership (`_verify_fixtures_*.py`). This is not a call change. |
| `_native_parallel_shard('faults_take')` with `MOJOLEARN_PAR_TRANSPORT_FAULT` | 3 | `_verify_distributed.py:67,231,264` | Keep it: it is the fault arm itself. Register it as a named fault hook beside the comparison-boundary hook (§2). |
| `_runtime_fallbacks`, `_optional_numpy`, `_x_sequence_model.Sequential`, `parallel_model_selection._capture`, `model_selection._NATIVE_CONTEXT`, `tokenizer._binding`, `importers._program.ImportedProgram`, `_estimator_type` | 8 | `_verify_native_runtime.py:168`, `_verify_parallel_cv.py:192,204`, `_verify_generation.py:107` | Public equivalents (`mojolearn.Sequential`, `ImportedProgram` through `ops`, the report's route), or delete the check. |
| Estimator and model private state in the harness (`_predict_with_proba`, `_write_checkpoint`, `_llama_config`, `_clone`, `_W_NAMES`, `_y_train_mean`/`_std`, `_optimizer_runs`, `_free_values`, `_threaded`, `_parameters`, `_x_sequence_device`, `_portable_math`, ...) | about 55 | `identity_break.py`, `identity_lanes/coverage.py`, `native_owners.py`, `neighbors.py`, `cluster.py`, `cnn.py`, `sequence.py` | Per lane: the public output (`predict_proba`, `save`/`load`, `state_dict`) or a typed report field. A lane that needs private state is testing an implementation, not the API. |

## 5. The lanes, smallest first, by file ownership

1. **`verify-dedupe-opened`.** Files: `python/mojolearn/_verify_distributed_program.py`,
   `_verify_partitioned_program.py`, `_verify_native_runtime.py`. Move `native` and `opened` into one
   `_verify_native_runtime.audited_context` and delete the two copies. Delete the
   `_backend.binding(...)`/`hasattr("ops_*_")` probes, because the public `distribute`/`partition` call already
   refuses when an entry is missing. This touches verify imports (protected): verify must still run both commands.
   Decided by `python/mojolearn/tests/test_distributed_program.py` and `test_partitioned_program.py`.
2. **`endpoint-census-fix`.** Files: `tools/public_runtime_inventory.py` (the `_call` crash at line 1224),
   `tools/verify_coverage_audit.py` (`lane_reach` and `NameClosure` are serial over 513 independent lanes; split
   them across cores with a fork pool), `tools/verification_endpoints.py`, and the regenerated
   `_verification_endpoint_data.py` and `_verification_api_targets.py`. Add census rows for members of exported
   classes that the census misses (decode sessions, `Coordinator`/`Worker`, ONNX inspection). The orchestrator runs
   `--write`. Decided by `tools/verification_endpoints.py --check` exiting 0.
3. **`public-surface-trim`.** Files: the `__all__` lists of `extratrees.py`, `randomforest.py`, `linear_model.py`,
   `density.py`, `neighbors.py` and `decomposition.py`, plus `docs-site/api_exclude.txt`. Stop re-exporting the
   helpers listed in §1, about 40 rows of the 104. This touches no verify code. Decided by
   `python3 tools/public_api_surface.py python/mojolearn` showing no `ForestProtocol`, `NumericModeMixin` or
   `argmax_rows` exports.
4. **`verify-api-routes-gaps`.** Files: `_verify_api_metadata.py` (one route per class for `vendor_used` and
   `numeric_mode_used`, which closes 133), `_verify_api_misc.py` and `_verify_api_neural.py` (decode-session
   `decode`, `state`, `load_state`, `discard`, `close` and `is_open`; scheduler `kind`; lowbit `format`;
   `TokenBatches`), and `_verification_api_targets.py`. Each new route records a captured output, so the reference
   must be re-admitted at the release verification. Decided by `verify --coverage` listing no documented-public
   UNMAPPED rows outside `api_exclude`.
5. **`verify-comparison-fault-hook`.** Files: `tools/identity_break.py` (`_h`, kept as a byte copy in the wheel),
   `_verify_all.py` (`run_cell`, `judge_rows`, `self_test`), `_verify_api.py` (`Recorder`) and
   `_verify_api_runner.py` (`differences`). Add the bit-flip twin digest on every part and every endpoint, required
   to read DIVERGENT, and the sampled one-ULP input arm (one lane per family). Give `training-device-cross-entropy`
   its first arm. Decided by `verify --self-test` reporting a CAUGHT count equal to the number of compared parts.
6. **`verify-fixture-home`.** Files: `_causal_lm_fixtures.py`, `_tokenizer_synthetic.py`, `_parallel_cv_witness.py`
   and `_gpu_witness.py`, renamed to `_verify_fixtures_*.py`, plus their importers in verify and the harness. This
   is a move with no behavior change. Decided by `verify --models-only` and `verify-causal-lm --device cpu` running.
7. **`verify-evidence-one-record`.** Files: `tools/record_identity_column.sh`, `tools/release_identity_*.py`,
   `tools/verify_reference_evidence.py`, `python/mojolearn/_verify_reference.py` (the builder reads
   `bench/results/verification/current/`), `tools/evidence_trim.py` and `.github/workflows/evidence-trim.yml`. This
   implements §3 and makes one tombstone removal of the dated verify directories. Decided by
   `_verify_reference.build_table` over `current/` reproducing the shipped `table.json` byte for byte.
8. **`verify-runtime-report`.** Files: a new native report entry (one binding that `_backend.py` already loads), a
   public `mojolearn.runtime_report()`, and every verify module and harness site in the §4 table. This removes the
   `_backend`, `_identity`, `_families`, `_extension`, `._b` and `._id` couplings. It is the largest lane and
   touches everything verify imports. Decided by `coupling_scan.py` reporting zero product-internal sites in wheel
   verify modules, and by `verify --self-test` passing.

**Removed by this lane:** nothing. It is an audit with no code change. The mechanisms the lanes above would remove
are named in each lane.
