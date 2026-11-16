# Verify harness decoupling, 2026-11-15

Lane `verify-harness-decouple`. Scope: `tools/identity_break.py` (byte-copied into the wheel as
`_identity_break.py`), `tools/identity_lanes/**` and `tools/verify_coverage_audit.py`. Source of the site list:
`docs/plans/VERIFY_COVERAGE_AUDIT_2026-11-15.md` section 4 and its `coupling_product.tsv` (85 rows under `tools/`).

The audit's scanner (`coupling_scan.py` then `coupling_refine.py`), rerun over this branch, reports **45** sites under
`tools/` where it reported 85: 40 removed. Every remaining site is listed below with its reason. Scan output:
`~/mojolearn-evidence/briefs-20261115/verify-harness-decouple/scan/coupling_product.tsv`.

## What replaced each private read

| site (before) | replacement |
|---|---|
| `m._predict_with_proba(X)` (rf-clf, et-clf, et-clf-weighted, train and infer) | `_labels_and_proba(m, X)` = `m.predict(X), m.predict_proba(X)`: two public calls, the same label bytes (predict is the first-max decode of predict_proba). One extra forest traversal in verify only. |
| `_portable_math` in the low-bit oracle (`inf`, `isinf`, `copysign`) | IEEE comparisons in the harness (`_LB_INF`, `_lb_isinf`, `-0.0 if v < 0 else 0.0` for a flushed subnormal). Same values. |
| `_portable_math.fsum`/`sqrt` (x-ann-tsne-pca scaled start) | exact `fractions.Fraction` sums rounded once (what fsum returns) and NumPy's IEEE `sqrt`. Same bits. |
| `_portable_math.exp` (GLM positive targets, `identity_lanes/linear.py`) | `decimal` software exp at 40 digits, rounded to binary64 then f32. See "Digest inputs". |
| `_forest_host.host_model`, `binary_path`, `host._binding` / `est._binding` (3 guards) | `mojolearn.host_model` and one helper `_require_named_forest_binding(ml)`: the LOADED path is `runtime_report()["bindings"]["mojolearn._host._mojolearn_forest_host"]["path"]` (when `loaded`), compared, as before, with that row's `expected_path` (the variable, else the install's default binding), set or unset. (Until 2026-11-15 the harness looked the row up under the bare basename, which is never a key, so the guard matched nothing.) |
| `m._y_train_mean`, `m._y_train_std` (gp-normalize-y) | the public saved file: `reals[1:3]` of `m.save(...)` (`_saved_fields`), the same binary64 floats. |
| `m.kernel_._free_values()` (gp optimizer lanes) | the saved kernel table `params` and `ls` (every node, binary64). See "Digest inputs". |
| `c._gamma`, `r._gamma`, `f._params[0]` (x-neighbors gamma='scale') | saved `hyper` arrays: SVC `[3]`, SVR `[4]`, RBFSampler `[0]`. Same floats. |
| `BpeTokenizer._synthetic()` (tokenizer lane, lm-corpus refusal) | `BpeTokenizer.from_token_bytes(_tokenizer_synthetic.vocabulary())` (`_synthetic_bpe`): `_synthetic` is that call plus a source label; identity, ids and bytes do not read the label. |
| `base._clone()`, `e._clone()` (ivf-extend train and batch) | `_ivf_copy`: public `save` then `IVFIndex.load`. |
| `dict(zip(est._W_NAMES, est._w))` (CPU inference of the blocks) | the weight dict the lane gave the block constructor, recorded by `_built(blk, weights=...)` at all 12 block constructions; the inference classes take the same constructor arguments. |
| `e._parameters`, `e._threaded`, `e._threads` (`_long_byte_lm`, `_rlpair_byte_lm_host`) | the constructor arguments recorded by `_built` (the two host-infer lanes and `_neural_inference`'s `from_checkpoint`); the rebuilt long model is held to `parameters_sha256()` of the lane model and refused if it differs. |
| `_cpu_reference.reference_training()` around `run` | deleted: its ContextVar is read by nothing ("Kept for callers; CPU fits no longer need it"). |
| `_backend.gpu_arch()`, `_backend.baseline_selection_receipt()` (the PTX column) | `runtime_report()["gpu_arch"]` and `runtime_report()["code_kind"] == "ptx"` (`code_mode` is the compiled numeric mode and never said "ptx"). |
| `blk._bns()` (cnn basic block) | the public `bn1`, `bn2`, `downsample[1]`, in `_bns`'s order. |
| `Tokenizer._from_json`, `Tokenizer._from_spelled` (hf-causal-lm-text) | `Tokenizer.from_pretrained(<tokenizer.json>)` and `from_pretrained(<its directory>)`. The fingerprint part does not read the source path (`tokenizer_vocab_fingerprint`). |

## Remaining sites (45) and why

**Verify fixtures that live in the product package (20).** `_causal_lm_fixtures` (`family_fixture`, `_llama_config`,
`_write_checkpoint`, `write_safetensors`) at identity_break 6495-6872 and coverage 344-1133; `_tokenizer_synthetic`
(`byte_to_char`, `vocabulary`) at identity_break 2106 and 6686 and coverage 967. These are testing modules, not
product. Lane `verify-fixture-home` moves them to `_verify_fixtures_*.py`; after that the scanner classes them as
verify-internal. Integrator: `_synthetic_bpe` (identity_break, one new `from mojolearn._tokenizer_synthetic import
vocabulary`) was added on this branch, so it must be renamed with the others when that lane merges.

**Needs product entry (18).** No public read exists; the site is left as it is.

| site | value needed |
|---|---|
| identity_break `host_record` 1302, 1337 (`_backend.host_families_built`, `host_module_path`, `load_host_module`, `_HOST_MODULES`) | `runtime_report()` fields: per host family `{column, detected_column, sabotage}` (each the string/bool the binding's `<prefix>_column()`, `_detected_column()`, `_sabotage()` returns) and `host_routes` (the `_HOST_MODULES` routing table as a dict). `bindings[name].path` already covers the file digest. |
| identity_break 6998, 6999 `ms._NATIVE(None, "family_msel_default_folds", ...)`, `ms._NATIVE_CONTEXT` (cross-val-folds) | a public route to cross_val_score's default folds for an integer cv, e.g. `model_selection.default_folds(y, cv, *, classifier)`. The public `check_cv(...).split` is NOT a replacement: it reaches `msel_stratified_split`, which does not apply `msel_sabotage()` (only `msel_default_folds` does, bindings/msel_split_door.mojo:512), so the msel sabotage arm could no longer move the cell; `tools/test_fold_lane_applicability.py` also holds the lane to this operation. |
| identity_break `_require_named_forest_binding` `_forest_host.binary_path()` | `runtime_report()["bindings"][name]["expected_path"]`: the file the process loads or would load, `_backend.door_host_binary_path(name)` (the `MOJOLEARN_<PREFIX>_BINARY` override, else the install's host binding as `host_module_path` resolves it). Done 2026-11-15 (lane smoke-triage-3): the report carries `expected_path` and the private read is gone. |
| identity_break 2754 `m2._y_cols` (knn-clf) | KNeighborsClassifier: the fitted label columns, int32 `(n_outputs, n_samples)`, e.g. `y_fit_`. |
| identity_break 5839 `m._optimizer_runs` (gp optimizer lanes) | GaussianProcessRegressor: per optimizer run `(iterations, evaluations, ..., log marginal likelihood)`, e.g. `optimizer_runs_`. |
| cluster 43 `m._tree_nodes` | BisectingKMeans: the bisecting tree node table, e.g. `tree_nodes_` (or in `save`). |
| cluster 202 `m._message_diag` | AffinityPropagation: the diagonal of the final message sum, e.g. `message_diag_`. |
| sequence 562 `m._opt_m` | Sequential: the optimizer's first-moment state, e.g. `optimizer_state_["m"]`. |
| coverage 400, 434 (x2) `_x_sequence_device.to_device`, `SequenceDeviceTensor` | a public `mojolearn.to_device` and its tensor type. |
| coverage 1172 `_training_impl.train_to_device` | a public `mojolearn.training.to_device`. |
| coverage 1034 `m._generate_resident` | `CausalLM`: a public resident-generation arm, or a product decision that `generate` is the only route (then the check is deleted). |
| coverage 1110, 1111, 1126 (x2), 1137 `SafetensorsFile._map`, `Checkpoint._file`, `CausalLM._read_weights` | the hf-model-surface lane covers these native doors through their private Python names. Either public names (`SafetensorsFile.mapped_bytes()`, `Checkpoint.shard(file)`, `CausalLM.read_weights(ckpt, plan, fmt)`) or a product decision that they are internal (then the lane's `mapped`, `shards` and `weights` parts and the gathered-head model move to native tests). |

**Status, 2026-11-16 (lane `verify-product-entries`).**

| site | status |
|---|---|
| `host_record` | DONE. `runtime_report()` host rows carry `column` and `detected_column` read-backs (beside `sabotage`), and the report has `host_paths` (every built host family -> its file). `host_record` reads those and `host_surface.routed_modules()`; it no longer loads every built family: a family no lane loaded is recorded `loaded: False` with its file bytes, and `column` is over the loaded families (`none-loaded` replaces `none-built`). |
| default folds | DONE. Public `model_selection.default_folds(y, n_splits=5, *, classifier=False)`, one call into the existing `family_msel_default_folds` entry; the lane calls it (sabotage arm unchanged). |
| forest `expected_path` | DONE earlier (smoke-triage-3). |
| knn-clf `_y_cols` | DONE with no product change: the `y_cols` member of the public `save` file (written from the fitted field as is). |
| gp `_optimizer_runs` | DONE. Renamed to the public fitted attribute `optimizer_runs_` (documented). |
| BisectingKMeans `_tree_nodes` | DONE. Renamed to `tree_nodes_` (documented). |
| AffinityPropagation `_message_diag` | DONE. Renamed to `message_diag_` (documented). |
| Sequential `_opt_m` | DONE. `_opt_m`, `_opt_v` renamed to `optimizer_m_`, `optimizer_v_` (documented). |
| `to_device`, `SequenceDeviceTensor`, `train_to_device` | LEFT, decision needed: the product has three resident-tensor types (`_x_sequence_device.SequenceDeviceTensor`, `_training_impl.TrainDeviceArray`, `_expansion_cnn.to_device`'s); publishing each under its own `to_device` would make three public names for one idea. Decide one public resident tensor first. |
| `CausalLM._generate_resident` | DONE (lane `verify-open-sites`, 2026-11-16). Public `CausalLM.generate(..., resident=True)` (documented; default False; one native call, which refuses sampling, EOS, layer-worker and session-less models instead of returning None). The private name on `CausalLM` and `ParallelCausalLM` and the parallel door's always-None `resident` operation are deleted; verify and `tools/lowbit_default/` call the keyword. |
| `SafetensorsFile._map`, `Checkpoint._file`, `CausalLM._read_weights` | DONE (lane `verify-open-sites`). They stay private. Verify reads the same values publicly: `SafetensorsFile.read` per tensor (through the map), each shard as its own `SafetensorsFile` against `Checkpoint.read`, the loader's weights through `CausalLM.load(...).parameters()` in `ModelPlan.layer_weights` layout. The two properties with no public read (the map is the whole file; one cached handle per shard) and `_read_weights`'s dict are held in `test_models_loader.py`: they are Python-object properties of the boundary (mmap, file handles, dicts), which a Mojo native check cannot reach. Digest parts that change: `mapped` (tensor reads, no header bytes) and one `flags` entry (`n_shard_files == 2` replaces the handle identity). |
| identity_break `language-model-config` `_backend.binding("_mojolearn_byte_lm")` | DONE (lane `verify-open-sites`). `profiles` reads `LanguageModelInference.profile` (the host binding that runs the lane's forward); the default flag reads a default inference's `profile`. Same strings on a passing column. |


**Fault arms (4).** native_owners 138 (`_SEQ_CONTEXT`, `to_device`) and 165: `_native_owner_failure_cleanup` swaps one
native entry for a builtin to force the allocation-cleanup arm. The audit (section 4) keeps fault arms that reach
internals on purpose; this one is registered as such.

**Tool to tool (1).** verify_coverage_audit 1122 `pri._refusal`: `pri` is `tools/public_runtime_inventory.py`, not the
product (the scanner's "defined in both" is a name clash).

## Digest inputs

- `x-glm-poisson`, `x-glm-gamma`, `x-glm-tweedie`, `x-glm-poisson-sw` (`identity_lanes/linear.py`
  `_linear_pos_target`): decimal exp replaces the product polynomial. Equality of the f32 targets cannot be shown from
  the arithmetic: the polynomial is not correctly rounded, so for an input whose e^x lies within its error of an f32
  rounding midpoint the two can round differently. **Treat these four lanes' digests as changed and re-admit them.**
  The fixture is the same on every column, so cross-vendor agreement is unaffected.
- gp optimizer lanes (`gp-optimize*`): the `params` part now hashes the saved kernel table (`params`, `ls`, every
  node) instead of the free-entry list: **bits of the verify record change** for that part (the reference must be
  re-admitted at the release verification). It covers strictly more values.
- Every other replacement reads the same bytes through a public route; no digest input changes.

## Product deletions this makes possible (not this lane's files)

With the harness off them, these private entries have no caller outside `python/mojolearn/tests/`:
`RandomForestClassifier._predict_with_proba`, `ExtraTreesClassifier._predict_with_proba`,
`HostForest._predict_with_proba`, `BpeTokenizer._synthetic`. `_cpu_reference.reference_training` is still entered
by `_verify_all.py`, `_verify_worker.py` and several tools, and is a no-op everywhere.

## Removed by this lane

The `reference_training` context around `run`; the three copies of the forest binding guard (one helper now). No
comparison, lane or sabotage arm was removed.
