# Verify harness decoupling, 2026-11-15

Lane `verify-harness-decouple`. Scope: `tools/identity_break.py` (byte-copied into the wheel as
`_identity_break.py`), `tools/identity_lanes/**` and `tools/verify_coverage_audit.py`. Source of the site list:
`docs/plans/VERIFY_COVERAGE_AUDIT_2026-11-15.md` section 4 and its `coupling_product.tsv` (85 rows under `tools/`).

The audit's scanner (`coupling_scan.py` then `coupling_refine.py`), rerun over this branch, reports **44** sites under
`tools/` where it reported 85: 41 removed. Every remaining site is listed below with its reason. Scan output:
`~/mojolearn-evidence/briefs-20261115/verify-harness-decouple/scan/coupling_product.tsv`.

## What replaced each private read

| site (before) | replacement |
|---|---|
| `m._predict_with_proba(X)` (rf-clf, et-clf, et-clf-weighted, train and infer) | `_labels_and_proba(m, X)` = `m.predict(X), m.predict_proba(X)`: two public calls, the same label bytes (predict is the first-max decode of predict_proba). One extra forest traversal in verify only. |
| `_portable_math` in the low-bit oracle (`inf`, `isinf`, `copysign`) | IEEE comparisons in the harness (`_LB_INF`, `_lb_isinf`, `-0.0 if v < 0 else 0.0` for a flushed subnormal). Same values. |
| `_portable_math.fsum`/`sqrt` (x-ann-tsne-pca scaled start) | exact `fractions.Fraction` sums rounded once (what fsum returns) and NumPy's IEEE `sqrt`. Same bits. |
| `_portable_math.exp` (GLM positive targets, `identity_lanes/linear.py`) | `decimal` software exp at 40 digits, rounded to binary64 then f32. See "Digest inputs". |
| `_forest_host.host_model`, `binary_path`, `host._binding` / `est._binding` (3 guards) | `mojolearn.host_model` and `_require_named_forest_binding(ml)`, which compares `runtime_report()["bindings"]["_mojolearn_forest_host"]["path"]` (when `loaded`) with `MOJOLEARN_FOREST_HOST_BINARY`. |
| `m._y_train_mean`, `m._y_train_std` (gp-normalize-y) | the public saved file: `reals[1:3]` of `m.save(...)` (`_saved_fields`), the same binary64 floats. |
| `m.kernel_._free_values()` (gp optimizer lanes) | the saved kernel table `params` and `ls` (every node, binary64). See "Digest inputs". |
| `c._gamma`, `r._gamma`, `f._params[0]` (x-neighbors gamma='scale') | saved `hyper` arrays: SVC `[3]`, SVR `[4]`, RBFSampler `[0]`. Same floats. |
| `BpeTokenizer._synthetic()` (tokenizer lane, lm-corpus refusal) | `BpeTokenizer.from_token_bytes(_tokenizer_synthetic.vocabulary())` (`_synthetic_bpe`): `_synthetic` is that call plus a source label; identity, ids and bytes do not read the label. |
| `base._clone()`, `e._clone()` (ivf-extend train and batch) | `_ivf_copy`: public `save` then `IVFIndex.load`. |
| `dict(zip(est._W_NAMES, est._w))` (CPU inference of the blocks) | the weight dict the lane gave the block constructor, recorded by `_built(blk, weights=...)` at all 12 block constructions; the inference classes take the same constructor arguments. |
| `e._parameters`, `e._threaded`, `e._threads` (`_long_byte_lm`, `_rlpair_byte_lm_host`) | the constructor arguments recorded by `_built` (the two host-infer lanes and `_neural_inference`'s `from_checkpoint`); the rebuilt long model is held to `parameters_sha256()` of the lane model and refused if it differs. |
| `_cpu_reference.reference_training()` around `run` | deleted: its ContextVar is read by nothing ("Kept for callers; CPU fits no longer need it"). |
| `_backend.gpu_arch()`, `_backend.baseline_selection_receipt()` (the PTX column) | `runtime_report()["gpu_arch"]` and `runtime_report()["code_mode"] == "ptx"`. |
| `blk._bns()` (cnn basic block) | the public `bn1`, `bn2`, `downsample[1]`, in `_bns`'s order. |
| `Tokenizer._from_json`, `Tokenizer._from_spelled` (hf-causal-lm-text) | `Tokenizer.from_pretrained(<tokenizer.json>)` and `from_pretrained(<its directory>)`. The fingerprint part does not read the source path (`tokenizer_vocab_fingerprint`). |

## Remaining sites (44) and why

**Verify fixtures that live in the product package (20).** `_causal_lm_fixtures` (`family_fixture`, `_llama_config`,
`_write_checkpoint`, `write_safetensors`) at identity_break 6495-6872 and coverage 344-1133; `_tokenizer_synthetic`
(`byte_to_char`, `vocabulary`) at identity_break 2106 and 6686 and coverage 967. These are testing modules, not
product. Lane `verify-fixture-home` moves them to `_verify_fixtures_*.py`; after that the scanner classes them as
verify-internal. Integrator: `_synthetic_bpe` (identity_break, one new `from mojolearn._tokenizer_synthetic import
vocabulary`) was added on this branch, so it must be renamed with the others when that lane merges.

**Needs product entry (17).** No public read exists; the site is left as it is.

| site | value needed |
|---|---|
| identity_break `host_record` 1302, 1337 (`_backend.host_families_built`, `host_module_path`, `load_host_module`, `_HOST_MODULES`) | `runtime_report()` fields: per host family `{column, detected_column, sabotage}` (each the string/bool the binding's `<prefix>_column()`, `_detected_column()`, `_sabotage()` returns) and `host_routes` (the `_HOST_MODULES` routing table as a dict). `bindings[name].path` already covers the file digest. |
| identity_break 6998, 6999 `ms._NATIVE(None, "family_msel_default_folds", ...)`, `ms._NATIVE_CONTEXT` (cross-val-folds) | a public route to cross_val_score's default folds for an integer cv, e.g. `model_selection.default_folds(y, cv, *, classifier)`. The public `check_cv(...).split` is NOT a replacement: it reaches `msel_stratified_split`, which does not apply `msel_sabotage()` (only `msel_default_folds` does, bindings/msel_split_door.mojo:512), so the msel sabotage arm could no longer move the cell; `tools/test_fold_lane_applicability.py` also holds the lane to this operation. |
| identity_break 2754 `m2._y_cols` (knn-clf) | KNeighborsClassifier: the fitted label columns, int32 `(n_outputs, n_samples)`, e.g. `y_fit_`. |
| identity_break 5839 `m._optimizer_runs` (gp optimizer lanes) | GaussianProcessRegressor: per optimizer run `(iterations, evaluations, ..., log marginal likelihood)`, e.g. `optimizer_runs_`. |
| cluster 43 `m._tree_nodes` | BisectingKMeans: the bisecting tree node table, e.g. `tree_nodes_` (or in `save`). |
| cluster 202 `m._message_diag` | AffinityPropagation: the diagonal of the final message sum, e.g. `message_diag_`. |
| sequence 562 `m._opt_m` | Sequential: the optimizer's first-moment state, e.g. `optimizer_state_["m"]`. |
| coverage 400, 434 (x2) `_x_sequence_device.to_device`, `SequenceDeviceTensor` | a public `mojolearn.to_device` and its tensor type. |
| coverage 1172 `_training_impl.train_to_device` | a public `mojolearn.training.to_device`. |
| coverage 1034 `m._generate_resident` | `CausalLM`: a public resident-generation arm, or a product decision that `generate` is the only route (then the check is deleted). |
| coverage 1110, 1111, 1126 (x2), 1137 `SafetensorsFile._map`, `Checkpoint._file`, `CausalLM._read_weights` | the hf-model-surface lane covers these native doors through their private Python names. Either public names (`SafetensorsFile.mapped_bytes()`, `Checkpoint.shard(file)`, `CausalLM.read_weights(ckpt, plan, fmt)`) or a product decision that they are internal (then the lane's `mapped`, `shards` and `weights` parts and the gathered-head model move to native tests). |

**Fault arms (4).** native_owners 138 (`_SEQ_CONTEXT`, `to_device`) and 165: `_native_owner_failure_cleanup` swaps one
native entry for a builtin to force the allocation-cleanup arm. The audit (section 4) keeps fault arms that reach
internals on purpose; this one is registered as such.

**Tool to tool (1).** verify_coverage_audit 1122 `pri._refusal`: `pri` is `tools/public_runtime_inventory.py`, not the
product (the scanner's "defined in both" is a name clash).

## Digest inputs

- `x-glm-*` lanes (`identity_lanes/linear.py` positive targets): decimal exp replaces the product polynomial. Both are
  within a binary64 ulp of e^x on [-2, 2], so the rounded f32 targets are expected to be the same bytes; if any
  differs, bits of the verify record change for those lanes (the fixture is the same on every column).
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
