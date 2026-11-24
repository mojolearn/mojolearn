# Verify through `device=`: what stays unmapped (vd-verify, lane vdv-ledger, 2026-11-24)

Static: read from the coverage tables and declarations on `lane/vd-verify` (`endpoints.tsv` `verify_lanes` /
`audit_status`, `mapped_cov_{est,generic,nn,parallel}.tsv`, `tools/identity_lanes/coverage_endpoints_declared.tsv`)
and from the `_one_sided(...)` calls in `tools/identity_break.py` and `tools/identity_lanes/*.py`. Nothing was run;
the first recording run decides the generic-fixture n/a set (section 4). Each entry says why it has no lane, or which
side is missing and why. A missing side of a lane not declared one-sided is a failure of the cell, never n/a.

Counts (endpoints.tsv, 4,494 endpoints): mapped 1,591 -> 3,476; not numerical (proposal `n/a: ...`) 953 -> 953;
unmapped 1,950 -> 65 (section 1). One-sided lanes: 41 (34 without a CPU side, 7 without a GPU side; section 2).

## 1. Endpoints still without a lane (65), by reason

### not numerical (44)

| endpoint | reason |
|---|---|
| `mojolearn.AlternatingLeastSquares.fit_transform` | unavailable by design (vdv-cov-est): available_if(transform) and the class has no transform; the call raises AttributeError |
| `mojolearn.BernoulliNB.fit_transform` | unavailable by design (vdv-cov-est): available_if(transform) and the class has no transform; the call raises AttributeError |
| `mojolearn.CategoricalNB.fit_transform` | unavailable by design (vdv-cov-est): available_if(transform) and the class has no transform; the call raises AttributeError |
| `mojolearn.ComplementNB.fit_transform` | unavailable by design (vdv-cov-est): available_if(transform) and the class has no transform; the call raises AttributeError |
| `mojolearn.ConstantKernel.__add__` | not numerical: kernel composition (__add__/__mul__ build a kernel object; the gp lanes compute with the composed kernels) |
| `mojolearn.ConstantKernel.__mul__` | not numerical: kernel composition (__add__/__mul__ build a kernel object; the gp lanes compute with the composed kernels) |
| `mojolearn.EllipticEnvelope.fit_transform` | unavailable by design (vdv-cov-est): available_if(transform) and the class has no transform; the call raises AttributeError |
| `mojolearn.ExperimentalTwoLevelFeatureFreq.predict_classes` | an RMSE-only regressor (vdv-cov-est): the call raises for a non-classification loss |
| `mojolearn.ExperimentalTwoLevelFeatureFreq.predict_proba` | an RMSE-only regressor (vdv-cov-est): the call raises for a non-classification loss |
| `mojolearn.GaussianNB.fit_transform` | unavailable by design (vdv-cov-est): available_if(transform) and the class has no transform; the call raises AttributeError |
| `mojolearn.LanguageModelConfig.__delattr__` | not numerical: dataclass protocol of a config (the byte-lm lane holds LanguageModelConfig(**cfg.to_dict()) == cfg) |
| `mojolearn.LanguageModelConfig.__eq__` | not numerical: dataclass protocol of a config (the byte-lm lane holds LanguageModelConfig(**cfg.to_dict()) == cfg) |
| `mojolearn.LanguageModelConfig.__hash__` | not numerical: dataclass protocol of a config (the byte-lm lane holds LanguageModelConfig(**cfg.to_dict()) == cfg) |
| `mojolearn.LanguageModelConfig.__post_init__` | not numerical: dataclass protocol of a config (the byte-lm lane holds LanguageModelConfig(**cfg.to_dict()) == cfg) |
| `mojolearn.Matern.__add__` | not numerical: kernel composition (__add__/__mul__ build a kernel object; the gp lanes compute with the composed kernels) |
| `mojolearn.Matern.__mul__` | not numerical: kernel composition (__add__/__mul__ build a kernel object; the gp lanes compute with the composed kernels) |
| `mojolearn.MinCovDet.fit_transform` | unavailable by design (vdv-cov-est): available_if(transform) and the class has no transform; the call raises AttributeError |
| `mojolearn.MultinomialNB.fit_transform` | unavailable by design (vdv-cov-est): available_if(transform) and the class has no transform; the call raises AttributeError |
| `mojolearn.OrderedRMSE.predict_classes` | an RMSE-only regressor (vdv-cov-est): the call raises for a non-classification loss |
| `mojolearn.ParallelByteLanguageModelTrainer.model_ownership` | metadata: native allocation byte counts per owned chunk; no numerical identity output |
| `mojolearn.ParallelByteLanguageModelTrainer.optimizer_ownership` | metadata: native ownership and byte counts per device; no numerical identity output |
| `mojolearn.RBF.__add__` | not numerical: kernel composition (__add__/__mul__ build a kernel object; the gp lanes compute with the composed kernels) |
| `mojolearn.RBF.__mul__` | not numerical: kernel composition (__add__/__mul__ build a kernel object; the gp lanes compute with the composed kernels) |
| `mojolearn.WhiteKernel.__add__` | not numerical: kernel composition (__add__/__mul__ build a kernel object; the gp lanes compute with the composed kernels) |
| `mojolearn.WhiteKernel.__mul__` | not numerical: kernel composition (__add__/__mul__ build a kernel object; the gp lanes compute with the composed kernels) |
| `mojolearn.gpu_arch_how.call` | not numerical: a diagnostic text answer about the GPU architecture |
| `mojolearn.linalg.require_identical.call` | not numerical: an identity-mode guard (raises or passes; no value) |
| `mojolearn.model_selection.ParameterGrid.__getitem__` | not numerical: parameter enumeration (x-metrics-splitters enumerates the grids) |
| `mojolearn.model_selection.ParameterGrid.__iter__` | not numerical: parameter enumeration (x-metrics-splitters enumerates the grids) |
| `mojolearn.model_selection.ParameterGrid.__len__` | not numerical: parameter enumeration (x-metrics-splitters enumerates the grids) |
| `mojolearn.model_selection.ParameterSampler.__iter__` | not numerical: parameter enumeration (x-metrics-splitters enumerates the grids) |
| `mojolearn.model_selection.ParameterSampler.__len__` | not numerical: parameter enumeration (x-metrics-splitters enumerates the grids) |
| `mojolearn.models.Checkpoint.__contains__` | not numerical: key membership / config lookup (hf-model-surface already uses `in` and [] on these; the audit walk does not see operators: wrong row) |
| `mojolearn.models.HFConfig.__contains__` | not numerical: key membership / config lookup (hf-model-surface already uses `in` and [] on these; the audit walk does not see operators: wrong row) |
| `mojolearn.models.HFConfig.__getitem__` | not numerical: key membership / config lookup (hf-model-surface already uses `in` and [] on these; the audit walk does not see operators: wrong row) |
| `mojolearn.models.SafetensorsFile.__contains__` | not numerical: key membership / config lookup (hf-model-surface already uses `in` and [] on these; the audit walk does not see operators: wrong row) |
| `mojolearn.numeric_profile.call` | not numerical: numeric-profile settings and readers (NON_COMPUTING in the device view; they select a profile, compute nothing) |
| `mojolearn.numeric_profile_measured.call` | not numerical: numeric-profile settings and readers (NON_COMPUTING in the device view; they select a profile, compute nothing) |
| `mojolearn.numeric_profiles.call` | not numerical: numeric-profile settings and readers (NON_COMPUTING in the device view; they select a profile, compute nothing) |
| `mojolearn.resample.BootstrapResult.__init__` | not numerical: result container constructors (the resample lanes digest the results these hold) |
| `mojolearn.resample.MonteCarloResult.__init__` | not numerical: result container constructors (the resample lanes digest the results these hold) |
| `mojolearn.resample.PermutationTestResult.__init__` | not numerical: result container constructors (the resample lanes digest the results these hold) |
| `mojolearn.set_numeric_mode.call` | not numerical: numeric-profile settings and readers (NON_COMPUTING in the device view; they select a profile, compute nothing) |
| `mojolearn.set_numeric_profile.call` | not numerical: numeric-profile settings and readers (NON_COMPUTING in the device view; they select a profile, compute nothing) |

### not reasonable: inherited control flow on an imported graph (8)

| endpoint | reason |
|---|---|
| `mojolearn.importers._program.ImportedProgram.decode` | not reasonable on an imported inference graph: an inherited Program training/control-flow method needs a captured loss, condition or iteration input that an ONNX import does not have; ops.Program holds them (sequence-tape-program lanes) |
| `mojolearn.importers._program.ImportedProgram.fit` | not reasonable on an imported inference graph: an inherited Program training/control-flow method needs a captured loss, condition or iteration input that an ONNX import does not have; ops.Program holds them (sequence-tape-program lanes) |
| `mojolearn.importers._program.ImportedProgram.fit_if` | not reasonable on an imported inference graph: an inherited Program training/control-flow method needs a captured loss, condition or iteration input that an ONNX import does not have; ops.Program holds them (sequence-tape-program lanes) |
| `mojolearn.importers._program.ImportedProgram.run_if` | not reasonable on an imported inference graph: an inherited Program training/control-flow method needs a captured loss, condition or iteration input that an ONNX import does not have; ops.Program holds them (sequence-tape-program lanes) |
| `mojolearn.importers._program.ImportedProgram.save_control` | not reasonable on an imported inference graph: an inherited Program training/control-flow method needs a captured loss, condition or iteration input that an ONNX import does not have; ops.Program holds them (sequence-tape-program lanes) |
| `mojolearn.importers._program.ImportedProgram.vjp_if` | not reasonable on an imported inference graph: an inherited Program training/control-flow method needs a captured loss, condition or iteration input that an ONNX import does not have; ops.Program holds them (sequence-tape-program lanes) |
| `mojolearn.importers._program.ImportedProgram.vjp_while` | not reasonable on an imported inference graph: an inherited Program training/control-flow method needs a captured loss, condition or iteration input that an ONNX import does not have; ops.Program holds them (sequence-tape-program lanes) |
| `mojolearn.importers._program.ImportedProgram.while_loop` | not reasonable on an imported inference graph: an inherited Program training/control-flow method needs a captured loss, condition or iteration input that an ONNX import does not have; ops.Program holds them (sequence-tape-program lanes) |

### not reasonable: the export does not resolve (3)

| endpoint | reason |
|---|---|
| `mojolearn.models.Tokenizer.unicode_version` | NOT REASONABLE: export does not resolve to an object (fix the export, then recount) |
| `mojolearn.models.catalog.?` | NOT REASONABLE: export does not resolve to an object (fix the export, then recount) |
| `mojolearn.models.verify_catalog_entry.?` | NOT REASONABLE: export does not resolve to an object (fix the export, then recount) |

### owed fixture (2)

| endpoint | reason |
|---|---|
| `mojolearn.BpeTokenizer.from_files` | owed: BpeTokenizer.from_files needs GPT-2 encoder.json + vocab.bpe files; no writer in the package fixtures (a fixture writer is owed) |
| `mojolearn.models.CausalLMState.__init__` | owed: CausalLMState is built by CausalLM.allocate_state with the model as owner (owner=); a caller-built state needs the per-layer block states, no public recipe |

### owed implementation (1)

| endpoint | reason |
|---|---|
| `mojolearn.KernelDensity.sample` | not implemented (vdv-cov-est): the native owner refuses by name; listed in docs-site/api_exclude.txt; owed implementation |

### owed lane: devices= loaders (5)

| endpoint | reason |
|---|---|
| `mojolearn.importers._program.ImportedProgram.distribute` | a devices= partition/distribute loader; vdv-cov-nn left it to vdv-cov-parallel, which did not map it. Owed: a single-device lane (devices=(0,), logical partitions held byte for byte to the plain load), as the psd-* lanes do for the drivers |
| `mojolearn.importers._program.ImportedProgram.partition` | a devices= partition/distribute loader; vdv-cov-nn left it to vdv-cov-parallel, which did not map it. Owed: a single-device lane (devices=(0,), logical partitions held byte for byte to the plain load), as the psd-* lanes do for the drivers |
| `mojolearn.ops.Program.distribute` | a devices= partition/distribute loader; vdv-cov-nn left it to vdv-cov-parallel, which did not map it. Owed: a single-device lane (devices=(0,), logical partitions held byte for byte to the plain load), as the psd-* lanes do for the drivers |
| `mojolearn.ops.Session.load_distributed` | a devices= partition/distribute loader; vdv-cov-nn left it to vdv-cov-parallel, which did not map it. Owed: a single-device lane (devices=(0,), logical partitions held byte for byte to the plain load), as the psd-* lanes do for the drivers |
| `mojolearn.ops.Session.load_partitioned` | a devices= partition/distribute loader; vdv-cov-nn left it to vdv-cov-parallel, which did not map it. Owed: a single-device lane (devices=(0,), logical partitions held byte for byte to the plain load), as the psd-* lanes do for the drivers |

### pytest boundary: framework interop (1)

| endpoint | reason |
|---|---|
| `mojolearn.importers._program.ImportedProgram.run_torch` | framework interop (torch tensors in and out): kept in pytest as boundary interop, adapters get no identity lanes (AGENTS.md) |

### verify tooling (1)

| endpoint | reason |
|---|---|
| `mojolearn.recipe.call` | the verify entry point itself (`python -m mojolearn verify --recipe`); its self-test is the recipe self-test (tools/tests/test_verify_recipe.py), not a lane |

## 2. One-sided lanes (41): the missing side, by name

From the `_one_sided("<lane>", missing=..., reason=...)` calls (registry `ONE_SIDED`). The report reads
"one-sided: no <side> implementation: <reason>" for the missing side.

| lane | missing side | fragment | reason |
|---|---|---|---|
| `saved-model-host-infer` | gpu | identity_break.py | the saved-model readers HostForest, HostGBDT, host_predict and host_predict_proba are the CPU inference surface and take no device |
| `training-device-cross-entropy` | cpu | coverage.py | the CPU training binding has no device array pool (TrainDeviceArray inputs exist on the GPU side only) |
| `fn-host-predict` | gpu | functions_host.py | host_model / host_predict / host_predict_proba are the CPU-inference entry points (host only by decision) |
| `fn-catboost-auto-learning-rate` | gpu | functions_host.py | catboost_auto_learning_rate is a host table lookup that takes no device (host only by decision) |
| `fn-jl-min-dim` | gpu | functions_host.py | johnson_lindenstrauss_min_dim takes no device on vd-functions (closed-form integer bound) |
| `optim-resident-io` | cpu | native_owners.py | the CPU training binding has no resident parameter or gradient buffers |
| `sequence-device-release` | cpu | native_owners.py | the CPU sequence binding has no resident tensor pool to release |
| `nn-ep-samba-inference` | gpu | nn_endpoints.py | SambaInference is the forward-only CPU inference class over the host neural binding; its GPU counterpart is SambaStack (nn-ep-samba-stack) |
| `nn-pg-array` | gpu | nn_programs.py | Array is the host container type (NON_COMPUTING in the device view); its reductions and comparisons run on the host only |
| `nn-tr-optim-resident` | cpu | nn_training.py | the CPU training binding keeps no resident parameter or gradient buffers (params_resident_available is False there by design; optim-io-capability holds that refusal) |
| `psd-logistic` | cpu | parallel_single.py | fit_logistic is a cooperative multi-GPU driver (glm_fit): its gradient columns are tiles inside the GPU binding, which no host binding restates |
| `psd-gram` | cpu | parallel_single.py | fit_gram_estimator is a cooperative multi-GPU driver: its Gram chunks live inside the GPU binding, which no host binding restates |
| `psd-gram-pca` | cpu | parallel_single.py | fit_gram_estimator is a cooperative multi-GPU driver: its Gram chunks live inside the GPU binding, which no host binding restates |
| `psd-cd` | cpu | parallel_single.py | fit_coordinate_descent is a cooperative multi-GPU driver with no host restatement |
| `psd-svm` | cpu | parallel_single.py | fit_svm and predict_svm are cooperative multi-GPU drivers with no host restatement |
| `psd-gp` | cpu | parallel_single.py | fit_gaussian_process and predict_gaussian_process are cooperative multi-GPU drivers with no host restatement |
| `psd-kmeans` | cpu | parallel_single.py | fit_kmeans is a cooperative multi-GPU driver with no host restatement |
| `psd-dbscan` | cpu | parallel_single.py | fit_dbscan is a cooperative multi-GPU driver with no host restatement |
| `psd-hdbscan` | cpu | parallel_single.py | fit_hdbscan is a cooperative multi-GPU driver with no host restatement |
| `psd-gmm` | cpu | parallel_single.py | fit_gaussian_mixture and predict_gaussian_mixture are cooperative multi-GPU drivers with no host restatement |
| `psd-resample` | cpu | parallel_single.py | parallel_classical bootstrap, permutation_test and monte_carlo_integrate are cooperative multi-GPU drivers with no host restatement |
| `psd-cholesky` | cpu | parallel_single.py | fit_cholesky and solve_cholesky are cooperative multi-GPU drivers (cholesky_fit, cholesky_solve) with no host restatement |
| `psd-kernel-ridge` | cpu | parallel_single.py | fit_kernel_method and apply_kernel_method are cooperative multi-GPU drivers (km_fit, km_apply) with no host restatement |
| `psd-nystroem` | cpu | parallel_single.py | fit_kernel_method and apply_kernel_method are cooperative multi-GPU drivers (km_fit, km_apply) with no host restatement |
| `psd-rbf-sampler` | cpu | parallel_single.py | transform_rbf_sampler is a cooperative multi-GPU driver with no host restatement |
| `psd-boosting` | cpu | parallel_single.py | fit_boosting is a cooperative multi-GPU driver with no host restatement |
| `psd-boosting-reg` | cpu | parallel_single.py | fit_boosting is a cooperative multi-GPU driver with no host restatement |
| `psd-iforest` | cpu | parallel_single.py | fit_isolation_forest and score_isolation_forest are cooperative multi-GPU drivers with no host restatement |
| `psd-ordered-rmse` | cpu | parallel_single.py | fit_ordered_rmse is a cooperative multi-GPU driver with no host restatement |
| `psd-feature-freq` | cpu | parallel_single.py | fit_feature_freq is a cooperative multi-GPU driver with no host restatement |
| `psd-forest-pool` | cpu | parallel_single.py | ParallelForestPredictor's groves are resident in the GPU binding; no host binding restates the pool |
| `psd-graph-agglomerative` | cpu | parallel_single.py | fit_graph is a cooperative multi-GPU driver with no host restatement |
| `psd-graph-umap` | cpu | parallel_single.py | fit_graph and transform_umap are cooperative multi-GPU drivers with no host restatement |
| `psd-byte-lm` | cpu | parallel_single.py | ParallelByteLanguageModelTrainer's sessions live inside the GPU byte-LM binding, which no host binding restates |
| `psd-byte-lm-worker` | cpu | parallel_single.py | ParallelByteLanguageModelTrainer's sessions live inside the GPU byte-LM binding, which no host binding restates |
| `psd-byte-lm-model-pool` | cpu | parallel_single.py | ParallelByteLanguageModelTrainer's sessions live inside the GPU byte-LM binding, which no host binding restates |
| `psd-byte-lm-offload` | cpu | parallel_single.py | ParallelByteLanguageModelTrainer's sessions live inside the GPU byte-LM binding, which no host binding restates |
| `psd-byte-lm-corpus` | cpu | parallel_single.py | ParallelByteLanguageModelTrainer's sessions live inside the GPU byte-LM binding, which no host binding restates |
| `psd-causal-lm` | cpu | parallel_single.py | models.ParallelCausalLM keeps devices= (a GPU list) and takes no device=, by the owner's direction of 2026-11-24 |
| `psd-causal-lm-text` | cpu | parallel_single.py | models.ParallelCausalLM keeps devices= (a GPU list) and takes no device=, by the owner's direction of 2026-11-24 |
| `public-numeric-metadata` | gpu | public_gap_methods.py | catboost_auto_learning_rate (host only by decision), johnson_lindenstrauss_min_dim and training.vendor_used take no device=, so the GPU side cannot be selected through the public parameter |

## 2b. One-sided by decision, and owed per the contract

- The 28 `psd-*` cooperative drivers (`fit_logistic`, `fit_gram_estimator`, `fit_coordinate_descent`, `fit_svm`,
  `predict_svm`, `fit_gaussian_process`, `predict_gaussian_process`, `fit_kmeans`, `fit_dbscan`, `fit_hdbscan`,
  `fit_gaussian_mixture`, `predict_gaussian_mixture`, `bootstrap`, `permutation_test`, `monte_carlo_integrate`,
  `fit_cholesky`, `solve_cholesky`, `fit_kernel_method`, `apply_kernel_method`, `transform_rbf_sampler`,
  `fit_boosting`, `fit_isolation_forest`, `score_isolation_forest`, `fit_ordered_rmse`, `fit_feature_freq`,
  `ParallelForestPredictor`, `fit_graph`, `transform_umap`) and `ParallelByteLanguageModelTrainer` have no CPU form:
  owed CPU implementations (a CPU-only install refuses them by design today: `bindings/parallel_plan_run.mojo`,
  `bindings/parallel_byte_native_door.mojo`).
- **ParallelCausalLM's CPU side** (`psd-causal-lm`, `psd-causal-lm-text`): one-sided by the owner's direction of
  2026-11-24 ("ParallelCausalLM keeps devices=", the `parallel_*` drivers take no `device`). It routes from the
  installation (`bindings/parallel_causal_native_run.mojo`), so a CPU side on a GPU installation cannot be selected.
  The CPU side returns only if Andrew reopens `device=` on it. `ParallelCausalLM` is in the device view's
  `NON_COMPUTING` list, so its GPU side runs on the installation's vendor.
- Resident buffers (`training-device-cross-entropy`, `optim-resident-io`, `sequence-device-release`,
  `nn-tr-optim-resident`): the CPU bindings keep no device array pool or resident parameter buffers by design;
  the CPU side is the host array, covered by the lanes' plain forms.
- Host-only by decision (`saved-model-host-infer`, `fn-host-predict`, `fn-catboost-auto-learning-rate`, `fn-jl-min-dim`,
  `nn-ep-samba-inference`, `nn-pg-array`, `public-numeric-metadata`): the CPU inference classes, host table lookups
  and host containers; no GPU form is owed unless the product adds one.

## 3. Mapped, but the cell fails by name until the product gives the side (product gaps)

These endpoints have a lane, so they count as mapped; the lane cannot pass until the product changes or vd-verify
decides. They are failures, not n/a (contract items 3 and 4).

- **`training.Generator` and the six LR schedules** (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, `StepLR`,
  `ExponentialLR`, `OneCycleLR`) take no `device=` and are not in `_verify_device.NON_COMPUTING`, so the strict view
  refuses them: "<name> takes no device=". Until vd-functions gives them `device=` (or vd-verify puts them in
  `NON_COMPUTING`), these lanes fail on both sides: `samba`, `samba-untied-dropout-accum` (identity_break.py),
  `samba-decode-session` (coverage.py), `native-schedule-series` (native_owners.py), `sequence-lr-schedulers`
  (sequence.py), `nn-tr-generator`, `nn-tr-schedules` (nn_training.py), `nn-ep-samba-stack` (nn_endpoints.py).
  `nn-ep-samba-inference` is one-sided, so its view passes them through.
- **Parallel drivers reached through the view** (vdv-fragfix moved them to `ml.<module>.<name>`): `forest-shard`
  (`parallel_ensemble.fit_forest`), `neighbors-reference-shard` (`ReferenceShardedNeighbors`), `ivf-distributed`
  (`DistributedIVFIndex`), `neural-trainer-unfused` (`ParallelNeuralTrainer`), `ordered-gradient-sum`
  (`parallel_training.ordered_sum_gradients`). Their signatures take `devices=`, so the strict view refuses them on
  both sides. The owner's direction of 2026-11-24 keeps `parallel_*` without `device`, so the refusal is permanent as
  written. The `psd-*` lanes import the drivers directly, and vdv-gaps routes a `device='cpu'` member to host workers.
  Decision for vd-verify: pass `devices`-mode drivers through the view when their member carries the side (as
  `psd-*` does), or declare these five `_one_sided(..., missing="cpu")` (`ordered_sum_gradients` has no member; it has
  a host owner in `_mojolearn_runtime_parallel_host`).
- **vd-functions not merged into vd-verify yet** (`origin/lane/vd-functions`): every lane that reaches a function or
  constructor that gets `device=` only there reads "takes no device=" until the merge. Per vdv-cov-func, the GPU
  side's device-resident run of the `fn-*` lanes also fails until vdf-xm-labels / vdf-xm-kernels / vdf-xm-transport
  (metrics), vdf-linalg2 (linalg, training, lstsq, solve, randomized_svd, sparse_encode), vdf-scope-{cluster,neural,trees}
  (model selection) and vdf-misc2 (chi2, resample, kpss_test, select_d, connected_components, refine,
  spectral_embedding) land. `layer_norm_forward` / `layer_norm_backward` need lane/vdf-seq.
- **`device="cpu"` refused on a GPU set** (inventory gap 7): `CalibratedClassifierCV`, `MultiOutputClassifier`,
  `MultiOutputRegressor`, `OneClassSVM`, `OneVsRestClassifier`, `StackingClassifier`, `StackingRegressor`,
  `VotingClassifier`, `VotingRegressor`, `Adam`, `AdamW`, `SGD`, the three explainers (inventory gap 7), and `RFE`, `GridSearchCV` (vdv-cov-est): their lanes' CPU side fails until
  vd-functions connects them (vdv-cov-est doubts).

## 4. Mapped to a generic lane, expected `n/a:generic-fixture` (static prediction; the first record decides)

The registry-driven lanes record `n/a:generic-fixture: <why>` for a class the 64x4 generic fixture cannot build or
fit. The set is known only from a run. At the release's recording, read the n/a parts of `check-generic-*` and
`generic-{params,attrs,pickle,digest}-*` and replace this list with the classes actually recorded n/a. Note that a
class's own constructor or fit exception is recorded as an n/a part that hashes the exception's type and words
(vdv-check addendum), so a one-sided failure shows as a CPU / GPU column difference.

- Predicted by vdv-check (`check-generic-*`): Adam, AdamW, SGD (no fit); Cholesky; ARIMA, AutoARIMA, ETS, DampedETS,
  Theta, OptimizedTheta, DynamicTheta, DynamicOptimizedTheta, AutoTheta, CrostonClassic, CrostonSBA,
  CrostonOptimized, GARCH, VAR, STL, ProphetForecaster (series fit); Sequential (no layers), TransformerBlock,
  Mamba1Block, Mamba2Block, Mamba3Block, Embedding, models.CausalLM, models.ParallelCausalLM (constructor or data
  shape); KernelExplainer, PermutationExplainer, TreeExplainer (require a model); model_selection.GridSearchCV,
  RandomizedSearchCV (require param_grid / param_distributions); CagraIndex, IVFIndex, the IVF*Index classes;
  LabelEncoder, LabelBinarizer, MultiLabelBinarizer (fit(y)); PageRank, Louvain (graphs); AlternatingLeastSquares;
  possibly IsotonicRegression, CategoricalNB, and the RNN / LSTM / GRU / CNN classifiers.
- Predicted by vdv-cov-generic (`generic-*`): HostForest.digest_, HostGBDT.digest_ / feature_importances_
  (CPU-inference readers with no fit; proposal: an extension lane on `saved-model-host-infer` digesting `est.digest_`);
  the optimizers (Adam, Lion, LAMB, ...: `resident_`, `t`, moments), the layers (Embedding, MoEBlock, GCNConv, Conv1d,
  pools), CausalLM / ParallelCausalLM and the tokenizer classes (constructors need parameters or weights; their
  numbers are held by the vdv-cov-nn lanes); splitters that need constructor input (PredefinedSplit test_fold).

## 5. Outside verify

- `tools/aft_idcheck.sh` still takes its CPU digest from a `MOJOLEARN_VENDOR=cpu` child, through
  `bench/speed/forest_speed_arm.py --host-digest`. That is bench tooling, not verify, and it changes when
  `forest_speed_arm.py` takes `device="cpu"` in the same process (vdv-tools; owner: bench).

## 6. Rows the static walk does not credit yet

- `models.HFConfig.__contains__` / `__getitem__`, `models.Checkpoint.__contains__`, `models.SafetensorsFile.__contains__`
  (section 1, not numerical): `hf-model-surface` / `hf-checkpoint` already use `in` and `[]` on them. vdv-gaps taught
  the coverage walk (`tools/verification_endpoint_calls.py`) to record `__contains__` and `__getitem__`; the next run of
  `tools/verify_coverage_audit.py` should credit them.
