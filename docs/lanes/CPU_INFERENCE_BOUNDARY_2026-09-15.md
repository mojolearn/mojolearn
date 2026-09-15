# Public CPU inference and internal reference training

This implements Andrew's September 15 direction: small development checks,
occasional broad CPU bitwise verification, public CPU inference, and one
Apple/AMD/NVIDIA certification set for PyPI releases.

## Result

- Future wheels ship eight host families: byte_lm, forest, tokenizer, core,
  linalg, estimators, metrics and svm. The 13 training-only families remain
  available as source reference builds. Linux packaging reads this manifest;
  setuptools rejects extra reference binaries in both source staging and
  stale build outputs, removing a rejected newly built wheel.
- Ordinary CPU estimator fit/partial_fit/fit_predict/fit_transform operations
  refuse, including passive k-NN/KDE fits and explicit host models used on
  GPU machines. GPU estimator fitting keeps its existing behavior.
- The private reference context enables source verification and restores the
  inference boundary even after an exception. It does not alter arithmetic.
  Mixed native inference bindings retain private reference helpers.
- The already-published LanguageModelHostTrainer remains supported.
- The public CPU identity command uses seven reference probes supported by
  inference-wheel dependencies. The source harness still covers all 117 lanes.
- Routine CI builds eight host families; weekly/manual/release full CI builds
  all 21. Their native caches have separate keys. No new GPU legs are run.

## Validation

On Apple M4, one numerical process/thread at a time:

- 220 boundary, manifest and CPU reference tests passed.
- All 117 source reference lanes, base fixture, two repeats, four sequential
  shards: unchanged versus the prior result. Training IDENTICAL=117;
  infer/model IDENTICAL=145, N/A=89; batch IDENTICAL=93, N/A=24.
- Built and pip-installed a local test wheel into an isolated target. All
  eight packaged host binaries loaded from that installed target. Ordinary
  CPU LinearRegression/KDE/k-NN fits refused before running.
- Installed-wheel classical saved-model inference matched all 306 recorded
  fixtures across the retained Apple/NVIDIA/AMD recordings.
- Installed-wheel forest inference matched all 24 recorded fixtures.
- Installed-wheel byte-LM training steps 1 and 64 matched all ten recorded
  array comparisons against Apple, with recorded SHA256 checks enabled.
- Packaging refused an injected training-only source binary and a stale
  training-only binary in setuptools' build directory. The prior valid wheel
  remained clean; the newly rejected wheel was removed.
- Workflow YAML/shell syntax, six orchestration tests, docs facts, wheel pins,
  package inventory and manifest readers passed.

## Follow-up: inference-only bindings (lane/inference-neighbors-density)

The eight-family count above is the state this document recorded. Wheels now
ship twelve: the eight, `neural` (lane/inference-tokenizer-neural), `forecast`
(lane/inference-forecast-umap-pca), and, since the neighbors and density
inference lane, `mixture_infer` and `hdbscan_infer`, two more INFERENCE-ONLY
host bindings. A
family whose reference binding carries a fit (mixture, hdbscan) stays a source
build; its scoring or prediction entries move into a shared module
(`bindings/mixture_host_scoring.mojo`, `bindings/hdbscan_host_predict.mojo`)
that both the reference binding and the inference binding register, so the two
binaries answer through one source. The manifest declares the inference family
with `routes=None`, the neural family's pattern, and `mojolearn.host_model`
loads a saved model into a host class that binds it, as the scalers are served
through the estimators binding. On the
M4 the inference files are 232,112 and 227,696 bytes against 406,456 and
359,688 for the reference ones, and `nm` finds no fit symbol in them
(bench/results/identity_break/2026-09-15_inference-iforest-gmm-hdbscan/fit_symbols.txt).
The routine CPU identity gate builds routed families only, so it does not
build these two yet; that workflow change is owed.

The test wheel reused existing local host binaries whose Mojo sources,
build scripts and lockfile are unchanged. No fresh numerical compilation or
GPU qualification was needed for this Python/package-policy change. This
is a local packaging/inference test, not release certification. Hosted CI for
the new commit must still be observed; PyPI publication is not part of this
change. Earlier full hosted run 34978769155 passed on all three CPU hosts.

## Addendum, the same day (lane/inference-tokenizer-neural)

- A ninth shipped family, `neural` (`_mojolearn_neural_host`, loaded by
  path), carries the forward entries of two training-only families:
  `MLPInference` and `TransformerBlockInference`. It exports no training
  entry, and `nm` finds no backward, optimizer or decode symbol in it.
  The same check on the GPU transformer and training binaries finds 15 and
  21, so a zero from this check is meaningful.
- `GPT2Tokenizer.encode_batch` joins the tokenizer family's surface.
- The results above are unchanged by either. The new cells are in
  `bench/results/identity_break/2026-09-15_tokenizer-batch` and
  `2026-09-15_neural-inference`.

## Forecasting, UMAP transform and full-SVD PCA (lane/inference-forecast-umap-pca, 2026-09-15)

- Saved ARIMA models (`ARIMA.save`/`load`, format `mojolearn-arima-1`) predict in sample and
  out of sample, forecast and answer the fitted attributes on a CPU with no GPU. The fit stays
  a reference build: the new host family `forecast` ships `_mojolearn_forecast_host`, which
  registers `arima_predict` and `arima_forecast` from `bindings/arima_host_predict.mojo` and no
  fit. The manifest's `serves` key routes `_mojolearn_arima` to it on a CPU-only install when
  the reference `_mojolearn_arima_host` is not built (`host_surface.inference_routes()`,
  `_backend._HOST_INFERENCE_MODULES`). With the neural family above, ten host families ship.
- Saved UMAP embeddings (`UMAP.save`/`load`, format `mojolearn-umap-1`) transform through the
  already shipped metrics host binding. The transform's answer depends on the query batch
  (umap/transform.mojo: the batch mean sigma floor, the batch maximum edge weight and the
  batch-position negative-sample draws), so the CPU claim is the GPU's bytes for the same batch.
- `pca-full-whiten` joins the inference lanes: a saved `svd_solver='full'` whitened PCA
  transforms and inverse transforms through the estimators host binding.
- `ExponentialSmoothing.fit` refuses on a CPU-only install outside `reference_training()`; it
  is not a `NumericModeMixin`, so the mixin's guard never reached it and it used to run.
- Holt-Winters saved-model inference waits for the line search change to its fit, which moves
  its fitted parameters and hashes.
- Evidence: bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/README.md.
