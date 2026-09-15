# Changelog

This file records release-level changes, not the development diary. Git history and archived evidence
contain the detailed investigation record.

## Unreleased (0.8.6 prep; the freeze commit names it 0.8.6 with the version bump)

Packaging release. Nothing in a kernel moves; what changes is what the two wheels carry and
what a user can check from a pip install. The freeze checks of docs/RELEASE_CHECKLIST.md,
the per-vendor GPU-box build and the byte compare of the host bindings across the three
Linux legs are OWED before this heading reads published.

- Public CPU `Cholesky` inference. On a CPU-only install `Cholesky().fit(A)` factors a given
  matrix and `solve` answers from it, and `Cholesky.save` / `Cholesky.load` (or
  `mojolearn.host_model`, which returns a `HostCholesky`) carry a factor from a GPU box to a
  CPU. The door moved into the linalg host binding, which ships in the inference wheel; the
  `cholesky` identity lane is now the linalg family's and a public CPU reference probe. Apple
  M4 CPU column: train, infer and batch IDENTICAL x4 against the 166-lane record, the new
  saved-factor model cells OWED to the release record, the sabotage build DIVERGENT on every
  train and owed cell.
- New `GPT2Tokenizer.encode_batch(documents, allow_endoftext=False)`, `decode_batch` and
  `decode_bytes_batch`. `encode_batch` is one call into the tokenizer host binding
  (`gpt2_encode_batch`) that encodes each document alone, so every document's ids equal
  `encode` on it; the decode calls loop over `decode_bytes`. The `tokenizer` identity lane
  now carries batch cells over 64 documents instead of `n/a`; its train and infer hashes
  are unchanged against the three committed GPU columns. A new
  `-D MOJOLEARN_TOKENIZER_BATCH_SABOTAGE=1` build must read BATCH_MOVED. Apple M4 CPU
  column only; the GPU columns' batch cells are owed to the release record.
- New public CPU neural inference from GPU-trained weights: `MLPInference` (the small
  8-16-3 MLP's `predict_logits`, from `SmallMLPTrainer.save_checkpoint` files or the four
  weights) and `TransformerBlockInference` (`TransformerBlock.forward` from a zero state,
  full causal or sliding window, ragged `lengths` included). Both run on a new shipped host
  binding, `_mojolearn_neural_host`, that exports forward entries only (no optimizer, loss,
  backward or decode step is compiled in). On a CPU column the `mlp`, `transformer` and
  `transformer-window` identity lanes now ask their held-out and batch cells through these
  classes: against the three committed GPU columns every train, infer, model and batch cell
  reads IDENTICAL (nine fixtures each), and a `-D MOJOLEARN_HOST_SABOTAGE=1` build of the new
  binding reads DIVERGENT on all 27 infer and 27 batch cells with every train cell unchanged.
  Training on the CPU stays internal to the verifier.
- `SVC(kernel='poly', degree, gamma, coef0)`, which was refused by name. The SVM Gram matrix is
  the identical linear GEMM followed by the kernel_methods polynomial epilogue (one fused
  multiply-add, then an ascending repeated product, DEVIATION 1663), so a negative base is
  legal; `degree` is an integer in [0, 32] and `coef0` any finite float. The svm host binding
  and its oracle carry the same arm, both bindings take `degree` and `coef0` in the fit and
  predict parameter lists, and saved poly models record `coef0`. New `svc-poly` identity lane
  (train, infer, model and batch). SVR still refuses 'poly'. Apple M4 Metal and CPU columns
  only; NVIDIA and AMD columns are owed to the release record.
- `GaussianProcessRegressor(normalize_y=True)`, which was refused. It follows the scikit-learn
  reference: y is centered and scaled by StandardScaler's pinned Float32 folds before the fit
  (a zero standard deviation scales by one), and the predictive mean and std are scaled back
  with one correctly rounded binary32 operation each on the host. New `gp-normalize-y`
  identity lane. Apple M4 Metal and CPU columns only; NVIDIA and AMD columns are owed to the
  release record.
- `score(X, y, sample_weight=...)` on `GradientBoostingClassifier`, `GradientBoostingRegressor`,
  the random forests and the Extra Trees, and `sample_weight` on `metrics.accuracy_score` and
  `metrics.r2_score`, all of which refused weights. They follow scikit-learn's reference definitions of weighted
  accuracy (`np.average(y == y_pred, weights=w)`) and weighted R2 (`force_finite=True`) in
  Float32 on the pinned-sum path (`metrics/impl/weighted_scores.mojo`); weights are 1-D,
  finite, non-negative and of positive total. Both metrics bindings export the weighted arms;
  the new `gbdt-adapter-score-weighted` and `rf-score-weighted` identity lanes cover them, with
  the metrics host sabotage build required to read DIVERGENT. Apple M4 Metal and CPU columns
  only; NVIDIA and AMD columns are owed to the release record.
- New `mojolearn.metrics.fowlkes_mallows_score`, following the scikit-learn reference definition (cuML
  has none): the device integer contingency matrix, exact Int64 pair counts, then
  `sqrt(tk / pk) * sqrt(tk / qk)` in Float64, 0.0 when `tk == 0` (no samples, one sample,
  all singletons). It was a named absence. The metrics GPU binding and the metrics host
  binding both export it; the new `metrics-fowlkes-mallows` identity lane covers it, with
  the metrics host sabotage build required to read DIVERGENT. Apple M4 Metal and CPU columns
  only; the NVIDIA and AMD columns are owed to the release record.
- New `HDBSCAN(prediction_data=True)` and `mojolearn.hdbscan.approximate_predict(clusterer,
  points_to_predict)`, mirroring cuML's prediction data and `approximate_predict`: the label
  and probability of new points under the fitted clustering, on the GPU binding and the CPU host
  binding. Without `prediction_data=True` it refuses by name, as cuML does. A tie in mutual
  reachability distance resolves in (distance, index) order (DEVIATION 1615). The fit is
  unchanged: the committed Apple, NVIDIA and AMD train hashes of the `hdbscan` and
  `hdbscan-leaf` lanes still match. Those lanes and `par-hdbscan` now carry infer and batch
  cells instead of `n/a:transductive`, with the batch and host sabotage builds required to
  move them. Apple M4 Metal and CPU columns only; the NVIDIA and AMD cells are owed to the
  release record. `membership_vector` and `all_points_membership_vectors` are not
  implemented and refuse by name.
- `GradientBoosting.fit` takes `group_id`, CatBoost's Pool argument: one string or integer id per
  row (an integer compares by its decimal spelling, as their Pool hashes it), each group's rows
  consecutive or the fit raises "group Ids are not consecutive". The grouping crosses into the GPU
  binding and the GBDT host binding as run lengths, and every loss this implementation trains
  refuses it BY NAME there, because no querywise loss is implemented yet. `subgroup_id` and `pairs`
  are refused by name in Python. A fit without them sends the same parameter layout as before.
- Every host (CPU) binding the manifest declares ships in both wheels under `mojolearn/host/`,
  namely the byte LM's, the forest's, the tokenizer's and the twelve routed families (core,
  linalg, estimators, metrics, preprocessing, tsa, solver, svm, trees, rf, gp, arima), fifteen in
  all. 0.8.5 carried the byte LM's alone. The
  list is read from `python/mojolearn/host_surface.py` by the two wheel builders, the Linux
  packer, both smokes and the Linux admission; `packaging/check_ext_lists.py` (and its
  `--host` mode, which needs no built binary) fails any of them that carries a host list of
  its own. Each binding builds pinned to the CPU kernel-matrix column with no accelerator
  target, reads back as vendor cpu, IDENTICAL and column cpu, and the packer refuses the wheel
  when any leg's copy of any binding differs by a byte from another leg's.
- The Linux copies of the host bindings now get a RUNPATH toward the staged MAX runtime and
  join the closure check (`packaging/linux/stage_libs.py` reached only the tier directories;
  the 0.8.5 Linux wheel's byte LM host binding shipped with whatever RUNPATH the build box
  left in it, and no Linux qualification of that release loaded it).
- `python -m mojolearn verify` works from a pip install because the wheel carries
  `mojolearn/reference_cards/` and a copy of `tools/identity_trace_diff.py`. The reference
  card is still the deliberate placeholder, so `verify` exits 5 and says so rather than
  failing to find its comparator; producing the card is the two-box procedure in
  docs/VERIFY.md.
- New `python -m mojolearn identity`. It runs the identity_break lanes on the local box under
  the identical tier and diffs the column against the three training GPU columns shipped in
  the wheel (the Apple M4, NVIDIA H100 and AMD MI325X columns of the record the manifest
  names, `bench/results/identity_break/2026-09-14_166-lanes` since the 166-lane record, copied to
  `mojolearn/identity_columns/<record>/` with a commit witness), requiring IDENTICAL x4 on
  every train cell it ran and IDENTICAL x4 or N/A on the infer and model cells. On a CPU-only
  install only the lanes with a CPU training path run. `--check` resolves the harness, the
  columns and the witness and runs nothing. Exit codes follow `verify`. Needs numpy.

## 0.8.5 (published 2026-09-14)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) and macOS arm64 wheel, both from
commit 8d16ce2f (tags alpha-api-0.8.5-20260913 and v0.8.5), on PyPI 2026-09-14 00:27Z and 00:36Z
(release runs 34792675705 and 34792714668). Not installed and qualified on GPUs; both wheels
install and import from PyPI on a clean amd64 Linux container and on the Mac. The HIP set was
built on a Hot Aisle MI300X inside the 22.04 ROCm container, as for 0.8.4. The release legs caught
two packaging regressions from the CPU training phase 0 merge, both fixed before the tag: the
Linux and macOS wheel builders now pin the CPU training binding's build to the cpu column (the
binding refuses any other column by name and the legs export the GPU column to every build), and
the two host bindings no longer carry a detected-column read-back, which had folded the build
machine's GPU name into a vendor-neutral binary so the NVIDIA and AMD legs' copies disagreed by
43 bytes and the packer refused the wheel. With it gone the three legs' copies are byte-identical.

- `ExperimentalTwoLevelFeatureFreq` gave different predictions on the three GPU vendors, and
  occasionally two different answers on one machine, because its histogram accumulator was sized
  and zeroed by a hard-coded dead flag while the kernels wrote the live number of cells past its
  end (DEVIATION 2710). The accumulator is sized by the live flag; every vendor now returns the
  same bits on all nine hostile fixtures, proven with the old code selectable beside the fix on an
  M4, an H100 and an MI325X. No other estimator's bits move (the five other gradient boosting
  lanes are identical before and after on every vendor).
- GEMM on AMD runs the gather staging body (`kpack_gs`, the AMD row of DEVIATION 2707): on an
  MI300X the lean language model step reads 0.957 of the previous default and the GEMM sum 0.947,
  every step witness equal and the card identical to the M4's, gated on a shipped build before the
  merge. NVIDIA keeps the `kpack_hg` body that 0.8.4 shipped; Apple compiles the line it compiled
  before.
- The CPU-only forest inference binding (`bindings/_mojolearn_forest_host.mojo`: RandomForest,
  ExtraTrees and the four GradientBoosting variants predict from a saved model on a machine with no
  GPU, the same bits as the GPU that trained it on seven CPUs) is in the source tree and its gate
  workflow, not in either wheel. The wheels carry the byte level language model's CPU training
  binding as in 0.8.4 and no other host binding.

## 0.8.4 (published 2026-09-13)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) and macOS arm64 wheel, both from
commit 0dcc1204 (tags alpha-api-0.8.4-20260913 and v0.8.4), on PyPI 2026-09-13 20:35Z and 20:45Z
(release runs 34781180731 and 34781309801). The Linux wheel was not installed and qualified on
GPUs; the identity evidence for this release is the source build at the same native inventory on
all three vendors, `bench/results/identity_break/2026-09-13_three-columns/`: every public lane on
nine hostile fixtures, 252 training cells, 189 held-out inference cells and 72 saved-model cells,
identical on an Apple M4, an NVIDIA H100 and an AMD MI325X. Both wheels install and import from
PyPI on a clean amd64 Linux container and on the Mac. The HIP set in the wheel was built on a Hot
Aisle MI300X inside the 22.04 ROCm container rather than on the DigitalOcean 24.04 image, because
the 24.04 linker gave the vendor-neutral CPU binding different bytes from the two CUDA legs and the
packer refuses a disagreeing copy (`bench/results/releases/2026-09-13-linux-0.8.4/README.md`).

Both wheels now carry the CPU training binding for the byte level language model, so
`LanguageModelHostTrainer` runs a forward pass, a backward pass and the AdamW update on a machine
with no GPU. The binding ships IDENTICAL only, like every family outside the tree lanes, and it is
the same binary the gate measures. One vendor neutral copy sits at `mojolearn/host/` in the Linux
wheel, beside the architecture trees rather than inside one, because it targets no GPU and reads
back vendor `cpu`.

What the binding does and does not claim is in docs/BYTE_LM_CPU_TRAINING.md. Identity is held per
batch shape rather than across a range, because nine of the weight gradients contract over the
token count, and two shapes are certified today.

There are no auditing switches to turn off for speed. A training step validates its token ids and
zeroes its output buffers, and that is all the Python side does; the finite checks and the state
copies happen once when the trainer is constructed, and the per array digest comparison belongs to
the gate rather than to the shipped class.

- Packaging gates extended to cover the new binary, since no existing check could see it. The
  build inventory counts it apart from the per tier GPU extensions so the architecture counts stay
  exact, the wheel audit admits its member and proves it by digest against the build proof, the
  installed record reads it back through its own path helper, and the release payload must name it.
  A wheel that declared CPU training and shipped no binary, or shipped an inference only build with
  no training entry, now fails qualification rather than reaching a user.

Neural training runs faster on NVIDIA and AMD with no bit moved. Every change below is a schedule
chosen through a kernel matrix row; the arithmetic, the fold order and the words at every address
are the ones the identity cards already pin, and each flip was gated on a shipped build against the
Apple card before it merged.

- GEMM on NVIDIA runs the `kpack_hg` body (DEVIATION 2707): a padded 16 byte aligned packed page,
  one 8 wide conflict free shared store per thread per window instead of sixteen scalar stores at a
  four way bank conflict, and the fold's flush spelled as the one hardware instruction the step seam
  already uses. H100, same pod, every step witness equal to the previous default and the card
  identical to the M4's: lean language model step 0.232 to 0.211 s on enwik8 and Pile GitHub,
  GEMM sum 143 to 122 ms. AMD and Apple compile the line they compiled before; the AMD row is
  measured separately.
- Attention on NVIDIA and AMD keeps the exp stash through the backward (DEVIATION 2657). H100 lean
  step 0.292 to 0.240 s; MI300X 0.757 to 0.736 s. Step glue on both vendors skips the optimizer
  shadow copy and the refuse scan (DEVIATION 2649). H100 0.291 to 0.284 s; MI300X 0.763 to 0.752 s.
  Apple stays on its previous schedule for both, unmeasured as a price.
- The byte level language model's initialization is a pinned function of the parameter index
  (`training/byte_lm_init.mojo`, exact in float32 by construction), and a gate regenerates the
  recorded step 0 parameters of all three vendor captures from it, so a seed, a corpus and a config
  determine the trained bits end to end. docs/BYTE_LM_CPU_TRAINING.md has the argument.
- A GPU resident array (a torch or CuPy tensor, a MAX device buffer) handed to any estimator is now
  refused by name, naming the type and the device, instead of failing later as "not a number"
  (DEVIATION 2692). Accepting device input directly is not started.
- Gradient boosting under IDENTICAL partitions leaves on the device (DEVIATION 2551) on every
  vendor; taxi at 1M rows confirms the bits against the previous default.

## 0.8.3 (published 2026-09-11)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) and macOS arm64 wheel, both from
commit f8b65ee2 (tags alpha-api-0.8.3-20260911 and v0.8.3), on PyPI 2026-09-11 20:17Z and
20:35Z (release runs 34643281339 and 34643372856). The installed Linux wheel passed its
identical qualification jobs on HIP gfx942 and CUDA sm_90a (29 smoke lanes, equal hashes on
both) and fit SVC at 400, 600 and 2,000 rows on an H100 with the source builds' bits; sm_89
was not qualified installed (no RunPod stock), and the release line's fast and deterministic
qualification jobs cannot pass without main's cc117fdf, which would have required new builds.

A patch on the 0.8.2 line. Branch release-0.8.3 starts at tag v0.8.2 (438a6e66) and carries
only the fixes below, their checks and the version bump; main's later speed defaults are not
in it.

- Fixed SVC on NVIDIA GPUs. Every `SVC` or `SVR` fit with more than 512 training rows failed
  on CUDA with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES since 0.8.0: the block solve's working set
  is `min(1024, n_train)`, and CUDA refuses the width 1024 kernel's reduction schedule. A
  kernel-matrix row now gives NVIDIA above width 512 the earlier halving-tree schedule, which
  selects the same elements, so the model bits are unchanged on every vendor (DEVIATION 2623).
  Verified on an H100 (fits at 400, 600 and 2,000 rows equal to the pre-0.8.0 schedule's bits,
  `svm/svc_main.mojo` 44/44 IDENTICAL, which failed seven gates before) and on the Apple M4.
- Fixed IDENTICAL `LinearRegression` on badly scaled designs. The float32 eigensolver's
  squared Frobenius norm overflowed on Gram matrices with eigenvalues near 1e19 and stopped
  before any rotation, and an absolute 1e-10 eigenvalue cutoff made the model's rank depend on
  the data's units. Istella-S (2,043,304 x 220) returned R^2 -115.6 where scikit-learn gets
  0.164. The Gram matrix is now equilibrated by exact power-of-two scales and the cutoff is
  relative, `n * eps32 * max|eigenvalue|` (DEVIATIONS 2620, 2621 for more rows than
  features; 2622 for more features than rows). Istella-S now gives R^2 0.332; taxi keeps R^2
  0.908837 with different coefficient bits. Coefficient hashes are identical across vendors.

## 0.8.2 (published 2026-09-11)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) and macOS arm64 wheel, both from
commit 438a6e66 (tags alpha-api-0.8.2-20260911 and v0.8.2), on PyPI 2026-09-11 18:20Z and
18:35Z (release runs 34632585818 and 34632738047).

A patch on the 0.8.1 line. Branch release-0.8.2 starts at tag v0.8.1 (343ffa35) and carries
only the fix below, its check and the version bump; main's later GBDT speed defaults
(DEVIATIONS 2550, 2551, 2581) are not in it. No change on Apple or NVIDIA: the fixed and
unfixed builds give identical bits there.

- Fixed IDENTICAL GBDT on AMD GPUs. The binary, half-byte and 5-/6-bit histogram kernels
  skipped a block-wide sync on some threads of AMD's 64-lane layout, so a fit on data with
  tied values (few distinct values per column) could differ between runs in one process
  and from NVIDIA and Apple. 0.8.1 moved on the `ties` fixture on an MI300X. Every thread
  now makes the same trips (DEVIATION 2600). Verified: 36/36 identity cells equal to the
  H100 on an MI300X and on the Apple M4, taxi 1M symmetric one hash in 10/10 rounds.
- Added `checks/gbdt_sub_byte_identity_check.py`, which fits the binary, half-byte, 5-bit
  and 6-bit arms twice and compares each to an H100 reference.

## 0.8.1 (published 2026-09-11)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) and macOS arm64 wheel, both from
commit 343ffa35 (tags alpha-api-0.8.1-20260911 and v0.8.1), on PyPI 2026-09-11 12:53Z and
13:08Z (release runs 34601165603 and 34601500930).

A patch on the 0.8.0 line. Branch release-0.8.1 starts at tag v0.8.0
(4a3c22c3, which is the 0.8.0 Linux build commit 9392320e plus two
qualification-tool commits) and carries only this fix, its regression check
and the version bump, because main has changed Random Forest outputs and the
attention default since 0.8.0. The 0.8.0 extension sets are not reused (their
build proofs bind `python/mojolearn/_buffer.py` and `_version.py`), so all
three Linux sets are rebuilt from this branch. No numerics change.

- Fixed buffer conversions in a process that also imports cuML. They were
  all refused with "argument 2: expected LP__PyBuffer instance instead of
  pointer to _PyBuffer", because `treelite.model` retypes the
  `ctypes.pythonapi.PyObject_GetBuffer` function pointer that ctypes caches
  process-wide and `_buffer.py` shared it. mojolearn now takes private
  function pointers for `PyObject_GetBuffer`, `PyBuffer_Release` and
  `PyMemoryView_FromMemory`.
- Added `tools/check_buffer_foreign_argtypes.py`, which reproduces that
  failure without cuML (`--real-cuml` imports cuML instead). The Linux and
  macOS release smokes arm the same foreign argtypes before their fits.

## 0.8.0 (published 2026-09-10)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) from commit 9392320e,
tag alpha-api-0.8.0-20260910; macOS arm64 wheel from tag v0.8.0 at 4a3c22c3.
Both on PyPI 2026-09-11 02:46Z and 02:52Z. Installed-wheel qualification on
the Linux architectures was not run (release policy of 2026-09-09: build,
retag, publish; test what changed). DEVIATION 2500 (labels in the base
binding) landed on main during the build and is not in these wheels.

- Only the three tree families (GBDT, Random Forest, Extra Trees) ship the
  `fast` and `deterministic` tiers. Every other binding, including SVC, SVR,
  isolation forest, k-means, k-NN, PCA, the linear models, UMAP, GP, ARIMA,
  preprocessing and the neural surface, builds and ships `identical` only.
  Asking one of them for a lower tier raises a named error, from
  `numeric_mode=` and from `MOJOLEARN_NUMERIC_MODE`, instead of an
  ImportError about a missing extension. Cross-vendor bitwise identity is
  the product; a fast tier ships only where it has a measured win over the
  opponent's own CPU. This removes 26 extension files from every three-tier
  wheel (13 bindings x 2 retired tiers) and is a minor break from 0.7.0,
  where `KMeans(numeric_mode="fast")` worked.
- Removed the runtime NumPy dependency. Estimators return `mojolearn.Array`
  and accept supported buffer inputs; `numpy.asarray(result)` provides a
  zero-copy view. Shared native conversion, validation and row-gather helpers
  support the GPU paths. New builds and qualification are required; this
  changes the array return API from 0.7.0.
- Added optional GPU `parallel_groves` prediction for Random Forest and Extra
  Trees, sharing resident forest storage, vector-leaf traversal and reusable
  prediction buffers. The default remains `sequential`; the two engines use
  different floating-point reduction orders. Packed-node traversal remains an
  experimental build option.
- Added bounded Random Forest classifier `class_weight` support, per-tree GBDT
  feature sampling and optional minimum child Hessian eligibility for Newton
  depthwise/lossguide growth. Unsupported combinations raise explicitly.
- Added GBDT classifier/regressor adapters and sklearn-style forest parameter
  and scoring protocols. Fitted numeric modes are retained for prediction.
- Added GPU regression errors, classification counts and scores, log loss,
  binary ROC AUC and precision-recall curves, with mode-aware arithmetic.
- Added GPU `StandardScaler` and `MinMaxScaler` with transformer protocols,
  and bounded serial GPU `cross_val_score` support for compatible pipelines.
- Improved automatic neighbor query batching and IDENTICAL wide full PCA.
- Added compiled host buffer conversion helpers and staged the NumPy-free
  buffer core. The estimator layer is not yet NumPy-free.
- Added `LanguageModelConfig` and `LanguageModelTrainer` aliases with
  configurable layer counts and token vocabularies, plus optional resident
  IDENTICAL model/optimizer sessions across Python calls. Inter-layer
  gradients stay on device; attention and prefill workspace allocation is
  reduced. Existing small byte-model defaults remain supported. Large-model
  fit and full-training time remain unqualified.
- The macOS release workflow now includes the IDENTICAL language-model
  extension and checks three-layer/vocab257 resident and stateless training
  in the installed wheel on each supported Python interpreter.
- Extended IDENTICAL IVF selection through k=1024, added the embedding
  backward total-key sort plan with scan/sort identity checks, and routed
  UMAP host math through portable binary64 seams including power.
- Flattened implementation directories and corrected release build policy pins.

## 0.7.0 (published 2026-09-09)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) from commit fe6067ba;
macOS arm64 wheel from tag v0.7.0. Installed per-architecture qualification
was not run for the Linux wheel.

- The identical path no longer calls the host C library. The seven host
  calls for float64 log and log2, float32 log and exp, and ceil (random-forest
  feature rule, boosting loss constant, min-entropy bin construction,
  extremely-randomized-trees builder) use the library's own portable
  implementations, the same code the device runs under identical mode, so
  every host computes the same bits by construction (DEVIATIONS 2260 to 2266).
  Verified on the M4 and an H100: the CatBoost bias oracle matches to the bit
  on every implemented arm, min-entropy and GreedyLogSum borders match CatBoost on
  every case, the RF predict check passes in both modes. Fast-mode bits at
  those three sites move; the fast profile carries no bit promise across
  versions.
- Mamba-3 deterministic mode on CUDA: the stable small-dt softplus
  spelling (float32 log1p of the vendor exp) now covers DETERMINISTIC as
  well as FAST. Before this, DETERMINISTIC evaluated log(exp(x) + 1) and the
  installed qualification failed the Mamba-3 key-state report against the
  float64 reference at one element on both an L40S and an H100, with the
  same excess FAST had shown before its own repair (DEVIATION 2300).
  IDENTICAL bits are untouched; DETERMINISTIC dt bits move on every vendor,
  within its same-box same-build contract.
- One Linux x86-64 wheel carrying CUDA sm_89, CUDA sm_90 and HIP gfx942, each
  in fast, deterministic and identical, plus the identical-mode byte-LM
  trainer extension per architecture (the combined-Linux profile authored
  for 0.6.1, published as 0.7.0). The 0.6.1 version number was never
  published.
- Repository size fences: pre-commit and pre-push hooks under tools/hooks
  refuse oversized blobs and wheels, tarballs or fixture dumps under
  bench/results; the incident is recorded in CONTRIBUTING.md.
- README rewritten around the gap the implementation fills and the identity contract,
  with a project-status section.
- Leg tool: a `checks` family runs named conformance checks on a rented GPU;
  `--allow-concurrent` covers recorded leases; the release-build gate ignores
  bench/results; the pixi installer has a 300 s budget with one retry.

### 0.6.1 (unpublished candidate)

- Version bump, alpha overlay, Linux and macOS packaging, serial job guards
  and release qualification tooling. Superseded by 0.7.0 without a PyPI
  release.

## 0.6.0 (published 2026-09-06)


- Added `UMAP.transform` to embed unseen samples against a frozen fitted model.
  Training input, embedding and fitted parameters are retained privately;
  changed parameters or numeric mode require refitting.
- Public UMAP fitting now stores the fuzzy graph in CSR form, using
  O(n_samples * n_neighbors) graph space. Exact neighbor computation remains
  quadratic; sparse storage is not an approximate-neighbor implementation.
- Preserved named IDENTICAL fit layouts and added held-out transform quality
  checks. Source transform fixtures match across Apple, NVIDIA and AMD;
  installed artifacts are qualified separately before publication.
- Bounded binding compilation to two workers by default, configurable through
  `MOJOLEARN_COMPILE_JOBS`.
- Retained an experimental specialized small-k selector behind an explicit
  build flag. It is not enabled in normal wheel builds.
- Removed the identical path's last host-libm dependency (IDENTITY_PATHS row 18,
  DEVIATION 2260). The eight `external_call` sites for `log`, `log2`, `logf`,
  `expf` and `ceil` in the random forest `max_features='log2'` rule, the extra
  trees host feature sampler, the MinEntropy border penalty and the
  boost-from-average logit now use the library's own portable logarithm and
  exponential (new `portable_log2_64`, exact at powers of two) and an exact
  `ceil`, so those bits are the same on every host and device. The two
  CatBoost-fidelity sites may differ from CatBoost's libm-computed value by one
  ulp on a near-tie; the CatBoost oracle cards are owed a re-baseline run.

## 0.5.0 — 2026-09-05

- Published the macOS arm64 wheel with 15 native extensions in all three modes.
  All Python 3.10–3.14/mode combinations passed isolated installed-wheel checks.
  The downloaded PyPI artifact matched the publication digest and passed smoke,
  Mamba, and Transformer API suites in all modes on Python 3.12 / Apple M4.
  A refreshed Linux wheel remains pending.

- Added `mojolearn.UMAP.fit` and `fit_transform` for dense Euclidean input,
  spectral initialization, and 2D/3D embeddings, with per-estimator numeric modes.
- Reject non-finite UMAP inputs and optimizer parameters before numerical work;
  gate the installed API against the named IDENTICAL layout fixture.
- Retained three-vendor Mamba backward and UMAP source certificates: five Mamba
  cases, 54 gradient tensors, and 186 UMAP stage cells match bitwise on Apple M4,
  NVIDIA RTX 4090, and AMD MI300X at `718495cd`. These are source-fixture claims,
  separate from installed-wheel platform coverage.

- Consolidated active documentation around one roadmap, support matrix, verification guide, and
  normative numerical contracts.
- Added and expanded Mamba, Transformer, training, embedding, and packaging validation lanes.
- Distinguished FAST, deterministic, and IDENTICAL promises across bindings and release tooling.
- Added guarded multi-vendor evidence collection and interleaved FAST/IDENTICAL performance harnesses.
- Added artifact admission checks for wheel contents, digests, platform tags, and native extensions.

## 0.4.0 — 2026-09-02

- Expanded cross-vendor identity coverage across classical ML, linear algebra, tree, sequence, and
  training components.
- Added Mamba 1/2/3 and Transformer forward/backward implementation work and Python bindings.
- Added the 15-extension packaging surface and stricter release refusal checks.
- Added representative price lanes for classical, unsupervised, linear-algebra, and tree workloads.

## Earlier releases

Versions 0.1.0 through 0.3.2 established the Mojo GPU implementation, derivation/refusal ledgers, identity-card
methodology, Python packaging, and the initial Apple/AMD/NVIDIA evidence. Exact changes are preserved
by Git tags and history rather than duplicated here.
