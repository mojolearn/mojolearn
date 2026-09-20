# Byte LM training on a CPU

DEVIATION 2680. The CPU runs one byte LM training step, forward, backward and
the AdamW update, and reproduces the recorded GPU bytes exactly. Two statements,
and the difference between them matters: seven CPUs reproduce every recorded
step when handed that step's own starting state, and one CPU, handed only the
first step's state, free-runs all 128 and ends on the recorded final parameters
and both Adam moments. The second is below under "The free-running run".

## What has been measured

On all seven CI runners (five x86-64 Linux draws, ARM64 Linux on Azure Cobalt
100, and Apple M1 macOS), replaying steps of the retained capture from each
step's own parameters, Adam moments, token ids and recorded optimizer. This is
the per-step statement; the free-running one is a separate section below:

| | |
|---|---|
| CPU step against the recorded Metal bytes | 640 of 640 array comparisons equal, **all 128 steps** |
| arrays compared per step | gradient, loss bits, post-step parameters, post-step `m`, post-step `v` |
| negative control, a wrong-gradient build | caught on every runner, 2 of 10 equal |
| certified CPU inference, unmoved | loss gate 33 of 33, DEVIATION 2612 catch 24 of 33 |

Because `vendors` mode separately shows Apple, CUDA and HIP agreeing on all 128
steps and all 11 arrays, a step that equals the Apple bytes equals all three
vendors. So the CPU step agrees bitwise with AMD and NVIDIA as well, through the
capture rather than through a fresh rental.

**The control fires for the right reason, which is the part worth checking.**
The wrong-gradient build got 8 of 10 arrays wrong and 2 right, and the 2 right
are the loss: a backward-only corruption leaves the forward exact and then
propagates from the gradient into both moments and the updated parameters. The
gate named the tensor, `block0.w_q` element 0, with both bit patterns.
- The GPU trainer class on a CPU-only install, for the internal verifier only
  (2026-09-15, lane/cpu-training-embedding-ivf). Inside
  `_cpu_reference.reference_training()` (which `tools/identity_break.py` enters),
  `SmallByteLanguageModelTrainer` and its alias `LanguageModelTrainer`,
  stateless or `resident=True`, run through
  `python/mojolearn/_byte_lm_trainer_host.py`, which serves the GPU binding's
  single-device entries (the step, evaluation, logits and the resident session)
  over this binding's step, loss and reference-path logits and holds no
  arithmetic; the multi-GPU entries are absent and refuse by name. Outside that
  scope the class refuses on a CPU-only install, as every public CPU fit does;
  `LanguageModelHostTrainer` stays the published CPU trainer. The `byte-lm` and
  `byte-lm-resident` identity_break lanes (three AdamW steps at that time, two
  and one now, see docs/lanes/FIXTURE_SHRINK_SCOPE.md section A3; weight decay
  0.01, the exported gradient, the logits, the checkpoint bytes and the batch
  part) read IDENTICAL x4 on all 72 train, infer, model and batch cells against
  the 166-lane record's Apple M4, NVIDIA H100 and AMD MI325X columns on the
  M4's CPU column (one core), and a build with `-D MOJOLEARN_HOST_SABOTAGE=1`
  (the host GEMM oracle's descending leaf) reads DIVERGENT on all 72. Before
  that change `byte_lm_host_sabotage` did not report that arm, and such a build
  loaded outside the gate reading clean.

## What this does NOT say

- Every step is covered, so this limit is retired. CI runs all 128 on every
  push, on all seven runners. An earlier revision sampled `every:16` because a
  replay was estimated at ten seconds per step; measured, the full job takes
  2m23s against the sample's 2m6s, so the sample bought seventeen seconds and
  cost 119 steps.
- One model profile, one batch shape. Nine of the gradients contract over the
  token count, so the same tokens in a different batch or microbatch schedule
  are a different sum. Identity here is per shape, exactly as inference is.
  This limit does not retire when a second shape passes. Each shape is its own
  certificate, so a third shape would be a third certificate. Everything
  measured above is the two-row shape `b2-l32` and nothing else.
- A second shape now holds, and it is a SECOND certificate rather than a wider
  first one. DEVIATION 2682 moves the shape into one shared helper,
  `tools/byte_lm_shape.py`, gives the capture harness and the verifiers a
  `--shape` argument, and commits the four-row schedule as
  `training/corpus/tinyshakespeare/manifest-b4-l32.json`, batch 4 length 32,
  128 tokens a step against the default's 64, over the same pinned corpus and
  the same 512 held-out target bytes.

| | |
|---|---|
| CPU step against the recorded four-row CUDA bytes | 640 of 640 per runner, **4480 of 4480** across seven CPUs, all 128 steps |
| capture | NVIDIA L40S, `bench/results/resume/2026-09-12-root-byte-lm-b4l32-nvidia` |
| held-out loss over the run | 5.5412992 to 2.7471306, ratio 0.4958 against a 0.9 threshold fixed beforehand |
| negative control at this shape | caught on every runner, 8 of 10 wrong, the 2 right being the losses |
| step cost | 44.5 to 72.5 ms on the Linux draws, 106.2 ms on Apple M1 |

  **One vendor.** That tree is CUDA on an L40S, so this is a CPU against CUDA
  statement at this shape. AMD and Apple at four rows are not captured and are
  not claimed, which is the difference between this shape and the two-row one,
  where the three retained vendor trees agree and byte equality is transitive.
  The single-vendor learning admission is also still owed here, because the
  tree was captured before the source inventory was fixed and genuinely lacks
  an entry the comparator requires.
- The reference path only, one thread, and that is a deliberate stop rather
  than an unfinished job. **Measured on 2026-09-12 across the seven runners of
  run 34701581834: a whole step is 23.5 to 38.7 ms on the Linux draws and
  68.9 ms on Apple M1**, forward, backward and the AdamW update, so all 128
  steps replay in under five seconds. Threading it would need
  two fast GEMM orientations that do not exist (the host fast kernel is NT only;
  the backward needs NN and TN) and leaf-parallel folding to keep the weight
  gradients' summation order exact, because they sum across every row and so
  cross every thread boundary. That is a large build whose payoff is making an
  already-fast thing faster, against a real risk to the bit identity that is the
  point. It should be justified by a workload that 37 ms a step makes painful,
  not by the limit's existence.
- Nothing about other algorithm families. Trees and the classical models have
  no backward pass; the CPU paths they do have are the host bindings in the
  table under "Where this sits in the CPU surface" below, not this trainer.
- `LanguageModelHostTrainer` is exported from `mojolearn/__init__.py` as of
  `31af2404`. What it promises is this profile and this shape, and a binding
  built before the training entry existed is refused at construction by name.

## The free-running run

The table above replays each step from its own recorded starting state, which is
what makes a disagreement at step 87 debuggable without the 86 steps before it.
It leaves one thing unsaid: whether a CPU left alone for a whole run arrives
where the GPU did. That should follow from the table by induction, since
`post_p`, `post_m` and `post_v` all agree at every step and so step N's output
is step N+1's input, but an induction argument is not a measurement and a
re-seed is exactly where a drift would hide. So it was run.

`tools/byte_lm_cpu_train_freerun.py` reads step 1's `initial_p`, `initial_m` and
`initial_v` and no recorded state after that. The model keeps its own parameters
and moments for the rest of the run; per step it reads only `ids`, because token
ids are the input data a training run consumes rather than state. The optimizer
is fixed at construction, so the tool refuses a capture whose steps disagree on
any hyperparameter rather than running 128 steps under step 1's numbers, and it
refuses a non-contiguous step range, which would feed one step's tokens to a
model that never consumed the step before it.

| | |
|---|---|
| free-running steps, no re-seeding after step 1 | 128 |
| arrays compared per step | 5: gradient, loss bits, post-step parameters, `m`, `v` |
| comparisons per vendor | **640 of 640 equal** |
| vendors | Apple Metal, NVIDIA CUDA and AMD HIP, 640 of 640 each |
| final state after step 128 | `post_p`, `post_m` and `post_v` all equal to the recorded bytes |
| recorded digests verified per run | 1408, being 128 steps of 11 arrays |
| step cost | 36.7 to 37.0 ms, reference path, one thread |
| whole run | 4.7 s |

The four-row shape free-runs too, and it is a second certificate rather than a
widening of the first, for the same reason the per-step result is: nine weight
gradients contract over the token count.

| | `b2-l32` | `b4-l32` |
|---|---|---|
| free-running comparisons | 640 of 640, three times | 640 of 640 |
| vendors | Apple, CUDA and HIP | CUDA only, no other capture exists |
| final `post_p`, `post_m`, `post_v` | all equal | all equal |
| step cost | 36.7 to 37.0 ms | 83.5 ms |
| whole run | 4.7 s | 10.7 s |

The step cost rises 2.2x for 2x the tokens, which is the shape doubling and not
a path that quietly reused the two-row batches.

So the per-step result composes. A CPU given the initialization a GPU started
from, and the same token stream, ends on that GPU's parameters and both Adam
moments bit for bit: on all three vendors at two rows, and on CUDA at four.

**One CPU, not seven.** These four runs are the bare-metal Apple M4 at commit
`8c8a500c`, host binding `eaada7a1`. The seven-runner result above is the
per-step one and stays that way; CI does not free-run yet. Widening this to the
other six runners is measurement, not construction.

Both captures recorded the same optimizer configuration, so one construction per
run is valid and the tool's refusal for a capture whose hyperparameters move
between steps has never fired. That guard is written, not exercised.

**What falsifies it.** The free run compares the same five arrays the `cpu` mode
of the gate compares, so the GEMM backward arm that fires there corrupts the
same arithmetic here. For compounding specifically, flipping one mantissa bit of
step 1's `initial_p` diverges `post_p` at step 1 and never recovers, ending on
different final parameters, with the difference localized to `embed` element 0
and tracking one ULP for all 128 steps. Both moments stayed equal under that
perturbation, so it is evidence about parameter propagation and says nothing
about the `post_m` and `post_v` comparisons. Those two are falsified only
through the gradient today: no sabotage arm reaches the AdamW update on this
path, because the host step calls `optimizer_step_oracle` and the twentyone
`MOJOLEARN_OPT_SABOTAGE_*` arms live in the device file, not the oracle. An arm
inside the host oracle, gated so it can never compile into a shipped binary and
reported by `byte_lm_host_sabotage`, is what would close that and it does not
exist.

## The initialization is a seed, not a gift

The free run above still reads step 1's parameters out of the capture, so the
statement it supports is "given these starting bytes". A reader can fairly answer
that the run was handed its initialization. It was not: the starting bytes are a
pinned deterministic function of an index, and the three retained trees all
record it under the same identifier,
`u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1`.

    h = fmix32((i + 1) ^ 0x42595445)          # Murmur3 finalizer, UInt32
    value = Float32(Int(h >> 24) - 128) * 2^-10
    # then the four RMS norm vectors are overwritten with exactly 1.0

**This is bit-exact by construction rather than by a measurement that passed.**
`h >> 24` is eight bits, so the numerator is an integer in [-128, 127] and the
divisor is a power of two; both are exactly representable in FP32 and the
quotient is exact, so there is nothing to round differently on another vendor.
There is no accumulator, so no fold order exists to disagree about, and no
transcendental, which is where cross-vendor agreement actually breaks
(DEVIATIONS 2260 to 2266). Element `i` depends on `i` alone, so no thread layout
can move a value. The draw lands on a 257-value dyadic grid: 256 points spanning
[-0.125, 0.125] plus the 1.0 the norm vectors carry.

`training/byte_lm_init.mojo` is the library surface and
`tools/byte_lm_seeded_init_gate.py` is the gate. The gate carries an INDEPENDENT
reimplementation rather than importing the generator in
`tools/byte_lm_real_text_capture.py`, because comparing that function against the
bytes it produced would prove nothing.

| | |
|---|---|
| vendor trees reproduced from the seed | Apple, CUDA and HIP, all three |
| parameters per tree | 34944 of 34944 bytes equal |
| recorded `initial_parameters_sha256` | `b87a6075...` on all three trees |
| distinct values | 257, the dyadic grid |
| **Mojo host implementation, full vector** | sha256 `b87a6075...`, byte-identical to the recorded `initial_p.f32` |
| negative control, XOR constant off by one | refused on all three, naming `embed` element 0, `3d080000` against `bdb00000` |

So the chain closes: a seed fixes the initialization, the initialization and the
token stream fix the trained weights, and both halves hold on a CPU and across
the three vendors. **No new capture and no rented GPU were needed** -- the
retained trees already recorded a reproducible initialization; nothing had
written down that it was reproducible.

**Two limits.** The seed here is the XOR constant inside a fixed index hash, so
this says the initialization is a pinned deterministic function, not that an
arbitrary seed gives a cross-vendor identical draw; that would need the hash
swept over many constants. And the proof above is the HOST path. Vendor agreement
comes from the three captures sharing one recorded digest, not from running a
device kernel; a device initializer that must equal the host vector is owed.

The run behind the table is retained in full at
`bench/results/gh-actions/2026-09-12_1513-byte-lm-cpu-gate-run34701581834`,
one directory per runner, each holding its own `cpu_train.json`,
`cpu_train_sab.json`, `train_capture.json`, the two inference gate reports and
the build logs. `local-three-vendor-train-capture.json` beside them is the
`vendors` mode run over all three retained trees, which the runners cannot do
because CI sparse-checkout fetches one.

## Why a gate came first

The retained three-vendor capture
(`bench/results/resume/2026-09-07-root-byte-lm-three-vendor`) holds, for every
one of the 128 training steps and on each of Apple Metal, NVIDIA CUDA and AMD
HIP, the full FP32 tensors of that step. The parameters, both Adam moments and
the state flags before the step, the token ids it consumed, the loss, the
gradient, and the parameters, moments and flags after the update. Nothing is
sampled and nothing is reduced to a digest; the digests in each `capture.json`
sit on top of the bytes.

So a CPU training step can be certified against recorded bytes from three
vendors without renting a GPU. Each step records its own starting state, so a
step is judged in isolation and a disagreement at step 87 needs no replay of
the 86 before it.

Until today nothing read those gradients. `tools/byte_lm_host_gate.py` reads
parameters and loss, and takes `comparison.json`'s `identity_admitted` flag on
trust for everything else. The one comparison that did cover gradients and
moments was root-only evidence code that ran once and compared the vendors to
each other.

## The gate

`tools/byte_lm_cpu_train_gate.py`, two modes.

`vendors` re-derives the three-vendor agreement from the raw bytes. For each
selected step, every array of every pair of vendor trees must be equal byte for
byte, and every array must match the SHA-256 its own `capture.json` records.
This needs no GPU, no binding and no build, and runs in about a second.

| | |
|---|---|
| steps | 128 of 128 |
| arrays per step | 11, including `grad`, `post_m`, `post_v` |
| comparisons | 4224 of 4224 equal |
| recorded digests verified | 4224 |
| vendors | Apple M4 Metal, NVIDIA CUDA, AMD MI325X HIP |

`cpu` is the gate proper and it runs. It replays selected steps through
`LanguageModelHostTrainer` and compares `grad`, `loss`, `post_p`, `post_m` and
`post_v` against one vendor tree, taking each step's optimizer configuration
from that step's own `capture.json` rather than from defaults. If the surface is
absent it still refuses with exit 2 and names what has to appear, rather than
reporting a pass over nothing.

`--expect-mismatch` inverts the verdict, which is how the wrong-gradient build
is required to be caught. CI runs both arms on every push: the clean binding
must agree, and a binding built with
`-D MOJOLEARN_GEMM_SABOTAGE_BWD_UNTRANSPOSED=1` must disagree.

That arm was chosen because **DEVIATION 2612's arm cannot reach this path.**
2612 reverses a fold inside `byte_host_logits`, and the training step calls
`gemm_oracle` directly, so a 2612 build computes a correct training step. Until
this control existed the training gate had never been shown capable of failing.
`byte_lm_host_sabotage` was widened in the same change, because it reported the
2612 flag alone and a binding carrying a GEMM backward arm would otherwise
compute wrong gradients while reading back as clean.

A difference is reported by tensor. The registry is a fixed order of 21
tensors, so a flat element index is localized to a name, an index inside that
tensor and the two IEEE-754 bit patterns.

## What a pass will and will not say

A pass is a statement about this model profile
(`mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1`), this batch
shape and this optimizer configuration, which the byte LM trainer restricts to
plain positive-lr AdamW.

It will not say anything about other batch schedules. The weight gradient
contracts over the token count, and `gemm/checks/gemm_backward.mojo` states the
consequence directly: the gradient at 1024 tokens is not the same bits as the
gradient at 512 tokens accumulated twice, and cannot be under any fixed
partition, because the two are different sums over different partitions. The
microbatch schedule is part of a training run's numerical specification. So CPU
training identity is claimed per shape, exactly as the inference sweep claims
it per shape.

The harness and the verifiers now take a `--shape` argument and a second
schedule is committed, which changes what can be attempted and changes nothing
about what has been shown. A pass recorded for one shape is evidence for that
shape alone. The `b4-l32` certificate is owed.

## What is still missing

That table was wrong when first written and is corrected here. Almost none of
it was missing. The decoder block backward exists in full as a host oracle,
`transformer/checks/transformer_backward_oracle.mojo`, 1598 lines, all 37
stages, alongside host RMS norm, SiLU, RoPE and softmax backward and the host
GEMM backward routing. `training/checks/train_step_check.mojo` already composes
an entire host step out of these and reports thirteen stages compared against
the device bitwise with four negative controls firing, at a one block fixture
with its own registry, on one device.

| Piece | Status |
|---|---|
| AdamW | normative host oracle, `training/checks/optimizer_oracle.mojo`, seams O1 to O14 |
| Cross-entropy backward | normative host oracle, `training/checks/loss_oracle.mojo` |
| Embedding gradient | normative host oracle, fixed ascending fold over sorted runs |
| GEMM backward | host routing over the reference GEMM, `_gemm_bwd_a` / `_gemm_bwd_b` |
| Decoder block backward, 37 stages | normative host oracle, already written |
| Byte LM shaped composition, two blocks | `training/byte_lm_host_backward.mojo`, compiles and **runs correctly on seven CPUs** |
| Binding entry | `byte_lm_host_train_step`, eight addresses, five AdamW scalars, returns the loss bits |
| Python surface | `LanguageModelHostTrainer`, unexported on purpose |
| The `cpu` mode of the gate | runs, and its negative control fires |

Nothing in the list above is now missing. What remains is coverage rather than
capability. One profile, one certified batch shape, and the reference path
only. The step sampling limit is gone, because CI replays all 128 on every
push. An earlier revision of this paragraph still said nine of 128 and was
stale. Widening what is left is measurement, not construction, and the second
shape is the case in point. Its schedule and its `--shape` plumbing are
committed, and its capture has not been taken, so nothing here counts it.

## Where this sits in the CPU surface

The byte LM host binding is one of the host families the package builds
for a CPU. The whole surface is declared once, in
`python/mojolearn/host_surface.py`, and this table is generated from it by
`tools/docs_facts.py --write` (`pixi run check-docs-facts` fails when they
disagree). The next wheels include **every** family in the table, built
by the two wheel builders through the same shims, read back
as vendor cpu and the CPU column, and (on Linux) byte-compared across the
three architecture legs before packing; 0.8.5 and earlier carry only the
byte LM's, and 0.8.6 was folded into 0.8.7 and never published, so 0.8.7 is
the first published release to carry all thirty-two. The sixteen training
families joined on 2026-09-16 (lane/ship-cpu-host-families): with them out,
an installed wheel could check 39 lanes, because a lane's host binding was
not in it; with them in, a measured `verify --all` checks 122. Do not quote
a registry total beside those two. It moves as lanes are added, and it moved
the same day: `tools/lane_select.py --count` read 211 in the morning and 212
by the evening. Ask the registry instead. Shipping the binding did not make
ordinary CPU `fit` public; it still refuses outside the verifier's scope. Each also builds from source through
`bindings/build_host_family.sh`, one shim per family.

<!--fact:host_surface_table-->| family | binding under `mojolearn/host/` | routes (CPU-only install) | internal CPU reference lanes | predicts on a CPU from a saved model | gate | in a wheel |
|---|---|---|---|---|---|---|
| byte_lm | `_mojolearn_byte_lm_host.so` | loaded by path | byte-lm-host-infer, byte-lm-host-infer-threaded, byte-lm-host-train, byte-lm, byte-lm-resident, language-model-config | LanguageModelInference, LanguageModelHostTrainer, SmallByteLanguageModelTrainer | .github/workflows/byte-lm-cpu-gate.yml and tools/identity_break.py (cpu-identity-gate.yml) | yes |
| forest | `_mojolearn_forest_host.so` | loaded by path | gbdt-categorical-ctr-tables, gbdt-tensor-ctr-tables, saved-model-host-infer | RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoosting, OrderedRMSE, ExperimentalTwoLevelFeatureFreq (rf_classifier, rf_regressor, et_classifier, et_regressor, gbdt_symmetric, gbdt_depthwise, gbdt_lossguide, gbdt_rmse, gbdt_ordered_rmse, gbdt_feature_freq, gbdt_pointwise_bayesian_eval, gbdt_categorical_onehot) | tools/forest_host_gate.py (.github/workflows/forest-host-gate.yml) | yes |
| tokenizer | `_mojolearn_tokenizer_host.so` | loaded by path | tokenizer, bpe-trainer, bpe-vocabulary, tokenized-corpus, hf-tokenizer | BpeTokenizer | pixi run check-tokenizer and python/mojolearn/tests/test_tokenizer_surface.py | yes |
| neural | `_mojolearn_neural_host.so` | loaded by path | hf-causal-lm, par-causal-lm | MLPInference, TransformerBlockInference, Mamba1BlockInference, Mamba2BlockInference, Mamba3BlockInference, SambaInference | python/mojolearn/tests/test_neural_inference.py, tools/step_vs_full_check.py and tools/identity_break.py (mlp, transformer, transformer-window, transformer-decode, mamba1, mamba2, mamba3, mamba1-decode, mamba2-decode, mamba3-decode, mamba2-dtlimit, samba, samba-decode, samba-untied-dropout-accum) | yes |
| core | `_mojolearn_core_host.so` | `_mojolearn` | knn, knn-clf, knn-reg, kmeans, kmeans-random, kmeans-array, kmeans-weighted, knn-sqeuclidean, knn-clf-distance, knn-reg-distance, knn-manhattan, knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc, radius, radius-manhattan, radius-chebyshev, radius-minkowski-p3, kmeans-sqrt, kmeans-classic-pp, par-queries-knn, par-queries-nn, par-queries-radius, par-reference-knn, par-reference-knn-reg | NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, KMeans, RadiusNeighbors (knn, knn-clf, knn-reg, knn-sqeuclidean, knn-manhattan, knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc, knn-clf-distance, knn-reg-distance, radius, radius-manhattan, radius-chebyshev, radius-minkowski-p3, kmeans, kmeans-random, kmeans-array, kmeans-weighted, kmeans-sqrt, kmeans-classic-pp) | tools/classical_host_gate.py (cpu-identity-gate.yml) | yes |
| linalg | `_mojolearn_linalg_host.so` | `_mojolearn_linalg` | gemm-pinned, gemm-transposed, cholesky, gemm-bf16, gemm-int8, linalg-qr, linalg-eigh, linalg-svdvals, lowbit-conversions, hf-checkpoint | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| estimators | `_mojolearn_estimators_host.so` | `_mojolearn_estimators` | kde, pca, pca-whiten, tsvd, ols, ridge, dbscan, logistic, dbscan-brute-l1, kde-tophat-sqeuclidean, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-cosine-minkowski, kde-weighted, ols-no-intercept, ols-weighted, ridge-no-intercept, logistic-unpenalized-no-intercept, dbscan-weighted, logistic-l1, logistic-elasticnet, logistic-multiclass, pca-full-whiten, par-queries-kde, linear-svc, linear-svc-squared-hinge, linear-svr, linear-svr-squared, qn-squared, qn-absolute | LinearRegression, Ridge, TruncatedSVD, LogisticRegression, PCA, KernelDensity, DBSCAN, StandardScaler, MinMaxScaler, Lasso, ElasticNet, KernelRidge, Nystroem, RBFSampler, AgglomerativeClustering, LinearSVC, LinearSVR, QNRegressor (ols, ridge, tsvd, logistic, logistic-multiclass, pca, pca-whiten, kde, ols-no-intercept, ols-weighted, ridge-no-intercept, logistic-l1, logistic-elasticnet, logistic-unpenalized-no-intercept, standard-scaler, standard-scaler-no-mean, standard-scaler-no-std, minmax-scaler, minmax-scaler-clip, lasso, elasticnet, elasticnet-l2end-no-intercept, kernel-ridge, nystroem, rbf-sampler, pca-full-whiten, kernel-ridge-poly, kernel-ridge-sigmoid, kernel-ridge-laplacian, nystroem-poly, nystroem-sigmoid, nystroem-laplacian, kde-tophat-sqeuclidean, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-cosine-minkowski, kde-weighted, dbscan, agglomerative) | tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml) | yes |
| metrics | `_mojolearn_metrics_host.so` | `_mojolearn_metrics` | metrics, spectral, spectral-precomputed, umap, metrics-classification, metrics-fowlkes-mallows, metrics-homogeneity-completeness, spectral-embedding | SpectralClustering, SpectralEmbedding, manifold.spectral_embedding, UMAP, metrics.accuracy_score, metrics.adjusted_rand_score, metrics.entropy, metrics.mutual_info_score, metrics.homogeneity_score, metrics.completeness_score, metrics.v_measure_score, metrics.r2_score, metrics.silhouette_score, metrics.silhouette_samples, metrics.rand_score, metrics.precision_score, metrics.recall_score, metrics.f1_score, metrics.log_loss, metrics.roc_auc_score, metrics.confusion_matrix, metrics.precision_recall_curve, metrics.mean_squared_error, metrics.mean_absolute_error, metrics.root_mean_squared_error, metrics.kl_divergence, metrics.trustworthiness, metrics.fowlkes_mallows_score (umap, spectral, spectral-precomputed) | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| preprocessing | `_mojolearn_preprocessing_host.so` | `_mojolearn_preprocessing` | standard-scaler, minmax-scaler, standard-scaler-no-mean, standard-scaler-no-std, minmax-scaler-clip, par-scaler, par-scaler-minmax | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| tsa | `_mojolearn_tsa_host.so` | `_mojolearn_tsa` | holtwinters, holtwinters-multiplicative, kpss, select-d, par-holtwinters, par-forecast-holtwinters | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| solver | `_mojolearn_solver_host.so` | `_mojolearn_solver` | lasso, elasticnet, agglomerative, elasticnet-l2end-no-intercept | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| svm | `_mojolearn_svm_host.so` | `_mojolearn_svm` | svc, iforest, svc-linear, iforest-tuned, svr, svr-linear, svc-poly | SVC, IsolationForest, SVR (svc, svc-linear, svc-poly, svr, svr-linear, iforest, iforest-tuned) | tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml) | yes |
| trees | `_mojolearn_trees_host.so` | `_mojolearn_trees` | et-clf, et-reg, et-clf-entropy-bestfirst, et-reg-bootstrap-parallel, par-forest-et, par-forest-et-clf | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| rf | `_mojolearn_rf_host.so` | `_mojolearn_rf` | rf-clf, rf-reg, rf-clf-entropy-log2-noboot, rf-clf-balanced-parallel, rf-reg-poisson, rf-reg-gamma-ig, par-forest, par-forest-reg, rf-score-weighted | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| gp | `_mojolearn_gp_host.so` | `_mojolearn_gp` | gp, gp-matern12, gp-matern32, gp-matern52-ard, gp-normalize-y, gpc, gpc-multiclass, gp-sample-y, gp-sample-y-normalize, gp-optimize, gp-optimize-restarts, par-gpc-fit, par-gpc-predict | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| kernel_methods | `_mojolearn_kernel_methods_host.so` | `_mojolearn_kernel_methods` | rbf-sampler, kernel-ridge, nystroem, par-rbf-sampler, kernel-ridge-poly, kernel-ridge-sigmoid, kernel-ridge-laplacian, nystroem-poly, nystroem-sigmoid, nystroem-laplacian | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| mixture | `_mojolearn_mixture_host.so` | `_mojolearn_mixture` | gmm, gmm-random-init, gmm-sample, gmm-random-init-sample | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| mixture_infer | `_mojolearn_mixture_infer_host.so` | loaded by path | no | GaussianMixture (gmm, gmm-random-init, gmm-sample, gmm-random-init-sample) | tools/classical_host_gate.py | yes |
| hdbscan | `_mojolearn_hdbscan_host.so` | `_mojolearn_hdbscan` | hdbscan, hdbscan-leaf | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| gp_infer | `_mojolearn_gp_infer_host.so` | loaded by path | no | GaussianProcessRegressor, GaussianProcessClassifier (gp, gp-matern12, gp-matern32, gp-matern52-ard, gp-normalize-y, gpc, gpc-multiclass) | tools/classical_host_gate.py | yes |
| hdbscan_infer | `_mojolearn_hdbscan_infer_host.so` | loaded by path | no | hdbscan.approximate_predict, hdbscan.membership_vector, hdbscan.all_points_membership_vectors (hdbscan, hdbscan-leaf) | tools/classical_host_gate.py | yes |
| gbdt | `_mojolearn_gbdt_host.so` | `_mojolearn_gbdt` | gbdt-symmetric, gbdt-rmse, gbdt-depthwise, gbdt-lossguide, cross-val, par-cross-val, gbdt-nan-modes, gbdt-adapter-clf, gbdt-adapter-reg, gbdt-parametric-losses, gbdt-exact-mae, gbdt-lossguide-newtoncosine, gbdt-multiclass, gbdt-onevsall, gbdt-multiclass-defaults, gbdt-ordered-rmse, gbdt-feature-freq, gbdt-pointwise-l2-bayesian-eval, gbdt-categorical-ctr, gbdt-adapter-score-weighted, gbdt-query-rmse, gbdt-pair-logit, gbdt-yeti-rank, gbdt-border-types, gbdt-ordered, gbdt-ordered-bayesian-noise, gbdt-bfa-quantile, gbdt-catboost-defaults, gbdt-symmetric-eval, gbdt-stochastic-arms | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| training | `_mojolearn_training_host.so` | `_mojolearn_training` | mlp, optim-sgd, optim-adam-clip, cross-entropy-arms, training-primitives, ordered-gradient-sum, par-mlp, samba, samba-untied-dropout-accum, par-samba, par-samba-clip, mlp-bf16w, mlp-int8w, samba-bf16w, samba-int8w, grad-accumulation | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| resample | `_mojolearn_resample_host.so` | `_mojolearn_resample` | bootstrap, permutation-test, monte-carlo | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| mamba | `_mojolearn_mamba_host.so` | `_mojolearn_mamba` | mamba2, mamba2-dtlimit, mamba1, mamba3, mamba1-bf16w, mamba1-int8w, mamba2-bf16w, mamba2-int8w, mamba3-bf16w, mamba3-int8w, mamba1-decode-session | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| arima | `_mojolearn_arima_host.so` | `_mojolearn_arima` | arima, arima-011, arima-seasonal-c, par-arima, arima-exog, arima-exog-seasonal, par-forecast-arima | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| embedding | `_mojolearn_embedding_host.so` | `_mojolearn_embedding` | embedding, embedding-sort | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| embedding_infer | `_mojolearn_embedding_infer_host.so` | `_mojolearn_embedding` when its reference binding is not built | no | Embedding (embedding) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| ivf | `_mojolearn_ivf_host.so` | `_mojolearn_ivf` | ivf, ivf-euclidean, ivf-extend, par-ivf | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| ivf_search | `_mojolearn_ivf_search_host.so` | `_mojolearn_ivf` when its reference binding is not built | no | IVFIndex (ivf, ivf-euclidean, ivf-extend) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| forecast | `_mojolearn_forecast_host.so` | `_mojolearn_arima`, `_mojolearn_tsa` when its reference binding is not built | no | ARIMA, ExponentialSmoothing, kpss_test (arima, arima-011, arima-seasonal-c, holtwinters, holtwinters-multiplicative, arima-exog, arima-exog-seasonal) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| transformer | `_mojolearn_transformer_host.so` | `_mojolearn_transformer` | transformer, transformer-window, transformer-bf16w, transformer-int8w, transformer-decode-session | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |<!--/fact-->

## The import question, measured

`transformer_backward_oracle.mojo` computes entirely on the host but imports
three names from `gemm/checks/gemm_backward.mojo`, which imports
`max.gpu.host`, and `IdentityTrace` from `core/identity_trace.mojo`, which does
too. `bindings/_mojolearn_byte_lm_host.mojo` says "HOST ONLY. No DeviceContext,
no kernel, no GPU, and nothing imported from the GPU side." Whether that rule
is enforced decided whether a CPU trainer needed a refactor of shared certified
files, so it was measured rather than argued from the comment.

`training/checks/byte_lm_host_bwd_probe.mojo` imports the oracle and the host
step and compiles with no accelerator target. **Exit 0 on all seven runners,
x86-64, ARM64 and Apple.** So the rule is policy the toolchain does not enforce
here, no shared file has to move, and no GPU code path is touched.

The same probe compiles `byte_lm_host_backward.mojo`, which nothing else
imports and which therefore no build would otherwise check. Its first pass
produced three errors, all the same rule, that reading a `List` out of a tuple
is an explicit copy or a transfer. Nothing structural failed, no import was
rejected and no oracle was missing.

The GPU backward is a useful starting point rather than a translation problem.
It contains no float atomic anywhere. Every place that needs a fold or a
scatter is written so one thread owns one output cell and accumulates in a
fixed ascending order, or work is partitioned so ownership is exclusive, and
there is an "eager" attention backward whose kernels are one thread per output
cell with serial chains. That is a sequential algorithm that happens to run on
a GPU.

One thing to establish rather than assume: the GPU's default attention
backward is the fused path, and a host version would follow the eager one, so the
two must be shown to agree bitwise.
