# Changelog

All notable changes to mojolearn are recorded here, newest first, in the style of Keep a Changelog.

## 0.8.20 (published 2026-09-26)

### Changed
- IDENTICAL arithmetic no longer depends on whether the compiler contracts a multiply into the add that follows it. Every such pair is written either as an explicit fused multiply-add or as a pinned product that no backend can fuse (`pinned_mul_f32` / `pinned_mul_f64` in `checks/numerics.mojo`: a fence on CPUs, `mul.rn` on NVIDIA, `v_mul` on AMD, an unflagged `fma(a, b, -0)` on Apple), and `tools/contraction_census.py` names any source line whose fused operations change between the default build and `--fp-mode contract=off`. The bits are 0.8.19's: 209 lanes, three fixtures each, equal to the 0.8.19 Metal, CUDA and HIP columns on Apple and CPU (default and `contract=off`, 627 of 627), on an NVIDIA L40S (621 of 621) and on an AMD MI300X (621 of 621) (`docs/lanes/PINNED_MUL_RESUME.md`).
- Apple GPU, IDENTICAL: the transposed GEMM is threadgroup-tiled, the tuned GEMM runs admitted windows on the simdgroup matrix unit (whose fp32 8x8x8 multiply-accumulate was shown to be the same ascending FMA chain), the Gram split-K and the quasi-Newton steps read their operands coalesced, and the byte language model's embedding and head weights are views of the flat parameters (less memory, same step). Same bits: the byte LM step witnesses equal on the M4, an H100 and an MI300X.
- Apple GPU, FAST (trees and classical, which do not promise identical bits): RandomForest and ExtraTrees histogram and split passes, SVC and SVR working sets, ARIMA (batched L-BFGS, the Kalman filter in parallel over time), UMAP's spectral initialization, IVF-Flat search and training, k-means++ seeding, DBSCAN, HDBSCAN and AgglomerativeClustering (Boruvka minimum spanning tree without the dense graph), Cholesky, Holt-Winters and the quasi-Newton gradient.
- Linux IDENTICAL CUDA sets (sm_89, sm_90a) ship machine code: every IDENTICAL kernel is compiled at build time by a pinned ptxas (CUDA 12.5.82) with `--fmad=false` and stored as an LZ4-compressed fatbin in place of its PTX (`packaging/linux/cubin_contract.py`), so the user's driver no longer JIT-compiles them. The toolkit is 12.5 on purpose: CUDA 13 binaries would need a 580 driver even under MAX's `MODULAR_NVPTX_COMPILER_PATH` escape, 12.5 ones do not, so the driver floor is unchanged. On 0.8.19 the whole release column ran with the driver JIT disabled on an RTX 4090 and an H100 (driver 580) and an RTX 2000 Ada (driver 570), with the same bits as the recorded Apple, AMD and NVIDIA columns (`bench/results/nvidia_fatbin_2026-09-26/`). The wheel audit refuses IDENTICAL CUDA PTX outside a small JIT-invariant exception.

## 0.8.19 (published 2026-09-25)

### Changed
- NVIDIA (sm_90a) training takes 30.7 s per optimizer step at the GPT-3 Small shape on one H100, down from 38.8 s, with the same bits (`bench/results/nvidia_step_time_2026-09-25/`). Three NVIDIA-only changes: GEMM window admission (a 16-deep window whose staged operands provably cannot produce a subnormal step result runs the bare `fma.rn`, which is then the contract step exactly; every other window keeps the two-instruction step; `-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1` reverts); a 128x64 kpack GEMM tile under a 512-thread launch bound (a schedule: every cell's chain, leaves and fold are unchanged; `-D MOJOLEARN_GEMM_NO_KPACK_NARROW=1` reverts); and launch bounds on the attention forward (1024) and dq (768) kernels, register allocation only (`-D MOJOLEARN_ATTN_NO_LAUNCH_BOUND=1` reverts). Every other column declares 1024 on those two attention kernels.
- AMD (gfx942) training takes 29.1 s per optimizer step at the same shape on one MI300X, down from 32.3 s in 0.8.18, with the same bits (`bench/results/amd_step_time_2026-09-24/`, "Pass 2"). The matrix-core GEMM applies the same exact-admission argument per window: where the block's staged operands prove every accumulator stays a multiple of 2^-126, the MFMA step alone is the contract step and the product by one is not issued (AMD only; `-D MOJOLEARN_GEMM_MFMA_NO_ADMIT=1` reverts). The AMD attention kernels on the matrix cores are not in this release; they stay off by default until their 16x16x1 step is explained.
- Every column: the embedding backward's run-start prefix sum runs in one block of 256 threads instead of one thread. Integer addition is exact, so every word equals the serial kernel's (`-D MOJOLEARN_EMB_SERIAL_RUN_BEGIN=1` reverts).
- Proofs of the merged source, built from source on each box (`bench/results/release_0819_proofs_2026-09-25/`): on a Hot Aisle MI300X and on a RunPod H100, the replays of steps 101 to 103 and 1999 to 2000 from the run's checkpoints equal the H100 chain line by line (state digests abc8b816b5c3fb15, a9421f91b947f82c, fcdb48b8ab51f2ef, dcb05e4e668a81e1, 0e39ed2bfe9bcbae; gradients, the 64 losses and the learning-rate bits); the T3-shape GEMM hashes equal the VALU kernels on AMD (64 of 64 lines over six operand kinds) and 0.8.18's NVIDIA GEMM on NVIDIA (72 of 72), the NVIDIA admission sabotage differs on 12 lines, and the 64 AMD lines equal the NVIDIA ones hash for hash; the GEMM device, backward and workspace checks are green on both; 181 of the 201 GEMM-reaching identity lanes read IDENTICAL on each vendor (6,813 cell parts) and none divergent, with the same 20 refused on both for bindings the boxes did not build. Apple: the changed kernels compile to AIR for apple-m4 on a Linux host; no Apple GPU ran for this release.

## 0.8.18 (published 2026-09-25)

### Changed
- AMD (gfx942) training is about 4.4 times faster per optimizer step at the GPT-3 Small shape, with the same bits. On a Hot Aisle MI300X one optimizer step of 64 shards (batch 4, length 2048, 162,147,840 parameters) went from 141.0 s to 32.3 s (`bench/results/amd_step_time_2026-09-24/`): the two IDENTICAL GEMM kernels declare their real 256-thread launch size so the gfx942 backend no longer spills the 128x128 register tile; the tuned GEMM calls run on the matrix cores (`v_mfma_f32_32x32x1f32`, one product per step, the flush spelled as a product by one) with a grouped launch; the AMD leaf-split dispatch; `ftz` spelled as one class compare in AMD device code; and the NVIDIA attention block map as AMD's default. Every replayed optimizer step from the run's checkpoints (steps 101 to 103 and 1999 to 2000) equals the H100 chain line by line on state, gradient, the 64 losses and the learning-rate bits; 181 of the 201 GEMM-reaching identity lanes read IDENTICAL on AMD against the shipped references and none divergent (20 refused for bindings the box did not build). The GEMM launch bound is the one change that reaches NVIDIA binaries (`.maxntid 256` on sm_89 and sm_90a); Apple carries no bound. The NVIDIA and AMD release columns and a two-step replay against the live chain on each vendor gate this release.

## 0.8.17 (published 2026-09-24)

### Fixed
- The live cross-vendor worker runs on Python 3.10 and 3.11 from the wheel. `ParallelByteLanguageModelTrainer.fold_export`, `cross_vendor.state_hash`, the worker's parameter count and its shard gradient views called `memoryview` on the package's own `Array`, which cannot export the buffer protocol below Python 3.12 (`TypeError: memoryview: a bytes-like object is required, not 'Array'`, seen on a Python 3.11 H100 box on 2026-09-23). They read through `_buffer.flat_bytes`, which works on every supported Python. No arithmetic changed; the native libraries are the 0.8.16 bytes.

## 0.8.16 (published 2026-09-23)

### Fixed
- Float16 safetensors checkpoints load on Python 3.10 and 3.11. `memoryview.cast("e")` exists only from Python 3.12; the F16 bytes are now copied as they are and widened from their bits on every version, the same bits as before (subnormal, negative zero and scalar tensors checked). Every F16 checkpoint read on 3.10 and 3.11 in 0.8.15 raised `memoryview: destination format must be a native single character format`.
- `tools/lm_segment.py`: an expected chain line written before chain lines named their hash scheme is compared as the first scheme (`sha256.v1`), and the recipe names the scheme a run hashes under, so a segment replayed on another vendor is held to the recorded digests rather than to the label.
- Loading a native binding no longer leaves `PYTHONEXECUTABLE`, `PYTHONPATH` and `MOJO_PYTHON_LIBRARY` set in the process environment. The bundled runtime sets them in the C environ when a binding loads, so a child interpreter started with `sys.executable` could lose its virtual environment (seen on Python 3.11: no NumPy in the child). Every binding load now restores the three variables as it found them, through the C environ.
- The Python test suite runs on Python 3.10 and 3.11 (`bench/results/python_versions_2026-09-23`): `bytes(Array)`, a `math.fma` use and two test fixtures that relied on Python 3.12 or 3.13 behavior are fixed, and the tools tests skip rather than abort without optional packages.

### Changed
- Native libraries rebuilt from source.

## 0.8.15 (published 2026-09-22)

### Fixed
- AMD gfx942 builds are byte-reproducible: the GEMM and Holt-Winters kernels compute their index arithmetic with unsigned division and loop counters, so repeated builds of the same source produce the same AMD binaries (23 of 23 identical bindings across six clean builds). No floating-point operation or order changed; NVIDIA binaries are unchanged.

### Changed
- Native libraries rebuilt from source.

### Added
- `tools/lm_segment.py`: one segment of a checkpoint-to-checkpoint language-model run on any box. A recipe (shape, K shards, optimizer, a learning-rate table of float32 bits, the token stream's identity) never changes between segments; steps are numbered globally; every step writes a hash-chained line (state, summed gradient, losses, learning-rate bits); checkpoints stream in the `mojolearn.byte-lm-stream.v1` format on a cadence, at a boundary minus two and at the end, pinned in a manifest and PUT to presigned URLs as written; `--expect-chain` holds a run to another run's chain step for step and stops on the first difference (route B against route A, an arrival replay against the sender); `--zero-moments` is the negative control; `compare` and `manifests` hold chains and checkpoints across runs. Verified on the M4 at a small shape: three segments, a second route from the first route's checkpoints, an arrival replay, the control failing and a wrong recipe refused.
- `ParallelByteLanguageModelTrainer.set_lr` (native `byte_lm_parallel_set_lr`): the learning rate every replica uses at its next update, for a per-step schedule; held bit for bit equal to a fresh open at that rate. `export_raw`: the four state arrays without the per-element admission pass, for hashing and checkpoints at scale.
- `mojolearn.cross_vendor` chained protocol (`--chained`): contiguous shard blocks, one gradient per worker on the wire instead of one per shard, the same bits as the gathered fold and the one-process column; `ordered_fold(..., prefix=)` and `fold_pair` with a vectorized NumPy spelling held equal to the pure-Python one; `Worker(lr_for_step=)`.
- `tools/fineweb_tokens.py`: FineWeb-Edu parquet shards (or `fineweb_text.py` text) to one pinned `mojolearn.byte-lm.tokens.v1` stream through the pinned vocabulary, one document per row, streamed, with a held-out tail range.
- The GPT-3 Small run's token stream is in the R2 dataset store: FineWeb-Edu shards 000 to 003 and 013 through the pinned vocabulary, 3.11B ids in seven pinned parts, produced byte-identically on three CPUs (`bench/results/fineweb_tokens_2026-09-22`).
- `ParallelByteLanguageModelTrainer.fold_reset`, `shard_gradient_fold`, `fold_add` and `fold_export` (native `byte_lm_parallel_fold_*`): a live worker's block of the ordered fold on its own device with the kernel `train_step` uses; the chained `cross_vendor` worker uses them when the binding has them, so no gradient is downloaded per shard and the fold costs milliseconds instead of the 184 s a host fold cost at 162M parameters. The coordinator's per-step record carries phase timings (gradients in, each worker's fold round trip, apply, the step).
- `tools/lm_segment.py --live-role coordinator|worker`: the multi-vendor segment inside the segment runner, writing the same chain lines and checkpoints as a one-box segment; `tools/lm_segment_body.sh`, `tools/lm_segment_leg.py` (renders a segment's leg body and mints its presigned URLs), `tools/lm_live_leg.sh` and `tools/lm_live_link.sh` (two rented boxes joined through an ssh tunnel), `tools/lm_run_driver.py` (every segment of every route in dependency order, with a ledger, the halt rule, a GPU-type walk and an AMD provider walk that waits for capacity). Chain lines name their hash scheme (`sliced-sha256-8.v2`: eight slices hashed in threads, about five times faster than one sha256 at this size) and compare only under one scheme. `tools/do_extra_leg.sh --size --region`.
- `tools/gemm_remote_leg.sh` and `tools/do_extra_leg.sh` take `--segment-lease N --dollar-cap USD`: a lease above one hour, named, and refused unless its worst case at the box's own hourly price is under the cap (RunPod: `costPerHr` after the create, the pod terminated on refusal; DigitalOcean: `price_hourly` from the sizes API before the create). `--minutes` above 60 stays refused.

## 0.8.14 (published 2026-09-22)

### Changed
- `ExponentialSmoothing` takes `initialization_method`, default `"estimated"`: the initial level, trend and seasonal states are fitted jointly with the smoothing parameters. `"heuristic"` (alias `"cuml"`) is the 0.8.13 fit, and saved models without the field load as `"heuristic"`.
- `GradientBoosting` trains small IDENTICAL pools, and pools of one-border columns, on the CPU host route with the same bits.
- Native libraries rebuilt from source, including ordered resident forest inference on every GPU vendor, per-vendor exact random forest training kernels and a parallel Holt-Winters fit kernel.

### Fixed
- Random forest `fit` no longer hangs on NaN in `X`; the tree family refuses NaN and infinite inputs with a `ValueError` naming them.
- `KMeans`, `DBSCAN` and `KernelDensity` refuse NaN and infinite inputs at `fit`, and `Embedding` refuses non-integer ids.
- Classical estimators have `get_params` and work with `clone` and `cross_val_score`; `mojolearn.Array` converts through `__array__`; `import mojolearn.metrics` works; `SVC` accepts string labels.
- `GradientBoostingRegressor` targets and the scalers accept nested Python lists.
- `LanguageModelHostTrainer` defaults `weight_decay` to 0.01, like the GPU trainer.
- The GreedyLogSum border penalty uses the portable logarithm, so borders near a tie agree across platforms.
- `python -m mojolearn verify` with no card runs the quick check instead of ending with no reference.

## 0.8.13 (published 2026-09-21)

### Fixed
- Quantile regression, extra trees and random forest `predict` and `predict_proba` no longer read Python objects after releasing the interpreter lock, which could crash the process.

## 0.8.12 (published 2026-09-21)

### Added
- Experimental live cross-vendor training (`mojolearn.cross_vendor`). GPUs from different vendors, in different machines, train one language model together, and every replica holds the same bits after every step, checked by a hash of each worker's full training state.
- `ParallelByteLanguageModelTrainer.shard_gradient` and `apply_gradient`.

### Fixed
- Apple language-model training with two or more blocks no longer aborts at small batch times length.

### Changed
- Native libraries rebuilt from source, including exact GPT backward fusions on NVIDIA and AMD and grouped symmetric-tree inference launches.
- Updated research preprint.

## 0.8.11 (published 2026-09-21)

### Added
- The research preprint ships in both platform wheels and is linked from the README.

Documentation-only release. Numerical code, reference data and native libraries are unchanged from 0.8.10.

## 0.8.10 (published 2026-09-20)

### Added
- Cross-validation accepts explicit Metal `devices=(0,)`.
- Parallel causal language models can assign every layer to Metal device 0.

### Changed
- `verify --par all` runs all fixtures by default, and every lane and fixture must carry its own placement witness.
- Parallel estimators with serialization must produce actual model bytes, and incomplete records are rejected.

### Fixed
- Parallel workers set native device counts to their own visible device group.
- Parallel ARIMA keeps the caller's trend configuration and saved-model metadata.
- The causal-language-model batch verifier closes its workers on success and failure.

## 0.8.9 (published 2026-09-20)

### Added
- CPU-only installs can train. Gradient boosting supports regression, classification, multiclass and ranking fits on the CPU, including bootstrap sampling and score noise.
- `LinearSVC`, `LinearSVR`, `QNRegressor` and the `mojolearn.svm` namespace.
- `SpectralEmbedding` and `manifold.spectral_embedding`.
- CatBoost-style boosting defaults, ordered boosting, seven border selection methods and quantile starting constants.
- Random Forest and Extra Trees select resident parallel-groves inference automatically in FAST mode.

### Fixed
- CPU HDBSCAN no longer produces NaN distances from a temporary-buffer lifetime error.
- GPU held-out loss curves, early stopping and model shrinking agree with the CPU when average boosting is disabled.

## 0.8.8 (published 2026-09-19)

### Fixed
- CPU replay of GPU-written CTR models no longer erases the recorded GPU model-byte reference.

### Changed
- Python and reference patches can reuse a published wheel's native libraries when their compile inputs are unchanged. Native binaries are the same as 0.8.7.

## 0.8.7 (published 2026-09-18)

Version 0.8.6 was prepared but never published. Its changes ship here.

### Added
- `python -m mojolearn verify --all`, `--self-test` and `--json`. Users can check an installed wheel against shipped reference hashes, watch the verifier fail on a deliberately perturbed input, and export per-cell evidence. See docs/VERIFY.md.
- `python -m mojolearn identity`, which diffs a local run against the Apple, NVIDIA and AMD columns shipped in the wheel.
- Every CPU host binding ships in both wheels, so a CPU-only install can verify many more lanes.
- Public CPU inference from GPU-saved models for most estimators, including k-means, DBSCAN, agglomerative and spectral clustering, neighbors, kernel density, Gaussian processes, Gaussian mixtures, HDBSCAN, isolation forest, ARIMA, Holt-Winters, UMAP, PCA, scalers, linear and kernel models, SVMs, IVF indexes, embeddings, Cholesky, gradient boosting with CTR tables, and the MLP, Transformer, Mamba and Samba blocks. `mojolearn.host_model(path)` loads any saved model on a CPU.
- Incremental CPU decoding with `allocate_state`, a carried state and `step`, bitwise equal to a full forward pass.
- Low-bit inference weights. Every inference class accepts bf16 or int8 packed weights through `mojolearn.lowbit.pack` and computes the same bits as the fp32 block on the materialized weights. int8 GEMM uses NVIDIA and AMD integer matrix units.
- `TransformerBlock` options for common decoder families (RoPE variants, biases, norm types, MLP types, QK norm, attention softcap).
- Experimental checkpoint loader `mojolearn.models.CausalLM.load` for Hugging Face `config.json` and `.safetensors` checkpoints, plus `models.Tokenizer.from_pretrained` for byte-level BPE families.
- `GaussianProcessClassifier`, GP kernel hyperparameter optimization, `GaussianProcessRegressor(normalize_y=True)` and `sample_y`.
- `GaussianMixture.sample`, `KMeans.transform`, `KMeans` save and load, `IVFIndex.extend`, `SpectralClustering.predict`.
- HDBSCAN `approximate_predict`, `membership_vector` and `all_points_membership_vectors`.
- ARIMA exogenous regressors.
- Ranking losses `QueryRMSE`, `PairLogit` and `YetiRank` for `GradientBoosting`, with `group_id` and explicit pairs.
- `SVC(kernel='poly')`, weighted `score` on tree estimators, weighted `accuracy_score` and `r2_score`, and `metrics.fowlkes_mallows_score`.
- `BpeTokenizer.encode_batch`, `decode_batch` and `decode_bytes_batch`.
- The bootstrap, permutation test, Monte Carlo integration and `kpss_test` work on a CPU-only install.

### Changed
- `GPT2Tokenizer` is renamed `BpeTokenizer`. The old name remains as a deprecated alias.
- `verify --all` reports INCOMPLETE and exits 4 when any part was refused. Only a run with nothing refused reads VERIFIED.
- `UMAP.transform` no longer depends on the query batch. A batch of N rows returns the same bytes as N single-row calls. Recorded UMAP transform outputs change.
- `GradientBoosting(l2_leaf_reg=None)` is the new default and takes each loss's own default.
- The README states Apple silicon support up front, and the Apple tree speed ratio was withdrawn.

### Removed
- The GPT-2 vocabulary and reference data. Tokenizers load a vocabulary the user supplies.

## 0.8.5 (published 2026-09-14)

### Fixed
- `ExperimentalTwoLevelFeatureFreq` returns the same predictions on every GPU vendor. Its histogram accumulator was undersized.
- Linux and macOS wheel packaging regressions for the CPU training binding.

### Changed
- AMD GEMM uses a new staging schedule with unchanged bits.

## 0.8.4 (published 2026-09-13)

### Added
- CPU training for the byte-level language model (`LanguageModelHostTrainer`) in both wheels, bitwise identical to the GPU.
- The language model's initialization is a pinned function of the parameter index, so seed, corpus and config determine the trained bits.

### Changed
- Shorter neural training steps on NVIDIA and AMD than in 0.8.3, with no bit changed (GEMM staging, attention exp stash, leaner step glue).
- Gradient boosting under IDENTICAL partitions leaves on the device on every vendor.

### Fixed
- A GPU-resident array passed to an estimator is refused with a clear error naming its type and device.

## 0.8.3 (published 2026-09-11)

### Fixed
- `SVC` and `SVR` fits with more than 512 training rows no longer fail on NVIDIA GPUs.
- IDENTICAL `LinearRegression` handles badly scaled designs. The eigensolver no longer overflows, and the rank cutoff is relative.

## 0.8.2 (published 2026-09-11)

### Fixed
- IDENTICAL gradient boosting on AMD GPUs. Histogram kernels could skip a block-wide sync, so fits on tied data could differ between runs and from other vendors.

## 0.8.1 (published 2026-09-11)

### Fixed
- Buffer conversions no longer fail in a process that also imports cuML.

## 0.8.0 (published 2026-09-10)

### Changed
- Only the tree families (GBDT, Random Forest, Extra Trees) ship the `fast` and `deterministic` tiers. Every other estimator is `identical` only and refuses other tiers by name.
- Estimators return `mojolearn.Array`, and NumPy is no longer a runtime dependency. `numpy.asarray(result)` gives a zero-copy view.

### Added
- Optional GPU `parallel_groves` prediction for Random Forest and Extra Trees.
- Random Forest `class_weight`, per-tree GBDT feature sampling and minimum child Hessian for Newton growth.
- GBDT classifier and regressor adapters and scikit-learn-style parameter and scoring protocols.
- GPU metrics (regression errors, classification scores, log loss, ROC AUC, precision-recall curves).
- GPU `StandardScaler`, `MinMaxScaler` and serial `cross_val_score`.
- `LanguageModelConfig` and `LanguageModelTrainer` with configurable layers and vocabularies and resident training sessions.
- IVF selection through k=1024.

## 0.7.0 (published 2026-09-09)

Version 0.6.1 was prepared but never published. Its changes ship here.

### Added
- One Linux x86-64 wheel carrying CUDA sm_89, CUDA sm_90 and HIP gfx942.

### Changed
- The identical path no longer calls the host C library, so every host computes the same bits.

### Fixed
- Mamba-3 DETERMINISTIC mode on CUDA uses the stable small-dt softplus.

## 0.6.0 (published 2026-09-06)

### Added
- `UMAP.transform` for unseen samples against a fitted model.

### Changed
- UMAP stores its fuzzy graph in CSR form.
- Binding compilation uses two workers by default, configurable with `MOJOLEARN_COMPILE_JOBS`.
- The identical path's last host libm calls were replaced with portable implementations.

## 0.5.0 (published 2026-09-05)

### Added
- macOS arm64 wheel with 15 native extensions in all three modes.
- `UMAP.fit` and `fit_transform` for dense Euclidean input with 2D and 3D embeddings.
- Wheel admission checks for contents, digests, platform tags and native extensions.

### Changed
- Documentation consolidated around one roadmap, support matrix (SUPPORT_MATRIX.md), verification guide and numerical contracts.

## 0.4.0 (unreleased 2026-09-02)

Prepared but never published to PyPI. Its contents shipped in 0.5.0.

### Added
- Mamba 1, 2 and 3 and Transformer forward and backward with Python bindings.
- Wider cross-vendor identity coverage across classical ML, linear algebra, trees, sequence models and training.
- The 15-extension packaging surface and stricter release checks.

## Earlier releases

Versions 0.1.0 through 0.3.1 established the Mojo GPU implementation, Python packaging and initial Apple, AMD and NVIDIA support. 0.3.0 is yanked on PyPI because its Linux wheel required AVX-512; use 0.3.1 or later.
