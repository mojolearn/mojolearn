# Changelog

All notable changes to mojolearn are recorded here, newest first, in the style of Keep a Changelog.

## Unreleased

### Added
- `mojolearn.linalg.matmul_int15`, with `quantize_int15` and `dequantize_int15`: a 15-bit integer matrix product under the contract `mojolearn.identical.gemm.int15i64.v1`. Each row of each operand is quantized from its float32 values to 15-bit integer codes with one power-of-two scale, split into two int8 planes, and multiplied on the integer matrix units with exact integer sums, so the result is the same bits on NVIDIA, AMD, Apple and the CPU. A contracted extent above `linalg.INT15_MAX_K` (65536) is refused by name. The same bits were recorded on an H100, an MI325X, an M3 Ultra and an M2 Pro over 178 cases, and every sabotage arm of its gates was seen failing. The technique is an application of known work (the Ozaki scheme on integer matrix units, and fixed-point arithmetic); nothing here is new.
- The tuned integer matrix unit kernels behind it, a parallel quantizer, their gates (`pixi run check-gemm-int15`, `check-gemm-lowbit`, `check-gemm-int8-mma-tuned`, `check-gemm-int8-pieces-tuned`, each with an arm that must fail), and the timing harnesses and their results under `bench/results/`.
- A selector for the number format of the matrix products: `mojolearn.numeric_profile()`, `mojolearn.set_numeric_profile()`, `mojolearn.numeric_profiles()`, `mojolearn.numeric_profile_measured()` and `numeric_profile=` on model loading and on `TransformerBlock`. Two profiles are registered: `fp32_v1` and `fixed15_v1`.

### Changed
- **FP32 remains the default; `fixed15_v1` is opt-in.** The unreleased default flip was reverted at the user's request. `mojolearn.models.CausalLM` and `ParallelCausalLM` use `fp32_v1` unless explicitly opted in through `numeric_profile="fixed15_v1"`, `set_numeric_profile("fixed15_v1")`, or `MOJOLEARN_NUMERIC_PROFILE=fixed15_v1`. Under that opt-in, transformer projections, the output head and Q.K^T use 15-bit integer products. P.V, norms, RoPE, softmax and residuals stay float32. The following results describe the optional profile, not default execution:
  - Different bits. The logits are not the ones earlier versions returned for the same model and input.
  - The same bits on every vendor. The new logits are identical on NVIDIA, AMD, Apple and the CPU, and decoding token by token gives the same logits as one prefill. SmolLM2-360M's full logits were recorded with the same hash on an RTX 4090, an H100, an MI325X, an M2 Pro and the CPU.
  - Quality. Held-out perplexity of SmolLM2-360M changed by +0.0006% on enwik8 and +0.0033% on pile_github against `fp32_v1`, both within +0.004%.
  - Time. A 512-token prefill of SmolLM2-360M takes 0.34 to 0.40 of `fp32_v1`'s time on an RTX 4090, 0.64 to 0.68 on an H100 and 0.85 to 0.89 on an MI325X. It is slower on Apple at large shapes: the 15-bit product at the model's 512-token rows takes 3.4 to 3.65 times the float32 product's time on an M3 Ultra and an M2 Pro. On Apple, the first model that uses the profile raises a `NumericProfileSpeedWarning` once.
  - Opt-in `generate` on NVIDIA runs one resident session under `fixed15_v1`, as it does under `fp32_v1`. The blocks' and the head's 15-bit weights stay on the device, products of up to 16 rows take the decode kernel, and the attention scores of every head are computed in one launch. Its tokens and logits are bit-identical to the per-layer route. From a 512-token SmolLM2-360M prompt, `generate` took this much of `fp32_v1`'s resident session's time: 0.92 (32 new tokens) and 0.95 (128) on an H100, and 0.91 and 0.88 on an RTX 4090. Per decode token it took 25.2 against 26.1 ms on the H100 and 19.1 against 22.3 ms on the 4090 (7 alternated rounds). Both profiles prescribe the same arithmetic across vendors; the default is FP32 everywhere. The opt-in has costs on AMD and Apple:
  - On AMD and Apple, the resident session under `fixed15_v1` has not been gated, so `generate` there runs the per-layer route, where under `fp32_v1` it ran the resident session. Against `fp32_v1`'s resident session, that `generate` has not been measured on AMD. On an MI325X the whole model on the per-layer route took 0.87 of `fp32_v1`'s per-layer time at prefill and 0.65 per decode token.
  - On Apple the opt-in profile is slower. SmolLM2-360M's whole model on an M2 Pro took 4.5 times `fp32_v1`'s time at prefill and 2.6 times per decode token (per-layer route, both profiles).
- **What does not change.** Mamba models, a `TransformerBlock` built directly, and every trainer (the optimizers, `SambaStack`, the byte language model trainers) still compute `fp32_v1` when you name nothing, and a model's `numeric_profile` attribute tells you which profile it runs. Training under `fixed15_v1` is refused by name. So is `fixed15_v1` on a Mamba model. A saved state with no profile field is read as `fp32_v1`. The `hf-causal-lm` and `par-causal-lm` verifier lanes load `fp32_v1` by name, because their reference digests were recorded under it.
- **Default arithmetic stays FP32 without configuration.** To override an explicit process/environment opt-in, pass `numeric_profile="fp32_v1"` to `CausalLM.load`. Existing states retain their recorded profile when restored. No fixed15 kernel arithmetic was removed or changed by restoring the default; the wheel still includes the new opt-in code and other merged improvements, so it is not the same artifact as the previous release.

### Measured
- The time of the complete 15-bit inference call over our own fp32.v1 call at the same 512-token rows, on the same box, in the same run: H100 0.43 to 0.50 (run 5); MI325X 0.19 to 0.30, with a stand-in recombination, one run; M3 Ultra 3.4 to 3.6, on the float unit, untuned.

## 0.8.28 (unreleased)

- Fix binary `LabelBinarizer.inverse_transform` for two-column input: read within the output allocation before selecting its last column. This fixes the `x-prep-inverse-transforms/wide` verifier assertion. The correction was already committed in source but absent from the prior verifier-only wheels.
- Reuse published 0.8.27 native binaries, references and verifier code; apply only the exact Python bounds fix and version metadata, recorded in `PYTHON_PATCH.json` and checked again before publication.

## 0.8.27 (published 2026-09-29)

### Fixed
- Verification reports now fingerprint every fixture used by bundled saved-model checks, including non-base fixtures in quick/default runs. Previously these missing fingerprints could make valid reports incomparable.
- `verify --compare` compares the bundled GPU and CPU cross-check values as well as the main cell table. Missing rows, failed workers and same-answer local divergences cannot become a successful comparison. Cross-check values and execution scope are now covered by commitments.
- Comparison tolerates different scopes: shared compatible checks are compared, missing checks are listed per side, and incompatible inputs are flagged per cell without discarding unrelated matches. Actual mismatches outrank incomplete coverage.
- `verify --compare A B --json-out comparison.json` saves its result using the same output option as verification. Scope still comes from the input reports; comparison does not rerun algorithms.
- Verifier-only update of 0.8.26: algorithm code, native binaries, fixtures and references are unchanged. Existing numerical runs remain valid for that unchanged payload.

## 0.8.26 (published 2026-09-29)

### Changed
- Verifier-only wheel update of 0.8.25: algorithm Python, native binaries, fixtures and reference hashes are reused byte-for-byte. Unreleased algorithm changes above are not included.
- Bare `verify` combines inference and classical training on the base fixture; `--quick` samples the same scope, and `--all` / `--full` use all nine fixtures. Neural training is separately selected with `--neural-training`.
- Routine GPU verification includes direct CPU/GPU inference comparisons. Full cross-checks have no lane cap, run sequential batches, include forest/GBDT routes, and fail on missing comparisons. Spectral batch comparisons now receive the same held-out affinity input on both devices.
- Verifier-only publication checks every wheel member against its published PyPI base, allowing only committed verifier files and version metadata changes. Existing numerical results are retained without repeating algorithm certification.

## 0.8.25 (published 2026-09-29)

### Added
- Optional `mojolearn[verify]` and `mojolearn[numpy]` extras. The base package does not require NumPy; optional array-based APIs provide installation guidance when it is absent.
- The full verifier includes applicable batch-invariance checks and reports missing, refused and inapplicable cases separately.
- `UMAP` option parity with umap-learn: `n_components` 1 to 32, any `local_connectivity`, `metric` sqeuclidean / cosine / manhattan / chebyshev / minkowski (`metric_kwds={'p': p}`), `init` 'random' / 'pca' / an array, `a` and `b` given directly, and supervised `fit(X, y)` with `target_metric` 'categorical' or 'l2', `target_weight` and `target_n_neighbors`. `densmap=True` and a non-euclidean `output_metric` are refused by name. Default fits keep their bits.
- `numpy`-style `mojolearn.linalg.qr` modes 'reduced', 'complete', 'r' and 'raw', and `mojolearn.linalg.svd` returning (U, S, Vh); `AlternatingLeastSquares(use_cg=True)`.

### Changed
- Consolidated Apple performance work across ANN, clustering, decomposition, linear models, metrics, neighbors, preprocessing and trees. Experimental paths remain opt-in where qualification is incomplete.
- `mojolearn.linalg.qr(a)` preserves its published `mode='r'` default and R-only result. Request `mode='reduced'` explicitly for the new `(Q, R)` result. The existing TSQR R preserves its bits; its row signs can differ from the R produced alongside Q.
- `TruncatedSVD.explained_variance_` and `explained_variance_ratio_` are computed in TruncatedSVD's own binding (`tsvd_explained`) instead of through the decomposition expansion binding; the default fit no longer loads `_mojolearn_x_decomp`. The values are scikit-learn's definition as before, with a different (pinned) summation order, so their last bits may differ from the previous unreleased build.
- `mojolearn.linalg.eigh(a, UPLO="L")` reads ONE triangle of `a`, as `numpy.linalg.eigh` does: the lower (the default) or, with `UPLO="U"`, the upper triangle mirrored across the diagonal; the other triangle is never read. Through 0.8.24 it fed the whole matrix to the Jacobi, so a NON-symmetric input returned a different answer from numpy's. Symmetric inputs return the same bits as before.
- `SpectralEmbedding(eigen_tol=<float>)` sets the Lanczos tolerance (a positive float; `'auto'` is cuVS's 1e-5). `PCA(svd_solver="arpack")`, `TruncatedSVD(algorithm="arpack")`, a nonzero `tol` on either, `SpectralEmbedding(eigen_solver=...)` other than None, and `Isomap` / `LocallyLinearEmbedding(eigen_solver="arpack")` are refused by name: those solvers are not implemented, and running an exact solver under their names would be a silent substitution.

### Fixed
- Cross-platform GP binding imports and Metal barrier declarations compile consistently.
- Verification jobs propagate failures, exclude timing fields from bitwise comparisons and bound optional timing work.
- Tree-wrapper packaging records its device-free dispatch role and checks its RF/GBDT device dependencies.

## 0.8.24 (published 2026-09-27)

### Fixed
- Linux ships again. 0.8.23 published the macOS wheel only: its AMD (gfx942) build failed because the Apple left-looking Cholesky kernel was launched behind a runtime guard and so compiled into every GPU target, where its Apple matrix-unit instructions do not exist (mixture, gp, kernel_methods). The launch is now compile-time gated (`cholesky/checks/potrf.mojo`); no bit moves on any column. Every binding changed since 0.8.22 cross-compiles for gfx942 (32 of 32). Linux users get 0.8.23's GaussianMixture and FAST-quality fixes with this release.

### Changed
- Apple GPU, IDENTICAL, same bits: the matrix-unit admission bound is exact, with a per-cell refusal in the Cholesky update; the Cholesky back solve stages operands ahead of the chain; ARIMA's forward-difference gradient runs in one stacked evaluation.

## 0.8.23 (published 2026-09-26)

### Fixed
- `GaussianMixture(init_params="kmeans")` seeds its k-means with the classic greedy k-means++ (scikit-learn's `KMeans` default) instead of cuVS's k-means|| (DEVIATION 3133). On few-valued data the k-means|| start left one cluster with most rows beside clusters of 1 to 7 rows, and the fit scored far below scikit-learn: on 20 taxi windows of 2,000 rows (10 offsets x 2 seeds) the median held-out mean log-likelihood gap to scikit-learn 1.7.2 went from -4240 to -356 (IDENTICAL) and from -5320 to -1.9 (FAST), and scikit-learn scored higher in 10 of 17 comparable fits instead of 17 of 18. IDENTICAL bits move for every `init_params="kmeans"` fit, on every column (`gmm`, `gmm-sample`, `par-gmm`); `init_params="random"` is unchanged.
- `MOJOLEARN_VENDOR=cpu` selects the CPU path on macOS. The macOS wheel returned before reading it, so through 0.8.22 a Mac asking for the CPU silently got the Metal set; it now loads the CPU-only bindings (`python/mojolearn/_backend.py`), whose results equal the Metal IDENTICAL fits bit for bit (KMeans, LinearRegression, RandomForest checked).

### Changed
- FAST keeps the reference quality: UMAP's spectral initialization solves to the reference tolerance again, and Lossguide grows best-first one leaf at a time again (the 16-leaf batch cost quality). FAST ARIMA uses the serial Kalman filter by default.
- Apple GPU, IDENTICAL, same bits: the transposed GEMM on the simdgroup matrix unit where the window admission holds, skinny GEMM shapes on split-K, Lanczos restarts on the device, the spectral kNN graph built on the device, the SVC/SVR block solve, ARIMA's Kalman loop at the state dimension, a left-looking Cholesky, IVF-Flat search in one launch, and DBSCAN and k-means++ schedules.

## 0.8.22 (published 2026-09-26)

### Changed
- The first release of the split Linux packages described under 0.8.21 (`mojolearn`, `mojolearn-nvidia`, `mojolearn-amd`; `pip install mojolearn` installs all three). 0.8.21 published the macOS wheel only: its Linux set was refused before upload because the two plugins carried two platform tags (`manylinux_2_34` and `manylinux_2_35`) and the core one. All three now carry the core's `manylinux_2_35_x86_64` alone (`packaging/linux/audit.sh`). No native library changed from 0.8.21.

## 0.8.21 (published 2026-09-26)

### Changed
- Linux ships as three PyPI packages: `mojolearn` (Python, the CPU bindings and the MAX runtime), `mojolearn-nvidia` (the CUDA sets) and `mojolearn-amd` (the HIP set). `pip install mojolearn` installs all three at the same version, so the install command does not change. The GPU code is byte for byte the combined wheel's, installed at the same paths (`python/mojolearn/gpu_plugins.py`). macOS stays one wheel.
- Apple GPU, IDENTICAL: the attention forward's scores and context and the backward's `dy` chain run on the simdgroup matrix unit. On the M4 the matrix multiply-accumulate equals the FMA chain on every cell probed (4.19M cells over 9 operand kinds, including products and partial sums straddling 2^-126), so the matrix chain is the attention chain; the causal diagonal keeps the scalar chain. Byte LM forward attention 127 to 85 ms per kernel at 1 x 2048, d768; per-step witnesses equal the shipped build's (`-D MOJOLEARN_ATTN_NO_APPLE_MMA` reverts).

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
