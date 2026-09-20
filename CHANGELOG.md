# Changelog

This file records release-level changes, not the development diary. Git history and archived evidence
contain the detailed investigation record.

## Unreleased (lane/close-no-cpu-path-gbdt)

**An eval set, the overfitting detector and `use_best_model` now have a CPU
verification route** on the Plain SymmetricTree Logloss fit, through the new
host oracle `gbdt/host/gbdt_oracle_eval.mojo`. Until this lane, `gbdt_fit`
refused `eval_set` BY NAME on every Plain arm, so no early-stopping fit could
be checked against a GPU column without owning a GPU.

- The oracle restates four pieces of `fit_with_test`: `CreateCursors`' test
  seed, `_apply_last_tree_to_test`, `_test_loss` through the same Logloss
  kernel the learn curve uses, and `ShrinkToBestIteration`'s second
  best-iteration tracker, which is not the detector's.
- It does NOT narrow the fit. An eval set does not reach the learn cursor,
  the borders, the splits or the leaves on the Plain doc-parallel path, and
  with `use_best_model` off a fit WITH an eval set has the same model bytes,
  learn curve and predictions as the same fit without one.
- New identity_break lane **`gbdt-symmetric-eval`**, and a negative control of
  its own, `-D MOJOLEARN_GBDT_EVAL_SABOTAGE=1`, which moves the held-out
  cells and nothing else. The family's own arm moves the leaves and so cannot
  tell a broken test cursor from a broken fit. The control earned its keep on
  the first run: it read DIVERGENT on all nine fixtures for the two held-out
  CURVES and did not move the detector or shrink parts at all, because at the
  lane's first shape the held-out curve never turned and those two fits were
  byte for byte the fit without an eval set. The stopping fits now overfit on
  purpose, and the lane raises if the detector stops firing.
- **GPU COLUMNS OWED.** The lane has a CPU column only; its cells read OWED
  against the three committed GPU records until the next coordinated record.

**`NO_CPU_PATH` no longer ends "among them".** Its single entry hid an unknown
count behind that phrase, and one of the things it named (eval sets) had
already been closed for Ordered boosting. It is now six entries, each naming
what refuses and the structural reason. The guard-by-guard enumeration behind
them, 50 by-name training refusal sites in 16 configuration families, is
`docs/lanes/BRIEF_gbdt_no_cpu_path_2026-09-20.md`, and
`tools/gbdt_cpu_refusal_probe.py` reproduces the table by fitting each
configuration rather than reading the source.

## Unreleased (lane/compare-challenge-nonce)

**`verify --compare` can now tell a run from a transcription.** A commitment
settled the ORDER of two evidence documents and nothing else: the reference
table ships in the wheel with the expected hash of every cell in it, so a
party could write a whole document out of that table, seal it, publish the
commitment first and hand over a file that never executed a line.

- **New `verify --challenge DOC --challenge-from C_A C_B`.** The challenge is
  `sha256(domain || min(c_a, c_b) || max(c_a, c_b))` over the two commitments
  published before the exchange, so neither party can compute it in advance
  and neither controls it alone. It reseeds `identity_break.fixture('hashed')`
  from that value, reruns the document's own lane set on it (about a ninth of
  what `--all` cost on the same box), appends a `challenge` block and prints a
  SECOND line to publish before the documents are exchanged. Two honest
  responses are identical, so an uncommitted response can simply be copied.
- **New `--challenge-commitment-a` / `--challenge-commitment-b`** on
  `--compare`, and a new verdict rung **`CHALLENGE BROKEN` (exit 1)**,
  directly under `COMMITMENT BROKEN` and above every cell outcome. A
  comparison with NO challenge is unchanged: still `AGREE`, still exit 0,
  labelled a weaker result on the RESULT line, exactly as a comparison
  without commitments already is. Documents from earlier releases compare as
  before, and answering a challenge does not move a document's round-one
  commitment.
- What it does NOT prove is in `docs/VERIFY_EXTERNALLY.md` beside what it
  does: it does not prove two PEOPLE; whoever publishes their commitment last
  can grind nonces to steer the challenge, so the fixture is not an unbiased
  draw; and it proves execution of what it covers, on one fixture of one
  kind. `bench/results/verify_reports/` predates the feature and sits at the
  weaker rung.

## Unreleased (lane/catboost-parity)

**BEHAVIOR CHANGE: `GradientBoosting`'s SymmetricTree defaults are now CatBoost's
GPU learner's** (catboost 1.2.10, pinned source 54a8143a). A default-constructed
model fits a different, larger model than before; pass the old values
explicitly to keep an old result. Old -> new, SymmetricTree only:

- `n_estimators` 100 -> 1000 (`boosting_options.cpp:13`).
- `learning_rate` 0.03 -> CatBoost's auto-selection from the pool when
  `learning_rate`, `l2_leaf_reg`, `leaf_estimation_method` and
  `leaf_estimation_iterations` are unset and the loss is RMSE, Logloss or
  MultiClass (GPU coefficient rows, `options_helper.cpp:221-288`); 0.03
  otherwise. The value used is `learning_rate_`.
- `random_strength` 0.0 -> 1.0 (`oblivious_tree_options.cpp:17`); unset under
  the L2/NewtonL2 scores, which carry no noise term, it resolves to 0.0.
- `bootstrap_type` no sampling -> Bayesian with `bagging_temperature` 1.0
  (`bootstrap_options.h:16-18`). For QueryRMSE, PairLogit and YetiRank that
  default samples whole queries, which is not implemented, so an unset
  bootstrap is refused by name for those losses (pass `bootstrap_type='No'`).
- `leaf_estimation_iterations` unset -> 1 when there are fewer than 200
  iterations and fewer than 20 features (`options_helper.cpp:290-307`).
- `boost_from_average` unset on MAE, Quantile and MAPE: False -> True
  (`options_helper.cpp:353-374`; all policies, as theirs). Their starting
  constant (`CalcSampleQuantile` with the 1e-6 delta adjust, and the MAPE
  weighted median) is now implemented and reproduces CatBoost 1.2.10 CPU's
  bias bit for bit on 40 cases. A model whose bias is -0.0 now writes it.

Depthwise and Lossguide keep 100 iterations, 0.03, no noise and no
bootstrap. `GradientBoostingClassifier` and `GradientBoostingRegressor` now
defer every default to `GradientBoosting` (their `l2_leaf_reg` default is
None rather than 3.0, because an explicit l2 turns the learning-rate
auto-selection off). Every GBDT lane of `tools/identity_break.py` and
`tools/repeat_run_stability.py` passes its earlier configuration explicitly,
and the covered lanes reproduce their shipped reference hashes on the CPU
route under those pins. A CPU-only install still refuses the new defaults'
Bayesian bootstrap and noise by name on most losses (`NO_CPU_PATH`).

- `boosting_type` ('Plain' or 'Ordered'): CatBoost's GPU Ordered boosting
  (`TDynamicBoosting`), with `fold_len_multiplier`, `fold_permutation_block`
  and `permutation_count` as its knobs. **Unset, it is CatBoost's GPU default:
  Ordered under SymmetricTree below 50,000 rows at 500 iterations or more**
  (`catboost_options.cpp:802-807`, `defaults_helper.h:33-42`), Plain otherwise,
  for the multiclass losses and for the L2 scores. Refused by name where
  CatBoost refuses (non-symmetric trees, multiclass, L2 scores, Exact leaves)
  and where it is not implemented (CTR categoricals, the ranking losses). An
  eval set, the overfitting detector and use_best_model work as on a Plain fit.
- `feature_border_type`: all seven of CatBoost's border selections
  (GreedyLogSum stays the default), matching CatBoost 1.2.10's own borders bit
  for bit on 294 oracle cases.
- The score-noise add in both pointwise scorers is now a pinned fma under
  IDENTICAL (IDENTITY_PATHS row 96); no recorded lane reached it.
- CPU host path: the symmetric Logloss fit now restates the Bayesian,
  Bernoulli and Poisson bootstraps and the score noise, so a CPU-only
  verifier can check a default-constructed Logloss fit (the
  gbdt-catboost-defaults lane). Ordered boosting and every border type have
  CPU host paths too; weighted and one-hot Ordered fits
  are GPU only and refused by name on the CPU.
- Multi-GPU: `fit_boosting` accepts Ordered fits and every border type
  (feature histograms partitioned by whole packed groups; the permutations,
  folds, cursors, leaves and border selection stay on the root device). New
  identity lanes par-ordered and par-border-types. The two-GPU columns are
  owed; see docs/lanes/LANE_STATUS_catboost-parity.md for a logical-shard
  finding on the Plain partition.
- New identity lanes: gbdt-catboost-defaults, gbdt-ordered,
  gbdt-ordered-bayesian-noise, gbdt-border-types, gbdt-bfa-quantile,
  par-ordered, par-border-types.

## 0.8.8 (published 2026-09-19)

A verifier/reference patch using the unchanged 0.8.7 native binaries. CPU replay
of GPU-written CTR models no longer erases the recorded GPU model-byte reference;
a real mismatch still fails. Targeted parallel-lane reference updates and exact
new-wheel light smoke evidence accompany publication. This patch does not add
CPU implementations of GPU-only parallel lanes or certify physical multi-GPU use.

Alpha Python/reference patches can now reuse a published wheel when its native
compile inputs are unchanged. The wheel records separate package and native
source commits, preserves the parent's build proof, and requires a new installed
smoke. Unchanged algorithms do not require a repeat library-wide campaign. Prepared
alpha publication now defaults to the light profile; full certification remains
an explicit choice.

## 0.8.7 (published 2026-09-18)

This release was built from integrated `main` at `4e1828f90`, including the
tokenizer/corpus additions, expanded CPU model bundle and verifier, loaded
language-model proof, and experimental distributed and cross-validation APIs.
The explicit light release profile checks the exact installed Apple/NVIDIA
wheels; it does not promote every exposed configuration to numerical
certification. Both macOS and Linux wheels are published on PyPI. See
`docs/lanes/RELEASE_087_LIGHT.md` for the release evidence and boundary, and
`docs/lanes/RELEASE_PROCESS_ALPHA.md` for the post-release simplification plan.

- `GPT2Tokenizer` is renamed `BpeTokenizer` (2026-09-18). It ships no GPT-2
  vocabulary and `TrainedBpeVocabulary.tokenizer()` returns one over mojolearn's own
  trained table, so the old name described a model it does not carry.
  `mojolearn.GPT2Tokenizer` and `mojolearn.tokenizer.GPT2Tokenizer` remain importable as a
  deprecated alias of the SAME class, with a DeprecationWarning. The host binding's
  entries moved from `gpt2_*` to `bpe_*`; the Python door still reads a binding built
  before the rename. No id moved: the `tokenizer` and `bpe-trainer` identity lanes hash
  the same before and after (docs/lanes/LANE_STATUS_tokenized-corpus.md).

**0.8.6 WAS NEVER PUBLISHED, and its number is skipped.** It was frozen on branch
release/0.8.6, built on three GPU boxes, packed, audited and partly recorded, and then folded
into 0.8.7 on 2026-09-16 rather than finished. The reason was a defect in the wheel it would
have shipped: `verify --all` returned VERIFIED as soon as one part read IDENTICAL, before it
looked at REFUSED, so a CPU-only install printed VERIFIED, exit 0 on 44 IDENTICAL and 288
REFUSED parts. The fix (below) is in an inventoried file, so shipping it meant rebuilding all
three Linux sets and the macOS wheel and re-recording, which is the whole release again. Main
never carried the 0.8.6 version bump, so nothing outside that branch ever claimed it. The
wheels that were built are kept as evidence, are not release candidates, and must not be
published; docs/lanes/RELEASE_087_PLAN.md on release/0.8.7 names each artifact.

This release expands native kernels, CPU host bindings, and the installed verifier.
It requires fresh wheels and installed-wheel qualification from the final release
source; the earlier 0.8.7 packaging freeze does not qualify these changes.

- The wheel verifier exposes 162 default CPU lanes, opt-in execution of pending
  CPU routes and 17 logical-shard drivers, and repeated portable saved-model
  checks (`verify --models-only`). Batch invariance and optional step/full,
  gradient, scaling, ragged-shape, and RL-pair probes retain their actual outcomes.
  Pending references remain OWED; CPU logical shards do not qualify physical
  multi-GPU execution. See `docs/VERIFY.md` and the coverage inventory for scope.

- **Every inference class can hold its projection weights as bf16 bits or as int8 codes with a power-of-two exponent per row, and computes bit for bit what the fp32 block computes from the exactly materialized weights.** Two new GEMM profiles sit beside `mojolearn.identical.gemm.fp32.v1` and are named in `gemm/IDENTICAL_LOWBIT_CONTRACT.md` (DEVIATIONS 2900 to 2909): `bf16f32.v1`, which widens bf16 exactly and runs the fp32 profile's arithmetic (a fused flat plan that reads the bf16 right operand directly at the decode shape, a widen plan above it, both required to agree bit for bit), and `int8i32.v1`, which quantizes each row to `[64, 128)` by a power-of-two scale with round-to-nearest-even codes clamped to `[-127, 127]`, accumulates in Int32 exactly, and dequantizes with one exact multiply. Six seams live in `checks/numerics.mojo`, the oracles in `gemm/host/gemm_lowbit_oracle.mojo`, the kernels in `gemm/checks/gemm_lowbit.mojo`, the gates in `gemm/checks/gemm_lowbit_check.mojo` (`pixi run check-gemm-lowbit` and two sabotage arms that must fail). The linalg extension and its host binding export `gemm_bf16`, `gemm_int8`, `quantize_int8`, `dequantize_int8`, `to_bf16`, `from_bf16` and `lowbit_profile_version`; `mojolearn.linalg` exposes `matmul_bf16`, `matmul_int8`, `to_bf16`, `from_bf16`, `quantize_int8` and `dequantize_int8`; `Array` learns uint16 and int8. `mojolearn.lowbit.pack(weights, "bfloat16" | "int8")` packs every 2-D tensor of a weight dict and `TransformerBlock`, `Mamba1Block`, `Mamba2Block`, `Mamba3Block`, their `*Inference` classes, `MLPInference`, `SambaInference` and `LanguageModelInference` (a dict keyed by `parameter_names`) accept the packed dict, materialize it exactly on the selected backend (the linalg kernels on a GPU, the linalg host binding on a CPU, the NumPy spelling with no binding) and expose `weight_format`. Fourteen lanes join the identity harness: `gemm-bf16`, `gemm-int8`, and `-bf16w` / `-int8w` forms of `transformer`, `mamba1`, `mamba2`, `mamba3`, `mlp` and `samba`, in their families' covered sets and in `PUBLIC_PENDING_LANES` as `no reference`. Measured on the Apple M4 (Metal) 2026-09-17: the nine GEMM gates pass, both sabotage arms fail the oracle gates, 16 Python linalg tests and 8 block tests pass (packed block equals fp32 block on the materialized weights, and moves against the raw fp32 model), and all fourteen lanes read STABLE on base, ties and denormal at two repeats. OWED: three-vendor columns for the fourteen lanes (the two profiles have them at the gate shapes, below), a fused bf16 plan inside the blocks (today the blocks materialize and run fp32), and any timing claim (none is made). (lane/identical-lowbit-inference, 2026-09-17)
- **`mojolearn.identical.gemm.int8i32.v1` runs its product on the integer matrix units of NVIDIA (IMMA, `mma.sync` m16n8k32 s8 to s32) and AMD CDNA3 (MFMA `v_mfma_i32_16x16x32_i8`), bit for bit what the flat kernel and the host oracle compute.** By construction: an int8 product is exact and an Int32 sum of exact integers is order-free, so the unit's tile and internal summation are scheduling and the only floating steps stay in `dequant_int8_pinned` (contract clause L-9, DEVIATION 2910). Ragged `k` and tile edges are padded with zero codes, never floats. `checks/kernel_matrix.mojo::lib_int8_matrix_unit_for` names the columns (NVIDIA, AMD; Apple and the CPU stay on the flat kernel), `identical_gemm_int8_into` dispatches, `-D MOJOLEARN_INT8_FORCE_FLAT=1` pins the flat plan, and `check_int8_mma_matches_flat` requires both plans' bits to match on every shape plus ragged shapes with k = 17, 31, 33, 100, 1000 and 4097. Measured 2026-09-17 on a RunPod H100 (`bench/results/lowbit/2026-09-17_h100-lowbit-mma/`) and a DigitalOcean MI325X (`bench/results/lowbit/2026-09-17_mi325x-lowbit-mma/`): int8 dispatch mma, 10 gates 0 failed on each, forced-flat the same, both sabotage arms failing; with the M4 record the nine gate shapes of both low-bit profiles have three vendor columns against one host oracle. OWED: a CDNA2 form and any timing claim (none is made). (lane/int8-mma, 2026-09-17)
- **`TransformerBlock` maps the common decoder families.** New constructor options `rope_theta`, `rope_scaling` (linear, llama3), `rope_dim`, `max_positions`, `qkv_bias`, `o_bias`, `norm` (rmsnorm, layernorm, rmsnorm_offset), `norm_eps`, `norm_bias`, `mlp` (swiglu, gelu, gelu_tanh, geglu, geglu_tanh), `mlp_bias`, `qk_norm`, `attn_softcap` (DEVIATIONS 2930 to 2939 and 2943 to 2948, `transformer/block_options.mojo`); the default record is the frozen profile bit for bit and sends the old binding lists; every option is spelled on the host oracle and the device through `checks/numerics.mojo` and gated by `transformer/checks/transformer_options_check.mojo` (device equal to oracle at all thirty stages, each option shown to move its own stage). Unsupported values and presence/flag mismatches refuse by name. The absolute-position ceiling is restated as the rotation angle's domain (8192.0), which a linear scaling factor widens. Backward: default options only. RUN OWED on every column. (lane/block-options, 2026-09-17)
- **A generalizable checkpoint loader, `mojolearn.models`.** `CausalLM.load(path, weight_format="float32"|"bfloat16"|"int8")` assembles embedding, N `TransformerBlock`/`Mamba1Block`/`Mamba2Block` layers, final norm and head from a Hugging Face `config.json` and its `.safetensors` shards (pure-Python reader, memory-mapped, one copy per tensor; F32/BF16/F16/I32/I64, F16 widened by bit construction, BF16 through `lowbit`'s exact shift) with the Samba stack's own three primitives, an option matrix keyed by `model_type` (llama, mistral, qwen2, qwen3, gemma, gemma2, phi3, mamba, mamba2) that refuses BY NAME every field a block cannot honor, `forward`/`allocate_state`/`step` and greedy `generate` (ties to the lowest index). `models.Tokenizer.from_pretrained` loads `tokenizer.json` for the byte-level BPE families with the pre-tokenization pattern a parameter (DEVIATION 2960: GPT-2 through the compiled binding; Llama 3 and Qwen 2 cut in Python over the byte codes, merged by the package's Python BPE); SentencePiece families are refused by name. `mojolearn.lowbit` is NumPy-free (packed tensors are `Array`s; the no-binding spelling is `checks/numerics.mojo`'s seams over Python ints, held to the NumPy oracle in its test). NO REAL CHECKPOINT HAS BEEN LOADED YET; tests load synthetic ones only. (lane/model-loader, 2026-09-17)
- **The model leg (`bench/model/`, `tools/model_leg/`): one real open model through `mojolearn.models` on four columns, and its time against the incumbent's fast default.** `bench/model/harness.py` greedy-generates the twenty prompts of `bench/model/prompts.txt` in each weight format (`float32`, `bfloat16`, `int8`) and records per prompt the SHA-256 of the generated ids and of the first-step logits, prefill and decode milliseconds per token (median of three runs after one warm-up), the numeric mode read back from the binaries and the model's config hash; `bench/model/torch_twin.py` runs the same through `transformers` + PyTorch as shipped (bf16 on a GPU, fp32 on a CPU) and under `torch.use_deterministic_algorithms(True)`; `bench/model/diff.py --diff` reads IDENTICAL xN or DIVERGENT per format and prompt and `--ratio` prints "identical mode takes X times the incumbent's time" per phase; `tools/model_leg/run_leg.sh`, `run_leg_amd.sh`, `run_leg_cpu.sh` and `run_local_m4.sh` are the four columns' runners, each staging the checkpoint from the R2 dataset store (`models/SmolLM2-360M` is pinned in `bench/results/dataset_store/manifest.tsv`; DEVIATION 2704 forbids a download on a rented box). RUN OWED on every column; every cell of `bench/model/README.md` reads RUN OWED. (lane/model-leg, 2026-09-17)
- **`UMAP.transform` no longer depends on the query batch.** A row asked alone and the same row
  asked inside a batch returned different embeddings, by up to 1.97 on a map whose clusters sit
  about 11 units apart, and adding one row to a request of ten thousand moved another row by
  1.36 because the refinement epoch count fell from 100 to 30 at that threshold. Four couplings
  read the whole request rather than the row: the sigma floor's mean, the edge-weight scale, the
  negative-sample counter (keyed on `row * k + j`, a POSITION in the request) and that epoch
  count. All four are per row now, so a batch of N returns the same bytes as N calls of one row,
  measured bitwise on the CPU host route and on Metal with the two routes agreeing to the last
  bit. This is a deliberate divergence from cuML and umap-learn, whose transforms couple a batch
  the same four ways. IT MOVES EVERY RECORDED UMAP TRANSFORM CELL and no fit cell: measured on
  all nine fixtures, the `train` and `model` hashes are bit for bit what they were and `infer`
  and `batch` both move, because the fit is untouched. `umap` and `par-graph-umap` carry
  `LANE_REVISIONS` entries so their committed cells read OWED rather than DIVERGENT, `umap` is
  held out of the public reference set as `stale reference`, and both lanes' batch EXEMPTION in
  the harness becomes a real batch part, which reads BATCH_MOVED on the old code and STABLE on
  this one. It costs a measured 3.07x on a request above ten thousand queries and 1.43x below
  it (lane/umap-batch-fix, 2026-09-16).

- **Incremental decoding is public on the CPU: `allocate_state`, a carried `state` and `step`
  on `TransformerBlockInference`, `Mamba1/2/3BlockInference` and `SambaInference`**
  (lane/stateful-cpu-decoding). They refused those three by name because the shipped neural
  host binding exported the fresh-state entries alone; it now also exports
  `transformer_forward`, `transformer_decode_step` and `mamba{1,2,3}_forward` /
  `mamba{1,2,3}_decode_step`, over the SAME host functions the fresh entries already call
  with the caller's state instead of a constructed zero. The wrappers therefore stop
  overriding `forward` and inherit the block classes' own call path, so prefill and decode
  are one spelling. Only `backward` (and Samba's `loss` and `train_step`) stays refused.
  What is claimed and measured, not assumed: decoding a sequence one token at a time with a
  carried state is BITWISE the same sequence run as one fresh-state forward pass, at every
  position, for the Transformer at both windows, Mamba-1, Mamba-2, Mamba-3 and the Samba
  stack. `tools/step_vs_full_check.py` is that comparison with a fail-first arm (one ULP on
  one carried cache cell must move a position), and `tools/identity_break.py --step-full` is
  the same question as a recorded part on eight lanes, whose hashes read IDENTICAL between
  the CPU column and the Apple column.
- Every host family now says in `python/mojolearn/host_surface.py` why it does or does not
  ship in the wheels (`wheel_note`, `--wheel-notes`), so an exclusion is never silent
  (lane/expose-inference-surface, for 0.8.7). That entry read "fifteen families ship and
  seventeen do not", each of the seventeen naming the shipping family that served its
  inference instead; lane/ship-cpu-host-families then shipped all thirty-two, so every note
  now begins "Ships:" and none of them names an exclusion.
- **The six k-means lanes are declared inference lanes now, with a GPU reference recording
  behind them, and `SAVED_MODEL_INFERENCE_OWED` is EMPTY** (lane/classical-host-recordings).
  `lane/kmeans-save` gave `KMeans` a `save` and a `load` the day before and put
  `mojolearn-kmeans-1` in the classical host door; nothing carried the lanes into
  `tools/classical_host_gate.py`, so no recording could exist for them, which is the same code
  gap the four predict lanes had. `kmeans`, `kmeans-random`, `kmeans-array`, `kmeans-weighted`,
  `kmeans-sqrt` and `kmeans-classic-pp` are recorded on nine fixtures each at
  `bench/results/classical_host/2026-09-16-nvidia-kmeans`, taken on an NVIDIA A100-SXM4-80GB
  (sm_80). `kmeans-cosine` is not among them and must not be: its fit was refused by name, so
  there is no model to save. The saved models are re-predicted from the CPU host bindings on
  both architectures, x86-64 and arm64, each reading `gate verdict IDENTICAL (54 fixtures,
  exit 0)`. Beside the identity pair each recorded fixture carries `labels` and
  `predict_training_rows`: the claim that a fit's own final assignment survives a file and a
  change of machine, which holds on all 54.
  **The negative control for that gate did not exist until this lane.** The family define's
  only k-means arm was in `host_accumulate`, which only the FIT walks, so a saved model's
  `predict` and `transform` could not be moved by it; rehearsed on a CPU recording before any
  box was rented, `check --expect-mismatch --every-fixture` read `SABOTAGE NOT CAUGHT ON
  FIXTURES` with all 54 in `unmoved`. `MOJOLEARN_KMEANS_PREDICT_SABOTAGE` moves both halves of
  the pair and the family define moves the transform half, which is deliberate: an arm that
  moves `predict` makes `tools/identity_break.py`'s own `predict(X) == labels_` assertion raise,
  and that lane's cells would read REFUSED instead of DIVERGENT. Both arms now read `EXPECTED
  MISMATCH SEEN` with an empty `unmoved` on both architectures.
  The same box retook the k-means identity arms `lane/kmeans-save` had to leave owed (its
  sabotage build had exited 127, so every cell of its control read REFUSED) and took
  `spectral`'s x86 CPU identity column at the published 512-row size, which
  `lane/saved-model-reference-gaps` left owed; both are at
  `bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu`.
- **`DBSCAN.predict`, `AgglomerativeClustering.predict` and `SpectralClustering.predict` are
  declared inference lanes now, with a GPU reference recording behind them**
  (lane/saved-model-reference-gaps). All three shipped on 2026-09-15 and no gate covered any of
  them: `mojolearn.host_model()` dispatched the saved files and nothing said what the answer
  should be. The reason was not policy. `tools/classical_host_gate.py record`, the only tool that
  can make such a recording, had been raising `AttributeError` on main since `--lane-rule-only`
  was added, before it ran a line of work, so nobody could have produced one. The four lanes
  (`dbscan`, `agglomerative`, `spectral`, `spectral-precomputed`) are recorded on nine fixtures
  each at `bench/results/classical_host/2026-09-16-nvidia-predict`, taken on an NVIDIA A100
  (sm_80), and the saved models are re-predicted from the CPU host bindings on two architectures,
  x86-64 and arm64, both reading `gate verdict IDENTICAL (36 fixtures, exit 0)`. The
  predict-only sabotage build is caught on all 36 cells on both, with an empty `unmoved` list.
  Two NVIDIA identity columns and a retaken Apple Metal column are at
  `bench/results/identity_break/2026-09-16_predict-nvidia`; the AMD recording is owed at the next
  release record.
- **The bootstrap, the permutation test, Monte Carlo integration and `kpss_test` now work on
  a CPU-only install.** They were unreachable: each computes a statistic from the caller's
  own data, trains no model and has nothing to save, so the saved-model inference boundary
  never had a side for them to fall on and they refused on a laptop. The boundary exists to
  keep CPU TRAINING OF MODELS internal, not to exclude analysis. The `resample` family now
  ships, taking the wheels from fifteen host bindings to sixteen; its binding already
  registered the three entries and no fit. `kpss_test` could not ship the same way, because
  its family `tsa` holds `holtwinters_fit`, so `kpss_test_binding` moved into a new shared
  module `bindings/kpss_host_test.mojo` that BOTH the tsa reference binding and the shipped
  `forecast` inference binding register, the pattern `holtwinters_host_predict.mojo` set: the
  `_mojolearn_tsa` route already falls back to the forecast binding on a CPU-only install, so
  no fit ships and the two binaries answer through one source. `forecast_host_sabotage` now
  also reports the KPSS arm.
- **`verify --all` can no longer pass a run it did not perform.** On a CPU-only install with
  stale bindings it printed `VERIFIED ... exit 0` while 288 of 332 cell parts REFUSED: the
  user had checked 13 percent of what they believed they had. A refused part did not run, so
  it is never evidence of success and no number of parts that did run makes up for it. Any
  refusal now reads `INCOMPLETE` and exits 4, only a run with nothing refused may read
  VERIFIED, and DIVERGENT is still read first because a wrong answer outranks an absent one.
  The verdict line leads with what was checked: `verified 44 of 332 cell parts (0 divergent,
  27 owed, 288 refused, 0 n/a)`. `docs/VERIFY.md` and `test_verdict_exit_codes` updated; the
  new test was watched failing against the old code first. A healthy install is unaffected
  (278 of 332, VERIFIED, exit 0).
- **`python -m mojolearn verify --self-test`: a user can now watch the verifier fail.** Reading
  VERIFIED meant trusting, unseen, that we wrote an honest table and a real comparison. The
  self-test runs one lane twice through the ordinary comparison path, untouched (must read
  IDENTICAL) and with every value of the input's first column moved up one ULP (must read
  DIVERGENT). The perturbation is real arithmetic at run time, so it needs no sabotage build
  and no second binding, and exit 0 requires BOTH arms, so a comparison stuck on either answer
  fails it. Proven against a deliberately broken comparator in both directions, and that is
  now a test rather than a one-off. The two-sided design earned itself immediately: the first
  perturbation moved a single value and was measured INERT for this lane, so the self-test
  reported NOT TRUSTWORTHY instead of passing quietly; a one-sided version would have shipped
  green. The size used is the smallest measured to move the hash, pinned by a test.
- **`verify --json` and `--json-out PATH` emit evidence rather than a verdict**, rendered from
  the same object as the human report so the two cannot drift. Per cell: the hash computed on
  this machine, the hash expected, the verdict and the wall time. Plus the version and commit,
  the sha256 and size of every binding actually loaded (host bindings included), the device,
  CPU, OS and Python, the committed column each reference came from as an openable path under
  `bench/results/identity_break/`, the self-test result in the same artifact, and lane counts
  kept separate from cell-part counts. Three defects were found by reading the output rather
  than assuming: the binding provenance was EMPTY on a CPU-only install (host bindings load
  under `mojolearn._host.*`, which the scanner did not look at), per-cell timings were dropped
  by `judge_rows`, and the lane counts first read "6 checked of 2 requested" because the
  portable models were folded in with the harness lanes.
- **Every CPU host binding ships in both wheels, and `verify --all` on a CPU-only install
  goes from 39 lanes to 122** (lane/ship-cpu-host-families). The manifest declared thirty-two
  host families and shipped sixteen; the sixteen held back were the CPU TRAINING families,
  kept out by the "inference only, CPU training internal" boundary. That boundary is about
  what a user may TRAIN with, and it was also deciding, as a side effect nobody chose, what a
  user may CHECK: a lane whose host binding is not in the wheel cannot be re-run on the
  machine it was installed on, whatever reference the shipped table carries. So
  `preprocessing`, `tsa`, `solver`, `trees`, `rf`, `gp`, `kernel_methods`, `mixture`,
  `hdbscan`, `gbdt`, `training`, `mamba`, `arima`, `embedding`, `ivf` and `transformer` now
  ship, and 83 lanes join `public_reference_lanes()`, which is now DERIVED from the covered
  lanes rather than hand-listed. Both numbers are measured on one machine: the after arm ran
  131 candidate lanes x 9 fixtures, 4,716 cell parts, 3,855 IDENTICAL, 0 DIVERGENT, 0
  REFUSED, one process per chunk at one core; the before arm ran origin/main's package
  against a host directory carrying exactly origin/main's sixteen wheel bindings, and read
  39. Nine lanes read IDENTICAL and are still held back beside `svc-poly`, because every
  IDENTICAL cell they carry rests on the Apple column alone and a two-column agreement is
  not what the other thirty were promoted on; they join when a release record carries NVIDIA
  and AMD. **CPU training did not become public.** An ordinary `fit` on a CPU-only install
  still refuses by name and still says to train on a GPU and load the saved model; these
  bindings answer the verifier, which fits inside its own private reference scope
  (`python/mojolearn/_cpu_reference.py`), and `test_cpu_inference_boundary.py` passes
  unchanged. The cost was measured rather than estimated: the sixteen add 7,663,488 bytes
  uncompressed and 2,189,221 compressed, taking the macOS wheel from 26,368,494 to
  28,639,807 bytes (+8.61%) and the Linux wheel, projected from the ratio measured over the
  fifteen families in both 0.8.6 candidate wheels, from 70,862,796 to 73,444,604 (+3.64%). The
  compressed figures are measurements: at deflate level 6 with a raw window this reproduces
  the 0.8.6 candidate wheel's recorded compressed sizes exactly on all fifteen of its host
  bindings, where level 9 reproduces none of them. Those wheels were built and audited but
  NEVER PUBLISHED (see the heading above), so the measurement is against an artifact on disk,
  not against anything on PyPI; the last published wheels are 0.8.5's, which carry the byte
  LM's host binding alone.
  A wheel is still dominated by its GPU bindings, 312 MB uncompressed across 91 files on
  Linux, which is why sixteen more CPU binaries move the total so little. Every family's
  `wheel_note` now begins "Ships:" and says what that binding makes checkable that nothing
  else could, and `test_public_inference_bindings_ship_and_packaging_reads_the_manifest`
  asserts that the held-back list is empty, so holding a family back again has to delete
  that assertion and write a reason. `docs/VERIFY.md` gains a section on what the CPU
  training bindings are for (verification, small data, reproducibility, air-gapped checking)
  and what they are not: a host binding is a device kernel restated as a serial host loop so
  it produces the device's bits exactly, single-threaded by construction, so timing one
  against a GPU fit measures that choice and nothing else.
- **What a CPU-only wheel user can verify goes from 9 lanes to 39, for zero extra wheel
  bytes.** `public_reference_lanes()` gained thirty lanes: k-means and its starts, the k-NN,
  radius and kernel-density variants, DBSCAN, the linear, ridge, logistic and decomposition
  lanes, SVC, SVR, the isolation forest, the saved UMAP embedding's transform, and the four
  analysis functions this release exposes. Every one was measured on an Apple M4 CPU column
  at 9 fixtures and 2 repeats against the Apple, NVIDIA and AMD columns: 0 DIVERGENT, and
  their sabotage arm seen to move rather than assumed. Nothing was added to the wheel,
  because each is served by a binding it already carried and each reference hash was already
  in the shipped table, merely never consulted. The promotion waited for the fixture shrink
  (`e2bb9e541`) to publish its scope, since these references ship in the wheel's table and a
  lane whose fixture moved would ship a reference a user's `verify` then fails against; none
  of the thirty is among the thirteen shrunk lanes. `svc-poly` was the one lane not promoted,
  because its cells rest on two columns and cannot meet `--require-columns 4` (nine more
  joined it on the same ground when lane/ship-cpu-host-families widened the set). Measured end
  to end on the CPU-only install afterwards, `verify --all --full` reads
  `VERIFIED (verified 1065 of 1412 cell parts (0 divergent, 158 owed, 0 refused, 189 n/a))`
  in 523.7 s, against 278 of 332 before.
- The same file now records the two measured exposure gaps rather than leaving them in a
  reader's head. `PUBLIC_REFERENCE_CANDIDATES` names the identity lanes that pass every static
  condition for `public_reference_lanes()` (a real train reference on all nine fixtures, a
  `cpu` column in the shipped table, all nine fixtures on all three training GPU columns so
  `--require-columns 4` can be met, and reachability from a binding that ships, so promoting
  them grows the wheel by nothing). Measured on an Apple M4 CPU column at 9 fixtures and 2
  repeats: 0 DIVERGENT and 26 of 27 IDENTICAL x4 on train across every fixture. They are held,
  not live, until the fixture shrink publishes its scope, because these references ship in
  the wheel's table and promoting a lane whose fixture then changes would ship a reference a
  user's `verify` fails against. Two criteria of the list were wrong and a check caught each:
  `svc-poly` rests on two columns and was dropped, and `kpss` was wrongly excluded by a rule
  that demanded the declaring family ship when a shipped binding serves its route.
  `SAVED_MODEL_INFERENCE_OWED` names the saved-model inference that IS implemented and that
  `mojolearn.host_model()` already dispatches but that no gate covers: DBSCAN,
  AgglomerativeClustering and SpectralClustering `predict`, each waiting on a GPU recording,
  and `KMeans.predict`, which has no save format yet.

- `ARIMA` takes exogenous regressors: `fit(y, exog)`, `forecast(steps, exog)` and
  `predict(start, end, exog)` (lane/arima-exog, for 0.8.7), regression with ARIMA errors as
  cuML's. `beta` is packed after `mu`, the regressors are differenced beside `y`, `beta` is
  started by a least-squares regression before the ARMA start values and then fitted jointly
  with them by the L-BFGS, and `x_t beta` is the observation intercept the Kalman filter adds
  to every prediction and forecast. `beta_` and `n_exog_` are new attributes; `exog` is
  `(batch_size, n_obs, n_exog)` (DEVIATION 996), at most 17 regressors (994), and a non-finite
  regressor is refused by name (997). The closed cuBLAS gemms of the observation intercept and
  of the start-value regression are ours to spell, serial ascending fma from zero (995).
  `trend='t'` and `'ct'` stay refused, now pointing at `exog`. Saved models: a fit without
  regressors is still `mojolearn-arima-1` byte for byte, and one with them is
  `mojolearn-arima-2`, carrying the regressors and `n_exog` (998), which the shipped forecast
  host binding predicts from on a CPU-only install. New lanes `arima-exog` and
  `arima-exog-seasonal`; the existing ARIMA lanes are unchanged. Evidence:
  bench/results/identity_break/2026-09-15_arima-exog.
- Gaussian process kernel hyperparameter optimization (lane/gp-optimizer, for 0.8.7).
  `GaussianProcessRegressor(optimizer="fmin_l_bfgs_b", n_restarts_optimizer=k, random_state=s)`
  maximizes the log marginal likelihood as scikit-learn does, with the same bits on every
  column rather than SciPy's bits. The kernels gain scikit-learn's `*_bounds` arguments ("fixed"
  included), `theta`, `bounds` and `n_dims`; `log_marginal_likelihood(theta, eval_gradient)`
  now answers at any theta. DEVIATION 2880: dK/dtheta for every node kind on the device and in
  the CPU verifier, K^-1 by the identical Cholesky solve, one pinned float32 trace fold.
  DEVIATION 2881: a projected L-BFGS in Python float64 (10 pairs, Armijo backtracking on the
  projected path, pgtol, ftol and a 200 step cap), restarts from Philox keyed by `random_state`,
  the best likelihood winning and ties going to the first run. `optimizer=None` stays the default,
  so every recorded gp cell is unchanged; a callable optimizer is refused by name. New identity
  lanes `gp-optimize` and `gp-optimize-restarts`. The classifier's optimizer is still refused.
- The full CPU identity verification covers every one-device lane (lane/cpu-verifier-gaps-7,
  for 0.8.7). Seven lanes had a CPU host function and no manifest entry, so the gate did not
  run them: gmm-sample and gmm-random-init-sample (the mixture family), gp-sample-y and
  gp-sample-y-normalize (gp), tokenizer, and the two CTR table lanes, whose CPU cells are the
  forest binding's predictions from Metal-saved models because CPU training of CTR tables
  refuses by name. The gate's sabotage host set now builds each family with the defines the
  manifest gives it (`host_surface.sabotage_build_defines`), so the tokenizer binding carries
  its own define and the forest binding the CTR arm, which `MOJOLEARN_HOST_SABOTAGE` does not
  reach; the covered sabotage run names that set's forest binary. A CPU column that loads a
  GPU-saved model reports the model part n/a (the file is the GPU column's bytes) and refuses a
  reload that predicts differently. On one x86 CPU pod all 63 cells read STABLE, the four-column
  diff is OK with 189 OWED parts and nothing DIVERGENT, and every one of those parts moves under
  the sabotage set. Evidence: `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7`.
- Sabotage value arms for the neighbor and IVF host oracles (lane/ties-sabotage, for 0.8.7).
  A fold walked in the other order is exact on the integer `ties` fixture, so the sabotage
  build left the ties cells of knn-cosine, knn-rbc, radius, radius-manhattan, ivf and
  ivf-euclidean unmoved. Sabotage builds now also move every returned distance's bits;
  production builds are unchanged (IDENTICAL to the committed records on all nine fixtures).
  `tools/classical_host_gate.py check` gains `--every-fixture` and `--lane-rule-only LANE`,
  and the CPU identity gate's saved-model sabotage step requires every fixture of every lane.
  Evidence: `bench/results/identity_break/2026-09-15_ties-sabotage`.
- Public CPU inference from a saved Holt-Winters model (lane/inference-holtwinters, for 0.8.7).
  `ExponentialSmoothing` gains `save` and `load` (format `mojolearn-holtwinters-1`) and
  `predict(start, end)`, the in-sample one-step predictions (NaN before `2 * seasonal_periods`,
  whose prediction reads the decomposition's start state the fit does not keep) and the forecast
  beyond `n`, so `predict(n, n + h)` is `forecast(h)` byte for byte. The shipped forecast host
  binding now registers `holtwinters_forecast` and `holtwinters_predict` from
  `bindings/holtwinters_host_predict.mojo` over `holtwinters/host/hw_predict.mojo`, and no fit,
  decomposition or line search; it serves the `_mojolearn_tsa` route on a CPU-only install and
  `mojolearn.host_model` binds a saved model to it on any machine. The reference tsa host
  binding registers both from the same source, and `hw_oracle.mojo::oracle_forecast` forecasts
  through the same body. Evidence: bench/results/identity_break/2026-09-15_inference-holtwinters.
- New `SpectralClustering(prediction_data=True)`, `SpectralClustering.predict`, `save` and `load`
  (`mojolearn-spectral-1`) (lane/spectral-predict, DEVIATION 2860, new capability that neither cuML
  nor scikit-learn has).
  - **Method.** The Nystrom out-of-sample extension of Bengio et al. (NIPS 2003), then the fit's
    own k-means assignment (ties to the lowest centroid index).
  - **New row's affinity.** For `nearest_neighbors`: its `n_neighbors` nearest training rows at
    0.5, the fit's symmetrization of a one-way edge. For `precomputed`: the caller's
    `(n_new, n_train)` affinity.
  - **Threshold.** A used column with `|1 + theta| < 1e-3` is refused by name.
  - **The fit.** It copies the eigenpairs, degree scaling and centroids out; no arithmetic is
    added, and `prediction_data=False` fits as before.
  - **Where it runs.** On the GPU binding and the CPU metrics host binding, which ships, so
    `mojolearn.host_model(path)` predicts on a CPU-only install.
  - **Not promised.** `predict(X_train) == labels_`; the evidence measures the rate
    (bench/results/identity_break/2026-09-15_spectral-predict).
- Public CPU inference from a saved gradient boosting model whose categorical columns are above
  `one_hot_max_size` (real CTR tables: Borders at three priors and FeatureFreq) or whose
  `ExperimentalTwoLevelFeatureFreq` tree splits on a combination (tensor CTRs with split history),
  fitted on a GPU (lane/inference-gbdt-ctr-tables). `mojolearn.host_model(path)` parses the
  `ctr_table`, `ctr_entry`, `tensor_ctr_registry` and `feature_freq_tensor` records and the shipped
  forest host binding applies them with the same `expand_raw_columns` and tensor apply body the GPU
  predict calls (the tensor apply moved to `gbdt/models/tensor_ctr_apply.mojo`, which has no device
  import). Unseen and seen-once categories take the tables' empty and learned values; a NaN, negative
  or non-integer categorical value and a CTR type with no apply-time arithmetic are refused by name.
  No saved-model record changed. On two new identity lanes the x86 CPU predictions from the Metal
  column's saved models equal the Metal cells (train, infer, model and batch, base, ties and odd), and
  both the forest and a new CTR sabotage build move every train, infer and batch cell; the NVIDIA and
  AMD cells are owed (bench/results/identity_break/2026-09-15_gbdt-ctr-tables).
- Public CPU inference from a saved model for `GaussianProcessRegressor` (every kernel the fit
  accepts, `normalize_y` included), `GaussianProcessClassifier` (binary and one-vs-rest) and
  `GaussianMixture.sample` (lane/inference-neighbors-density). `GaussianProcessRegressor` gains
  `save` and `load` (`mojolearn-gp-1`); `mojolearn.host_model(path)` returns the predictive mean
  and std, and a saved classifier's labels and probabilities, through a new INFERENCE-ONLY host
  binding, `_mojolearn_gp_infer_host`, which registers `gpr_predict` and `gpc_predict` and no fit,
  Laplace Newton loop, log marginal likelihood or Cholesky door; the gp reference binding still
  does not ship. `_mojolearn_mixture_infer_host` also registers `gmm_sample`. On the M4 the 27
  models saved by the Metal classes (base, ties and dupes) match every committed infer cell that
  carries one (the 166-lane record's Apple, NVIDIA and AMD columns for the four RBF and Matern GP
  lanes, the gpc and gmm-sample lanes' own columns); gp-normalize-y has no committed column, and its
  x86 CPU infer hashes equal the Metal recording on all three fixtures, so its NVIDIA and AMD cells are
  owed. On x86 the 27 recordings read IDENTICAL through the shipped bindings and EXPECTED MISMATCH SEEN
  under the host sabotage build, with a new sample sabotage arm in `gmm_sample_binding` because the
  existing arm cannot reach a sample drawn from a saved model
  (bench/results/identity_break/2026-09-15_inference-gp-gpc-gmm-sample).
- Public CPU inference from a model saved on a GPU for more neighbor and density lanes
  (lane/inference-neighbors-density). `NearestNeighbors` on the sqeuclidean, manhattan,
  chebyshev, cosine and minkowski metrics and over the random ball cover, the
  distance-weighted `KNeighborsClassifier` and `KNeighborsRegressor`, and `KernelDensity`
  on the five kernel and metric pairs and with sample weights, all through host bindings
  that already shipped. `RadiusNeighbors` gains `save` and `load` (`mojolearn-radius-1`) and
  answers `radius_neighbors` on a CPU on its four metrics. `IsolationForest` gains `save`
  and `load` (`mojolearn-iforest-1`; the file holds the training matrix and the knobs, since
  every scoring call rebuilds the forest, DEVIATION 874). `GaussianMixture` gains `save` and
  `load` (`mojolearn-gmm-1`) and `HDBSCAN(prediction_data=True)` gains `save` and `load`
  (`mojolearn-hdbscan-2`) for `mojolearn.hdbscan.approximate_predict`, `membership_vector` and
  `all_points_membership_vectors`; their CPU entries ship
  in two new INFERENCE-ONLY host bindings, `_mojolearn_mixture_infer_host` (232,112 bytes
  on the M4, against 406,456 for the reference binding with the fit) and
  `_mojolearn_hdbscan_infer_host` (287,280 bytes with the two soft clustering entries), which
  register the scoring or prediction entries and no fit: `nm` finds no EM step, Boruvka MST or prediction data
  generation in either file. `mojolearn.host_model(path)` loads a saved model into a host
  class that binds them, as the scalers are served through the estimators binding; CPU fits
  still refuse. On the
  M4, one core: the 54 neighbor and KDE models saved by the Metal classes predict IDENTICAL
  on the CPU against their recordings and the 166-lane record's Apple, NVIDIA and AMD infer
  cells, and the host sabotage build reads DIVERGENT on 50 of 54
  (bench/results/identity_break/2026-09-15_inference-neighbors-density). The 16 iforest,
  gmm and hdbscan models saved by the Metal classes predict IDENTICAL through the
  inference-only bindings against their recordings and every committed infer cell, and the
  sabotage build reads DIVERGENT on all 16
  (bench/results/identity_break/2026-09-15_inference-iforest-gmm-hdbscan). The NVIDIA and AMD
  recordings, and a CPU identity gate workflow that builds the inference-only families, are
  owed.
- New `python -m mojolearn verify --all` (`--quick`, `--full`, `--lanes`, `--fixtures`,
  `--repeats`, `--json`): runs the identity_break lanes from the installed package (the
  wheel's byte copy of `tools/identity_break.py`, fixtures generated from its seeds) and
  compares every train, infer, model and batch part with
  `mojolearn/verify_reference/table.json`, a table generated from the committed records
  with each hash's record directory and commit, plus small GPU-trained saved models whose
  bytes and loaded answers must equal the recorded ones. Parts read IDENTICAL, DIVERGENT,
  OWED or REFUSED; exit codes are `verify`'s. On a CPU-only install it runs the public CPU
  reference lanes and the portable models. The card verifier's list-every-stage flag is now
  `--all-stages`. Maintainers regenerate the table with `verify --all --emit-reference` and
  the models with `verify --emit-models` (docs/VERIFY.md).
- New `GaussianProcessClassifier`, the Laplace approximation with
  `optimizer=None`: binary fits by the posterior-mode Newton loop, one-vs-rest past two classes,
  `predict`, `predict_proba`, `latent_mean_and_variance`, `log_marginal_likelihood_value_`, `save` and
  `load`. The loop stops by the reference's rule read on an identical float32 likelihood, so its
  iteration count (`n_iter_`) is the same on every column (DEVIATION 2830, closing DEVIATION 1766);
  the float32 orders are pinned (DEVIATION 2831); the probability runs in float64 through a new
  portable float64 erf (DEVIATION 2832); the one-vs-rest composition is DEVIATION 2833.
  `optimizer`, `n_restarts_optimizer`, `warm_start`, `copy_X_train=False`, `random_state`,
  `multi_class='one_vs_one'` and `n_jobs` are refused by name. The device path is
  `gaussian_process/classifier.mojo`, the CPU host restatement `gaussian_process/host/gpc_oracle.mojo`
  in the gp host family; fit on a CPU-only install stays internal, and a saved classifier predicts on
  the CPU through `GaussianProcessClassifier.load` or `mojolearn.host_model`. New identity lanes
  `gpc` and `gpc-multiclass` with batch declarations. Apple M4: the Metal column STABLE on all 18
  train, 18 infer, 18 model and 18 batch cells, and the Metal fit and prediction equal to the host
  binding's bit for bit; the gp regressor's cells unchanged, IDENTICAL x4 against the 166-lane
  record. Against scikit-learn 1.9.0 on the lanes' fixtures: every label agrees, the largest
  probability difference is 1.6e-4 (the `wide` fixture) and at most 1.1e-5 elsewhere. The NVIDIA and
  AMD columns are owed to the release record.
- New `KMeans.transform` and `KMeans.fit_transform`, with cuML's `KMeans.transform` as the
  reference: the distance from every row to every fitted center under the model's `metric`
  (squared for the default `'euclidean'`, cuVS `L2Expanded`; the root for
  `'l2_sqrt_expanded'`), float32 `(n_samples, n_clusters)`. Each cell is the fused
  assignment kernel's cell, so the distance at `predict`'s label is the row minimum bit for
  bit. On the GPU binding and the CPU core host binding; cosine refuses by name as its fit
  does. The seven fitting k-means lanes add `transform` to their infer and batch cells.
  Apple M4 Metal and CPU columns only; the NVIDIA and AMD cells are owed to the release
  record.
- Public CPU inference for saved ARIMA models, UMAP embeddings and the whitened full-SVD PCA.
  `ARIMA.save`/`load` and `UMAP.save`/`load` are new; `mojolearn.host_model(path)`, or the
  classes on a CPU-only install, predict (in sample and out of sample), forecast and read the
  fitted ARIMA attributes, and transform with a saved UMAP embedding. A UMAP transform's answer
  depends on the query batch by its contract; the CPU answers the GPU's bytes for the same batch.
  ARIMA inference ships in a new host binding, `_mojolearn_forecast_host`, which carries
  predict and forecast and no fit (about 270 KB on macOS arm64); the ARIMA fit stays a source
  reference build. `ExponentialSmoothing.fit` now refuses on a CPU-only install outside the
  internal reference context, as every other CPU fit does. Apple M4 CPU column against the
  committed Apple, NVIDIA and AMD columns; the model cells of the new save formats are owed to
  the release record.
- New `GaussianProcessRegressor.sample_y(X, n_samples=1, random_state=0)`, with
  scikit-learn's `sample_y` as the reference: float32 `(n_rows, n_samples)` draws from the
  posterior, the mean plus a factor of the predictive covariance `k(X, X) - V^T V` times
  standard normals, un-normalized under `normalize_y`. The covariance is factored by the
  identical Cholesky at its pinned `2^-20` jitter, the normals are position-mapped Philox keyed
  by `random_state` (tag "GPSY") through the guarded Box-Muller, and the products are the
  identical GEMM (DEVIATION 2793), so one model, `X`, `n_samples` and `random_state` give the
  same bits on every vendor; they are not scikit-learn's bits. A covariance that does not
  factor, `random_state=None` and the unfitted-prior arm refuse by name. On the GPU binding,
  with an internal verifier arm in the gp host binding; public CPU `sample_y` from a saved
  model is left to the neighbors and density inference lane. New `gp-sample-y` and
  `gp-sample-y-normalize` identity lanes (batch n/a: the rows of one call are jointly
  correlated), so the gp lanes' recorded cells do not move.
- New `GaussianMixture.sample(n_samples)`, with scikit-learn's `BaseMixture.sample` as the
  reference: `(X, y)` with the component counts a multinomial draw over `weights_` and the
  rows grouped by component ascending. Every draw is position-mapped Philox keyed by
  `random_state` (DEVIATION 2791), the normals are a Box-Muller transform with the pinned seams,
  and each row is the forward substitution through the fitted `precisions_cholesky_`
  (DEVIATION 2792), so one model and one `random_state` give the same bits on every vendor;
  they are not scikit-learn's bits. `X` is float32 and `y` int32. On the GPU mixture binding
  and the internal mixture host binding; public CPU exposure from a saved model is left to
  the neighbors and density inference lane. New `gmm-sample` and `gmm-random-init-sample`
  identity lanes, so the gmm lanes' recorded cells do not move. Apple M4 Metal and CPU
  columns only; the NVIDIA and AMD cells are owed to the release record.

- CPU inference from saved models for StandardScaler, MinMaxScaler, Lasso, ElasticNet,
  KernelRidge (linear and rbf kernels), Nystroem (linear and rbf) and RBFSampler: each
  gains `save` and `load`, and `mojolearn.host_model(path)` transforms or predicts on a
  CPU-only install. The shipped estimators host binding serves the six entries
  (`standard_transform`, `minmax_transform`, `cd_predict`, `kernel_ridge_predict`,
  `nystroem_transform`, `rbf_sampler_transform`); the preprocessing, solver and
  kernel_methods reference bindings still do not ship. The saved-model classical lanes
  grow from 12 to 29, adding the ols, ridge and logistic option variants. On a CPU-only
  install `StandardScaler.fit`, `MinMaxScaler.fit`, `Lasso.fit` and `ElasticNet.fit` now
  refuse by name outside the internal reference scope, as every other estimator fit does.
- New `IVFIndex.extend(X)`, with cuVS `ivf_flat::extend` (fixed centres) as the reference.
  - The new rows are assigned to the built index's fixed centres by the build's own
    assignment and tie rule, then appended to their lists under the ids `n_rows_`,
    `n_rows_ + 1`, and so on. `extend_labels_` names each row's list.
  - Extending by a set of rows in one call, or in several calls in the same order,
    gives the same index bytes, so a search after it is the same everywhere.
  - It runs on the GPU binding and on the CPU host bindings, the shipped
    `_mojolearn_ivf_search_host` included, so a GPU-built index saved and loaded
    on a CPU can be extended there.
  - Caller-chosen ids and `adaptive_centers` are not implemented.
  - New identity lane `ivf-extend` with a batch declaration. Apple M4 Metal and
    RunPod x86 CPU columns only; the NVIDIA and AMD cells are owed to the release
    record. Evidence: bench/results/identity_break/2026-09-15_ivf-extend/.
- Public CPU inference for saved `IVFIndex` indexes and `Embedding` tables.
  - `IVFIndex.fit` now builds the index and `search` answers from it, as two
    binding calls. The train, infer and batch hashes are unchanged against
    the committed Apple, NVIDIA and AMD columns, on Metal and on the CPU.
  - `IVFIndex.save` / `load` (`mojolearn-ivf-flat-1`) and `Embedding.save` /
    `load` (`mojolearn-embedding-1`) carry GPU-built state to a CPU, and
    `mojolearn.host_model` returns a host instance for either file.
  - Two new host bindings ship in the wheels, `_mojolearn_ivf_search_host`
    (search only) and `_mojolearn_embedding_infer_host` (lookup only). Each
    serves its family on a CPU-only install that has no reference binding.
  - `Embedding.backward` refuses on a CPU-only install outside the internal
    verifier.
  - Evidence: bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/.
- CPU inference from saved SVM models for `SVC(kernel='linear')`, `SVC(kernel='poly')`,
  `SVR` and `SVR(kernel='linear')`: `SVR` gains `save` and `load` (format
  `mojolearn-svr-1`), and `mojolearn.host_model(path)`, or the classes on a CPU-only
  install, answer `decision_function` and `predict` through the shipped svm host binding's
  `svc_predict` and `svr_predict`. No entry was added to that binding. The svm family's
  saved-model lanes grow from one (`svc`) to five. The svm host sabotage build gains an
  intercept arm in the fit and a decision arm in predict, so on an integer-grid fixture
  such as `ties` a saved model's file and its predictions now move there too. Apple M4 Metal recording checked on an x86 CPU against the committed Apple,
  NVIDIA and AMD columns; the SVR model cells and every `svc-poly` GPU cell are owed to the
  release record.
- Public CPU `Cholesky` inference. On a CPU-only install `Cholesky().fit(A)` factors a given
  matrix and `solve` answers from it, and `Cholesky.save` / `Cholesky.load` (or
  `mojolearn.host_model`, which returns a `HostCholesky`) carry a factor from a GPU box to a
  CPU. The door moved into the linalg host binding, which ships in the inference wheel; the
  `cholesky` identity lane is now the linalg family's and a public CPU reference probe. Apple
  M4 CPU column: train, infer and batch IDENTICAL x4 against the 166-lane record, the new
  saved-factor model cells OWED to the release record, the sabotage build DIVERGENT on every
  train and owed cell.
- mojolearn ships no third-party vocabulary or data file. The GPT-2 rank table and the GPT-2
  reference fixture are removed from the tree; `GPT2Tokenizer` is the GPT-2 format's byte-level
  BPE algorithm over a vocabulary the user supplies: `GPT2Tokenizer.from_files(encoder_json,
  vocab_bpe)`, `from_ranks_file(path)` or `from_token_bytes(tokens)`, and `GPT2Tokenizer()` refuses
  by name. `<|endoftext|>` is the id after the last rank (`eot_token`, `n_vocab - 1`); the
  `data_directory` argument, `MOJOLEARN_TOKENIZER_DATA` and `mojolearn.tokenizer.data_dir` are
  gone. The Unicode class table is generated at build time from Python's `unicodedata` (Unicode
  16.0.0, sha256 pinned) and compiled into the tokenizer binding. The gates and the `tokenizer`
  identity lane use a synthetic vocabulary mojolearn trains itself, so the lane's hashes changed;
  its record cells read OWED until the next release record (`LANE_REVISIONS`). NOTICE carries
  only the copyright, the license and the trademark sentence.
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
- New public CPU inference from GPU-trained weights for the Mamba blocks and the Samba
  stack. `Mamba1BlockInference`, `Mamba2BlockInference` (with `dt_limit`) and
  `Mamba3BlockInference` run the block's `forward` from a zero state. `SambaInference`
  runs `SambaStack.forward`'s logits, from `SambaStack.save_checkpoint` files or a config
  and its weights. Both honor ragged `lengths`. A carried state, `step`, `allocate_state` and
  `backward` refuse by name. The byte LM's public class stays `LanguageModelInference.from_checkpoint`.
  - **Binding:** `_mojolearn_neural_host` gains six forward-only entries. `nm` on the Linux
    build finds no backward, optimizer, loss or decode symbol, where the reference training,
    Mamba and Transformer bindings show 20, 71 and 12. The Linux test wheel grows from
    1,891,155 to 1,972,140 bytes.
  - **Identity:** on a CPU column, the mamba1, mamba2, mamba2-dtlimit, mamba3, samba,
    samba-untied-dropout-accum, byte-lm and byte-lm-resident lanes ask their infer, batch,
    batchscale and ragged cells through these classes. The transformer lanes do the same for
    batchscale and ragged. Against the three committed GPU columns, all 90 train, 90 batch,
    90 batchscale and 90 ragged cells read IDENTICAL; the infer and model cells read 126
    IDENTICAL and 54 N/A, with none owed.
  - **Sabotage:** a sabotage build of the neural and byte LM bindings reads DIVERGENT on
    every batch, batchscale and ragged cell, and the batch sabotage reads BATCH_MOVED on
    all ten lanes.
  - **Wheel check:** the installed test wheel answers 16 of 16 public calls byte for byte
    against the source tree.
  - **Evidence:** `bench/results/identity_break/2026-09-15_neural-forward-inference`, one
    AMD EPYC CPU column.
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
  points_to_predict)`: the label and probability of new points under the fitted clustering, on
  the GPU binding and the CPU host binding. Without `prediction_data=True` it refuses by name. A tie in mutual
  reachability distance resolves in (distance, index) order (DEVIATION 1615). The fit is
  unchanged: the committed Apple, NVIDIA and AMD train hashes of the `hdbscan` and
  `hdbscan-leaf` lanes still match. Those lanes and `par-hdbscan` now carry infer and batch
  cells instead of `n/a:transductive`, with the batch and host sabotage builds required to
  move them. Apple M4 Metal and CPU columns only; the NVIDIA and AMD cells are owed to the
  release record.
- New `mojolearn.hdbscan.membership_vector(clusterer, points_to_predict, batch_size=4096)` and
  `all_points_membership_vectors(clusterer, batch_size=4096)`, soft clustering: for each
  point and each selected cluster, the probability of membership. They need
  `HDBSCAN(prediction_data=True)` and refuse by name without it. cuML computes four of the steps in
  float64, which an Apple GPU cannot run; here every step is float32 with pinned seams on the GPU
  binding and the CPU host binding, and rows where cuML overflows to NaN (duplicated points) are
  finite (DEVIATION 1616). The `hdbscan` and `hdbscan-leaf` identity lanes hash both calls in their
  infer cells, and their batch part asks `membership_vector` alone and split.
- `GradientBoosting.fit` takes `group_id`: one string or integer id per
  row (an integer compares by its decimal spelling), each group's rows
  consecutive or the fit raises "group Ids are not consecutive". The grouping crosses into the GPU
  binding and the GBDT host binding as run lengths. `loss="QueryRMSE"` reads it; every other loss
  refuses it BY NAME. `subgroup_id` and `pairs` are refused by name in Python. A fit without them
  sends the same parameter layout as before.
- New `loss="QueryRMSE"` for `GradientBoosting`, the first learning-to-rank loss: a querywise
  target (group means per query, leaf estimation in inverse bin order) on the GPU and restated in
  the GBDT host binding for the CPU reference column. SymmetricTree with the greedy searcher, Newton leaves at
  one iteration by default; a bootstrap, categorical features, an eval set, the pointwise searcher
  and the non-symmetric policies are refused by name. Without `group_id` every row is its own
  query, so the fit learns nothing. Prediction is the ordinary row-wise raw
  score, so saved-model CPU inference covers it. New identity lane `gbdt-query-rmse` with a batch
  part. Apple M4 Metal and CPU columns only; NVIDIA and AMD are owed to the release record.
- New `loss="PairLogit"` for `GradientBoosting`, a pairwise ranking loss through the querywise
  target, on the same arm as QueryRMSE. Without `pairs` the
  pairs are generated from `group_id` and the grades as the reference's default does (every two rows
  of a query with different grades, the higher grade the winner); `fit(pairs=..., pairs_weight=...)`
  takes explicit `[winner, loser]` row pairs inside groups. Two named DEVIATIONs: each row's pair
  derivatives are summed in one fixed order where the reference sums them in thread arrival order,
  and the search weight plane uses second derivatives as weights as the pointwise target does, where
  the reference's querywise branch has the two arms reversed (so at the default Cosine score the trees
  can differ from the reference's GPU and follow its CPU weighting). Each tree's leaf values are
  shifted to average zero after estimation, as the reference does for this loss
  (a shift no pairwise loss or ranking metric can see, but raw predictions do). `pairs` without
  `group_id`, the `max_pairs` subsample and the pairwise-scored variant are not implemented. New identity lane
  `gbdt-pair-logit` with a batch part. Apple M4 Metal and CPU columns only; NVIDIA and AMD are owed
  to the release record.
- New `loss="YetiRank"` for `GradientBoosting`, the sampled-permutation ranking loss of the CatBoost
  reference (`yeti_rank_pointwise.cu` and its two radix-sort passes, through the querywise target), on
  the same arm as QueryRMSE, at the reference's defaults: 10 permutations, decay 0.85, Newton leaves at
  one iteration (changing the method is refused in the reference's words) and an L2 of 0. A query over
  1023 rows is refused as the reference refuses it. Two named DEVIATIONs: each task of at most 1024
  rows runs sequentially on one device thread in the reference's per-document order (draws, the stable
  sort, then each lane's two phases), and the derivative seeds come from a YetiRank stream of
  `random_state` kept apart from the searcher's, so the trees cannot match the reference's GPU bit
  for bit. Leaves are shifted to average zero as for PairLogit; `loss_curve_` is zero because the
  reference's target writes no value. `GradientBoosting(l2_leaf_reg=...)` now defaults to `None`,
  which takes the loss's default (0 for YetiRank, 3.0 for every other loss, the value it was); an
  explicit value is used as given. New identity lane `gbdt-yeti-rank` with a batch part. Apple M4
  Metal and CPU columns only; NVIDIA and AMD are owed to the release record.
- The host (CPU) bindings `python/mojolearn/host_surface.py` marks `ships_in_wheel` ship in
  both wheels under `mojolearn/host/`: ten families, byte_lm, forest, tokenizer, neural, core,
  linalg, estimators, metrics, svm and forecast. The other seventeen families the manifest
  declares (preprocessing, tsa, solver, trees, rf, gp, kernel_methods, mixture, hdbscan, gbdt,
  training, resample, mamba, arima, embedding, ivf, transformer) are source reference builds
  for the verifier and do not ship. 0.8.5 carried the byte LM's alone. The
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
- **The Apple silicon backend is now stated in the third paragraph of README.md, which is the
  PyPI long description for both wheels** (`packaging/macos/build_release_wheel.sh` copies it
  into `python/`, `packaging/linux/pack_wheel.py` reads it from the repository root), and the
  PyPI summary line in `python/pyproject.toml` names Apple silicon, NVIDIA and AMD by vendor.
  It was buried two thirds of the way down the file. The paragraph is a capability claim, that
  one Mojo source builds for Metal, CUDA and HIP so the tree and classical estimators fit on an
  M-series GPU, and it says in the same breath that the optional `fast` tier is not the
  bitwise-identical default and promises no repeatability at all, so a reader cannot come away
  thinking the accelerated tree training is the certified thing.
- **The Apple tree speed ratio is withdrawn from README.md and ENGINEERING_RULES.md, and no
  tree speed claim replaces it.** Both files said ExtraTrees measured a range against
  scikit-learn on all ten M4 cores at covtype 581k, framed as the win that earns the `fast`
  tier its place. The range is a splice of two rows of a deleted file
  (`bench/results/WINDOW_2026-08-22_extratrees-batched.md`, removed by `e08cda5bc` on
  2026-09-04, six days before `92928a2cd` wrote the sentence), it reports speedup where
  `bench/OPPONENT_REFERENCE.md` reports its inverse so the same digits mean the opposite thing
  in two files of this repository, its own source was already superseded by a later addendum
  and by three later Apple covtype windows reading 1.04x slower, 1.13x slower and 1.12x
  faster, covtype is neither of the two datasets a training-speed claim requires and is below
  the million-row floor, and `bench/OPPONENT_REFERENCE.md`'s "Rows never to quote" section
  covers both `bench/results/fast_speed/mac-*` and our own fast and deterministic arms on any
  vendor. On the qualifying datasets the standing is the reverse of the withdrawn claim, ours
  over theirs where lower is better, extra trees 1.53x of scikit-learn on taxi and 1.63x on
  Istella-S (slower, and its fast arm returns the identical arm's hash in the same time),
  random forest 0.29x and 0.14x, symmetric trees 0.44x and 0.30x of CatBoost with no XGBoost
  oblivious grower to check against. The tier rule is unchanged; what changes is that it now
  rests on the structural argument, tree fitting calls no BLAS while the classical families
  do, and a qualifying Apple measurement is recorded as owed.
- **`KMeans` can be saved and loaded, so a k-means model fitted on a GPU predicts on a machine
  with none.** `KMeans.predict` and `KMeans.transform` already shipped and
  `mojolearn/host/_mojolearn_core_host.so` already exported both, but the class had no `save`,
  so there was no file for `mojolearn.host_model()` to open and the whole train-here,
  infer-there route stopped at serialization; `host_surface.py`'s own gap registry said so.
  `save` writes the format `mojolearn-kmeans-1` through the same deterministic npz writer
  every other portable model uses, and `mojolearn.host_model(path)` returns a `HostKMeans`
  bound to the core host binding. ONE format covers every k-means lane: the metric and the
  start are members of the file rather than tags of their own. The fit's own `labels_` travels
  with the centroids, because it is what `predict` on the training rows must equal. A file of
  another format or another estimator, a truncated one, one whose centroid count disagrees
  with its dimensionality, one whose metric name disagrees with its code member, and one whose
  arrays are at another dtype are each refused by name rather than loaded into a plausible
  wrong answer. What is still owed for `kmeans` is the GPU recording under
  `bench/results/classical_host/`, as for `dbscan`, `agglomerative` and `spectral`.

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
