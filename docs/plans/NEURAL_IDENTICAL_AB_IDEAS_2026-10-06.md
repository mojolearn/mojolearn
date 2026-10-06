# Neural-only IDENTICAL A/B ideas — 2026-10-06

64 cards drafted before implementation fan-out. These are hypotheses inferred from local source, not measured bottleneck or speedup claims. Source baseline `fd6cf80453a6f18eb02e81566c824e7da106ccf0`; branch `ideas/neural-identical-ab-20261006-r3`.

## Contract

- **Scope:** NEURAL ONLY: neural GEMM, attention/transformers, language models, Mamba/Samba, RNN/LSTM/GRU, MLP/MoE, CNN/ResNet, neural GCN/GraphSAGE, embeddings, neural losses/optimizers and their native runtime. Excludes classical ML, classical forecasts, ANN/vector search, trees, clustering and Apple FAST tuning. Shared primitive changes must remain opt-in for named neural callers.
- **Identity:** A and B may have different bits. Within each arm/version, NVIDIA, AMD, Apple and host must match bitwise wherever promised: outputs, gradients, fitted/state/checkpoint buffers, selection/stopping and errors. Changed arithmetic requires a coherent same-version host/device/forward/backward/decode contract, not an old-version equality gate.
- **Quality:** No degraded quality is accepted. Use the repository's existing model-specific acceptance rules and full tasks: LM held-out loss/perplexity, classification/regression metrics, neural forecasting task loss, gradients/finite state and multi-step training stability. Numerical errors and identity alone do not establish task quality. Do not invent relaxed thresholds.
- **Scope of this request:** Generate ideas, then program opt-in source in this isolated worktree. Do NOT compile, run tests/checkers/static verification, execute candidate ML code, benchmark, rent machines, or update measured boards. All new switches default OFF. Source drafting is not qualification.
- **Future measurement:** Freeze one commit; reuse matching accepted compilation/identity evidence. Full-dataset/full-workload end-to-end A/B for every affected model on NVIDIA and AMD, one excluded warmup and one scored sample, vendors in parallel and cells serial per GPU. Include preparation, required synchronization and consumed outputs; distinguish fit/train, inference/prefill/decode, cold/repeated and complete default combinations. Component timings never qualify a default.
- **Promotion:** Joint NVIDIA+AMD improvement with neither materially slower, required four-column identity and model quality preserved. Apple is an identity witness, never a timing vote for IDENTICAL. Preserve loser/neutral/failure/pending evidence and use board tools when actual measurements exist. No new default or merge to main from these uncompiled drafts.
- **Routing:** No benchmark dimensions/names/seeds or near-board thresholds. Derive geometry/route/caps from work, bytes, supported hardware and numeric contract; explain in source. Cover adjacent shapes and a non-board dataset later. Removing a targeted rule uses old rule as B.
- **Runtime:** All runtime data work is Mojo, GPU work stays on device for GPU routes; Python is API shell and experiment metadata/orchestration only. No Python arithmetic, loops/sorting/label work or worker threads in runtime.
- **Unsupported:** Use supported Mojo/Modular features only. Record exact upstream asks if unavailable; no compiler-output rewrites, toolchain patches, unsupported modes or invented vendor instructions.
- **Resources:** Future CPU arms use the full actual machine allocation, Linux cgroup allocation rather than host count, serial measurement arms and documented nested pools. Record effective worker pools; do not alter another active freeze.

## Future full-workload map

Each card names affected neural caller families and source anchors. Before any later timing, map every transitive affected public caller to its saved full recipe, dataset/corpus version/hash/split, actual dimensions, dtype/mode, full model/optimizer/seed settings, A/B flags and timed boundary. Audit all internal caps. Missing mappings remain pending; do not substitute a tiny driver.

- **primary_neural:** tools/bench_board_neural.py: LANES, MODEL_OF, LM_SHAPES/GEMM_SHAPES/BLOCK_SHAPES/SAMBA_SHAPES/MLP_SHAPES, make_inputs, build_runner, quality; retain exact corpus/data hashes and complete model config.
- **neural_layers_and_estimators:** tools/bench_board_algos.py: ONLY neural sequence, CNN, embedding, loss and optimizer _add entries plus _build_seqmodel/_build_cnnclf/_build_layer/_build_optim. Exclude its classical time-series and classical estimator entries despite shared directories.
- **board_orchestration:** tools/bench_board.py: plan_races and neural-family call sites; recipes and intrinsic caps must be read before future timing. --rows full or --neural-shape full is not evidence of actual full input.
- **existing_candidates:** experiments/performance_ideas/I01-I11, A01-A05 and N01-N06 manifests/coverage/native_arms; component drivers are starting references, not full-workload qualification.

## Priorities and arms

A = candidate; B = frozen incumbent. Hold unrelated switches fixed; an all-off build is not the normal control. P0 = low arithmetic risk and broad memory/launch opportunity; P1 = caller integration or explicit numerical revision; P2 = high design cost or capability uncertainty. No priority claims a measured speedup. Within an arm compare all vendors with that arm's host. Between arms compare quality and complete timing, not old-versus-new bit equality.

Each compound card names independently attributable sub-arms. A profile candidate needs a written arithmetic contract and actual coherent implementation; a flag or manifest is not an implementation. Reuse existing candidates/evidence, and retain existing loser controls rather than restarting them unchanged.

## Inventory

| Card | Lane | Type | Priority | Idea |
| --- | --- | --- | --- | --- |
| [NN01](#nn01) | gemm | schedule | P0 | Attribute GEMM schedules at real neural callers |
| [NN02](#nn02) | gemm | schedule | P0 | Stream GEMM partial planes into a bounded fold |
| [NN03](#nn03) | gemm | profile | P2 | Versioned GEMM leaf lengths |
| [NN04](#nn04) | gemm | profile | P2 | Versioned independent accumulator chains inside GEMM leaves |
| [NN05](#nn05) | gemm | schedule | P0 | Group independent projection jobs |
| [NN06](#nn06) | gemm | schedule | P0 | Rounded neural GEMM epilogues |
| [NN07](#nn07) | gemm | schedule | P1 | Reuse transposed operand staging across neural jobs |
| [NN08](#nn08) | gemm | schedule | P2 | Bounded asynchronous operand loading |
| [NN09](#nn09) | gemm | schedule | P1 | Shared-memory page count and bank layout |
| [NN10](#nn10) | gemm | schedule | P0 | Neural GEMM cost-based dispatch |
| [NN11](#nn11) | gemm | schedule | P1 | Specialize fold storage to the logical tree |
| [NN12](#nn12) | gemm | schedule | P0 | Model-owned GEMM plan and workspace reuse |
| [NN13](#nn13) | gemm | schedule | P0 | Tile sequence-family same-chain GEMM |
| [NN14](#nn14) | gemm | schedule | P1 | Group convolution-lowered GEMMs with bounded im2col |
| [NN15](#nn15) | gemm | schedule | P2 | Hardware-specific schedules under one profile |
| [NN16](#nn16) | gemm | schedule | P0 | Avoid unnecessary zero-tail and scratch passes |
| [NN17](#nn17) | attention | schedule | P0 | Share attention K/V tiles across query heads |
| [NN18](#nn18) | attention | schedule | P0 | Skip structurally masked attention tiles |
| [NN19](#nn19) | attention | schedule | P0 | Retain versus recompute attention intermediates |
| [NN20](#nn20) | attention | profile | P2 | Versioned streaming stable attention softmax |
| [NN21](#nn21) | attention | schedule | P0 | Fuse attention pointwise score transforms |
| [NN22](#nn22) | attention | schedule | P1 | Canonical dK/dV task geometry |
| [NN23](#nn23) | attention | schedule | P1 | Fuse attention backward row dot and pointwise gradients |
| [NN24](#nn24) | attention | profile | P1 | Versioned RMSNorm/LayerNorm fixed-lane reductions |
| [NN25](#nn25) | attention | schedule | P0 | Separate norm scalar fold from parallel cell scaling |
| [NN26](#nn26) | attention | schedule | P0 | Training SwiGLU forward/backward pass fusion |
| [NN27](#nn27) | attention | schedule | P0 | Session-owned RoPE frequency and position state |
| [NN28](#nn28) | attention | schedule | P0 | Remove training-only dead KV-cache writes |
| [NN29](#nn29) | attention | schedule | P1 | Bounded decode KV layout and append fusion |
| [NN30](#nn30) | attention | schedule | P0 | Backward residual and gradient buffer views |
| [NN31](#nn31) | attention | schedule | P2 | Bounded model activation checkpoint policy |
| [NN32](#nn32) | attention | schedule | P0 | Retain Samba attention forward state for backward |
| [NN33](#nn33) | state_cnn | schedule | P0 | Attribute SSD/SISO shared tile reuse |
| [NN34](#nn34) | state_cnn | profile | P2 | Versioned absolute-chunk selective scan |
| [NN35](#nn35) | state_cnn | schedule | P0 | Parallel causal depthwise-convolution cells |
| [NN36](#nn36) | state_cnn | schedule | P0 | Cache Mamba decay exponent values |
| [NN37](#nn37) | state_cnn | schedule | P1 | Prune unused triangular SSD work |
| [NN38](#nn38) | state_cnn | profile | P1 | Versioned shared Mamba-3 angle-gradient suffixes |
| [NN39](#nn39) | state_cnn | profile | P1 | Versioned Mamba parameter-gradient folds |
| [NN40](#nn40) | state_cnn | schedule | P0 | Mamba immutable weight generations and workspace |
| [NN41](#nn41) | state_cnn | schedule | P1 | One-launch ordered recurrent inference/training segments |
| [NN42](#nn42) | state_cnn | schedule | P0 | Fuse recurrent gate pointwise updates |
| [NN43](#nn43) | state_cnn | profile | P1 | Versioned recurrent weight and bias gradients |
| [NN44](#nn44) | state_cnn | schedule | P0 | MoE stable token grouping and grouped expert jobs |
| [NN45](#nn45) | state_cnn | schedule | P0 | Fuse CNN activation/bias/residual passes |
| [NN46](#nn46) | state_cnn | schedule+profile | P1 | Deterministic CNN gradient and pooling schedules |
| [NN47](#nn47) | state_cnn | schedule+profile | P1 | Neural BatchNorm statistics and running-state fusion |
| [NN48](#nn48) | state_cnn | schedule | P1 | Neural graph aggregation tile reuse |
| [NN49](#nn49) | training | schedule | P0 | Stable embedding grouping with touched-row gradients |
| [NN50](#nn50) | training | schedule | P0 | Reuse immutable token grouping across backward uses |
| [NN51](#nn51) | training | schedule | P0 | Resident token and target validation |
| [NN52](#nn52) | training | schedule | P0 | Cross-entropy elementwise pass fusion |
| [NN53](#nn53) | training | schedule+profile | P1 | Stream LM head and exact loss without full logits |
| [NN54](#nn54) | training | profile | P1 | Versioned loss reduction across tokens and microbatches |
| [NN55](#nn55) | training | schedule | P0 | Fuse optimizer post-update status production |
| [NN56](#nn56) | training | schedule | P0 | Batch optimizer parameter groups with per-group scalars |
| [NN57](#nn57) | training | profile | P1 | Versioned global gradient-norm reduction |
| [NN58](#nn58) | training | schedule | P0 | Fuse accumulation with gradient finishing |
| [NN59](#nn59) | training | schedule | P1 | Fuse neural dropout RNG and pointwise consumers |
| [NN60](#nn60) | training | schedule | P0 | Parameter/gradient views with generation-safe ownership |
| [NN61](#nn61) | training | schedule | P0 | One final neural step status/readback boundary |
| [NN62](#nn62) | training | schedule | P1 | Live-range neural scratch and activation arenas |
| [NN63](#nn63) | training | schedule | P2 | Deterministic neural multi-device shard merge |
| [NN64](#nn64) | training | schedule | P1 | Neural inference state batching with isolated sessions |

<a id="nn01"></a>
## NN01 — Attribute GEMM schedules at real neural callers

**Affected callers:** LM; transformer; Mamba/Samba; MLP; CNN.

**Source anchors:** `gemm/checks/gemm_identical.mojo`; `training/dev_tensors.mojo`.

**A:** Select one existing stepped MFMA/kpack/body-tile schedule at a time and expose its route for full caller A/B.

**B:** Frozen incumbent GEMM dispatch with every unrelated switch unchanged.

**Same-version identity:** Logical products, leaf boundaries, FMA/FTZ and fold stay fixed; physical tile choice may differ by vendor.

**Quality:** Full forward, gradients, optimizer state and model loss/quality across shapes and tails.

**Reject/park:** Do not repeat decided arms unchanged or claim kernel timing establishes train-step benefit.

**Prior work:** Reuse I01/A01/N01 and retained outcomes; caller coverage and attribution first.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn02"></a>
## NN02 — Stream GEMM partial planes into a bounded fold

**Affected callers:** Neural GEMM; forward/backward linear layers.

**Source anchors:** `gemm/checks/gemm_identical.mojo`; `gemm/experiments/bounded_workspace.mojo`.

**A:** Use bounded groups of canonical leaf partials with an explicit streaming adjacent-pair stack.

**B:** Materialize the incumbent full partial plane before its fold.

**Same-version identity:** The exact logical tree including odd carries and +0 initialization is unchanged by streaming.

**Quality:** Contraction words and all callers' full forward/backward outputs and task metrics.

**Reject/park:** Reject spills, extra launches, in-flight aliasing or a stack that depends on physical grid size.

**Prior work:** Extends I02/N02; distinguish live storage from a revised arithmetic tree.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn03"></a>
## NN03 — Versioned GEMM leaf lengths

**Affected callers:** FP32 neural GEMM consumers.

**Source anchors:** `gemm/contract.mojo`; `gemm/host/gemm_oracle.mojo`; `gemm/checks/gemm_identical.mojo`.

**A:** Separate canonical 64/128/256-term leaf experiments with the same specified within-leaf FMA and pair tree.

**B:** Incumbent contraction profile for the same inputs and public model.

**Same-version identity:** A new version may change bits; each arm has a complete common host/device profile, portable FMA and fixed tails. Leaf selection cannot depend on vendor.

**Quality:** Conditioned error, neural loss/perplexity, forward and gradient quality, finite state and full training trajectories.

**Reject/park:** Reject no credible throughput mechanism, precision reduction or incomplete migration of a caller/oracle.

**Prior work:** I04 has only prerequisite scalar probes; do not call them a device implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn04"></a>
## NN04 — Versioned independent accumulator chains inside GEMM leaves

**Affected callers:** FP32 neural projections and weight gradients.

**Source anchors:** `gemm/contract.mojo`; `gemm/host/identical_gemm.mojo`; `gemm/checks/gemm_identical.mojo`.

**A:** Use a specified small number of interleaved FMA chains per logical leaf, then combine them in a fixed tree.

**B:** One dependent ascending FMA chain per incumbent leaf.

**Same-version identity:** Lane membership, initial zero, FMA and final merge are numerical profile data shared across vendors, independent of warp width.

**Quality:** Cancellation/adversarial errors, forward/backward/optimizer state and task quality; all dtype contracts remain unchanged.

**Reject/park:** Reject extra registers or quality loss; no native matrix instruction with unknown internal order may stand in for the declared chain.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn05"></a>
## NN05 — Group independent projection jobs

**Affected callers:** Q/K/V; gate/up; independent dX/dW products; Samba.

**Source anchors:** `gemm/experiments/grouped_jobs.mojo`; `training/dev_tensors.mojo`; `transformer/impl/llama/modeling_llama.mojo`.

**A:** Launch separate logical products as grid jobs, sharing read-only operand staging when compatible; attribute grouping and sharing independently.

**B:** Independent GEMM launches with the same operands.

**Same-version identity:** Each contraction's k, leaves, rounded products and output tree remain unchanged; concatenating backward contractions is excluded.

**Quality:** Every projection and gradient, complete layer/step quality and memory lifetime.

**Reject/park:** Reject descriptor overhead or extra scratch/serialization at small batches.

**Prior work:** Reuse I03; new caller combinations need their own reach mapping.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn06"></a>
## NN06 — Rounded neural GEMM epilogues

**Affected callers:** MLP; transformer projections; CNN dense layers.

**Source anchors:** `gemm/experiments/rounded_epilogue.mojo`; `transformer/impl/llama/modeling_llama.mojo`; `training/mlp_ops.mojo`.

**A:** Independently fuse bias, residual, scaling or activation output passes into the producer with explicit stored-value rounding seams.

**B:** GEMM writes followed by separate pointwise kernels.

**Same-version identity:** FMA formation cannot cross an incumbent materialization boundary; preserve the exact portable activation and FTZ order.

**Quality:** Intermediate/final words, activation derivatives and full step/model quality.

**Reject/park:** Reject register pressure slowing GEMM or fusion of a nonlinear operation without its backward contract.

**Prior work:** Extend I05 separately for each epilogue/caller.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn07"></a>
## NN07 — Reuse transposed operand staging across neural jobs

**Affected callers:** Weight gradients; repeated linear layers; attention projections.

**Source anchors:** `gemm/checks/gemm_identical.mojo`; `training/dev_tensors.mojo`.

**A:** Stage immutable strided/transposed operand tiles once for several independent consumers, keeping distinct outputs.

**B:** Each consumer repeats the same gather/staging.

**Same-version identity:** No change to each output's product order or required strided-load rounding; cached weights invalidate by owner/version.

**Quality:** All transpose modes, ragged strides, repeated calls, gradients and model quality.

**Reject/park:** Reject materialization more expensive than strided access or stale transpose caches.

**Prior work:** N04 supplies transpose controls; extend only new sharing opportunities.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn08"></a>
## NN08 — Bounded asynchronous operand loading

**Affected callers:** Neural GEMM forward/backward.

**Source anchors:** `gemm/experiments/async_operand_pipeline.mojo`; `gemm/experiments/async_api_probe.mojo`.

**A:** Overlap the next canonical operand tile load with current arithmetic using supported Mojo pipeline primitives.

**B:** Synchronous loading of the same tiles.

**Same-version identity:** Arithmetic order identical; barriers and page ownership explicit, no use-before-ready.

**Quality:** Every GEMM/gradient output, tail masks and complete neural operation quality.

**Reject/park:** If a required primitive is unsupported, record a Modular ask; no compiler patch, assembly rewrite or unsupported build mode.

**Prior work:** Reuse N03 capability evidence where compiler/target match.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn09"></a>
## NN09 — Shared-memory page count and bank layout

**Affected callers:** AMD/NVIDIA neural GEMM.

**Source anchors:** `gemm/experiments/bounded_staging.mojo`; `gemm/checks/gemm_identical.mojo`.

**A:** Separate one-page/two-page/padded or swizzled staging arms derived from shared bytes and supported bank access.

**B:** Incumbent page count and operand layout.

**Same-version identity:** Address permutation only, preserving every logical operand and arithmetic order; document invertible layouts and tails.

**Quality:** Contraction and model outputs plus resource footprint on both voting vendors.

**Reject/park:** Reject additional address arithmetic, register pressure or a layout valid only for a board dimension.

**Prior work:** A02/A03 already cover baseline variants; add only clearly distinct arms.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn10"></a>
## NN10 — Neural GEMM cost-based dispatch

**Affected callers:** All affected neural GEMM callers.

**Source anchors:** `gemm/checks/gemm_identical.mojo`; `gemm/contract.mojo`.

**A:** Replace any identified shape-fitted rule with a byte/work/device-fill rule valid for neighboring shapes; each removal is its own arm.

**B:** Explicitly retain the old route rule as the designated B arm.

**Same-version identity:** The numerical contract is independent of route; if removing a rule changes its profile, all columns move together.

**Quality:** Full affected models, adjacent shapes and at least one non-board dataset, including every fallback.

**Reject/park:** Do not invent hardware properties if Mojo cannot report them; park that rule and request support.

**Prior work:** Roadmap A8 and AGENTS.md no-dimension-targeting requirements.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn11"></a>
## NN11 — Specialize fold storage to the logical tree

**Affected callers:** Neural GEMM folds.

**Source anchors:** `gemm/checks/gemm_identical.mojo`; `gemm/experiments/bounded_workspace.mojo`.

**A:** Allocate only the mathematically required tree levels/active slots for each supported profile; separate register and shared stack arms.

**B:** Maximum-capacity fold storage at every contraction.

**Same-version identity:** Capacity follows the logical reduction depth, never a dataset or arbitrary nearby dimension.

**Quality:** Tiny/huge contractions and ragged partial trees, full gradients and quality.

**Reject/park:** Reject excessive specialization count or stack overflow hidden by fallback.

**Prior work:** N02 has prior stack-capacity work; reuse and extend actual uncovered callers.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn12"></a>
## NN12 — Model-owned GEMM plan and workspace reuse

**Affected callers:** LM; transformer; Samba; MLP; CNN.

**Source anchors:** `training/dev_tensors.mojo`; `training/byte_lm_model_pool.mojo`; `gemm/checks/gemm_identical.mojo`.

**A:** Retain immutable plans and bounded scratch by device/context, profile, operation and strides, with explicit generation invalidation.

**B:** Re-plan and provision temporaries per product/call.

**Same-version identity:** Scheduling/lifetime only; prevent in-flight reuse, cross-session sharing and stale flags/strides.

**Quality:** Cold/repeated calls, growing/shrinking workloads, destruction/error paths, model output and gradients.

**Reject/park:** Reject unbounded high-water retention or excluding first-use allocation from cold timing.

**Prior work:** Extend I02 and existing workspaces; no global mutable cache.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn13"></a>
## NN13 — Tile sequence-family same-chain GEMM

**Affected callers:** RNN/LSTM/GRU; sequence MLP; MoE.

**Source anchors:** `sequence/gemm_tiled.mojo`; `sequence/ops.mojo`; `sequence/exec_device.mojo`.

**A:** Tile strided operands while preserving each output's original scalar contraction chain; batch independent cells.

**B:** Per-output serial dot with repeated operand loads.

**Same-version identity:** The sequence family's numerical profile is authoritative; do not silently substitute the main GEMM profile.

**Quality:** All forward/backward recurrent state and optimizer words, strided tails and training quality.

**Reject/park:** Reject shared-memory costs or changing the fold while claiming schedule-only.

**Prior work:** Roadmap D1 and existing tiled executor are source anchors.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn14"></a>
## NN14 — Group convolution-lowered GEMMs with bounded im2col

**Affected callers:** Conv1d/2d; CNNClassifier; ResNet blocks.

**Source anchors:** `x_cnn/device.mojo`; `x_cnn/ops.mojo`; `gemm/experiments/grouped_jobs.mojo`.

**A:** Build bounded patches for independent spatial tiles and consume them immediately in grouped exact products.

**B:** Materialize/reload full im2col or launch each patch group separately.

**Same-version identity:** Channel/kernel/tap product order, padding, dilation and grouping semantics fixed.

**Quality:** Forward/dInput/dWeight/dBias, receptive-field boundaries and complete CNN quality.

**Reject/park:** Reject repeated patch generation or smaller measured image batches masquerading as an improvement.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn15"></a>
## NN15 — Hardware-specific schedules under one profile

**Affected callers:** AMD/NVIDIA neural GEMM.

**Source anchors:** `gemm/checks/gemm_identical.mojo`; `gemm/experiments/subwave_membership.mojo`.

**A:** Map independent logical groups to physical waves/warps and use already supported scalar-equivalent instruction bodies.

**B:** Incumbent physical mapping of the same logical contractions.

**Same-version identity:** Neither physical wave width nor instruction grouping defines FP summation order; operand membership must be explicit.

**Quality:** Four-column words, full neural quality and vendor-specific whole-workload timings.

**Reject/park:** Reject undocumented native instruction arithmetic, subgroup cross-talk or a material slowdown on either voting vendor.

**Prior work:** Reuse A04/A05 existing probes; not an AMD portable-toolchain workaround.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn16"></a>
## NN16 — Avoid unnecessary zero-tail and scratch passes

**Affected callers:** Neural GEMM and tensor preparation.

**Source anchors:** `gemm/checks/gemm_identical.mojo`; `core/device_zero.mojo`; `training/dev_tensors.mojo`.

**A:** Have a producer fully initialize its owned scratch/output region, fusing required tail zeroing with useful stores.

**B:** Clear full buffers separately before producers overwrite them.

**Same-version identity:** Only prove-away cells never read before write; explicitly handle signed zero, nonfinite inputs, padding and reuse.

**Quality:** Poisoned workspace, alternating shapes, all recorded stages and downstream models.

**Reject/park:** Reject undefined cells or suppressing arithmetic on zeros where NaN/Inf semantics require it.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn17"></a>
## NN17 — Share attention K/V tiles across query heads

**Affected callers:** GQA/MQA/MHA; transformer; LM; Samba attention.

**Source anchors:** `transformer/impl/llama/fused_attention.mojo`; `transformer/impl/llama/attention_v2.mojo`.

**A:** Extend supported compatible-head/query-tile sharing with bounded query accumulators, one geometry per arm.

**B:** Each query-head group reloads the same K/V tiles.

**Same-version identity:** Canonical score, max, denominator, output and gradient order unchanged for each query.

**Quality:** Causal/full masks, GQA ratios, sequence tails, gradients and full model quality.

**Reject/park:** Reject shared memory/extra barriers erasing reuse or assuming a head ratio exists on all inputs.

**Prior work:** I06 already has two-head GQA reuse; extend new schedules, preserve retained rejection evidence.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn18"></a>
## NN18 — Skip structurally masked attention tiles

**Affected callers:** Causal attention prefill and training.

**Source anchors:** `transformer/impl/llama/fused_attention.mojo`; `transformer/impl/llama/attention_v2.mojo`.

**A:** Omit fully masked tile tasks and bound partial-tile work while preserving public/trace outputs and backward masks.

**B:** Schedule full rectangular score tiles then mask.

**Same-version identity:** Masked values' logical contribution and initialization remain explicit; all-masked row behavior unchanged.

**Quality:** Prefix/causal boundaries, all-masked rows, dQ/dK/dV, trace and checkpoint semantics.

**Reject/park:** Reject omission of a recorded/backward-consumed value or a changed -Inf/NaN convention.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn19"></a>
## NN19 — Retain versus recompute attention intermediates

**Affected callers:** Transformer/LM/Samba training.

**Source anchors:** `transformer/impl/llama/fused_attention.mojo`; `transformer/checks/transformer_backward.mojo`.

**A:** Independently retain probabilities, exponent state or only canonical row summaries under an explicit storage/recompute budget.

**B:** Incumbent stored/recomputed state choice.

**Same-version identity:** Recomputation repeats identical operations and RNG coordinates; owner/generation prevents backward using another forward's state.

**Quality:** Complete step, gradients, checkpoint replay and usable batch capacity, repeated/mixed lengths.

**Reject/park:** Reject a backward-only win that slows forward+backward or reduces available batch size.

**Prior work:** Reuse I07 cost-model and checked lifetime APIs.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn20"></a>
## NN20 — Versioned streaming stable attention softmax

**Affected callers:** Attention forward/backward; full transformer and LM.

**Source anchors:** `transformer/impl/llama/fused_attention.mojo`; `transformer/host/transformer_block_host.mojo`; `transformer/checks/transformer_backward_oracle.mojo`.

**A:** Combine fixed key tiles as canonical (max,scaled_sum,weighted_value) summaries with a prescribed merge tree.

**B:** Incumbent multi-pass canonical max/exponent/denominator/value evaluation.

**Same-version identity:** New arithmetic version on every column, with exact rescaling, portable exp/div, masked/empty tiles, tails and backward/checkpoint definitions.

**Quality:** Attention error, gradient quality, full training loss/perplexity and multi-step state; extreme logits.

**Reject/park:** Reject quality loss or unimplemented host/backward migration; hardware-native FlashAttention is not an identity proof.

**Prior work:** A separate attention_v2 implementation already exists with retained nonpromotion evidence; it does not establish this new balanced-summary-tree proposal. Preserve that control and its limitations.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn21"></a>
## NN21 — Fuse attention pointwise score transforms

**Affected callers:** Scaled/masked/biased attention.

**Source anchors:** `transformer/impl/llama/fused_attention.mojo`; `transformer/impl/llama/modeling_llama.mojo`.

**A:** Apply prescribed scale, mask and supported bias at score production before its existing reduction.

**B:** Write scores then run separate transforms.

**Same-version identity:** Preserve operation order, intermediate rounding, causal positions and special mask values.

**Quality:** Every transformed score/softmax output, gradients and model quality, including partial tiles.

**Reject/park:** Reject new contraction across a rounding seam or dropping requested attention outputs.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn22"></a>
## NN22 — Canonical dK/dV task geometry

**Affected callers:** Attention backward; GQA/MQA.

**Source anchors:** `transformer/impl/llama/fused_attention.mojo`; `transformer/checks/transformer_backward.mojo`.

**A:** Share useful dK/dV input tiles among independent key/feature tasks while merging the same query partials in order.

**B:** Incumbent gradient grid and separate input loads.

**Same-version identity:** No floating atomics; each gradient's query/head contribution order is fixed regardless of scheduler.

**Quality:** All dQ/dK/dV and upstream parameter gradients, long sequences, GQA and accumulation tails.

**Reject/park:** Reject revival of cooperative/stacked variants with recorded losses unless the changed mechanism is explicit.

**Prior work:** I06/N06 retain earlier backward rejection evidence; new geometry must be distinct.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn23"></a>
## NN23 — Fuse attention backward row dot and pointwise gradients

**Affected callers:** Attention backward.

**Source anchors:** `transformer/impl/llama/fused_attention.mojo`; `transformer/checks/transformer_backward.mojo`.

**A:** Reuse probability/dOutput tiles while producing fixed-order row-dot leaves and their gradient consumers.

**B:** Separate full-array row-dot and pointwise gradient passes.

**Same-version identity:** The row-dot's reduction contract and dependencies complete before the corresponding consumer; dropout mask unchanged.

**Quality:** All softmax/attention gradients, finite behavior and full step quality.

**Reject/park:** Reject replicated reduction work or hidden global synchronization assumptions.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn24"></a>
## NN24 — Versioned RMSNorm/LayerNorm fixed-lane reductions

**Affected callers:** Transformer; Mamba/Samba; MLP; sequence LayerNorm.

**Source anchors:** `transformer/impl/llama/modeling_llama.mojo`; `transformer/checks/transformer_backward.mojo`; `sequence/layernorm.mojo`.

**A:** Use a fixed logical lane count and pair tree for norm/mean/variance and backward dots, independent of hardware width.

**B:** Incumbent sequential or existing pinned row fold.

**Same-version identity:** New common version: epsilon placement, centered variance definition, FTZ/sqrt/div and derivative folds explicit; host adopts same graph.

**Quality:** Forward/gradient errors, train loss/perplexity, constant/extreme/canceling rows and narrow/tail widths.

**Reject/park:** Reject raw-moment variance cancellation, changed epsilon or hardware-shaped subgroup reductions.

**Prior work:** Roadmap C4/B11; schedule-only row scaling is a separate arm.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn25"></a>
## NN25 — Separate norm scalar fold from parallel cell scaling

**Affected callers:** RMSNorm/LayerNorm forward/backward.

**Source anchors:** `transformer/impl/llama/modeling_llama.mojo`; `transformer/checks/transformer_backward.mojo`; `sequence/layernorm.mojo`.

**A:** Retain the incumbent row-scalar fold and parallelize independent normalization/affine/residual cells.

**B:** One row thread performs both reduction and all output cells.

**Same-version identity:** Scalar result and each cell's arithmetic exactly retained; no newly fused FMA across seams.

**Quality:** Norm outputs, residual state, parameter/input gradients and training quality.

**Reject/park:** Reject launch overhead at narrow rows or incorrect broadcast lifetime.

**Prior work:** Distinct schedule counterpart to NN24.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn26"></a>
## NN26 — Training SwiGLU forward/backward pass fusion

**Affected callers:** Transformer FFN; LM; Samba.

**Source anchors:** `transformer/impl/llama/modeling_llama.mojo`; `transformer/checks/transformer_backward.mojo`.

**A:** Compute SiLU and gate product together, optionally retain derivative-needed state; fuse paired backward pointwise outputs.

**B:** Separate activation/product/derivative passes.

**Same-version identity:** Portable sigmoid/exp and original derivative rounding sequence maintained; distinct saved versus recomputed arms.

**Quality:** All activation/gradient words, extremes and full step loss.

**Reject/park:** Reject saved-state memory exceeding recomputation savings or missing training outputs.

**Prior work:** Existing forward fusion does not establish training fusion.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn27"></a>
## NN27 — Session-owned RoPE frequency and position state

**Affected callers:** Transformer prefill/decode; LM; Samba attention.

**Source anchors:** `transformer/impl/llama/modeling_llama.mojo`; `transformer/host/transformer_block_host.mojo`.

**A:** Retain validated immutable frequency data and share portable sin/cos position tiles across independent layers/heads.

**B:** Repeated frequency scan/position transform work each layer/call.

**Same-version identity:** Exact position, base/scaling, portable trig and rotation pair order; cache keys include owner/config/generation.

**Quality:** Prefill/decode agreement, long positions, reset/resume and gradients.

**Reject/park:** Reject stale config/positions, approximate trig or cross-session reuse.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn28"></a>
## NN28 — Remove training-only dead KV-cache writes

**Affected callers:** Transformer/LM/Samba training prefill.

**Source anchors:** `transformer/impl/llama/modeling_llama.mojo`; `training/byte_lm.mojo`.

**A:** Explicit training caller capability omits KV-cache append/copy operations whose buffers have no training consumer.

**B:** Populate decode-style KV state during every training prefill.

**Same-version identity:** Only dead storage disappears; recorded stages, backward and optional requested cache outputs still execute.

**Quality:** Training gradients/loss and decode callers' untouched contract; alternate training and inference on supported owners.

**Reject/park:** Reject inference cache breakage or proving deadness only from one fixture.

**Prior work:** Roadmap C8; inspect existing switches before extending.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn29"></a>
## NN29 — Bounded decode KV layout and append fusion

**Affected callers:** Autoregressive transformer/LM/Samba inference.

**Source anchors:** `transformer/impl/llama/modeling_llama.mojo`; `transformer/impl/llama/fused_attention.mojo`.

**A:** Fuse new K/V writes with canonical cache layout conversion and batch independent heads/requests under explicit capacity.

**B:** Separate append/copy/layout kernels.

**Same-version identity:** No cache quantization, token reordering or changed attention fold; preserve prefix lengths, capacity/refusal and reset semantics.

**Quality:** Every decode-step logits and cache words, prefill/decode agreement and long context quality.

**Reject/park:** Reject unsupported paging primitives or moving cache allocation outside cold latency.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn30"></a>
## NN30 — Backward residual and gradient buffer views

**Affected callers:** Transformer/LM training.

**Source anchors:** `transformer/checks/transformer_backward.mojo`; `training/byte_lm.mojo`.

**A:** Alias explicitly immutable incoming residual-gradient views or transfer ownership rather than copy whole tensors.

**B:** Copy input gradients into identical temporary buffers at every block boundary.

**Same-version identity:** Ownership and lifetime part of API; preserve caller buffers, accumulation semantics and asynchronous use.

**Quality:** Repeated backward, accumulation, shared tensors, checkpoint replay and all model gradients.

**Reject/park:** Reject aliasing in-place consumers or assuming ownership from pointer equality.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn31"></a>
## NN31 — Bounded model activation checkpoint policy

**Affected callers:** Transformer/LM/Samba training.

**Source anchors:** `training/byte_lm_layer_pool.mojo`; `training/byte_lm_offload.mojo`; `transformer/checks/transformer_backward.mojo`.

**A:** Compare explicit layer groups of stored versus recomputed activation state from a byte/recompute model, with full cold and repeated steps.

**B:** Incumbent activation retention/replay schedule.

**Same-version identity:** Same forward/RNG/replay graph, checkpoints include normalization/position/cache metadata; no reduced sequence/batch length.

**Quality:** Full gradients, state, multi-step loss and memory-capacity behavior.

**Reject/park:** Reject offload dependence on unsupported async primitives or a recompute policy chosen by benchmark row.

**Prior work:** Related I07 state tradeoff expanded to complete model layers.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn32"></a>
## NN32 — Retain Samba attention forward state for backward

**Affected callers:** Samba training.

**Source anchors:** `training/samba_ops.mojo`; `transformer/impl/llama/modeling_llama.mojo`; `transformer/checks/transformer_backward.mojo`.

**A:** Pass an explicit owner/generation-tagged saved-forward handle into backward instead of recomputing from an empty cache.

**B:** Re-run identical attention forward during the training backward handoff.

**Same-version identity:** Handle includes inputs/weights/config versions; invalidate on mutation and preserve deliberate checkpoint mode.

**Quality:** Complete Samba gradients/loss, alternating sessions, weight updates and failure cleanup.

**Reject/park:** Reject stale state, Python runtime comparison of input arrays or unbounded retained activations.

**Prior work:** Roadmap B4; lifetime extension beyond an attention-only benchmark.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn33"></a>
## NN33 — Attribute SSD/SISO shared tile reuse

**Affected callers:** Mamba-2; Mamba-3; Samba.

**Source anchors:** `mamba/impl/modules/ssd_minimal.mojo`; `mamba/impl/ops/mamba3_siso.mojo`.

**A:** Isolate shared B/C/decay and G×L retained tiles, then test new bounded channel/task sharing one arm at a time.

**B:** Incumbent SSD/SISO staging and recomputation.

**Same-version identity:** Same canonical products and per-channel sums; recorded/backward-consumed stages retained.

**Quality:** All forward/state/backward words, prefill/decode and training quality across state sizes/tails.

**Reject/park:** Reject extra memory traffic or recreating already recorded tile regressions.

**Prior work:** Reuse I08 tile and retained-product arms; no duplicate implementation claims.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn34"></a>
## NN34 — Versioned absolute-chunk selective scan

**Affected callers:** Mamba-1 selective scan; Samba.

**Source anchors:** `mamba/impl/ops/selective_scan_interface.mojo`; `mamba/impl/ops/selective_scan_backward.mojo`; `mamba/host/gen`.

**A:** Compose affine recurrence summaries at fixed absolute-position chunk boundaries with a specified carry tree and replay.

**B:** Incumbent sequential recurrence evaluation.

**Same-version identity:** New full profile covers host, GPU, backward, checkpoint and decode; retain enough prefix state so chunk boundaries do not depend on total length.

**Quality:** Loss/perplexity, gradients, long-context stability, prefix/decode agreement and resumed-state words.

**Reject/park:** Reject incomplete decode/backward contract, associative-real arithmetic claims used as bit proofs or FAST length-dependent chunks.

**Prior work:** I09 leaves this profile unimplemented pending a complete contract.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn35"></a>
## NN35 — Parallel causal depthwise-convolution cells

**Affected callers:** Mamba-1/2; Samba; neural Conv1d.

**Source anchors:** `mamba/impl/modeling/modeling_mamba.mojo`; `mamba/impl/modules/mamba2.mojo`.

**A:** Map independent batch/time/channel cells to parallel tasks and share overlapping input windows, with each tap chain fixed.

**B:** Serial time loop per channel or repeated window loads.

**Same-version identity:** Causal padding, convolution state, tap order and activation rounding unchanged.

**Quality:** Prefill/decode state agreement, dInput/dWeight, short sequences and tails.

**Reject/park:** Reject changing padding or crossing recurrent state updates that are not independent.

**Prior work:** I09 already supplies token-parallel conv arms; extend actual window reuse/caller reach.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn36"></a>
## NN36 — Cache Mamba decay exponent values

**Affected callers:** Mamba-2/3; Samba.

**Source anchors:** `mamba/impl/modules/ssd_minimal.mojo`; `mamba/impl/ops/mamba3_siso.mojo`.

**A:** Compute each immutable portable exp/decay value once per logical position/head and reuse across channels and later consumers.

**B:** Recompute identical exponentials inside independent channel tasks.

**Same-version identity:** Store explicitly rounded original exp/decay words; distinguish diagonal, inter-chunk and backward dependencies.

**Quality:** Complete stages and gradients, long/short sequences and extreme decays.

**Reject/park:** Reject retained buffers costing more than recomputation or reusing numerically similar but nonidentical exponent expressions.

**Prior work:** I08 already retains some decay stages; only uncovered uses are new.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn37"></a>
## NN37 — Prune unused triangular SSD work

**Affected callers:** Mamba-2 SSD; Mamba-3 SISO.

**Source anchors:** `mamba/impl/modules/ssd_minimal.mojo`; `mamba/impl/ops/mamba3_siso.mojo`.

**A:** Schedule only lower-triangular causal tile tasks while explicitly initializing any trace/backward-visible unused cells.

**B:** Compute full Q×Q intermediate matrices.

**Same-version identity:** Causal inclusion and logical arithmetic for used cells fixed; no skipping upper cells if a consumer reads them.

**Quality:** Forward/backward complete stages, chunk tails, checkpoints and prefix behavior.

**Reject/park:** Reject assumptions based only on forward output; include initialization cost.

**Prior work:** I08 lower-triangle arm exists; scope is attributable downstream reach.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn38"></a>
## NN38 — Versioned shared Mamba-3 angle-gradient suffixes

**Affected callers:** Mamba-3 backward; Samba training.

**Source anchors:** `mamba/impl/modules/mamba3_backward.mojo`; `mamba/host/gen/mamba3_backward.mojo`.

**A:** Compute canonical suffix summaries once and reuse across d_dt/angle-gradient consumers instead of repeating each suffix.

**B:** Each token folds its entire suffix independently.

**Same-version identity:** New version defines reverse leaf boundaries, carry and multiplication placement in device/host/backward oracle; old bits need not match.

**Quality:** All parameter/input gradients, multi-step loss and numerical error on long cancellation-heavy sequences.

**Reject/park:** Reject a prefix reversal that changes the intended causal derivative or incomplete host generation.

**Prior work:** Programming found IDN_M3_ANGLE_DT_SUFFIX already default-on with chunk=64 and a host counterpart. Reuse it as incumbent; new work must isolate extra suffix-seed reuse or an explicitly different profile.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn39"></a>
## NN39 — Versioned Mamba parameter-gradient folds

**Affected callers:** Mamba-1/2/3 backward; Samba training.

**Source anchors:** `mamba/impl/ops/mamba2_ssd_backward.mojo`; `mamba/impl/modules/mamba3_backward.mojo`; `mamba/host/gen`.

**A:** Produce fixed-position per-token gradient leaves then merge with a common tree; attribute each parameter family separately.

**B:** Long serial time/batch gradient chains.

**Same-version identity:** Logical partition shared by every vendor and host, with exact FMA/FTZ/odd-tail specification; accumulation API semantics remain explicit.

**Quality:** Every gradient tensor, optimizer moments/parameters and full model quality; cancellation and scale range.

**Reject/park:** Reject output-partial traffic dominating or changing only device folds.

**Prior work:** Roadmap B6 and I09 profile extension.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn40"></a>
## NN40 — Mamba immutable weight generations and workspace

**Affected callers:** Mamba/Samba prefill/decode/training.

**Source anchors:** `bindings/_mojolearn_mamba.mojo`; `mamba/impl/modules/idn_gemm_ws.mojo`; `training/samba_ops.mojo`.

**A:** Use explicit owner/generation metadata to retain validated packed weights and bounded scratch until a weight/config mutation.

**B:** Repeated full weight comparisons/copies and temporary allocations.

**Same-version identity:** No Python data comparison/runtime work; all external mutation paths invalidate or take the safe uncached route.

**Quality:** Forward/decode/backward after update/load/reset, independent sessions and exception lifetime.

**Reject/park:** Reject pointer-only cache keys or silently trusting externally mutable buffers.

**Prior work:** Roadmap B10 and existing workspace infrastructure.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn41"></a>
## NN41 — One-launch ordered recurrent inference/training segments

**Affected callers:** RNN; GRU; LSTM.

**Source anchors:** `sequence/recurrent_scan.mojo`; `sequence/recurrent.mojo`; `sequence/exec_device.mojo`.

**A:** Use supported within-task synchronization for bounded recurrent segments, batching independent sequences while keeping time steps ordered.

**B:** One host-dispatched device operation per recurrent timestep.

**Same-version identity:** Hidden/cell state and gate arithmetic exactly preserved; supported memory ordering must suffice without a compiler/runtime workaround.

**Quality:** All sequence outputs/states/gradients, ragged lengths, bidirectionality if supported, and model quality.

**Reject/park:** Park on unsupported ordering/launch capability; no cross-block spin barriers.

**Prior work:** Roadmap D2; existing FAST Apple scan is not IDENTICAL qualification.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn42"></a>
## NN42 — Fuse recurrent gate pointwise updates

**Affected callers:** LSTM; GRU; RNN; sequence MLP.

**Source anchors:** `sequence/recurrent.mojo`; `sequence/ops.mojo`; `sequence/recurrent_scan.mojo`.

**A:** Share projected gate loads for portable activation, state update and derivative-state stores in one task.

**B:** Separate gate activation/state/derivative kernels.

**Same-version identity:** Gate order, portable nonlinear functions, state dependencies and materialization rounding fixed.

**Quality:** Forward hidden/cell states, gradients and full recurrent model loss.

**Reject/park:** Reject different gate formulas or changed saved derivative semantics.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn43"></a>
## NN43 — Versioned recurrent weight and bias gradients

**Affected callers:** RNN; LSTM; GRU; sequence MLP.

**Source anchors:** `sequence/recurrent.mojo`; `sequence/ops.mojo`; `sequence/checks/oracle.mojo`.

**A:** Fixed batch/time leaves and a canonical fold replace long sequential gradient reductions; separate dWeight and dBias arms.

**B:** Incumbent ascending reduction chains.

**Same-version identity:** Host executor/oracle and device use exactly the new tree, masked-time rules and gradient accumulation ordering.

**Quality:** Gradient error, multi-step training loss, all optimizer state and padded/ragged sequences.

**Reject/park:** Reject skipping padded terms if old semantics require a signed-zero/nonfinite operation.

**Prior work:** An existing blocked weight-gradient profile is implemented. New arms change a declared leaf or schedule rather than claiming its first introduction.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn44"></a>
## NN44 — MoE stable token grouping and grouped expert jobs

**Affected callers:** Mixture-of-experts neural layers.

**Source anchors:** `sequence/moe_group.mojo`; `sequence/moe_tiled.mojo`; `sequence/moe_weights.mojo`.

**A:** Stable-group tokens by chosen expert and batch independent expert products, scattering outputs with original token order.

**B:** Per-token/expert dispatch and repeated expert-weight loads.

**Same-version identity:** Router scores, top-k ties, gate normalization, capacity and combine order unchanged; no dropped-token shortcut.

**Quality:** Router/expert outputs and gradients, expert skew, empty experts, overflow and task quality.

**Reject/park:** Reject sort/setup costs or reordered weighted expert sums.

**Prior work:** Reuse existing grouped/tiled MoE; isolate new scheduling geometry.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn45"></a>
## NN45 — Fuse CNN activation/bias/residual passes

**Affected callers:** CNNClassifier; Conv1d/2d; ResNet blocks.

**Source anchors:** `x_cnn/ops.mojo`; `x_cnn/device.mojo`; `x_cnn/host/ops_host.mojo`.

**A:** Share convolution output loads for exact bias/activation/residual operations and save required backward masks in the same pass.

**B:** Separate output passes and repeated mask creation.

**Same-version identity:** Explicit rounded seams, activation-zero derivative policy and layout retained; no precision change.

**Quality:** Forward/gradients, residual branch behavior and full CNN classification quality.

**Reject/park:** Reject register/live-state growth or non-equivalent order of residual and activation.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn46"></a>
## NN46 — Deterministic CNN gradient and pooling schedules

**Affected callers:** Conv1d/2d; max/average pooling; CNN training.

**Source anchors:** `x_cnn/ops.mojo`; `x_cnn/device.mojo`; `x_cnn/host/ops_host.mojo`.

**A:** Tile gather-style dInput and dWeight work, and fuse pool value/arg-index output; separately evaluate a versioned fixed dWeight tree.

**B:** Incumbent independent gathers/reductions and pooling passes.

**Same-version identity:** Gather schedule keeps arithmetic order; tree variant is a separate version on all columns. Max ties/NaNs and overlap multiplicities fixed.

**Quality:** All gradients and pool indices, dilation/stride/padding tails and model quality.

**Reject/park:** Reject unordered floating scatter-add or changed tie winners.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn47"></a>
## NN47 — Neural BatchNorm statistics and running-state fusion

**Affected callers:** BatchNorm1d/2d; CNN training/inference.

**Source anchors:** `x_cnn/ops.mojo`; `x_cnn/device.mojo`; `x_cnn/host/ops_host.mojo`.

**A:** Fuse independent statistic loads and normalization output; separately explore a versioned centered fixed-tree moment profile.

**B:** Separate reductions, normalization and running-statistic passes.

**Same-version identity:** Training versus inference, biased/unbiased variance uses, epsilon and momentum/running-count updates remain explicit and shared.

**Quality:** Statistics, outputs/gradients, running buffers and complete CNN loss; constant channels and small batches.

**Reject/park:** Reject cancellation from raw moments or updated running state on a failed step.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn48"></a>
## NN48 — Neural graph aggregation tile reuse

**Affected callers:** GCN; GraphSAGE neural layers.

**Source anchors:** `x_cnn/ops.mojo`; `x_cnn/device.mojo`; `x_cnn/host/ops_host.mojo`.

**A:** Share fixed CSR neighbor-feature tiles across output channels and fuse degree normalization at the specified point.

**B:** Independent channel neighbor scans and separate normalization matrices.

**Same-version identity:** Graph edges, stable neighbor order, reduction profile, self-loop and degree semantics unchanged; neural graph layers only.

**Quality:** Node outputs/gradients, isolated/skewed-degree nodes and graph-learning quality.

**Reject/park:** Reject neighbor sampling, graph sparsification or including classical PageRank in this lane.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn49"></a>
## NN49 — Stable embedding grouping with touched-row gradients

**Affected callers:** Embedding backward; LM; transformer; neural sequence.

**Source anchors:** `embedding/checks/embedding_identical.mojo`; `embedding/checks/embedding_sort.mojo`.

**A:** Use existing stable radix grouping and process touched rows; separately vary group task geometry and dense zero initialization.

**B:** Incumbent row scan/bitonic grouping path.

**Same-version identity:** Token/original-position order, padding, duplicate multiplicity and accumulation semantics fixed; no floating atomics.

**Quality:** Dense gradient words and full neural step, all-same/unique/skew IDs and repeated calls.

**Reject/park:** Reject sort cost or excluding required dense output allocation/zeroing from timing.

**Prior work:** Reuse I11 actual candidates; new experiments must identify distinct geometry or caller coverage.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn50"></a>
## NN50 — Reuse immutable token grouping across backward uses

**Affected callers:** Tied embeddings; repeated embedding backward; microbatch replay.

**Source anchors:** `embedding/checks/embedding_sort.mojo`; `training/byte_lm.mojo`.

**A:** Retain canonical sorted IDs, position map and segments for an explicitly owned unchanged token batch.

**B:** Rebuild the same grouping for each backward consumer.

**Same-version identity:** Token owner/version, padding and vocabulary are cache keys; position order not inferred from a pointer or shape.

**Quality:** All embedding/head gradients, vocabulary changes, mutated IDs, repeated backward and accumulation.

**Reject/park:** Reject unbounded caches or changing tied-weight gradient contribution order.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn51"></a>
## NN51 — Resident token and target validation

**Affected callers:** LM training; embedding; cross entropy.

**Source anchors:** `training/byte_lm.mojo`; `embedding/checks/embedding_identical.mojo`; `training/checks/loss.mojo`.

**A:** Validate immutable input IDs once in the native owner and pass explicit validated views to embedding/loss consumers; reuse bounded upload buffers.

**B:** Repeated upload/readback and index scans in each consumer.

**Same-version identity:** Validation must precede unsafe access; retain canonical first-invalid-index and refusal timing, invalidate on mutation.

**Quality:** Invalid IDs/targets, vocabulary tails, gradients, loss and repeated multi-step training.

**Reject/park:** Reject a caller-controlled trust flag used without a validation witness.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn52"></a>
## NN52 — Cross-entropy elementwise pass fusion

**Affected callers:** CrossEntropy; LM; MLP/CNN classification.

**Source anchors:** `training/checks/loss.mojo`; `training/checks/loss_contract.mojo`; `training/loss_host_rows.mojo`.

**A:** Share logits/max/exp loads for weights, target contribution and dLogits stores while preserving the denominator and reduction profile.

**B:** Separate shifted-exp, target-weight and gradient elementwise passes.

**Same-version identity:** Ignore index, label smoothing, reduction, weights and portable log/exp/div fixed; no approximate softmax.

**Quality:** Loss, dLogits, empty/ignored rows, extreme logits and full classification/LM quality.

**Reject/park:** Reject extra unrequested gradient work in inference or changed weighted denominator.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn53"></a>
## NN53 — Stream LM head and exact loss without full logits

**Affected callers:** Byte LM train step; large-vocabulary neural heads.

**Source anchors:** `training/chunked_lm_head_v2.mojo`; `training/byte_lm_pooled_head.mojo`; `training/checks/loss.mojo`.

**A:** Use bounded vocabulary panels, canonical row statistics and tiled backward, retaining or recomputing exact head pieces as separate arms.

**B:** Materialize full logits and dLogits before loss and head backward.

**Same-version identity:** The chosen profile fully specifies panel merges; if changed, all host/device/head/loss contracts move together. Requested full logits still materialize.

**Quality:** Loss/perplexity, all head/hidden/tied-embedding gradients, full optimizer state and memory capacity.

**Reject/park:** Reject sampled/adaptive softmax, fewer vocabulary items or scoring only the loss without required gradients.

**Prior work:** Extend existing chunked-head work with explicit same-version contracts.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn54"></a>
## NN54 — Versioned loss reduction across tokens and microbatches

**Affected callers:** LM; neural classification/regression losses.

**Source anchors:** `training/checks/loss_contract.mojo`; `training/checks/loss.mojo`; `training/checks/loss_oracle.mojo`.

**A:** Use a fixed logical token-leaf tree for loss numerator/weight sums, independently comparing a bounded streaming storage form.

**B:** Incumbent per-row/token reduction profile.

**Same-version identity:** Same tree in host/all GPUs, explicit normalization/ignore/smoothing/zero-weight rules. Microbatch equivalence only where promised, with a defined merge.

**Quality:** Loss error, gradients, training trajectories and all degenerate-row refusals.

**Reject/park:** Reject accepting changed bits as proof of unchanged quality or hardware-dependent leaves.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn55"></a>
## NN55 — Fuse optimizer post-update status production

**Affected callers:** SGD; Adam; AdamW; LM/Samba/MLP training.

**Source anchors:** `training/checks/optimizer.mojo`; `training/opt_gate.mojo`; `training/byte_lm.mojo`.

**A:** Emit per-block integer first-nonfinite records while update values are live, then reduce once in canonical field/index order.

**B:** Rescan updated parameters and state in separate kernels.

**Same-version identity:** Keep pre-access/pre-update checks, same failure step, field/index/message and commit/rollback behavior; only redundant output scans disappear.

**Quality:** Parameters, moments, counters, refusals and multi-step loss, including deliberately nonfinite gradients/states.

**Reject/park:** Reject partially committed failed steps or a later-step error report.

**Prior work:** Reuse I10 finish-scan design; full owner commit path is necessary.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn56"></a>
## NN56 — Batch optimizer parameter groups with per-group scalars

**Affected callers:** SGD; Adam/AdamW; other existing neural optimizers.

**Source anchors:** `training/opt_gate.mojo`; `training/checks/optimizer.mojo`; `sequence/opt_resident.mojo`.

**A:** Precompute canonical step scalars once per group and use bounded group/tensor descriptors in one update grid.

**B:** Per-tensor launches or repeated scalar calculation per element.

**Same-version identity:** Respect per-tensor momentum initialization, group hyperparameters, bias correction, weight decay order and maximize semantics.

**Quality:** Every parameter/moment/step counter, heterogeneous groups, empty tensors and full model quality.

**Reject/park:** Reject changed pow recurrence, epsilon placement or flattening away group semantics.

**Prior work:** Existing one-launch SGD is a foundation; new arms concern descriptor/scalar reuse.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn57"></a>
## NN57 — Versioned global gradient-norm reduction

**Affected callers:** Gradient clipping; all neural trainers.

**Source anchors:** `training/checks/optimizer_contract.mojo`; `training/checks/optimizer.mojo`; `training/clip_multi_gpu.mojo`.

**A:** Fixed logical leaves over the declared parameter/tensor order with a canonical norm fold; optionally fuse squared-gradient production with existing readers.

**B:** Incumbent per-tensor norms and scalar combination.

**Same-version identity:** New profile shared host/GPU includes tensor order, squared-term rounding, sqrt, epsilon, clip comparison and nonfinite behavior.

**Quality:** Norms, clipped gradients, multi-step parameter trajectories and model task quality, especially clip-boundary cases.

**Reject/park:** Reject clipping each shard independently or changing tensor grouping without a new contract.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn58"></a>
## NN58 — Fuse accumulation with gradient finishing

**Affected callers:** Neural microbatch training; tied/shared parameter gradients.

**Source anchors:** `training/accumulate_multi_gpu.mojo`; `training/byte_lm.mojo`; `training/checks/optimizer.mojo`.

**A:** Consume each produced gradient into the canonical accumulation buffer while emitting allowed status leaves, avoiding separate copy/add scans.

**B:** Materialize gradient then separately add/scan it.

**Same-version identity:** Microbatch and shared-parameter contribution order fixed, one explicit rounding at each old add, no unordered atomics.

**Quality:** Complete accumulated gradients and optimizer state over varied microbatch schedules and failures.

**Reject/park:** Reject changing effective batch size, averaging order or reusing a buffer before its producer completes.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn59"></a>
## NN59 — Fuse neural dropout RNG and pointwise consumers

**Affected callers:** Dropout; residual blocks; attention/MLP/CNN training.

**Source anchors:** `core/philox_neural.mojo`; `transformer/impl/llama/modeling_llama.mojo`; `x_cnn/ops.mojo`; `sequence/ops.mojo`.

**A:** Generate the existing counter-indexed mask in its pointwise consumer and reuse exact mask coordinates in backward/replay.

**B:** Materialize/read the entire mask or regenerate it in redundant separate passes.

**Same-version identity:** Seed, step, layer, tensor coordinate and dropout scaling order unchanged; graph scheduling never consumes an RNG stream differently.

**Quality:** Masks, outputs, gradients, checkpoint reproducibility and training quality for p=0 and nonzero p.

**Reject/park:** Reject changing stochastic semantics, probability, or omitting a publicly requested mask.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn60"></a>
## NN60 — Parameter/gradient views with generation-safe ownership

**Affected callers:** LM; Samba; neural optimizers.

**Source anchors:** `training/byte_lm.mojo`; `training/byte_lm_model_pool.mojo`; `training/byte_lm_optimizer_pool.mojo`.

**A:** Bind native model and optimizer views to the same owned storage on supported vendors, refreshing after handle swaps.

**B:** Copy all parameters and gradients at each step boundary.

**Same-version identity:** Explicit alias/owner generation, mutation invalidation and synchronization; never silently enable an Apple-only assumption on other vendors.

**Quality:** Load/reset/step/serialize sequences, failed updates, tied weights, concurrent models and multi-step state.

**Reject/park:** Reject unsupported buffer/view capabilities; record the exact Modular API ask.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn61"></a>
## NN61 — One final neural step status/readback boundary

**Affected callers:** LM; Samba; MLP training.

**Source anchors:** `training/byte_lm.mojo`; `training/samba_ops.mojo`; `core/step_glue.mojo`.

**A:** Accumulate safe device status fields and loss summaries for one final owner drain, retaining mandatory pre-access gates.

**B:** Repeated intermediate status/loss readbacks and waits.

**Same-version identity:** Canonical refusal priority and rollback/commit state remain identical; do not defer bounds checks that protect memory access.

**Quality:** Every success/failure state, correct step/counter, consumed outputs and full train quality.

**Reject/park:** Reject hidden CPU work, errors exposed late or removing lifetime barriers rather than replacing ownership.

**Prior work:** I10 supports pieces; complete public-step adoption remains its own experiment.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn62"></a>
## NN62 — Live-range neural scratch and activation arenas

**Affected callers:** LM; transformer; Mamba/Samba; CNN; recurrent models.

**Source anchors:** `training/byte_lm_layer_pool.mojo`; `core/device_arena.mojo`; `training/dev_tensors.mojo`.

**A:** Reuse disjoint-lifetime activation/gradient slabs within a bounded model/session arena; separate allocation and aliasing changes.

**B:** Per-stage temporaries or overprovisioned high-water buffers.

**Same-version identity:** Liveness includes pending device consumers, saved backward state, error paths and replay; all cells read are initialized.

**Quality:** Alternating model sizes, repeated inference/train, checkpointing, memory pressure and full outputs.

**Reject/park:** Reject unbounded retention, inter-session aliases or moving allocation outside cold timing.

**Prior work:** I02 covers GEMM workspace; this extends model-wide liveness.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn63"></a>
## NN63 — Deterministic neural multi-device shard merge

**Affected callers:** Existing multi-GPU neural GEMM/optimizers/training only.

**Source anchors:** `training/optimizer_multi_gpu.mojo`; `training/clip_multi_gpu.mojo`; `training/accumulate_multi_gpu.mojo`; `core/shard_merge_device.mojo`.

**A:** Batch transport and merge canonical logical gradient leaves in one prescribed global order, preserving a single-device equivalent profile where promised.

**B:** Incumbent shard transport/merge schedule.

**Same-version identity:** Device count/topology may schedule transfer but cannot redefine arithmetic leaf membership; no nondeterministic library collective as an identity substitute.

**Quality:** Full global gradients, clip norms, optimizer state and model quality; uneven shards and failure cleanup.

**Reject/park:** If supported Mojo communication/ordering is missing, park and request it; no unsupported transport workaround.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

<a id="nn64"></a>
## NN64 — Neural inference state batching with isolated sessions

**Affected callers:** Transformer/LM decode; Mamba/Samba decode; recurrent inference.

**Source anchors:** `training/byte_lm_model_pool.mojo`; `transformer/impl/llama/modeling_llama.mojo`; `mamba/impl/modeling/modeling_mamba.mojo`; `sequence/recurrent.mojo`.

**A:** Batch independent inference sessions/requests with explicit per-session cache/state lengths and gather/scatter descriptors.

**B:** Separate launches and repeated model-weight staging per request.

**Same-version identity:** Each request retains its own token order, cache/hidden state, RNG and arithmetic profile; length buckets follow work/bytes only.

**Quality:** Every request's logits/state against its own sequential execution, reset/cancel/tails and task quality.

**Reject/park:** Reject cross-session contamination, changed padding arithmetic or latency regressions hidden by aggregate throughput.

**Prior work:** New extension; inspect existing switches before implementation.

**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.

**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage.

## Interaction rounds

- NN01–NN16: GEMM physical schedule × arithmetic profile × workspace × grouped jobs × epilogues; isolate before the proposed complete configuration.
- NN17/18/19/20/21/22/23: attention sharing, masks, saved state, softmax version and backward; measure a full layer and full train step.
- NN24/25/26/27/28/30/31/32: norm arithmetic versus schedule, activation fusion, positions, caches, views and saved forward lifetime.
- NN33–NN40: SSD tiles, scan/suffix/gradient profiles, decay reuse, causal pruning and generations; prefill/decode/backward must compose.
- NN13/41/42/43/44: sequence GEMM, recurrent segments, gates, gradient profile and MoE grouping without changing dependent update order.
- NN14/45/46/47/48: CNN lowering/fusion/gradients/statistics and neural graph operators, including host mirrors and full estimators.
- NN49/50/51/52/53/54: embedding grouping, IDs, head/loss streaming and reduction; account for tied-weight gradient order.
- NN55/56/57/58/59/60/61/62: optimizer status, group scalars, clipping, accumulation, RNG, aliases and arenas; full committed or failed step.
- NN63/64 × NN03/04/20/24/34/38/39/43/54/57: versioned arithmetic must not accidentally depend on GPU count, batching or session routing.

## Excluded shortcuts

No lower precision/TF32 substitution, model-size/rank/sequence/batch/vocabulary reduction, fewer layers/epochs/steps, smaller corpora, skipped gradients or uncertainty checks, sampled softmax, approximate attention, changed dropout/seed, relaxed quality thresholds, removal of correctness settings, Python runtime compute or hidden CPU fallback. Supported low-precision products retain their own semantics; they are not a way to accelerate the FP32 contract by changing it.

## Programming lanes

NN01–NN16 GEMM; NN17–NN32 attention/transformer; NN33–NN48 state-space/recurrent/CNN; NN49–NN64 embeddings/loss/optimizer/runtime. All source stays opt-in in this new worktree. No build, checker, test, execution, benchmark or promotion is authorized. If a full caller/profile cannot be programmed coherently, record concrete remaining code, not a fabricated completion. Unsupported features are parked with Modular asks.

> Keep logs out of context: save complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or infer full success from filtered output.
