# NEURAL-only IDENTICAL experiments — 2026-10-06

Source baseline: `35b2d4976550a6a9a47c4563ca1c08b6de21670b`.
Working branch: `lane/neural-identical-ideas-20261006`.

This is a source-development campaign, requested without compilation,
verification, tests, GPU execution, or timing. Every proposed speedup and
identity argument below is a hypothesis. Newly introduced controls stay OFF.
Existing controls are recorded as existing, not represented as new kernels.
The implementation ledger in `experiments/neural_identical_20261006/` separates
wired source, reusable experiments, prerequisites, and toolchain blockers.

## The numerical rule

**Bits may change between versions. Within each version the NVIDIA, AMD,
Apple GPU and host columns must agree bit for bit.** Compare candidate A
across columns and baseline B across columns. An A-versus-B digest difference
is allowed for a declared arithmetic revision; it is never evidence of a
cross-vendor failure. Equal hashes also do not establish task quality.

Class S changes scheduling/storage with the intention of keeping arithmetic
unchanged. Class V explicitly changes the arithmetic profile and must change
every implementation, independent host oracle, backward path and relevant
decode/checkpoint interpretation together. Class C is a combination experiment.
One numerical version cannot silently select a different fold by GPU vendor,
physical warp width, available workspace, output batch size or tile geometry.
Profiles must name leaf order, zero seed, odd tails, FMA versus separate
multiply/add, rounding, FTZ, NaNs, infinities and signed zeros. This campaign
does not silently substitute TF32/BF16/FP16 for FP32.

Quality means unchanged model/API semantics and no admitted task-quality
degradation: loss, held-out NLL/perplexity or classification/regression metrics
as appropriate, complete gradients and optimizer state, convergence, refusal
semantics, seeded randomness, checkpoint/resume and prefix/decode behavior.
Numerical revisions need a baseline comparison and the existing independent
reference quality gates, including adversarial inputs. A one-step loss or
finite outputs alone do not establish equal training quality. Fix the quality
metric, dataset/split, seed set, training budget and acceptance margin before
future measurement; do not choose margins after seeing a candidate result.

## Coverage and future A/B protocol (not executed here)

The NEURAL board is `tools/bench_board_neural.py`. It includes byte LM,
transformer, Mamba 1/2/3, Samba, training MLP and neural GEMM. Also include
neural public callers in `sequence/` (MLP, recurrent nets, LayerNorm, MoE),
`x_cnn/`, embedding and optimizers when affected. Exclude classical estimators,
trees, clustering, generic time-series/statistical models and Apple FAST.
Shared GEMM changes require a NEURAL-specific opt-in call path or isolated
experiment profile; do not change classical defaults to run this campaign.

| Workload code | Saved recipe and required coverage |
| --- | --- |
| LM | `tools/bench_board_neural.py`: `LM_SHAPES`, byte corpus acquisition and batches; forward, full train step, held-out quality, cold and repeated sessions. Full declared shape is B=1,L=2048,D=384,H=6,KV=6,Dh=64,FF=1024,layers=8,V=8192. This shape alone does not prove full-corpus coverage. |
| TR | Same file: `BLOCK_SHAPES['transformer']`; full forward plus actual backward/LM callers, GQA, causal/window masks and decode. |
| M1/M2/M3 | Same file: `BLOCK_SHAPES`; B=1,L=2048,D=384, respective complete block operations, backward and streaming callers where changed. |
| SA | Same file: `SAMBA_SHAPES`; B=2,L=512,D=384,V=256, four alternating Mamba3/attention layers; forward and complete train step, byte corpus coverage. |
| ML | Same file: `MLP_SHAPES`, `MLP_DIMS`; full declared B=256 synthetic fixture. Also actual `sequence/mlp_fit.mojo` classifier/regressor full-dataset recipes when affected. |
| GE | Same file: `GEMM_SHAPES`; 4096 cubed is a component recipe, not a substitute for affected full neural caller operations. Cover NN/NT/TN, gradients and decode-shaped products in those callers. |
| SQ | `tools/classical_two_datasets.py`, `tools/bench_board_algos.py` and `sequence/` neural caller definitions: record exact neural estimator/data mapping before timing; unresolved mappings remain pending. |
| CN | `tools/classical_host_lanes/cnn.py`, `tools/identical_wave_prepare_neural.py`, `x_cnn/`; record full CNN fit/predict datasets, steps and consumed state before timing. |
| EB/OP | Embedding/optimizer components qualify only through LM, SA, ML, SQ or CN callers that actually reach them; standalone fixtures are diagnostics. |

The existing host NEURAL board has `CPU_LENGTH_CAP=512`; its `full` selector
is not an uncapped GPU-shape host witness or evidence of full-corpus timing.
Record actual dataset version/hash/split, dimensions, intrinsic caps, model
settings, RNG, numeric profile, defines/env, binding/source/compiler/binary
hashes, physical device and allocation, and whole-operation boundary before
future timing. Include required input preparation/upload, fit/training,
synchronization and consumed outputs; separate inference, cold and repeated
use. For synthetic-only lanes use the full declared fixture. Do not silently
reduce a dataset or claim the board's finite batch stream is a whole corpus.

Future rounds freeze one commit, reuse accepted compilation/identity evidence
where exactly applicable, and use one excluded warmup plus one scored sample
per arm and workload on NVIDIA and AMD. Run vendors on separate owned boxes
and cells serially per GPU. Score identity from the execution, add Apple and
host witnesses, and retain all failures. NVIDIA and AMD jointly decide speed:
combined improvement, neither materially slower. Apple never votes on
IDENTICAL performance. Full-machine CPU arms obey cgroups and documented
nested-pool policy. No source promotion or board-result admission is authorized
by this uncompiled campaign. Future results go through board tools only.

Every route/threshold needs an explicit memory, hardware or operation-cost
argument valid for neighboring shapes. No benchmark dimensions or dataset
names in dispatch. Removing an old targeted rule uses that old rule as B and
requires neighboring shapes plus a non-board dataset. Test interactions and
the proposed full default configuration before promotion. Preserve losers,
neutral cases and pending coverage next to the control and in the ledger.

## 64 A/B ideas

A is the candidate; B is the frozen current route or an explicitly specified
control. Priorities P1/P2/P3 indicate expected reach and implementation risk,
not measured performance. A control sweep is an experiment definition, not
permission to turn on every switch together.

### Dense neural products — G01–G10

Primary sources: `gemm/checks/gemm_identical.mojo`, `gemm/contract.mojo`,
`gemm/experiments/`, `training/dev_tensors.mojo`, transformer projections.

| ID | Class / priority / callers | A versus B, mechanism and particular acceptance condition |
| --- | --- | --- |
| G01 | S/P1/GE,LM,TR,SA,M1–3 | Isolate existing AMD small-MFMA and NVIDIA stepped-kpack schedules versus their documented OFF arms. Reduce scalar fallback without changing logical leaves. Record actual route and full caller time; no repeating already decided identical cells. |
| G02 | S/P1/GE,LM,TR,SA | One versus two staging pages, and smaller register tiles versus current tiles. Reduce occupancy pressure; evaluate scratch/register traffic and tail lanes. Existing A02/N01 arms are reusable, not new implementations. |
| G03 | S/P1/LM,TR,SA | Group independent Q/K/V, gate/up or dA/dW products in one launch versus separate calls. Separate output ownership and original K per product; grouping cannot concatenate a reduction or alter rounding. Wire actual callers beyond the existing grouped-jobs fixture. |
| G04 | S/P1/LM,SA,TR | Retain bounded GEMM scratch per session versus per-product allocation. Cap by a documented byte budget; large/short alternating shapes, reset and concurrent owners must not alias or retain unbounded memory. |
| G05 | S/P2/LM,TR,SA | Fuse bias/residual/scaling into the final product write versus separate passes, explicitly materializing original rounding/FTZ seams. Measure increased register pressure and every backward stash consumer. |
| G06 | S/P2/GE,LM,SA | Cache deterministic execution plans by complete operation/layout/profile/device key versus repeat plan search. Invalidate on workspace policy/device changes; never cache an unproven numerical route. |
| G07 | S/P2/GE,LM,TR | Bounded leaf groups with in-kernel fold stacks versus global partial tensors. Keep logical leaf indices and adjacent-pair tree exact, including odd tails; count fold traffic and available parallelism together. |
| G08 | S/P2/LM,SA,ML | Persistent immutable weight packing versus per-call packing/gathering. Optimizer-owned generation stamps invalidate after updates and external mutations; no pointer-identity-only cache. Include cold packing and memory capacity. |
| G09 | S/P2/GE,LM,TR | Output-tile swizzle or grouped jobs sharing read-only operands versus linear tile order. Improve cache reuse with no changed arithmetic. Neighbor shapes and transposed operands guard against a board-only win. |
| G10 | V/P3/GE,all neural GEMM callers | A named shorter-leaf FP32 profile (existing 64-element research arm first) versus current leaf profile. Update host/GPU contracts together; inspect conditioning and full training quality. Register both version digests rather than reject A because it differs from B. |

### Attention and transformer blocks — A01–A10

Primary sources: `transformer/impl/llama/{modeling_llama,fused_attention,attention_v2}.mojo`,
transformer backward sources and corresponding host implementations.

| ID | Class / priority / callers | A versus B, mechanism and particular acceptance condition |
| --- | --- | --- |
| A01 | S/P1/TR,LM,SA | Existing GQA K/V head-reuse schedule versus independent query-head loads. Cover true GQA, full multi-head, causal tails and windows; no altered head mapping or reduction order. |
| A02 | S/P1/TR,LM,SA | Retain exponent/probability state versus exact backward recomputation. Evaluate entire train step and peak memory, not backward alone; preserve masks, RNG, lifetime and recomputation refusal behavior. |
| A03 | S/P1/TR,LM,SA | Two-output fused SiLU-plus-gate versus separate passes, retaining the activation needed by backward. Keep nonlinear/divide and rounded product seams; existing forward-only fusion is a control, not evidence for training. |
| A04 | S/P1/LM,SA | Cache-free prefill for training versus allocate/append/copy a KV cache whose outputs are unused. Limit to explicit non-decode callers; preserve prefill traces and all retained backward inputs. |
| A05 | S/P2/TR,LM,SA | Compute canonical row stats once, then parallel scaling/residual application versus serial row walks. Preserve stat chain; include extra launch and scratch in timing. Distinct from the numerical tree revision V02. |
| A06 | S/P2/TR,LM | Hoist validated RoPE frequencies and retain position tables versus per-layer scan/rebuild. Key by full RoPE options, absolute range, dtype and device; extension/window/reset must invalidate correctly. |
| A07 | S/P2/TR,LM,SA | Reuse backward d_out as a read-only input view versus a full entry copy. Prove the destination's mutation lifetime in source, preserve ownership and alias rejection, include caller reuse. |
| A08 | S/P2/TR,LM,SA | Cooperative query/output lanes with shared score state versus each output lane recomputing normalizers. Preserve score, maximum, denominator and value order. Revisit rejected cooperative dK/dV only with a concrete changed resource mechanism. |
| A09 | V/P3/TR,LM,SA | Fixed absolute key blocks with versioned softmax partials/merge versus sequential row normalization. Shared host/forward/backward/decode profile, canonical masks and empty rows; online recurrence is not inherently vendor-identical. |
| A10 | C/P2/TR,LM,SA | Winning G03+A01+A02+A03+A04 combinations versus individual winners and frozen baseline. Measure memory/occupancy interactions and full configuration; independent wins never establish the combination. |

### Mamba and Samba — M01–M10

Primary sources: `mamba/impl/modules/`, `mamba/impl/ops/`,
`mamba/host/gen/`, `training/samba_ops.mojo` and Mamba bindings.

| ID | Class / priority / callers | A versus B, mechanism and particular acceptance condition |
| --- | --- | --- |
| M01 | S/P1/M2 | Existing SSD tiles plus retained G⊙L versus tiles without retention and scalar control. Use a byte-cost budget, preserve triangular/backward trace semantics, include storage and tail chunks. |
| M02 | S/P1/M1,M2 | Token-parallel depthwise convolution versus channel-owned sequence loops. Original tap order/zero padding and streaming state; existing route needs full caller A/B, not another duplicate kernel. |
| M03 | S/P1/M3,SA | Retain per-token/head exponential decay once versus recompute per feature. Cache only mathematically identical operands with the same primitive; backward/reset/sequence extension invalidate correctly. |
| M04 | S/P2/M3,SA | Extend existing intra-chunk shared tiles to resource-feasible AMD shapes versus scalar recomputation. Resource reasoning replaces vendor/dimension folklore; padded tails and finite checks stay live. |
| M05 | S/P1/SA | Retain attention/Mamba forward stages across backward versus replay forward in the step. Key by owned input/weight generation, include activation memory and checkpoint mode; never Python data processing. |
| M06 | S/P2/M1–3,SA | Retain scratch and weight descriptors with mutation stamps versus allocate/byte-compare/reupload each call. Warm and cold sessions, model interleaving, updates and checkpoint restoration must agree. |
| M07 | S/P2/M1–3,SA | Sweep fixed launch geometry and independent output tiling versus current geometry. Arithmetic ownership stays fixed; choose eventual rule from hardware resource/cost, not exact state or sequence sizes. |
| M08 | V/P3/M1 | Absolute-position fixed-chunk affine scan versus serial recurrence. Forward, reverse scan, host, streaming/decode and resume use one new profile; variable request segmentation cannot redefine chunk boundaries. |
| M09 | V/P2/M3,SA | Shared suffix carry for angle-gradient sums versus per-token O(L²) suffix folds. Explicit new suffix order shared by host/backward; compare gradients and training convergence, especially cancellation/long sequences. |
| M10 | V/P2/M2 | Fixed leaves and balanced folds for backward parameter gradients versus serial token chains. Cover convolution, projection and dt gradients, tails and batch composition; no unordered atomics. |

### Training, loss and optimizer — T01–T10

Primary sources: `training/{byte_lm,mlp_ops,samba_ops,chunked_lm_head_v2}.mojo`,
`training/checks/{loss,optimizer}.mojo`, corresponding host/oracle files.

| ID | Class / priority / callers | A versus B, mechanism and particular acceptance condition |
| --- | --- | --- |
| T01 | S/P1/LM,SA,ML | Emit post-update status while p/m/v are live versus rescan full arrays. Preserve ordered first-error field/index and transaction behavior; existing I10 is the starting point. |
| T02 | S/P1/LM,SA | Device status accumulation and one completion read versus intermediate waits. Only defer checks that cannot cause unsafe access; retain original error precedence and roll back failed updates before exposure. |
| T03 | S/P1/LM,SA | Device arena views for parameters/gradients versus full copies around layers. Rebind after swaps; checkpoint, tied weights, optimizers and all vendor ownership rules are included. |
| T04 | S/P1/LM,SA | Resident token/target batches and reusable pinned staging versus allocate/upload/read back repeatedly. Validate IDs before use; update batches explicitly, no stale-byte assumptions. Cold transfer remains timed. |
| T05 | S/P1/LM,SA,ML | Fuse CE weights and dlogits, reuse shift-exp values versus separate elementwise passes. Keep denominator and loss folds; ignore-index, class tails, extreme logits and gradient scale must match. |
| T06 | S/P2/LM | Bounded GEMM chunked LM head versus scalar streamed logits and dense head. Compare complete loss+dhidden+dweight and inference; no approximate/sampled softmax or missing dense gradient cost. Existing V2 is a distinct arithmetic control. |
| T07 | S/P2/LM,SA,ML,SQ,CN | One descriptor-driven SGD/momentum arena launch versus per-tensor launches. Keep weight decay, clipping, Nesterov operation order and tensor error order; mirror host semantics. |
| T08 | S/P2/LM,SA,ML,SQ,CN | Reuse deterministic norm/clip partials versus duplicate parameter scans. Respect global tensor order, overflow, empty tensors and disabled clipping; no change to the clipping decision. |
| T09 | S/P2/LM,SA | Microbatch gradient buffers and explicit immutable reduction schedule versus repeated allocation/clear. Existing accumulation semantics and RNG coordinates are invariant; no changed effective batch size or training steps. |
| T10 | C/P2/LM,SA,ML | Combine admitted residency, status, loss and optimizer arms versus individual winners. Include cold start, full step, repeated epochs and checkpoint; expose interacting regressions before any default changes. |

### Embedding and output handling — E01–E06

Primary sources: `embedding/checks/{embedding_identical,embedding_sort}.mojo`,
embedding host/bindings, byte-LM embedding and output paths.

| ID | Class / priority / callers | A versus B, mechanism and particular acceptance condition |
| --- | --- | --- |
| E01 | S/P1/EB,LM,SA | Existing stable radix (token, original position) grouping versus vocabulary scan/bitonic. Include sort workspace, zeros for untouched rows, duplicates/padding and highly skewed IDs. |
| E02 | S/P2/EB,LM,SA | Group only touched row descriptors and sparse temporary accumulation versus scanning every vocabulary row. Materialize required dense gradients in the timed boundary; optimizer weight decay still visits required state. |
| E03 | S/P2/EB,LM,SA | Reuse sorted IDs/segments between same-step consumers versus repeat grouping. Key by owned ID generation, sequence extent and vocab/padding settings; input changes invalidate. |
| E04 | S/P2/EB,LM,SA | Coalesced vector-width gather/scatter over independent feature cells versus scalar loads. Tail/alignment-safe, no float atomics, no changed duplicate accumulation order. |
| E05 | S/P2/LM,SA | Shared prerefused ID/target validation within a step versus repeated download-and-check. Public unvalidated inputs still pass complete range/refusal checks; cached validity belongs to a versioned buffer. |
| E06 | S/P2/LM,ML,CN | Device argmax/metric sufficient outputs for APIs requesting only those outputs versus materialize/download full logits. Preserve ties/NaNs and API contract; full-logit APIs still return all logits. |

### Other neural families — S01–S10

Primary sources: `sequence/{gemm_tiled,recurrent,recurrent_scan,layernorm,
mlp_fit,moe,adafactor}.mojo`, `sequence/{ops,exec_device}.mojo`, `x_cnn/`.

| ID | Class / priority / callers | A versus B, mechanism and particular acceptance condition |
| --- | --- | --- |
| S01 | S/P1/SQ | Shared tiles for same-chain strided sequence GEMM versus per-cell global reads. Preserve K chain, transpose/stride and zero seed; actual recurrent and MLP fit callers required. |
| S02 | S/P1/SQ | Supported single-block recurrent scan versus per-timestep launches. Enforce hidden-state dependencies and supported barrier semantics across vendors; park if Mojo cannot guarantee required ordering. |
| S03 | V/P2/SQ | Versioned fixed-leaf recurrent weight/bias gradients versus serial T×B folds. Host/device executor and oracle agree; forward quality alone does not qualify training. |
| S04 | S/P2/SQ | Split LayerNorm stats and cell application versus serial row walks. Reuse stats in backward when lifetime permits; epsilon, population variance and output rounding unchanged. |
| S05 | V/P2/SQ,ML | Device epoch permutation with a declared new RNG mapping versus host Fisher–Yates. Same permutation on every column for seed/epoch/sample count; assess full training quality, no Python row processing. |
| S06 | S/P2/CN | General implicit/direct convolution replicating GEMM leaves versus im2col+GEMM. No model-size targeting; tensor indexing, dilation, groups, stride and tails have explicit coverage. |
| S07 | S/P1/CN | Retain im2col across convolution backward versus recompute, with byte-budget policy. Include forward memory/copies and batch capacity; avoid stale input or changed padding semantics. |
| S08 | S/P2/CN,SQ | Compute row exponentials once and reuse for softmax versus two exponent calls. Preserve canonical max, denominator, NaNs and probability normalization; assess saved bytes versus extra scratch. |
| S09 | S/P2/SQ | Stable MoE expert grouping and bounded batched products versus independent tiny expert jobs. Stable token/tie order, same top-k routes, capacity/overflow and gather/scatter order; no changed model semantics. |
| S10 | V/P2/SQ,OP | Fixed-leaf Adafactor/LAMB norm and factor reductions versus serial chains. Shared host/device primitive and exact profile identity; preserve epsilon, clipping/trust ratios and update ordering, then assess convergence. |

### Deliberate arithmetic revisions and supported future work — V01–V08

These are experiments specifically enabled by permission to change version
bits. They are not permission to use vendor-dependent native approximations.

| ID | Class / priority / callers | A versus B, mechanism and particular acceptance condition |
| --- | --- | --- |
| V01 | V/P3/GE,LM,TR,SA | Multiple independent FP32 accumulators per logical leaf with a fixed merge versus serial FMA dependency. Specify chain assignment/merge/FTZ identically on host and all GPUs; whole-caller quality, error under cancellation and actual throughput mechanism required. |
| V02 | V/P2/TR,M1–3,SQ | Balanced row-normalization sum/dot profile versus serial sums. Fixed logical leaves independent of physical warp width; include forward variance/RMS, backward dot, dweight/dbias and epsilon placement. |
| V03 | V/P3/TR,LM,SA | One deterministic shared reciprocal per row plus multiply versus repeated division. Implement or reuse a supported correctly specified primitive, preserve exceptional rules, update all host/device paths; task quality and near-zero conditioning are critical. |
| V04 | V/P3/LM,SA,ML,CN | Fixed-block CE max/sum/loss and gradient profile versus current folds. Keep full vocabulary/objective; finite, large-negative and ignored rows, gradient scale, all-class tails and training quality included. |
| V05 | V/P3/neural nonlinear callers | A versioned shared polynomial/range-reduction primitive for exp/sigmoid/SiLU/tanh versus current primitive, only with an explicit error/domain bound and all-column implementation. Native vendor math is not a valid substitute. Assess accumulated training error, not isolated ulps only. |
| V06 | V/P3/LM,SA,ML,SQ,CN | Fused multiply-add at selected optimizer/activation seams versus separately rounded multiply/add. One precisely named seam per A/B; shared oracle changes and model-state/convergence gates. No global fast-math. |
| V07 | S/P3/LM,SA,TR | Supported graph replay or command batching for stable resident steps versus ordinary enqueue. Only implement when Mojo/Modular exposes required supported lifetime/error semantics; otherwise record the concrete upstream request. No toolchain patching or raw backend workaround. |
| V08 | C/P2/all affected neural callers | Complete proposed numerical version (selected G10/M08–10/S03/S05/S10/V01–06 plus schedule winners) versus last version. New version identity is a separate matrix; ablate interactions and keep rejected/neutral arms off. Never promote from a component-only or one-vendor win. |

## Implementation order and ownership

1. Write this complete inventory before delegation (done by the primary agent).
2. Program independent opt-in source and selectable A/B definitions in parallel:
   transformer/training (A/T plus V03/V04/V06/V07), Mamba/Samba (M),
   embedding/sequence/CNN (E/S plus V02), dense profiles and catalog/runner
   (G plus V01/V05/V08, primary agent).
3. Reuse existing valid arms; distinguish caller wiring from a component-only
   prototype. Do not fabricate a define for unimplemented source. Numerical
   revisions without complete host/device coverage remain visibly unqualified
   and cannot masquerade as S-class optimizations. Unsupported facilities are
   blocked, with an upstream ask, rather than emulated.
4. Keep source-development status separate from validation. No builds, static
   check commands, tests, verification scripts, timings, rentals or board runs
   in this task. Do not merge to main or change any enabled production default.

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output.

Historical context: `docs/plans/IDENTICAL_NEURAL_ROADMAP.md`,
`experiments/performance_ideas/IMPLEMENTATION_STATUS.md`, the I01–I11/A01/A02/N01/N02
cards and `bench/evidence/` attention rejection notes. Historical compilation
and rejected kernels do not establish success for newly edited source.
