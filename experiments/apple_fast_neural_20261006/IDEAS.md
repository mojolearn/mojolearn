# Apple FAST neural A/B ideas — 2026-10-06

Scope: neural networks only, on Apple GPU FAST. Forked from local `main`
at `fd6cf8045`. No classical estimators, trees, nearest-neighbor ANN, or
cross-vendor IDENTICAL tuning. Bits may change from the previous version;
quality, model semantics, safety checks, and public API behavior may not.

**Not tested.** This is a source-only experiment delivery. Do not compile,
verify, run, or measure it in this assignment. Every new mechanism remains
opt-in, with `not tested` beside its switch. Existing defaults and historical
evidence are preserved. Source availability is not proof of quality or speed.

This list was written before implementation was delegated. It covers 44
mechanism cards and eight interaction recipes. Some cards deliberately reuse
existing disabled implementations; their value is a controlled, full-workload
A/B recipe, not a claim of newly invented code. New topology variants and
storage changes are separately selectable. A card may contain several B arms.

## A/B and quality contract for future work

A is the existing FAST implementation with the tested mechanism absent. For a
geometry variant, A enables the same parent mechanism with its incumbent
geometry, so the comparison isolates the geometry. B changes only the named
mechanism; prerequisites appear on both sides unless the entire fusion is the
experiment. Current default-on switches are held constant. No `ALL` define is
an implicit baseline, and no candidate is promoted here.

Future qualification must use the full intended dataset and complete operation
for **every affected estimator**. Record input/corpus version and hash, split,
actual dimensions, all architecture and optimizer settings, seed, numeric mode,
defines, source/binary provenance, hardware, and completion status. Audit caps
in existing recipes; `full` alone is not evidence of full coverage. Keep cold
setup, repeated calls, forward/inference, fit/training, checkpoint, and decode
separate. Include required preparation, upload, synchronization, output
consumption and teardown where the public operation requires them. Do not
substitute a kernel, tiny caller or opponent-only measurement.

Quality is a gate, not a speed tradeoff. Use the existing neural task-quality
rules without widening their tolerances. Compare held-out loss/perplexity,
accuracy where applicable, full training trajectories, logits/output error,
gradients, and recurrent/cache state. Accept bit changes only inside those
rules. Keep optimizer hyperparameters, clipping, steps, data order, seeds,
precision contract, labels, masking and padding semantics fixed. Refusal,
rollback, accumulation, checkpoint/resume and continuation behavior must hold.
An ambiguous quality rule or missing full-workload recipe stays pending.

Later A/B rounds should retain winners, losers, neutral and failed arms, one
excluded warmup and one scored sample under the repository policy, and state
the limited sample count. More robust future quality coverage should include
multiple seeds and held-out examples. Neighboring supported shapes, ragged
tails, short/long sequences, odd widths, empty cases, repeated token IDs,
non-finite inputs and a non-board dataset expose overfitting. No dispatch is
allowed to match benchmark dimensions or dataset names. Geometry choices below
are hardware/resource hypotheses, never measured winners.

## Workload map

| Group | Public operations and future full-workload source | Quality emphasis |
| --- | --- | --- |
| A | TransformerBlock; ByteLM and Samba attention/FFN callers; `tools/bench_board_neural.py` full transformer, LM and Samba recipes; `tools/neural_fast_quality.py` block/Samba definitions | Mask and RoPE semantics, output/logit error, held-out NLL, cache append and continuation, full training when the changed forward is used during fit |
| T | ByteLMTrainer, SambaTrainer, SmallMLPTrainer and optimizer/loss callers reached by them; board LM/Samba/MLP training recipes; `tools/neural_fast_quality.py` and `experiments/performance_ideas/apple_fast/lm_task.py` | Full trajectory and held-out quality; gradients, clipping, parameter/moment updates, refusal rollback, save/resume; the existing small LM task alone is not full-dataset evidence |
| M | Mamba1/2/3 block forward; Samba mixed stack and training where applicable; board full Mamba/Samba definitions | Output and every returned state, long recurrence stability, chunk boundaries, warm state continuation, full-stack held-out quality |
| E | Conv2d, CNNClassifier, Embedding forward/backward and actual neural consumers; `tools/apple_speed_cnn/fastq.py`, `tools/neural_fast_quality.py`, and existing embedding callers | CNN accuracy/loss and backward gradients; embedding duplicate/padding/accumulation semantics and consumer quality. Full CNN/embedding dataset mappings must be resolved before timing; do not claim a board lane exists for them |

The standalone generic GEMM board is not a qualification substitute for neural
callers. Matrix ideas below are scoped to neural projection/SSD/convolution
code so classical operations are unaffected.

## A — Transformer, attention, normalization and projections

Primary source: `transformer/impl/llama/afn_apple_fast.mojo`, its orchestration in
`modeling_llama.mojo`, and the transformer binding. Each existing-arm card gets
explicit controls and an inline not-tested qualification note.

| ID | A → B idea | Why it could help; quality/risk and scope |
| --- | --- | --- |
| A01 | Serial row RMSNorm → existing simdgroup RMSNorm (`AFN_ATTN_NORM_SG`) | Cooperative reduction and coalesced loads expose more parallelism. Keep epsilon and f32 accumulation; reduction order may change. Cover narrow/wide/tail rows and both norms. |
| A02 | Separate RoPE/cache operations → existing fused RoPE + cache append (`AFN_ATTN_ROPE_CACHE`) | Remove intermediate traffic and launches. Preserve absolute positions, rotary pairing and cache ownership; retain the incumbent route outside supported fresh-prefill cases. |
| A03 | Materialized attention → existing online-softmax FLASH (`AFN_ATTN_FLASH`) | Avoid quadratic score traffic and use matrix units. Preserve causal/window masking, max subtraction and denominator stability; cover long sequences and head tails. |
| A04 | FLASH per query head → existing grouped-query FLASH (`AFN_ATTN_GQA_TILE`) | Reuse one K/V tile across query heads. Compare against FLASH on both sides; preserve head mapping and fallback for unsupported groups. |
| A05 | Materialized norm/QKV/gate inputs → existing normalized projection staging (`AFN_ATTN_FUSE_PRE`) | Apply normalization while staging GEMM operands and fold RoPE/cache writes into the epilogue. Check cached and fresh paths separately; avoid recomputing normalization incorrectly. |
| A06 | Separate gate/up/SiLU/down/residual work → existing fused FFN epilogues (`AFN_ATTN_FUSE_MLP`) | Reduce activation intermediates, memory traffic and launches. Preserve activation formula, residual order contract and backward-visible intermediates. |
| A07 | Many block allocations → existing transformer arena (`AFN_ATTN_ARENA`) | Reduce per-buffer and command submission overhead. Include cold allocation and repeated use; guard buffer lifetime, resize and output aliases. |
| A08 | Eight RMSNorm simdgroups/block → four or sixteen | Add an independent norm-only launch-width control. Trade occupancy and block count without changing the flash/projection launch size. Use all 32 lanes per row and correct tail bounds. |
| A09 | FLASH key tile 32 → 16 | Reduce shared storage and potentially improve occupancy/short-window waste, at the cost of more softmax updates/barriers. Recompute resource formulas and retain legal head-class coverage; do not select by a board dimension. |
| A10 | FLASH query tile 32 → 16 | Increase query-block parallelism and reduce per-block accumulator pressure; extra K/V reloads may lose. Preserve fragment ownership, masking and complete writes. |
| A11 | Fused projection K tile 32 → 16 | Reduce threadgroup storage per block; trade additional staging/barriers for occupancy. Apply only to neural fused projection kernels and retain supported alignment/fallback semantics. |
| A12 | Fused projection output-row tile 64 → 32 | Reduce accumulators and increase grid parallelism on short prefills. Keep column/head layout and epilogues unchanged; tail predicates and resource math must follow the selected row tile. |

## T — Training, byte-LM head/backward, optimizers and small MLP

Primary sources: `training/byte_lm_afn.mojo`, `byte_lm_afn_grad.mojo`,
`afn_optim.mojo`, `mlp_fast.mojo`, and their existing public call sites.

| ID | A → B idea | Why it could help; quality/risk and scope |
| --- | --- | --- |
| T01 | Existing step orchestration → existing one-completion LM step (`AFN_LM_NOSYNC`) | Batch refusal status and loss readback; remove intermediate host waits on one ordered context. Keep transactional failure/rollback and host-buffer lifetimes. |
| T02 | Separate softmax, CE and dlogits → existing fused LM head (`AFN_LM_HEAD_FUSE`) | Reuse per-row max/denominator and avoid intermediate memory. Preserve mean scaling, target handling, large-logit stability and held-out loss. |
| T03 | Per-layer pack/unpack → existing flat parameter/gradient views (`AFN_LM_PARAM_VIEWS`) | Remove copy kernels and redundant allocations. Preserve flat registry offsets, alias ownership and checkpoint/resume data. |
| T04 | Unsplit weight gradients → existing occupancy-driven split reduction (`AFN_LM_WGRAD_SPLIT`) | Parallelize long reductions with too few output tiles. This implementation zeroes the destination then atomically adds f32 partial tiles; include the zero launch and atomic contention. Keep f32 and assess gradient/trajectory error. |
| T05 | Separate backward operations → existing independent backward fusion arms (`AFN_LM_BWD_FUSE`, `AFN_LM_BWD_EPILOGUE`, `AFN_LM_BWD_NORM1_RESID`) | Reuse activation/gradient reads and fold residual additions into producers. Keep separate B variants and a combination; leave current backward-no-sync default fixed. |
| T06 | Existing optimizer/loss orchestration → existing scan, clip, multitensor SGD, vector Adam, resident scratch and fused loss arms | Select each `AFN_OPT_*` / `AFN_LOSS_FUSED` arm independently before combined use. No skipped scans, clipping changes or moment approximation. Exercise AdamW and applicable SGD consumers separately. |
| T07 | Existing SmallMLP step → existing two-launch fused step (`AFN_MLP_FUSED_STEP`) | Fuse forward/loss/backward then gradient fold/update. Preserve public architecture, gradients, logits and optimizer semantics. This is not permission to add shape-specific dispatch. |
| T08 | Per-step SmallMLP transport → existing resident and multistep arms (`AFN_MLP_RESIDENT`, `AFN_MLP_MULTISTEP`) | Retain parameters/moments and batch host submission. Keep every logical optimizer step, loss/output contract, state download and rollback. Resident and multistep are distinct B variants. |
| T09 | Weight-gradient minimum K work 256 / split cap 16 → minimum 128 and/or split cap 8 | Add controls based on work per split and bounded reduction overhead; compare with split enabled on both sides. More tasks can improve occupancy or lose to atomic contention. Neighbor shapes are essential. |
| T10 | Optimizer 256-thread launches → 128 or 512 | Vary block size without changing update equations, refusal coverage or vector width. Derive scan storage/grid consistently; retain finite-state handling and clipped updates. |
| T11 | Fused CE launch/reduction geometry → alternative legal block sizes | Separate CE-specific geometry from optimizer update geometry. Preserve full class/row coverage, ignore/padding and mean denominators; no sampled vocabulary or reduced training. |
| T12 | SmallMLP 64 rows/block → 32 or 128 | Trade number of partial-gradient blocks against shared memory/register pressure. Derive maximum partial slots from row capacity and tile size, rather than assuming four blocks. Keep public row limits unchanged. |

## M — Mamba recurrence, SSD and Samba

Primary sources: `mamba/impl/modules/afn_defines.mojo`, `afn_ssd_mma.mojo`,
`afn_refusal.mojo`, `mamba/impl/ops/afn_selective_scan.mojo`, and fused forwards.

| ID | A → B idea | Why it could help; quality/risk and scope |
| --- | --- | --- |
| M01 | Serial Mamba-1 recurrence → existing chunk scan (`AFN_MAMBA1_CHUNKSCAN`) | More time-axis parallelism with carried chunk summaries; f32 reassociation may change long-horizon error. Preserve initial/final state and every token. |
| M02 | Separate convolution/SiLU/split work → existing Mamba-1 input fusion (`AFN_MAMBA1_FUSE_IN`) | Cut launches and intermediates while preserving causal convolution window state and dt/A transforms. |
| M03 | Scalar SSD products → existing Mamba-2 f32 matrix-unit products (`AFN_MAMBA2_SSD_MMA`) | Accelerate C·B, diagonal output and carried-state products. Preserve causal zeros, decay and tail fallback, and compare final state as well as output. |
| M04 | Many Mamba-3 elementwise launches → existing SISO fusion (`AFN_MAMBA3_SISO_FUSED`) | Reduce launch and stage traffic without dropping reports or angle state. Cover mixed Samba stacks and training consumers. |
| M05 | Per-buffer allocations → existing Mamba arena (`AFN_MAMBA_ARENA`) | Lower allocation/submission costs and stage waits. Preserve initialization, resize, lifetime and cold/repeated behavior. |
| M06 | Scalar device refusal reads → float4 device refusal reads | Device refusal is already mandatory; do not recreate a host baseline. Vectorize contiguous reads and local first-failure selection with scalar tails. Preserve named error precedence, non-finite coverage and refusal before published mutation. |
| M07 | 32 recurrence chunks → 16 or 64 | Add a bounded chunk-count control based on parallelism versus carry overhead and shared storage. Support sequences shorter than chunk count without changing recurrence length. |
| M08 | SSD K tile 32 → 16 | Reduce staged operand storage while increasing loop/barrier count. Keep f32, causal clipping and full K coverage; compare on full Mamba2 workloads. |
| M09 | Mamba-3 fused elementwise 128-thread launches → 256 or 64 | Change launch occupancy while keeping cell mapping and state/report writes. Include both forward-only and full Samba use. |
| M10 | Refusal 256 threads / existing work-sized grid → 128 threads and/or four target cells/thread | The current grid already uses min(ceil(n/256),32); increase target work per thread while retaining the grid-stride walk over every cell. Derive work from tensor extent, not dataset identity; keep bounded scratch and ordered failure selection. |

## E — CNN and neural embedding

Primary sources: `x_cnn/afn_direct.mojo`, `x_cnn/device.mojo`,
`embedding/checks/embedding_fast_apple.mojo` (runtime source despite its path),
and `bindings/_mojolearn_embedding.mojo`.

| ID | A → B idea | Why it could help; quality/risk and scope |
| --- | --- | --- |
| E01 | im2col + GEMM + epilogue → existing implicit convolution (`AFN_CNN_DIRECT`) | Gather input taps directly into shared tiles; fuse bias and optional ReLU. Preserve stride, dilation, padding and saved backward columns. Full CNN training quality is required. |
| E02 | Convolution 64 spatial rows/tile → 32 | Reduce input staging and per-block work; increase grid parallelism. Keep all spatial tails and padding predicates. |
| E03 | Convolution 32 output channels/tile → 16 | Reduce per-thread accumulators and shared weights; trade more blocks/input reloads. Do not dispatch on exact channel counts. |
| E04 | Convolution K tile 32 → 16 | Reduce shared memory and tail waste; more barriers may regress. Keep the complete convolution and f32 accumulation. |
| E05 | Every output-channel tile writes saved im2col → only the first output-channel tile writes it | Remove redundant identical global writes and duplicate ownership of backward columns. All channel tiles still gather/read their own operands; the complete columns must be ready before backward. |
| E06 | Existing sorted embedding gradient + scalar gather → existing atomic scatter / vector gather (`AFN_EMB_ATOMIC_BWD`) | Eliminate sorting/run metadata and reduce gather instructions. Keep padding, accumulation and ID refusal. Atomic order changes require repeated-token gradient and consumer-quality gates. |
| E07 | Embedding float4 gather → float8 where width alignment permits | Amortize ID/index work and vector memory operations. Retain float4/scalar fallback for all other widths, no rounding or quantization. |
| E08 | Embedding incumbent launch geometry (256 desired, hardware limited) → alternative 64/128 geometry | Independent launch control for Apple FAST gather/scatter/seed/pad helpers, with grids derived consistently. Compare duplicate-heavy and uniform IDs; no workload-name targeting. |
| E09 | Repeated embedding table upload/scan → opt-in token-owned resident table in FAST | A shared DeviceContext already exists. Reuse the existing table ownership facility to avoid redundant upload/scans until weight assignment changes the token. Preserve refusal, invalidation and error paths; do not change IDENTICAL policy. |
| E10 | Fresh embedding scratch → opt-in existing device scratch pooling in FAST | Reuse shape/capacity-owned buffers across calls. Require full initialization, no stale gradient accumulation, safe resize and resident dependency. Include cold and repeated calls separately. |

## Interaction recipes (after single-mechanism qualification, never run here)

| ID | Combined B configuration | Required boundary and likely interaction |
| --- | --- | --- |
| X01 | A01+A02+A03 | Full transformer and LM prefill: norm/rope fusion can move the dominant cost to attention/projection. |
| X02 | A04+A05+A06+A07 | Grouped attention + fused block + arena: inspect retained stages and storage lifetimes, then full LM/Samba quality. |
| X03 | A08+A09 or A10; A11+A12 | Separate attention and projection geometry pairs against their same-parent baselines; combined shared-memory and occupancy effects are not additive. |
| X04 | T01+T02+T03+T05 | Complete LM training with refusal and checkpoint/resume; fused CE, views and waits share buffer lifetimes. |
| X05 | T04+T09+T06 | Complete training: parallel weight-gradient atomic folding changes optimizer inputs; include zeroing, atomic contention, optimizer scans and updates. |
| X06 | T07+T08+T12 | Full MLP training: tile geometry must agree with resident partial-slot layout and every batched step. |
| X07 | M01+M02+M05+M06; M03+M05+M06; M04+M05+M06 | Separate complete Mamba1, Mamba2 and Mamba3 configurations; also full mixed Samba where reached. |
| X08 | E01+E02+E03+E04+E05; E06+E07+E08+E09+E10 | Separate full CNN and embedding-consumer configurations. Convolution tiles interact; embedding pooling must preserve zero/accumulate behavior. |

## Ideas deliberately outside this source campaign

No lower precision, approximate exponentials, quantization, pruning, fewer
layers/steps/tokens, altered batching semantics, dropout removal, vocabulary
sampling or looser tolerances. These change additional quality contracts and
are not necessary for the f32 scheduling/storage campaign above. No custom
Metal compiler path, hand-patched AIR, compiler-output rewriting, or invented
unsupported command-graph/asynchronous-copy API. If an implementation requires
missing Mojo/Modular support, record the exact missing capability and leave
that arm blocked rather than build a workaround.

No speedups or quality passes are claimed. Future build, verification,
measurement, board updates and default promotion are outside this assignment.

Source-reading refinements during implementation: the attention orchestration
admits A01–A06/A08–A12 only for `forward_only` calls without materialize/trace/
plant requests; backward-visible forwards retain their incumbent route. A07
is the standalone transformer binding arena. Do not claim that these switches
accelerate training kernels merely because LM/Samba also expose training APIs.
M06 and E09 replace already-shipped orchestration ideas with the distinct
mechanisms above. T04/T09 use atomic split accumulation, not split scratch and
a separate final-fold launch. These are implementation-scope corrections, not
verification or performance results.
