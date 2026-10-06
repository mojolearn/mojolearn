# Neural IDENTICAL performance experiments

This catalog contains 60 neural-only A/B ideas for GEMM used by neural models,
CNNs, attention and transformer blocks, language-model training, embedding,
losses and optimizers, Mamba 1/2/3, Samba, recurrent networks, neural MLPs, MoE,
GCN/GraphSAGE message passing and neural channel dropout.
Trees, classical estimators and the non-neural algorithms sharing `sequence/`
are excluded. Every speed claim below is a hypothesis. The requested work is
source only: no compilation, execution, tests, lint, verification or measurements.

## Required identity and quality

**Bits may change between versions. Within each version, NVIDIA, AMD, Apple and
the host must produce the same bits.** Historical output equality is not an
acceptance requirement for a new numerical version. Compare candidate A across
columns, and separately compare reference B across columns. A versus B checks
quality and semantics; it checks exact bits only for arms claiming unchanged
arithmetic. A candidate with a new numerical graph needs its host counterpart,
backward, state/checkpoint implications and every affected consumer together.

- **S** means a scheduling/layout/lifetime candidate intended to retain the
  present arithmetic graph. This label is a design intent, not verified identity.
- **V** means an explicitly new arithmetic version. Define logical leaf lengths,
  operation/merge order, odd tails, FTZ, FMA contraction, division and portable
  transcendental behavior independently of warp/wave width and physical tiles.
- Maintain estimator settings, precision profile, objective, optimizer, seed and
  training budget. Do not get a speedup by reducing tokens, context, layers,
  iterations, gradient coverage or model capacity. No approximate softmax,
  truncated recurrence, hidden mixed precision, TF32 substitution or dropped
  nonfinite checks. New FP32 reduction orders are permitted; reduced precision is
  a separate API/quality decision and is not proposed as an automatic default.
- Retain supported error names/priority, transactional failed-step behavior,
  mask semantics, output lifetimes, weight updates and checkpoint/resume behavior.
  Training quality includes held-out loss/accuracy and multi-step trajectories,
  gradients and optimizer state, not only one finite forward result.
- No exact board dimension, dataset name or threshold fitted around a board row
  may control dispatch. A threshold follows bytes, work or hardware resources
  and has a source comment. Removing a suspicious old rule retains it as B and
  measures neighboring shapes plus a non-board workload later.
- Python remains the API/experiment glue. Tensor loops, sorting, arithmetic,
  sampling and training computation belong in Mojo on the selected device or
  Mojo host binding. Do not add a Python runtime implementation.
- Use supported Mojo/Modular features. If a required capability is unavailable,
  record the upstream ask and stop that candidate; do not patch compiler output,
  hand-edit IR or invent an unsupported launch/portable build mode.

## A and B and eventual measurement

For every card, **A is the candidate and B is the current/reference route**.
Existing opt-ins are reused rather than recreated. New switches are independent,
IDENTICAL-only, default OFF, with `MOJOLEARN_IDN_ALL_OFF` restoring the reference.
No existing accepted default is changed by this task. A source implementation,
helper or arm recipe is not a compiled/identity-qualified/performance-qualified
result. Keep these statuses separate.

Before future timing, map each card to every affected full estimator recipe,
dataset/corpus version and hash, split, actual dimensions, settings, seeds,
defines, numerical version and complete timing boundary. Audit intrinsic caps:
`--shape full` does not establish full intended coverage. Record compiler,
source/binary/harness hashes and hardware. Missing mappings remain pending.
Freeze one source commit per A/B round, compile once on the designated cheap
machines and reuse accepted evidence thereafter. None of that runs in this task.

Measure complete operations including preparation, forward/backward, required
sync, optimizer where applicable and consumed outputs. Distinguish cold setup,
repeated calls, training steps/trajectories, prefill and decode. Use one excluded
warmup and one scored sample initially. Run NVIDIA and AMD on separate owned
boxes, cells serially per GPU. Promotion requires combined improvement with
neither vendor materially slower, all-column identity and no task-quality
regression on all affected full workloads. Apple verifies identity and its timing
does not vote on IDENTICAL defaults. Set quality gates before observing results.
Retain winners, losers, failures and pending cells; update boards through board
tools only when actual results exist. Component timing never closes full coverage.

## Full workload references

| Key | Workload and recipe sources |
| --- | --- |
| LM | `tools/bench_board_neural.py` and `tools/torch_lm_step_opponent.py`: full LM forward/train, exact corpus bytes/hash, architecture, token/context settings, AdamW, clip and state policies |
| TR | Same neural driver: transformer forward/inference; transformer public backward/training consumers, full prefill and decode recipes where available |
| MM | Same neural driver and `mamba/corpus/gen_corpus.py`: full Mamba 1/2/3 and Samba forward/train; backward/prefill/decode and adversarial recurrence coverage in native public callers |
| CNN | Neural CNN lanes/settings in `tools/bench_board_algos.py`, `tools/classical_two_datasets.py`, `tools/bench_board_more.py` and public CNN training callers; record the actual full image set and preprocessing |
| SEQ | Neural-only LSTM/GRU/RNN, MLP and MoE entries in those expanded drivers; exclude time-series/statistical estimators even where they share executor code |
| OP | Embedding, attention, cross-entropy, norms, optimizers and GEMM component recipes plus all full LM/TR/MM/CNN/SEQ consumers they affect; standalone timing is diagnostic only |

These are saved recipe locations, not claims that every required mapping already
exists. In particular, CPU neural lanes have a documented sequence cap and
auxiliary backward/decode/MoE tasks may need new full-workload recipes. Dataset
hashes, precise dimensions and missing public consumer recipes stay pending until
they are recorded. No invented corpus, substitute tiny fixture or opponent-only
measurement can fill that gap.

## GEMM and CNN lane

### NI01 Neural GEMM scratch lifetime

**A/B:** retain capacity for live partials/admission/folds across a neural session
versus allocate and drain per operation. **Savings:** allocation, zeroing and host
waits. **Contract:** S; no numerical changes, no stale partial reads.
**Source:** `gemm/checks/gemm_identical.mojo` GemmWorkspace and
`training/dev_tensors.mojo`; reuse I02. **Full scope:** LM/TR/MM/CNN/SEQ.
**Risks:** changing shapes, noncontiguous inputs, failed calls, stream ownership,
peak memory and cold-vs-repeat tradeoff. Reuse capacity, not mutable operand data.

### NI02 Bounded streaming of GEMM partial planes

**A/B:** compute and merge bounded windows of canonical leaf partials versus retain
all planes. **Savings:** scratch bandwidth/peak memory.
**Contract:** S only when the original logical merge tree and odd-tail behavior
are preserved; an online left fold is a distinct V arm.
**Source:** GEMM grouped kernels/fold and host contract; reuse existing I02/N02
streaming work. **Full scope:** all neural GEMM consumers.
**Risks:** window-boundary carries, numerical repair flags, small-K overhead.

### NI03 GEMM launch plans from resource costs

**A/B:** choose tile/group/split schedule by live bytes, occupancy and work versus
the current rule; replace dimension-targeted rules with the old rule as B.
**Savings:** better device fill without excessive partials.
**Contract:** S; logical leaves/folds independent of schedule.
**Source:** GEMM plan selection and `checks/kernel_matrix_gemm.mojo`.
**Full scope:** all neural GEMM consumers, neighboring shapes and non-board model.
**Risks:** unsupported resource queries must wait for Modular; no fabricated CU
counts or constants tuned around board rows. Retain old rejected-plan evidence.

### NI04 Smaller exact AMD matrix tiles

**A/B:** supported exact matrix-instruction body at smaller tiles versus scalar
fallback for underfilled calls. **Savings:** arithmetic throughput and fill.
**Contract:** S only for a documented supported instruction mode matching the
canonical scalar steps; unknown multi-K internal accumulation is not admissible.
**Source:** GEMM MFMA body/staging and existing AMD profiles.
**Full scope:** LM/TR/MM/MLP and all changed neural callers on AMD and NVIDIA.
**Risks:** registers, shared memory, exact fallback/admission, tail cells; NVIDIA
still participates in complete candidate acceptance even for AMD-oriented work.

### NI05 NVIDIA register tiles without partial workspace

**A/B:** smaller kpack/register tiles selected by resource costs versus grouped
partials. **Savings:** avoid global partial-plane traffic.
**Contract:** S; same canonical K leaves and tree.
**Source:** GEMM kpack bodies and existing NVIDIA candidate profiles.
**Full scope:** LM/TR/MM/CNN/SEQ plus all transpose forms.
**Risks:** register spills, low block count, odd tails and tuning to one shape.
Compare several analytically motivated tile/page choices without flipping defaults.

### NI06 GEMM operand staging and page depth

**A/B:** one vs two staging pages, cached FTZ inputs and coalesced stride-aware
loads versus present staging. **Savings:** shared-memory pressure or load overlap.
**Contract:** S; copied bits and contraction boundaries unchanged.
**Source:** GEMM operand/FTZ staging and existing one-page controls.
**Full scope:** all neural callers, NN/NT/TN/TT and noncontiguous public inputs.
**Risks:** data hazards, barriers, subnormals, exact numerical repair, extra staging
cost. Hardware schedules may differ; their numerical graph cannot.

### NI07 Batched independent neural projections

**A/B:** one multi-job launch/workspace for compatible Q/K/V, gate/up or dA/dW
projections versus separate calls. **Savings:** launches and repeated operand reads.
**Contract:** S; independent jobs keep their own GEMM graph; concatenating reduction
dimensions does not automatically preserve it.
**Source:** GEMM multi-job API and neural callsites; reuse I03.
**Full scope:** LM/TR/MM/MLP forward and backward.
**Risks:** mixed strides, buffer aliases, job tails, output lifetime and resource
imbalance. Integrate real callers, not only a synthetic grouped benchmark.

### NI08 Versioned GEMM leaf lengths and balanced folds

**A/B:** explicitly shared alternative logical FP32 leaf length/tree versus current
contract. **Savings:** shorter serial dependency or fewer partials, if demonstrated.
**Contract:** V; update host and every GPU, all transposes, tails, FTZ and repair.
**Source:** `gemm/contract.mojo`, host identical GEMM and device kernels; reuse I04
design where applicable. **Full scope:** every affected neural estimator.
**Risks:** cancellation, overflow, changed gradients/convergence and model-wide
interaction; no speed claim from a leaf microbenchmark or unknown tensor-core order.

### NI09 Epilogues adjacent to canonical GEMM

**A/B:** bias/residual/activation epilogues on final reduced output versus separate
passes. **Savings:** global rereads and launches.
**Contract:** S when explicit post-reduction statements retain rounding; V only
as a separate shared contract. Never fold bias into initial FMA accumulator silently.
**Source:** GEMM epilogue seam and neural linear/CNN/MLP callers; reuse I05.
**Full scope:** actual forward/backward training consumers.
**Risks:** saved preactivations, aliasing, signed zero, tail masks and refusal order.

### NI10 Retain convolution columns for backward

**A/B:** keep exact im2col output until dW versus rebuild in backward.
**Savings:** avoid repeated input transformation.
**Contract:** S; identical padding, dilation, layout and data ownership.
**Source:** `x_cnn/device.mojo`, conv training tape and bindings.
**Full scope:** full CNN fit and forward/backward calls.
**Risks:** large columns memory, checkpoint/recompute interactions, changed input
after forward, multiple live tapes. A byte budget must select reuse vs recompute.

### NI11 Direct exact convolution for bounded reductions

**A/B:** direct convolution preserving the GEMM graph versus im2col plus GEMM.
**Savings:** eliminate expansion reads/writes.
**Contract:** S for a supported exact reduction; large reductions need explicit
canonical leaves, not a free sequential dot.
**Source:** existing DIRECT_CONV routes in `x_cnn/device.mojo` and `x_cnn/ops.mojo`.
**Full scope:** CNN fit, pointwise/depthwise/grouped cases and neighboring kernels.
**Risks:** padding/stride/dilation/tails, crossover costs and input-gradient parity;
the Apple FAST direct implementation is not automatically IDENTICAL-safe.

### NI12 Implicit convolution with canonical tiles

**A/B:** generate im2col addresses inside canonical GEMM tiles versus materialize
columns. **Savings:** peak memory and expansion bandwidth.
**Contract:** S; logical K mapping and operand FTZ exactly defined.
**Source:** CNN launchers, GEMM load seam and matching host indexing.
**Full scope:** CNN training and inference including large spatial images.
**Risks:** address arithmetic overhead, padding zeros versus skipped products,
group offsets and arbitrary strides. No tensor-core shortcut with an unknown fold.

### NI13 Cache convolution weight layouts by generation

**A/B:** retain packed/transposed immutable weights until optimizer generation
changes versus repack each call. **Savings:** preparation traffic and launches.
**Contract:** S; bytes and model ownership unchanged.
**Source:** CNN model/tape and device packing paths.
**Full scope:** CNN inference and repeated training steps.
**Risks:** refit/update invalidation, public mutable weights, dtype/layout changes,
multiple streams and memory release; pointer identity alone is insufficient.

### NI14 Convolution backward gather tiling

**A/B:** reuse neighboring gradient tiles in the existing gather-style col2im
versus repeated scattered loads. **Savings:** input-gradient memory traffic.
**Contract:** S; same contribution sequence per output, no floating scatter atomics.
**Source:** `x_cnn/ops.mojo` col2im and device launchers.
**Full scope:** CNN fit with overlapping windows, stride/dilation and groups.
**Risks:** boundary tiles, many overlapping taps, occupancy and shared-memory limits.
Keep integer address transformations separate from the numerical reduction graph.

### NI15 Dedicated canonical bias and parameter folds

**A/B:** direct canonical column folds versus GEMM against a ones buffer or long
single-thread reductions. **Savings:** operand traffic and unused GEMM work.
**Contract:** S if existing leaf/tree is reproduced; V for a new documented fold.
**Source:** CNN bias-gradient kernels and host counterpart; reuse existing bias-fold
and ones-cache routes. **Full scope:** full CNN/MLP training.
**Risks:** broadcast axes, batch/spatial tails, cancellation and signed zero.
Record whether a card maps to existing code instead of duplicating it.

### NI16 Fuse convolution activation and residual passes

**A/B:** fuse adjacent output elementwise stages or their backward VJPs versus
separate launches. **Savings:** activation bandwidth and launch overhead.
**Contract:** S; preserve pre/postactivation values required by the tape and exact
ReLU/tie/NaN semantics. **Source:** CNN device blocks and `x_cnn/ops.mojo`.
**Full scope:** CNN blocks and complete CNN training.
**Risks:** shared residual aliases, in-place updates, branch fanout and saved-mask
lifetime. Unsupported multi-consumer lifetimes retain the reference path.

### NI17 CNN pooling and normalization canonical reductions

**A/B:** logical chunked reductions for normalization statistics/backward and
parallel exact pooling selection versus current long chains.
**Savings:** shorten row/channel dependencies.
**Contract:** V for changed sums; S for complete pooling comparator with unchanged
tie order. **Source:** CNN normalization/pooling operators and host counterparts.
**Full scope:** CNN fit and inference; running statistics across several steps.
**Risks:** max-pool tie indices, zero variance, epsilon placement, train/eval mode
and cancellation; output-only accuracy is insufficient to qualify saved state.

### NI18 CNN device epoch preparation

**A/B:** deterministic device permutation/gather and retained batches versus
host-produced epoch order and repeated staging.
**Savings:** training preparation/transfers.
**Contract:** S only if RNG mapping is identical; V for an explicitly versioned
shared shuffle while retaining seed reproducibility and full sample coverage.
**Source:** existing CNN epoch preparation and native training bindings.
**Full scope:** full CNN dataset and multi-epoch fit.
**Risks:** dropped/repeated rows, last batch, resume epoch state and leakage; do not
replace a shuffle with a biased mapping merely to avoid host work.

## Transformer and training lane

### NI19 Reuse attention query key and value tiles

**A/B:** bounded shared/register tiles and grouped-query K/V reuse versus repeated
loads/recomputation. **Savings:** attention bandwidth.
**Contract:** S; preserve score/PV folds and mask order.
**Source:** `transformer/impl/llama/fused_attention.mojo`; reuse I06/A05/N05.
**Full scope:** TR/LM/Samba forward and backward, prefill and decode.
**Risks:** GQA head mapping, causal/padding masks, empty rows, tail tiles, scratch
capacity and full training tape; device schedules must not change softmax sums.

### NI20 Versioned online attention softmax

**A/B:** canonical fixed logical score blocks with online max/denominator/PV merge
versus materialized or existing fold. **Savings:** quadratic state traffic.
**Contract:** V; fixed merge/rescale graph on all columns, portable exp and explicit
all-masked-row behavior; backward follows the same defined forward.
**Source:** attention V2 contract/device/host implementations, not vendor library
SDPA. **Full scope:** LM/TR/Samba training and long prefill.
**Risks:** underflow, sharp logits, mask tails, gradient accuracy, changed loss and
decode/prefill relation; reuse existing versioned attention rather than fork it.

### NI21 Attention save versus recompute by memory budget

**A/B:** retain selected probabilities/statistics/checkpoints or regenerate them
from immutable stages versus current policy.
**Savings:** avoid either excessive writes or expensive recomputation.
**Contract:** S if regenerated values use the same graph.
**Source:** attention tape/session and existing I07 work.
**Full scope:** LM/TR/Samba backward and complete train steps.
**Risks:** peak memory, weight generation, dropout counter replay, long contexts,
multiple live tapes and cold setup; no hidden tape truncation to claim speed.

### NI22 Sparse causal tile scheduling

**A/B:** launch only logically visible causal/local-mask tiles versus compute then
discard masked work. **Savings:** score/value arithmetic on impossible edges.
**Contract:** S where omitted tiles are exactly excluded by the contract; skipping
masked additions of zero may need V if old signed-zero/rounding behavior changes.
**Source:** attention tile descriptors and host mask definition.
**Full scope:** TR/LM/Samba causal attention and irregular mask tails.
**Risks:** dense user masks, all-masked rows, window endpoints, rectangular decode,
zero versus negative-infinity fill and backward dK/dV contribution coverage.

### NI23 Fuse RoPE and attention layout transforms

**A/B:** apply exact rotation while writing the layout consumed by attention
versus separate rotate/transpose/copy passes.
**Savings:** Q/K memory passes.
**Contract:** S; same angle, portable trig, multiply/add order and position offsets.
**Source:** transformer model/RoPE and backward layout kernels.
**Full scope:** TR/LM/Samba prefill, decode and training.
**Risks:** grouped heads, odd/nonrotary channels, cache offsets, in-place aliases,
inverse VJP and checkpoint positions; preserve existing supported shape semantics.

### NI24 Training prefill without unused KV cache writes

**A/B:** explicit no-decode-cache training session versus ordinary prefill append
and cache copies. **Savings:** dead KV writes, allocations and copies.
**Contract:** S; only remove state absent from the caller's supported outputs.
**Source:** transformer model sessions and LM/Samba training handoff.
**Full scope:** full LM/Samba steps; independently preserve decode public callers.
**Risks:** retained backward stages, mixed prefill/decode session use, checkpoint
state and stale handles; do not drop a cache a later public operation can observe.

### NI25 Canonical parallel RMSNorm and LayerNorm

**A/B:** fixed logical lane/chunk trees for statistics and backward dot versus
serial row chains; separately compare S cell-parallel scaling after old statistics.
**Savings:** shorten wide-row dependency and reuse row reads.
**Contract:** V for stats; all columns share variance formulation/FTZ/epsilon/divide.
**Source:** transformer forward/backward norms, training norm and sequence norm
integration coordinated with the sequence lane. **Full scope:** TR/LM/Samba.
**Risks:** cancellation, tiny variance, long rows, gradient norms and training
quality; never set a reduction boundary from a board width.

### NI26 Fuse training SwiGLU and its saved outputs

**A/B:** one pass producing SiLU preactivation state plus gate product, and paired
VJP outputs, versus separate stages.
**Savings:** launch and activation traffic.
**Contract:** S; original multiply/portable sigmoid order and tape values retained.
**Source:** transformer model and backward kernels, existing SWIGLU_FUSED.
**Full scope:** TR/LM/Samba forward/backward and complete training.
**Risks:** extreme activations, aliasing, tape ownership and multi-consumer state;
an inference-only fused output is insufficient for the training candidate.

### NI27 Parameter and gradient arena views

**A/B:** block parameters/gradients alias owned flat arenas versus unpack/pack
copies every step. **Savings:** copies, buffers and launch overhead.
**Contract:** S; bind views after optimizer swaps and preserve exclusive mutation.
**Source:** `training/byte_lm.mojo`, model/layer pools and bindings.
**Full scope:** LM and compatible Samba training.
**Risks:** shadow/committed parameter generations, zero gradients, checkpoint,
failed-step rollback and output survival; qualify NVIDIA/AMD explicitly, not just
reuse an Apple FAST policy with its numerical assumptions.

### NI28 Consolidate status collection transactionally

**A/B:** canonical device status cells and one final collection versus per-stage
host waits. **Savings:** synchronization/launch seams.
**Contract:** S; preserve first error priority/index and commit no invalid step;
shadow state or a supported replay boundary is required.
**Source:** training step, device scan and transformer refusal calls; reuse I10.
**Full scope:** LM/Samba/MLP training with all requested outputs.
**Risks:** invalid loss/gradient/variance, replay mutation, async lifetimes and
hidden synchronization; do not simply remove waits guarding error handling.

### NI29 Fuse optimizer update and state refusal scans

**A/B:** update kernels emit per-block integer first-invalid witnesses while values
are live versus rescanning parameters/moments afterward.
**Savings:** several full-buffer reads and launches.
**Contract:** S; same update statements, same error priority and rollback semantics.
**Source:** optimizer kernels and training step finalization; reuse existing scans.
**Full scope:** LM/Samba/MLP/CNN/SEQ optimizer consumers.
**Risks:** preexisting invalid inputs, negative variance, failure observed one step
late, clipping interaction and master/shadow state; timing includes final status.

### NI30 Batched optimizers and canonical global clipping

**A/B:** descriptor/arena update over all parameters plus a shared canonical norm
fold versus per-tensor launches and repeated clip preparation.
**Savings:** launch and descriptor/upload overhead.
**Contract:** S for updates and existing norm graph; V for a changed clipping fold.
**Source:** `training/checks/optimizer.mojo`, optimizer pool, clip modules.
**Full scope:** every neural training caller using these optimizers.
**Risks:** tensor order, empty tensors, bias-correction step, weight-decay convention,
nonfinite norm, sparse gradients and zero norm; do not parallelize competing arms.

### NI31 Stable segmented embedding gradients

**A/B:** radix-sort (token ID, original position) and fold touched rows versus
vocabulary-wide scan or bitonic sort.
**Savings:** avoid V×T scans and unnecessary untouched rows.
**Contract:** S if within-row original contribution order is retained; V for a
shared balanced segment fold. **Source:** embedding identical/sort and host entry;
reuse I11. **Full scope:** LM/Samba training plus full embedding workloads.
**Risks:** repeated IDs, padding IDs, stable sort, zeroing old gradient rows and
large vocabulary; dense Adam semantics cannot silently become sparse Adam.

### NI32 Reuse token validation and resident batches

**A/B:** validate IDs/targets once at an owned immutable batch boundary and retain
upload buffers versus re-download/recheck at each embedding/loss entry.
**Savings:** transfers and waits.
**Contract:** S; public untrusted callers retain validation; provenance is tied to
buffer contents/generation, not a freely asserted flag.
**Source:** embedding prerefused paths, LM input and loss entrypoints.
**Full scope:** full LM/Samba epoch/step boundaries.
**Risks:** mutable inputs, target shift, last batch, resume offsets and invalid IDs;
include input-step work in end-to-end timing instead of hiding it off the clock.

### NI33 Fused loss intermediates and logits gradient

**A/B:** reuse exp/log-normalizer/weights to produce CE and dlogits versus separate
read/exp passes. **Savings:** logits traffic and portable transcendental work.
**Contract:** S with existing max/denominator/mean folds; V only if explicitly
versioned. **Source:** loss kernels/contracts and LM head; reuse existing I10 arms.
**Full scope:** LM/Samba/MLP/CNN/SEQ full training.
**Risks:** extreme logits, ignore index, label smoothing/weights, all-ignored rows,
target refusal and exact mean denominator; a changed mean is an objective change.

### NI34 Chunked LM head with canonical vocabulary fold

**A/B:** stream vocabulary tiles through logits/loss/backward versus materialize
all logits. **Savings:** activation memory and write/read traffic.
**Contract:** V if the reduction graph changes, with host parity; S when existing
chunked-head V2 contract already defines the same graph. Reuse that implementation.
**Source:** `training/chunked_lm_head_v2.mojo`, pooled head and host loss.
**Full scope:** LM/Samba full training and logits-returning APIs separately.
**Risks:** tied embedding gradient merge, class tails, recomputation costs,
requested full logits, gradient coverage and peak-memory/cold-time accounting.

### NI35 Versioned loss and gradient reduction trees

**A/B:** common fixed logical leaves/balanced trees for token mean loss, parameter
gradients and residual fan-in versus present serial/heterogeneous folds.
**Savings:** remove long dependent chains and enable reusable parallel kernels.
**Contract:** V; each operation pins its own scale, fold order and rounding; no
floating atomics or vendor-dependent reduction primitive.
**Source:** loss, transformer backward, LM host/backward and optimizer interfaces.
**Full scope:** LM/TR/Samba and every shared training consumer.
**Risks:** error cancellation, near-zero gradients, optimizer trajectory and resume;
per-operation gains must survive a complete step with the new numerical version.

### NI36 Live range based activation retention

**A/B:** share disjoint scratch and retain expensive forward stages needed by
backward versus repeated allocation, unnecessary copies or duplicate forward.
**Savings:** memory pressure and recomputation.
**Contract:** S; tape references immutable input/weight generations, deterministic
replay preserves the same arithmetic/RNG.
**Source:** LM/Samba layer pools and transformer backward session ownership.
**Full scope:** LM/Samba trajectories, gradient accumulation and checkpoints.
**Risks:** concurrent tapes, user-visible stage outputs, failed steps, checkpoint
tradeoff and accidental aliasing; use supported buffer ownership, not unsafe caches.

## Mamba and neural sequence lane

### NI37 Parallel causal depthwise convolution

**A/B:** one cell task per batch/token/channel with original tap order versus a
thread looping over the full sequence.
**Savings:** expose independent token work.
**Contract:** S; same causal boundary, tap order, FTZ/FMA and bias/activation.
**Source:** Mamba 1/2 model convolution and host counterpart.
**Full scope:** MM/Samba forward, backward and decode.
**Risks:** carried convolution cache, short/partial sequences, grouped channels,
saved preactivations and reverse gradient coverage; no recurrence is reordered.

### NI38 Versioned Mamba 1 affine chunk scan

**A/B:** fixed absolute-position chunks, canonical affine composition/carry and
re-walk versus a full serial recurrence.
**Savings:** parallelize long sequences without truncation.
**Contract:** V; composition changes floating order. Host/GPU forward, backward,
prefill/decode, state serialization and chunk boundary semantics move together.
**Source:** selective_scan_interface/backward and Mamba host generation; reuse I09.
**Full scope:** Mamba1/Samba full sequences plus long-horizon state quality.
**Risks:** exp decay, underflow, unstable states, arbitrary request chunking and
checkpoint boundaries; length-dependent FAST chunk rules are not a canonical graph.

### NI39 Cache Mamba exponentials and decays

**A/B:** compute exact portable exp/decay once per logical timestep/head and reuse
over output/state channels versus recomputation.
**Savings:** transcendental cost and redundant loads.
**Contract:** S; cache the exact existing expression, not an algebraic replacement.
**Source:** SSD minimal and Mamba3 SISO; reuse existing candidates.
**Full scope:** Mamba2/3/Samba forward/backward.
**Risks:** extra storage, cache reuse boundaries, tails, very long decay intervals,
underflow and matching which consumers use which decay expression.

### NI40 Tile Mamba 2 SSD interactions

**A/B:** shared/register reuse of C×B/decay interactions and bounded intermediate
tiles versus recompute per channel.
**Savings:** repeated products and global loads.
**Contract:** S if each output chain is unchanged; V for an explicitly shared fold.
**Source:** `mamba/impl/modules/ssd_minimal.mojo`; reuse I08.
**Full scope:** Mamba2/Samba full forward and backward.
**Risks:** partial chunks, head groups, register/shared-memory pressure, retained
backward stages and deterministic initial/final states.

### NI41 Skip mathematically unused SSD triangle

**A/B:** compute only consumed causal C×B entries versus full chunk square.
**Savings:** unused pair products and storage.
**Contract:** S for outputs if no backward/trace consumer reads skipped entries;
fill required diagnostic slots deterministically or define the trace version.
**Source:** SSD C×B formation and every backward/tape consumer.
**Full scope:** Mamba2/Samba forward/backward.
**Risks:** off-diagonal reads in gradients, inclusive diagonal, partial chunks,
stage-export API and savings lost to irregular indexing.

### NI42 Tile Mamba 3 interactions on all supported GPUs

**A/B:** reuse query/key×decay tiles on AMD and NVIDIA with resource-based geometry
versus recompute scalar paths. **Savings:** repeated intra-chunk products.
**Contract:** S; same angle and output folds, no vendor-specific arithmetic.
**Source:** `mamba3_siso.mojo`, existing yintra/S16 arms.
**Full scope:** Mamba3/Samba forward/backward.
**Risks:** existing shape-fitted routing, shared-memory budgets and waves; old rule
is B and neighboring/non-board cases are required when replacing it.

### NI43 Versioned linear work Mamba 3 angle gradient

**A/B:** shared canonical reverse suffix scan/carries versus each timestep folding
its whole suffix. **Savings:** remove quadratic repeated angle-gradient work.
**Contract:** V for d_dt/angle gradients; specify reverse leaf/merge order and host
backward counterpart, not a vendor scan primitive.
**Source:** `mamba3_backward.mojo`, theta reverse and host backward.
**Full scope:** Mamba3/Samba full train/backward and long sequences.
**Risks:** suffix endpoint, angle channels, sequence tails, accumulated rounding,
gradient oracle and training quality; forward identity alone proves nothing here.

### NI44 Canonical Mamba parameter gradient folds

**A/B:** logical token chunks and shared merge for convolution/dt/projection
parameter gradients versus serial per-parameter chains.
**Savings:** parallelize long reductions and reuse per-token work.
**Contract:** V; fixed graph for all columns and batch/sequence traversal.
**Source:** Mamba2 SSD backward, Mamba1/3 backward and host counterparts.
**Full scope:** MM/Samba full training/backward.
**Risks:** cancellation, grouped dimensions, multiple gradient terms, padding and
optimizer trajectory. Separate per-cell scheduling-only S arms where possible.

### NI45 Mamba scratch arenas and generation keyed weights

**A/B:** reuse owned arena capacity and optimizer generation stamps versus repeated
allocation, byte comparisons and uploads.
**Savings:** preparation, transfers and waits.
**Contract:** S; only valid immutable/canonical weight generations are reused.
**Source:** Mamba arenas, native binding sessions and existing IDN_MAMBA switches.
**Full scope:** Mamba1/2/3/Samba repeated forward/backward/train.
**Risks:** external weight mutation, shape changes, shared sessions, failed calls,
state cache and memory release; report cold setup and repeated use separately.

### NI46 Fuse Mamba elementwise state preparation

**A/B:** fuse exact bias/softplus/decay preparation or output gate/residual passes
versus materialize each intermediate.
**Savings:** launches and elementwise traffic.
**Contract:** S; retain portable exp/log/softplus order and required saved tensors.
**Source:** Mamba model and SISO/selective-scan launch seams.
**Full scope:** MM/Samba forward/backward, prefill and decode.
**Risks:** extreme dt, missing stage exports, duplicated recomputation in backward,
live range overlap and altered FMA contraction; keep explicit arithmetic statements.

### NI47 State space chunk boundary contract

**A/B:** stable logical chunk scheduling and reused boundary carries versus repeated
whole-sequence setup or length-dependent chunk layout.
**Savings:** reuse prefix state and avoid redundant setup for streamed calls.
**Contract:** S only if current graph is preserved; V for a new absolute-position
graph. **Source:** Mamba prefill/decode/session and checkpoint contracts.
**Full scope:** MM/Samba full prefill, streaming decode and resumed training.
**Risks:** user call boundaries, final-state bits, cache offsets, padding and
batch reorder; state updates must remain coherent across vendors within a version.

### NI48 Reuse Samba forward stages in backward

**A/B:** retain owned attention/Mamba stages from the actual forward versus rerun
forward from a fresh cache during backward.
**Savings:** duplicate forward projections and attention/scan work.
**Contract:** S; stages tied to exact inputs, parameter generation and cache state.
**Source:** Samba native handoff, training/samba_ops and transformer/Mamba sessions;
coordinate with NI36. **Full scope:** full Samba train trajectories.
**Risks:** mutable inputs, memory pressure, alternate layer types, tape fanout,
checkpoint/recompute and failed-step rollback; integrate native code, not Python
data comparisons or tensor work.

### NI49 Repair recurrent persistent scan before timing

**A/B:** repaired one-block-per-independent-row recurrence versus per-timestep
launches. **Savings:** T forward/backward launches become bounded scans.
**Contract:** S if all cell statements, timestep dependencies and barriers match.
**Source:** `sequence/recurrent_scan.mojo` and recurrent/host executors.
**Full scope:** full LSTM/GRU/RNN classification/regression training.
**Known risk:** source comments report constant-prediction training failure for
existing SCAN arms. Keep them disabled and explicitly unqualified; diagnose source
and retain old failure evidence, do not label the old arm ready because it is fast.

### NI50 Tiled same chain neural sequence GEMM

**A/B:** shared-memory operand reuse with the same ascending per-cell dot versus
unshared executor GEMM. **Savings:** repeated operand loads.
**Contract:** S for the sequence operation's own graph; do not substitute another
GEMM graph merely because it is also called IDENTICAL.
**Source:** `sequence/gemm_tiled.mojo`, neural executor/callers; reuse existing arms.
**Full scope:** LSTM/GRU/RNN, neural MLP and MoE only.
**Risks:** strided views, output aliasing, transpose forms and shared executor reach;
avoid expanding the experiment to non-neural statistical estimators.

### NI51 Versioned recurrent gradients and LayerNorm folds

**A/B:** logical chunk/tree reductions for timestep×batch weight/bias gradients
and neural sequence LayerNorm versus long serial folds.
**Savings:** parallelize training reduction bottlenecks.
**Contract:** V; host/device executor share the same leaves, ordering and rounding.
**Source:** sequence recurrent/layernorm/ops and host executor.
**Full scope:** full neural recurrent/MLP training.
**Risks:** masking and last sequence, gradient accumulation, normalization epsilon,
multiple steps and all supported recurrent gates; keep non-neural callsites intact.

### NI52 Neural MLP device epoch shuffle and batches

**A/B:** Mojo device-defined seeded permutation and retained batch buffers versus
host row shuffling and uploads per epoch.
**Savings:** setup/transfer per epoch.
**Contract:** S for exact existing random mapping; V for a declared shared new
permutation with complete row coverage and resume state.
**Source:** `sequence/mlp_fit.mojo` and host training counterpart.
**Full scope:** full neural MLP multi-epoch fit.
**Risks:** tail batches, reproducibility, convergence and train/validation leakage;
runtime sampling must not move into Python.

### NI53 MoE stable routing and grouped expert execution

**A/B:** integer-key stable token/expert packing and bounded grouped expert GEMMs
versus separate scatter/launches. **Savings:** dispatch overhead and sparse fill.
**Contract:** S for stable routing and unchanged expert/fan-in order; V only for
a separately defined canonical combine fold.
**Source:** `sequence/moe*.mojo`, grouped/tiled expert kernels and host executor.
**Full scope:** full neural MoE forward/backward/training recipes, pending if absent.
**Risks:** equal router scores, capacity/overflow policy, empty experts, repeated
tokens and gate gradients; never drop routed tokens for performance.

### NI54 Adafactor tiled and versioned factored statistics

**A/B:** tiled same-order row/column moment reads, then separately fixed canonical
parallel folds versus current serial factor reductions.
**Savings:** memory reuse and shorter optimizer dependency chains.
**Contract:** S for tiled reads; V for new mean/RMS folds and shared host parity.
**Source:** `sequence/adafactor.mojo`, optimizer executor and host counterparts.
**Full scope:** neural training callers using Adafactor; full recipe may be pending.
**Risks:** epsilon placement, relative-step/clipping semantics, zeros, rectangular
parameters, step state and training convergence; no silently changed optimizer.

## Graph neural and dropout additions

The full-workload source map also contains graph neural and channel-dropout
callers. These additional cards are defined before their root-owned programming.
They operate on GCN/GraphSAGE tensors, not graph clustering or neighbor search.

### NI55 Graph neural feature register tiles

**A/B:** one task accumulates four neighboring feature outputs while sharing CSR
edge/weight reads versus one independent task per output.
**Savings:** repeated graph metadata and weight loads.
**Contract:** S; each output visits the same CSR entries in the same order.
**Source:** `x_cnn/ops.mojo` spmm and device graph launchers.
**Full scope:** GCN/GraphSAGE forward/backward and full neural consumers.
**Risks:** feature tails, empty rows, mean normalization and backward transpose;
four registers is a fixed resource choice, not a board feature-count selector.

### NI56 Reuse graph normalization workspace

**A/B:** retain buffer capacities for edge weights, CSR, degrees and normalized
weights versus allocate them for every normalization call.
**Savings:** allocation and setup overhead.
**Contract:** S; recompute all values for each supplied graph, with no address-only
cache of mutable data. **Source:** `x_cnn/device.mojo` gcn_norm_device.
**Full scope:** GCN preparation plus full forward/backward consumers.
**Risks:** changed topology, new self loops, graph sizes, zero edges and lifetime;
include every graph preparation pass in timing, not only propagation.

### NI57 Versioned graph neural edge reduction

**A/B:** fixed 64-edge logical leaves and adjacent balanced merge versus one full
CSR-row sum. **Savings:** an explicit shorter-dependency graph available for later
parallel leaf scheduling; the initial scalar-stack arm may cost more.
**Contract:** V; share exact FTZ/mul/div and odd-tail tree across host and GPUs.
**Source:** common x_cnn message-passing element function and host/device callers.
**Full scope:** GCN/GraphSAGE sum/mean forward and transposed backward.
**Risks:** high degree, cancellation, duplicate edges, empty rows, graph normalization
and training quality. Do not claim tree notation alone makes the kernel parallel.

### NI58 GraphSAGE max feature tiling

**A/B:** process four feature maxima/counts together per CSR row versus separate
metadata traversals. **Savings:** repeated row/column index reads.
**Contract:** S; original ascending edge order, NaN handling and equal-maximum
counts, so backward retains its exact equal-split semantics.
**Source:** GraphSAGE max element function and graph dispatch.
**Full scope:** GraphSAGE max aggregation and requested backward outputs.
**Risks:** equal maxima, signed zeros, NaNs, empty rows and feature tails; never
replace equal-gradient sharing with a single arbitrary winning edge.

### NI59 One launch channel dropout

**A/B:** one block per channel draws the existing counter mask once, broadcasts it
and writes output/mask versus separate mask-table and apply launches.
**Savings:** one launch and an intermediate table.
**Contract:** S; same Philox channel counter, integer threshold and scaling.
**Source:** dropout2d device entry and shared mask function.
**Full scope:** full channel dropout forward/backward and CNN training consumers.
**Risks:** low channel count, long spatial planes, tails and supported seed offsets;
preserve the full observable mask ABI and keep elementwise dropout semantics distinct.

### NI60 Tile dropout mask application

**A/B:** apply a cached channel mask to four contiguous spatial values per task
versus independent per-element loads/division.
**Savings:** repeated mask loads and address decoding, independently of NI59.
**Contract:** S; original per-element FTZ/multiply and complete mask output.
**Source:** dropout apply element function and channel-table path.
**Full scope:** channel dropout and actual full CNN consumers.
**Risks:** partial spatial tiles, no cross-channel tail reads, occupancy and wider
register use. NI59 and NI60 are alternative launch schedules, not additive flags.

## Required interaction experiments

Individual gains do not establish the complete configuration. Later measure:

1. NI01/02/03/04/05/06 on complete neural GEMM schedules; include baseline,
   individually changed arms and the chosen combined profile.
2. NI07/09 with NI26/27/36: projections, epilogues, views and activation ownership.
3. NI10/11/12/13/14/15/16: full CNN preparation/forward/backward/optimizer boundary;
   direct, implicit and retained-columns policies are alternatives where exclusive.
4. NI19/20/21/22/23/24: attention numerical version, masks, tape and KV lifetime.
5. NI25/28/29/30/33/35: full training reductions, clipping, refusal and optimizer.
6. NI31/32/34: embedding updates, resident tokens and chunked tied LM head.
7. NI37/38/39/40/41/42/43/44/46/47: each complete Mamba model and its backward;
   isolate which numerical contracts changed instead of one opaque all-on flag.
8. NI36/45/48: full Samba ownership, forward reuse and checkpoint/recovery behavior.
9. NI49/50/51/52/53/54: relevant recurrent, MLP, MoE and optimizer combinations;
   a known failed scan cannot enter the accepted combination unchanged.
10. Any adopted NI08 GEMM numerical version with every dependent training version,
    on all affected neural estimators and every column. Preserve versions in
    receipts/checkpoints or explicitly reject incompatible continuation.
11. NI55/56/57/58 on full graph-neural workloads, and NI59 versus NI60 with complete
    CNN consumers. Keep scalar version-contract experiments separate from claims
    of parallel throughput; record absence of full training recipes as pending.

## Implementation and evidence status

The complete idea list precedes implementation delegation. Program in three
disjoint lanes: NI01–NI18 GEMM/CNN; NI19–NI36 transformer/training/embedding;
NI37–NI54 Mamba/neural sequence. Root owns NI55–NI60 after finding those neural
surfaces in saved recipes. Coordinate shared files explicitly. Reuse existing
opt-in runtime arms, add real disabled candidate paths where supported, and record
an honest per-card status. A selector that does not reach different code is not
an implementation; an unintegrated helper is not a complete public caller.

Each lane writes arm controls, paths, dependencies and unresolved coverage under
`experiments/neural_identical_20261006/`. Preserve known failures and make missing
host parity/integration explicit. Do not manufacture result/compile receipts.
No lane compiles, tests, lints, imports candidate modules, runs manifest checkers,
verifies, measures, promotes, merges or pushes during this task. Reading existing
source/evidence and editing are allowed; no commands that execute a candidate.

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output. Propagate this
reminder to any delegated agents.

## Prior work and limitations

Reuse `docs/plans/IDENTICAL_NEURAL_ROADMAP.md`, neural cards under
`experiments/performance_ideas/`, the attention/GEMM/Mamba/loss/optimizer contracts,
and exact source switch comments. Older roadmap statements preserving old folds
are not a constraint against this owner's explicit permission to change version
bits. Historical component evidence remains component evidence.

`tools/neural_experiments.py` is an older same-bits diagnostic runner: its blanket
rejection of a changed baseline digest cannot judge V experiments. New arm
metadata must state the numerical contract and same-version cross-column rule;
do not loosen the older tool's checks indiscriminately or pretend its small
stage timings complete the full-dataset campaign.

No numerical speedup factor is promised. Quality preservation and cross-vendor
identity remain pending for all new source until a separately requested frozen
measurement/qualification round is performed.

The owner subsequently requested source integration, commit and push. The
integration delivery is recorded in
[`IMPLEMENTATION_STATUS.md`](../../experiments/neural_identical_20261006/IMPLEMENTATION_STATUS.md):
model callers, shared host arithmetic, numerical profiles, selectable variants,
the existing frozen native builder and the full-operation A/B queue. This expands
source delivery only; it does not authorize compilation or model verification.
The root performs commit/push with normal repository policy hooks. New candidates
remain OFF, and missing qualification and broader unfinished variants stay visible.
