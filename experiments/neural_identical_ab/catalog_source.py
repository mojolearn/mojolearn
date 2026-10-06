"""Write the neural-only design inventory. No builds, checks, or ML execution."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CARDS = []


def card(n, title, callers, paths, candidate, control, identity, quality, reject,
         prior="New extension; inspect existing switches before implementation.", kind="schedule", priority="P1"):
    lane = "gemm" if n <= 16 else "attention" if n <= 32 else "state_cnn" if n <= 48 else "training"
    CARDS.append(dict(id=f"NN{n:02d}", title=title, lane=lane, callers=callers.split("; "),
                      source_anchors=paths.split("; "), arm_a=candidate, arm_b=control,
                      identity_contract=identity, quality_gate=quality, rejection_conditions=reject,
                      prior_work=prior, kind=kind, priority=priority, mode="identical",
                      default_enabled=False, implementation="planned", compilation="not_run_by_request",
                      verification="not_run_by_request", measurements="not_run_by_request"))


card(1, "Attribute GEMM schedules at real neural callers", "LM; transformer; Mamba/Samba; MLP; CNN",
     "gemm/checks/gemm_identical.mojo; training/dev_tensors.mojo",
     "Select one existing stepped MFMA/kpack/body-tile schedule at a time and expose its route for full caller A/B.",
     "Frozen incumbent GEMM dispatch with every unrelated switch unchanged.",
     "Logical products, leaf boundaries, FMA/FTZ and fold stay fixed; physical tile choice may differ by vendor.",
     "Full forward, gradients, optimizer state and model loss/quality across shapes and tails.",
     "Do not repeat decided arms unchanged or claim kernel timing establishes train-step benefit.",
     "Reuse I01/A01/N01 and retained outcomes; caller coverage and attribution first.", priority="P0")
card(2, "Stream GEMM partial planes into a bounded fold", "Neural GEMM; forward/backward linear layers",
     "gemm/checks/gemm_identical.mojo; gemm/experiments/bounded_workspace.mojo",
     "Use bounded groups of canonical leaf partials with an explicit streaming adjacent-pair stack.",
     "Materialize the incumbent full partial plane before its fold.",
     "The exact logical tree including odd carries and +0 initialization is unchanged by streaming.",
     "Contraction words and all callers' full forward/backward outputs and task metrics.",
     "Reject spills, extra launches, in-flight aliasing or a stack that depends on physical grid size.",
     "Extends I02/N02; distinguish live storage from a revised arithmetic tree.", priority="P0")
card(3, "Versioned GEMM leaf lengths", "FP32 neural GEMM consumers",
     "gemm/contract.mojo; gemm/host/gemm_oracle.mojo; gemm/checks/gemm_identical.mojo",
     "Separate canonical 64/128/256-term leaf experiments with the same specified within-leaf FMA and pair tree.",
     "Incumbent contraction profile for the same inputs and public model.",
     "A new version may change bits; each arm has a complete common host/device profile, portable FMA and fixed tails. Leaf selection cannot depend on vendor.",
     "Conditioned error, neural loss/perplexity, forward and gradient quality, finite state and full training trajectories.",
     "Reject no credible throughput mechanism, precision reduction or incomplete migration of a caller/oracle.",
     "I04 has only prerequisite scalar probes; do not call them a device implementation.", "profile", "P2")
card(4, "Versioned independent accumulator chains inside GEMM leaves", "FP32 neural projections and weight gradients",
     "gemm/contract.mojo; gemm/host/identical_gemm.mojo; gemm/checks/gemm_identical.mojo",
     "Use a specified small number of interleaved FMA chains per logical leaf, then combine them in a fixed tree.",
     "One dependent ascending FMA chain per incumbent leaf.",
     "Lane membership, initial zero, FMA and final merge are numerical profile data shared across vendors, independent of warp width.",
     "Cancellation/adversarial errors, forward/backward/optimizer state and task quality; all dtype contracts remain unchanged.",
     "Reject extra registers or quality loss; no native matrix instruction with unknown internal order may stand in for the declared chain.", kind="profile", priority="P2")
card(5, "Group independent projection jobs", "Q/K/V; gate/up; independent dX/dW products; Samba",
     "gemm/experiments/grouped_jobs.mojo; training/dev_tensors.mojo; transformer/impl/llama/modeling_llama.mojo",
     "Launch separate logical products as grid jobs, sharing read-only operand staging when compatible; attribute grouping and sharing independently.",
     "Independent GEMM launches with the same operands.",
     "Each contraction's k, leaves, rounded products and output tree remain unchanged; concatenating backward contractions is excluded.",
     "Every projection and gradient, complete layer/step quality and memory lifetime.",
     "Reject descriptor overhead or extra scratch/serialization at small batches.", "Reuse I03; new caller combinations need their own reach mapping.", priority="P0")
card(6, "Rounded neural GEMM epilogues", "MLP; transformer projections; CNN dense layers",
     "gemm/experiments/rounded_epilogue.mojo; transformer/impl/llama/modeling_llama.mojo; training/mlp_ops.mojo",
     "Independently fuse bias, residual, scaling or activation output passes into the producer with explicit stored-value rounding seams.",
     "GEMM writes followed by separate pointwise kernels.",
     "FMA formation cannot cross an incumbent materialization boundary; preserve the exact portable activation and FTZ order.",
     "Intermediate/final words, activation derivatives and full step/model quality.",
     "Reject register pressure slowing GEMM or fusion of a nonlinear operation without its backward contract.", "Extend I05 separately for each epilogue/caller.", priority="P0")
card(7, "Reuse transposed operand staging across neural jobs", "Weight gradients; repeated linear layers; attention projections",
     "gemm/checks/gemm_identical.mojo; training/dev_tensors.mojo",
     "Stage immutable strided/transposed operand tiles once for several independent consumers, keeping distinct outputs.",
     "Each consumer repeats the same gather/staging.",
     "No change to each output's product order or required strided-load rounding; cached weights invalidate by owner/version.",
     "All transpose modes, ragged strides, repeated calls, gradients and model quality.",
     "Reject materialization more expensive than strided access or stale transpose caches.", "N04 supplies transpose controls; extend only new sharing opportunities.")
card(8, "Bounded asynchronous operand loading", "Neural GEMM forward/backward",
     "gemm/experiments/async_operand_pipeline.mojo; gemm/experiments/async_api_probe.mojo",
     "Overlap the next canonical operand tile load with current arithmetic using supported Mojo pipeline primitives.",
     "Synchronous loading of the same tiles.",
     "Arithmetic order identical; barriers and page ownership explicit, no use-before-ready.",
     "Every GEMM/gradient output, tail masks and complete neural operation quality.",
     "If a required primitive is unsupported, record a Modular ask; no compiler patch, assembly rewrite or unsupported build mode.", "Reuse N03 capability evidence where compiler/target match.", priority="P2")
card(9, "Shared-memory page count and bank layout", "AMD/NVIDIA neural GEMM",
     "gemm/experiments/bounded_staging.mojo; gemm/checks/gemm_identical.mojo",
     "Separate one-page/two-page/padded or swizzled staging arms derived from shared bytes and supported bank access.",
     "Incumbent page count and operand layout.",
     "Address permutation only, preserving every logical operand and arithmetic order; document invertible layouts and tails.",
     "Contraction and model outputs plus resource footprint on both voting vendors.",
     "Reject additional address arithmetic, register pressure or a layout valid only for a board dimension.", "A02/A03 already cover baseline variants; add only clearly distinct arms.")
card(10, "Neural GEMM cost-based dispatch", "All affected neural GEMM callers",
     "gemm/checks/gemm_identical.mojo; gemm/contract.mojo",
     "Replace any identified shape-fitted rule with a byte/work/device-fill rule valid for neighboring shapes; each removal is its own arm.",
     "Explicitly retain the old route rule as the designated B arm.",
     "The numerical contract is independent of route; if removing a rule changes its profile, all columns move together.",
     "Full affected models, adjacent shapes and at least one non-board dataset, including every fallback.",
     "Do not invent hardware properties if Mojo cannot report them; park that rule and request support.", "Roadmap A8 and AGENTS.md no-dimension-targeting requirements.", priority="P0")
card(11, "Specialize fold storage to the logical tree", "Neural GEMM folds",
     "gemm/checks/gemm_identical.mojo; gemm/experiments/bounded_workspace.mojo",
     "Allocate only the mathematically required tree levels/active slots for each supported profile; separate register and shared stack arms.",
     "Maximum-capacity fold storage at every contraction.",
     "Capacity follows the logical reduction depth, never a dataset or arbitrary nearby dimension.",
     "Tiny/huge contractions and ragged partial trees, full gradients and quality.",
     "Reject excessive specialization count or stack overflow hidden by fallback.", "N02 has prior stack-capacity work; reuse and extend actual uncovered callers.")
card(12, "Model-owned GEMM plan and workspace reuse", "LM; transformer; Samba; MLP; CNN",
     "training/dev_tensors.mojo; training/byte_lm_model_pool.mojo; gemm/checks/gemm_identical.mojo",
     "Retain immutable plans and bounded scratch by device/context, profile, operation and strides, with explicit generation invalidation.",
     "Re-plan and provision temporaries per product/call.",
     "Scheduling/lifetime only; prevent in-flight reuse, cross-session sharing and stale flags/strides.",
     "Cold/repeated calls, growing/shrinking workloads, destruction/error paths, model output and gradients.",
     "Reject unbounded high-water retention or excluding first-use allocation from cold timing.", "Extend I02 and existing workspaces; no global mutable cache.", priority="P0")
card(13, "Tile sequence-family same-chain GEMM", "RNN/LSTM/GRU; sequence MLP; MoE",
     "sequence/gemm_tiled.mojo; sequence/ops.mojo; sequence/exec_device.mojo",
     "Tile strided operands while preserving each output's original scalar contraction chain; batch independent cells.",
     "Per-output serial dot with repeated operand loads.",
     "The sequence family's numerical profile is authoritative; do not silently substitute the main GEMM profile.",
     "All forward/backward recurrent state and optimizer words, strided tails and training quality.",
     "Reject shared-memory costs or changing the fold while claiming schedule-only.", "Roadmap D1 and existing tiled executor are source anchors.", priority="P0")
card(14, "Group convolution-lowered GEMMs with bounded im2col", "Conv1d/2d; CNNClassifier; ResNet blocks",
     "x_cnn/device.mojo; x_cnn/ops.mojo; gemm/experiments/grouped_jobs.mojo",
     "Build bounded patches for independent spatial tiles and consume them immediately in grouped exact products.",
     "Materialize/reload full im2col or launch each patch group separately.",
     "Channel/kernel/tap product order, padding, dilation and grouping semantics fixed.",
     "Forward/dInput/dWeight/dBias, receptive-field boundaries and complete CNN quality.",
     "Reject repeated patch generation or smaller measured image batches masquerading as an improvement.")
card(15, "Hardware-specific schedules under one profile", "AMD/NVIDIA neural GEMM",
     "gemm/checks/gemm_identical.mojo; gemm/experiments/subwave_membership.mojo",
     "Map independent logical groups to physical waves/warps and use already supported scalar-equivalent instruction bodies.",
     "Incumbent physical mapping of the same logical contractions.",
     "Neither physical wave width nor instruction grouping defines FP summation order; operand membership must be explicit.",
     "Four-column words, full neural quality and vendor-specific whole-workload timings.",
     "Reject undocumented native instruction arithmetic, subgroup cross-talk or a material slowdown on either voting vendor.", "Reuse A04/A05 existing probes; not an AMD portable-toolchain workaround.", priority="P2")
card(16, "Avoid unnecessary zero-tail and scratch passes", "Neural GEMM and tensor preparation",
     "gemm/checks/gemm_identical.mojo; core/device_zero.mojo; training/dev_tensors.mojo",
     "Have a producer fully initialize its owned scratch/output region, fusing required tail zeroing with useful stores.",
     "Clear full buffers separately before producers overwrite them.",
     "Only prove-away cells never read before write; explicitly handle signed zero, nonfinite inputs, padding and reuse.",
     "Poisoned workspace, alternating shapes, all recorded stages and downstream models.",
     "Reject undefined cells or suppressing arithmetic on zeros where NaN/Inf semantics require it.", priority="P0")

card(17, "Share attention K/V tiles across query heads", "GQA/MQA/MHA; transformer; LM; Samba attention",
     "transformer/impl/llama/fused_attention.mojo; transformer/impl/llama/attention_v2.mojo",
     "Extend supported compatible-head/query-tile sharing with bounded query accumulators, one geometry per arm.",
     "Each query-head group reloads the same K/V tiles.",
     "Canonical score, max, denominator, output and gradient order unchanged for each query.",
     "Causal/full masks, GQA ratios, sequence tails, gradients and full model quality.",
     "Reject shared memory/extra barriers erasing reuse or assuming a head ratio exists on all inputs.", "I06 already has two-head GQA reuse; extend new schedules, preserve retained rejection evidence.", priority="P0")
card(18, "Skip structurally masked attention tiles", "Causal attention prefill and training",
     "transformer/impl/llama/fused_attention.mojo; transformer/impl/llama/attention_v2.mojo",
     "Omit fully masked tile tasks and bound partial-tile work while preserving public/trace outputs and backward masks.",
     "Schedule full rectangular score tiles then mask.",
     "Masked values' logical contribution and initialization remain explicit; all-masked row behavior unchanged.",
     "Prefix/causal boundaries, all-masked rows, dQ/dK/dV, trace and checkpoint semantics.",
     "Reject omission of a recorded/backward-consumed value or a changed -Inf/NaN convention.", priority="P0")
card(19, "Retain versus recompute attention intermediates", "Transformer/LM/Samba training",
     "transformer/impl/llama/fused_attention.mojo; transformer/checks/transformer_backward.mojo",
     "Independently retain probabilities, exponent state or only canonical row summaries under an explicit storage/recompute budget.",
     "Incumbent stored/recomputed state choice.",
     "Recomputation repeats identical operations and RNG coordinates; owner/generation prevents backward using another forward's state.",
     "Complete step, gradients, checkpoint replay and usable batch capacity, repeated/mixed lengths.",
     "Reject a backward-only win that slows forward+backward or reduces available batch size.", "Reuse I07 cost-model and checked lifetime APIs.", priority="P0")
card(20, "Versioned streaming stable attention softmax", "Attention forward/backward; full transformer and LM",
     "transformer/impl/llama/fused_attention.mojo; transformer/host/transformer_block_host.mojo; transformer/checks/transformer_backward_oracle.mojo",
     "Combine fixed key tiles as canonical (max,scaled_sum,weighted_value) summaries with a prescribed merge tree.",
     "Incumbent multi-pass canonical max/exponent/denominator/value evaluation.",
     "New arithmetic version on every column, with exact rescaling, portable exp/div, masked/empty tiles, tails and backward/checkpoint definitions.",
     "Attention error, gradient quality, full training loss/perplexity and multi-step state; extreme logits.",
     "Reject quality loss or unimplemented host/backward migration; hardware-native FlashAttention is not an identity proof.", "A separate attention_v2 implementation already exists with retained nonpromotion evidence; it does not establish this new balanced-summary-tree proposal. Preserve that control and its limitations.", "profile", "P2")
card(21, "Fuse attention pointwise score transforms", "Scaled/masked/biased attention",
     "transformer/impl/llama/fused_attention.mojo; transformer/impl/llama/modeling_llama.mojo",
     "Apply prescribed scale, mask and supported bias at score production before its existing reduction.",
     "Write scores then run separate transforms.",
     "Preserve operation order, intermediate rounding, causal positions and special mask values.",
     "Every transformed score/softmax output, gradients and model quality, including partial tiles.",
     "Reject new contraction across a rounding seam or dropping requested attention outputs.", priority="P0")
card(22, "Canonical dK/dV task geometry", "Attention backward; GQA/MQA",
     "transformer/impl/llama/fused_attention.mojo; transformer/checks/transformer_backward.mojo",
     "Share useful dK/dV input tiles among independent key/feature tasks while merging the same query partials in order.",
     "Incumbent gradient grid and separate input loads.",
     "No floating atomics; each gradient's query/head contribution order is fixed regardless of scheduler.",
     "All dQ/dK/dV and upstream parameter gradients, long sequences, GQA and accumulation tails.",
     "Reject revival of cooperative/stacked variants with recorded losses unless the changed mechanism is explicit.", "I06/N06 retain earlier backward rejection evidence; new geometry must be distinct.")
card(23, "Fuse attention backward row dot and pointwise gradients", "Attention backward",
     "transformer/impl/llama/fused_attention.mojo; transformer/checks/transformer_backward.mojo",
     "Reuse probability/dOutput tiles while producing fixed-order row-dot leaves and their gradient consumers.",
     "Separate full-array row-dot and pointwise gradient passes.",
     "The row-dot's reduction contract and dependencies complete before the corresponding consumer; dropout mask unchanged.",
     "All softmax/attention gradients, finite behavior and full step quality.",
     "Reject replicated reduction work or hidden global synchronization assumptions.")
card(24, "Versioned RMSNorm/LayerNorm fixed-lane reductions", "Transformer; Mamba/Samba; MLP; sequence LayerNorm",
     "transformer/impl/llama/modeling_llama.mojo; transformer/checks/transformer_backward.mojo; sequence/layernorm.mojo",
     "Use a fixed logical lane count and pair tree for norm/mean/variance and backward dots, independent of hardware width.",
     "Incumbent sequential or existing pinned row fold.",
     "New common version: epsilon placement, centered variance definition, FTZ/sqrt/div and derivative folds explicit; host adopts same graph.",
     "Forward/gradient errors, train loss/perplexity, constant/extreme/canceling rows and narrow/tail widths.",
     "Reject raw-moment variance cancellation, changed epsilon or hardware-shaped subgroup reductions.", "Roadmap C4/B11; schedule-only row scaling is a separate arm.", "profile", "P1")
card(25, "Separate norm scalar fold from parallel cell scaling", "RMSNorm/LayerNorm forward/backward",
     "transformer/impl/llama/modeling_llama.mojo; transformer/checks/transformer_backward.mojo; sequence/layernorm.mojo",
     "Retain the incumbent row-scalar fold and parallelize independent normalization/affine/residual cells.",
     "One row thread performs both reduction and all output cells.",
     "Scalar result and each cell's arithmetic exactly retained; no newly fused FMA across seams.",
     "Norm outputs, residual state, parameter/input gradients and training quality.",
     "Reject launch overhead at narrow rows or incorrect broadcast lifetime.", "Distinct schedule counterpart to NN24.", priority="P0")
card(26, "Training SwiGLU forward/backward pass fusion", "Transformer FFN; LM; Samba",
     "transformer/impl/llama/modeling_llama.mojo; transformer/checks/transformer_backward.mojo",
     "Compute SiLU and gate product together, optionally retain derivative-needed state; fuse paired backward pointwise outputs.",
     "Separate activation/product/derivative passes.",
     "Portable sigmoid/exp and original derivative rounding sequence maintained; distinct saved versus recomputed arms.",
     "All activation/gradient words, extremes and full step loss.",
     "Reject saved-state memory exceeding recomputation savings or missing training outputs.", "Existing forward fusion does not establish training fusion.", priority="P0")
card(27, "Session-owned RoPE frequency and position state", "Transformer prefill/decode; LM; Samba attention",
     "transformer/impl/llama/modeling_llama.mojo; transformer/host/transformer_block_host.mojo",
     "Retain validated immutable frequency data and share portable sin/cos position tiles across independent layers/heads.",
     "Repeated frequency scan/position transform work each layer/call.",
     "Exact position, base/scaling, portable trig and rotation pair order; cache keys include owner/config/generation.",
     "Prefill/decode agreement, long positions, reset/resume and gradients.",
     "Reject stale config/positions, approximate trig or cross-session reuse.", priority="P0")
card(28, "Remove training-only dead KV-cache writes", "Transformer/LM/Samba training prefill",
     "transformer/impl/llama/modeling_llama.mojo; training/byte_lm.mojo",
     "Explicit training caller capability omits KV-cache append/copy operations whose buffers have no training consumer.",
     "Populate decode-style KV state during every training prefill.",
     "Only dead storage disappears; recorded stages, backward and optional requested cache outputs still execute.",
     "Training gradients/loss and decode callers' untouched contract; alternate training and inference on supported owners.",
     "Reject inference cache breakage or proving deadness only from one fixture.", "Roadmap C8; inspect existing switches before extending.", priority="P0")
card(29, "Bounded decode KV layout and append fusion", "Autoregressive transformer/LM/Samba inference",
     "transformer/impl/llama/modeling_llama.mojo; transformer/impl/llama/fused_attention.mojo",
     "Fuse new K/V writes with canonical cache layout conversion and batch independent heads/requests under explicit capacity.",
     "Separate append/copy/layout kernels.",
     "No cache quantization, token reordering or changed attention fold; preserve prefix lengths, capacity/refusal and reset semantics.",
     "Every decode-step logits and cache words, prefill/decode agreement and long context quality.",
     "Reject unsupported paging primitives or moving cache allocation outside cold latency.")
card(30, "Backward residual and gradient buffer views", "Transformer/LM training",
     "transformer/checks/transformer_backward.mojo; training/byte_lm.mojo",
     "Alias explicitly immutable incoming residual-gradient views or transfer ownership rather than copy whole tensors.",
     "Copy input gradients into identical temporary buffers at every block boundary.",
     "Ownership and lifetime part of API; preserve caller buffers, accumulation semantics and asynchronous use.",
     "Repeated backward, accumulation, shared tensors, checkpoint replay and all model gradients.",
     "Reject aliasing in-place consumers or assuming ownership from pointer equality.", priority="P0")
card(31, "Bounded model activation checkpoint policy", "Transformer/LM/Samba training",
     "training/byte_lm_layer_pool.mojo; training/byte_lm_offload.mojo; transformer/checks/transformer_backward.mojo",
     "Compare explicit layer groups of stored versus recomputed activation state from a byte/recompute model, with full cold and repeated steps.",
     "Incumbent activation retention/replay schedule.",
     "Same forward/RNG/replay graph, checkpoints include normalization/position/cache metadata; no reduced sequence/batch length.",
     "Full gradients, state, multi-step loss and memory-capacity behavior.",
     "Reject offload dependence on unsupported async primitives or a recompute policy chosen by benchmark row.", "Related I07 state tradeoff expanded to complete model layers.", priority="P2")
card(32, "Retain Samba attention forward state for backward", "Samba training",
     "training/samba_ops.mojo; transformer/impl/llama/modeling_llama.mojo; transformer/checks/transformer_backward.mojo",
     "Pass an explicit owner/generation-tagged saved-forward handle into backward instead of recomputing from an empty cache.",
     "Re-run identical attention forward during the training backward handoff.",
     "Handle includes inputs/weights/config versions; invalidate on mutation and preserve deliberate checkpoint mode.",
     "Complete Samba gradients/loss, alternating sessions, weight updates and failure cleanup.",
     "Reject stale state, Python runtime comparison of input arrays or unbounded retained activations.", "Roadmap B4; lifetime extension beyond an attention-only benchmark.", priority="P0")

card(33, "Attribute SSD/SISO shared tile reuse", "Mamba-2; Mamba-3; Samba",
     "mamba/impl/modules/ssd_minimal.mojo; mamba/impl/ops/mamba3_siso.mojo",
     "Isolate shared B/C/decay and G×L retained tiles, then test new bounded channel/task sharing one arm at a time.",
     "Incumbent SSD/SISO staging and recomputation.",
     "Same canonical products and per-channel sums; recorded/backward-consumed stages retained.",
     "All forward/state/backward words, prefill/decode and training quality across state sizes/tails.",
     "Reject extra memory traffic or recreating already recorded tile regressions.", "Reuse I08 tile and retained-product arms; no duplicate implementation claims.", priority="P0")
card(34, "Versioned absolute-chunk selective scan", "Mamba-1 selective scan; Samba",
     "mamba/impl/ops/selective_scan_interface.mojo; mamba/impl/ops/selective_scan_backward.mojo; mamba/host/gen",
     "Compose affine recurrence summaries at fixed absolute-position chunk boundaries with a specified carry tree and replay.",
     "Incumbent sequential recurrence evaluation.",
     "New full profile covers host, GPU, backward, checkpoint and decode; retain enough prefix state so chunk boundaries do not depend on total length.",
     "Loss/perplexity, gradients, long-context stability, prefix/decode agreement and resumed-state words.",
     "Reject incomplete decode/backward contract, associative-real arithmetic claims used as bit proofs or FAST length-dependent chunks.", "I09 leaves this profile unimplemented pending a complete contract.", "profile", "P2")
card(35, "Parallel causal depthwise-convolution cells", "Mamba-1/2; Samba; neural Conv1d",
     "mamba/impl/modeling/modeling_mamba.mojo; mamba/impl/modules/mamba2.mojo",
     "Map independent batch/time/channel cells to parallel tasks and share overlapping input windows, with each tap chain fixed.",
     "Serial time loop per channel or repeated window loads.",
     "Causal padding, convolution state, tap order and activation rounding unchanged.",
     "Prefill/decode state agreement, dInput/dWeight, short sequences and tails.",
     "Reject changing padding or crossing recurrent state updates that are not independent.", "I09 already supplies token-parallel conv arms; extend actual window reuse/caller reach.", priority="P0")
card(36, "Cache Mamba decay exponent values", "Mamba-2/3; Samba",
     "mamba/impl/modules/ssd_minimal.mojo; mamba/impl/ops/mamba3_siso.mojo",
     "Compute each immutable portable exp/decay value once per logical position/head and reuse across channels and later consumers.",
     "Recompute identical exponentials inside independent channel tasks.",
     "Store explicitly rounded original exp/decay words; distinguish diagonal, inter-chunk and backward dependencies.",
     "Complete stages and gradients, long/short sequences and extreme decays.",
     "Reject retained buffers costing more than recomputation or reusing numerically similar but nonidentical exponent expressions.", "I08 already retains some decay stages; only uncovered uses are new.", priority="P0")
card(37, "Prune unused triangular SSD work", "Mamba-2 SSD; Mamba-3 SISO",
     "mamba/impl/modules/ssd_minimal.mojo; mamba/impl/ops/mamba3_siso.mojo",
     "Schedule only lower-triangular causal tile tasks while explicitly initializing any trace/backward-visible unused cells.",
     "Compute full Q×Q intermediate matrices.",
     "Causal inclusion and logical arithmetic for used cells fixed; no skipping upper cells if a consumer reads them.",
     "Forward/backward complete stages, chunk tails, checkpoints and prefix behavior.",
     "Reject assumptions based only on forward output; include initialization cost.", "I08 lower-triangle arm exists; scope is attributable downstream reach.")
card(38, "Versioned shared Mamba-3 angle-gradient suffixes", "Mamba-3 backward; Samba training",
     "mamba/impl/modules/mamba3_backward.mojo; mamba/host/gen/mamba3_backward.mojo",
     "Compute canonical suffix summaries once and reuse across d_dt/angle-gradient consumers instead of repeating each suffix.",
     "Each token folds its entire suffix independently.",
     "New version defines reverse leaf boundaries, carry and multiplication placement in device/host/backward oracle; old bits need not match.",
     "All parameter/input gradients, multi-step loss and numerical error on long cancellation-heavy sequences.",
     "Reject a prefix reversal that changes the intended causal derivative or incomplete host generation.", "Programming found IDN_M3_ANGLE_DT_SUFFIX already default-on with chunk=64 and a host counterpart. Reuse it as incumbent; new work must isolate extra suffix-seed reuse or an explicitly different profile.", "profile", "P1")
card(39, "Versioned Mamba parameter-gradient folds", "Mamba-1/2/3 backward; Samba training",
     "mamba/impl/ops/mamba2_ssd_backward.mojo; mamba/impl/modules/mamba3_backward.mojo; mamba/host/gen",
     "Produce fixed-position per-token gradient leaves then merge with a common tree; attribute each parameter family separately.",
     "Long serial time/batch gradient chains.",
     "Logical partition shared by every vendor and host, with exact FMA/FTZ/odd-tail specification; accumulation API semantics remain explicit.",
     "Every gradient tensor, optimizer moments/parameters and full model quality; cancellation and scale range.",
     "Reject output-partial traffic dominating or changing only device folds.", "Roadmap B6 and I09 profile extension.", "profile", "P1")
card(40, "Mamba immutable weight generations and workspace", "Mamba/Samba prefill/decode/training",
     "bindings/_mojolearn_mamba.mojo; mamba/impl/modules/idn_gemm_ws.mojo; training/samba_ops.mojo",
     "Use explicit owner/generation metadata to retain validated packed weights and bounded scratch until a weight/config mutation.",
     "Repeated full weight comparisons/copies and temporary allocations.",
     "No Python data comparison/runtime work; all external mutation paths invalidate or take the safe uncached route.",
     "Forward/decode/backward after update/load/reset, independent sessions and exception lifetime.",
     "Reject pointer-only cache keys or silently trusting externally mutable buffers.", "Roadmap B10 and existing workspace infrastructure.", priority="P0")
card(41, "One-launch ordered recurrent inference/training segments", "RNN; GRU; LSTM",
     "sequence/recurrent_scan.mojo; sequence/recurrent.mojo; sequence/exec_device.mojo",
     "Use supported within-task synchronization for bounded recurrent segments, batching independent sequences while keeping time steps ordered.",
     "One host-dispatched device operation per recurrent timestep.",
     "Hidden/cell state and gate arithmetic exactly preserved; supported memory ordering must suffice without a compiler/runtime workaround.",
     "All sequence outputs/states/gradients, ragged lengths, bidirectionality if supported, and model quality.",
     "Park on unsupported ordering/launch capability; no cross-block spin barriers.", "Roadmap D2; existing FAST Apple scan is not IDENTICAL qualification.", priority="P1")
card(42, "Fuse recurrent gate pointwise updates", "LSTM; GRU; RNN; sequence MLP",
     "sequence/recurrent.mojo; sequence/ops.mojo; sequence/recurrent_scan.mojo",
     "Share projected gate loads for portable activation, state update and derivative-state stores in one task.",
     "Separate gate activation/state/derivative kernels.",
     "Gate order, portable nonlinear functions, state dependencies and materialization rounding fixed.",
     "Forward hidden/cell states, gradients and full recurrent model loss.",
     "Reject different gate formulas or changed saved derivative semantics.", priority="P0")
card(43, "Versioned recurrent weight and bias gradients", "RNN; LSTM; GRU; sequence MLP",
     "sequence/recurrent.mojo; sequence/ops.mojo; sequence/checks/oracle.mojo",
     "Fixed batch/time leaves and a canonical fold replace long sequential gradient reductions; separate dWeight and dBias arms.",
     "Incumbent ascending reduction chains.",
     "Host executor/oracle and device use exactly the new tree, masked-time rules and gradient accumulation ordering.",
     "Gradient error, multi-step training loss, all optimizer state and padded/ragged sequences.",
     "Reject skipping padded terms if old semantics require a signed-zero/nonfinite operation.", "An existing blocked weight-gradient profile is implemented. New arms change a declared leaf or schedule rather than claiming its first introduction.", kind="profile", priority="P1")
card(44, "MoE stable token grouping and grouped expert jobs", "Mixture-of-experts neural layers",
     "sequence/moe_group.mojo; sequence/moe_tiled.mojo; sequence/moe_weights.mojo",
     "Stable-group tokens by chosen expert and batch independent expert products, scattering outputs with original token order.",
     "Per-token/expert dispatch and repeated expert-weight loads.",
     "Router scores, top-k ties, gate normalization, capacity and combine order unchanged; no dropped-token shortcut.",
     "Router/expert outputs and gradients, expert skew, empty experts, overflow and task quality.",
     "Reject sort/setup costs or reordered weighted expert sums.", "Reuse existing grouped/tiled MoE; isolate new scheduling geometry.", priority="P0")
card(45, "Fuse CNN activation/bias/residual passes", "CNNClassifier; Conv1d/2d; ResNet blocks",
     "x_cnn/ops.mojo; x_cnn/device.mojo; x_cnn/host/ops_host.mojo",
     "Share convolution output loads for exact bias/activation/residual operations and save required backward masks in the same pass.",
     "Separate output passes and repeated mask creation.",
     "Explicit rounded seams, activation-zero derivative policy and layout retained; no precision change.",
     "Forward/gradients, residual branch behavior and full CNN classification quality.",
     "Reject register/live-state growth or non-equivalent order of residual and activation.", priority="P0")
card(46, "Deterministic CNN gradient and pooling schedules", "Conv1d/2d; max/average pooling; CNN training",
     "x_cnn/ops.mojo; x_cnn/device.mojo; x_cnn/host/ops_host.mojo",
     "Tile gather-style dInput and dWeight work, and fuse pool value/arg-index output; separately evaluate a versioned fixed dWeight tree.",
     "Incumbent independent gathers/reductions and pooling passes.",
     "Gather schedule keeps arithmetic order; tree variant is a separate version on all columns. Max ties/NaNs and overlap multiplicities fixed.",
     "All gradients and pool indices, dilation/stride/padding tails and model quality.",
     "Reject unordered floating scatter-add or changed tie winners.", kind="schedule+profile")
card(47, "Neural BatchNorm statistics and running-state fusion", "BatchNorm1d/2d; CNN training/inference",
     "x_cnn/ops.mojo; x_cnn/device.mojo; x_cnn/host/ops_host.mojo",
     "Fuse independent statistic loads and normalization output; separately explore a versioned centered fixed-tree moment profile.",
     "Separate reductions, normalization and running-statistic passes.",
     "Training versus inference, biased/unbiased variance uses, epsilon and momentum/running-count updates remain explicit and shared.",
     "Statistics, outputs/gradients, running buffers and complete CNN loss; constant channels and small batches.",
     "Reject cancellation from raw moments or updated running state on a failed step.", kind="schedule+profile")
card(48, "Neural graph aggregation tile reuse", "GCN; GraphSAGE neural layers",
     "x_cnn/ops.mojo; x_cnn/device.mojo; x_cnn/host/ops_host.mojo",
     "Share fixed CSR neighbor-feature tiles across output channels and fuse degree normalization at the specified point.",
     "Independent channel neighbor scans and separate normalization matrices.",
     "Graph edges, stable neighbor order, reduction profile, self-loop and degree semantics unchanged; neural graph layers only.",
     "Node outputs/gradients, isolated/skewed-degree nodes and graph-learning quality.",
     "Reject neighbor sampling, graph sparsification or including classical PageRank in this lane.", priority="P1")

card(49, "Stable embedding grouping with touched-row gradients", "Embedding backward; LM; transformer; neural sequence",
     "embedding/checks/embedding_identical.mojo; embedding/checks/embedding_sort.mojo",
     "Use existing stable radix grouping and process touched rows; separately vary group task geometry and dense zero initialization.",
     "Incumbent row scan/bitonic grouping path.",
     "Token/original-position order, padding, duplicate multiplicity and accumulation semantics fixed; no floating atomics.",
     "Dense gradient words and full neural step, all-same/unique/skew IDs and repeated calls.",
     "Reject sort cost or excluding required dense output allocation/zeroing from timing.", "Reuse I11 actual candidates; new experiments must identify distinct geometry or caller coverage.", priority="P0")
card(50, "Reuse immutable token grouping across backward uses", "Tied embeddings; repeated embedding backward; microbatch replay",
     "embedding/checks/embedding_sort.mojo; training/byte_lm.mojo",
     "Retain canonical sorted IDs, position map and segments for an explicitly owned unchanged token batch.",
     "Rebuild the same grouping for each backward consumer.",
     "Token owner/version, padding and vocabulary are cache keys; position order not inferred from a pointer or shape.",
     "All embedding/head gradients, vocabulary changes, mutated IDs, repeated backward and accumulation.",
     "Reject unbounded caches or changing tied-weight gradient contribution order.", priority="P0")
card(51, "Resident token and target validation", "LM training; embedding; cross entropy",
     "training/byte_lm.mojo; embedding/checks/embedding_identical.mojo; training/checks/loss.mojo",
     "Validate immutable input IDs once in the native owner and pass explicit validated views to embedding/loss consumers; reuse bounded upload buffers.",
     "Repeated upload/readback and index scans in each consumer.",
     "Validation must precede unsafe access; retain canonical first-invalid-index and refusal timing, invalidate on mutation.",
     "Invalid IDs/targets, vocabulary tails, gradients, loss and repeated multi-step training.",
     "Reject a caller-controlled trust flag used without a validation witness.", priority="P0")
card(52, "Cross-entropy elementwise pass fusion", "CrossEntropy; LM; MLP/CNN classification",
     "training/checks/loss.mojo; training/checks/loss_contract.mojo; training/loss_host_rows.mojo",
     "Share logits/max/exp loads for weights, target contribution and dLogits stores while preserving the denominator and reduction profile.",
     "Separate shifted-exp, target-weight and gradient elementwise passes.",
     "Ignore index, label smoothing, reduction, weights and portable log/exp/div fixed; no approximate softmax.",
     "Loss, dLogits, empty/ignored rows, extreme logits and full classification/LM quality.",
     "Reject extra unrequested gradient work in inference or changed weighted denominator.", priority="P0")
card(53, "Stream LM head and exact loss without full logits", "Byte LM train step; large-vocabulary neural heads",
     "training/chunked_lm_head_v2.mojo; training/byte_lm_pooled_head.mojo; training/checks/loss.mojo",
     "Use bounded vocabulary panels, canonical row statistics and tiled backward, retaining or recomputing exact head pieces as separate arms.",
     "Materialize full logits and dLogits before loss and head backward.",
     "The chosen profile fully specifies panel merges; if changed, all host/device/head/loss contracts move together. Requested full logits still materialize.",
     "Loss/perplexity, all head/hidden/tied-embedding gradients, full optimizer state and memory capacity.",
     "Reject sampled/adaptive softmax, fewer vocabulary items or scoring only the loss without required gradients.", "Extend existing chunked-head work with explicit same-version contracts.", "schedule+profile", "P1")
card(54, "Versioned loss reduction across tokens and microbatches", "LM; neural classification/regression losses",
     "training/checks/loss_contract.mojo; training/checks/loss.mojo; training/checks/loss_oracle.mojo",
     "Use a fixed logical token-leaf tree for loss numerator/weight sums, independently comparing a bounded streaming storage form.",
     "Incumbent per-row/token reduction profile.",
     "Same tree in host/all GPUs, explicit normalization/ignore/smoothing/zero-weight rules. Microbatch equivalence only where promised, with a defined merge.",
     "Loss error, gradients, training trajectories and all degenerate-row refusals.",
     "Reject accepting changed bits as proof of unchanged quality or hardware-dependent leaves.", kind="profile", priority="P1")
card(55, "Fuse optimizer post-update status production", "SGD; Adam; AdamW; LM/Samba/MLP training",
     "training/checks/optimizer.mojo; training/opt_gate.mojo; training/byte_lm.mojo",
     "Emit per-block integer first-nonfinite records while update values are live, then reduce once in canonical field/index order.",
     "Rescan updated parameters and state in separate kernels.",
     "Keep pre-access/pre-update checks, same failure step, field/index/message and commit/rollback behavior; only redundant output scans disappear.",
     "Parameters, moments, counters, refusals and multi-step loss, including deliberately nonfinite gradients/states.",
     "Reject partially committed failed steps or a later-step error report.", "Reuse I10 finish-scan design; full owner commit path is necessary.", priority="P0")
card(56, "Batch optimizer parameter groups with per-group scalars", "SGD; Adam/AdamW; other existing neural optimizers",
     "training/opt_gate.mojo; training/checks/optimizer.mojo; sequence/opt_resident.mojo",
     "Precompute canonical step scalars once per group and use bounded group/tensor descriptors in one update grid.",
     "Per-tensor launches or repeated scalar calculation per element.",
     "Respect per-tensor momentum initialization, group hyperparameters, bias correction, weight decay order and maximize semantics.",
     "Every parameter/moment/step counter, heterogeneous groups, empty tensors and full model quality.",
     "Reject changed pow recurrence, epsilon placement or flattening away group semantics.", "Existing one-launch SGD is a foundation; new arms concern descriptor/scalar reuse.", priority="P0")
card(57, "Versioned global gradient-norm reduction", "Gradient clipping; all neural trainers",
     "training/checks/optimizer_contract.mojo; training/checks/optimizer.mojo; training/clip_multi_gpu.mojo",
     "Fixed logical leaves over the declared parameter/tensor order with a canonical norm fold; optionally fuse squared-gradient production with existing readers.",
     "Incumbent per-tensor norms and scalar combination.",
     "New profile shared host/GPU includes tensor order, squared-term rounding, sqrt, epsilon, clip comparison and nonfinite behavior.",
     "Norms, clipped gradients, multi-step parameter trajectories and model task quality, especially clip-boundary cases.",
     "Reject clipping each shard independently or changing tensor grouping without a new contract.", kind="profile", priority="P1")
card(58, "Fuse accumulation with gradient finishing", "Neural microbatch training; tied/shared parameter gradients",
     "training/accumulate_multi_gpu.mojo; training/byte_lm.mojo; training/checks/optimizer.mojo",
     "Consume each produced gradient into the canonical accumulation buffer while emitting allowed status leaves, avoiding separate copy/add scans.",
     "Materialize gradient then separately add/scan it.",
     "Microbatch and shared-parameter contribution order fixed, one explicit rounding at each old add, no unordered atomics.",
     "Complete accumulated gradients and optimizer state over varied microbatch schedules and failures.",
     "Reject changing effective batch size, averaging order or reusing a buffer before its producer completes.", priority="P0")
card(59, "Fuse neural dropout RNG and pointwise consumers", "Dropout; residual blocks; attention/MLP/CNN training",
     "core/philox_neural.mojo; transformer/impl/llama/modeling_llama.mojo; x_cnn/ops.mojo; sequence/ops.mojo",
     "Generate the existing counter-indexed mask in its pointwise consumer and reuse exact mask coordinates in backward/replay.",
     "Materialize/read the entire mask or regenerate it in redundant separate passes.",
     "Seed, step, layer, tensor coordinate and dropout scaling order unchanged; graph scheduling never consumes an RNG stream differently.",
     "Masks, outputs, gradients, checkpoint reproducibility and training quality for p=0 and nonzero p.",
     "Reject changing stochastic semantics, probability, or omitting a publicly requested mask.", priority="P1")
card(60, "Parameter/gradient views with generation-safe ownership", "LM; Samba; neural optimizers",
     "training/byte_lm.mojo; training/byte_lm_model_pool.mojo; training/byte_lm_optimizer_pool.mojo",
     "Bind native model and optimizer views to the same owned storage on supported vendors, refreshing after handle swaps.",
     "Copy all parameters and gradients at each step boundary.",
     "Explicit alias/owner generation, mutation invalidation and synchronization; never silently enable an Apple-only assumption on other vendors.",
     "Load/reset/step/serialize sequences, failed updates, tied weights, concurrent models and multi-step state.",
     "Reject unsupported buffer/view capabilities; record the exact Modular API ask.", priority="P0")
card(61, "One final neural step status/readback boundary", "LM; Samba; MLP training",
     "training/byte_lm.mojo; training/samba_ops.mojo; core/step_glue.mojo",
     "Accumulate safe device status fields and loss summaries for one final owner drain, retaining mandatory pre-access gates.",
     "Repeated intermediate status/loss readbacks and waits.",
     "Canonical refusal priority and rollback/commit state remain identical; do not defer bounds checks that protect memory access.",
     "Every success/failure state, correct step/counter, consumed outputs and full train quality.",
     "Reject hidden CPU work, errors exposed late or removing lifetime barriers rather than replacing ownership.", "I10 supports pieces; complete public-step adoption remains its own experiment.", priority="P0")
card(62, "Live-range neural scratch and activation arenas", "LM; transformer; Mamba/Samba; CNN; recurrent models",
     "training/byte_lm_layer_pool.mojo; core/device_arena.mojo; training/dev_tensors.mojo",
     "Reuse disjoint-lifetime activation/gradient slabs within a bounded model/session arena; separate allocation and aliasing changes.",
     "Per-stage temporaries or overprovisioned high-water buffers.",
     "Liveness includes pending device consumers, saved backward state, error paths and replay; all cells read are initialized.",
     "Alternating model sizes, repeated inference/train, checkpointing, memory pressure and full outputs.",
     "Reject unbounded retention, inter-session aliases or moving allocation outside cold timing.", "I02 covers GEMM workspace; this extends model-wide liveness.", priority="P1")
card(63, "Deterministic neural multi-device shard merge", "Existing multi-GPU neural GEMM/optimizers/training only",
     "training/optimizer_multi_gpu.mojo; training/clip_multi_gpu.mojo; training/accumulate_multi_gpu.mojo; core/shard_merge_device.mojo",
     "Batch transport and merge canonical logical gradient leaves in one prescribed global order, preserving a single-device equivalent profile where promised.",
     "Incumbent shard transport/merge schedule.",
     "Device count/topology may schedule transfer but cannot redefine arithmetic leaf membership; no nondeterministic library collective as an identity substitute.",
     "Full global gradients, clip norms, optimizer state and model quality; uneven shards and failure cleanup.",
     "If supported Mojo communication/ordering is missing, park and request it; no unsupported transport workaround.", priority="P2")
card(64, "Neural inference state batching with isolated sessions", "Transformer/LM decode; Mamba/Samba decode; recurrent inference",
     "training/byte_lm_model_pool.mojo; transformer/impl/llama/modeling_llama.mojo; mamba/impl/modeling/modeling_mamba.mojo; sequence/recurrent.mojo",
     "Batch independent inference sessions/requests with explicit per-session cache/state lengths and gather/scatter descriptors.",
     "Separate launches and repeated model-weight staging per request.",
     "Each request retains its own token order, cache/hidden state, RNG and arithmetic profile; length buckets follow work/bytes only.",
     "Every request's logits/state against its own sequential execution, reset/cancel/tails and task quality.",
     "Reject cross-session contamination, changed padding arithmetic or latency regressions hidden by aggregate throughput.", priority="P1")


POLICY = {
    "scope": "NEURAL ONLY: neural GEMM, attention/transformers, language models, Mamba/Samba, RNN/LSTM/GRU, MLP/MoE, CNN/ResNet, neural GCN/GraphSAGE, embeddings, neural losses/optimizers and their native runtime. Excludes classical ML, classical forecasts, ANN/vector search, trees, clustering and Apple FAST tuning. Shared primitive changes must remain opt-in for named neural callers.",
    "identity": "A and B may have different bits. Within each arm/version, NVIDIA, AMD, Apple and host must match bitwise wherever promised: outputs, gradients, fitted/state/checkpoint buffers, selection/stopping and errors. Changed arithmetic requires a coherent same-version host/device/forward/backward/decode contract, not an old-version equality gate.",
    "quality": "No degraded quality is accepted. Use the repository's existing model-specific acceptance rules and full tasks: LM held-out loss/perplexity, classification/regression metrics, neural forecasting task loss, gradients/finite state and multi-step training stability. Numerical errors and identity alone do not establish task quality. Do not invent relaxed thresholds.",
    "scope_of_this_request": "Generate ideas, then program opt-in source in this isolated worktree. Do NOT compile, run tests/checkers/static verification, execute candidate ML code, benchmark, rent machines, or update measured boards. All new switches default OFF. Source drafting is not qualification.",
    "future_measurement": "Freeze one commit; reuse matching accepted compilation/identity evidence. Full-dataset/full-workload end-to-end A/B for every affected model on NVIDIA and AMD, one excluded warmup and one scored sample, vendors in parallel and cells serial per GPU. Include preparation, required synchronization and consumed outputs; distinguish fit/train, inference/prefill/decode, cold/repeated and complete default combinations. Component timings never qualify a default.",
    "promotion": "Joint NVIDIA+AMD improvement with neither materially slower, required four-column identity and model quality preserved. Apple is an identity witness, never a timing vote for IDENTICAL. Preserve loser/neutral/failure/pending evidence and use board tools when actual measurements exist. No new default or merge to main from these uncompiled drafts.",
    "routing": "No benchmark dimensions/names/seeds or near-board thresholds. Derive geometry/route/caps from work, bytes, supported hardware and numeric contract; explain in source. Cover adjacent shapes and a non-board dataset later. Removing a targeted rule uses old rule as B.",
    "runtime": "All runtime data work is Mojo, GPU work stays on device for GPU routes; Python is API shell and experiment metadata/orchestration only. No Python arithmetic, loops/sorting/label work or worker threads in runtime.",
    "unsupported": "Use supported Mojo/Modular features only. Record exact upstream asks if unavailable; no compiler-output rewrites, toolchain patches, unsupported modes or invented vendor instructions.",
    "resources": "Future CPU arms use the full actual machine allocation, Linux cgroup allocation rather than host count, serial measurement arms and documented nested pools. Record effective worker pools; do not alter another active freeze.",
}
RECIPES = {
    "primary_neural": "tools/bench_board_neural.py: LANES, MODEL_OF, LM_SHAPES/GEMM_SHAPES/BLOCK_SHAPES/SAMBA_SHAPES/MLP_SHAPES, make_inputs, build_runner, quality; retain exact corpus/data hashes and complete model config.",
    "neural_layers_and_estimators": "tools/bench_board_algos.py: ONLY neural sequence, CNN, embedding, loss and optimizer _add entries plus _build_seqmodel/_build_cnnclf/_build_layer/_build_optim. Exclude its classical time-series and classical estimator entries despite shared directories.",
    "board_orchestration": "tools/bench_board.py: plan_races and neural-family call sites; recipes and intrinsic caps must be read before future timing. --rows full or --neural-shape full is not evidence of actual full input.",
    "existing_candidates": "experiments/performance_ideas/I01-I11, A01-A05 and N01-N06 manifests/coverage/native_arms; component drivers are starting references, not full-workload qualification.",
}
INTERACTIONS = [
    "NN01–NN16: GEMM physical schedule × arithmetic profile × workspace × grouped jobs × epilogues; isolate before the proposed complete configuration.",
    "NN17/18/19/20/21/22/23: attention sharing, masks, saved state, softmax version and backward; measure a full layer and full train step.",
    "NN24/25/26/27/28/30/31/32: norm arithmetic versus schedule, activation fusion, positions, caches, views and saved forward lifetime.",
    "NN33–NN40: SSD tiles, scan/suffix/gradient profiles, decay reuse, causal pruning and generations; prefill/decode/backward must compose.",
    "NN13/41/42/43/44: sequence GEMM, recurrent segments, gates, gradient profile and MoE grouping without changing dependent update order.",
    "NN14/45/46/47/48: CNN lowering/fusion/gradients/statistics and neural graph operators, including host mirrors and full estimators.",
    "NN49/50/51/52/53/54: embedding grouping, IDs, head/loss streaming and reduction; account for tied-weight gradient order.",
    "NN55/56/57/58/59/60/61/62: optimizer status, group scalars, clipping, accumulation, RNG, aliases and arenas; full committed or failed step.",
    "NN63/64 × NN03/04/20/24/34/38/39/43/54/57: versioned arithmetic must not accidentally depend on GPU count, batching or session routing.",
]


def write_catalog():
    record = dict(schema=1, baseline="fd6cf80453a6f18eb02e81566c824e7da106ccf0", policy=POLICY,
                  workload_recipe_sources=RECIPES, interaction_rounds=INTERACTIONS, experiments=CARDS)
    (HERE / "catalog.json").write_text(json.dumps(record, indent=2) + "\n")
    out = ["# Neural-only IDENTICAL A/B ideas — 2026-10-06", "",
           "64 cards drafted before implementation fan-out. These are hypotheses inferred from local source, not measured bottleneck or speedup claims. Source baseline `fd6cf80453a6f18eb02e81566c824e7da106ccf0`; branch `ideas/neural-identical-ab-20261006-r3`.", "",
           "## Contract", ""]
    out.extend(f"- **{k.replace('_', ' ').capitalize()}:** {v}" for k, v in POLICY.items())
    out.extend(["", "## Future full-workload map", "",
                "Each card names affected neural caller families and source anchors. Before any later timing, map every transitive affected public caller to its saved full recipe, dataset/corpus version/hash/split, actual dimensions, dtype/mode, full model/optimizer/seed settings, A/B flags and timed boundary. Audit all internal caps. Missing mappings remain pending; do not substitute a tiny driver.", ""])
    out.extend(f"- **{k}:** {v}" for k, v in RECIPES.items())
    out.extend(["", "## Priorities and arms", "",
                "A = candidate; B = frozen incumbent. Hold unrelated switches fixed; an all-off build is not the normal control. P0 = low arithmetic risk and broad memory/launch opportunity; P1 = caller integration or explicit numerical revision; P2 = high design cost or capability uncertainty. No priority claims a measured speedup. Within an arm compare all vendors with that arm's host. Between arms compare quality and complete timing, not old-versus-new bit equality.", "",
                "Each compound card names independently attributable sub-arms. A profile candidate needs a written arithmetic contract and actual coherent implementation; a flag or manifest is not an implementation. Reuse existing candidates/evidence, and retain existing loser controls rather than restarting them unchanged.", "",
                "## Inventory", "", "| Card | Lane | Type | Priority | Idea |", "| --- | --- | --- | --- | --- |"])
    out.extend(f"| [{c['id']}](#{c['id'].lower()}) | {c['lane']} | {c['kind']} | {c['priority']} | {c['title']} |" for c in CARDS)
    for c in CARDS:
        out.extend(["", f"<a id=\"{c['id'].lower()}\"></a>", f"## {c['id']} — {c['title']}", "",
                    "**Affected callers:** " + "; ".join(c['callers']) + ".", "",
                    "**Source anchors:** " + "; ".join(f"`{p}`" for p in c['source_anchors']) + ".", "",
                    f"**A:** {c['arm_a']}", "", f"**B:** {c['arm_b']}", "",
                    f"**Same-version identity:** {c['identity_contract']}", "",
                    f"**Quality:** {c['quality_gate']}", "",
                    f"**Reject/park:** {c['rejection_conditions']}", "",
                    f"**Prior work:** {c['prior_work']}", "",
                    "**Later A/B coverage:** affected full neural models/datasets, preparation through consumed outputs, plus isolated phase diagnostics; adjacent shapes and a non-board dataset. Separate forward/backward/train, prefill/decode, cold/repeated use. Record bytes, launches/waits, peak scratch, compile/profile/source/binary/hardware/harness provenance and per-vendor failures. One excluded warmup and one scored sample initially; no unsupported statistical confidence claim.", "",
                    "**Evidence at fan-out:** idea only. Compilation, verification and timing intentionally not run. Follow lane handoffs for exact implemented versus component-only versus pending coverage."])
    out.extend(["", "## Interaction rounds", ""])
    out.extend(f"- {s}" for s in INTERACTIONS)
    out.extend(["", "## Excluded shortcuts", "",
                "No lower precision/TF32 substitution, model-size/rank/sequence/batch/vocabulary reduction, fewer layers/epochs/steps, smaller corpora, skipped gradients or uncertainty checks, sampled softmax, approximate attention, changed dropout/seed, relaxed quality thresholds, removal of correctness settings, Python runtime compute or hidden CPU fallback. Supported low-precision products retain their own semantics; they are not a way to accelerate the FP32 contract by changing it.", "",
                "## Programming lanes", "",
                "NN01–NN16 GEMM; NN17–NN32 attention/transformer; NN33–NN48 state-space/recurrent/CNN; NN49–NN64 embeddings/loss/optimizer/runtime. All source stays opt-in in this new worktree. No build, checker, test, execution, benchmark or promotion is authorized. If a full caller/profile cannot be programmed coherently, record concrete remaining code, not a fabricated completion. Unsupported features are parked with Modular asks.", "",
                "> Keep logs out of context: save complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or infer full success from filtered output.", ""])
    (ROOT / "docs/plans/NEURAL_IDENTICAL_AB_IDEAS_2026-10-06.md").write_text("\n".join(out))


if __name__ == "__main__":
    write_catalog()
