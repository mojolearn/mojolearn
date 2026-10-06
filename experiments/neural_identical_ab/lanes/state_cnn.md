# Neural state/CNN lane source handoff

Branch `ideas/neural-identical-ab-20261006-r3`; base `main` at `fd6cf80453a6f18eb02e81566c824e7da106ccf0`. Source changes are uncommitted; no hooks were run.

Compilation, static verification, tests, model execution, vendor identity, quality and timing were **not run by request**. No default was promoted. Every new switch is opt-in, IDENTICAL-only and disabled by `MOJOLEARN_IDN_ALL_OFF`. Existing inherited defaults are not rebranded as new work. There are no measurement sample counts other than zero.

Arithmetic changes are allowed between versions. NN39 and NN43 define new shared within-version graphs. Scheduling arms retain incumbent per-cell arithmetic; this is intended design, not executed proof.

Repository-generated host sources were directly mirrored where new arithmetic/internal state requires it. The repository generator was not executed; source freshness and any generator compatibility remain pending. NN36 has an authored host fallback substitution in `tools/mamba_host_gen.py`.

The catalog scope is broader than these draft implementations. Per-card remaining work below must remain visible; do not label a card fully qualified merely because a source switch is wired.

## NN33 — wired_draft

Candidate defines: `MOJOLEARN_NN33_YDIAG_ROWS8`, `MOJOLEARN_NN33_CSTATE_P16`.

Implemented: New independent geometries on inherited I08 SSD tiles: eight Ydiag rows with 32-column input staging; sixteen Cstate value channels. Existing per-cell leaves, order, padded terms and saved stages retained.

Pending: Mamba-3 SISO geometry extension. Generated-source refresh for changed source text.

Source: `mamba/impl/modules/ssd_minimal.mojo`.

Prior work: I08 shared tiles/defaults and optional retained G*L already existed; no duplicate implementation or measurement claim.

Route requirements: Inherited M2_SSD_TILED route active; the existing shared-memory resource guard still applies.

## NN34 — component_draft

Candidate define: `MOJOLEARN_NN34_AFFINE_PREFIX`.

Implemented: an isolated prepared-factor recurrence A/B with real host/device prefill and one-token decode entry points. Candidate arithmetic uses fixed absolute 32-token chunks, shared affine compose/evaluate helpers and canonical adjacent-pair prefix trees. Three-phase prefill constructs independent prefixes, propagates completed-chunk boundaries per chain and evaluates outputs. Decode retains binary-prefix slots and uses the same graph. Initialization and checkpoint profile-id metadata checks are present.

The new graph may change bits from the previous version. It is intended to match across host/NVIDIA/AMD/Apple within this version; no execution or quality evidence exists. No public Mamba/Samba route imports the component.

Pending: raw model factor preparation/emission, all backward/optimizer consequences, public ownership, serialized checkpoint validation/restore, failure handling, ragged streams and complete model/host migration. Repeated prefix subtree work is still a scheduling opportunity. Component A/B is not full-dataset model evidence.

Source: `mamba/impl/ops/neural_scan_profile.mojo`. Contract: `experiments/neural_identical_ab/NN34_AFFINE_PREFIX_CONTRACT.md`.

## NN35 — wired_draft

Candidate defines: `MOJOLEARN_NN35_CONV_PAIR`.

Implemented: Two adjacent independent time cells share input windows and weight loads for Mamba-1 and Mamba-2 causal depthwise convolution. Original bias-seeded ordered FMA taps, portable SiLU, retained preactivation and separate window-state update remain. Odd tail loads only the inputs used by its one live output.

Pending: Generated-source refresh; old host per-cell arithmetic remains the same intended output contract. Additional standalone neural Conv1d callers and gradient-specific window reuse.

Source: `mamba/impl/modeling/modeling_mamba.mojo`, `mamba/impl/modules/mamba2.mojo`.

Prior work: I09 token-parallel cell arms are inherited. Pair window reuse is the new arm.

## NN36 — wired_draft

Candidate defines: `MOJOLEARN_NN36_SHARED_DECAY`.

Implemented: Mamba-2 off-diagonal SSD output uses one block per batch/token/head, sharing one explicitly rounded exp word across value channels. No retained global scratch or extra synchronization lifetime. Generator substitution selects unchanged host cell exp/dot expression for the shared-memory kernel.

Pending: Run repository host source regeneration when authorized; not run here. Remaining Mamba-2 inter-chunk/backward and Mamba-3 decay consumers.

Source: `mamba/impl/modules/ssd_minimal.mojo`, `tools/mamba_host_gen.py`.

Prior work: Existing retained decay stages and I08 reuse are not claimed as new.

## NN37 — wired_draft

Candidate defines: `MOJOLEARN_NN37_TRIANGLE_TASKS`.

Implemented: Mamba-2 CB tasks enumerate only lower-triangle coordinates using exact integer upper-bound search. Explicit full cb_g zero-fill preserves upper and padded stage cells for traces/backward; include its launch and traffic in A/B.

Pending: Mamba-3 causal task scheduling. Generated-source refresh.

Source: `mamba/impl/modules/ssd_minimal.mojo`.

Prior work: I08 lower-triangle arithmetic pruning already existed. Compact task indexing and explicit clearing are new.

Route requirements: Inherited IDN_M2_CB_LOWER must remain enabled.

## NN38 — wired_draft

Candidate defines: `MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS`.

Implemented: Reuse each descending chunk-summary chain once to create exclusive suffix seeds in place before replaying token chunks. Device and direct host source mirror include seed scratch transformation and retain inherited angle/rate/dt operations.

Pending: Generated-source refresh remains pending; mirrors were source-authored without running the generator. Full Mamba-3/Samba caller mapping and scored quality/identity later.

Source: `mamba/impl/modules/mamba3_backward.mojo`, `mamba/host/gen/mamba3_backward.mojo`.

Arithmetic profile: The fixed-64 chunk angle suffix profile already exists and is default-on; this new extension changes only repeated computation scheduling.

Route requirements: Inherited IDN_M3_ANGLE_DT_SUFFIX active.

## NN39 — wired_draft

Candidate defines: `MOJOLEARN_NN39_M2_GRAD_TREE`.

Implemented: New shared pure Mojo Mamba-2 gradient fold profile uses existing row leaves and adjacent binary subtree merges, with explicitly carried odd tails and no added zero leaf. Device and host import the same arithmetic helper; single leaf is returned verbatim. Reaches existing Mamba-2 conv weight/bias, A/A_log and dt_bias terminal-gradient merge caller.

Pending: Mamba-1/Mamba-3 parameter families and separately attributable per-family switches. Generated-source refresh and future full optimizer/quality acceptance.

Source: `mamba/impl/ops/neural_gradient_profile.mojo`, `mamba/impl/ops/mamba2_ssd_backward.mojo`, `mamba/host/gen/mamba2_ssd_backward.mojo`.

Prior work: Mamba-2 fixed 256-row gradient leaves and serial partial merge were already default-on; the balanced merge is new.

Arithmetic profile: mojolearn.neural-ab.mamba2.gradient-fold.fp32.v2; version-to-version bits may change, within-version columns share the exact helper.

## NN40 — wired_draft

Candidate defines: `MOJOLEARN_NN40_WS_HEADROOM`.

Implemented: Workspace-only sub-arm reserves 25% grow-ahead capacity capped at 16 MiB additional scratch. Existing per-context ownership and wait-before-replacement semantics stay.

Pending: Immutable packed-weight owner/generation API for all mutation/load/update routes. Cross-session invalidation and public caller ownership mapping.

Source: `mamba/impl/modules/idn_gemm_ws.mojo`.

Prior work: Existing decode sessions already own copied weights; ordinary external mutable buffers are not trusted by this arm.

Scope limit: This does not implement or claim a weight-generation cache.

Route requirements: Existing IDN_MAMBA_GEMM_WS enabled; GPU workspace route only.

## NN41 — reused_existing

Candidate defines: `MOJOLEARN_IDN_SEQ_LSTM_SCAN`.

Implemented: Existing IDENTICAL whole-sequence scan is reused rather than reimplemented. Source repair restricts SCAN_WIDE to FAST because its backward folds are a different graph and the host scan uses the original graph.

Pending: Recorded FAST scan quality failure still needs diagnosis; source alone does not establish its cause. IDENTICAL forward/backward scan vendor ordering and quality qualification. Explicit bounded-segment launch/state contract beyond existing whole-sequence scan.

Source: `sequence/recurrent_scan.mojo`, `sequence/recurrent.mojo`, `sequence/exec_device.mojo`.

Prior failure: Source records constant-predictor quality collapse in FAST scan even after Args repair; NOT repaired or qualified by this lane.

Modular ask: If the existing team_barrier memory ordering is insufficient on a vendor, request a documented supported workgroup barrier that orders device-memory stores/loads across loop iterations, or a supported persistent launch primitive. No cross-block spin barrier or compiler workaround.

## NN42 — wired_draft

Candidate defines: `MOJOLEARN_NN42_INPUT_BIAS_FUSED`.

Implemented: Fold input projection bias materialization into the per-step recurrent hidden/gate kernel. Each owner stores the same rounded GX word before reading its gates; shared host/GPU element body. Saved GX, gate activations, hidden/cell words and derivative inputs keep their seams.

Pending: Scan path deliberately retains its separate bias launch. Sequence MLP-specific fusion and further derivative-state fusions.

Source: `sequence/ops.mojo`, `sequence/recurrent.mojo`.

Route requirements: Per-step recurrent path; when SEQ_LSTM_SCAN is compiled, this arm deliberately yields to the existing bias path.

## NN43 — wired_draft

Candidate defines: `MOJOLEARN_NN43_WGRAD_FIXED128`.

Implemented: New recurrent weight/bias profile uses absolute 128-row leaves, incumbent per-leaf chain and ascending partial merges. Host and GPU execute the same generic Exec split-K operations; independent oracle source states the same partition. For exceptional K whose single-cell partials exceed the fixed scratch budget, double the leaf identically on every column until it fits.

Pending: Separate dWeight/dBias attribution switches. Non-recurrent sequence MLP gradient profile caller coverage.

Source: `sequence/recurrent.mojo`, `sequence/checks/oracle.mojo`, `sequence/ops.mojo`.

Prior work: The default square-root/512-minimum blocked gradient profile already existed; this arm is a distinct profile.

Arithmetic profile: mojolearn.neural-ab.recurrent.gradient-leaves.fp32.v2; numerical profile may change previous-version bits.

Route requirements: Existing blocked recurrent weight-gradient route enabled.

## NN44 — wired_draft

Candidate defines: `MOJOLEARN_NN44_STABLE_GROUP`, `MOJOLEARN_NN44_EXPERT_BISECT`.

Implemented: Stable per-expert ascending pair scatter reuses existing device counts/offsets and preserves original pair output positions. Independent integer binary-search expert lookup handles empty experts and retains scheduling semantics. Router/top-k/weighted combine arithmetic remains inherited.

Pending: Scalable parallel stable grouping with explicit workspace for large expert counts. Additional expert gradient/training grouped caller coverage.

Source: `sequence/moe_group.mojo`, `sequence/moe_tiled.mojo`, `sequence/exec_device.mojo`.

Risk: Stable scatter does E*pair_count comparisons; it may lose despite less cursor contention. Existing FP32 pair indices/range contract is inherited, not enlarged.

## NN45 — wired_draft

Candidate defines: `MOJOLEARN_NN45_CONV_RELU`.

Implemented: Fuse GEMM-output layout, bias, retained preactivation and ReLU for unpooled CNN blocks. Shared host/device element function preserves canonical NaNs, bias rounding and ReLU derivative inputs. Existing direct-convolution path still uses its original separate ReLU.

Pending: Residual epilogue fusion and additional Conv1d/ResNet callers. Pooled block path already has inherited ReLU/maxpool fusion; no new claim.

Source: `x_cnn/ops.mojo`, `x_cnn/device.mojo`, `x_cnn/host/ops_host.mojo`.

## NN46 — wired_draft

Candidate defines: `MOJOLEARN_NN46_GATHER_BOUNDS`.

Implemented: Derive exact candidate tap intervals for col2im and maxpool-backward gathers from stride/dilation/padding inequalities. Keep all original divisibility checks, tap order and floating operations; reversed negative-control path keeps incumbent bounds. Fused pool-ReLU backward consumers share the same maxpool value helper.

Pending: New dWeight tree and shared-memory gather tiling. Host optimized plane routines retain the same old fold; new host scheduling is not claimed.

Source: `x_cnn/ops.mojo`.

Prior work: Maxpool value/arg-index already fused; tiled pool and several bounded backward paths were inherited.

## NN47 — wired_draft

Candidate defines: `MOJOLEARN_NN47_APPLY_RUNNING`.

Implemented: Training normalization output and running-state updates share one launch using the same element helper on host/GPU. Exactly one first-image pixel per channel owns running updates after existing statistics; output does not read running state. Existing mean/centered variance folds, biased/unbiased conversions, momentum and epsilon unchanged.

Pending: Separate centered fixed-tree moment profile and additional statistics-load reuse. Alias/exception and small-batch acceptance when verification is authorized.

Source: `x_cnn/ops.mojo`, `x_cnn/device.mojo`, `x_cnn/host/ops_host.mojo`.

## NN48 — wired_draft

Candidate defines: `MOJOLEARN_NN48_CSR_TILES`.

Implemented: Neural CSR aggregation stages 32 edge columns/weights once per 128-feature tile, preserving ascending edge folds. Weighted, mean, inverse-weight and backward normalization modes retain original floating seams; inactive feature lanes still join barriers. Host retains the incumbent exact arithmetic definition.

Pending: GraphSAGE max-specific tiles and degree-normalization fusion. Graph learning forward/backward quality and skew/tail coverage later.

Source: `x_cnn/device.mojo`, `x_cnn/ops.mojo`.

Scope limit: Neural spmm only; no PageRank, classical graph, sparsification or neighbor sampling changes.

## NN13 — reused_existing

Candidate defines: none newly introduced.

Implemented: Inherited same-chain sequence tiled GEMM serves neural recurrent/MLP projections; source was not changed by this lane.

Pending: GEMM lane owns any new neural caller/profile API.

Source: `sequence/gemm_tiled.mojo`, `sequence/exec_device.mojo`.

Scope limit: No change to generic shared dispatcher or classical sequence callers.

## NN14 — wired_draft

Candidate defines: `MOJOLEARN_NN14_BOUNDED_IM2COL`.

Implemented: Public Conv2d forward device and host paths batch complete K rows through bounded im2col/GEMM scratch with global NCHW scatter. Target cols+y2 scratch is 8 MiB, except one indivisible row; full K reduction and bias seam retained.

Pending: CNNClassifier saved-column block/training callers, backward dInput/dWeight caller integration. Direct-convolution comparison must be included: this candidate public-forward path uses im2col/GEMM even where baseline direct route applies. The per-call requested scratch bound does not shrink existing process context caches; cold/repeated peak memory remain separate measurements.

Source: `x_cnn/ops.mojo`, `x_cnn/device.mojo`, `x_cnn/host/ops_host.mojo`.

Scope limit: Forward caller sub-arm only; not complete NN14 training coverage.

## Later evidence requirements

For each active arm, first map actual full-workload recipes, dataset/version/hash and dimensions, estimator settings, exact A/B defines and the timed boundary. Preparation, training/fit, synchronization and consumed outputs belong inside the declared operation. Separate forward/backward/train, prefill/decode, cold/repeated use, neighboring shapes and a non-board dataset. Do not substitute components for the affected full neural workloads.

Promotion remains disallowed without future accepted frozen compilation, required same-version host/NVIDIA/AMD/Apple identity, quality and full-dataset end-to-end NVIDIA+AMD A/B evidence. Include independent arms and combinations. Reuse already accepted evidence where applicable. Apple identity matters; Apple timing does not vote.

Keep logs out of context: save complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or infer full success from filtered output.
