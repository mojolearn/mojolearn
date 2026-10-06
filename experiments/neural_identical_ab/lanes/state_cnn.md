# Neural state/CNN source handoff

Branch `ideas/neural-identical-ab-20261006-r3`, forked from `main` at `fd6cf80453a6f18eb02e81566c824e7da106ccf0`. These edits follow root commit `88c68d132`; root owns the next commit/push.

All selected arms below have actual runtime caller source or explicitly reuse an existing caller. They remain **uncompiled and unverified source drafts**. Compilation, checks, model execution, same-version vendor identity, quality and timing were not run by request. No default was promoted; all new switches are OFF, IDENTICAL-only and respect `MOJOLEARN_IDN_ALL_OFF`.

Host source was authored with the explicit `--source-write-only` generator mode (exit0; `/tmp/neural_mamba_source_authoring.log`). It disables post-generation verification and stale comparisons. The ordinary write/check paths remain separate. Private stack arrays are preserved; only actual shared-memory/barrier bodies are replaced by host stubs. This is source authoring, not compile or identity evidence.

NN03/04 integration covers dense Mamba projections/backward and CNN contractions. The handwritten Mamba-2/3 SSD/SISO forward kernels retain their own incumbent numerical profile, and their host SSD helper calls deliberately retain the matching graph. Sequence retains its separate same-chain profile.

## Chosen source arms

### NN33 — wired_draft

Mamba-2 inherited tiled Ydiag rows8 and Cstate value-width16 geometries.

Defines: `MOJOLEARN_NN33_YDIAG_ROWS8`, `MOJOLEARN_NN33_CSTATE_P16`.

New independent geometries on inherited I08 SSD tiles: eight Ydiag rows with 32-column input staging; sixteen Cstate value channels. Existing per-cell leaves, order, padded terms and saved stages retained.

Further research, outside this chosen arm: Mamba-3 SISO geometry extension.

Sources: `mamba/impl/modules/ssd_minimal.mojo`.

### NN34 — wired_draft

Mamba-1 full prefill/decode and zero-state prefill backward affine-prefix numerical profile; public checkpoint and host inference integration.

Defines: `MOJOLEARN_NN34_AFFINE_PREFIX`.

Fixed absolute32 prefix tree and shared compose/evaluate; actual Mamba-1 factor preparation, emission, D skip and complete model forward/decode route. Optional device-owned boundary and affine slots; profile id and scalar absolute position in Python checkpoint schema; host and resident GPU session open/export/load; scalar metadata and named finite refusal for added state. Real canonical-tree VJP for zero-state prefill: descending prefix losses, reverse node creation order, chunk-boundary adjoints, separate prepared-factor A/B gradients feeding all existing downstream parameter/input gradients. Public Mamba host generated model plus alternate neural-host inference/oracle share new profile. Source-only generator refresh; no check or compilation execution.

Further research, outside this chosen arm: Parallel-prefix subtree reuse; segmented training/checkpoint VJP beyond the established zero-state prefill-backward API; Mamba-2/3 numerical profiles; independent-position ragged carried-state API (existing ragged fresh inference still invokes independent rows).

Sources: `mamba/impl/ops/neural_scan_profile.mojo`, `mamba/impl/ops/neural_mamba_scan.mojo`, `mamba/impl/ops/selective_scan_backward.mojo`, `mamba/impl/modeling/modeling_mamba.mojo`, `mamba/checks/mamba_oracle.mojo`, `mamba/checks/mamba_backward.mojo`, `mamba/host/gen/neural_scan_profile.mojo`, `mamba/host/gen/neural_mamba_scan.mojo`, `mamba/host/gen/selective_scan_backward.mojo`, `mamba/host/gen/modeling_mamba.mojo`, `bindings/_mojolearn_mamba.mojo`, `bindings/_mojolearn_mamba_host.mojo`, `bindings/_mojolearn_neural_host.mojo`, `python/mojolearn/_mamba_impl.py`, `experiments/neural_identical_ab/NN34_AFFINE_PREFIX_CONTRACT.md`.

### NN35 — wired_draft

Mamba-1/2 paired causal-convolution time outputs with shared window loads.

Defines: `MOJOLEARN_NN35_CONV_PAIR`.

Two adjacent independent time cells share input windows and weight loads for Mamba-1 and Mamba-2 causal depthwise convolution. Original bias-seeded ordered FMA taps, portable SiLU, retained preactivation and separate window-state update remain. Odd tail loads only the inputs used by its one live output.

Further research, outside this chosen arm: Additional standalone neural Conv1d callers and gradient-specific window reuse.

Sources: `mamba/impl/modeling/modeling_mamba.mojo`, `mamba/impl/modules/mamba2.mojo`.

### NN36 — wired_draft

Mamba-2 off-diagonal SSD output shares the rounded token/head exp across value channels.

Defines: `MOJOLEARN_NN36_SHARED_DECAY`.

Mamba-2 off-diagonal SSD output uses one block per batch/token/head, sharing one explicitly rounded exp word across value channels. No retained global scratch or extra synchronization lifetime. Generator substitution selects unchanged host cell exp/dot expression for the shared-memory kernel.

Further research, outside this chosen arm: Remaining Mamba-2 inter-chunk/backward and Mamba-3 decay consumers.

Sources: `mamba/impl/modules/ssd_minimal.mojo`, `tools/mamba_host_gen.py`.

### NN37 — wired_draft

Mamba-2 lower-triangle CB task enumeration with explicit upper/padded stage zeroing.

Defines: `MOJOLEARN_NN37_TRIANGLE_TASKS`.

Mamba-2 CB tasks enumerate only lower-triangle coordinates using exact integer upper-bound search. Explicit full cb_g zero-fill preserves upper and padded stage cells for traces/backward; include its launch and traffic in A/B.

Further research, outside this chosen arm: Mamba-3 causal task scheduling.

Sources: `mamba/impl/modules/ssd_minimal.mojo`.

### NN38 — wired_draft

Mamba-3 existing angle/dt suffix profile reuses descending chunk-summary seeds before replay.

Defines: `MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS`.

Reuse each descending chunk-summary chain once to create exclusive suffix seeds in place before replaying token chunks. Device and direct host source mirror include seed scratch transformation and retain inherited angle/rate/dt operations.

Sources: `mamba/impl/modules/mamba3_backward.mojo`, `mamba/host/gen/mamba3_backward.mojo`.

### NN39 — wired_draft

Mamba-2 existing fixed256-row conv/A/dt gradient partials merge with one shared adjacent-pair tree.

Defines: `MOJOLEARN_NN39_M2_GRAD_TREE`.

New shared pure Mojo Mamba-2 gradient fold profile uses existing row leaves and adjacent binary subtree merges, with explicitly carried odd tails and no added zero leaf. Device and host import the same arithmetic helper; single leaf is returned verbatim. Reaches existing Mamba-2 conv weight/bias, A/A_log and dt_bias terminal-gradient merge caller.

Further research, outside this chosen arm: Mamba-1/Mamba-3 parameter families and separately attributable per-family switches.

Sources: `mamba/impl/ops/neural_gradient_profile.mojo`, `mamba/impl/ops/mamba2_ssd_backward.mojo`, `mamba/host/gen/mamba2_ssd_backward.mojo`.

### NN40 — wired_draft

Retained projection workspace headroom plus explicit Mamba-3 immutable snapshot install/update/export and generation-owned fresh forward/backward on GPU and host.

Defines: `MOJOLEARN_NN40_WS_HEADROOM`, `MOJOLEARN_NN40_OWNED_WEIGHTS`.

Projection workspace grow-ahead capacity (25%, at most16MiB) retains context ownership and wait-before-replacement. Explicit Mamba3Block.install_owned_weights creates an immutable copied snapshot; generation-owned fresh forward/backward never trust mutable external pointers; reinstall after optimizer/load mutation invalidates device stages. Host implements the same snapshot semantics; export reads installed bytes; pickle stores those bytes and lazily reinstalls a new generation. Binding changes, stale generation, busy/unusable sessions and disabled ownership arm refuse by name. Ordinary borrowed-buffer calls retain byte validation.

Ownership recipe: explicitly call `install_owned_weights()` for A and reinstall after every intended optimizer/load update. B uses the ordinary borrowed-buffer path. Include snapshot installation/update in cold/update timing. Ordinary mutable weight semantics do not change when the compile flag alone is set.

Further research, outside this chosen arm: Packed/pretransposed immutable weight layouts with additional retained-memory accounting; Mamba-1/2 snapshot-generation APIs.

Sources: `mamba/impl/modules/idn_gemm_ws.mojo`, `bindings/_mojolearn_mamba.mojo`, `bindings/_mojolearn_mamba_host.mojo`, `python/mojolearn/_mamba_impl.py`.

### NN41 — reused_existing

Existing IDENTICAL whole-sequence LSTM scan and source correction preventing FAST-wide backward from entering IDENTICAL.

Defines: `MOJOLEARN_IDN_SEQ_LSTM_SCAN`.

Existing IDENTICAL whole-sequence scan is reused rather than reimplemented. Source repair restricts SCAN_WIDE to FAST because its backward folds are a different graph and the host scan uses the original graph.

The inherited FAST scan constant-predictor quality failure remains unresolved. Restricting FAST-wide backward to FAST is a source routing correction; it does not establish the cause or qualify the IDENTICAL arm.

Further research, outside this chosen arm: Explicit bounded-segment launch/state contract beyond existing whole-sequence scan.

Sources: `sequence/recurrent_scan.mojo`, `sequence/recurrent.mojo`, `sequence/exec_device.mojo`.

### NN42 — wired_draft

Sequence per-step recurrent input-bias materialization fused into gate/hidden operation.

Defines: `MOJOLEARN_NN42_INPUT_BIAS_FUSED`.

Fold input projection bias materialization into the per-step recurrent hidden/gate kernel. Each owner stores the same rounded GX word before reading its gates; shared host/GPU element body. Saved GX, gate activations, hidden/cell words and derivative inputs keep their seams.

Further research, outside this chosen arm: Scan path deliberately retains its separate bias launch. Sequence MLP-specific fusion and further derivative-state fusions.

Sources: `sequence/ops.mojo`, `sequence/recurrent.mojo`.

### NN43 — wired_draft

Recurrent weight and bias gradients use fixed128-row leaves, with size-derived portable scratch-cap growth.

Defines: `MOJOLEARN_NN43_WGRAD_FIXED128`.

New recurrent weight/bias profile uses absolute 128-row leaves, incumbent per-leaf chain and ascending partial merges. Host and GPU execute the same generic Exec split-K operations; independent oracle source states the same partition. For exceptional K whose single-cell partials exceed the fixed scratch budget, double the leaf identically on every column until it fits.

Further research, outside this chosen arm: Separate dWeight/dBias attribution switches. Non-recurrent sequence MLP gradient profile caller coverage.

Sources: `sequence/recurrent.mojo`, `sequence/checks/oracle.mojo`, `sequence/ops.mojo`.

### NN44 — wired_draft

Neural MoE stable expert-local ascending pair grouping and upper-bound expert tile lookup.

Defines: `MOJOLEARN_NN44_STABLE_GROUP`, `MOJOLEARN_NN44_EXPERT_BISECT`.

Stable per-expert ascending pair scatter reuses existing device counts/offsets and preserves original pair output positions. Independent integer binary-search expert lookup handles empty experts and retains scheduling semantics. Router/top-k/weighted combine arithmetic remains inherited.

Further research, outside this chosen arm: Scalable parallel stable grouping with explicit workspace for large expert counts. Additional expert gradient/training grouped caller coverage.

Sources: `sequence/moe_group.mojo`, `sequence/moe_tiled.mojo`, `sequence/exec_device.mojo`.

### NN45 — wired_draft

CNN unpooled conv-layout/bias/preactivation/ReLU fusion, with inherited direct convolution continuing its established path.

Defines: `MOJOLEARN_NN45_CONV_RELU`.

Fuse GEMM-output layout, bias, retained preactivation and ReLU for unpooled CNN blocks. Shared host/device element function preserves canonical NaNs, bias rounding and ReLU derivative inputs. Existing direct-convolution path still uses its original separate ReLU.

Further research, outside this chosen arm: Residual epilogue fusion and additional Conv1d/ResNet callers. Pooled block path already has inherited ReLU/maxpool fusion; no new claim.

Sources: `x_cnn/ops.mojo`, `x_cnn/device.mojo`, `x_cnn/host/ops_host.mojo`.

### NN46 — wired_draft

CNN col2im and maxpool backward gathers prune impossible taps using exact stride/dilation bounds.

Defines: `MOJOLEARN_NN46_GATHER_BOUNDS`.

Derive exact candidate tap intervals for col2im and maxpool-backward gathers from stride/dilation/padding inequalities. Keep all original divisibility checks, tap order and floating operations; reversed negative-control path keeps incumbent bounds. Fused pool-ReLU backward consumers share the same maxpool value helper.

Further research, outside this chosen arm: New dWeight tree and shared-memory gather tiling. Host optimized plane routines retain the same old fold; new host scheduling is not claimed.

Sources: `x_cnn/ops.mojo`.

### NN47 — wired_draft

CNN training batchnorm apply also owns running-stat updates once/channel; statistics unchanged.

Defines: `MOJOLEARN_NN47_APPLY_RUNNING`.

Training normalization output and running-state updates share one launch using the same element helper on host/GPU. Exactly one first-image pixel per channel owns running updates after existing statistics; output does not read running state. Existing mean/centered variance folds, biased/unbiased conversions, momentum and epsilon unchanged.

Further research, outside this chosen arm: Separate centered fixed-tree moment profile and additional statistics-load reuse.

Sources: `x_cnn/ops.mojo`, `x_cnn/device.mojo`, `x_cnn/host/ops_host.mojo`.

### NN48 — wired_draft

Neural weighted/mean/inverse/normalized CSR SpMM shares index/value tiles across feature lanes.

Defines: `MOJOLEARN_NN48_CSR_TILES`.

Neural CSR aggregation stages 32 edge columns/weights once per 128-feature tile, preserving ascending edge folds. Weighted, mean, inverse-weight and backward normalization modes retain original floating seams; inactive feature lanes still join barriers. Host retains the incumbent exact arithmetic definition.

Further research, outside this chosen arm: GraphSAGE max-specific tiles and degree-normalization fusion.

Sources: `x_cnn/device.mojo`, `x_cnn/ops.mojo`.

### NN13 — reused_existing

Existing sequence tiled same-chain GEMM callers reused; independent sequence numerical profile retained.

Defines: .

Inherited same-chain sequence tiled GEMM serves neural recurrent/MLP projections; source was not changed by this lane.

Further research, outside this chosen arm: GEMM lane owns any new neural caller/profile API.

Sources: `sequence/gemm_tiled.mojo`, `sequence/exec_device.mojo`.

### NN14 — wired_draft

Bounded-im2col full Conv2d/CNNClassifier forward/recompute, virtual-im2col dWeight and dInput backward, and reduced saved-column allocation.

Defines: `MOJOLEARN_NN14_BOUNDED_IM2COL`.

Forward and CNNClassifier training/recompute tile complete-K GEMMs over bounded row slabs (8MiB cols+y2 target, one indivisible row minimum), scattering to absolute NCHW coordinates. Backward dWeight contracts against generated im2col values using full-row selected GEMM leaf/chain/tree; dInput generates each dcols contraction within the incumbent tap-ordered gather. No full cols/dcols materialization. Standalone Conv2d and resident CNNClassifier block backward select the same shared host/device element functions. Bias gradients stay routed to selected GEMM. Saved-column owner allocates one sentinel word under the native compile-flag query; saved preactivation remains full.

Further research, outside this chosen arm: Tiling/caching generated backward operands for speed without changing the fixed contraction graph.

Sources: `x_cnn/ops.mojo`, `x_cnn/device.mojo`, `x_cnn/host/ops_host.mojo`, `x_cnn/host/gemm_host.mojo`, `bindings/_mojolearn_x_cnn.mojo`, `bindings/_mojolearn_x_cnn_host.mojo`, `python/mojolearn/_expansion_cnn.py`.

## Remaining acceptance

All compile/static checks, same-version host/NVIDIA/AMD/Apple identity, checkpoint/resume and gradient behavior, estimator quality, and full-dataset end-to-end NVIDIA+AMD timing remain unrun. Sample count0. Mathematical equivalence or shared source is not executed evidence. NN34 changes the scan graph and requires particular attention to long sequences, cancellation, strong decay and multi-step training before any promotion.

No unsupported compiler workaround was introduced. NN41 still needs a documented vendor workgroup barrier ordering guarantee if its existing team-barrier semantics prove insufficient; request Modular support rather than add cross-block spin protocols.

Integration follow-up after `d39587b1d` (source only)

The machine-readable `state_cnn_integration_inventory.json` maps NN33–48 and
NN13/14 to exact baseline/candidate defines, runtime APIs, callers, generated
host files, existing experiment IDs/files and full-workload recipe coverage.
Inherited I08/I09 routes, the recurrent scan quality failure, inactive
combinations and unsupported Modular requests remain explicit. No source read
is presented as a compilation, reachability or numerical acceptance result.

The dedicated Mamba host binding now exports the owned-session report and
counter probes used by the shared Python/Samba caller. Host counters describe
snapshot generations, reuse and backward recomputation; they do not claim
retained device stages or GPU transfers.

`tools/bench_board_algos.py race/worker --neural-ab-config PATH` consumes the
selector's schema1 IDENTICAL JSON, propagates a frozen copy and environment to
workers, and retains requested source flags in receipts. It never builds or
admits a binary. Classical lanes and ordinary invocations do not opt in.
The NN45 recipe applies `runtime.estimator_settings.pool_size=1` to both ours
and Torch CNN construction; both arms use the full existing image dataset and
common `MOJOLEARN_XCNN_NO_DIRECT_CONV=1`. Torch skips a pool of size1 to match
our public CNN model semantics. NN14 is absent for this selected fusion arm.
Runtime `training` applies to BN/ResNet layer mode; GraphSAGE's optional
`graphsage_aggregator="mean"` applies only to that lane. An explicit config
refuses smoke-row caps and ours-fast relabeling and uses the existing whole
operation boundary. Original dataset sizes and intrinsic caps are unchanged.
Compilation flags in the receipt are requests, not proof of the loaded build.

No compilation, checker/static/syntax verification, tests, ML execution,
identity, gradient/quality or timing was run. Source generation was unnecessary
for this binding-only follow-up. All acceptance and default promotion remain
pending, including the unresolved inherited recurrent-scan failure.
