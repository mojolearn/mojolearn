# Neural source delivery status

Branch `ideas/neural-identical-ab-20261006-r3`; forked from main `fd6cf80453a6f18eb02e81566c824e7da106ccf0`.

This ledger summarizes source authoring, not verification. No compilation, tests, static checkers, candidate execution, quality evaluation or timing were run. All new candidates remain opt-in. Existing enabled switches are identified in their handoffs; no new default was promoted.

64 idea cards were completed before four-lane programming (root plus three agents). The completion pass connects selected arms to native/public callers and matching arithmetic profiles. Broad research variants, prior source and remaining work are distinct below. A wired arm is source programming, never a claim of successful compilation or qualification. See MODEL_INTEGRATION.md for the selected delivery scope.

| Card | Idea | Recorded source status by owner | Handoff |
| --- | --- | --- | --- |
| NN01 | Attribute GEMM schedules at real neural callers | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN02 | Stream GEMM partial planes into a bounded fold | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN03 | Versioned GEMM leaf lengths | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN04 | Versioned independent accumulator chains inside GEMM leaves | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN05 | Group independent projection jobs | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN06 | Rounded neural GEMM epilogues | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN07 | Reuse transposed operand staging across neural jobs | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN08 | Bounded asynchronous operand loading | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN09 | Shared-memory page count and bank layout | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN10 | Neural GEMM cost-based dispatch | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN11 | Specialize fold storage to the logical tree | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN12 | Model-owned GEMM plan and workspace reuse | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN13 | Tile sequence-family same-chain GEMM | gemm: reused_existing; state_cnn: reused_existing | [gemm.json](lanes/gemm.json); [state_cnn.json](lanes/state_cnn.json) |
| NN14 | Group convolution-lowered GEMMs with bounded im2col | gemm: delegated_source_owner; state_cnn: wired_draft | [gemm.json](lanes/gemm.json); [state_cnn.json](lanes/state_cnn.json) |
| NN15 | Hardware-specific schedules under one profile | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN16 | Avoid unnecessary zero-tail and scratch passes | gemm: wired_draft | [gemm.json](lanes/gemm.json) |
| NN17 | Share attention K/V tiles across query heads | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN18 | Skip structurally masked attention tiles | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN19 | Retain versus recompute attention intermediates | attention: reused_existing | [attention.json](lanes/attention.json) |
| NN20 | Versioned streaming stable attention softmax | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN21 | Fuse attention pointwise score transforms | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN22 | Canonical dK/dV task geometry | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN23 | Fuse attention backward row dot and pointwise gradients | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN24 | Versioned RMSNorm/LayerNorm fixed-lane reductions | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN25 | Separate norm scalar fold from parallel cell scaling | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN26 | Training SwiGLU forward/backward pass fusion | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN27 | Session-owned RoPE frequency and position state | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN28 | Remove training-only dead KV-cache writes | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN29 | Bounded decode KV layout and append fusion | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN30 | Backward residual and gradient buffer views | attention: reused_existing | [attention.json](lanes/attention.json) |
| NN31 | Bounded model activation checkpoint policy | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN32 | Retain Samba attention forward state for backward | attention: wired_draft | [attention.json](lanes/attention.json) |
| NN33 | Attribute SSD/SISO shared tile reuse | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN34 | Versioned absolute-chunk selective scan | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN35 | Parallel causal depthwise-convolution cells | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN36 | Cache Mamba decay exponent values | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN37 | Prune unused triangular SSD work | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN38 | Versioned shared Mamba-3 angle-gradient suffixes | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN39 | Versioned Mamba parameter-gradient folds | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN40 | Mamba immutable weight generations and workspace | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN41 | One-launch ordered recurrent inference/training segments | state_cnn: reused_existing | [state_cnn.json](lanes/state_cnn.json) |
| NN42 | Fuse recurrent gate pointwise updates | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN43 | Versioned recurrent weight and bias gradients | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN44 | MoE stable token grouping and grouped expert jobs | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN45 | Fuse CNN activation/bias/residual passes | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN46 | Deterministic CNN gradient and pooling schedules | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN47 | Neural BatchNorm statistics and running-state fusion | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN48 | Neural graph aggregation tile reuse | state_cnn: wired_draft | [state_cnn.json](lanes/state_cnn.json) |
| NN49 | Stable embedding grouping with touched-row gradients | training: wired_draft | [training.json](lanes/training.json) |
| NN50 | Reuse immutable token grouping across backward uses | training: wired_draft | [training.json](lanes/training.json) |
| NN51 | Resident token and target validation | training: wired_draft | [training.json](lanes/training.json) |
| NN52 | Cross-entropy elementwise pass fusion | training: wired_draft | [training.json](lanes/training.json) |
| NN53 | Stream LM head and exact loss without full logits | training: wired_draft | [training.json](lanes/training.json) |
| NN54 | Versioned loss reduction across tokens and microbatches | training: wired_draft | [training.json](lanes/training.json) |
| NN55 | Fuse optimizer post-update status production | training: wired_draft | [training.json](lanes/training.json) |
| NN56 | Batch optimizer parameter groups with per-group scalars | training: wired_draft | [training.json](lanes/training.json) |
| NN57 | Versioned global gradient-norm reduction | training: wired_draft | [training.json](lanes/training.json) |
| NN58 | Fuse accumulation with gradient finishing | training: wired_draft | [training.json](lanes/training.json) |
| NN59 | Fuse neural dropout RNG and pointwise consumers | residual: wired_draft; training: wired_draft | [residual.json](lanes/residual.json); [training.json](lanes/training.json) |
| NN60 | Parameter/gradient views with generation-safe ownership | training: wired_draft | [training.json](lanes/training.json) |
| NN61 | One final neural step status/readback boundary | training: reused_existing | [training.json](lanes/training.json) |
| NN62 | Live-range neural scratch and activation arenas | training: wired_draft | [training.json](lanes/training.json) |
| NN63 | Deterministic neural multi-device shard merge | training: wired_draft | [training.json](lanes/training.json) |
| NN64 | Neural inference state batching with isolated sessions | training: wired_draft | [training.json](lanes/training.json) |

## Material remaining work

- Per-card handoffs describe the selected native caller, ownership and arithmetic contract. Additional variants in a broad idea are not silently claimed as implemented; see [model integration scope](MODEL_INTEGRATION.md).
- Mamba repository host source was authored through the new explicit source-write-only generator mode. Its post-generation verification and stale comparison were disabled. This writes source, not compiler IR; no compiler or toolchain was patched.
- NN41 preserves an inherited recurrent-scan quality failure in its lane notes. Existence of an opt-in route does not resolve that failure or justify a promotion.
- NN20 preserves earlier attention-v2 nonpromotion evidence. Its newly connected profile cannot inherit qualification or erase prior failures.
- Full dataset/corpus hashes, intrinsic-cap audits, settings, exact workload commands and transitive affected-model coverage must be resolved before a future campaign. No diagnostic fixture substitutes for those recipes.
- All compilation, same-version four-column identity, model-quality and joint NVIDIA/AMD full-workload A/B measurements are intentionally unrun. Apple does not vote on IDENTICAL timing.

## Source evidence

The [idea list](../../docs/plans/NEURAL_IDENTICAL_AB_IDEAS_2026-10-06.md), [catalog](catalog.json), lane JSON/Markdown files and actual Mojo changes are the retained evidence. There are no newly produced build/test/GPU logs or performance results to cite. The metadata-only [planner](../../tools/neural_identical_ab.py) is authored but unexecuted.

For cross-owner NN13/NN14, read both GEMM and state/CNN handoffs: the GEMM owner records its boundary; the state/CNN supplement records the actual caller work. This ledger preserves both rather than erasing partial coverage.
