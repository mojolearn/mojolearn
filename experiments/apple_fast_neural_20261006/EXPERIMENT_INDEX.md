# Neural A/B experiment and file index

Source inventory dated 2026-10-06 for branch `ideas/apple-fast-neural-20261006`, including the kernel/source delivery `1ebead86d` and planning integration `806d87769`. The branch began at local main `fd6cf8045`.

**Not tested.** This document inventories source and authored recipes only. No experiment, compiler, checker, test, or measurement was run to prepare it. Existing manifest status fields and old result comments are historical declarations, not new evidence.

The new catalog contains **44 mechanism cards and eight interaction groups (52 entries), with 116 named A/B variants**. Some cards reuse existing implementations; their kind is shown explicitly. This is not a claim of 44 newly invented kernels. The legacy inventories below are separate and overlap several new catalog cards.

Every path is repository-relative and linked. In the new tables, A and B are the complete bare compiler-define lists from the authored recipe. No experiment defines means the incumbent FAST configuration, not an empty compiler command. Geometry variants retain their parent mechanism in both arms. New definitions are opt-in; no default was promoted.

The global `AFN26-` namespace avoids collisions: `AFN26-A01` is the Apple FAST RMSNorm card; legacy `A01` is an AMD IDENTICAL GEMM experiment. Runtime environment toggles in the older runner are distinguished from compiler defines.

## Index and integration files

| File | Purpose / current status |
| --- | --- |
| [IDEAS.md](IDEAS.md) | Original hypotheses, quality risks, workload map and scope corrections. |
| [README.md](README.md#integration-status-and-entry-points) | Delivery and integration status, including work that remains pending. |
| [catalog.py](catalog.py) | Standalone metadata list/show/select; no build or numerical execution. |
| [tools/apple_fast_neural_ideas.py](../../tools/apple_fast_neural_ideas.py) | Shared catalog adapter, namespaced IDs, binding targets and prospective workload routes. |
| [tools/performance_ideas.py](../../tools/performance_ideas.py) | Main discovery and `plan --variant` entry point; AFN26 execution is disabled. |
| [tools/neural_experiments.py](../../tools/neural_experiments.py) | New `--apple-fast-list` / `--apple-fast-plan`; also contains the separate historical runtime-toggle runner. |
| [tools/afn_ab.sh](../../tools/afn_ab.sh) | New `--experiment-list` / `--experiment-plan`; legacy positional entry builds and measures and was not invoked. |
| [tools/bench_board_neural.py](../../tools/bench_board_neural.py) | Existing public neural workload definitions; actual routing, caps and full-dataset coverage remain to be established for new arms. |
| [tools/neural_fast_quality.py](../../tools/neural_fast_quality.py) and [tools/afn_ab.py](../../tools/afn_ab.py) | Existing task-quality and paired-output tools; not run here. |

Runtime source controls and planning entry points are programmed. **Complete executable full-workload A/B harness integration is still pending**, especially true MLP multistep and full CNN/embedding consumers. Binding/driver references below do not prove that a runtime branch is reached. Compile acceptance, correctness, quality, measurements, board admission and merging to main are not established.

## New catalog overview

| Family | IDs | Recipe | Source notes |
| --- | --- | --- | --- |
| Attention and transformer projections | `AFN26-A01`–`AFN26-A12` | [attention.json](attention.json) | [attention.md](attention.md) |
| Training, losses, optimizers and MLP | `AFN26-T01`–`AFN26-T12` | [training.json](training.json) | [training.md](training.md) |
| Mamba and Samba | `AFN26-M01`–`AFN26-M10` | [mamba.json](mamba.json) | [mamba.md](mamba.md) |
| CNN and neural embedding | `AFN26-E01`–`AFN26-E10` | [cnn_embedding.json](cnn_embedding.json) | [cnn_embedding.md](cnn_embedding.md) |
| Combined A/B configurations | `AFN26-X01`–`AFN26-X08` | [interactions.json](interactions.json) | [IDEAS.md](IDEAS.md#interaction-recipes-after-single-mechanism-qualification-never-run-here) |

## New catalog: Attention and transformer projections

Recipe: [attention.json](attention.json). Every entry and variant below is **not tested**.

### AFN26-A01 — Cooperative RMSNorm

**Kind:** `existing_arm`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `enable_norm_sg` | No experiment defines | `MOJOLEARN_AFN_ATTN_NORM_SG` |

**Scope notes:** Keep epsilon, vector-four input contract and f32 accumulation; reduction reassociation requires existing quality gates. Cover norm1, residual+norm2 and chained next-norm callers, including tail token rows.

### AFN26-A02 — Fused RoPE and cache append

**Kind:** `existing_arm`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `enable_rope_cache` | No experiment defines | `MOJOLEARN_AFN_ATTN_ROPE_CACHE` |

**Scope notes:** Fresh full-causal prefill only; carried cache and sliding-window cases retain orchestration fallback. Compare absolute positions, rotary passthrough columns, packed cache layout and continuation.

### AFN26-A03 — Online-softmax FLASH attention

**Kind:** `existing_arm`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `enable_flash` | No experiment defines | `MOJOLEARN_AFN_ATTN_FLASH` |

**Scope notes:** Historical component timings remain documented in source; they are not full-workload qualification for this campaign. Cover masks/window, empty visible rows, head padding, amax/denom outputs and consumed context.

### AFN26-A04 — Grouped-query FLASH tile reuse

**Kind:** `existing_arm`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `grouped_vs_flash` | `MOJOLEARN_AFN_ATTN_FLASH` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN_ATTN_GQA_TILE` |

**Scope notes:** FLASH is enabled on both sides; GQA_TILE implicitly implies FLASH, but the recipe keeps the common parent explicit. Cover n_rep 1, 2, 4 and unsupported-group fallback, with unchanged head mapping. Historical mixed component outcomes are preserved.

### AFN26-A05 — Normalized fused projection staging

**Kind:** `existing_arm`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `enable_fuse_pre` | No experiment defines | `MOJOLEARN_AFN_ATTN_FUSE_PRE` |

**Scope notes:** Complete mechanism includes the cooperative sums-of-squares path; fresh-pre QKV RoPE/cache and foldable norm2 gate/up are separately affected. Backward/materialized stages remain on existing fallback.

### AFN26-A06 — SwiGLU and residual projection epilogues

**Kind:** `existing_arm`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `enable_fuse_mlp` | No experiment defines | `MOJOLEARN_AFN_ATTN_FUSE_MLP` |

**Scope notes:** Exercise gate/up, output residual and down residual through complete forward-only callers. Keep activation formula, f32 precision and residual semantics; no approximation is introduced.

### AFN26-A07 — Transformer binding arena

**Kind:** `existing_arm`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo).

**Affected operations:** TransformerBlock full public operation, including SambaStack transformer binding callers.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `enable_arena` | No experiment defines | `MOJOLEARN_AFN_ATTN_ARENA` |

**Scope notes:** Allocation and submission experiment in the standalone transformer binding; include cold and repeated calls plus resize/cache lifetimes. ByteLM uses its own orchestration: do not claim this arena affects its internal block allocations.

### AFN26-A08 — RMSNorm-only 128/512-thread launches

**Kind:** `new_variant`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `norm_128` | `MOJOLEARN_AFN_ATTN_NORM_SG` | `MOJOLEARN_AFN_ATTN_NORM_SG`<br>`MOJOLEARN_AFN26_ATTN_NORM_TPB128` |
| `norm_512` | `MOJOLEARN_AFN_ATTN_NORM_SG` | `MOJOLEARN_AFN_ATTN_NORM_SG`<br>`MOJOLEARN_AFN26_ATTN_NORM_TPB512` |
| `pre_norm_128` | `MOJOLEARN_AFN_ATTN_FUSE_PRE` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN26_ATTN_NORM_TPB128` |
| `pre_norm_512` | `MOJOLEARN_AFN_ATTN_FUSE_PRE` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN26_ATTN_NORM_TPB512` |

**Scope notes:** Four or sixteen simdgroups replace eight only in RMSNorm; FLASH, RoPE/cache and projection launches retain their independent sizes. Selecting both widths is a compile-time configuration error; neither flag enables a parent mechanism. All targets and quality remain untested.

### AFN26-A09 — FLASH key tile 16

**Kind:** `new_variant`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `flash_bk16` | `MOJOLEARN_AFN_ATTN_FLASH` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN26_ATTN_FLASH_BK16` |
| `gqa_bk16` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN_ATTN_GQA_TILE` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN_ATTN_GQA_TILE`<br>`MOJOLEARN_AFN26_ATTN_FLASH_BK16` |

**Scope notes:** Two keys per each of eight softmax lanes replace four; score fragments, shared strides/pages and key loops derive from BK. Existing compiled padded head classes remain 16..80; do not claim expanded head coverage or quality from a smaller page.

### AFN26-A10 — FLASH query tile 16

**Kind:** `new_variant`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `flash_tq16` | `MOJOLEARN_AFN_ATTN_FLASH` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN26_ATTN_FLASH_TQ16` |
| `gqa_tq16` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN_ATTN_GQA_TILE` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN_ATTN_GQA_TILE`<br>`MOJOLEARN_AFN26_ATTN_FLASH_TQ16` |

**Scope notes:** The candidate uses four simdgroups/128 threads to preserve eight softmax lanes per query row and integral context fragments for every existing head class. GROUP4 then has four tokens per query head; each eight-row matrix fragment may span two heads that share K/V. This new combination is explicitly untested. Pair with BK16 only in a separate interaction arm; no geometry or quality result is claimed.

### AFN26-A11 — Fused neural projection K tile 16

**Kind:** `new_variant`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `pre_kb16` | `MOJOLEARN_AFN_ATTN_FUSE_PRE` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN26_ATTN_PROJ_KB16` |
| `mlp_kb16` | `MOJOLEARN_AFN_ATTN_FUSE_MLP` | `MOJOLEARN_AFN_ATTN_FUSE_MLP`<br>`MOJOLEARN_AFN26_ATTN_PROJ_KB16` |
| `pre_mlp_kb16` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN_ATTN_FUSE_MLP` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN_ATTN_FUSE_MLP`<br>`MOJOLEARN_AFN26_ATTN_PROJ_KB16` |

**Scope notes:** Shared allocation is the maximum of operand staging and the later RoPE C tile, with the existing barrier separating lifetimes. Whole-K-window routing follows KB; k divisible by 16 but not 32 can enter the candidate while baseline falls back. Treat newly covered shapes separately from same-route geometry. Only neural afn projection kernels change; generic GEMM remains untouched.

### AFN26-A12 — Fused neural projection output-row tile 32

**Kind:** `new_variant`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

**Affected operations:** TransformerBlock forward-only full operation; ByteLM full-corpus forward/held-out evaluation reaching transformer forward-only; SambaStack full-corpus mixed-stack forward reaching TransformerBlock.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `pre_bm32` | `MOJOLEARN_AFN_ATTN_FUSE_PRE` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN26_ATTN_PROJ_BM32` |
| `mlp_bm32` | `MOJOLEARN_AFN_ATTN_FUSE_MLP` | `MOJOLEARN_AFN_ATTN_FUSE_MLP`<br>`MOJOLEARN_AFN26_ATTN_PROJ_BM32` |
| `pre_mlp_bm32` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN_ATTN_FUSE_MLP` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN_ATTN_FUSE_MLP`<br>`MOJOLEARN_AFN26_ATTN_PROJ_BM32` |

**Scope notes:** The 256-thread block keeps eight simdgroups; each row subgroup owns half as many row fragments. Grid, row statistics, shared strides and epilogues derive from BM. A-staging uses rounded-up predicated slots so BM32+KB16 still loads the complete tile when float4 slots are fewer than threads. Column tile remains 64 so RoPE partner ownership is unchanged; full row tails and prefill/cache semantics remain pending.


## New catalog: Training, losses, optimizers and MLP

Recipe: [training.json](training.json). Every entry and variant below is **not tested**.

### AFN26-T01 — One-completion LM step

**Kind:** `existing_arm`. **Files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `one_completion` | No experiment defines | `MOJOLEARN_AFN26_LM_NOSYNC` |

**Scope notes:** AFN26 alias selects the existing one-completion path; no default changed. Future full-step boundary includes token upload, status and loss download, final completion, consumed outputs and transactional refusal/recovery. Default-on LM backward no-sync remains fixed in A and B.

### AFN26-T02 — Fused byte-LM head loss and dlogits

**Kind:** `existing_arm`. **Files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `head_fused` | No experiment defines | `MOJOLEARN_AFN26_LM_HEAD_FUSE` |

**Scope notes:** Existing implementation uses reset plus per-row CE kernel; f32 atomic loss mean can change bits. Compare complete training trajectory and held-out loss; preserve target validation, divisor and non-finite behavior. Chunked head configurations may bypass this arm; future recipe must document reached route rather than treat the flag as proof.

### AFN26-T03 — Flat LM parameter and gradient views

**Kind:** `existing_arm`. **Files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `parameter_views` | No experiment defines | `MOJOLEARN_AFN26_LM_PARAM_VIEWS` |

**Scope notes:** Existing view routing avoids per-layer packing/unpacking; checkpoint/resume, registry offsets, mutations and aliased-buffer lifetime require future qualification. Include cold construction, full steps, and state/save/resume separately.

### AFN26-T04 — Occupancy-driven split weight-gradient products

**Kind:** `existing_arm`. **Files:** [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending; SambaStack full samba-train-step where transformer backward reaches these switches; full-workload route mapping pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `wgrad_split` | No experiment defines | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT` |

**Scope notes:** Existing split implementation zeroes output then atomically accumulates split GEMM partial tiles; it does not allocate split scratch or run a separate final fold. Compare complete training with gradient/trajectory quality and stochastic atomic fold sensitivity; include zero launch and atomic contention.

### AFN26-T05 — Independent backward fusion paths

**Kind:** `existing_arm`. **Files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo), [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending; SambaStack full samba-train-step where transformer backward reaches these switches; full-workload route mapping pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `backward_fusion` | No experiment defines | `MOJOLEARN_AFN26_LM_BWD_FUSE` |
| `gemm_epilogue_fan_in` | No experiment defines | `MOJOLEARN_AFN26_LM_BWD_EPILOGUE` |
| `norm1_residual` | No experiment defines | `MOJOLEARN_AFN26_LM_BWD_NORM1_RESID` |
| `three_backward_fusions` | No experiment defines | `MOJOLEARN_AFN26_LM_BWD_FUSE`<br>`MOJOLEARN_AFN26_LM_BWD_EPILOGUE`<br>`MOJOLEARN_AFN26_LM_BWD_NORM1_RESID` |

**Scope notes:** Each named AFN26 alias independently selects the existing mechanism; the combined arm lists all three explicitly. Keep incumbent LM backward no-sync default unchanged in both arms. Check full backward gradients, residual fan-in and trajectory quality.

### AFN26-T06 — Optimizer and loss orchestration alternatives

**Kind:** `existing_arm`. **Files:** [training/afn_optim.mojo](../../training/afn_optim.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending; SmallMLPTrainer full declared mlp-train-step workload; tools/bench_board_neural.py; preserve public row cap and cover complete training sequence; SambaStack full samba-train-step where shared optimizer/loss is reached; Neural optimizer consumers using SGD, Adam or AdamW and cross-entropy; full dataset recipes and reachability pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `refusal_scan_fusion` | No experiment defines | `MOJOLEARN_AFN26_OPT_FUSE_SCAN` |
| `clip_fusion` | No experiment defines | `MOJOLEARN_AFN26_OPT_CLIP_FUSE` |
| `sgd_multitensor` | No experiment defines | `MOJOLEARN_AFN26_OPT_MULTITENSOR` |
| `adam_vector4` | No experiment defines | `MOJOLEARN_AFN26_OPT_VEC4` |
| `resident_scratch` | No experiment defines | `MOJOLEARN_AFN26_OPT_RESIDENT_STATE` |
| `fused_cross_entropy` | No experiment defines | `MOJOLEARN_AFN26_LOSS_FUSED` |
| `adam_optimizer_combined` | No experiment defines | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` |
| `sgd_optimizer_combined` | No experiment defines | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_MULTITENSOR`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` |
| `adam_loss_combined` | No experiment defines | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_LOSS_FUSED` |
| `sgd_loss_combined` | No experiment defines | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_MULTITENSOR`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_LOSS_FUSED` |

**Scope notes:** Prior measured losses/wins remain in source and are not repeated evidence for this campaign. Shared loss path uses three kernel launches (row, partial reduction, final fold), preserving ignore index, smoothing and reduction denominator. A neural consumer's actual route must reach this shared optimizer/loss; one-completion LM, chunked LM heads and fused MLP can bypass it. No opponent-only or optimizer component result qualifies estimator-wide performance. Cold pool allocation and repeated use are separate boundaries.

### AFN26-T07 — Fused SmallMLP full step

**Kind:** `existing_arm`. **Files:** [training/mlp_fast.mojo](../../training/mlp_fast.mojo).

**Affected operations:** SmallMLPTrainer full declared mlp-train-step workload; tools/bench_board_neural.py; preserve public row cap and cover complete training sequence.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `fused_step` | No experiment defines | `MOJOLEARN_AFN26_MLP_FUSED_STEP` |

**Scope notes:** Uses existing public SmallMLP architecture and two-launch GPU path; no dimension dispatch introduced. Future quality includes logits, input/parameter gradients, moments, loss, refusal and full training trajectory.

### AFN26-T08 — Resident and multistep SmallMLP execution

**Kind:** `existing_arm`. **Files:** [training/mlp_fast.mojo](../../training/mlp_fast.mojo).

**Affected operations:** SmallMLPTrainer full declared mlp-train-step workload; tools/bench_board_neural.py; preserve public row cap and cover complete training sequence.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `resident` | `MOJOLEARN_AFN26_MLP_FUSED_STEP` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT` |
| `multistep` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT`<br>`MOJOLEARN_AFN26_MLP_MULTISTEP` |

**Scope notes:** Resident and multistep are separate transport experiments. The multistep control implies resident in source, which is still explicit on both recipe sides. Python/binding caller must explicitly exercise advertised resident/multistep methods; a compiled flag alone is not evidence they ran. Preserve downloads, rollback, loss/logit outputs, batch order and checkpoint/resume. Do not redefine batch size or omit per-step outcomes.

### AFN26-T09 — Weight-gradient split minimum work and split cap

**Kind:** `new_variant`. **Files:** [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending; SambaStack full samba-train-step where transformer backward reaches these switches; full-workload route mapping pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `minimum_k128` | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT` | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT`<br>`MOJOLEARN_AFN26_LM_WGRAD_MIN128` |
| `split_cap8` | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT` | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT`<br>`MOJOLEARN_AFN26_LM_WGRAD_CAP8` |
| `minimum_k128_cap8` | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT` | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT`<br>`MOJOLEARN_AFN26_LM_WGRAD_MIN128`<br>`MOJOLEARN_AFN26_LM_WGRAD_CAP8` |

**Scope notes:** Incumbent minimum K per split is 256 and cap is 16; candidate minimum 128 and cap 8 are independent. Policy continues to use output tile count, hardware occupancy target, K work and KB alignment for all neighboring shapes; no benchmark-name or exact-size dispatch. Smaller minimum can increase parallelism and atomic overhead; smaller cap bounds atomic fan-in. Neither geometry define enables the parent.

### AFN26-T10 — Optimizer launch blocks of 128 or 512 threads

**Kind:** `new_variant`. **Files:** [training/afn_optim.mojo](../../training/afn_optim.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending; SmallMLPTrainer full declared mlp-train-step workload; tools/bench_board_neural.py; preserve public row cap and cover complete training sequence; SambaStack full samba-train-step where shared optimizer/loss is reached; Neural optimizer consumers using SGD, Adam or AdamW and cross-entropy; full dataset recipes and reachability pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `scan_block128` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_BLOCK128` |
| `clip_block128` | `MOJOLEARN_AFN26_OPT_CLIP_FUSE` | `MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_BLOCK128` |
| `adam_vector_block128` | `MOJOLEARN_AFN26_OPT_VEC4` | `MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_BLOCK128` |
| `sgd_multitensor_block128` | `MOJOLEARN_AFN26_OPT_MULTITENSOR` | `MOJOLEARN_AFN26_OPT_MULTITENSOR`<br>`MOJOLEARN_AFN26_OPT_BLOCK128` |
| `adam_combined_block128` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_OPT_BLOCK128` |
| `sgd_combined_block128` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_MULTITENSOR`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_MULTITENSOR`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_OPT_BLOCK128` |
| `scan_block512` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_BLOCK512` |
| `clip_block512` | `MOJOLEARN_AFN26_OPT_CLIP_FUSE` | `MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_BLOCK512` |
| `adam_vector_block512` | `MOJOLEARN_AFN26_OPT_VEC4` | `MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_BLOCK512` |
| `sgd_multitensor_block512` | `MOJOLEARN_AFN26_OPT_MULTITENSOR` | `MOJOLEARN_AFN26_OPT_MULTITENSOR`<br>`MOJOLEARN_AFN26_OPT_BLOCK512` |
| `adam_combined_block512` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_OPT_BLOCK512` |
| `sgd_combined_block512` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_MULTITENSOR`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` | `MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_CLIP_FUSE`<br>`MOJOLEARN_AFN26_OPT_MULTITENSOR`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_OPT_BLOCK512` |

**Scope notes:** Incumbent optimizer block is 256; power-of-two candidates share one constant across scan scratch, grid, reductions and update launches. CE uses a separate constant so this card cannot silently change loss geometry. Choose one block define per family; 128 has deterministic precedence if both are accidentally supplied. Parent mechanisms are held equal in A and B. Refusal and clipping reductions can change fold order; vector width and update equations are unchanged.

### AFN26-T11 — CE-specific legal launch and reduction geometries

**Kind:** `new_variant`. **Files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo), [training/afn_optim.mojo](../../training/afn_optim.mojo).

**Affected operations:** LanguageModelTrainer full lm-train-step workload; tools/bench_board_neural.py corpus and full-shape recipe; full-dataset coverage mapping pending; SmallMLPTrainer full declared mlp-train-step workload; tools/bench_board_neural.py; preserve public row cap and cover complete training sequence; SambaStack full samba-train-step where shared optimizer/loss is reached; Neural optimizer consumers using SGD, Adam or AdamW and cross-entropy; full dataset recipes and reachability pending.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `lm_head_block128` | `MOJOLEARN_AFN26_LM_HEAD_FUSE` | `MOJOLEARN_AFN26_LM_HEAD_FUSE`<br>`MOJOLEARN_AFN26_LM_CE_BLOCK128` |
| `loss_block128` | `MOJOLEARN_AFN26_LOSS_FUSED` | `MOJOLEARN_AFN26_LOSS_FUSED`<br>`MOJOLEARN_AFN26_LOSS_BLOCK128` |
| `resident_loss_block128` | `MOJOLEARN_AFN26_LOSS_FUSED`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` | `MOJOLEARN_AFN26_LOSS_FUSED`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_LOSS_BLOCK128` |
| `lm_head_block512` | `MOJOLEARN_AFN26_LM_HEAD_FUSE` | `MOJOLEARN_AFN26_LM_HEAD_FUSE`<br>`MOJOLEARN_AFN26_LM_CE_BLOCK512` |
| `loss_block512` | `MOJOLEARN_AFN26_LOSS_FUSED` | `MOJOLEARN_AFN26_LOSS_FUSED`<br>`MOJOLEARN_AFN26_LOSS_BLOCK512` |
| `resident_loss_block512` | `MOJOLEARN_AFN26_LOSS_FUSED`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` | `MOJOLEARN_AFN26_LOSS_FUSED`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_LOSS_BLOCK512` |

**Scope notes:** LM CE uses 256 incumbent threads, candidate 128 or 512; status-scan geometry remains unchanged. General loss retains 32 threads when vocabulary fits one Apple simdgroup, and changes the incumbent wider-row 256 to 128 or 512; row/partial/final-fold templates and launch dimensions use that same CE geometry. These are complete power-of-two reductions with strided coverage of all classes and rows; ignore index, label smoothing, divisor, gate writes and full output writes are retained. Small-vocabulary general-loss cases intentionally have identical geometry and are scope controls, not claimed affected cells.

### AFN26-T12 — SmallMLP row tiles of 32 or 128

**Kind:** `new_variant`. **Files:** [training/mlp_fast.mojo](../../training/mlp_fast.mojo).

**Affected operations:** SmallMLPTrainer full declared mlp-train-step workload; tools/bench_board_neural.py; preserve public row cap and cover complete training sequence.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `fused_rows32` | `MOJOLEARN_AFN26_MLP_FUSED_STEP` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_ROWS32` |
| `resident_rows32` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT`<br>`MOJOLEARN_AFN26_MLP_ROWS32` |
| `multistep_rows32` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT`<br>`MOJOLEARN_AFN26_MLP_MULTISTEP` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT`<br>`MOJOLEARN_AFN26_MLP_MULTISTEP`<br>`MOJOLEARN_AFN26_MLP_ROWS32` |
| `fused_rows128` | `MOJOLEARN_AFN26_MLP_FUSED_STEP` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_ROWS128` |
| `resident_rows128` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT`<br>`MOJOLEARN_AFN26_MLP_ROWS128` |
| `multistep_rows128` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT`<br>`MOJOLEARN_AFN26_MLP_MULTISTEP` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT`<br>`MOJOLEARN_AFN26_MLP_MULTISTEP`<br>`MOJOLEARN_AFN26_MLP_ROWS128` |

**Scope notes:** Incumbent rows per block is 64; selected tile controls threadgroup activation storage, row ownership, gradient folding and launch grid. Maximum resident partial slots are ceil(public 256-row step capacity / selected row tile), removing the hardcoded four-slot assumption. Public row cap stays fixed. Multistep submissions reuse partial slots sequentially on the ordered context. Full batches, partial tiles and complete logical step sequences need future qualification. 32 trades smaller shared storage for more partials; 128 trades fewer partials for larger shared storage. No shape-specific choice is made.


## New catalog: Mamba and Samba

Recipe: [mamba.json](mamba.json). Every entry and variant below is **not tested**.

### AFN26-M01 — Mamba1 serial recurrence versus existing chunk scan

**Kind:** `existing_arm`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/ops/afn_selective_scan.mojo](../../mamba/impl/ops/afn_selective_scan.mojo), [mamba/impl/ops/selective_scan_interface.mojo](../../mamba/impl/ops/selective_scan_interface.mojo).

**Affected operations:** Mamba1Block.forward full workload, cold and repeated; Mamba1 continuation and returned state; Samba full forward and training configurations that contain Mamba1.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `chunkscan32` | No experiment defines | `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN` |

**Scope notes:** Historical component evidence stays in the source; this campaign has no new qualification. Keep input fusion, arena and refusal geometry fixed. Compare all outputs, incoming/final state, recurrence stability and full consumer quality.

### AFN26-M02 — Mamba1 existing convolution and projection-input fusion

**Kind:** `existing_arm`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/modeling/afn_mamba1_fused.mojo](../../mamba/impl/modeling/afn_mamba1_fused.mojo).

**Affected operations:** Mamba1Block.forward full workload, cold and repeated; Mamba1 stateful continuation; Samba full forward and training configurations that contain Mamba1.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `input_fusion` | No experiment defines | `MOJOLEARN_AFN_MAMBA1_FUSE_IN` |

**Scope notes:** Keep recurrence algorithm fixed to isolate input fusion. Quality includes causal convolution window state, dt/A transforms, outputs and training consumers; no current result is claimed.

### AFN26-M03 — Mamba2 existing f32 SSD matrix-unit products

**Kind:** `existing_arm`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/modules/afn_ssd_mma.mojo](../../mamba/impl/modules/afn_ssd_mma.mojo), [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo).

**Affected operations:** Mamba2Block.forward full workload, cold and repeated; Mamba2 warm-state continuation and every returned state; Samba full forward and training configurations that contain Mamba2.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `ssd_mma_k32` | No experiment defines | `MOJOLEARN_AFN_MAMBA2_SSD_MMA` |

**Scope notes:** Includes C.B, causal diagonal output and carried-state products with their normal dispatch/fallback behavior. Full workload must exercise the intended route; component timings and the generic GEMM lane are not qualification.

### AFN26-M04 — Mamba3 existing fused SISO elementwise stages

**Kind:** `existing_arm`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/ops/afn_mamba3_fused.mojo](../../mamba/impl/ops/afn_mamba3_fused.mojo).

**Affected operations:** Mamba3Block.forward full workload, cold and repeated; Mamba3 continuation and h/k/v/theta reports; Samba full forward and full training with Mamba3 blocks.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `siso_fused128` | No experiment defines | `MOJOLEARN_AFN_MAMBA3_SISO_FUSED` |

**Scope notes:** Keep all four fused launches at incumbent 128 threads for this card. Compare every state/report and downstream training quality; angle chain and optimizer settings stay fixed.

### AFN26-M05 — Mamba block per-buffer allocation versus existing arena

**Kind:** `existing_arm`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/modules/afn_arena.mojo](../../mamba/impl/modules/afn_arena.mojo).

**Affected operations:** Mamba1Block.forward full workload; Mamba2Block.forward full workload; Mamba3Block.forward full workload; Samba full forward and training where arena-backed block callers are reached.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `arena` | No experiment defines | `MOJOLEARN_AFN_MAMBA_ARENA` |

**Scope notes:** Separate cold allocation from repeated use; include upload, completion and consumed outputs. Exercise shape changes, output alias lifetimes, state initialization and failure cleanup. Existing IDENTICAL arena policy is unchanged.

### AFN26-M06 — Mandatory device refusal: scalar versus contiguous vector4 loads

**Kind:** `new_variant`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/modules/afn_refusal.mojo](../../mamba/impl/modules/afn_refusal.mojo).

**Affected operations:** Mamba1Block.forward full workload including named-input refusal; Mamba2Block.forward full workload including named-input refusal; Mamba3Block.forward full workload including named-input refusal; Samba full forward and training where AfnRefusalBatch is reached.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `refusal_vec4` | No experiment defines | `MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4` |

**Scope notes:** The initial host-versus-device idea is superseded: device refusal is already mandatory; its legacy enable define does not produce distinct A/B arms. New vector4 scan retains scalar tails, integer first-index selection, name order, messages and the same single finish readback. It introduces no host-data path. Future correctness coverage includes non-finite entries in every vector lane, multiple errors, nonmultiple-of-four extents, empty names and refusal before state publication.

### AFN26-M07 — Mamba1 chunk-scan parallelism: 16 or 64 chunks versus 32

**Kind:** `new_variant`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/ops/afn_selective_scan.mojo](../../mamba/impl/ops/afn_selective_scan.mojo).

**Affected operations:** Mamba1Block.forward full workload; Mamba1 continuation and returned state; Samba full forward and training configurations that contain Mamba1.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `chunks16` | `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN` | `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN`<br>`MOJOLEARN_AFN26_MAMBA1_CHUNKS16` |
| `chunks64` | `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN` | `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN`<br>`MOJOLEARN_AFN26_MAMBA1_CHUNKS64` |

**Scope notes:** Both arms enable the same chunk-scan parent. Do not select both chunk flags. Logical chunks own separate summaries; the 16-chunk arm launches 32 lanes and the 64-chunk arm launches 64. Empty/surplus lanes do not write final state. Trade shared storage and serial carry length against time-axis parallelism; retain every token and the walked final state. Quality must cover long recurrence and ragged/short sequences.

### AFN26-M08 — Mamba2 SSD staged K window: 16 versus 32

**Kind:** `new_variant`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/modules/afn_ssd_mma.mojo](../../mamba/impl/modules/afn_ssd_mma.mojo).

**Affected operations:** Mamba2Block.forward full workload; Mamba2 warm-state continuation and returned state; Samba full forward and training configurations that contain Mamba2.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `ssd_k16` | `MOJOLEARN_AFN_MAMBA2_SSD_MMA` | `MOJOLEARN_AFN_MAMBA2_SSD_MMA`<br>`MOJOLEARN_AFN26_MAMBA2_SSD_K16` |

**Scope notes:** Both arms enable SSD MMA; K16 does not enable it on its own. Shared extents, operand stride, staging loops and fragment loops derive from the K tile; causal clipping and the full K extent remain unchanged. Reduced shared storage may lose to extra barriers. Existing fallback applies outside the legal MMA tile plan.

### AFN26-M09 — Mamba3 fused elementwise blocks: 64 or 256 threads versus 128

**Kind:** `new_variant`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/ops/afn_mamba3_fused.mojo](../../mamba/impl/ops/afn_mamba3_fused.mojo).

**Affected operations:** Mamba3Block.forward full workload; Mamba3 continuation and all state reports; Samba full forward and full training with Mamba3 blocks.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `threads64` | `MOJOLEARN_AFN_MAMBA3_SISO_FUSED` | `MOJOLEARN_AFN_MAMBA3_SISO_FUSED`<br>`MOJOLEARN_AFN26_MAMBA3_THREADS64` |
| `threads256` | `MOJOLEARN_AFN_MAMBA3_SISO_FUSED` | `MOJOLEARN_AFN_MAMBA3_SISO_FUSED`<br>`MOJOLEARN_AFN26_MAMBA3_THREADS256` |

**Scope notes:** Both arms enable the same SISO fusion; choose one thread-count flag. All four launch grids use the selected block size and cell indexing uses block_dim.x. Quality covers every stage/report and consumer training behavior.

### AFN26-M10 — Device refusal: smaller reductions and four-element work target

**Kind:** `new_variant`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/modules/afn_refusal.mojo](../../mamba/impl/modules/afn_refusal.mojo).

**Affected operations:** Mamba1Block.forward full workload including refusal; Mamba2Block.forward full workload including refusal; Mamba3Block.forward full workload including refusal; Samba full forward and training where AfnRefusalBatch is reached.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `threads128` | No experiment defines | `MOJOLEARN_AFN26_MAMBA_REFUSAL_THREADS128` |
| `grid_work4` | No experiment defines | `MOJOLEARN_AFN26_MAMBA_REFUSAL_GRID4` |
| `threads128_grid_work4` | No experiment defines | `MOJOLEARN_AFN26_MAMBA_REFUSAL_THREADS128`<br>`MOJOLEARN_AFN26_MAMBA_REFUSAL_GRID4` |
| `vec4_grid_work4` | `MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4` | `MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4`<br>`MOJOLEARN_AFN26_MAMBA_REFUSAL_GRID4` |
| `vec4_threads128_grid_work4` | `MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4` | `MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4`<br>`MOJOLEARN_AFN26_MAMBA_REFUSAL_THREADS128`<br>`MOJOLEARN_AFN26_MAMBA_REFUSAL_GRID4` |

**Scope notes:** Incumbent already launches min(ceil(n/256),32) partial blocks; GRID4 changes the work target, not the existence of work-sized dispatch. Candidate grid is min(ceil(n/(threads*4)),32); grid-stride scalar/vector loops cover the entire extent. Scratch stays bounded to 32 partials/name, with unused columns initialized to NONE. Name order and minimum error code preserve refusal precedence; include sparse and multiple invalid entries beyond the first grid pass.


## New catalog: CNN and neural embedding

Recipe: [cnn_embedding.json](cnn_embedding.json). Every entry and variant below is **not tested**.

### AFN26-E01 — Implicit convolution with bias/ReLU epilogue

**Kind:** `existing_arm`. **Files:** [x_cnn/afn_direct.mojo](../../x_cnn/afn_direct.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo).

**Affected operations:** Conv2d.forward/backward; CNNClassifier.fit/predict_proba.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `direct` | No experiment defines | `MOJOLEARN_AFN_CNN_DIRECT` |

**Scope notes:** Retains the incumbent direct-kernel admission; compare full CNN fit and inference, not an isolated convolution.

### AFN26-E02 — Convolution spatial tile 32

**Kind:** `new_variant`. **Files:** [x_cnn/afn_direct.mojo](../../x_cnn/afn_direct.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo).

**Affected operations:** Conv2d.forward/backward; CNNClassifier.fit/predict_proba.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `rows32` | `MOJOLEARN_AFN_CNN_DIRECT` | `MOJOLEARN_AFN_CNN_DIRECT`<br>`MOJOLEARN_AFN26_CNN_ROWS32` |

**Scope notes:** All row tails retain predicates; tile size trades staged storage against input reloads and block count.

### AFN26-E03 — Convolution channel tile 16

**Kind:** `new_variant`. **Files:** [x_cnn/afn_direct.mojo](../../x_cnn/afn_direct.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo).

**Affected operations:** Conv2d.forward/backward; CNNClassifier.fit/predict_proba.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `channels16` | `MOJOLEARN_AFN_CNN_DIRECT` | `MOJOLEARN_AFN_CNN_DIRECT`<br>`MOJOLEARN_AFN26_CNN_CHANNELS16` |

**Scope notes:** All channel tails retain predicates; fewer accumulators per thread can reduce register pressure.

### AFN26-E04 — Convolution reduction tile 16

**Kind:** `new_variant`. **Files:** [x_cnn/afn_direct.mojo](../../x_cnn/afn_direct.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo).

**Affected operations:** Conv2d.forward/backward; CNNClassifier.fit/predict_proba.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `k16` | `MOJOLEARN_AFN_CNN_DIRECT` | `MOJOLEARN_AFN_CNN_DIRECT`<br>`MOJOLEARN_AFN26_CNN_K16` |

**Scope notes:** Complete reduction remains f32; smaller staged windows add barriers.

### AFN26-E05 — Single owner for saved convolution columns

**Kind:** `new_variant`. **Files:** [x_cnn/afn_direct.mojo](../../x_cnn/afn_direct.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo).

**Affected operations:** Conv2d.forward/backward; CNNClassifier.fit/predict_proba.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `columns_once` | `MOJOLEARN_AFN_CNN_DIRECT` | `MOJOLEARN_AFN_CNN_DIRECT`<br>`MOJOLEARN_AFN26_CNN_COLS_ONCE` |

**Scope notes:** Output-channel tile zero writes saved columns; this ownership condition is structural, not dimension targeting. Only backward-saving callers benefit.

### AFN26-E06 — Atomic embedding backward and float4 forward

**Kind:** `existing_arm`. **Files:** [embedding/checks/embedding_fast_apple.mojo](../../embedding/checks/embedding_fast_apple.mojo), [bindings/_mojolearn_embedding.mojo](../../bindings/_mojolearn_embedding.mojo).

**Affected operations:** Embedding.forward/backward; full neural consumers of the public Embedding binding.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `atomic` | No experiment defines | `MOJOLEARN_AFN_EMB_ATOMIC_BWD` |

**Scope notes:** Existing relaxed f32 atomics change gradient fold order; preserve padding, accumulation, ID refusal and repeated-ID task quality.

### AFN26-E07 — Embedding float8 gather

**Kind:** `new_variant`. **Files:** [embedding/checks/embedding_fast_apple.mojo](../../embedding/checks/embedding_fast_apple.mojo), [bindings/_mojolearn_embedding.mojo](../../bindings/_mojolearn_embedding.mojo).

**Affected operations:** Embedding.forward/backward; full neural consumers of the public Embedding binding.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `gather8` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD`<br>`MOJOLEARN_AFN26_EMB_GATHER8` |

**Scope notes:** Width divisible by eight is vector-load legality; otherwise the existing float4 or scalar route handles the entire row.

### AFN26-E08 — Embedding launch width

**Kind:** `new_variant`. **Files:** [embedding/checks/embedding_fast_apple.mojo](../../embedding/checks/embedding_fast_apple.mojo), [bindings/_mojolearn_embedding.mojo](../../bindings/_mojolearn_embedding.mojo).

**Affected operations:** Embedding.forward/backward; full neural consumers of the public Embedding binding.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `threads64` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD`<br>`MOJOLEARN_AFN26_EMB_THREADS64` |
| `threads128` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD`<br>`MOJOLEARN_AFN26_EMB_THREADS128` |

**Scope notes:** Same selected block width derives gather/scatter/seed/pad grids. Select one width; thread variants are mutually exclusive.

### AFN26-E09 — Resident embedding table on FAST Apple

**Kind:** `new_variant`. **Files:** [embedding/checks/embedding_fast_apple.mojo](../../embedding/checks/embedding_fast_apple.mojo), [bindings/_mojolearn_embedding.mojo](../../bindings/_mojolearn_embedding.mojo).

**Affected operations:** Embedding.forward/backward; full neural consumers of the public Embedding binding.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `resident` | No experiment defines | `MOJOLEARN_AFN26_EMB_RESIDENT` |
| `resident_atomic` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD`<br>`MOJOLEARN_AFN26_EMB_RESIDENT` |

**Scope notes:** The public immutable-weight token/address/extent contract owns the cached table. Context residency was already implemented; this changes table upload/scan reuse. Include table reassignment and zero-token callers.

### AFN26-E10 — Pooled embedding scratch on FAST Apple

**Kind:** `new_variant`. **Files:** [embedding/checks/embedding_fast_apple.mojo](../../embedding/checks/embedding_fast_apple.mojo), [bindings/_mojolearn_embedding.mojo](../../bindings/_mojolearn_embedding.mojo).

**Affected operations:** Embedding.forward/backward; full neural consumers of the public Embedding binding.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `scratch` | `MOJOLEARN_AFN26_EMB_RESIDENT` | `MOJOLEARN_AFN26_EMB_RESIDENT`<br>`MOJOLEARN_AFN26_EMB_SCRATCH` |
| `scratch_atomic` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD`<br>`MOJOLEARN_AFN26_EMB_RESIDENT` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD`<br>`MOJOLEARN_AFN26_EMB_RESIDENT`<br>`MOJOLEARN_AFN26_EMB_SCRATCH` |

**Scope notes:** Scratch requires resident flag; pools exact sizes and returns buffers only after consumed-output synchronization. Fresh gradients seed all cells, accumulation uploads all cells, and empty dY uses an unread dummy cell.


## New catalog: Combined A/B configurations

Recipe: [interactions.json](interactions.json). Every entry and variant below is **not tested**.

### AFN26-X01 — Norm, RoPE/cache and FLASH complete forward

**Kind:** `interaction_recipe`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo).

**Affected operations:** full TransformerBlock, LM and Samba forward-only callers.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `forward` | No experiment defines | `MOJOLEARN_AFN_ATTN_NORM_SG`<br>`MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN_ATTN_ROPE_CACHE` |

**Scope notes:** A01+A02+A03. Preserve inference route guards; compare cold and repeated completion separately.

### AFN26-X02 — Grouped attention, fused projections and arena

**Kind:** `interaction_recipe`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo).

**Affected operations:** full TransformerBlock and mixed Samba forward; ByteLM only for reached non-arena pieces.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `fused_grouped_block` | No experiment defines | `MOJOLEARN_AFN_ATTN_GQA_TILE`<br>`MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN_ATTN_FUSE_MLP`<br>`MOJOLEARN_AFN_ATTN_ARENA` |

**Scope notes:** A04+A05+A06+A07. Arena is in the standalone transformer binding; declare constituent routing per consumer. No training-kernel speed claim.

### AFN26-X03 — Attention and projection geometry interactions

**Kind:** `interaction_recipe`. **Files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo).

**Affected operations:** full transformer/LM/Samba forward-only public routes.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `norm128_keys16` | `MOJOLEARN_AFN_ATTN_NORM_SG`<br>`MOJOLEARN_AFN_ATTN_FLASH` | `MOJOLEARN_AFN_ATTN_NORM_SG`<br>`MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN26_ATTN_NORM_TPB128`<br>`MOJOLEARN_AFN26_ATTN_FLASH_BK16` |
| `norm128_queries16` | `MOJOLEARN_AFN_ATTN_NORM_SG`<br>`MOJOLEARN_AFN_ATTN_FLASH` | `MOJOLEARN_AFN_ATTN_NORM_SG`<br>`MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN26_ATTN_NORM_TPB128`<br>`MOJOLEARN_AFN26_ATTN_FLASH_TQ16` |
| `flash_keys16_queries16` | `MOJOLEARN_AFN_ATTN_FLASH` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN26_ATTN_FLASH_BK16`<br>`MOJOLEARN_AFN26_ATTN_FLASH_TQ16` |
| `gqa_keys16_queries16` | `MOJOLEARN_AFN_ATTN_GQA_TILE`<br>`MOJOLEARN_AFN_ATTN_FLASH` | `MOJOLEARN_AFN_ATTN_GQA_TILE`<br>`MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN26_ATTN_FLASH_BK16`<br>`MOJOLEARN_AFN26_ATTN_FLASH_TQ16` |
| `projection_rows32_k16` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN_ATTN_FUSE_MLP` | `MOJOLEARN_AFN_ATTN_FUSE_PRE`<br>`MOJOLEARN_AFN_ATTN_FUSE_MLP`<br>`MOJOLEARN_AFN26_ATTN_PROJ_KB16`<br>`MOJOLEARN_AFN26_ATTN_PROJ_BM32` |

**Scope notes:** A08+A09/A10 and A11+A12. Identical parent defines in A/B isolate geometry. TQ16 includes its necessary thread ownership changes; full GQA mapping and smaller staging remain not tested.

### AFN26-X04 — Whole LM fused training and resident parameter views

**Kind:** `interaction_recipe`. **Files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo), [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo), [training/afn_optim.mojo](../../training/afn_optim.mojo), [training/mlp_fast.mojo](../../training/mlp_fast.mojo).

**Affected operations:** full LanguageModelTrainer training, checkpoint/resume, held-out evaluation, refusal and recovery.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `lm_full_step` | No experiment defines | `MOJOLEARN_AFN26_LM_NOSYNC`<br>`MOJOLEARN_AFN26_LM_HEAD_FUSE`<br>`MOJOLEARN_AFN26_LM_PARAM_VIEWS`<br>`MOJOLEARN_AFN26_LM_BWD_FUSE`<br>`MOJOLEARN_AFN26_LM_BWD_EPILOGUE`<br>`MOJOLEARN_AFN26_LM_BWD_NORM1_RESID` |

**Scope notes:** T01+T02+T03+T05; existing default-on backward no-sync held fixed. May bypass shared optimizer paths; qualify this actual complete configuration.

### AFN26-X05 — Split weight gradients and optimizer interactions

**Kind:** `interaction_recipe`. **Files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo), [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo), [training/afn_optim.mojo](../../training/afn_optim.mojo), [training/mlp_fast.mojo](../../training/mlp_fast.mojo).

**Affected operations:** full affected LM/Samba training and actual reached optimizer consumers.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `split_optimizer` | No experiment defines | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT`<br>`MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_LM_WGRAD_MIN128`<br>`MOJOLEARN_AFN26_LM_WGRAD_CAP8` |
| `split_geometry_with_optimizer` | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT`<br>`MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE` | `MOJOLEARN_AFN26_LM_WGRAD_SPLIT`<br>`MOJOLEARN_AFN26_OPT_FUSE_SCAN`<br>`MOJOLEARN_AFN26_OPT_VEC4`<br>`MOJOLEARN_AFN26_OPT_RESIDENT_STATE`<br>`MOJOLEARN_AFN26_LM_WGRAD_MIN128`<br>`MOJOLEARN_AFN26_LM_WGRAD_CAP8` |

**Scope notes:** T04+T09+T06. Include output zero launch, atomic partial accumulation, all optimizer refusals and updates. Parent versus geometry-only comparisons are distinct.

### AFN26-X06 — Fused resident multistep MLP with smaller row tile

**Kind:** `interaction_recipe`. **Files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo), [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo), [training/afn_optim.mojo](../../training/afn_optim.mojo), [training/mlp_fast.mojo](../../training/mlp_fast.mojo).

**Affected operations:** full SmallMLPTrainer ordered training trajectory, checkpoint and failure rollback.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `resident_multistep_rows32` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT` | `MOJOLEARN_AFN26_MLP_FUSED_STEP`<br>`MOJOLEARN_AFN26_MLP_RESIDENT`<br>`MOJOLEARN_AFN26_MLP_MULTISTEP`<br>`MOJOLEARN_AFN26_MLP_ROWS32` |

**Scope notes:** T07+T08+T12. A consumes the same minibatches through resident single steps; B batches host submission, not optimizer semantics. Include required losses/logits/state transport on both sides.

### AFN26-X07 — Complete Mamba-family and Samba configurations

**Kind:** `interaction_recipe`. **Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/modules/afn_refusal.mojo](../../mamba/impl/modules/afn_refusal.mojo).

**Affected operations:** full Mamba1/2/3 forward/continuation and each reached full Samba stack.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `mamba1` | No experiment defines | `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN`<br>`MOJOLEARN_AFN_MAMBA1_FUSE_IN`<br>`MOJOLEARN_AFN_MAMBA_ARENA`<br>`MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4` |
| `mamba2` | No experiment defines | `MOJOLEARN_AFN_MAMBA2_SSD_MMA`<br>`MOJOLEARN_AFN_MAMBA_ARENA`<br>`MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4` |
| `mamba3` | No experiment defines | `MOJOLEARN_AFN_MAMBA3_SISO_FUSED`<br>`MOJOLEARN_AFN_MAMBA_ARENA`<br>`MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4` |

**Scope notes:** M01/M02/M03/M04+M05+M06. Baseline already uses mandatory device refusal. Compare every returned recurrent state; inference cells alone cannot qualify any affected training route.

### AFN26-X08 — Complete CNN and embedding-consumer configurations

**Kind:** `interaction_recipe`. **Files:** [x_cnn/afn_direct.mojo](../../x_cnn/afn_direct.mojo), [embedding/checks/embedding_fast_apple.mojo](../../embedding/checks/embedding_fast_apple.mojo), [bindings/_mojolearn_embedding.mojo](../../bindings/_mojolearn_embedding.mojo).

**Affected operations:** full CNNClassifier fit/inference and Conv2d backward consumers; full actual consumers of public Embedding forward/backward.

| Variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `cnn_complete` | No experiment defines | `MOJOLEARN_AFN_CNN_DIRECT`<br>`MOJOLEARN_AFN26_CNN_ROWS32`<br>`MOJOLEARN_AFN26_CNN_CHANNELS16`<br>`MOJOLEARN_AFN26_CNN_K16`<br>`MOJOLEARN_AFN26_CNN_COLS_ONCE` |
| `cnn_geometry` | `MOJOLEARN_AFN_CNN_DIRECT` | `MOJOLEARN_AFN_CNN_DIRECT`<br>`MOJOLEARN_AFN26_CNN_ROWS32`<br>`MOJOLEARN_AFN26_CNN_CHANNELS16`<br>`MOJOLEARN_AFN26_CNN_K16`<br>`MOJOLEARN_AFN26_CNN_COLS_ONCE` |
| `embedding_complete` | No experiment defines | `MOJOLEARN_AFN_EMB_ATOMIC_BWD`<br>`MOJOLEARN_AFN26_EMB_GATHER8`<br>`MOJOLEARN_AFN26_EMB_THREADS64`<br>`MOJOLEARN_AFN26_EMB_RESIDENT`<br>`MOJOLEARN_AFN26_EMB_SCRATCH` |
| `embedding_storage_geometry` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD` | `MOJOLEARN_AFN_EMB_ATOMIC_BWD`<br>`MOJOLEARN_AFN26_EMB_GATHER8`<br>`MOJOLEARN_AFN26_EMB_THREADS64`<br>`MOJOLEARN_AFN26_EMB_RESIDENT`<br>`MOJOLEARN_AFN26_EMB_SCRATCH` |

**Scope notes:** Separate E01–E05 CNN and E06–E10 embedding comparisons. Full dataset recipe mapping is pending; do not replace it with a reduced fixture. Atomic order, pooled buffer initialization, immutable-table invalidation and all output contracts remain quality gates.

## Existing Apple FAST neural experiments found

The following experiments predate this branch. The controls are transcribed from the existing neural A/B request documents; source links identify their implementation locations. These are historical recipe descriptions, not a claim that every old control is still a distinct active arm or has been fully qualified. Their original `ALL` bundles are listed for completeness, not recommended as new baselines.

Old request documents sometimes refer to `.txt` queue files. This index links the `.md` request files actually found; it creates no queue request and submits no job. Generic GEMM entries are included because the old neural campaign names them; their shared implementation can also serve other families, which are outside this document's experiment scope.

### Existing — Transformer forward

**A/B definition:** [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md). **Notes:** [docs/apple-fast/notes/neural-attn.md](../../docs/apple-fast/notes/neural-attn.md).

**Implementation files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo).

RMSNorm, RoPE/cache, FLASH, grouped-query reuse, normalized projection staging, fused FFN/residual epilogues, arena and bundle. Reused by AFN26-A01–A07.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_ATTN_NORM_SG` | [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md) |
| `MOJOLEARN_AFN_ATTN_ROPE_CACHE` | [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md) |
| `MOJOLEARN_AFN_ATTN_FLASH` | [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md) |
| `MOJOLEARN_AFN_ATTN_GQA_TILE` | [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md) |
| `MOJOLEARN_AFN_ATTN_FUSE_PRE` | [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md) |
| `MOJOLEARN_AFN_ATTN_FUSE_MLP` | [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md) |
| `MOJOLEARN_AFN_ATTN_ARENA` | [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md) |
| `MOJOLEARN_AFN_ATTN_ALL` | [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md) |

### Existing — Byte-LM step

**A/B definition:** [docs/apple-fast/ab-neural/lm.md](../../docs/apple-fast/ab-neural/lm.md). **Notes:** [docs/apple-fast/notes/neural-lm.md](../../docs/apple-fast/notes/neural-lm.md).

**Implementation files:** [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

One-completion step, backward waits/fusion, parameter views, head fusion and bundle. Reused by AFN26-T01–T03/T05. Backward no-sync is already default-on in this snapshot; OFF is the control arm for the historical F08 comparison.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_LM_NOSYNC` | [docs/apple-fast/ab-neural/lm.md](../../docs/apple-fast/ab-neural/lm.md) |
| `MOJOLEARN_AFN_LM_BWD_NOSYNC` | [docs/apple-fast/ab-neural/lm.md](../../docs/apple-fast/ab-neural/lm.md) |
| `MOJOLEARN_AFN_LM_BWD_FUSE` | [docs/apple-fast/ab-neural/lm.md](../../docs/apple-fast/ab-neural/lm.md) |
| `MOJOLEARN_AFN_LM_PARAM_VIEWS` | [docs/apple-fast/ab-neural/lm.md](../../docs/apple-fast/ab-neural/lm.md) |
| `MOJOLEARN_AFN_LM_HEAD_FUSE` | [docs/apple-fast/ab-neural/lm.md](../../docs/apple-fast/ab-neural/lm.md) |
| `MOJOLEARN_AFN_LM_ALL` | [docs/apple-fast/ab-neural/lm.md](../../docs/apple-fast/ab-neural/lm.md) |

### Existing — Optimizer and loss

**A/B definition:** [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md). **Notes:** [docs/apple-fast/notes/neural-optim.md](../../docs/apple-fast/notes/neural-optim.md).

**Implementation files:** [training/afn_optim.mojo](../../training/afn_optim.mojo), [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo), [training/estimator.mojo](../../training/estimator.mojo).

Refusal scans, clipping, multitensor SGD, vector Adam, resident scratch, fused CE and bundle. Reused by AFN26-T06.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_OPT_FUSE_SCAN` | [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md) |
| `MOJOLEARN_AFN_OPT_CLIP_FUSE` | [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md) |
| `MOJOLEARN_AFN_OPT_MULTITENSOR` | [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md) |
| `MOJOLEARN_AFN_OPT_VEC4` | [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md) |
| `MOJOLEARN_AFN_OPT_RESIDENT_STATE` | [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md) |
| `MOJOLEARN_AFN_LOSS_FUSED` | [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md) |
| `MOJOLEARN_AFN_OPTIM_ALL` | [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md) |

### Existing — MLP, embedding and CNN

**A/B definition:** [docs/apple-fast/ab-neural/mlp.md](../../docs/apple-fast/ab-neural/mlp.md). **Notes:** [docs/apple-fast/notes/neural-mlp.md](../../docs/apple-fast/notes/neural-mlp.md).

**Implementation files:** [training/mlp_fast.mojo](../../training/mlp_fast.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [embedding/checks/embedding_fast_apple.mojo](../../embedding/checks/embedding_fast_apple.mojo), [bindings/_mojolearn_embedding.mojo](../../bindings/_mojolearn_embedding.mojo), [x_cnn/afn_direct.mojo](../../x_cnn/afn_direct.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo).

Fused, resident and multistep MLP; atomic embedding; implicit convolution. Reused by AFN26-T07/T08/E01/E06. The board single-step MLP route is not multistep coverage.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_MLP_FUSED_STEP` | [docs/apple-fast/ab-neural/mlp.md](../../docs/apple-fast/ab-neural/mlp.md) |
| `MOJOLEARN_AFN_MLP_RESIDENT` | [docs/apple-fast/ab-neural/mlp.md](../../docs/apple-fast/ab-neural/mlp.md) |
| `MOJOLEARN_AFN_MLP_MULTISTEP` | [docs/apple-fast/ab-neural/mlp.md](../../docs/apple-fast/ab-neural/mlp.md) |
| `MOJOLEARN_AFN_MLP_ALL` | [docs/apple-fast/ab-neural/mlp.md](../../docs/apple-fast/ab-neural/mlp.md) |
| `MOJOLEARN_AFN_EMB_ATOMIC_BWD` | [docs/apple-fast/ab-neural/mlp.md](../../docs/apple-fast/ab-neural/mlp.md) |
| `MOJOLEARN_AFN_CNN_DIRECT` | [docs/apple-fast/ab-neural/mlp.md](../../docs/apple-fast/ab-neural/mlp.md) |

### Existing — Mamba forward

**A/B definition:** [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md). **Notes:** [docs/apple-fast/notes/neural-mamba.md](../../docs/apple-fast/notes/neural-mamba.md).

**Implementation files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo), [mamba/impl/ops/afn_selective_scan.mojo](../../mamba/impl/ops/afn_selective_scan.mojo), [mamba/impl/modeling/afn_mamba1_fused.mojo](../../mamba/impl/modeling/afn_mamba1_fused.mojo), [mamba/impl/modules/afn_ssd_mma.mojo](../../mamba/impl/modules/afn_ssd_mma.mojo), [mamba/impl/ops/afn_mamba3_fused.mojo](../../mamba/impl/ops/afn_mamba3_fused.mojo), [mamba/impl/modules/afn_arena.mojo](../../mamba/impl/modules/afn_arena.mojo), [mamba/impl/modules/afn_refusal.mojo](../../mamba/impl/modules/afn_refusal.mojo).

Chunk scan, input fusion, SSD MMA, SISO fusion, arena and device-refusal bundle. AFN26-M01–M05 reuse existing mechanisms. Device refusal is now mandatory, so its old enable define is not a new host-versus-device arm; AFN26-M06 instead vectorizes the device scan.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN` | [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md) |
| `MOJOLEARN_AFN_MAMBA1_FUSE_IN` | [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md) |
| `MOJOLEARN_AFN_MAMBA2_SSD_MMA` | [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md) |
| `MOJOLEARN_AFN_MAMBA3_SISO_FUSED` | [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md) |
| `MOJOLEARN_AFN_MAMBA_ARENA` | [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md) |
| `MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL` | [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md) |
| `MOJOLEARN_AFN_MAMBA_ALL` | [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md) |

### Existing — Samba and Mamba backward

**A/B definition:** [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md). **Notes:** [docs/apple-fast/notes/neural-samba.md](../../docs/apple-fast/notes/neural-samba.md).

**Implementation files:** [training/samba_afn.mojo](../../training/samba_afn.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo), [mamba/impl/modules/mamba3_prefill_backward.mojo](../../mamba/impl/modules/mamba3_prefill_backward.mojo).

Fused stack-tail calls, scratch arena, device admission, embedding atomics, Mamba3 backward chunk scan and backward arena. Separate sibling-binding runs do not prove the full combined configuration.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_SAMBA_FUSE` | [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md) |
| `MOJOLEARN_AFN_SAMBA_ARENA` | [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md) |
| `MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT` | [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md) |
| `MOJOLEARN_AFN_SAMBA_EMB_ATOMIC` | [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md) |
| `MOJOLEARN_AFN_MAMBA3_BWD_CHUNK` | [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md) |
| `MOJOLEARN_AFN_MAMBA3_BWD_ARENA` | [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md) |
| `MOJOLEARN_AFN_SAMBA_ALL` | [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md) |

### Existing — Shared neural GEMM wave 1

**A/B definition:** [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md). **Notes:** [docs/apple-fast/notes/neural-gemm.md](../../docs/apple-fast/notes/neural-gemm.md).

**Implementation files:** [gemm/afn_apple_fast.mojo](../../gemm/afn_apple_fast.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo).

Simdgroup f32 products, split-K, tile shape, bf16/int8 matrix routes, epilogues and bundle. The bf16/int8 experiments are separate historical precision lanes, not new f32 quality-preserving proposals.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_GEMM_SIMDGROUP` | [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md) |
| `MOJOLEARN_AFN_GEMM_SPLITK` | [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md) |
| `MOJOLEARN_AFN_GEMM_TILESHAPE` | [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md) |
| `MOJOLEARN_AFN_GEMM_BF16_MMA` | [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md) |
| `MOJOLEARN_AFN_GEMM_INT8_MMA` | [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md) |
| `MOJOLEARN_AFN_GEMM_EPILOGUE` | [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md) |
| `MOJOLEARN_AFN_GEMM_ALL` | [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md) |

### Existing — Shared neural GEMM wave 2

**A/B definition:** [docs/apple-fast/ab-neural/w2-gemm2.md](../../docs/apple-fast/ab-neural/w2-gemm2.md). **Notes:** [docs/apple-fast/notes/neural-w2-gemm2.md](../../docs/apple-fast/notes/neural-w2-gemm2.md).

**Implementation files:** [gemm/afn_apple_fast2.mojo](../../gemm/afn_apple_fast2.mojo), [gemm/afn_apple_fast.mojo](../../gemm/afn_apple_fast.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo).

Larger tiles, double buffering, direct B loads, grid swizzle and bundle. Historical A retains wave-1 SIMDGROUP/BF16_MMA parents; B adds a wave-2 control.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_GEMM2_BIGTILE` | [docs/apple-fast/ab-neural/w2-gemm2.md](../../docs/apple-fast/ab-neural/w2-gemm2.md) |
| `MOJOLEARN_AFN_GEMM2_DBUF` | [docs/apple-fast/ab-neural/w2-gemm2.md](../../docs/apple-fast/ab-neural/w2-gemm2.md) |
| `MOJOLEARN_AFN_GEMM2_DIRECT_B` | [docs/apple-fast/ab-neural/w2-gemm2.md](../../docs/apple-fast/ab-neural/w2-gemm2.md) |
| `MOJOLEARN_AFN_GEMM2_SWIZZLE` | [docs/apple-fast/ab-neural/w2-gemm2.md](../../docs/apple-fast/ab-neural/w2-gemm2.md) |
| `MOJOLEARN_AFN_GEMM2_ALL` | [docs/apple-fast/ab-neural/w2-gemm2.md](../../docs/apple-fast/ab-neural/w2-gemm2.md) |

### Existing — LM gradient wave 2

**A/B definition:** [docs/apple-fast/ab-neural/w2-lmgrad.md](../../docs/apple-fast/ab-neural/w2-lmgrad.md). **Notes:** [docs/apple-fast/notes/neural-w2-lmgrad.md](../../docs/apple-fast/notes/neural-w2-lmgrad.md).

**Implementation files:** [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

Weight-gradient split-K, backward GEMM residual epilogue, norm1/residual fusion and bundle. Reused by AFN26-T04/T05 and extended by T09.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_LM_WGRAD_SPLIT` | [docs/apple-fast/ab-neural/w2-lmgrad.md](../../docs/apple-fast/ab-neural/w2-lmgrad.md) |
| `MOJOLEARN_AFN_LM_BWD_EPILOGUE` | [docs/apple-fast/ab-neural/w2-lmgrad.md](../../docs/apple-fast/ab-neural/w2-lmgrad.md) |
| `MOJOLEARN_AFN_LM_BWD_NORM1_RESID` | [docs/apple-fast/ab-neural/w2-lmgrad.md](../../docs/apple-fast/ab-neural/w2-lmgrad.md) |
| `MOJOLEARN_AFN_LMGRAD_ALL` | [docs/apple-fast/ab-neural/w2-lmgrad.md](../../docs/apple-fast/ab-neural/w2-lmgrad.md) |

### Existing — Mamba projection and Samba head epilogues

**A/B definition:** [docs/apple-fast/ab-neural/w2-epi.md](../../docs/apple-fast/ab-neural/w2-epi.md). **Notes:** [docs/apple-fast/notes/neural-w2-epi.md](../../docs/apple-fast/notes/neural-w2-epi.md).

**Implementation files:** [mamba/impl/modules/afn_proj_gemm.mojo](../../mamba/impl/modules/afn_proj_gemm.mojo), [training/samba_afn.mojo](../../training/samba_afn.mojo).

Projection MMA/residual epilogue, projection split-K, tied-head GEMM and bundle. Split-K compares against the projection-epilogue parent; Samba head compares with SAMBA_FUSE on both sides.

| Existing compiler control | Definition and A/B details |
| --- | --- |
| `MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE` | [docs/apple-fast/ab-neural/w2-epi.md](../../docs/apple-fast/ab-neural/w2-epi.md) |
| `MOJOLEARN_AFN_MAMBA_PROJ_SPLITK` | [docs/apple-fast/ab-neural/w2-epi.md](../../docs/apple-fast/ab-neural/w2-epi.md) |
| `MOJOLEARN_AFN_SAMBA_HEAD_GEMM` | [docs/apple-fast/ab-neural/w2-epi.md](../../docs/apple-fast/ab-neural/w2-epi.md) |
| `MOJOLEARN_AFN_EPI_ALL` | [docs/apple-fast/ab-neural/w2-epi.md](../../docs/apple-fast/ab-neural/w2-epi.md) |

### Existing — numeric-tier baselines

[docs/apple-fast/ab-neural/tier.md](../../docs/apple-fast/ab-neural/tier.md) describes A=IDENTICAL (`ours`) versus B=FAST (`ours-fast`) with no AFN experiment defines, across the neural GPU board lanes. [tools/afn_ab.sh](../../tools/afn_ab.sh) owns the positional driver and [tools/bench_board_neural.py](../../tools/bench_board_neural.py) owns the lanes. This compares numeric tiers, unlike new FAST-versus-FAST geometry experiments.

## Existing central experiment manifests relevant to neural work

The next entries retain their legacy IDs. Each manifest links its own build/caller recipes. Values in `status` are copied historical declarations; none was checked or requalified for this document. Empty compiler arrays can still describe a real A/B when the caller switches API behavior. A central manifest alone does not establish full-dataset coverage.

### Existing F03 — Actual LM bounded resident session versus stateless calls

**Mode:** `fast`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/F03/manifest.json](../../experiments/performance_ideas/F03/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/F03/caller.py](../../experiments/performance_ideas/F03/caller.py), [experiments/performance_ideas/apple_fast/lm_task.py](../../experiments/performance_ideas/apple_fast/lm_task.py), [bindings/_mojolearn_byte_lm.mojo](../../bindings/_mojolearn_byte_lm.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | No experiment defines | No experiment defines |

API A/B: resident=False versus resident=True on the same LM task; no define difference.

**Declared operation boundary:** Train step through consumed scalar, first step separate from repeated calls; A resident=False B resident=True; no new default

### Existing F07 — FLASH and true GQA caller qualification with reach counters

**Mode:** `fast`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/F07/manifest.json](../../experiments/performance_ideas/F07/manifest.json).

**Implementation/caller files:** [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [experiments/performance_ideas/F07/caller.py](../../experiments/performance_ideas/F07/caller.py).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | `MOJOLEARN_AFN_ATTN_AUDIT` | `MOJOLEARN_AFN_ATTN_FLASH`<br>`MOJOLEARN_AFN_ATTN_AUDIT` |
| `gqa` | `MOJOLEARN_AFN_ATTN_AUDIT` | `MOJOLEARN_AFN_ATTN_GQA_TILE`<br>`MOJOLEARN_AFN_ATTN_AUDIT` |

**Declared operation boundary:** Public transform through first output read; cold/repeated calls separate; no matrix-only claim

### Existing F08 — Independent LM backward fusion and view A/B training task

**Mode:** `fast`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/F08/manifest.json](../../experiments/performance_ideas/F08/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/F08/caller.py](../../experiments/performance_ideas/F08/caller.py), [experiments/performance_ideas/apple_fast/lm_task.py](../../experiments/performance_ideas/apple_fast/lm_task.py), [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo), [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | `MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF` | `MOJOLEARN_AFN_LM_BWD_NOSYNC` |
| `fused` | `MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF` | `MOJOLEARN_AFN_LM_BWD_FUSE`<br>`MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF` |
| `views` | `MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF` | `MOJOLEARN_AFN_LM_PARAM_VIEWS`<br>`MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF` |

**Declared operation boundary:** Whole train step through consumed loss; forward-only timing cannot certify backward

### Existing F09 — Memory-bounded LM head in a fixed actual SGD task

**Mode:** `fast`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/F09/manifest.json](../../experiments/performance_ideas/F09/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/F09/caller.py](../../experiments/performance_ideas/F09/caller.py), [training/chunked_lm_head_v2.mojo](../../training/chunked_lm_head_v2.mojo), [training/afn_optim.mojo](../../training/afn_optim.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | No experiment defines | No experiment defines |

API A/B: materialized logits + CE + backward versus chunked_lm_head_loss, preserving the same SGD task and explicitly requested logits. Caller fixtures are not full-dataset qualification.

**Declared operation boundary:** Eight whole head train steps through read of loss/gradients; requested logits measured separately; A full logits B chunked public API

### Existing F10 — Independent SSD MMA and Mamba fusion caller arms

**Mode:** `fast`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/F10/manifest.json](../../experiments/performance_ideas/F10/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/F10/caller.py](../../experiments/performance_ideas/F10/caller.py), [mamba/impl/modules/afn_ssd_mma.mojo](../../mamba/impl/modules/afn_ssd_mma.mojo), [mamba/impl/modeling/afn_mamba1_fused.mojo](../../mamba/impl/modeling/afn_mamba1_fused.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | No experiment defines | `MOJOLEARN_AFN_MAMBA2_SSD_MMA` |
| `mamba1-input` | No experiment defines | `MOJOLEARN_AFN_MAMBA1_FUSE_IN` |
| `chunk-scan` | No experiment defines | `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN` |
| `mamba3-elementwise` | No experiment defines | `MOJOLEARN_AFN_MAMBA3_SISO_FUSED` |
| `arena` | No experiment defines | `MOJOLEARN_AFN_MAMBA_ARENA` |

**Declared operation boundary:** Full caller through first read; missing corpus coverage fails explicitly

### Existing F20 — Task-level optimizer fusion and FAST blocked LayerNorm qualification

**Mode:** `fast`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/F20/manifest.json](../../experiments/performance_ideas/F20/manifest.json).

**Implementation/caller files:** [sequence/layernorm.mojo](../../sequence/layernorm.mojo), [training/afn_optim.mojo](../../training/afn_optim.mojo), [experiments/performance_ideas/F20/caller.py](../../experiments/performance_ideas/F20/caller.py).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | No experiment defines | `MOJOLEARN_AFN_OPT_MULTITENSOR` |
| `status` | No experiment defines | `MOJOLEARN_AFN_OPT_FUSE_SCAN` |
| `normalization` | No experiment defines | `MOJOLEARN_LN_FAST_BLOCK_FOLD` |

**Declared operation boundary:** Whole two-head train step through losses+updated parameters, separate full normalization forward/backward

### Existing I06 — qualify attention KV grid reuse across GQA tails

**Mode:** `identical`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/I06/manifest.json](../../experiments/performance_ideas/I06/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/I06/check.mojo](../../experiments/performance_ideas/I06/check.mojo), [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | `MOJOLEARN_ATTN_ARM_TRIAL=1` | `MOJOLEARN_ATTN_ARM_TRIAL=1`<br>`MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE=1` |

**Declared operation boundary:** Frozen source; compile once on cheap boxes; device runs only; full public-operation timing includes all scratch/preparation/readback. Apple IDENTICAL bit check only. Native builds explicitly bind device column and accelerator; independent host replay remains a separate identity gate, not a DeviceContext execution vendor.

### Existing I07 — exercise retained and recomputed attention backward lifetimes

**Mode:** `identical`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/I07/manifest.json](../../experiments/performance_ideas/I07/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/I07/check.mojo](../../experiments/performance_ideas/I07/check.mojo), [experiments/performance_ideas/I07/state_cost.mojo](../../experiments/performance_ideas/I07/state_cost.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | `MOJOLEARN_ATTN_ARM_TRIAL=1`<br>`MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD=1` | `MOJOLEARN_ATTN_ARM_TRIAL=1` |

**Declared operation boundary:** Frozen source; compile once on cheap boxes; device runs only; full public-operation timing includes all scratch/preparation/readback. Apple IDENTICAL bit check only. Native builds explicitly bind device column and accelerator; independent host replay remains a separate identity gate, not a DeviceContext execution vendor.

### Existing I08 — attribute SSD tile reuse on multiple state cases and chunk tails

**Mode:** `identical`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/I08/manifest.json](../../experiments/performance_ideas/I08/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/I08/check.mojo](../../experiments/performance_ideas/I08/check.mojo), [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo), [experiments/performance_ideas/I08/backward_check.mojo](../../experiments/performance_ideas/I08/backward_check.mojo), [mamba/impl/modules/mamba2_prefill_backward.mojo](../../mamba/impl/modules/mamba2_prefill_backward.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | No experiment defines | `MOJOLEARN_IDN_M2_RETAIN_GL=1` |

**Declared operation boundary:** Frozen source; compile once on cheap boxes; device runs only; full public-operation timing includes all scratch/preparation/readback. Apple IDENTICAL bit check only. Native builds explicitly bind device column and accelerator; independent host replay remains a separate identity gate, not a DeviceContext execution vendor.

### Existing I09 — gate token-parallel recurrence on prefix and decode boundaries

**Mode:** `identical`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/I09/manifest.json](../../experiments/performance_ideas/I09/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/I09/check.mojo](../../experiments/performance_ideas/I09/check.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | `MOJOLEARN_IDN_MAMBA_CONV_CELL_OFF=1` | No experiment defines |

**Declared operation boundary:** Frozen source; compile once on cheap boxes; device runs only; full public-operation timing includes all scratch/preparation/readback. Apple IDENTICAL bit check only. Native builds explicitly bind device column and accelerator; independent host replay remains a separate identity gate, not a DeviceContext execution vendor.

### Existing I10 — reduce ordered training status from per-tile contributions

**Mode:** `identical`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/I10/manifest.json](../../experiments/performance_ideas/I10/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/I10/check.mojo](../../experiments/performance_ideas/I10/check.mojo), [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | `MOJOLEARN_STEP_GLUE_TRIAL=1` | `MOJOLEARN_TRAIN_LIVE_STATUS=1`<br>`MOJOLEARN_STEP_GLUE_TRIAL=1` |

**Declared operation boundary:** Frozen source; compile once on cheap boxes; device runs only; full public-operation timing includes all scratch/preparation/readback. Apple IDENTICAL bit check only. Native builds explicitly bind device column and accelerator; independent host replay remains a separate identity gate, not a DeviceContext execution vendor.

### Existing I11 — qualify canonical radix embedding updates under skew and vocabulary reuse

**Mode:** `identical`. **Historical declared status:** `source_ready`. **Manifest:** [experiments/performance_ideas/I11/manifest.json](../../experiments/performance_ideas/I11/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/I11/check.mojo](../../experiments/performance_ideas/I11/check.mojo).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | `MOJOLEARN_IDN_EMB_RADIX_SORT_OFF=1` | No experiment defines |

**Declared operation boundary:** Frozen source; compile once on cheap boxes; device runs only; full public-operation timing includes all scratch/preparation/readback. Apple IDENTICAL bit check only. Native builds explicitly bind device column and accelerator; independent host replay remains a separate identity gate, not a DeviceContext execution vendor.

### Existing N06 — Keep compact feature gradients in registers without query barriers

**Mode:** `identical`. **Historical declared status:** `build_passed`. **Manifest:** [experiments/performance_ideas/N06/manifest.json](../../experiments/performance_ideas/N06/manifest.json).

**Implementation/caller files:** [experiments/performance_ideas/N06/compact_grad.mojo](../../experiments/performance_ideas/N06/compact_grad.mojo), [experiments/performance_ideas/N06/check.mojo](../../experiments/performance_ideas/N06/check.mojo), [transformer/impl/llama/attention_v2.mojo](../../transformer/impl/llama/attention_v2.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py).

| Manifest variant | A — baseline defines | B — candidate defines |
| --- | --- | --- |
| `default` | No experiment defines | No experiment defines |

**Declared operation boundary:** Frozen source, one arm run; device completion includes launch and wait, excludes fixture upload/reference/readback. Full caller timing and compiled resource counters remain required; Apple identity only.

### Existing shared GEMM experiments also found

These are shared linear-algebra mechanisms relevant to neural products, not new Apple FAST neural cards. Their existing IDENTICAL vendor/identity rules remain separate. Alternative-arm lists and execution details stay in their manifests and native campaign files; no flattened list here should be interpreted as one combined build.

| Legacy ID | Idea | Manifest | Implementation/campaign files |
| --- | --- | --- | --- |
| `I01` | Attribute existing GEMM schedules | [experiments/performance_ideas/I01/manifest.json](../../experiments/performance_ideas/I01/manifest.json) | [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)<br>[gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py) |
| `I02` | Bound session GEMM scratch retention | [experiments/performance_ideas/I02/manifest.json](../../experiments/performance_ideas/I02/manifest.json) | [gemm/experiments/bounded_workspace.mojo](../../gemm/experiments/bounded_workspace.mojo)<br>[gemm/experiments/bounded_workspace_check.mojo](../../gemm/experiments/bounded_workspace_check.mojo)<br>[gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| `I03` | Batch independent products on separate grid jobs | [experiments/performance_ideas/I03/manifest.json](../../experiments/performance_ideas/I03/manifest.json) | [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo)<br>[gemm/experiments/grouped_jobs_check.mojo](../../gemm/experiments/grouped_jobs_check.mojo)<br>[gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| `I04` | Qualify a coherently shared opt-in 64-element IDENTICAL leaf version | [experiments/performance_ideas/I04/manifest.json](../../experiments/performance_ideas/I04/manifest.json) | [gemm/contract.mojo](../../gemm/contract.mojo)<br>[gemm/experiments/fold_profile_probe.mojo](../../gemm/experiments/fold_profile_probe.mojo)<br>[gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/experiments/profile_identity_check.mojo](../../gemm/experiments/profile_identity_check.mojo)<br>[gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| `I05` | Fuse bias after the explicit rounded product seam | [experiments/performance_ideas/I05/manifest.json](../../experiments/performance_ideas/I05/manifest.json) | [gemm/experiments/rounded_epilogue.mojo](../../gemm/experiments/rounded_epilogue.mojo)<br>[gemm/experiments/rounded_epilogue_check.mojo](../../gemm/experiments/rounded_epilogue_check.mojo)<br>[gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| `A01` | Isolate AMD smaller MFMA and band routing | [experiments/performance_ideas/A01/manifest.json](../../experiments/performance_ideas/A01/manifest.json) | [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py)<br>[gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `A02` | Compare production one/two-page staging and bounded one/two/four-plane resource controls | [experiments/performance_ideas/A02/manifest.json](../../experiments/performance_ideas/A02/manifest.json) | [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py)<br>[gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)<br>[experiments/performance_ideas/A02/check.mojo](../../experiments/performance_ideas/A02/check.mojo)<br>[gemm/experiments/bounded_staging.mojo](../../gemm/experiments/bounded_staging.mojo)<br>[gemm/experiments/bounded_staging_check.mojo](../../gemm/experiments/bounded_staging_check.mojo) |
| `A03` | Vary vector-aligned LDS operand strides | [experiments/performance_ideas/A03/manifest.json](../../experiments/performance_ideas/A03/manifest.json) | [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py)<br>[gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `A04` | Pair independent logical groups with supported XOR membership | [experiments/performance_ideas/A04/manifest.json](../../experiments/performance_ideas/A04/manifest.json) | [gemm/experiments/subwave_membership.mojo](../../gemm/experiments/subwave_membership.mojo)<br>[gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| `A05` | Halve the packed body row accumulator live range | [experiments/performance_ideas/A05/manifest.json](../../experiments/performance_ideas/A05/manifest.json) | [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py)<br>[gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `N01` | Isolate packed64 body tiles and their interaction | [experiments/performance_ideas/N01/manifest.json](../../experiments/performance_ideas/N01/manifest.json) | [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py)<br>[gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `N02` | Prove and dispatch a two-level logical fold stack | [experiments/performance_ideas/N02/manifest.json](../../experiments/performance_ideas/N02/manifest.json) | [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py)<br>[gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `N03` | Pipeline supported asynchronous operand loads with explicit completion | [experiments/performance_ideas/N03/manifest.json](../../experiments/performance_ideas/N03/manifest.json) | [gemm/experiments/async_operand_pipeline.mojo](../../gemm/experiments/async_operand_pipeline.mojo)<br>[gemm/experiments/async_operand_pipeline_check.mojo](../../gemm/experiments/async_operand_pipeline_check.mojo)<br>[gemm/experiments/async_api_probe.mojo](../../gemm/experiments/async_api_probe.mojo)<br>[gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| `N04` | Separate contiguous and gather staging with counted caller passes | [experiments/performance_ideas/N04/manifest.json](../../experiments/performance_ideas/N04/manifest.json) | [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py)<br>[gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `N05` | Batch compatible launches with fixed-address changing inputs | [experiments/performance_ideas/N05/manifest.json](../../experiments/performance_ideas/N05/manifest.json) | [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo)<br>[gemm/experiments/changing_batch_check.mojo](../../gemm/experiments/changing_batch_check.mojo)<br>[gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |

## Existing runtime-toggle experiments

All names below are defined in [tools/neural_experiments.py](../../tools/neural_experiments.py) under `EXPERIMENTS`, with groupings in `SETS`; the numerical driver is [tools/neural_stage_timing.py](../../tools/neural_stage_timing.py). These are environment-variable A/Bs, not the new compiler defines. A is the runner's inherited/default configuration and B applies the listed environment delta. Establish the effective baseline environment before future use. The legacy runner rejects digest movement; that policy is not the new Apple FAST acceptance rule.

| Existing experiment name | B environment change / purpose | Definition or runtime implementation |
| --- | --- | --- |
| `baseline` | No environment delta; control arm. | [tools/neural_experiments.py](../../tools/neural_experiments.py) |
| `no_retain_weights` | MOJOLEARN_TRANSFORMER_RETAIN_WEIGHTS=0; MOJOLEARN_MAMBA3_RETAIN_WEIGHTS=0 | [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo)<br>[bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo) |
| `legacy_fresh_entry` | MOJOLEARN_TRANSFORMER_SESSION_FRESH=0 | [tools/neural_experiments.py](../../tools/neural_experiments.py)<br>[bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo) |
| `legacy_everything` | MOJOLEARN_TRANSFORMER_LEGACY_SETUP=1; MOJOLEARN_MAMBA3_LEGACY_SETUP=1 | [tools/neural_experiments.py](../../tools/neural_experiments.py)<br>[bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo)<br>[bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo) |
| `no_stage_reset` | MOJOLEARN_TRANSFORMER_STAGE_RESET=0 | [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo) |
| `speculative_attn` | MOJOLEARN_ATTN_SPECULATIVE=1 | [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo) |
| `swiglu_fused` | MOJOLEARN_SWIGLU_FUSED=1 | [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo) |
| `no_layer_sync` | MOJOLEARN_BYTE_LM_LAYER_SYNC=0 | [training/byte_lm.mojo](../../training/byte_lm.mojo) |
| `mamba3_legacy` | MOJOLEARN_MAMBA3_LEGACY_SETUP=1 | [tools/neural_experiments.py](../../tools/neural_experiments.py)<br>[bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo) |
| `mamba3_no_retain_stages` | MOJOLEARN_MAMBA3_RETAIN_STAGES=0 | [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo) |
| `norm_dw_own_ws` | MOJOLEARN_TRANSFORMER_NORM_DW_OWN_WS=1 | [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo) |
| `s16_naive` | MOJOLEARN_MAMBA3_S16_QK_ARM=naive | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo) |
| `s16_shared` | MOJOLEARN_MAMBA3_S16_QK_ARM=shared | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo) |
| `s16_regs` | MOJOLEARN_MAMBA3_S16_QK_ARM=regs | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo) |
| `s16_regs2` | MOJOLEARN_MAMBA3_S16_QK_ARM=regs2 | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo) |
| `s16_smem48` | MOJOLEARN_MAMBA3_S16_QK_ARM=smem48 | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo) |
| `s16_regs2h` | MOJOLEARN_MAMBA3_S16_QK_ARM=regs2h | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo) |
| `s16_regsh` | MOJOLEARN_MAMBA3_S16_QK_ARM=regsh | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo) |
| `opt_host` | MOJOLEARN_OPTIMIZER_RESIDENT=0; historical host-state control, not a new runtime-policy endorsement. | [python/mojolearn/_training_impl.py](../../python/mojolearn/_training_impl.py) |
| `all_on` | MOJOLEARN_ATTN_SPECULATIVE=1; MOJOLEARN_SWIGLU_FUSED=1; MOJOLEARN_BYTE_LM_LAYER_SYNC=0; MOJOLEARN_TRANSFORMER_STAGE_RESET=0 | [tools/neural_experiments.py](../../tools/neural_experiments.py) |
| `--gemm-arms` | MOJOLEARN_GEMM_ARM per requested arm (documented examples: shipped, tuned128, half, quarter, kpack); requires the corresponding supported trial configuration. | [tools/neural_experiments.py](../../tools/neural_experiments.py)<br>[gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |

Existing runtime sets: `priority`, `s16`, `s16_apple`, `optimizer`, `default`, `nvidia`, and `amd`. They select combinations of the rows above; they are not additional kernels. GPU/vendor legality and current route effectiveness are not established by this inventory.

## Other existing neural A/B harnesses and comparisons found

| Existing A/B | Files | Scope and distinction |
| --- | --- | --- |
| Between-commit Apple neural A/B | [tools/apple_speed_neural/ab.sh](../../tools/apple_speed_neural/ab.sh)<br>[tools/apple_speed_neural/pyprof.py](../../tools/apple_speed_neural/pyprof.py)<br>[tools/lm_step_memory_probe.py](../../tools/lm_step_memory_probe.py) | Uses AB_VARIANTS label=commit worktrees, AB_BUILDS, AB_LANES and alternating repetitions. Script explicitly selects IDENTICAL; optional byte-LM shape and Samba-profile comparisons. Not run here. |
| Byte-LM optimizer-pool rollback/replay | [tools/byte_lm_pool_ab_matrix.sh](../../tools/byte_lm_pool_ab_matrix.sh)<br>[tools/byte_lm_pool_ab_compare.py](../../tools/byte_lm_pool_ab_compare.py)<br>[tools/byte_lm_optimizer_pool_check.py](../../tools/byte_lm_optimizer_pool_check.py)<br>[training/byte_lm_optimizer_pool.mojo](../../training/byte_lm_optimizer_pool.mojo) | Baseline/candidate fault binaries, logical shard counts 2/3/5, exact state/recovery comparison. IDENTICAL correctness A/B, not Apple FAST timing qualification. |
| Byte-LM one/two-device shard comparison | [tools/lm_shards_ab_matrix.sh](../../tools/lm_shards_ab_matrix.sh)<br>[tools/lm_shards_ab_compare.py](../../tools/lm_shards_ab_compare.py)<br>[tools/lm_shards_probe.py](../../tools/lm_shards_probe.py)<br>[training/byte_lm_parallel.mojo](../../training/byte_lm_parallel.mojo) | Baseline/candidate binaries on one and two devices with reversed second-round order and complete-state comparison. IDENTICAL; not a new Apple FAST experiment. |
| Embedding/CNN component A/B driver | [tools/afn_custom_time.py](../../tools/afn_custom_time.py)<br>[tools/afn_ab.sh](../../tools/afn_ab.sh) | Existing custom lane for Embedding forward/backward and Conv2d. It is not a full CNNClassifier fit or complete embedding-consumer dataset recipe. |
| Frozen Apple caller pairs | [experiments/performance_ideas/apple_fast/build_pair.py](../../experiments/performance_ideas/apple_fast/build_pair.py)<br>[experiments/performance_ideas/apple_fast/pair.py](../../experiments/performance_ideas/apple_fast/pair.py)<br>[experiments/performance_ideas/apple_fast/lm_task.py](../../experiments/performance_ideas/apple_fast/lm_task.py) | Existing F-card build/caller orchestration, provenance and paired outputs; supports the historical manifest recipes above. It is not an executable AFN26 harness. |

## Overlap and remaining integration work

| Existing mechanism | New catalog relationship |
| --- | --- |
| Attention wave-1 and F07 | AFN26-A01–A07 reuse controls; A08–A12 add geometry variants. |
| LM wave-1, gradient wave-2, F08 | AFN26-T01–T05 preserve/select existing mechanisms; T09 and the LM side of T11 add geometry. Existing backward-no-sync default is held fixed in the new recipes. |
| Optimizer wave-1, F20 | AFN26-T06 reuses mechanisms; T10/T11 add launch alternatives. Existing LayerNorm F20 normalization is separate. |
| MLP wave-1 | AFN26-T07/T08 reuse fused/resident/multistep routes; T12 adds tile/capacity variants. |
| Mamba wave-1, F10 | AFN26-M01–M05 reuse mechanisms; M06–M10 add vectorization/geometry. Old DEVICE_REFUSAL is not a distinct new arm. |
| CNN and embedding wave-1 | AFN26-E01/E06 reuse mechanisms; E02–E05 and E07–E10 add tiles, ownership, gather width and residency/scratch variants. |
| F03 resident LM and F09 chunked head | Additional existing neural A/Bs; not relabeled as new AFN26 kernels. |
| AFN26-X01–X08 | Explicit combinations of catalog mechanisms, not evidence that individual wins compose. |

Before any later execution, complete the full-dataset recipe and binding-dependency mapping, distinguish inference-only from training routes, provide actual multistep and CNN/embedding consumer harnesses, and retain quality/provenance evidence. Preserve existing quality rules even though bits may change. No compile/test/measurement result is implied by a source link, manifest status, catalog entry or this document.

This inventory is limited to the neural and shared neural-math material found in this worktree. It does not inventory unrelated classical, tree or nearest-neighbor experiments, other worktrees, or unmerged external branches.
