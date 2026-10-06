# Apple FAST NEURAL: the plan (lane afn-tier, 2026-10-03)

2026-10-06 source-only follow-up:
[44 neural ideas, eight interaction groups, and integration status](../../experiments/apple_fast_neural_20261006/README.md).
The `AFN26-*` cards share A/B metadata through `tools/performance_ideas.py`,
`tools/neural_experiments.py --apple-fast-plan`, and
`tools/afn_ab.sh --experiment-plan`. Those new entry points only emit plans.
They are **not tested**, not compiled or measured, and do not inherit the
historical qualification below. Complete executable full-workload harnesses,
including actual multistep and CNN/embedding consumer recipes, remain pending.

## The tier

FAST is the Apple GPU tier: no bit promise, speed and quality only. Until 2026-10-03 it covered
trees and classical ML; Andrew's order extends it to EVERY neural network algorithm, GPU path only,
Apple only. The neural bindings (linalg, transformer, mamba, training, embedding, x_cnn) already
build FAST (`python/mojolearn/_backend.py` `_CLASSICAL_FAST`): today FAST runs the IDENTICAL kernels
with the pins of checks/numerics.mojo on the free schedule. The byte LM binding
(`_mojolearn_byte_lm`) refused fast (bindings/build_byte_lm.sh); lane afn-lm lifts that. Every
change of this pass compiles ONLY under `comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()`, behind its own `-D MOJOLEARN_AFN_<NAME>` define (default OFF);
IDENTICAL compiles main's code unchanged, so the same bits on NVIDIA, AMD, Apple and the host
column stay the product. FAST never degrades quality: f32 accumulation stays f32, fold order may
change, no lower-precision arithmetic on an fp32 lane, no approximate transcendental, no caps or
subsampling. The quality judge is tools/neural_fast_quality.py (one seed per A/B, Andrew 2026-10-03)
plus tools/afn_ab.py's output judge.

Apple's measured costs set the levers: a launch costs ~20 us host and grows ~0.25 us per live Metal
buffer; a launch + readback + sync round trip ~180 us; a device-to-host readback ~21 ms per 64 MB.
The LM's small stages on the M3 (norm 1.24 ms vs 0.25 on AMD, rope_and_cache 1.19 vs 0.06) are launch
and buffer bound. So: FUSION and FEWER LAUNCHES/WAITS first, arena views for scratch second,
simdgroup_matrix MMA tiles third.

## The lanes and their defines (from the briefs in ~/mojolearn-evidence/briefs-2026-10-03/afn-*.md)

| lane | branch | binding | board lanes | defines |
|---|---|---|---|---|
| gemm | lane/apple-fast-neural-gemm | linalg | gemm, gemm-bf16, gemm-int8 | MOJOLEARN_AFN_GEMM_SIMDGROUP (fp32 tiles through simdgroup_matrix 8x8 f32 MMA), _GEMM_SPLITK (split-K for small-M/N, large-K and under-filled grids), _GEMM_TILESHAPE (shape-class tile selection at dispatch), _GEMM_BF16_MMA (bf16 x bf16 -> f32 simdgroup tiles), _GEMM_INT8_MMA (exact i32 int8 path), _GEMM_EPILOGUE (fused bias/residual/gate epilogues), _GEMM_ALL |
| attn | lane/apple-fast-neural-attn | transformer | transformer-forward (and the LM's blocks through afn-lm) | MOJOLEARN_AFN_ATTN_FLASH (one-launch flash-style attention per head x query tile, online softmax), _ATTN_FUSE_PRE (norm1 + qkv + rope + cache write fused), _ATTN_FUSE_MLP (SwiGLU gate in the up-projection epilogue, residual add fused), _ATTN_ARENA (per-forward scratch as sub-buffer views of one arena), _ATTN_GQA_TILE (one KV tile serves a GQA group), _ATTN_ALL |
| mamba | lane/apple-fast-neural-mamba | mamba | mamba1-forward, mamba2-forward, mamba3-forward | MOJOLEARN_AFN_MAMBA1_CHUNKSCAN (chunked parallel scan over time), _MAMBA1_FUSE_IN (in_proj -> conv1d -> SiLU -> dt/B/C fused), _MAMBA2_SSD_MMA (SSD intra-chunk and state matmuls through simdgroup tiles), _MAMBA3_SISO_FUSED (per-step elementwise work fused into the scan), _MAMBA_ARENA (scratch views, no per-stage synchronize), _MAMBA_ALL |
| samba | lane/apple-fast-neural-samba | training | samba-forward, samba-train-step | MOJOLEARN_AFN_SAMBA_RESIDENT (weights, grads, optimizer state and activations resident), _SAMBA_FWD_FUSE (embedding + first norm, residual adds fused), _MAMBA3_BWD_CHUNK (Mamba-3 backward scan chunked over time), _MAMBA2_BWD_MMA (SSD backward matmuls through simdgroup tiles), _SAMBA_GRAD_FOLD (block-parallel gradient accumulation), _SAMBA_ALL |
| lm | lane/apple-fast-neural-lm | byte_lm | lm-train-step, lm-forward | (required, no define) the byte LM FAST build; MOJOLEARN_AFN_LM_BWD_FUSE (fused backward stages), _LM_NOSYNC (no host waits inside a step but the last), _LM_HEAD_FUSE (logits GEMM + softmax + CE + its backward fused), _LM_LAYER_PIPE (every layer's launches in one command stream), _LM_WGRAD_SPLIT (split-K weight-gradient GEMMs), _LM_ALL |
| optim | lane/apple-fast-neural-optim | training | lm-train-step, samba-train-step, mlp-train-step (the optimizer and loss every train step uses) | MOJOLEARN_AFN_OPT_MULTITENSOR (one launch updates every tensor), _OPT_FUSE_SCAN (non-finite scan fused into the update), _OPT_CLIP_FUSE (grad-norm clip fold fused), _LOSS_FUSED (CE forward + backward in one launch per row tile), _OPT_RESIDENT_STATE (optimizer state resident in an arena), _OPTIM_ALL |
| mlp | lane/apple-fast-neural-mlp | training (+ embedding, x_cnn) | mlp-train-step (Embedding and the CNN are public GPU surface with no board lane) | MOJOLEARN_AFN_MLP_FUSED_STEP (forward + backward + update in as few launches as the shape allows), _MLP_RESIDENT (X and y uploaded once per fit), _MLP_MULTISTEP (k steps in one command buffer), _EMB_ATOMIC_BWD (embedding backward scatter-add with f32 atomics), _CNN_DIRECT (direct convolution tiles, no im2col), _MLP_ALL |
| tier | lane/apple-fast-neural-tier | (tools only) | all | none: the board arm, afn_ab.sh, the quality judge, this plan |

The sibling lanes' notes (docs/apple-fast/notes/neural-<lane>.md) and request files
(docs/apple-fast/ab-neural/<lane>.txt and .md) carry what each define finally does; this table is
the planned list. A define a lane did not finish is deleted before its final commit, not left
behind a switch.

## How to A/B each (the M3 manager measures; lanes only write lines)

One line per board-lane x define in docs/apple-fast/ab-neural/<lane>.txt:

    CMD lane/apple-fast-neural <tag> bash tools/afn_ab.sh <tag> <binding> <board-lane> full 2 "" "-D MOJOLEARN_AFN_<DEFINE>"

tools/afn_ab.sh builds the binding twice under MOJOLEARN_NUMERIC_MODE=fast (arm A with no define,
arm B with the define), installs each into the package in turn and races OUR arm alone
(`bench_board_neural.py race --arms ours-fast`, no torch opponent re-run), alternated A B A B,
3 timed rounds each; it prints `AFN-AB <tag> arm=A|B ... median_ms=` per arm and
`AFN-DEF-SUMMARY <tag> A= B= ratio=B/A`, then judges quality: the kept outputs of the last rep
(train lanes: step losses, forward lanes: the output; tolerances in tools/afn_ab.py) and, on the
samba, mlp and block lanes, tools/neural_fast_quality.py for arm A then arm B at one seed from the
same init and batches, judged by its `pair` rule (`AFN-QUALITY ... status=OK|DIFF`). The baseline
form `identical fast` races the tier itself (docs/apple-fast/ab-neural/tier.txt: every GPU board
lane). The directory is not auto-queued (ab-neural/README.md): the lines name the merged branch.

## The keep rule

A define is KEPT when, on the M3, arm B's median is below arm A's (AFN-DEF-SUMMARY ratio below 1.0
on every board lane it touches, none above) AND every AFN-QUALITY line reads OK (quality within the
FAST spread: f32 reassociation noise on the outputs, the paired held-out rule on the train lanes).
Kept means: the change becomes the FAST default on Apple (its `comptime if` reads true without the
define), the `MOJOLEARN_AFN_<NAME>` switch is removed, and a `MOJOLEARN_AFN_<NAME>_OFF` define is
kept so the old FAST path stays reachable for one more A/B. Faster but DIFF is not kept: fix the
quality inside the parallel form or drop the change. Not faster is dropped (the define and its code
deleted), never left behind a switch. IDENTICAL is never touched by any of this. A kept define's
afn_ab.sh line flips to `"" "-D MOJOLEARN_AFN_<NAME>_OFF"` for the confirmation run.

## How the board shows the tier

tools/bench_board.py on an Apple box (default modes fast,identical) plans every GPU neural race with
`ours` (IDENTICAL) and `ours-fast` (FAST, the same public classes under MOJOLEARN_NUMERIC_MODE=fast)
beside the torch arms: 12 races, 64 cells; the rendered race table's `ours FAST / arm` column fills
for neural as it does for trees and classical, and the `ours-fast` row's quality columns are its
differences from `ours` (loss diffs on train lanes, max abs/rel output diff on forward lanes). NVIDIA
and AMD plan `ours` only, and `--modes fast` with the neural family is refused there by name. The
CPU neural lanes (*-infer, lm-host-train-step) are on no board in any mode. The M3 FAST board page
(docs/apple-fast/BOARD_M3_FAST.md) gains the neural rows from the first Apple board run of the
merged branch's wheel.
