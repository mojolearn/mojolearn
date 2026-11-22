# Apple FAST neural training source experiments

**Not tested.** T01–T12 are programmed on `ideas/apple-fast-neural-20261006`,
forked from `main` at `fd6cf8045`. No source in this delivery has been compiled,
verified, measured, linted, or run. There are no new quality or performance
results. Existing historical comments remain evidence for their original
scope only; they do not qualify this campaign.

The independently selectable recipes are in [training.json](training.json).
Defines there are bare names for future `-D` arguments. Every new switch uses
`MOJOLEARN_AFN26_*`, is guarded by the existing Apple GPU FAST predicate, and
defaults off. Existing default-on `AFN_LM_BWD_NOSYNC` remains fixed. Do not add
legacy `*_ALL` defines to these recipes: they change more than the named arm.

## What is programmed

| Cards | Source changes | A/B interpretation |
| --- | --- | --- |
| T01–T03 | Independent aliases in `training/byte_lm_afn.mojo` select existing one-completion, head CE fusion, and flat parameter/gradient view paths. | A retains incumbent execution; B enables only that alias. Aliases preserve the existing disabled defaults. |
| T04 | Independent alias in `training/byte_lm_afn_grad.mojo` selects existing occupancy-driven split-K weight gradients. | A unsplit, B split when hardware/work policy allows. The split path zeroes the output then uses global f32 atomic tile accumulation; there is no separate split scratch/fold kernel. |
| T05 | Separate aliases select block backward fusion, GEMM residual epilogue, and norm1 residual fusion. | Three independent comparisons and one explicit combined comparison. Default backward no-sync stays identical across A/B. |
| T06 | Independent optimizer scan, clipping, SGD multitensor, Adam vector4, scratch residency and general CE aliases in `training/afn_optim.mojo`. | Six single-mechanism variants plus optimizer-only and optimizer/loss combinations for the applicable SGD and Adam/AdamW consumers. |
| T07 | Independent fused-step alias in `training/mlp_fast.mojo`. | Existing fused SmallMLP step versus incumbent public operation. |
| T08 | Independent resident and multistep aliases in `training/mlp_fast.mojo`. | Resident versus per-call fused arithmetic; multistep versus resident single-step calls with the same ordered minibatches. |
| T09 | Weight-gradient minimum work 128 and maximum splits 8 controls. | Keep split enabled on both sides; compare minimum, cap, and both. Incumbents remain minimum 256/cap 16. |
| T10 | Optimizer 128- or 512-thread block controls. | Keep the same selected optimizer mechanism(s) on both sides. Scan grids, shared reductions and update launches follow the selected constant. Incumbent is 256. |
| T11 | Separate LM-head and general-loss 128- or 512-thread CE controls. | Keep the applicable fusion parent on both sides. CE geometry is independent of optimizer and LM status-scan geometry. |
| T12 | SmallMLP 32- or 128-row tile controls and derived resident partial capacity. | Keep fused, resident, or multistep parents fixed on both sides. Incumbent tile is 64. |

All affected constants still have incumbent values without new defines.
Geometry defines do not silently enable a parent fusion. Use one alternative
block/tile size per family. If both alternatives are supplied accidentally,
the first branch wins (128 before 512 for optimizer/CE, 32 before 128 for MLP);
such a combination is not an experiment recipe.

T09 changes work partitioning only. Its policy continues to use output tile
count, the hardware occupancy target, K length, and GEMM K-tile alignment.
Fewer K elements per split may expose more parallel tasks and may lose through
atomic contention; a smaller cap bounds fan-in. Neither is a measured winner.

T10 and T11 use power-of-two reduction widths. Shared arrays, reduction trees,
strides and launch sizes use the same family constant. General fused CE keeps
its one-simdgroup (32-thread) path when the vocabulary fits those lanes, and
uses the selected alternative above that hardware boundary. LM fused CE uses
its own selected block for all vocabulary widths. Full vocabulary/row coverage,
mean divisors, smoothing and ignore-index contracts remain the intended
semantics. New geometries can change summation bits and must be qualified.

T12 derives `MLP_FT_MAX_BLOCKS` as
`(MLP_FT_MAX_ROWS + MLP_FT_ROWS - 1) // MLP_FT_ROWS`.
The existing public per-step row capacity is 256; this is a public API cap,
not an experiment dispatch threshold. Changing the tile therefore sizes the
resident slot region for every legal public batch. The nonresident path
already allocates partial slots from actual block count. A multistep call
reuses the slot region serially on its ordered context, while its input/output
transport regions retain capacity for every submitted batch. Smaller tiles
reduce shared storage and increase partial count; larger tiles do the reverse.

## Future caller and binding map

These are source destinations for a later authorized qualification campaign,
not commands run by this delivery.

| Scope | Binding and caller destinations | Full workload definition and caveats |
| --- | --- | --- |
| LM orchestration, head and views (T01–T03) | `bindings/_mojolearn_byte_lm.mojo`, `bindings/build_byte_lm.sh`, `training/byte_lm.mojo`; public `LanguageModelTrainer` | `tools/bench_board_neural.py` `lm-train-step` corpus and full-shape recipe. Record actual dataset hash, sequence dimensions, all model settings, seed, steps and timed boundary. Audit chunked-head settings and confirm the intended head route in future qualification. |
| Transformer backward (T04–T05, T09) | LM binding above and transformer backward bindings consumed by Samba; implementation callers in `transformer/checks/transformer_backward.mojo` | Full LM training and every Samba training route that reaches the changed backward operation. Caller reachability and full-data mapping remain pending. A backward-only component result is insufficient. |
| Shared optimizer and CE (T06, T10, general-loss T11) | `bindings/_mojolearn_training.mojo`, `bindings/build_training.sh`, `training/checks/optimizer.mojo`, `training/estimator.mojo`; neural consumers only | Full intended LM, Samba and MLP training wherever those consumers reach the shared path, plus actual neural SGD/Adam/AdamW consumers. Fused MLP, one-completion LM and chunked heads can bypass shared paths. Map the actual route before claiming coverage. |
| LM-head CE geometry (LM variants of T11) | Byte-LM binding and caller above | Full LM training with `LM_HEAD_FUSE` on both sides. The geometry-only flag does not enable the head fusion. |
| SmallMLP (T07–T08, T12) | `bindings/_mojolearn_training.mojo`, `bindings/build_training.sh`, `training/mlp_ops.mojo`; public `SmallMLPTrainer` | `tools/bench_board_neural.py` `mlp-train-step` full declared workload. Resident and multistep bindings must actually be called. Compare the same complete sequence of optimizer steps, not larger or fewer logical steps. |

`tools/neural_fast_quality.py` and
`experiments/performance_ideas/apple_fast/lm_task.py` provide existing quality
task definitions. They are not substitutes for each affected estimator's full
dataset and complete operation. A `full` selector alone does not establish
coverage; audit intrinsic caps and record the real consumed rows/tokens.
Missing dataset, route, or quality mappings stay pending. Standalone optimizer
or CE component timings cannot qualify the full neural consumers.

## Qualification still required

No build, compile acceptance, binding-load check, quality result, performance
measurement, source verification or default promotion is included here. Every
card and variant is `not_tested`. Source availability alone cannot show that a
kernel instantiates successfully or that the intended runtime branch is used.

A future campaign must preserve architecture, optimizer settings, clipping,
seed, step count, data order, target/mask semantics and output consumption.
Compare full trajectories, held-out loss/perplexity or classification quality,
logits and gradients, optimizer moments, refusal/rollback and checkpoint/resume
under the existing acceptance rules. Bits may change across versions; quality
tolerances may not be widened. Use ragged tiles, neighboring supported shapes,
non-board data, repeated calls and non-finite cases as applicable, without
dispatching on those cases in source.

Measure complete operations with required preparation, transfer, fit/training,
completion and consumed outputs. Retain cold allocation and repeated use as
separate cases. Resident experiments include required state download and
checkpoint paths. Multistep comparisons retain every logical step and output
contract. Record all toggle combinations and their source/binary/hardware
provenance; individual wins do not prove the combined configuration wins.
Preserve failures, losers, neutral results and pending scope. No new result is
claimed by this source-only delivery.
