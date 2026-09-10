# Tree code reuse audit — 2026-09-10

This is a source audit of the current tree lane, not a performance claim. RF lives in `ensemble/`, ET in `extratrees/`, and boosting in `gbdt/`. The user's current instruction prioritizes shared GPU implementations. Historical comments preserving duplicate files to avoid concurrent lane edits are not a lasting architectural justification.

`PORTING_RULES.md` requires preserving the actual upstream computation and dispatch, including fused kernels. Its GPU-only rule permits required host control-plane work and independent test oracles; it does not justify a public CPU training or prediction backend. Sharing a scalar host/device arithmetic primitive does not introduce such a backend.

## Consolidation implemented

RF and ET's generic `_log_seam` had identical executable tokens after removing their explanatory docstrings and whitespace. Both dispatch Float32 through `checks.numerics.identical_log`, then cast back; all other floating widths use `std.math.log`. The normalized pre-extraction body hash was `46693d137c30098a470e6533e5513d38c7c72277d04ee2be3c9691825138bebc`.

`core/tree_math.mojo:20` now owns `tree_log`. Compatibility imports retain `_log_seam` at every existing objective call in `ensemble/decisiontree/batched_levelalgo/objectives.mojo:307` and `extratrees/impl/decisiontree/batched_levelalgo/objectives.mojo:383`. This preserves Float32 FAST/DETERMINISTIC stdlib log and IDENTICAL portable log; it does not change fusion, gain association, thresholds, or objective policy. The helper records both source pins: cuML `00094f7` for ET and v26.08 `265b9da6` for RF. Unused imports were removed.

The independent existing checks are `ensemble/checks/objectives_check.mojo` and `extratrees/checks/objectives_check.mojo`. Each is run with `pixi run mojo run -I . PATH`, once without a mode define, once with `-D MOJOLEARN_NUMERIC_DETERMINISTIC=1`, and once with `-D MOJOLEARN_NUMERIC_IDENTICAL=1`, under `tools/with_build_lock.sh`. Commands and exit codes are retained in `bench/results/tree_reuse_2026-09-10/`. All six local runs completed with exit code 0: RF reported `objectives_check: ALL OK`, and ET reported `PASS -- every cell`. Source identity and local execution are not a cross-vendor execution result.

## Further concrete reuse candidates

| Candidate and source locations | Actual overlap | Safe boundary and required gate |
| --- | --- | --- |
| FNV byte fold: `ensemble/decisiontree/batched_levelalgo/random_utils.mojo:100–120`, `extratrees/checks/pcg_rng.mojo:47–94` | Same UInt32 constants and four low-byte-first xor/multiply rounds; actual cuML `cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh:100–119` confirms the source. | Move constants and byte fold to a core hash module with old-name reexports. Keep seed width handling, optional high-word fold, salts, PCG and Philox schedules separate. Gate existing RNG vectors including high seed bits and full model fingerprints. Product ET importing RNG from `checks/` is another reason to correct ownership. |
| Float sortable-key transform: `core/segmented_sort.mojo:101–115`, `gbdt/gpu_util/kernel/segmented_sort.mojo:87–101` | Same reversible Float32 bit ordering. The core header cites temporary lane ownership as the duplication reason. | Share bit transforms first. Preserve signed zero, subnormal bits, NaN policy and payload stability; the full sort APIs differ. |
| Three-phase scan and segmented radix plumbing: `core/segmented_sort.mojo:145–242`, `gbdt/gpu_util/kernel/segmented_sort.mojo:140–249`, `gbdt/gpu_util/kernel/scan.mojo:71–162` | Repeated block scan, block totals, carry propagation and scatter shapes. | A shared typed scan can parameterize payload/operator while retaining block geometry, signedness, overflow bounds and launch barriers. Core sort has keys-only uniform segments; GBDT also has payloads and explicit segment offsets/sizes. Do not replace both with one superficial API or introduce extra global-memory passes. Gate empty/tail/multiblock segments and stable ties. |
| Generic flush seam: RF objectives `:282–288`, ET objectives `:386–392` | Same Float32 `ftz` dispatch, other widths pass through. | Small follow-up beside `tree_log`, with the same objective checks. No reason to move entire objectives to remove a small seam. |
| `max_nodes`: RF builder `:104–113`, ET builder `:101–105` | Same valid-depth reserve hint: complete-tree size below depth 13, otherwise 8191. | Can share after defining valid input domain; cast/add order differs. This is an allocation hint, **not a depth cap**. |
| Sparse node storage: `ensemble/flatnode.mojo:94–231`, `extratrees/impl/decisiontree/flatnode.mojo:114–210` | Same conceptual five fields and sibling convention (`right = left + 1`), leaf marker `left == -1`. | A canonical core node plus compatibility accessors is plausible, but a plain reexport now breaks APIs: RF uses private-prefixed fields and explicit default/field constructors; ET exposes fields, fieldwise initialization and additional traits. Audit direct field access and constructor/trait requirements before migration. Gate serialization layouts, leaf markers, traversal and model bits. |

## Differences that must remain explicit

`TreeMetaDataNode` is not an identical duplicate: RF's definition in `ensemble/decisiontree/decisiontree.mojo:424–451` includes `train_time`, while ET's `extratrees/impl/decisiontree/flatnode.mojo:313` has different field ordering and helper methods. Reexporting one as the other without migration is unsafe even though their names match.

Split winners are also not interchangeable. RF's `ensemble/decisiontree/batched_levelalgo/split.mojo` carries bin-interval midpoint and non-associative equal-gain reduction rules (`update` and `update_bin`). ET's split helpers support keyed feature/threshold ties and explicit salts. Their common integer counts are shareable; tie order and threshold generation are model semantics.

RF bins candidate thresholds and can share feature quantization across trees (`ensemble/randomforest.mojo:2441–2499`). ET evaluates random per-node thresholds without RF's bin histogram. GBDT uses gradient/Hessian planes and distinct symmetric, depthwise and loss-guided growth. A shared RF histogram is not automatically an ET or GBDT replacement. Share storage, count/reduction primitives and launch scaffolding only where their semantics match; keep algorithm-specific candidate generation and dispatch.

## Reuse already present

RF and ET already use `core.philox` (`ensemble/randomforest.mojo:40`, `extratrees/impl/randomforest/randomforest.mojo:99`, ET builder `:97`). Shared numeric seams, scan/reduction utilities and device infrastructure already exist in `core/` and `checks/numerics.mojo`. ET's single-tree builder already calls its forest driver (ET builder `:3624`); duplicating another unified single-tree driver would add code rather than remove it.

## GPU-only follow-through and priority

1. Complete the small arithmetic/hash/bit-key extractions with compatibility imports and existing independent gates. These reduce maintenance immediately without changing the execution graph.
2. Unify flat-node storage deliberately, then implement shared GPU flat-forest inference with per-row traversal and the existing tree accumulation order. Current ET binding `bindings/_mojolearn_trees.mojo:305` rebuilds host metadata and invokes host traversal; RF's `bindings/_mojolearn_rf.mojo:485` similarly exposes implemented host prediction. Moving that product work onto the GPU is higher impact than merging superficially similar training functions. Classification argmax/ties, regression scaling, accumulation mode, empty batches and archive models need explicit coverage.
3. ET's public CPU-training route has now been retired in the follow-up GPU-only slice: Python rejects non-GPU devices, the binding refuses non-1 selectors before pointer access, and unsuffixed native fits dispatch to the existing GPU trainers. Explicit `*_reference` host routines retain oracle checks; see `extratrees/README.md` for migration. Prediction remains a separate GPU migration.
4. Consolidate scan/sort kernels after their contracts are explicit. Next share prepared training storage and scheduling where useful; preserve RNG schedules, floating reduction order, integer-count overflow checks and fused histogram dispatch. Every kernel replacement needs full model/prediction equivalence in all supported modes plus isolated full-fit timing before changing defaults.

The original shared-log extraction changed no GBDT native source, binding, or Python API; the separately authorized GPU-only follow-up updates ET training entrypoints as described above. Numeric identity is a requirement for these refactors; multi-vendor claims require actual multi-vendor evidence beyond these local checks.
