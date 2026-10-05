# Remaining unmatched classical/tree requests — 2026-10-04

Read-only source/history review at main `1c51f03cbfb271b513387ca4755a373e11f192ed`.
Input: `~/mojolearn-evidence/apple-fast/branch-audit-20261004/unmatched-requests.json`.
No cloud access, builds, numerical runs or queue changes. No new measured winners.

| Original unmatched tag | Disposition | Evidence / next action |
|---|---|---|
| `linsvr-fix-taxi` | Renamed historical trial; do not resurrect | `LSVR_FASTPATH_FIX` ledger row records `linsvr-fix-taxi-x` on M2 (-0.2%, noise) and M3 istella +1.5%, dropped. Local `m2runner/verdicts-1.md` confirms the renamed trial. Taxi already ships `LSVR_ALL` (M3 117.2 -> 27.0 ms), including a stronger fused/batched objective path. No unmeasured current-main speed claim follows from the missing original tag. |
| `sym-ordered-fbo-istella` | Implementation already bundled in accepted wide default | `ORD_FOLD_BINS_ONE` standalone flag removed; implementation ships in `ORD_ALL` for more than 32 features. Bundle istella 75,758 -> 63,554 ms with improved AUC/logloss. Standalone taxi was -1.0%, labeled noise in historical ledger. No independent istella result established by this review; a fresh independent attribution experiment would require current-main toggles and is not an additional missing implementation. |
| `sym-ordered-std-taxi` | Unproven narrow standalone; stale request, low priority | Standalone flag removed; code retained inside wide-only `ORD_ALL`. Historical independent istella result -1.8%, labeled noise. Narrow bundle failed quality. This does NOT establish that STD alone fails quality on taxi; its reordered sum needs independent quality evidence if reconstructed. No validated standalone taxi result found. |
| `sym-ordered-lean-taxi` | Unproven narrow standalone; stale request, low priority | Standalone flag removed; code retained inside wide-only `ORD_ALL`. Historical independent istella result about -0.8%, labeled noise. Narrow bundle quality failure is not evidence against buffer reuse alone. A fresh scoped current-main candidate could isolate buffer reuse with independent quality checks, but the old branch is not queue-ready. |

Sources: `docs/apple-fast/EXPERIMENTS.md` (`ORD_*`, `LSVR_*` rows),
`gbdt/methods/ordered_fast_switches.mojo`, historical
`lane/apple-fast-sym-ordered@27b912397` and
`lane/apple-fast-linsvr@c649076a4` request explanations.
The local batch/m2runner request files contain renamed `*-x` requests for all
four original tags; request presence alone is not execution evidence.

## SDK control checkpoint

`73bb9ac6ebec3d83e86b580a7b686abee0c9524d` explicitly says uncompiled, with no
quality helper/binding. It must not be admitted directly. It adapts caller-owned
contiguous buffers into SDK NN/NT matmul for non-split decomp calls, avoiding
new allocations, uploads or transpose materialization. It leaves transpose-A,
N=1, zero-K, aliasing, row views and incumbent split plans on existing routes.
Potential value is establishing whether SDK outperforms the *actual shipped AFN*
backend for eligible resident/Kit operations. No numerical or timing benefit is
established. Existing RSVD/NMF AFN gains were against a former per-cell backend,
not against SDK; the generic G0 screen is not this comparison.

Before queuing: review/compile API and lifetimes on M2, implement actual-wrapper
quality and reach counters, independent FP64 checks, NN/NT/pool reuse/alias and
offset controls, split boundaries and downstream caller coverage; then provide
pinned artifacts, manifests and a prepared M3 helper. Square 1024 cubed currently
selects the decomp split plan and is deliberately ineligible, so it cannot be
used to claim this adapter's reach. This is source work to finish, not a missed
ready measurement.
