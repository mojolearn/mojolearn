# rf-score-weighted: the mechanism is a LOST CANDIDATE in the cross-block merge

**2026-09-16, legs 11-14. The divergence is introduced in `find_best_splits_kernel`, over
histogram data that is bit-identical between the two runs, and it takes the form of a
candidate VANISHING from the merge rather than a tie being resolved differently.**

**A fix is identified and its EFFECT is replicated. Its MECHANISM is NOT demonstrated, and
nothing has been landed as a default.**

> **SCOPE, AND THIS IS THE PART THAT MATTERS BEYOND RANDOM FORESTS.**
> `extratrees/impl/decisiontree/batched_levelalgo/split.mojo:512` and
> `neighbors/impl/detail/fused_l2_knn.mojo:624,730` use the **identical mutex spelling**,
> verbatim, and **have never been measured**. If the ordering hole is real, this is not a
> random-forest bug; it is a shared-primitive bug that happened to surface in random forests,
> and ExtraTrees and fused kNN on AMD are exposed by the same argument.

## Leg 11: the first differing STAGE

Stage trace on the SHIPPED 0.8.5 wheel, no code change (`instr.trace.enabled` is a runtime
check inside `fit_forest`, which builds a live `FitInstruments()` whose `IdentityTrace()`
reads `getenv(MOJOLEARN_IDENTITY_TRACE)`).

Untraced positive control 4/100 REPRODUCES. Traced 6/150, `identical_trace_but_model_moved=0`.

    seq=124  tree0.batch6.round0.colsamples   i32  976      MATCHED
    seq=125  tree0.batch6.round0.cols0.hist   u8   624640   MATCHED
    seq=126  tree0.batch6.round0.cols10.hist  u8   374784   MATCHED
    seq=139  tree0.batch6.round0.cand         u32  610      <-- FIRST DIFFERENCE

Ordering CHECKED, not assumed: the K=4 pipeline interleaves trees, so tree1-3's batch6 records
sit at 128-138 and tree0's own histograms are genuinely earlier. The whole differing set is 8
of 693 records, a causal chain inside ONE tree, ending at `nodes`/`leaves`.

## Leg 12: WHICH FIELD moves

`.cand` packs 10 u32 lanes per split. Dumping the sidecars and diffing lane by lane:

| repeat | `colid` | `best_metric_val` | reading |
|---|---|---|---|
| 34 | 15 -> **13** | **identical** | lower colid won |
| 71 | 14 -> **13** | **identical** | lower colid won |
| 98 | 15 -> **7** | **identical** | lower colid won |
| 86 | 15 -> **3** | ref **higher** | lower gain won |

`Split::update` awards an equal-gain tie to the **higher** `colid`. Three of four show a LOWER
colid winning at **bit-identical gain**, which the total order forbids -- so the higher-colid
candidate was **ABSENT from the merge**, not outvoted. The fourth lost a strictly higher gain.
One direction every time: **a lost candidate**.

## The suspect line

`_publish_to_global`'s claim. The ACQUIRE sits on the spin **load**; the lock is taken by a
weak **RELAXED** compare-exchange. An RMW must read the latest value in the coherence order; a
plain acquire-load must not. When the two observe different releases, the previous holder's
**plain non-atomic** store to `split[node]` is unordered against this thread's plain read of
it. DEVIATION 106 is amended in place with this reasoning.

## Legs 13-14: the A/B, replicated

`-D MOJOLEARN_RF_ACQUIRE_CAS=1` moves the ACQUIRE onto the claim.

| leg | control (`stock`, 24c92e17) | test (`acqcas`, a705f9a1) | P(0 at control rate) |
|---|---|---|---|
| 13 | 7/300 moved | **0/300** | 9.1e-4 |
| 14 | 6/300 moved | **0/300** | ~1e-3 |

Digests provably distinct; arms independent. The stock digest is byte-identical to leg 9's
pre-edit binary, so **both diagnostic defines are provably inert when absent** -- demonstrated
by digest, not asserted.

## What is NOT established, and why I am not calling it proven

A `0/300` is consistent with the acquire ordering repairing the edge. It is **equally
consistent with different codegen perturbing timing enough to hide a 2-5% race.** That is the
same masking confound raised against the `N_BLKS_FOR_COLS` cap arm, and a result that suits
the hypothesis does not get less scepticism than one that does not.

The primitive probe is what separates them, and **it has not yet done so**:

- **Leg 13**: the probe failed to compile, both arms, from my own error --
  `'comptime if' must be contained in a function`. Mojo rejects a module-scope `comptime if`.
- **Leg 14**: the probe ran and returned a **NULL**: `relaxed: got 512 want 512 shortfall 0`
  with `unlocked(SABOTAGE)` losing 508 of 512. The sabotage proves the cell contends, so the
  probe can see lost updates -- the shipped spelling simply did not lose any.

**That null is underpowered, and the reason is my probe's design**: each block acquired the
mutex ONCE with a two-instruction critical section, so the spin almost never looped -- and the
hypothesised window REQUIRES the spin to loop, with the load observing one release while the
claim lands on another. The probe is now rewritten with `ROUNDS = 64` acquisitions per block
and a `HOLD` widening whose result is stored into a sink the host reads, so a compiler cannot
elide it and silently restore the narrow section.

## Measured capability

`column_has_acquire_rmw` (`ensemble/checks/atomic_matrix.mojo`), AMD **MEASURED** two ways on
gfx942: the probe printed `acquire_arm_compiled True` with `acquire: got 512 want 512`, and
`_mojolearn_rf.so` built and ran under the define with a distinct digest. Apple is **False**
(documented by name in DEVIATION 106). NVIDIA and the rest are **UNPROVEN** and conservatively
False, because unproven is an honest value and a transcribed guess is not.

## What would settle it

Re-run the strengthened probe. If the relaxed spelling loses updates where the acquire
spelling does not, the mechanism is demonstrated away from forests and the fix is proven in
the strong sense. If it stays exact, the A/B remains an effect without a mechanism, and
landing an ordering change across three subsystems on that basis is a judgement call rather
than a measurement -- which is why it has not been landed.

**Owed regardless of outcome:** ExtraTrees and fused kNN carry the same spelling and no
measurement at all.
