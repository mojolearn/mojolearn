# lane/rf-mutex-claim-acquire

**2026-09-16. The repair for the MI300X random forest nondeterminism, with the memory-model
argument written out, applied to every site that shares the primitive. Cut from `main`
9257fd6de. NOT MERGED. Andrew wants this reconciled with Codex's work on the same defect
before anything reaches `main`.**

The defect itself, its history, the twelve MI300X legs and the trace evidence live in
`lane/rf-score-weighted-nondeterminism` (`docs/lanes/LANE_STATUS_lane-rf-score-weighted-
nondeterminism.md` and `docs/lanes/LANE_STATUS_MECHANISM.md` on that branch). This file
answers the five questions the addendum asked, names the repair, and carries the evidence
for it.

## The repair, in one paragraph

Every cross-block mutex in this repository spun on an ACQUIRE load and then took the lock
with a weak RELAXED compare-exchange. Those are two separate reads of the mutex, and only
the first carried an acquire. The repair adds one ACQUIRE load of the mutex after the claim
succeeds. Nothing else changes. The commit is `39b9b8750`.

| site | claim | repaired |
|---|---|---|
| `ensemble/decisiontree/batched_levelalgo/split.mojo` `_publish_to_global` | 1 | yes, with the control define |
| `extratrees/impl/decisiontree/batched_levelalgo/split.mojo` block publish | 1 | yes |
| `neighbors/impl/detail/fused_l2_knn.mojo` consumer and producer | 2 | yes |
| `neighbors/mutex_probe_main.mojo` consumer and producer | 2 | yes |

`-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` compiles the pre-repair claim in the forest binding.
It exists only so the A/B leg has a control that must be SEEN to move. It is not a default
and not a matrix row.

## 1. Is the memory-model reasoning correct? YES

The claim under test is that a relaxed compare-exchange taking the lock, preceded by a
separate acquire load, does not establish the synchronizes-with edge that an acquire
read-modify-write would. That is correct, and it follows from the definition rather than
from hardware folklore.

An acquire operation synchronizes with the release operation whose value it reads (or a
value in that release's release sequence). The spin load is an acquire, so it synchronizes
with whichever release wrote the zero it read. The compare-exchange is relaxed, so it
synchronizes with nothing, whatever value it reads. The two reads need not see the same
release. Concretely, with holders A, B and contender C on one node's mutex:

1. A releases (store-release of 0, call it R1).
2. C's acquire spin load reads R1's zero. C now synchronizes with A, not with anyone later.
3. B's claim consumes R1's zero first. C's claim on that zero fails and C spins again, OR
   C's claim is simply late.
4. B loads `split[node]`, merges, stores `split[node]`, releases (R2).
5. C's relaxed claim consumes R2's zero and succeeds.

C holds the lock without any acquire that reads R2. B's plain store of `split[node]` is
sequenced before R2, but R2 does not happen-before anything C does, so B's store and C's
plain load of `split[node]` are unordered. In the C++ model that is a data race with
undefined behavior. In hardware terms C may be served a copy of the slot that predates B's
store. C then merges its own candidate into that stale copy and writes the whole struct
back, and B's candidate is gone.

The only caveat to the addendum's wording is the sentence "an RMW must read the latest
value in coherence order". A read-modify-write reads the last value in the modification
order immediately before its own write, which is the property that puts it in the release
sequence. That is exactly why a plain acquire load AFTER the claim repairs it (section 3).

Why MI300X and not the other columns. Each of the MI300X's eight XCDs has its own L2, and
a plain load on one XCD can be served from a line that XCD cached before another XCD's
holder wrote the slot back. The acquire load at the spin invalidates that XCD's caches, but
it does so BEFORE B's store, and nothing invalidates them again between the spin and the
claim. A part with one coherent L2 has the same formal hole with a far smaller window,
which is consistent with the NVIDIA and Apple columns never having shown a move without
proving they cannot.

## 2. Does the fix close the hole, or narrow it? It CLOSES it

The failing step is "C holds the lock with no acquire that reads a value written by or
after R2". After the repair, C performs an acquire load after its claim succeeds. That load
reads the value C's own claim wrote. A compare-exchange is a read-modify-write, and the
release sequence headed by R2 is the maximal contiguous run of read-modify-writes that
follow R2 in the modification order of the mutex, so C's claim is in R2's release sequence.
An acquire that reads any value in a release sequence synchronizes with the release that
heads it. R2 therefore synchronizes with C's post-claim load, and B's store of
`split[node]` happens-before C's load of it. There is no remaining path by which C can
hold the lock without having synchronized with the release it claimed against. The window
is not narrower; it does not exist.

Two things the repair does NOT rely on. It does not need the spin's acquire load, which
stays only because removing it would change more text than the defect requires. It does
not need any vendor-specific instruction. The acquire load is the same one the spin already
used on every column, so it legalizes on Apple, where the compiler rejects an acquire
compare-exchange by name.

## 3. Is there a better-shaped fix? The three alternatives, evaluated

**Acquire success ordering on the compare-exchange.** Formally equivalent to the repair,
and the direct translation of the reference's `atomicCAS` plus `__threadfence`. Apple
rejects it by name (`neighbors/mutex_probe_main.mojo` reproduces the message), so it needs
a per-vendor `comptime if` and leaves Apple on a second spelling. Whether gfx942 legalizes
it is what the other lane's probe measures. The post-claim acquire load reaches the same
edge with one spelling for every column, which is why it was chosen.

**Make the `split[node]` store itself a release, or the loads acquires.** Wrong side of the
edge. B's stores are already ordered before B's release store of the mutex by the release.
The missing half is on the acquirer, and it is missing for the mutex, not for the slot.
Field-wise acquire loads of the seven fields would have to synchronize with field-wise
release stores of the same fields, seven atomics each way per publish, and `Split` holds a
Float64 threshold and an Int64 count for which 64-bit atomics are a compile error on Apple
(`ensemble/checks/atomic_width_probe.mojo`). Heavier, less portable, no more correct.

**Remove the mutex.** One block per (node, sampled column) could write its candidate to a
per-node scratch of `n_sampled_cols` slots, and a second kernel per round could reduce them
in column order. Because sampled columns are a permutation, `update` over distinct colids
is a plain maximum on a total order, so the result would be identical to a correct mutex
merge. The costs are one extra launch per round per tree batch, a scratch of
`max_nodes * n_sampled_cols` splits, and a redesign of a seam the reference implementation
does not have. The Apple column already pays eight hours to launch overhead, so adding
launches to fix an ordering bug that one load repairs is the wrong trade. Kept as the
fallback if the leg shows the repaired claim still moves.

## 4. Is the diagnosis right on four traced divergences? YES, and not because of the count

Four is thin for a rate, and it would be thin for choosing between two mechanisms that
predict different distributions. It is not thin for this conclusion, because the argument
is structural. With every candidate merged correctly, the merged result is the maximum of
the candidates under a total order on `(gain, colid, ...)`, which does not depend on
arrival order at all. Any fit whose merged split differs from another fit's, over
bit-identical histograms and an identical column sample (which leg 11 checked at the exact
diverging round), has by definition dropped a candidate or read a partial state. Three of
the four traces show a lower colid winning an equal-gain tie that `update` awards to the
higher colid, and the fourth shows a strictly lower gain winning. Both are impossible under
a correct merge and both are exactly what a stale read of `split[node]` followed by a
write-back produces. One trace would have sufficed for "a candidate was lost". Four in the
same direction, and zero in the other, is what one expects from a mechanism that can only
ever erase.

What the count cannot yet settle is that THIS hole is the only one. That is what the A/B
leg is for. The control arm has to move at the rate the earlier legs measured, and the
repaired arm has to be stable at a size where a null is strong.

## 5. Other spellings of the same hole

Every `compare_exchange` in the repository was listed (`git grep -ln compare_exchange`);
the four files above are the complete set, and every claim in them now carries the
post-claim acquire. The gbdt single-pass reorder (`gbdt/gpu_util/kernel/
reorder_single_pass.mojo`) uses acquire loads and release stores without a claim, and it
reads its payload out of the atomic word itself, so the acquire load IS the read of the
released value and the edge is sound. The plain store of `leaf_total_zeros` there is read
by a later launch, which is ordered by the queue. No other spin-and-claim exists.

The DEVIATION 2502 word store of the purity flag outside the lock
(`builder_kernels_impl.mojo`) writes the value every merge writes and regression never
marks, so it is inert for this defect; noted so nobody re-derives it.

## Evidence

Filled in as the leg lands. Layout follows the earlier legs, outside the repo under
`~/mojolearn-evidence/rf-mutex-claim-acquire/leg-N/remote/identity/`.

| arm | binding | fits | moved | distinct |
|---|---|---|---|---|
| stock_prerepair (control, `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1`) | pending | | | |
| claimfix (default) | pending | | | |

Reading the table. The control must move, or the leg proves nothing about the repair. At
the 4.3% to 5.3% rate the earlier legs measured for this fixture, 0/300 on the repaired arm
has p under 1e-6 against "unchanged"; 0/100 would still leave p near 0.01 and would need a
second leg. Digests of the two `.so` files must differ.

Apple. The repaired forest binding was built on this Mac at one core in identical mode to
show the spelling legalizes on Metal (build log in the scratchpad, result recorded below
when it finishes). The repair adds an ordering constraint and no arithmetic, so no Apple
bit can move; an rf-reg spot check on the base fixture against the current reference is
the confirmation, taken under the Metal lock.
