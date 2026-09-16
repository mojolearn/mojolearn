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

> **2026-09-16, measured on Apple after the sections below were written. THE
> REPAIR IS NOT IN THE BINARY.** Building this branch with and without the
> acquire load, and building the extratrees file with the line deleted, gives
> byte-identical `__TEXT` and `__DATA`; a one-line arithmetic change in the
> same function moves 270981 bytes, so the comparison can see a code change.
> The compiler deletes an `Atomic.load` whose result is thrown away with
> `_ =`. Sections 1 to 5 below argue the memory model and remain the argument
> for what the repair SHOULD say; they are not a statement about what the
> shipped binary does, and no claim here about the hole being closed has been
> demonstrated on any column. The line has to be made to survive the optimizer
> before any A/B of it means anything. See the Evidence section.

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

## Evidence, Apple (Metal) and the binaries, 2026-09-16

Run on the shared M4 under `mac_slot.sh` (one core, `nice -n 19`,
`MOJOLEARN_COMPILE_JOBS=1`, one Metal job at a time), from the worktree at
`lane/rf-mutex-claim-acquire` `2e9d7dc65`. Logs and JSONs under
`/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/460f8328-3642-45a3-a14f-579fc7e7363f/scratchpad`.

### THE HEADLINE: the post-claim acquire load EMITS NO CODE

The repair does not survive compilation. Built two ways, the arms are the same
program.

| pair | how the arms differ | bytes differing | `__TEXT` + `__DATA` |
|---|---|---|---|
| `_mojolearn_rf.so` fixed vs stock | `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` | 54 | IDENTICAL |
| `_mojolearn_trees.so` fixed vs stock | the acquire line DELETED from the source | 54 | IDENTICAL |
| `_mojolearn_rf.so` fixed vs sabotage | one arithmetic line added | 270981 | DIFFERENT |

The 54 differing bytes are the same three regions in both pairs and carry no
code: the `mktemp` install-name suffix that `build_rf.sh` and `build_trees.sh`
bake in (6 bytes), `LC_UUID` (16 bytes) and the ad-hoc code-signature slot in
`__LINKEDIT` (32 bytes). Both files of each pair have the same length.

The third row is the control and it is why the first two rows mean something.
A one-line arithmetic change in the same function moves 270981 bytes, so the
comparison can see a code change; it does not see this one.

WHAT WAS MEASURED is the effect, not the mechanism: adding or removing the
line produces byte-identical code and data, and the Metal AIR blobs live in
`__TEXT`, so they are identical too. On the Apple column the repair is not
narrow, it is ABSENT from the artifact.

THE LIKELY CAUSE, not yet confirmed by reading emitted IR, is that
`_ = Atomic.load[ordering = Ordering.ACQUIRE](mutex)` throws its result away,
and a non-volatile atomic load whose value is dead and whose ordering is not
sequentially consistent may be eliminated. Confirming that, and finding a
spelling that survives (the value has to be consumed, or the ordering has to
be attached to an operation that is kept), is the lane's next step. A stale
compiler cache is excluded: the extratrees pair differ by FILE CONTENT, which
the cache must key on, and the sabotage build of the same file family did
change the binary.

This was NOT caught by digest comparison, and it would not have been caught by
the A/B leg either. `build_rf.sh` builds into a fresh `mktemp` directory and
the compiler bakes that path into the install name, so two builds of the SAME
source always have different sha256. The leg's guard
(`tools/rf_nondeterminism/rf_claimfix_ab_body.template.sh`, the
`DIGEST COLLISION -- arms are NOT independent, results VOID` branch) compares
exactly those sha256 values, so it can never fire. Its two arms would have
been one program run twice, and whatever it reported, moved or stable, would
have said nothing about the repair. Compare `__TEXT` and `__DATA`, not the
file digest.

### What the identity comparison therefore says

The Apple A/B was run in full before the binaries were compared, so it is
recorded here, but it rests on two builds of one program and CANNOT fail. It
is not evidence that the repair is inert on output; it is evidence of nothing.

Nine rf lanes (`rf-clf`, `rf-reg`, `rf-clf-entropy-log2-noboot`,
`rf-clf-balanced-parallel`, `rf-reg-poisson`, `rf-reg-gamma-ig`,
`rf-score-weighted`, `par-forest`, `par-forest-pool`) on all nine fixtures,
two repeats, `--diff`: `IDENTICAL=70` train, `IDENTICAL=140` infer and model,
`IDENTICAL=70` batch. `rf-score-weighted` REFUSED all nine cells in both arms
(it needs `_mojolearn_metrics.so`, not built in this worktree), so 9 cells
rest on no hash.

### Two Apple run-to-run movers, one on each arm, neither replicating

Taken while the Mac was at load 20 to 25 with several agents:

  - `rf-reg-gamma-ig/wide` MOVED on the stock arm: two fits of the same seed
    gave `fbdbb762bf85ec73` and `f76e788124ef87d3`.
  - `par-forest/denormal_ftz` REFUSED on the fixed arm, from the lane's own
    check: `fit_forest predict_proba and plain predict_proba differ: 1236
    bytes of 16384`.

Replicated at six repeats on a quiet machine, both arms: all four cells
STABLE and IDENTICAL across arms, `par-forest/denormal_ftz` settling on
`9ee018b4ad7d8c6c`, the value the stock arm had already recorded. Neither
event reproduced.

NOTHING IS ATTRIBUTED TO THE SPELLING. One event landed on each arm, and the
arms are the same program, so the spelling cannot be the variable. What the
two events do show is that the Apple column moves run to run in these lanes,
which contradicts this file's earlier sentence that the NVIDIA and Apple
columns have "never shown a move". The rate is not established: one event in
two repeats, then zero in six. `rf-reg-gamma-ig` fits `inverse_gaussian` with
`max_features=None`, which resolves to a fraction of 1.0, so all 16 columns
are merged through the mutex per node, and it is on the `wide` fixture; that
is the same shape the MI300X legs named as their reproducer. That is a
coincidence worth a powered run, not a result.

### Binaries built, all identical tier, Apple, one core

| file | spelling | sha256 |
|---|---|---|
| `_mojolearn_rf.so` | repaired (default) | `b3479092fed5a6c23749a9f1b120d51cfddfe1b7baf46aad5b06a1085f843727` |
| `_mojolearn_rf.so` | `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` | `2c501521b6459396209e33cdb6d79ddbf4e9b28938ecac5ec42a8b76e8619d70` |
| `_mojolearn_rf.so` | repaired + scratch sabotage | `0696aad3ce81da9d25d8710e84ce1271dc03bccc551376939f5413911976e781` |
| `_mojolearn_trees.so` | repaired (default) | `173ab5164db9b0f670bd4378bd1096fcf2492708daa555ccdb709e23eda7e1e0` |
| `_mojolearn_trees.so` | acquire line deleted | `be3a27506a2950b8977fca5d09d39d2d98c25005aec0c7405c56f19d15ebcd4c` |
| `_mojolearn.so` | repaired (base binding, kNN sites never reached) | `6edf210855760da43e37a65bdda8526fa80e7b8eb8c864d53a8b4dcf145b6b2d` |

The define was seen on the compiler command line, not inferred: `build_rf.sh`
was run under `bash -x` and the traced line reads
`pixi run mojo build -j 1 --emit shared-lib --target-cpu apple-m1 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1 ...`.
It reached the compiler and changed nothing.

### The sabotage: the check was made to fail first

In a scratch build one line was added INSIDE `_publish_to_global`, immediately
after the repaired claim, so that a null would indict either the fingerprint
or the reachability of that block:

    self.quesval = self.quesval + Scalar[Self.dtype](1.0)

It moved 270981 bytes of the binary and `__TEXT` differs, which is the control
the two inert pairs are read against. The source was restored from a byte copy
(never `git checkout --`) and `git diff` is empty.

### CPU host column: the repaired lines are not on it

Not run, because it would be a pass that cannot fail, and the source says so.

  - The rf CPU route is `bindings/_mojolearn_rf_host.mojo`, whose header says
    "HOST ONLY. No DeviceContext, no kernel launch, no GPU", and whose fit is
    `ensemble/host/rf_oracle.mojo::rf_host_fit`.
  - `grep -c "Atomic\|compare_exchange\|mutex"` over that closure
    (`_mojolearn_rf_host.mojo`, `ensemble/host/rf_oracle.mojo`,
    `core/forest_host_predict.mojo`, `bindings/forest_export_binding.mojo`,
    `bindings/forest_host_groves_binding.mojo`) returns 0 for every file. The
    same grep on `ensemble/decisiontree/batched_levelalgo/split.mojo` returns
    31, so the grep has teeth.
  - `bindings/_mojolearn_trees_host.mojo` is 0 as well and does not import
    `batched_levelalgo`.
  - It could not have been selected here anyway: `python/mojolearn/_backend.py`
    loads the host set only when `_CPU_ONLY is not None`, "so a box with a GPU
    never serves host arithmetic under a GPU label".

A digest experiment on the host binding was attempted and is NOT reported as
evidence: its contrast arm failed. Three builds to one fixed `-o` gave one
sha256 with and without the define, but the same experiment on the GPU binding
ALSO gave one sha256, and the GPU binding is the arm that had to move. Two
causes were found and both are recorded so the next session does not repeat
them: the Bash tool runs zsh, where an unquoted `$flags` does not word-split,
so the `-D` never reached argv on the first attempt; and after that was fixed
the builds completed in seconds, which are compiler-cache hits. The byte
comparison of `__TEXT` above replaces it and needs no such control.

### kNN: the two fused sites are unreachable from the public surface

Not run, for the same reason, and again from the source.

  - `bindings/_mojolearn.mojo:234` is the only `knn_search(` call site and it
    passes `KNN_METHOD_AUTO`. `python/mojolearn/neighbors.py:487` says the arm
    "is NOT a parameter of this" surface.
  - Under `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL`,
    `neighbors/impl/detail/knn_brute_force.mojo:1482` sets `want_fused = False`
    unconditionally for AUTO (DEVIATION 509), and the launch at `:1560` is
    guarded by `and want_fused`.
  - Even under an explicit `KNN_METHOD_FUSED`, `fused_l2_knn.mojo:563` takes a
    `gdx == 1` early path that never touches the mutex.

So `neighbors/impl/detail/fused_l2_knn.mojo`'s two claim sites cannot be
exercised by any kNN lane in the identical tier on any column, and a kNN
identity comparison would have been a third check that cannot fail. DEVIATION
106's sentence that the kNN producer and consumer "carry the same post-claim
acquire load" is true of the text and says nothing about the shipped path.

### Still owed

  - THE REPAIR ITSELF. It must first be made to survive the optimizer. As
    written it is deleted, so nothing downstream of it can be tested.
  - The MI300X A/B (`tools/rf_nondeterminism/rf_claimfix_ab_body.template.sh`).
    Blocked on Hot Aisle stock; leg-1 under
    `~/mojolearn-evidence/rf-mutex-claim-acquire/leg-1/` records `exit=3`, no
    box created and nothing spent. Its digest guard must be changed to compare
    `__TEXT` and `__DATA` before it is run, or it will pass while running one
    binary twice.
  - `rf-score-weighted` on Apple, once `_mojolearn_metrics.so` is built.
  - A powered run on `rf-reg-gamma-ig/wide`, the shape with the most mutex
    traffic per node, to put a rate on the two Apple movers.
