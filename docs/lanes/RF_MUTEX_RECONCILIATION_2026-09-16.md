# The cross-block mutex ordering defect, reconciled

**2026-09-16. Three branches independently attacked one defect. This file is the single
account Andrew asked for before anything reaches `main`. Written on
`lane/rf-mutex-reconcile`, cut from `main` 453374781. NOTHING IS MERGED AND NO SHIPPED
DEFAULT IS CHANGED BY THIS BRANCH.**

The three branches are `lane/rf-score-weighted-nondeterminism` (the diagnosis),
`lane/rf-mutex-claim-acquire` (the repair, inline), and `fix/amd-merge-ordering` (the
repair as a portable helper, plus a model checker and a device check).

**REVISED LATER THE SAME DAY. The first version of this file recommended a repair that
DOES NOT EMIT. The reasoning survives intact. The spelling does not, and neither does the
A/B evidence that was thought to support it. Section 0 is the correction and it should be
read before anything else.**

---

## 0. THE CORRECTION. A discarded acquire load is not a fence

All three branches converged on the same repair, an ACQUIRE load of the mutex word
performed after the claim succeeds and whose value is discarded. That repair was carried
inline on `lane/rf-mutex-claim-acquire` at four production sites and two probe sites, and
as `core/device_mutex.mojo:45` on `fix/amd-merge-ordering` under a comment instructing a
future reader not to remove the unused value.

**Measured 2026-09-16. `_ = Atomic.load[ordering = Ordering.ACQUIRE](mutex)` produces ZERO
instructions on Metal AIR, on PTX and on GCN alike.** The fixed and stock arms of
`_mojolearn_rf.so`, and a second pair of `_mojolearn_trees.so` differing only by the
deleted line itself, had bit-identical `__TEXT` and `__DATA`. The compiler drops the load
because nothing consumes it. Every acquire load in this repository that DOES emit uses its
value, in the shape `if Atomic.load[...](mtx) != 0` or `var w = Atomic.load[...]`. The
comment telling a reader not to remove the unused value describes exactly what the
compiler does anyway.

This was owed item 6 of the first version of this file, the ISA-level confirmation that
the acquire load survives. **It came back NEGATIVE.** The first version was right to owe
it and wrong to recommend landing before it was paid.

### 0.1 The spelling that does emit

**`fence[ordering = Ordering.ACQUIRE]()` from `std.atomic`.** One source, no vendor
divergence, no capability row. Measured the same day:

- The rf binding's sections differ with it. `__TEXT,__const` grew 16 bytes of AIR.
- The disassembled metallib shows one added `fence acquire` sitting between
  `air.atomic.global.cmpxchg.weak.i32` and the plain loads of `split[node]`, which is
  exactly where the argument in section 1 places it.
- gfx942 shows `buffer_inv sc0 sc1` at the claim where the stock build has nothing.

**`std.atomic.fence` is a DIFFERENT SYMBOL from `std.gpu.intrinsics.threadfence`.**
DEVIATION 106 reasoned from `threadfence` being comptime-asserted NVIDIA-only and
concluded that no fence was available on this path at all. That conclusion is wrong, it
has stood in the file since the mutex was first written, and it is the reason three
separate agents went looking for an ordering they could hang off an existing atomic
operation instead of simply fencing. Correcting it is the most useful single line of this
reconciliation.

A second correction to the same deviation. Apple rejects `acquire` on **every**
read-modify-write, not only on a compare-exchange.

### 0.2 What this does to the evidence

**The replicated A/B effect is no longer explained by the repair. Say it plainly.**

`lane/rf-score-weighted-nondeterminism` legs 13 and 14 reported a control moving 7/300 and
then 6/300 against a repaired arm of 0/300 twice. That contrast cannot be the repair
working, because the repair emitted nothing. What remains is the alternative that lane
explicitly refused to rule out and wrote down in its own status file, that a different
codegen perturbs timing enough to hide a 2% to 5% race. The effect is real and it is
**UNEXPLAINED**. It is not evidence for the ordering repair and it must not be quoted as
if it were.

One part of that is still open and section 3.2 says how it gets settled. Legs 13 and 14
compiled the ACQUIRE-CAS spelling, `Atomic.compare_exchange[success_ordering =
Ordering.ACQUIRE, ...]`, which is a different spelling from the discarded load and which
has its own reason to emit. Whether it did is a question the same section comparison
answers, and it is now owed and cheap.

### 0.3 A second defect, in the A/B harness

`tools/rf_nondeterminism/rf_claimfix_ab_body.template.sh` guards arm independence with a
whole-file `sha256 A != sha256 B`, at `:156` for the refusal and `:165` for the VOID
branch. `bindings/build_rf.sh:113` bakes a fresh `mktemp -d` path into the output's install
name. Two builds of BYTE-IDENTICAL source therefore always differ, in about 54 bytes of
mktemp install-name suffix, `LC_UUID` and code-signature slot.

**That guard can never fire its VOID branch.** It is a verification that cannot fail. A leg
run under it would have compared one program with itself and reported the result as two
arms, and the log line claiming the digests differ would have been true and meaningless.

Every "the arms are provably distinct binaries" and "digests provably distinct" claim in
this lane family rests on that guard and is **unestablished** until redone. That includes
legs 13 and 14.

The fix is to compare SECTIONS and not segments. `__TEXT,__text`, `__TEXT,__const` and the
`__DATA` sections. The `__TEXT` SEGMENT starts at file offset 0 and carries the mktemp
install name, so a segment-level digest reports a false DIFFER and is the same trap one
level up.

**Andrew's rule, set here. Run the byte compare BEFORE spending a box, never after.**

---

## 1. The defect, stated once

Every cross-block mutex in this repository spins on an ACQUIRE load until the mutex reads
free, then takes the lock with a WEAK RELAXED compare-exchange, and hands it back with a
RELEASE store. Those are two separate reads of the mutex word and only the first carries
an acquire.

An acquire operation synchronizes with the release whose value it reads, or with a value
in that release's release sequence. The spin load is an acquire, so it synchronizes with
whichever release wrote the free value it observed. The compare-exchange is relaxed, so it
synchronizes with nothing at all, whatever value it reads. The two reads need not observe
the same release. With holders A and B and contender C on one node's mutex:

1. A releases, a store-release of the free value. Call it R1.
2. C's acquire spin load reads R1. C now synchronizes with A and with nobody later.
3. B claims on R1 first. C's claim on that value fails, or C is simply late.
4. B loads `split[node]`, merges its candidate, stores `split[node]`, releases. Call it R2.
5. C's relaxed claim consumes R2 and succeeds.

C now holds the lock having performed no acquire that reads R2. B's plain store to
`split[node]` is sequenced before R2, but R2 happens-before nothing that C does, so B's
store and C's plain load of the same slot are unordered. C merges its own candidate into a
stale copy and writes the whole struct back. **B's candidate is erased.** The failure mode
is a LOST CANDIDATE, in one direction only, which is what the MI300X traces show.

Mutual exclusion on the lock word is not the missing property. The lock word is fine. What
is missing is the happens-before edge that orders the previous holder's ordinary payload
accesses before the new holder's.

**This reasoning is unaffected by section 0.** The protocol is invalid for the reason
given. What section 0 changes is which instruction repairs it.

---

## 2. What each branch contributes, and what survives from it

| branch | author | contribution that survives | superseded or withdrawn |
|---|---|---|---|
| `lane/rf-score-weighted-nondeterminism` | Opus | the shipped-0.8.5 reproduction, twelve MI300X legs, the stage trace (leg 11) and the field diff (leg 12), `column_has_acquire_rmw` in `ensemble/checks/atomic_matrix.mojo`, and the honest leg-14 NULL | its legs 13 and 14 A/B, whose arms rest on the guard of section 0.3 and whose effect is unexplained per section 0.2. Its `-D MOJOLEARN_RF_ACQUIRE_CAS=1` is a diagnostic. Its `N_BLKS_FOR_COLS` defines are spent and default to the reference 10 |
| `lane/rf-mutex-claim-acquire` | Fable | the memory-model argument written out in DEVIATION 106, `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` as the control define, the A/B harness structure, the site census | the REPAIR, which does not emit (section 0). The SHAPE, six verbatim inline copies. The harness's distinctness guard (section 0.3) |
| `fix/amd-merge-ordering` | Codex | `core/device_mutex.mojo` as a SHAPE, `core/device_mutex_check.mojo`, `tools/check_mutex_handoff_model.py` with a printed counterexample and a negative control, `tools/device_mutex_leg.sh`, the gfx942 device run, and the correct scepticism about what the traces establish | the same non-emitting repair, including its comment at `:45` instructing a reader not to remove the unused value. Its relaxed spin test. Its unapplied integration patch |

No branch found a fact another branch contradicts. All three were defeated by the same
compiler behavior, which none of them measured until it was owed and paid.

---

## 3. The repair that lands

**One `fence[ordering = Ordering.ACQUIRE]()` from `std.atomic`, executed after the claim
succeeds, before any payload access. Nothing else changes.**

The fence orders this thread's subsequent plain loads after every release that this
thread's preceding atomic operations observed, including the claim's. The claim is a
read-modify-write, so its write sits inside the release sequence headed by whichever
release it consumed. An acquire fence placed after an atomic operation that read a value
in a release sequence gives the same synchronizes-with edge that an acquire on the
operation itself would. The previous holder's payload stores therefore happen-before this
holder's payload loads.

**This closes the hole rather than narrowing it,** and unlike the discarded load it is an
instruction that survives to the ISA on all three targets, shown by the three measurements
in section 0.1.

Why this spelling and not the alternatives:

- **Acquire success ordering on the compare-exchange** is formally equivalent. Apple
  rejects `acquire` on every read-modify-write, so shipping it needs a per-vendor
  `comptime if` and leaves Apple on a second spelling, which is the situation that
  produced this defect. Kept on `lane/rf-score-weighted-nondeterminism` as a diagnostic
  and, per section 3.2, as the arm that settles what legs 13 and 14 actually compared.
- **`std.gpu.intrinsics.threadfence`** is comptime-asserted NVIDIA-only. This is the
  symbol DEVIATION 106 reasoned from, and confusing it with `std.atomic.fence` is what
  made a fence look unavailable (section 0.1).
- **Making the payload accesses atomic** is on the wrong side of the edge. The previous
  holder's stores are already ordered before its release store. The missing half belongs
  to the acquirer and it is missing for the mutex, not for the slot. Field-wise atomics
  would need seven acquire loads against seven release stores per publish, `Split` carries
  a Float64 threshold and an Int64 count for which 64-bit atomics are a compile error on
  Apple (`ensemble/checks/atomic_width_probe.mojo`), and per-field atomics still give no
  consistent snapshot of the struct.
- **Removing the mutex** by publishing one candidate per (node, sampled column) into
  scratch and reducing in column order is sound, because sampled columns are a permutation
  and `update` over distinct colids is a maximum on a total order. It costs one extra
  launch per round per tree batch, scratch of `max_nodes * n_sampled_cols` splits,
  initialization, separate handling of the purity and metadata words, and a redesign of a
  seam the reference does not have. The Apple column already pays hours to launch
  overhead. This stays the named fallback if a real A/B shows the repaired claim still
  moves.

### 3.1 The shape

**Codex's helper ships. `core/device_mutex.mojo` becomes the one definition of the claim,
and all six claim sites call it. The spin test stays ACQUIRE.**

1. **Six copies is the defect's own transmission mechanism.** The census is exact. `git
   grep -n compare_exchange` on `main`, excluding `bench/results`, returns exactly six
   lines. Four production: `ensemble/decisiontree/batched_levelalgo/split.mojo:640`
   (`_publish_to_global`), `extratrees/impl/decisiontree/batched_levelalgo/split.mojo:511`,
   and `neighbors/impl/detail/fused_l2_knn.mojo:623` (the kNN consumer, `-2 -> -1`) and
   `:729` (the kNN producer, `0 -> 1`). Two more in `neighbors/mutex_probe_main.mojo:154`
   and `:183`. All six are the same protocol written out by hand, so a bug in the protocol
   is a bug in three subsystems at once, which is what happened.
2. **A runnable check can only test the shipped text if the shipped text is a callable.**
   `core/device_mutex_check.mojo` imports `core.device_mutex`. With the helper it
   exercises the code the forest, ExtraTrees and kNN actually run. With inline copies it
   exercises a seventh copy that nothing ships, and
   `lane/rf-score-weighted-nondeterminism`'s own method note applies against that, "verify
   a fix in the GENERATED artifact, not the template".
3. **It is the established shape here.** `core/device_scan.mojo` with
   `core/device_scan_check.mojo` and `core/device_zero.mojo` with
   `core/device_zero_check.mojo` are the same primitive-plus-check pair.
4. **The include path already exists at every site.** `bindings/build_rf.sh:123` and
   `bindings/build_trees.sh:134` build with `-I . -I bindings`, and sibling files in all
   three packages already import from `core`.
5. **The spin test stays ACQUIRE**, against Codex's relaxed version. Relaxing it is a
   performance change, one fewer cache invalidate per spin iteration on AMD, and it must
   not ride in on a correctness fix.

`-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` moves into the helper, where it compiles the
pre-repair claim for every caller rather than for the forest alone. It is never a default
and never a matrix row. Its only job is to give a leg an arm that has to be SEEN to move.

`-D MOJOLEARN_MUTEX_SECTION_SABOTAGE=1` is added to the helper as the positive control for
the section comparator of section 0.3. It changes the held value the claim writes, which
must move `__TEXT,__text`. It is build-only and is never run.

### 3.2 Integration, site by site

| file | claim | shape after |
|---|---|---|
| `ensemble/decisiontree/batched_levelalgo/split.mojo` `_publish_to_global` | `0 -> 1` | `claim_device_mutex(mutex, Int32(0), Int32(1))` |
| `extratrees/impl/decisiontree/batched_levelalgo/split.mojo` block publish | `0 -> 1` | same call, still inside the `sab != SPLIT_SAB_NO_LOCK` guard so the existing sabotage arm stays reachable |
| `neighbors/impl/detail/fused_l2_knn.mojo` consumer | `-2 -> -1` | `claim_device_mutex(mtx, Int32(-2), Int32(-1))`, the `barrier()` after it unchanged |
| `neighbors/impl/detail/fused_l2_knn.mojo` producer | `0 -> 1` | `claim_device_mutex(mtx, Int32(0), Int32(1))`, the `barrier()` after it unchanged |
| `neighbors/mutex_probe_main.mojo` consumer and producer | both | folded into the helper too, so the probe measures the shipped primitive rather than a copy of it |

DEVIATION 106 carries forward with four corrections. The `threadfence` conclusion is wrong
(section 0.1). Apple rejects acquire on every RMW, not only on compare-exchange. The
sentence about a read-modify-write reading the latest value in the coherence order is
imprecise, since an RMW reads the last value in the modification order immediately before
its own write, which is the property that puts it in the release sequence. The XCD and L2
paragraph is a HARDWARE HYPOTHESIS and must say so, because the formal hole exists on
every column.

---

## 4. Evidence ledger

The schedule enumeration is a MODEL CHECK. The mutex lines in the same log are a DEVICE
RUN. They were printed by one script within seconds of each other and they carry different
weight. Neither borrows the other's authority.

### 4.1 MODELED

`tools/check_mutex_handoff_model.py` enumerates the interleavings of two successful lock
holders and asks whether their payload accesses are ordered by happens-before. Run on
gfx942 at 2026-09-16T12:14:09Z, and reproducible anywhere because it is pure Python:

```
stock:    4/6 schedules lack payload ordering
witness:  T0.test -> T1.test -> T0.cas -> T0.payload -> T0.release ->
          T1.cas -> T1.payload -> T1.release
acqcas:   0/6 schedules lack payload ordering
postload: 0/6 schedules lack payload ordering
```

The witness is the abstract form of section 1's scenario. **It has a negative control and
the control was run.** `--protocol stock` exits 1, so the checker was observed to FAIL on
the unfixed side.

**What it is NOT.** A small bounded enumerator written by hand by the author of the repair,
not a C++ memory-model checker such as herd7 or CDSChecker. Two threads, no spurious
compare-exchange failures, and deliberately STRICTER than C++ in that every load sees the
latest modification, which means it cannot exhibit the stale-load half of the argument at
all. It nonetheless exposes the stock protocol, which is the point.

**And note what section 0 does to it.** Its `postload` protocol models an acquire
operation that the compiler deletes. The model was checking a program that does not exist.
A model check cannot catch that, and this is the sharpest available illustration of why a
model check never substitutes for looking at the generated artifact.

### 4.2 MEASURED, on a device

**gfx942, Hot Aisle MI300X, 2026-09-16T12:14:09Z.** Evidence at
`/Users/andrewhendel/mojolearn-evidence/codex-device-mutex-gfx942-2026-09-16b/box_out/gemm_leg_out/device_mutex.log`.
Pulled by hand after the crash killed its parent. The log prints `GPU architecture: gfx942`
and the sha256 of both source files, so the text that ran is identified.

- Sixteen contention launches, eight of `0 -> 1` and eight of `-2 -> -1`, two blocks and
  128 claims each. Every one reports `count 256 handoff-error 0 accepted True`.
- The skip-write sabotage reports `count 0 handoff-error 0 accepted False`, REJECTED.
- `PASS device_mutex: both state pairs, sabotage rejected`.

**What it does NOT establish.** There was no stock arm. It shows the repaired primitive
passing, not the stock primitive failing. Its sabotage breaks the COUNT, not the ordering
edge. And per section 0 the primitive it passed contained no ordering instruction at the
claim, so what it actually demonstrated is that the check's contention pattern does not
expose the hole, on a build where the hole was fully present. **That is a NULL, and a
useful one.** It sets the bar for the stock arm of section 6.

**Apple M4, Metal, under the exclusive Metal lock, one CPU thread.** Recorded on
`fix/amd-merge-ordering`. Sixteen launches, all counts 256, no handoff errors, skip-write
rejected. Same reading as above, same null.

**Apple M4, the forest binding.** `lane/rf-mutex-claim-acquire` built the repaired
`_mojolearn_rf.so` for `apple-m1` in identical mode on this Mac at one core, exit 0. The
log ends `built python/mojolearn/identical/_mojolearn_rf.so (gate SKIPPED by
MOJOLEARN_SKIP_BUILD_GATE)`, so no Apple bit was compared to a reference. Per section 0
that build is byte-equal in `__TEXT` and `__DATA` to its own stock arm.

**MI300X, the forest, as an EFFECT.** Legs 13 and 14, 300 fits per arm, control moving
7/300 then 6/300 against 0/300 twice. **This effect is real and UNEXPLAINED (section 0.2),
and its arm-independence claim is unestablished (section 0.3).** It is not evidence for
the ordering repair.

**The defect itself, on the published wheel.** `pip install mojolearn==0.8.5` on one
MI300X, `max_features=1.0` which is the shipped default, 13/300 fits moved and all 13
models distinct, against a clean 0/300 control at `max_features=0.625`. The fit path is
semantically unchanged back through v0.8.0. **This stands. Users on MI300X are affected
today, and nothing in section 0 touches it.**

**Capability.** `column_has_acquire_rmw` in `ensemble/checks/atomic_matrix.mojo` on
`lane/rf-score-weighted-nondeterminism`. AMD True measured two ways on gfx942, Apple False
documented by name, NVIDIA and the rest UNPROVEN and conservatively False. That row
governs the diagnostic spelling only. **The shipped repair needs no capability row at all,
which is now the second reason it is the shipped repair.**

### 4.3 UNPROVEN, per column

| column | protocol argument | fence EMITS | primitive on device | forest A/B | identity after the repair |
|---|---|---|---|---|---|
| **AMD gfx942** | holds, counterexample printed | **MEASURED**, `buffer_inv sc0 sc1` at the claim | repaired-but-non-emitting arm only, a NULL | **NONE that survives section 0** | **UNPROVEN** |
| **Apple Metal** | holds | **MEASURED**, `fence acquire` in the disassembled metallib, `__TEXT,__const` +16 bytes | same NULL | not applicable | **UNPROVEN** |
| **NVIDIA** | holds | **MEASURED** as PTX emission in the same pass | **NEVER RUN.** `core/device_mutex_check.mojo` has never been compiled for an NVIDIA target | never run | **UNPROVEN** |

The protocol argument is column independent. That is its value and also its limit.

---

## 5. What is still OWED

1. **A section-level byte comparison of the arms, before any box is rented.** Section 0.3.
   Stock against fence, with a sabotage arm that must move the sections so the comparator
   is seen able to differ. This is the gate on everything below it.
2. **A stock arm of `core/device_mutex_check.mojo` on gfx942.** The top owed item and the
   one that converts effect into mechanism. The existing gfx942 run passes only the
   repaired primitive, and its skip-write sabotage breaks the count rather than the
   ordering edge. **Andrew has approved a rented AMD box for this.**
3. **Whether the acquire-CAS spelling of legs 13 and 14 emits.** Settled by the same
   section comparison as item 1. If it does not, those legs compared one program with
   itself and the 0/300 arms are void. If it does, their arms were genuinely distinct and
   the effect is a real contrast between two real programs that still is not the ordering
   repair.
4. **A forest A/B at the spelling that ships,** with the fixed harness guard.
5. **An rf identity cell with the repair in.** The fence adds an ordering constraint and
   no arithmetic, so no bit should move, and that is currently an argument rather than a
   measurement. The Apple build was taken with the build gate skipped and compared
   nothing.
6. **One NVIDIA build and run of `core/device_mutex_check.mojo`.**
7. **ExtraTrees and fused kNN have no measurement at all.** Both carry the identical
   protocol. Note the asymmetry `fix/amd-merge-ordering` records. The kNN `-2 -> -1`
   consumer has a single consumer, which restricts the interleavings, so a matching
   spelling there is not independent proof of the same failure. The kNN producer at `:729`
   has multiple producers and the argument applies to it directly.
8. **An emission audit of every other discarded atomic in the repository.** Section 0 is a
   general fact about this compiler, not a fact about mutexes, and nothing has checked
   whether the same mistake is spelled anywhere else.

---

## 6. What each branch should not be read as saying

- `lane/rf-score-weighted-nondeterminism`'s leg 14 probe returned EXACT 512/512 for the
  shipped spelling and that lane correctly called it a NULL, because one acquisition per
  block almost never makes the spin loop. Nothing here upgrades that null.
- `fix/amd-merge-ordering`'s `PASS device_mutex` is a pass of a primitive that, per
  section 0, carried no ordering instruction at the claim. It is not an A/B.
- `lane/rf-mutex-claim-acquire`'s Apple build proves the SOURCE compiles. It does not prove
  the ordering reached the binary, and section 0 shows it did not.
- None of the three has measured ExtraTrees or fused kNN.
- The four traced divergences support a lost-candidate reading. They do not by themselves
  identify a unique cause. What makes that reading strong is structural rather than
  statistical. A correct merge is a maximum over a total order and does not depend on
  arrival order, so any fit whose merged split differs from another's over bit-identical
  histograms and an identical column sample has dropped a candidate or read a partial
  state. An invalid lock is worth repairing on the protocol argument alone, whatever the
  traces turn out to mean, and that argument is what survives today intact.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
