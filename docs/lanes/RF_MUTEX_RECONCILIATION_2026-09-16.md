# The cross-block mutex ordering defect, reconciled

**2026-09-16. Three branches independently attacked one defect. This file is the single
account Andrew asked for before anything reaches `main`. Written on
`lane/rf-mutex-reconcile`, cut from `main` 453374781. NOTHING IS MERGED AND NO SHIPPED
DEFAULT IS CHANGED BY THIS BRANCH.**

The three branches are `lane/rf-score-weighted-nondeterminism` (the diagnosis),
`lane/rf-mutex-claim-acquire` (the repair, inline), and `fix/amd-merge-ordering` (the
repair as a portable helper, plus a model checker and a device check). They reached the
same repair by three routes and do not contradict each other on any fact. What follows
decides the one shape that ships, separates what is MEASURED from what is MODELED from
what is UNPROVEN, and names what is still owed.

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
is a LOST CANDIDATE, in one direction only, which is exactly what the MI300X traces show.

Mutual exclusion on the lock word is not the missing property. The lock word is fine. What
is missing is the happens-before edge that orders the previous holder's ordinary payload
accesses before the new holder's.

---

## 2. What each branch contributes, and what survives from it

| branch | author | contribution that survives | superseded or withdrawn |
|---|---|---|---|
| `lane/rf-score-weighted-nondeterminism` | Opus | the shipped-0.8.5 reproduction, twelve MI300X legs, the stage trace (leg 11) and the field diff (leg 12), the acquire-CAS A/B replicated twice, `column_has_acquire_rmw` in `ensemble/checks/atomic_matrix.mojo`, the honest leg-14 NULL | its `-D MOJOLEARN_RF_ACQUIRE_CAS=1` spelling is a diagnostic, not the shipped repair, because Apple rejects acquire success ordering on a compare-exchange by name. Its `N_BLKS_FOR_COLS` defines are spent diagnostics and default to the reference 10 |
| `lane/rf-mutex-claim-acquire` | Fable | the repair itself (one post-success acquire load), the memory-model argument written out in DEVIATION 106, `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` as the A/B control, the A/B harness `tools/rf_nondeterminism/rf_claimfix_ab_body.template.sh`, the Apple build that proves the spelling legalizes on Metal | the SHAPE. Six verbatim copies of the claim is how one defect reached three subsystems, and an inline copy cannot be exercised by a runnable check |
| `fix/amd-merge-ordering` | Codex | `core/device_mutex.mojo`, `core/device_mutex_check.mojo`, `tools/check_mutex_handoff_model.py` with a printed counterexample and a negative control, `tools/device_mutex_leg.sh`, and the gfx942 device run | its relaxed spin test (section 4.2). Its `docs/lanes/AMD_MUTEX_ORDERING_INTEGRATION.patch` is superseded by the integration described in section 4 |

No branch found a fact another branch contradicts. The disagreements are about shape and
about how strongly the evidence may be stated, not about what happened.

---

## 3. The repair that lands

**One ACQUIRE load of the mutex word, performed after the claim succeeds. Nothing else
changes.**

After a successful claim, the holder performs an acquire load of the mutex. That load
reads the value the holder's own claim wrote, because a claim is sequenced before the load
and no other participant may modify a held mutex. A compare-exchange is a
read-modify-write, and the release sequence headed by R2 is the maximal contiguous run of
read-modify-writes following R2 in the mutex's modification order, so the holder's own
claim sits inside R2's release sequence. An acquire that reads any value in a release
sequence synchronizes with the release heading it. R2 therefore synchronizes with the
post-claim load, and the previous holder's payload stores happen-before this holder's
payload loads.

**This closes the hole rather than narrowing it.** There is no remaining interleaving in
which a thread holds the lock without having synchronized with the release it claimed
against. That is a statement about the protocol, not about any particular chip.

Why this spelling and not the two alternatives:

- **Acquire success ordering on the compare-exchange** is formally equivalent and is the
  direct translation of the reference's `atomicCAS` plus `__threadfence`. The Apple
  backend rejects it by name. `neighbors/mutex_probe_main.mojo:34` records the exact Mojo
  1.0 message, "Apple GPU does not support `acquire` atomic ordering", alongside the
  refusal of a strong compare-exchange and the NVIDIA-only `threadfence`. Shipping it
  would need a per-vendor `comptime if` and would leave Apple on a second spelling, which
  is the situation that produced this defect. Kept only as the measured diagnostic on
  `lane/rf-score-weighted-nondeterminism`.
- **Making the payload accesses atomic** is on the wrong side of the edge. The previous
  holder's stores are already ordered before its release store. The missing half belongs
  to the acquirer and it is missing for the mutex, not for the slot. Field-wise atomics
  would need seven acquire loads against seven release stores per publish, and `Split`
  carries a Float64 threshold and an Int64 count for which 64-bit atomics are a compile
  error on Apple (`ensemble/checks/atomic_width_probe.mojo`). Heavier, less portable, no
  more correct, and per-field atomics still give no consistent snapshot of the struct.
- **Removing the mutex** by publishing one candidate per (node, sampled column) into
  scratch and reducing in column order is sound, because sampled columns are a permutation
  and `update` over distinct colids is a maximum on a total order. It costs one extra
  launch per round per tree batch, scratch of `max_nodes * n_sampled_cols` splits,
  initialization, separate handling of the purity and metadata words, and a redesign of a
  seam the reference does not have. The Apple column already pays hours to launch
  overhead. This stays the named fallback if the A/B shows the repaired claim still moves.

---

## 4. Which implementation shape ships

**The helper ships. `core/device_mutex.mojo` becomes the one definition of the claim, and
all six claim sites call it.** `lane/rf-mutex-claim-acquire`'s inline edit is the text that
was measured and it is correct, but it is the wrong shape to keep.

### 4.1 Why the helper and not six inline copies

1. **Six copies is the defect's own transmission mechanism.** The census is exact. `git
   grep -n compare_exchange` on `main`, excluding `bench/results`, returns exactly six
   lines. Four production: `ensemble/decisiontree/batched_levelalgo/split.mojo:640`
   (`_publish_to_global`), `extratrees/impl/decisiontree/batched_levelalgo/split.mojo:511`,
   and `neighbors/impl/detail/fused_l2_knn.mojo:623` (the kNN consumer, `-2 -> -1`) and
   `:729` (the kNN producer, `0 -> 1`). Two more in `neighbors/mutex_probe_main.mojo:154`
   and `:183`. All six are the same protocol written out by hand. A bug in the protocol is
   therefore a bug in three subsystems at once, which is precisely what happened.
2. **A runnable check can only test the shipped text if the shipped text is a callable.**
   `core/device_mutex_check.mojo` imports `core.device_mutex`. With the helper it
   exercises the code the forest, ExtraTrees and kNN actually run. With inline copies it
   would exercise a seventh copy that nothing ships, and
   `lane/rf-score-weighted-nondeterminism`'s own method note applies against that, "verify
   a fix in the GENERATED artifact, not the template".
3. **It is the established shape in this repository, not a new idea.**
   `core/device_scan.mojo` with `core/device_scan_check.mojo` and `core/device_zero.mojo`
   with `core/device_zero_check.mojo` are the same primitive-plus-check pair.
4. **The include path already exists at every site.** `bindings/build_rf.sh:123` and
   `bindings/build_trees.sh:134` both build with `-I . -I bindings`, and sibling files in
   all three packages already import from `core` (`core.device_zero` and
   `core.launch_log` in `ensemble/decisiontree/batched_levelalgo/builder.mojo`,
   `core.philox` in the ExtraTrees builder, `core.gemm` and `core.row_norms` in
   `neighbors/impl/detail/knn_brute_force.mojo`). Adding `from core.device_mutex import
   claim_device_mutex` to the three files adds no new build surface.

### 4.2 The one amendment the helper needs before it ships

`core/device_mutex.mojo` as written on `fix/amd-merge-ordering` makes the spin test
**RELAXED**. **Ship it ACQUIRE, unchanged from today's default.**

The relaxed spin is formally sufficient. It is also a second change, and it is not the
change anybody measured. Every arm that has been built and run, on every branch, kept the
acquire spin. Keeping it means the claim sequence the helper emits at the forest site is
the same sequence `lane/rf-mutex-claim-acquire`'s measured text emits, so the A/B result
transfers to the helper instead of having to be repurchased. The spin's acquire is
redundant under the argument in section 3 and removing it is a defensible follow-on with
its own cost story, one fewer cache invalidate per spin iteration on AMD. That is a
performance change, it belongs in its own arm with its own measurement, and it must not
ride in on a correctness fix.

### 4.3 The integration, site by site

| file | claim | shape after |
|---|---|---|
| `ensemble/decisiontree/batched_levelalgo/split.mojo` `_publish_to_global` | `0 -> 1` | `claim_device_mutex(mutex, Int32(0), Int32(1))` |
| `extratrees/impl/decisiontree/batched_levelalgo/split.mojo` block publish | `0 -> 1` | same call, still inside the `sab != SPLIT_SAB_NO_LOCK` guard so the existing sabotage arm stays reachable |
| `neighbors/impl/detail/fused_l2_knn.mojo` consumer | `-2 -> -1` | `claim_device_mutex(mtx, Int32(-2), Int32(-1))`, the `barrier()` after it unchanged |
| `neighbors/impl/detail/fused_l2_knn.mojo` producer | `0 -> 1` | `claim_device_mutex(mtx, Int32(0), Int32(1))`, the `barrier()` after it unchanged |
| `neighbors/mutex_probe_main.mojo` consumer and producer | both | folded into the helper too, so the probe measures the shipped primitive rather than a copy of it |

`fix/amd-merge-ordering`'s unapplied patch already covers the four production sites
correctly and preserves the ExtraTrees sabotage guard. It does not touch
`neighbors/mutex_probe_main.mojo`, and it does not carry
`-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK`.

### 4.4 What the A/B control define becomes

`-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` must survive the refactor and must move into the
helper, where it compiles the pre-repair claim for every caller rather than for the forest
alone. On `lane/rf-mutex-claim-acquire` it guards the ensemble site only, which is correct
for a forest A/B and wrong as a general control. It is never a default and never a matrix
row. Its only job is to give a leg an arm that has to be SEEN to move.

### 4.5 Text that must be carried across, not dropped

DEVIATION 106 in `ensemble/decisiontree/batched_levelalgo/split.mojo` is the authoritative
statement of why this repository's mutex is spelled the way it is. Both RF branches amend
it and their amendments are compatible. The merged text keeps
`lane/rf-mutex-claim-acquire`'s wording, with three corrections that
`fix/amd-merge-ordering` is right about:

- The sentence "an RMW must read the latest value in the coherence order" is imprecise. A
  read-modify-write reads the last value in the modification order immediately before its
  own write. That is the property that puts it in the release sequence, and it is the
  property the repair uses.
- The retained sentence "no ABA hides in the relaxed claim" needs qualification. Reuse of
  the free value across the two reads is exactly the interleaving section 1 describes. It
  is true that only the holder writes the free value, and it is not true that this makes
  the two reads equivalent.
- The XCD and L2 paragraph is a HARDWARE HYPOTHESIS and must say so. It is a plausible
  account of why MI300X is where this fired and nothing in the evidence establishes it.
  The formal hole exists on every column.

---

## 5. Evidence ledger

This is the part that must not blur. The schedule enumeration is a MODEL CHECK. The mutex
lines in the same log file are a DEVICE RUN. They were printed by one script within
seconds of each other and they carry different weight. Neither borrows the other's
authority.

### 5.1 MODELED

`tools/check_mutex_handoff_model.py` enumerates the interleavings of two successful lock
holders and asks whether their payload accesses are ordered by happens-before. Run on
gfx942 at 2026-09-16T12:14:09Z as part of the same leg, and reproducible anywhere because
it is pure Python:

```
stock:    4/6 schedules lack payload ordering
witness:  T0.test -> T1.test -> T0.cas -> T0.payload -> T0.release ->
          T1.cas -> T1.payload -> T1.release
acqcas:   0/6 schedules lack payload ordering
postload: 0/6 schedules lack payload ordering
```

The witness is the abstract form of section 1's scenario. T1 leaves its acquire test
before T0 has released anything, T1's claim is relaxed and consumes T0's release, and
nothing orders T0's payload against T1's.

**It has a negative control and the control was run.** `--protocol stock` exits 1, so the
checker was observed to FAIL on the unfixed side. It is not a check that cannot fail.

**What it is NOT.** It is a small bounded enumerator written by hand by the author of the
repair, not a C++ memory-model checker such as herd7 or CDSChecker. It models exactly two
threads, omits spurious compare-exchange failures, and is deliberately STRICTER than C++
in that every load sees the latest modification, which means it cannot exhibit the stale
load half of the argument at all. It nonetheless exposes the stock protocol, which is the
point. It says nothing about a compiler's lowering and nothing about any chip.

### 5.2 MEASURED, on a device

**gfx942, Hot Aisle MI300X, 2026-09-16T12:14:09Z.** Evidence at
`/Users/andrewhendel/mojolearn-evidence/codex-device-mutex-gfx942-2026-09-16b/box_out/gemm_leg_out/device_mutex.log`.
This is the run whose parent process died in the crash and which was pulled by hand before
the dead-man fired. The log prints `GPU architecture: gfx942` and the sha256 of both
source files it compiled, so the text that ran is identified.

- Sixteen contention launches, eight of `0 -> 1` and eight of `-2 -> -1`, two blocks and
  128 claims each. Every one reports `count 256 handoff-error 0 accepted True`.
- The skip-write sabotage arm reports `count 0 handoff-error 0 accepted False`, so the
  host check REJECTED it.
- `PASS device_mutex: both state pairs, sabotage rejected`.

This is the mechanism evidence `lane/rf-score-weighted-nondeterminism` recorded as MISSING
at leg 14, in the sense that it reaches the primitive on the column where the defect
fires. It settles that the repaired primitive runs correctly under contention on gfx942
and that both state pairs, including kNN's negative-state consumer claim, are exercised.

**What it does NOT establish, and this is important.** `core/device_mutex_check.mojo`
calls `claim_device_mutex` from `core/device_mutex.mojo`, which is the REPAIRED helper.
**There was no stock arm on the device.** So the device run shows the repaired primitive
passing. It does not show the stock primitive failing. The sabotage that was rejected is a
skip-write sabotage, which breaks the COUNT, not the ordering edge. The check has
therefore been shown to fail on one kind of fault and has NOT been shown to fail on the
fault this lane is about. A stock arm is owed (section 6, item 1).

**Apple M4, Metal, under the exclusive Metal lock, one CPU thread.** Recorded on
`fix/amd-merge-ordering`. Sixteen contention launches, all counts 256, no handoff errors,
skip-write sabotage rejected. Also recorded there, `mojo build -j 1 -I .
core/device_mutex_check.mojo` PASS, so the helper's spelling legalizes on Metal.

**Apple M4, the forest binding.** `lane/rf-mutex-claim-acquire` built the REPAIRED
`_mojolearn_rf.so` for `apple-m1` in identical mode on this Mac at one core. Exit 0, no
errors. That proves the post-claim acquire legalizes in the real binding on Metal, where
the acquire compare-exchange does not. It is a BUILD result and not an identity result.
The log ends `built python/mojolearn/identical/_mojolearn_rf.so (gate SKIPPED by
MOJOLEARN_SKIP_BUILD_GATE)`, so no Apple bit has been compared to a reference.

**MI300X, the forest, as an EFFECT.** `lane/rf-score-weighted-nondeterminism` legs 13 and
14, 300 fits per arm, binaries provably distinct by sha256:

| leg | control (stock) | test (acquire-CAS) |
|---|---|---|
| 13 | 7/300 moved | 0/300 |
| 14 | 6/300 moved | 0/300 |

Two independent replications with a control that moved both times. Treating the control
rate as known gives P near 1e-3 each; `fix/amd-merge-ordering` recomputes leg 13 as a
two-sided Fisher exact at approximately 0.01508 and notes that 0 in 300 gives a one-sided
95% upper bound on the failure rate of 0.9936%, not zero. Both readings are right and the
Fisher figure is the one to quote. Note that these legs measured the ACQUIRE-CAS spelling,
which is the diagnostic, not the spelling that ships.

**The defect itself, on the published wheel.** `pip install mojolearn==0.8.5` on one
MI300X, `max_features=1.0` which is the shipped default, 13/300 fits moved and all 13
models distinct, against a clean 0/300 control at `max_features=0.625`. The fit path is
semantically unchanged back through v0.8.0. Users on MI300X are affected today.

**Capability.** `column_has_acquire_rmw` in `ensemble/checks/atomic_matrix.mojo` on
`lane/rf-score-weighted-nondeterminism`. AMD True, MEASURED two ways on gfx942. Apple
False, DOCUMENTED by name and not measured. NVIDIA and everything else UNPROVEN and
conservatively False. That row governs the diagnostic spelling only. The shipped repair
needs no capability row, which is the main reason it is the shipped repair.

### 5.3 UNPROVEN, per column

| column | protocol argument | primitive on device | forest A/B | identity after the repair |
|---|---|---|---|---|
| **AMD gfx942** | holds, and the model check prints the counterexample | **MEASURED**, repaired arm only, no stock arm | **MEASURED as an EFFECT** for the acquire-CAS spelling, twice. The post-claim spelling has **NOT** been run on a device at the forest level | **UNPROVEN**. No rf identity cell has been taken against a reference with the repair in |
| **Apple Metal** | holds | **MEASURED**, repaired arm only, helper check under the Metal lock | not applicable, the defect has never been observed here and has never been shown unable to occur here | **UNPROVEN**. The forest binding BUILDS with the repair, gate skipped. No bit compared |
| **NVIDIA** | holds | **NOT RUN AT ALL.** `core/device_mutex_check.mojo` has never been compiled for an NVIDIA target | never run | **UNPROVEN** |

The protocol argument is column independent. That is its value and also its limit. It says
the stock protocol is invalid everywhere and the repaired one is valid everywhere, and it
says nothing about whether any given backend lowers an acquire load into the instruction
the argument assumes.

---

## 6. What is still OWED before this can be a default

Ordered by how much they change the conclusion.

1. **A stock arm of `core/device_mutex_check.mojo` on gfx942.** The check has been shown
   to fail only against a skip-write sabotage, which is a different fault. Build a second
   binary whose helper carries the pre-repair claim, under
   `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` once the define moves into the helper, and run it
   on the same box. If it reports a nonzero `handoff-error`, the mechanism is demonstrated
   at the primitive, away from forests, and the fix is proven in the strong sense. If it
   stays at zero, that is a NULL and must be reported as one, not as a clearance. Cheap.
   One box, minutes.
2. **The forest A/B at the spelling that ships.** Legs 13 and 14 measured acquire-CAS. The
   post-claim acquire load has never been A/B'd on a device. This is exactly what the
   in-flight `rf-claimfix-ab` leg is for. See section 7.
3. **An rf identity cell with the repair in, on at least one column.** The repair adds an
   ordering constraint and no arithmetic, so no bit should move, and that claim is
   currently an argument rather than a measurement on every column. The Apple build was
   taken with `MOJOLEARN_SKIP_BUILD_GATE`, so nothing was compared. An rf-reg spot check
   on the base fixture against the current reference, under the Metal lock at one core, is
   the cheapest honest version of this and it is owed on Apple.
4. **One NVIDIA build of `core/device_mutex_check.mojo`.** Nothing in any of the three
   branches has compiled the helper for an NVIDIA target. A compile plus a run on any
   NVIDIA box closes a column that is currently blank rather than negative.
5. **ExtraTrees and fused kNN have no measurement at all.** Both carry the identical
   protocol and neither has ever been shown to move or to be stable on AMD. This is the
   scope statement `lane/rf-score-weighted-nondeterminism` made and it is still owed. Note
   the asymmetry `fix/amd-merge-ordering` records. The kNN `-2 -> -1` consumer has a
   single consumer, which restricts the interleavings, so a matching spelling there is not
   independent proof of the same failure. The kNN producer at `:729` has multiple
   producers and the argument applies to it directly.
6. **An ISA-level confirmation that the acquire load survives.** The post-claim load's
   value is discarded. Distinct sha256 digests between arms prove a define reached the
   compiler and changed codegen, which is the right guard and is already enforced by the
   A/B body. They do not prove that an acquire-ordered load landed at the claim. An
   `llvm-objdump` of the AMDGPU code object, looking for the cache invalidate that an
   acquire lowers to after the compare-exchange, would settle it. Both helper and inline
   versions carry a comment telling a future reader not to remove the load, and a comment
   is not a guarantee.
7. **A re-run of the gfx942 device check against the FINAL helper text.** Section 4.2
   changes the spin test back to ACQUIRE, so the text that was measured at
   `sha256 9449c1f2…` is not the text that would ship. The log already prints the source
   sha256, which is the right discipline, and the re-run keeps that discipline honest.
   Relaxing to acquire only adds an ordering constraint and cannot introduce a handoff
   error, so the expected result is another PASS, and an expected result still has to be
   taken.

Until items 2 and 3 land, the repair stays on a branch. Items 1, 4, 5, 6 and 7 are owed
before it would be fair to call the mechanism demonstrated rather than the effect
replicated.

---

## 7. Does the in-flight `rf-claimfix-ab` leg cover it

**Partly. It covers owed item 2 and nothing else.**

The leg is `lane/rf-claimfix-ab`, launched at 2026-09-16T12:12:02Z from
`lane/rf-mutex-claim-acquire` at commit `39b9b8750`, Hot Aisle, AMD, 13core spec, 60
minutes, slot 2. Its metadata is at
`/Users/andrewhendel/mojolearn-evidence/rf-mutex-claim-acquire/leg-1/leg.txt` and its
results will land in that same directory. **It is a live process and must not be killed.**

What it will answer. `tools/rf_nondeterminism/rf_claimfix_ab_body.template.sh` builds two
forest bindings on the box, `stock_prerepair` under
`-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` and `claimfix` at the default, and runs 300
`RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7)` fits per arm on the
wide 16-column fixture. It refuses to run the second arm if the two sha256 digests match,
and it logs `DIGEST COLLISION -- arms are NOT independent, results VOID` if they do. Every
line carries the token `CLAIMFIX-cols16` so a stale R2 object from an earlier leg cannot
be read as this leg's numbers. That is the right construction.

What it will NOT answer.

- It is a control for the FOREST claim site only. `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` on
  `lane/rf-mutex-claim-acquire` guards `_publish_to_global` alone, so the ExtraTrees and
  kNN sites carry the REPAIRED claim in BOTH arms. That is correct for a forest A/B and it
  means the leg says nothing at all about the other two subsystems. Owed item 5 is
  untouched by it.
- It is an EFFECT measurement, the same shape as legs 13 and 14. A 0/300 on the repaired
  arm remains consistent with different codegen perturbing timing enough to hide a race.
  That confound is what owed item 1 exists to remove, and this leg does not remove it.
- It takes no identity cell, so owed item 3 is untouched.
- It measures the INLINE spelling. If the helper ships as section 4 recommends, the result
  transfers only because section 4.2 keeps the acquire spin, which makes the emitted claim
  sequence the same at the forest site. That transfer is an argument, not a measurement.
  A repeat of the repaired arm against the helper build is a fair thing to ask for and
  costs one box at roughly the price the earlier legs paid.

Reading rule for when it lands. **The control must move or the leg proves nothing.** At
the 4.3% to 5.3% rate the earlier legs measured for this fixture, 0/300 on the repaired
arm is strong, 0/100 is not and would need a second leg. Check `so_stock_prerepair.sha256`
against `so_claimfix.sha256` first. Check `runs` reached 300 on both arms and that
`notes` carries no `deadline` entry, because the body sets `done_<arm>` after a deadline
break as well as after a clean finish, and an incomplete arm must be preserved as
incomplete evidence rather than read as a result.

**Status as of 2026-09-16T12:28Z. The leg has NO results yet.** Its rent log at
`/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/67533bb6-a42c-4bd2-91e7-6c31d4f5435b/scratchpad/leg1_rent.log`
shows it has been retrying every 60 seconds since 12:12Z and Hot Aisle has reported
`quantity 0` on 13core every time. It has never acquired a box. Nothing has been measured
by it. The `/Users/andrewhendel/mojolearn-evidence/rf-mutex-claim-acquire/leg-1/` directory
holds only the bundle and body metadata.

One correction to `fix/amd-merge-ordering`'s review of this harness. It objects that
`build_arm` hardcodes `MOJOLEARN_COMPILE_JOBS=4` and asks for 1 to respect the one-core
cap. That objection does not apply. The one-core rule binds this Mac, whose memory is the
constraint that took the machine down. `MOJOLEARN_COMPILE_JOBS=4` there is inside the
REMOTE body, running on a rented 13-core box, where four jobs is a reasonable use of
something Andrew is paying for by the minute. The rest of that review's objections to the
harness stand, in particular that the control's movement is a reading rule rather than an
enforced exit status.

---

## 8. Landing order

Nothing here is merged by this branch. When Andrew approves, this is the order that wastes
the least.

1. Let `rf-claimfix-ab` finish. It is an owed run. Do not cancel it, do not relaunch it,
   and do not start a second Hot Aisle leg while it holds slot 2.
2. Amend `core/device_mutex.mojo` on `fix/amd-merge-ordering` to keep the ACQUIRE spin
   test and to carry `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` as the stock arm for every
   caller.
3. Take owed items 1 and 7 together on one box. Same leg, same log, stock arm and repaired
   arm of `core/device_mutex_check.mojo` on gfx942, source sha256 printed for both.
4. Integrate the six call sites onto the helper. Carry DEVIATION 106's merged text with
   the three corrections in section 4.5.
5. Take owed item 3 on Apple under the Metal lock, at one core, through the slot helper.
6. Take owed item 4, one NVIDIA build, on whatever NVIDIA box is next rented for another
   purpose.
7. Merge. Retire `lane/rf-score-weighted-nondeterminism`'s two diagnostic defines
   (`MOJOLEARN_RF_ACQUIRE_CAS`, `MOJOLEARN_RF_BLKS_COLS8`, `MOJOLEARN_RF_BLKS_COLS16`) or
   keep them with their existing "DIAGNOSTIC ONLY, not a default" headers. They are inert
   when absent, proved by digest rather than asserted, so either is defensible. Keep
   `column_has_acquire_rmw` either way. It is a measured row and it is what tells the next
   reader why the acquire compare-exchange was not chosen.
8. Owed item 5, ExtraTrees and fused kNN, is a separate measurement and should not hold
   the forest repair. Record it as owed against those subsystems.

---

## 9. What each branch should not be read as saying

- `lane/rf-score-weighted-nondeterminism`'s leg 14 probe returned EXACT 512/512 for the
  shipped spelling. That lane correctly called it a NULL rather than a clearance, because
  one acquisition per block almost never makes the spin loop and the hypothesized window
  requires the loop. Nothing in this reconciliation upgrades that null. The model check
  reaches the window the probe could not, and it reaches it as a model.
- `fix/amd-merge-ordering`'s `PASS device_mutex` line is a pass of the REPAIRED primitive.
  It is not an A/B and it is not a demonstration that the stock primitive fails.
- `lane/rf-mutex-claim-acquire`'s Apple build is a build. It proves the spelling legalizes
  on Metal. It compares no bits.
- None of the three has measured ExtraTrees or fused kNN. Everything said about those two
  subsystems here is the protocol argument applied to identical text, which is a reason to
  repair them and not a measurement of them.
- The four traced divergences support a lost-candidate reading. They do not by themselves
  identify a unique cause. A stale read, a competing write, an omitted candidate upstream
  or a compiler defect could each produce the same symptom. What makes the lost-candidate
  reading strong is structural rather than statistical. A correct merge is a maximum over
  a total order and does not depend on arrival order, so any fit whose merged split
  differs from another's over bit-identical histograms and an identical column sample has
  dropped a candidate or read a partial state. An invalid lock is worth repairing on the
  protocol argument alone, whatever the traces turn out to mean.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
