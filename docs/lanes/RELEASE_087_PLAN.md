# 0.8.7 release plan, opened 2026-09-16

**Nothing has been published and nothing here publishes anything.** Publishing
needs Andrew's separate explicit "ship". No PyPI upload, no release tag, no
`release-provenance.yml` dispatch.

Branch `release/0.8.7`, cut from `origin/main` at **0d313ad11**.

---

## 1. What 0.8.7 is, and what happened to 0.8.6

### 1.1 0.8.6 was folded in and NEVER PUBLISHED

Andrew's decision, 2026-09-16, after the audit in
`docs/lanes/RELEASE_086_STATE_2026-09-16.md` on `release/0.8.6`. **0.8.6 is
abandoned as a release. Its number is skipped. Nothing with 0.8.6 on it was
ever distributed.**

The reason, in one line. The 0.8.6 wheels ship a `verify --all` whose
`verdict()` returns VERIFIED as soon as one part is IDENTICAL, before it looks
at REFUSED, and that file is on the native build inventory, so fixing it meant
rebuilding all three Linux sets and the macOS wheel and re-recording. Cutting
fresh from main costs the same builds and buys 149 commits of work as well.

**The fold is clean at the version level, which is checked, not assumed.**
`python/mojolearn/_version.py` and `python/pyproject.toml` on this base both
read `0.8.5`. The 0.8.6 version bump exists only on commit 274a9d161 on the
folded branch and was never merged. **No commit on main has ever declared
0.8.6.** The published PyPI version remains 0.8.5.

### 1.2 The abandoned artifacts, named, so nobody mistakes them for a release

These exist on disk and in R2 and are **not** 0.8.7 inputs. They are evidence
of work, not candidates to ship.

| artifact | where | what it is |
|---|---|---|
| `mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl` | `~/mojolearn-evidence/release-0.8.6/linux-wheel/dist/final/`, sha256 `7cab1aa3cfcd...` | packed, audited, stripped, never published |
| the same wheel | R2, key `releases/0.8.6/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl` | uploaded so record legs could install it |
| `mojolearn-0.8.6-py3-none-macosx_11_0_arm64.whl` | `~/mojolearn-evidence/release-0.8.6/macos-wheel/`, sha256 `eba69f83c9a9...` | built, verified on five interpreters, never published |
| three Linux build sets at db9047b9f | `~/mojolearn-evidence/release-0.8.6/linux-builds/` | sm_89, sm_90a, gfx942, complete with proofs |
| four identity columns | `~/mojolearn-evidence/release-0.8.6/records/` and `apple-record/` | see section 3 |

**Do not delete any of it without asking.** The columns are the only recording
of the `rf-score-weighted` finding, and the R2 object is the only copy of a
wheel that cost three GPU rentals. Deleting the R2 key is a separate decision
for Andrew; this plan does not touch it.

### 1.3 Six user-facing documents claim a 0.8.6 that will not exist

Measured on this base, `git grep -c "0\.8\.6"` over tracked files. Seventeen
files mention it. Ten are internal comments recording when a change landed in
the TREE (`packaging/check_ext_lists.py`, `packaging/linux/build_sets.sh`,
`packaging/linux/pack_wheel.py`, `packaging/linux/test_release_061_inventory.py`,
`packaging/macos/build_release_wheel.sh`,
`tools/check_linux_release_qualification.py`,
`tools/test_native_source_rule.py`,
`tools/test_optional_byte_lm_build_layout.py`,
`tools/test_release061_end_to_end.py`), plus
`tools/identity_break.py:803`, which correctly cites the 0.8.6 AMD gap as
history and should be left alone.

**Seven say something a reader would take as a version promise, and each is
now false.**

| file and line | what it says |
|---|---|
| `CHANGELOG.md:6` | heading `## Unreleased (0.8.6 prep; the freeze commit names it 0.8.6 with the version bump)` |
| `README.md:273` | "From 0.8.6 every one of these host bindings ships in both" |
| `SUPPORT_MATRIX.md:57` | "The published 0.8.5 wheels carry the byte LM host binding alone and 0.8.6 sixteen of the thirty-two" |
| `docs/VERIFY_EXTERNALLY.md:83` | "From 0.8.6" |
| `docs/BYTE_LM_CPU_TRAINING.md:366` | "0.8.6 sixteen of the thirty-two" |
| `docs/RELEASE_CHECKLIST.md:64` | "Since 0.8.6 the wheel also carries every host binding" |
| `tools/verify_external.sh:40` | "From 0.8.6 the wheel" |

Two of them are wrong twice over. `SUPPORT_MATRIX.md:57` and
`docs/BYTE_LM_CPU_TRAINING.md:366` say **sixteen** families, which
`lane/ship-cpu-host-families` already superseded with thirty-two before the
fold was decided.

All seven move to 0.8.7 as part of the version bump, and
`tools/docs_facts.py --check` is what holds them to it.

### 1.4 What 0.8.7 inherits

Everything 0.8.6 was going to be, plus the 149 commits that made folding
cheaper than fixing.

* **Every CPU host binding ships.** `0d313ad11`. Thirty-two families, measured
  on this base as `ships_in_wheel=True` 32 and `False` 1. A CPU-only
  `verify --all` checks **122 lanes against 39**, recorded in
  `docs/VERIFY.md:188` and `docs/lanes/LANE_STATUS_lane-ship-cpu-host-families.md`.
* **The verifier verdict fix**, `87085a5eb`. Confirmed on this base by reading
  `python/mojolearn/_verify_all.py`: `REFUSED` is tested before `IDENTICAL`
  and returns `INCOMPLETE`. The defect that killed 0.8.6 does not exist here.
* **The lane selector and the three tiers**, `26916b0cb`, with five defects
  fixed.
* **The tokenizer trainer determinism close-out**, `14aa85848`. The HF pin
  recommendation is withdrawn, the owned BPE trainer stays, and the stale
  `test_tokenizer_manifest.py` assertion is fixed.
* The release-record scope in code, the fixture shrink, the admission-rule
  fix, `stale_reference_lanes`, one host-binding resolution path, the sabotage
  arms that could not do their job, and the capability merged since the 0.8.6
  freeze (ARIMA exogenous regressors, the GP optimizer,
  `SpectralClustering.predict`, Holt-Winters CPU inference, GBDT CTR tables,
  YetiRank, the BPE vocabulary trainer).

### 1.5 Do not write a lane total into this file

The registry is the answer and it moves. `tools/lane_select.py --count` read
**211** on this base on 2026-09-16. Any number in this document is a dated
measurement with the command beside it, never a constant to be trusted later.

---

## 2. Release machinery that must not be lost with the folded branch

Three pieces of release machinery live ONLY on `release/0.8.6`. They are real
code with real tests and they are the reason 0.8.6 got as far as it did. If
0.8.7 is cut from main and the folded branch is forgotten, all three are gone
and 0.8.7 rediscovers them.

### 2.1 Verified absent from this base, with controls

Repo-wide `git grep -l` on `origin/main` (0d313ad11), each with a control that
prints its matches on `release/0.8.6`. A grep returning zero on both refs is
not evidence; every row below has a non-empty control.

| token | 0.8.7 base | control on `release/0.8.6` |
|---|---|---|
| `POST_RECORD_FILES` | ABSENT | `packaging/linux/pack_wheel.py`, `tools/verify_linux_surface_qualification.py`, `tools/test_post_record_allowlist.py` |
| `POST_RECORD_NAMES` | ABSENT | `packaging/macos/repack_post_record.py`, `tools/verify_linux_surface_qualification.py`, `tools/test_post_record_allowlist.py` |
| `post_record_differences` | ABSENT | `packaging/linux/pack_wheel.py`, `tools/check_linux_release_qualification.py`, `tools/test_post_record_allowlist.py` |
| `post_record_equivalent` | ABSENT | `packaging/macos/repack_post_record.py`, `tools/verify_linux_surface_qualification.py`, `tools/test_post_record_allowlist.py` |
| `NOTICE_FORBIDDEN` | ABSENT | `tools/release_wheel_content_audit.py` |
| `repack_post_record` | ABSENT | `packaging/macos/repack_post_record.py`, `packaging/macos/test_repack_post_record.py` |
| `release_wheel_content_audit` | ABSENT | `tools/release_wheel_content_audit.py`, `tools/test_release_wheel_content_audit.py` |

And by path, with `git ls-tree`, which does not depend on any phrase:

| file | 0.8.7 base | `release/0.8.6` |
|---|---|---|
| `tools/release_wheel_content_audit.py` | ABSENT | PRESENT |
| `tools/test_release_wheel_content_audit.py` | ABSENT | PRESENT |
| `packaging/macos/repack_post_record.py` | ABSENT | PRESENT |
| `packaging/macos/test_repack_post_record.py` | ABSENT | PRESENT |
| `tools/test_post_record_allowlist.py` | ABSENT | PRESENT |

### 2.2 What each one does, and why 0.8.7 needs it more than 0.8.6 did

**The post-record allowlist** (`POST_RECORD_FILES`,
`post_record_differences`, `post_record_equivalent`). A release is built,
installed and recorded, and only then does the record land in the tree. The
shipped binaries must be the recorded ones, so the final wheel is packed from
the recorded build proofs and exactly one inventoried file may differ from
them, `python/mojolearn/host_surface.py`, and only in the top-level
assignments `TRAINING_GPU_COLUMNS`, `TRAINING_FIX_COLUMNS` and
`TRAINING_FIX_LANES`. `post_record_equivalent` proves it by AST comparison
with those assignments removed, so a comment cannot smuggle a change through.
Without this the packer either refuses the record commit outright or admits
anything.

**The wheel content audit** (`tools/release_wheel_content_audit.py`,
`NOTICE_FORBIDDEN`). Checks the built wheel's own bytes: NOTICE byte-equal to
the branch NOTICE and free of "used under license", no GPT-2 table, no
fixture or vocabulary, no vendored environment, and `host_surface.py` imports
and declares its bindings. Andrew's "no third-party data in distribution" rule
has no other enforcement.

**The macOS post-record repack** (`packaging/macos/repack_post_record.py`).
Replaces only `host_surface.py`, `verify_reference/table.json` and the
identity column JSONs inside the finished macOS wheel, so the record can land
without rebuilding the Mac wheel. On 0.8.7 that is worth more than it was on
0.8.6, because the macOS wheel now carries thirty-two host families and is
correspondingly more expensive to rebuild.

None of the three hard-codes a family count. `pack_wheel.py` takes
`HOST_NAMES` from `--wheel-bindings`, so all three follow `host_surface.py`
and pick up thirty-two families with no edit. Checked by grepping each file
for a literal `15`, a `wheel_families` constant and a `len(...bindings)`; the
only hit is an unrelated `set_fast_count=15` comment in `pack_wheel.py:90`.

### 2.3 The merge-back plan, with the conflict measured

Three commits carry all of it. Pick them onto `release/0.8.7` in this order.

| order | commit | files | expected |
|---|---|---|---|
| 1 | `6cb6fa571` the post-record allowlist and host binaries against their own proof | `packaging/linux/pack_wheel.py`, `packaging/linux/test_release_061_inventory.py`, `tools/verify_linux_surface_qualification.py` | base UNMOVED on main, **clean pick** |
| | | `tools/test_post_record_allowlist.py` | new file, still absent, **clean add** |
| | | `tools/check_linux_release_qualification.py` | **the one conflict** |
| 2 | `aa23e7b3a` the wheel content audit and the macOS repack | all four files | new files, still absent, **clean add** |
| 3 | `45fd47f2f` NOTICE must not carry "used under license" | the audit and its test | applies on top of 2 |

**The one conflict, in full, so nobody has to rediscover it.** Main moved
`tools/check_linux_release_qualification.py` exactly once since
`6cb6fa571^`, at `f9e49346a`, the native-source rule gaining `tokenizer/tools/`:

```python
             if (name.endswith('.mojo') or rel.startswith((
-                    'bindings/', 'packaging/linux/', 'python/mojolearn/'))
+                    'bindings/', 'packaging/linux/', 'python/mojolearn/', 'tokenizer/tools/'))
```

`release/0.8.6` carries the same change as `9cc557e3d`, so the two sides agree
on intent. Resolve by keeping both hunks, main's prefix line and
`6cb6fa571`'s post-record additions. `tools/test_native_source_rule.py` on
this base is the check that the three copies of the rule still agree; run it
after the pick.

After the picks, the fail-first control is to run each grep in the table of
section 2.1 against `release/0.8.7` and see it **print matches** where it read
ABSENT here. A commit message is not a file change.

---

## 3. The record scope on the new base, and whether the columns carry

### 3.1 The scope moved, because the registry moved

Measured on this base by importing the registry, not by grep:

| quantity | 0.8.6 harness | 0.8.7 base |
|---|---|---|
| registry lanes (`lane_select.py --count`) | 192 in the record run | **211** |
| `par-*` lanes excluded by `RECORD_EXCLUDED_PREFIXES` | 39 | **50** |
| **release record scope** | 153 | **161** |
| `LANE_REVISIONS` entries | 1 | **14** |

`RECORD_EXCLUDED_PREFIXES = ("par-",)` is unchanged at
`tools/identity_break.py:818`, so the rule my audit relied on still holds.
What moved is both sides of it.

### 3.2 THE THREE EXISTING COLUMNS CANNOT BE CARRIED INTO 0.8.7

Saying this loudly, because it was the single biggest saving on the table and
it is not available. **All three GPU columns must be retaken, and so must the
CPU column.**

There are three independent reasons and **each one alone is fatal**. They were
established by reading `merge()` in `tools/identity_break.py:7458`, not by
running a merge.

**Reason 1, the commit.** `MERGE_SAME` includes `"commit"`. The existing
columns record `db9047b9f`; a 0.8.7 column records the 0.8.7 build commit.
`merge()` raises `SystemExit` on the first differing `MERGE_SAME` key.
`--allow-separate-builds` does **not** relax this; it relaxes only the binding
digest test, and the key loop runs before it.

**Reason 2, the fixtures.** `MERGE_SAME` also includes `"lane_revisions"`. The
columns carry `{"tokenizer": "synthetic-vocab-1"}`; this harness carries
fourteen entries. `stale_revision_lanes` run against each column returns the
**same thirteen lanes** for all three:

```
byte-lm, byte-lm-resident, gbdt-lossguide-newtoncosine, gbdt-nan-modes,
gbdt-pair-logit, gbdt-parametric-losses, hdbscan, hdbscan-leaf, holtwinters,
mamba2-dtlimit, samba, samba-untied-dropout-accum, spectral
```

Those cells describe inputs this harness can no longer produce. This is not a
technicality to route around; it is the check `fc28d435f` added precisely so
that a stale reference cannot be reported as DIVERGENT and read as the
identity claim being false.

**Reason 3, the bytes.** `_build_digests` compares every loaded binding's
sha256 across parts, and "the parts of one column must run one build" is what
"recorded bytes are shipped bytes" rests on. The existing columns ran a
fifteen-family wheel. 0.8.7 ships thirty-two families and main carries new GPU
code besides, so every digest differs.

**And coverage falls short anyway.** Even ignoring all three, measured against
the 161-lane scope:

| column | recorded | in scope | minus the 13 stale | usable | in-scope lanes with no cell |
|---|---|---|---|---|---|
| nvidia-h100-sm_90a | 192 | 153 | 13 | **140 of 161** | 8 |
| amd-mi300x-gfx942 | 162 | 151 | 13 | **138 of 161** | 10 |
| apple-m4 (125 parts) | 151 | 151 | 13 | **138 of 161** | 10 |

The lanes with no cell, because they did not exist when the columns were
taken: `arima-exog`, `arima-exog-seasonal`, `bpe-trainer`,
`gbdt-categorical-ctr-tables`, `gbdt-tensor-ctr-tables`, `gbdt-yeti-rank`,
`gp-optimize`, `gp-optimize-restarts`, plus `iforest` and `iforest-tuned` on
AMD and Apple.

### 3.3 What the old columns are still worth

They are not waste and they should not be deleted.

* They are the only recording of the **`rf-score-weighted` MOVED cell**, and
  that finding stands on its own, because `de981fa3f` confirmed the same
  defect in the **published 0.8.5 wheel** from PyPI (13 of 300 fits moved at
  the shipped default on an MI300X, 0 of 300 on the control, all thirteen
  distinct). The defect is in 0.8.7's scope as a defect, not as a record
  problem.
* When the 0.8.7 AMD column is taken, whether `rf-score-weighted` moves again
  is a direct third and fourth reading against legs 3 and 4.
* The Apple part files are a **measured cost model** for the next Apple run,
  including the per-group seconds and queue counts that settled two lanes per
  process. That is worth real hours.

### 3.4 What the fold buys back on the CPU column

0.8.6's first CPU column attempt failed because the wheel shipped fifteen
families and 747 of 1386 cells read REFUSED by name, so the column had to be
built from source on a rented box. With thirty-two families shipping, a
wheel-installed CPU run now reaches **122 lanes against 39**, which removes
that failure mode. 122 is still short of the 161-lane scope, so the
from-source route through `tools/cpu_identity_gate_check.py run-column`
remains the way to take the full column. The measured saving is the failure
mode, not the route.

### 3.5 The Apple column is guarded now, and needs its release named

`refuse_routine_apple_column` refuses more than `APPLE_COLUMN_LANE_LIMIT`
(24) lanes in one Apple process unless `MOJOLEARN_APPLE_RELEASE_RECORD` names
the release. The 0.8.7 column therefore runs under
`MOJOLEARN_APPLE_RELEASE_RECORD=0.8.7`, one small group per process through
`mac_slot.sh metal`, two lanes per process on the 0.8.6 evidence.

Cost, stated honestly and not minimized. The 0.8.6 Apple run reached 151 lanes
across 125 part files written between 18:16 on 2026-09-15 and 05:40 the next
morning, and main's own harness comment records a full pass measuring **seven
hours** and stopping at 125 of 158 lanes. The
0.8.7 scope is 161 lanes on a wheel with thirty-two host families. **Plan on
at least a full night, and take it once.** This is the largest single cost in
the release and nothing in the fold reduces it.

**Nothing was rented and no column was taken for this plan.**

---

## 3A. THE CRITICAL PATH. The random forest repair is INERT, and the builds wait

**Do not start a build against this source. It is about to change.**

The decision I ranked eighth moved on the morning of 2026-09-16 and it is now
the item the release schedule hangs off.

**The repair that three branches agreed on does nothing.**
`_ = Atomic.load[ordering = Ordering.ACQUIRE](mutex)` **emits zero
instructions** on Metal AIR, PTX and GCN alike, proved by bit-identical
`__TEXT` and `__DATA` across the arms. A discarded load is dead code, and the
ordering rides on the load rather than standing on its own, so the optimizer
takes both away. That is why the A/B had a replicated EFFECT and no
demonstrated MECHANISM. A 0 of 300 from a build whose only change emits nothing
was never measuring the repair; it was measuring codegen noise, which is
exactly what `41fa431ce` said it could not rule out.

The inert spelling is the one actually on the branches, checked rather than
assumed:

| branch | sites |
|---|---|
| `lane/rf-mutex-claim-acquire` | `ensemble/.../split.mojo:676` and `:695`, `extratrees/.../split.mojo:504` and `:524`, `neighbors/impl/detail/fused_l2_knn.mojo:618` |
| `fix/amd-merge-ordering` | `core/device_mutex.mojo:45` |

**The spelling that emits is `fence[ordering = Ordering.ACQUIRE]()` from
`std.atomic`.** A lane is correcting that now. `git grep "fence\[ordering"` over
this base returns **nothing**, so the corrected repair is not yet written
anywhere in the tree, while `Ordering.ACQUIRE` returns matches in five files,
which is the control that says the search works.

**Item 8 is therefore not "pick one of three branches".** It is "wait for the
corrected repair to be written, measured and landed". Nothing chooses between
the three branches today, because all three carry the same spelling and that
spelling is inert.

### A question this opens about the SHIPPED code, flagged and not concluded

The four claim sites on `origin/main` **already** use
`Atomic.load[ordering = Ordering.ACQUIRE]` in their spin conditions, today, in
shipped source:

```
ensemble/decisiontree/batched_levelalgo/split.mojo:635
extratrees/impl/decisiontree/batched_levelalgo/split.mojo:504
neighbors/impl/detail/fused_l2_knn.mojo:618, :724
```

These differ from the repair's form in one way that matters. The repair's load
is DISCARDED, so the whole statement is dead and emits nothing. These loads'
values ARE used, so a load instruction must be emitted. What is not settled is
whether the **acquire ordering** on them is honored or is dropped the same way,
and if it is dropped then the shipped spin-wait has been resting on an ordering
it never had. That is a question for the lane holding the instrument, not a
conclusion here, and it is not evidence about the `rf-score-weighted` defect on
its own. It is written down because the same measurement that proved the repair
inert can answer it at no extra cost, and because the four sites are the shipped
ones.

### What this means for sequencing

1. The corrected repair touches `ensemble/`, `extratrees/`, `neighbors/` and
   possibly `core/` Mojo source, all of it on the native build inventory.
2. So **no Linux set and no macOS wheel may be built until it lands or is
   ruled out**, and the freeze commit cannot be named before then.
3. And because the columns come from the built wheel, **no column may be taken
   either.** The Apple night in particular must not start; it is the single
   most expensive thing here and a source change after it throws the whole
   night away. That is precisely what the fold just did to the 0.8.6 Apple run,
   and it is not worth doing twice.

Items 3 to 7 of the ranked list are all downstream of this. Items 1 and 2 are
not, which is why they are done first.

---

## 4. Ranked list of what 0.8.7 needs before Andrew could say ship

**Items 1 and 2 are DONE. Everything from 3 down is blocked on section 3A.**

1. ~~**Merge back the three pieces of release machinery**~~ (section 2). DONE
   on this branch, 2026-09-16. The predicted conflict on
   `tools/check_linux_release_qualification.py` did not happen; git auto-merged
   it and both hunks were checked by hand. `test_native_source_rule` 2 OK,
   `test_post_record_allowlist` 6 OK, `test_release_wheel_content_audit` 5 OK,
   `test_repack_post_record` 4 OK.
2. ~~**Fix the false 0.8.6 claims**~~ (section 1.3). DONE, and **on main**, not
   only here, because they were wrong for a reader of main today. Seven
   statements corrected in `CHANGELOG.md`, `README.md`, `SUPPORT_MATRIX.md`,
   `docs/BYTE_LM_CPU_TRAINING.md`, `docs/RELEASE_CHECKLIST.md`,
   `docs/VERIFY_EXTERNALLY.md` and `tools/verify_external.sh`, including the
   two that still said sixteen families, one that still said ten, and two
   CHANGELOG lines that called the 0.8.6 wheels published. `docs_facts --check`
   reads OK. Still owed at freeze: the version bump itself, in
   `python/mojolearn/_version.py` and `python/pyproject.toml` (both read 0.8.5
   today, correctly, since 0.8.7 is not frozen), and `CITATION.cff`.
3. **THE BLOCKER, section 3A. The corrected random forest repair.** The agreed
   repair is inert, the emitting spelling is `fence[ordering = Ordering.ACQUIRE]()`,
   and it is not written anywhere in the tree yet. Nothing below can start.
4. **Freeze 0.8.7 at a named commit** and run the freeze checks of
   `docs/RELEASE_CHECKLIST.md`. Every build below runs from that commit.
5. **Three Linux build sets** at the freeze commit, sm_89, sm_90a and gfx942,
   then the byte compare of the host bindings across all three. Now thirty-two
   host families instead of fifteen, so expect longer compiles and a larger
   wheel. Needs Andrew's approval to rent.
6. **Pack, audit, strip and upload the Linux wheel**, and build the macOS
   wheel, with the content audit from item 1 run on both.
7. **Three of the four columns**, all retaken (section 3.2), over the 161-lane
   scope. NVIDIA and AMD from the installed Linux wheel, the CPU column from
   source. Needs Andrew's approval to rent.
8. **The Apple column** (section 3.5), LAST, once, under
   `MOJOLEARN_APPLE_RELEASE_RECORD=0.8.7`. Plan a night. Do not start it until
   items 3 to 6 are settled, because a source change after it is taken throws
   the whole night away, which is exactly what the fold just did to the 0.8.6
   Apple run.
9. **The diff, the record commit, the final pack and repack**, and the final
   content audit on both wheels.

The `rf-score-weighted` decision that was item 8 is now item 3, because it
turned out to gate the builds rather than follow them. Its standing facts are
unchanged and are in section 3.3: the defect is already published in 0.8.5, so
not shipping 0.8.7 protects nobody, and either the corrected repair lands with
its mechanism settled or 0.8.7 ships with the defect and the CHANGELOG says so
in plain words.

**STOP BEFORE PUBLISH.**

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
