# 0.8.6 audit, 2026-09-16, after the Mac session crash

**Nothing has been published and nothing here publishes anything.** This file
is an audit of state, reconciled against the repository and the evidence on
disk rather than against the previous state file. Publishing needs Andrew's
separate explicit "ship".

Read this instead of `docs/lanes/RELEASE_086_STATE.md`, which is a 2026-09-15
evening snapshot and is wrong in three ways that matter (below).

---

## The headline, in one paragraph

0.8.6 is much closer to done than the old state file says, and the two things
that looked like walls are gone. The AMD column does not need ten to thirty
more rentals and the Apple column does not need another ten hours, because
`lane/identity-fixtures-light` (merged to main on 2026-09-16) put the release
record's scope into code and that scope **excludes every `par-*` lane**. Under
that scope the record is 153 lanes, and NVIDIA is complete, AMD is 151 of 153
and Apple is 151 of 153, each missing exactly two lanes, `iforest` and
`iforest-tuned`. Against that, one new item outranks everything else on the
list. The wheel 0.8.6 would ship carries a `verify --all` that can print
`VERIFIED, exit 0` on a run in which most parts refused to execute, and the
fix for that is on main and not on this branch.

---

## 1. What the old state file got wrong

### 1.1 The stale state file was the LOCAL branch, not the release branch

The local `release/0.8.6` sat at 7e23bc670 with a 355-line
`RELEASE_086_STATE.md`. `origin/release/0.8.6` is at **f7943e757**, which is
**the same commit as `fix/release-post-record-allowlist`**, and carries a
**1189-line** one. The only difference between 7e23bc670 and f7943e757 is that
single file (`git diff --stat` reads one file changed, 834 insertions).

So `fix/release-post-record-allowlist` is not a fix branch any more. It is the
branch the release pushes were made from, it IS the published release branch,
and it holds everything from 2026-09-15 20:00 onward, including the complete
NVIDIA column, the CPU column, AMD legs 2 to 5 and the `rf-score-weighted`
blocker. The long state file was never lost; only the local pointer was stale,
for the second time (the file records the first).

**Done in this commit.** The local branch was rebased onto
`origin/release/0.8.6`, so the two agree again and this audit sits on top of
the current state. The lesson stands for the next session. Read
`origin/release/0.8.6`, never the local pointer, and check
`git rev-parse origin/release/0.8.6` against `HEAD` before believing any state
file on this branch.

### 1.2 The Apple column is at 151 lanes, not at chunk 00

The old file describes Apple as chunk 00 plus a plan for chunks 01 to 06 and
an estimate of 10.6 more hours. The run continued overnight. There are **125
Apple part files** in `~/mojolearn-evidence/release-0.8.6/apple-record/`, the
last written at 05:40 on 2026-09-16, covering **151 of the 192 harness lanes,
1449 cell entries, zero MOVED cells**.

Measured properties of that set, because a merge has to be safe before it is
run:

* 90 cells appear in more than one part file and **all 90 agree byte for
  byte**, so `identity_break.py --merge` will not refuse.
* The three contended parts are quarantined in
  `apple-record/contended-2026-09-15/`, outside the merge glob, and all five
  lanes they touched (`radius`, `standard-scaler`, `minmax-scaler`,
  `gbdt-feature-freq`, `mlp`) were re-recorded in clean `c16s-*` parts.
* The run stopped at `c16s-group118`, which wrote no JSON. Its log is a
  `BrokenPipeError` inside `_parallel_worker.py`, so it was a `par-*` lane and
  is out of scope anyway.

### 1.3 "Main carries each fix alone" is false for three of the fixes

The old file says "Main carries each fix alone: c25fc046c, 81cc87979,
3f8b5bd69 (and the lanes' own merges)". Those three commits exist and do carry
their fixes. But the sentence reads as though every 0.8.6 fix is also on main,
and three of them are not on main at all. See section 3.

---

## 2. Evidence the release still owes

### 2.1 The record's scope changed, in code, on main

`tools/identity_break.py` on main now carries a record-scope block
(`lane/identity-fixtures-light`, 2026-09-16):

```python
RECORD_EXCLUDED_PREFIXES = ("par-",)
```

with three measured reasons stated in the source, the first of which cites
this release by name ("The `par-*` lanes were the whole remaining AMD gap for
0.8.6: 28 of the 30 lanes that column never reached"), the second of which is
that several `par-*` lanes cannot be covered honestly at all, and the third of
which is that `_par_devices()` defaults to device 0, so every `par-*` cell in
a record is a ONE-device run and is not the two-device claim anyone thinks it
is.

The 0.8.6 harness defines 192 record lanes, 39 of them `par-*`. **The release
record's scope is therefore 153 lanes.**

### 2.2 Column state against that scope

Computed from the column JSONs on disk, not from prose.

| column | in-scope lanes | missing | MOVED | complete |
|---|---|---|---|---|
| nvidia-h100-sm_90a | **153 / 153** | none | 0 | yes |
| amd-mi300x-gfx942 | **151 / 153** | `iforest`, `iforest-tuned` | **1** | no |
| apple-m4 (125 parts, unmerged) | **151 / 153** | `iforest`, `iforest-tuned` | 0 | no |
| cpu-amd-epyc-7713 | 148 / 153 | `gmm-random-init-sample`, `gmm-sample`, `gp-sample-y`, `gp-sample-y-normalize`, `tokenizer` | 0 | its own scope |

The CPU column runs a different scope on purpose, the 159 lanes
`host_surface.py --covered-lanes` names, and the four-column diff is scoped to
the 154 that `--record-covered-lanes` names. Its five in-scope gaps are lanes
with no CPU training path in that list, which is the designed behavior and not
a gap to fill.

So all three GPU legs of the record **exist**. Two are complete against scope
and one is complete but for one cell. What is owed is:

1. **AMD**, two lanes, `iforest` and `iforest-tuned`. The `--lanes` subset path
   is already wired and was verified in both directions
   (`scripts/make_record_body_uploading.sh`, 8th argument), so this is one
   short Hot Aisle lease, not ten.
2. **Apple**, the same two lanes, one small group under the Metal lock, plus
   the merge of the 125 parts into one column.
3. **The merge and diff**, then pending steps 8 to 11 of the old file.

**Nothing was rented and no column was taken for this audit.** Both of the
above need Andrew's word first.

### 2.3 The record's scope rule must NOT be picked onto this branch

`packaging/linux/pack_wheel.py:598` copies `tools/identity_break.py` into the
wheel as `mojolearn/_identity_break.py`, and every record leg checked that the
installed harness was byte equal to the checkout's copy. Editing
`tools/identity_break.py` here would change a packed wheel member and break
that equality for every leg already taken.

`tools/identity_break.py` is **not** on the native build inventory (the 1687
file list in each build proof contains no `tools/` entry), so the packer's
post-record check would not catch it. That makes the hazard quieter, not
smaller.

**The scope is therefore applied as a stated scope for this record**, written
in the record README, with the 39 excluded `par-*` lanes named and main's
`RECORD_EXCLUDED_PREFIXES` cited as where the rule lives. No shipped byte
moves. 0.8.7, built after the rule is on its branch, gets it in code.

---

## 3. Per fix branch, verified by reading the code

Every claim below was checked by comparing blobs and by grepping distinctive
single-line tokens on both refs, with a control that prints matches on the ref
where the thing is present. A grep that returns zero on both refs is not
evidence, which this audit hit once and corrected (section 6).

| branch | tip | content on `release/0.8.6` | content on `main` |
|---|---|---|---|
| `fix/release-build-preflight-import` | 65a9e9302 | **yes**, ancestor | **yes**, as c25fc046c, both files byte equal |
| `fix/release-verifier-bindings` | 70d0940a6 | **yes**, ancestor | **yes**, as 81cc87979 |
| `fix/release-archive-tokenizer-generator` | 9cc557e3d | **yes**, ancestor | **yes**, all four copies of the native source rule carry `tokenizer/tools/` |
| `fix/macos-smoke-identical-only` | 68906d504 | **yes**, cherry-picked as db9047b9f and ef2b077e4, blobs byte equal (the tip is NOT an ancestor, which is why blobs were compared) | **yes**, `packaging/macos/smoke.py` and `verify_wheel.sh` byte equal, the latter as 3f8b5bd69 |
| `fix/release-post-record-allowlist` | f7943e757 | **the code, yes; the 31 state-file commits, no** | **NO. The whole post-record allowlist, the content audit and the macOS repack are absent from main** |

### 3.1 The three release fixes that are on this branch and not on main

Repo-wide `git grep -l` on `main` with a control on `release/0.8.6`:

| token | main | release/0.8.6 |
|---|---|---|
| `POST_RECORD_FILES` | ABSENT | `packaging/linux/pack_wheel.py`, `tools/verify_linux_surface_qualification.py`, `tools/test_post_record_allowlist.py` |
| `post_record_differences` | ABSENT | `packaging/linux/pack_wheel.py`, `tools/check_linux_release_qualification.py`, `tools/test_post_record_allowlist.py` |
| `post_record_equivalent` | ABSENT | `packaging/macos/repack_post_record.py`, `tools/verify_linux_surface_qualification.py` |
| `NOTICE_FORBIDDEN` | ABSENT | `tools/release_wheel_content_audit.py` |
| `repack_post_record` | ABSENT | `packaging/macos/repack_post_record.py`, `packaging/macos/test_repack_post_record.py` |
| `release_wheel_content_audit` | ABSENT | `tools/release_wheel_content_audit.py`, `tools/test_release_wheel_content_audit.py` |

`git ls-tree main -- tools/release_wheel_content_audit.py` and
`git ls-tree main -- packaging/macos/repack_post_record.py` both return
nothing; the same commands on `release/0.8.6` return blobs.

This is not a release blocker, because the release packs from this branch. It
is a **merge-back debt**. Step 9 of the old plan says the record is
cherry-picked back to main, and if only the record goes back, main keeps a
packer with no post-record allowlist, no wheel content audit and no macOS
repack, and 0.8.7 rediscovers all three.

### 3.2 Direction two, what main has that this branch does not

Checked because a fix present on main and absent from the release branch is
exactly what ships broken.

* **`python/mojolearn/_verify_all.py`, commit 87085a5eb.** See section 4.1.
  This is the serious one.
* **`_verify_reference.admit`, commit 323c8980d.** The admission rule matched
  its exclusion tokens against the whole lowercased path, so a clean column
  recorded beside a negative control was silently discarded for its
  neighbor's directory name. Measured over 495 committed columns, 220 admitted
  before and 222 after. `verify --all --emit-reference` is release step 9, so
  0.8.6's shipped reference table would be built by the rule with this defect.
* **`stale_reference_lanes`, commit fc28d435f.** Refuses to compare a lane
  whose fixture moved past its reference. 0.8.6 does not need it (its
  `LANE_REVISIONS` is `{"tokenizer": "synthetic-vocab-1"}` alone and its
  record is taken at its own fixtures), but see section 5.3.
* **`b4adb8432`, one resolution path for every host binding.** `MOJOLEARN_HOST_DIR`
  moved the training door and not the inference door, so under the override
  the inference path served the package binding while another set was asked
  for. It affects host-set overrides, not the default install path.
* Everything else merged to main since the freeze is new capability (ARIMA
  exogenous regressors, the GP optimizer, `SpectralClustering.predict`,
  Holt-Winters CPU inference, GBDT CTR tables, YetiRank, the BPE vocabulary
  trainer) and belongs in 0.8.7. Deferring it is correct.

---

## 4. What would make 0.8.6 wrong if it shipped today, by severity

### 4.1 SEVERITY 1. The shipped verifier can print VERIFIED on a run it did not perform

On this branch, `python/mojolearn/_verify_all.py`:

```python
def verdict(counts):
    """(exit code, headline) from the state counts of every judged part."""
    if counts[vref.DIVERGENT]:
        return EXIT_MISMATCH, "MISMATCH"
    if counts[vref.IDENTICAL]:
        return EXIT_VERIFIED, "VERIFIED"
    if counts[vref.REFUSED]:
        return EXIT_CANNOT_RUN, "CANNOT RUN"
    return EXIT_NO_REFERENCE, "NO REFERENCE"
```

One IDENTICAL part outranks any number of REFUSED ones. Main's 87085a5eb
reorders it so any refusal reads INCOMPLETE and exits non-zero, and records
the measurement behind it: a CPU-only install with stale bindings printed
`VERIFIED, exit 0` on 44 IDENTICAL and 288 REFUSED parts, so **the user had
checked 13 percent of what they believed they checked**.

Why this lands hardest on this particular release. 0.8.6 ships **15 of the 32
host families**, and this release's own CPU column attempt 1 measured what
that does on a CPU-only install from the wheel: of 1386 cells, 639 STABLE and
**747 REFUSED**, by name, "no host binding covers `_mojolearn_rf`". That is a
54 percent refusal rate on exactly the install that would print VERIFIED. The
verifier is the product's central claim, and this is the configuration in
which it is least true.

The cost of fixing it is real and must be stated plainly. `python/mojolearn/_verify_all.py`
**IS** on the native build inventory. It is entry
`['python/mojolearn/_verify_all.py', ...]` in
`linux-builds/cuda-sm_90a-db9047b9f/source_inventory_local.json`, 1687 files,
and `POST_RECORD_FILES` admits only `python/mojolearn/host_surface.py` and
only three assignments inside it. So picking 87085a5eb onto this branch means
**all three Linux sets rebuilt and the macOS wheel rebuilt**, a new pack, and
re-recording at the new commit. That is the whole release again.

The three options, for Andrew:

1. **Rebuild.** Correct, and costs the release over again.
2. **Ship 0.8.6 as it is and fix it in 0.8.7**, with the CHANGELOG saying in
   plain words that `verify` on this version reports VERIFIED when parts
   refused and that a user must read the counts. Honest, and leaves a wrong
   verdict in a published artifact.
3. **Do not ship 0.8.6 at all**, fold it into 0.8.7 with the verifier fix, the
   16th family and whatever else lands. The freeze was 2026-09-15 and main has
   moved 149 commits since.

This audit does not choose. It reports that the choice exists and that nobody
has made it.

### 4.2 SEVERITY 2. The AMD column carries a MOVED cell, and it is a defect already shipped in 0.8.5

`rf-score-weighted/wide` in the merged AMD column reads
`49be8ea935a47640` then `50ce4a9f62cddf8e`. Every other column reads the first
value, twice. One machine disagreeing with itself.

What the old state file did not have, and what changes the decision. The lane
`lane/rf-score-weighted-nondeterminism` has 17 commits of work on exactly this
defect, and commit de981fa3f, 2026-09-16 07:30, records `pip install
mojolearn==0.8.5` from PyPI on an MI300X:

```
cols16_SHIPPED_DEFAULT  max_features=1.0    13/300 moved (4.3%)  distinct=13
cols10_CONTROL          max_features=0.625   0/300 stable
```

All 13 moved fits produced distinct models, so it is a race and not two
attractors, and the control is clean, so the probe is sensitive. The fit path
is semantically unchanged from v0.8.0 through db9047b9f (every changed line in
`split.mojo` over that span is comment text).

That reframes it completely. **This is not a 0.8.6 regression and not a 0.8.6
release gate.** A user on an MI300X calling
`RandomForestRegressor(...).fit(X, y)` with 11 or more features and the
shipped default already gets a different model about 4 percent of the time,
today, from the published 0.8.5 wheel. 0.8.6's record would be the first time
this is written down honestly.

The repair exists and is not landed anywhere. Three branches agree on a
post-success acquire load at the four claim sites
(`lane/rf-score-weighted-nondeterminism`, `lane/rf-mutex-claim-acquire`,
`fix/amd-merge-ordering`). `git grep MOJOLEARN_RF_MUTEX_CLAIM_STOCK` returns
two matches on `lane/rf-mutex-claim-acquire` and **nothing on `main` or on
`release/0.8.6`**. Its EFFECT is replicated (control 7/300 then 6/300 against
0/300 twice, p about 1e-3 each) but its MECHANISM is not demonstrated, so
41fa431ce says nothing is landed as a default. It is also in `ensemble/` and
`extratrees/` Mojo source, so landing it would force the same full rebuild as
4.1.

The decision rule already fixed in advance still holds. The lane is not
dropped, not marked n/a and not excluded from the diff. Whether 0.8.6 ships
with a known moving cell is Andrew's call with the record in front of him, and
the new fact is that refusing to ship does not protect anyone, because the
same defect is already on PyPI.

### 4.3 SEVERITY 3. The wheel's host-family surface is being changed under the release

Three different answers to "what does the wheel ship" are live at once:

| ref | `host_surface.py --wheel-families` |
|---|---|
| `release/0.8.6` | **15** |
| `main` | **16** (`resample`, added by `lane/expose-inference-surface`) |
| `lane/ship-cpu-host-families` | **32** |

The release is internally consistent at 15. Its three Linux build proofs each
carry 15 host bindings, the packed wheel carries 15, and every record leg read
back 15.

`lane/ship-cpu-host-families` flips `ships_in_wheel` from False to True on 16
families, and both wheel builders and the packer read `--wheel-families`. If
that branch merges and anyone assumes 0.8.6 should carry it, **every build and
every column is void**. Its own reasoning is aimed squarely at this release
("an installed wheel could check 39 of the harness's 199 identity lanes"), and
it is the same argument as 4.1 from the other side.

This needs one sentence from Andrew and it costs nothing to get it now.
Either 0.8.6 ships 15 families and the 32-family change is 0.8.7, or 0.8.6
waits for it and every build is retaken. The branch is also incomplete on its
own terms; its CHANGELOG and LANE_STATUS carry a literal `<!--TBD-->` where a
`verify --all` count goes, and the run that would produce it was killed by the
crash.

### 4.4 SEVERITY 4. The shipped reference table would be built by a defective admission rule

`verify --all --emit-reference` is release step 9 and it runs from this
branch, where `_verify_reference.admit` still matches its exclusion tokens
against the whole path (main's 323c8980d). On main's measurement that silently
discards two clean columns of 495. The table would ship slightly thinner than
it should be, with no report that anything was refused. Same rebuild cost as
4.1 if fixed here, since `python/mojolearn/_verify_reference.py` is in the
same 1687 file build inventory.

### 4.5 SEVERITY 5. The record spans two build commits, and the reason is sound but must be in the record

The three Linux columns record db9047b9f; the Apple column records 2f53960ca,
the commit the macOS wheel was built at. 2f53960ca is an ancestor, four
commits and eight files separate them, exactly one of the eight is on the
native inventory (`tools/linux_surface_qualification.sh`, which takes no part
in a macOS build), and none of the eight ships in either wheel, checked
against both member lists with a control pair that reads 1 and 1. That
analysis is already written in the long state file and it holds. It has to be
in the record README, not only in a lane document, because `--diff` compares
no commit and will not raise it on its own.

### 4.6 SEVERITY 6. 206 MB of leg evidence is sitting in a directory the OS reaps

`/private/tmp/claude-501/.../scratchpad/wt-release-086/bench/results/releases/2026-09-16-linux-0.8.6/`
holds the Hot Aisle and RunPod leg outputs, untracked, in the crashed
session's worktree. The build sets that matter were copied to
`~/mojolearn-evidence/release-0.8.6/linux-builds/` already, so this is the
superseded and console material, but it is the only copy of it.

---

## 5. Things that are true and easy to get wrong

### 5.1 The AMD merge refuses, by design, and the decision is already recorded

`identity_break.py --merge` refuses legs 3 and 4 together because both carry
`rf-score-weighted` and they disagree (leg 3 moved on `wide`, leg 4 moved on
`base`, each on a different VM). Andrew's recorded decision is that the AMD
column carries **leg 3's** recording, as the larger part, and leg 4 is kept
beside it in `records/amd-leg-4/` as the second reading, because the finding
is that the two disagree. The merged column on disk
(`records/amd-mi300x-gfx942.json`) already reflects this. Its `merged_from`
reads leg 1 at 558 cells, leg 3 at 882 cells and
`amd_leg4_minus_contested.json` at 18 cells, which is leg 4's `par-dbscan` and
`par-scaler` only.

### 5.2 The wheels and the R2 upload are done and should not be redone

Unless 4.1 or 4.3 forces a rebuild. The macOS wheel
(sha256 `eba69f83c9a9...`) and the final Linux wheel
(sha256 `7cab1aa3cfcd...`, in R2 at
`releases/0.8.6/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl`) are
packed, audited, stripped and content-audited. The record repack at the end
replaces only `host_surface.py`, `verify_reference/table.json` and the
identity column JSONs.

### 5.3 Merging the record back to main is not a plain cherry-pick any more

Main's `LANE_REVISIONS` now names 13 lanes whose fixtures were shrunk on
2026-09-16 (`gbdt-parametric-losses`, `gbdt-nan-modes`,
`gbdt-lossguide-newtoncosine`, `gbdt-pair-logit`, `hdbscan`, `hdbscan-leaf`,
`spectral`, `holtwinters`, `byte-lm`, `byte-lm-resident`, `samba`,
`samba-untied-dropout-accum`, `mamba2-dtlimit`). This branch's is
`{"tokenizer": "synthetic-vocab-1"}` alone. The 0.8.6 record is taken at the
OLD fixtures, so once it lands on main, main's own `stale_reference_lanes`
check will correctly mark those 13 lanes stale and drop them from comparison.
That is the check working, not a defect, but it means the record contributes
13 fewer lanes to main's table than it does to the release's own.

Main's harness also carries 188 `@lane(...)` decorators to this branch's 171,
so any lane count taken from main does not describe this record.

### 5.4 The Metal reading that must not be quoted

There is still no pre-slowdown Metal GBDT timing committed anywhere, so no
healthy per-fit number is claimed. Two measurements of the same degraded
window disagree (about 2.2x to 2.7x on `gbdt_direct.py` against 1.06x on the
seven identity lanes) and nobody has established why. What is settled is that
the rerun of those seven lanes was bit for bit identical across all 63 cells,
so the degraded window cost time and not one bit.

---

## 6. A probe of mine that returned zero on both sides

Worth recording because it is the failure this project keeps catching. To test
whether the README trademark section was on main I grepped for
`not affiliated with`. It returned **zero on main and zero on
`release/0.8.6`**, and the first reading of that was "absent from main". The
phrase wraps across a line in the file ("not affiliated\nwith, sponsored
by"), so the grep could not have matched anywhere. Re-probed with the
single-line tokens `Trademarks and affiliation` and `sponsored by, or endorsed
by Modular`, both refs print line 543 and line 546. **The section is on main.**

Every absence claim in section 3.1 was run with a control on the ref where the
token is present, and the matches are printed above rather than the counts.

---

## 7. Ranked list of what 0.8.6 needs before anyone can say ship

1. **A decision on the verifier defect (4.1).** Rebuild, ship with it stated
   in the CHANGELOG, or fold 0.8.6 into 0.8.7. Everything else is cheap; this
   one is the release.
2. **A decision on the moving AMD cell (4.2)**, now that it is known to be a
   defect already published in 0.8.5 rather than a 0.8.6 regression.
3. **A decision on the wheel's host families (4.3).** 15 for 0.8.6 and 32 for
   0.8.7, or retake every build.
4. **Two AMD lanes**, `iforest` and `iforest-tuned`, one short lease with
   `--lanes`. Needs Andrew's approval to rent.
5. **Two Apple lanes**, the same two, one small group under the Metal lock,
   then merge the 125 Apple parts into one column.
6. **Write the record scope into the record README** (2.3), naming the 39
   excluded `par-*` lanes and citing main's `RECORD_EXCLUDED_PREFIXES`. Do not
   edit `tools/identity_break.py` on this branch.
7. **Run the diff and the owed check** (`scripts/release_diff.sh`), then the
   record commit, the final pack and repack, and the final content audit.
8. **Plan the merge-back of the three release fixes to main** (3.1), so 0.8.7
   does not rediscover the post-record allowlist, the content audit and the
   macOS repack.
9. **Move the 206 MB of leg evidence out of `/private/tmp`** (4.6).

**STOP BEFORE PUBLISH.** No PyPI upload, no tag that triggers one, no
`release-provenance.yml` dispatch.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
