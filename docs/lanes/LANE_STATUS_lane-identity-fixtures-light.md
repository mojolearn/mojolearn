# LANE STATUS: lane/identity-fixtures-light (2026-09-16)

Making identity verification LIGHT again. Branch `lane/identity-fixtures-light`,
cut from `main` at bfb8f725a. Three jobs: shrink the expensive identity
fixtures, take the `par-*` lanes out of the release record's scope, and write
down the AMD position for 0.8.6.

**Nothing here touches the frozen 0.8.6 commit db9047b9f or `release/0.8.6`.**
Fixture changes are code changes and would invalidate every column recorded;
they land on main for the NEXT release record. The 0.8.6 Apple column was
recording on the Metal lock throughout this lane; no job here took the Metal
lock and no box was rented.

## 1. What the record actually costs, measured

Read from the running Apple column's own logs, not re-run:
`~/mojolearn-evidence/release-0.8.6/apple-record/c16*-group*.log` (the harness
header timestamp) against each group's JSON mtime, joined with the same
group's `.queues` file (`ioclasscount AGXCommandQueue`, sampled through the
run).

**WITHDRAWN: the queue column is not a signal about our lanes.** The two
tables below sort lanes by the Metal command-queue count sampled during their
group, and an earlier revision of this file treated the high-count groups as
"contaminated" and discounted their times. **That classification is wrong and
is withdrawn.** It is kept below only as a record of what was observed, with
its conclusion removed.

Three explanations were tried and all three are dead, in order:

1. *A leak in our bindings.* No. lane/metal-queue-leak could not reproduce
   accumulation: eleven GBDT fits flat at 34, four later workloads flat at 22
   to 34.
2. *Per-call `DeviceContext` creation.* No. Ruled out by the same health check.
3. *Contexts held as fields on model objects*, so that many live models mean
   many queues. This fitted my data and I believed it. **Also no.**

What was actually measured on 2026-09-16: the queue count climbed 2105 to 4663
in front of the leak lane, and the holder was **pid 37288, `DockHelper`, an
Apple XPC service under launchd**, gaining 12 to 18 queues a second, while a
seven-lane `identity_break.py` run holding the Metal slot at that moment held
**zero to one queue**. The queues were never ours.

And the pressure is harmless anyway: at 4663 queues, nine times the stated 512
limit, with a real identity run in flight, there were **zero kernel failure
lines, zero refusals, zero NaNs, and normal progress**. So queue count does not
degrade a run, which removes the mechanism by which it could ever have
explained a slow lane.

So the split below classified our lanes by **an unrelated Apple system
service's activity that happened to overlap parts of the record**. That is a
confound, not a weak signal. Any number derived from it, including the
retracted queue-leak share in 1b-i, is derived from nothing. The GBDT lanes'
cost is therefore UNATTRIBUTED, and it is being re-measured rather than
re-attributed (`docs/lanes/FIXTURE_SHRINK_SCOPE.md` bucket C).

(Unrelated, since it appears throughout our Metal logs and reads alarmingly:
the `Context leak detected, CoreAnalytics returned false` lines are an Apple
diagnostic, present in committed evidence since 2026-09-10, and nothing to do
with any of this.)

### 1a. Clean-queue lanes: the time is really theirs

| lane | sec | queues first -> max |
|---|---:|---|
| byte-lm | 1995 | 23 -> 24 |
| byte-lm-resident | 1735 | 23 -> 27 |
| samba-untied-dropout-accum | 1561 | 23 -> 24 |
| samba | 955 | 23 -> 24 |
| mamba2-dtlimit | 835 | 23 -> 25 |
| transformer-window | 685 | 23 -> 24 |
| mamba2 | 597 | 23 -> 25 |
| transformer | 520 | 23 -> 24 |
| hdbscan | 471 | 23 -> 24 |
| hdbscan-leaf | 439 | 23 -> 24 |
| mamba1 | 429 | 23 -> 25 |
| mamba3 | 409 | 23 -> 24 |
| arima-seasonal-c | 231 | 23 -> 25 |
| arima-011 | 165 | 23 -> 25 |

### 1b. Leak-contaminated lanes: the time is NOT a fixture cost

| lane | sec | queues first -> max |
|---|---:|---|
| gbdt-parametric-losses | 1701 | 137 -> 341 |
| gbdt-nan-modes | 969 | 23 -> 332 |
| gbdt-lossguide-newtoncosine | 838 | 24 -> 335 |
| gbdt-pair-logit | 794 | 23 -> 352 |
| cross-val | 791 | 23 -> 449 |
| gbdt-ordered-rmse | 706 | 22 -> 344 |
| gbdt-adapter-clf | 603 | 23 -> **1809** |
| gbdt-pointwise-l2-bayesian-eval | 580 | 24 -> 657 |
| gbdt-adapter-score-weighted | 573 | 23 -> 369 |
| gpc-multiclass, umap | 525 | 40 -> 1531 |
| gbdt-categorical-ctr | 457 | 23 -> 330 |
| gbdt-adapter-reg | 279 | 1137 -> 1137 |

### 1b-i. The column, counted per lane

Counting per GROUP double-counts reruns and hides the multi-lane chunk-00
groups, so the totals here count each of the 192 lanes once, taking its
cleanest-queue measurement and splitting a multi-lane group evenly:

| | lanes | seconds |
|---|---:|---:|
| measured on the Apple column | 151 | **32,202** (8.94 h) |
| of that, clean-queue | | 19,139 |
| of that, leak-suspect | | **13,063** |
| never reached (39 `par-*`, plus iforest and iforest-tuned) | 41 | - |

Projecting the 41 unreached lanes at the measured median of 80 s (a LOWER
bound for the `par-*` ones, which ran far above median on AMD):

| full 192-lane Apple column | seconds | hours |
|---|---:|---:|
| as scoped for 0.8.6 | 35,497 | **9.86** |
| with `par-*` out of scope (section 2) | 32,362 | **8.99** |

### Keep the proportions honest, and label how each one was produced

An earlier revision of this file split the ~9.9 h column three ways and
presented all three as measurements. **One of them was an inference and it is
retracted here.** Each number below now carries its provenance.

| figure | value | how it was produced |
|---|---|---|
| hdbscan + hdbscan-leaf | 910 s, ~2.5% | **MEASURED.** Their record times (471 s, 439 s), cross-checked against direct Metal re-runs at 45.7 and 47.8 s per fixture |
| `par-*` | ~0.9 h | **ESTIMATED.** 39 lanes never run on Apple x the 80 s measured median. A lower bound, not a measurement; no `par-*` lane was ever timed on this column |
| "queue leak" | ~3.6 h | **RETRACTED. This was an inference presented as a measurement.** |

**The retraction, in full.** The 13,063 s figure is the summed wall time of
lanes whose group ran while the sampled queue count was above 60. That is an
ASSOCIATION between two observations. It is not a subtraction of observed cost
minus healthy cost, and no such subtraction was ever performed, so it never
measured time LOST. The true lost time is unknown, bounded above by 13,063 s
and plausibly far below it.

**And the cause it named does not exist.** lane/metal-queue-leak established
that device contexts are held as FIELDS ON MODEL AND POOL OBJECTS, so a
process holding many live models legitimately holds many queues. That fits the
data better than "leak" did: the groups whose counts climbed are exactly the
lanes that construct many models (`gbdt-parametric-losses` builds ten
`GradientBoosting` models, `gbdt-adapter-clf` builds them and saves and
reloads them), while `byte-lm` holds one trainer and stays flat at 23 for its
whole life. **Queue count tracks live model objects.** It is a proxy for model
count, not evidence of slowness.

What that costs: the GBDT family's time is now attributed to NOTHING, so the
"leave big" verdict that rested on it has been withdrawn (they are undecided
in `docs/lanes/FIXTURE_SHRINK_SCOPE.md` pending measurement). Their fixtures
are in fact LARGE, since they fit the full 20,000 x 16 fixture, so they may
turn out to be the best shrink target in the harness.

The lesson for this repository is its own rule, measure before attributing. A
queue count going up while a lane is slow does not make the queues the reason
the lane is slow.

**Finding: roughly half the Apple column's wall time is a runtime queue leak,
not fixture size.** Shrinking the GBDT fixtures would be treating a runtime
defect with a weaker test. Those lanes are left alone here and the cost is
referred to lane/metal-queue-leak phase 2, which is the lane that owns it.

### 1c. The expensive lanes are not big, they are chatty

The neural lanes have no large input to shrink. Read the bodies:

- `byte-lm` / `byte-lm-resident`: `ByteLanguageModelConfig()` at b2-l32-d32,
  **34,944 parameters**, three AdamW steps on three (2, 33) windows
  (`_ids(X, 3 * shape.batch, shape.length + 1)`). Six windows of 33 bytes.
- `mamba1` / `mamba2` / `mamba3` / `mamba2-dtlimit` / `transformer` /
  `transformer-window`: a **(2, 16, 32)** activation slab (`_seq(X, 2, 16, dm)`)
  at the smallest legal d_model.
- `samba`: `_ids(X, 6, 17)`, three steps on (2, 17) windows.

What costs the time is the number of device round trips per cell. Per cell,
per repeat, the batch part alone asks (`_eval_batch_rows`, `_eval_batch_prefix`,
`_prefix_lengths`): the whole batch (1) + each of `BATCH_ALONE`=16 rows alone
(16) + the uneven split 1,7,rest (3) + the whole length and prefixes 1,7,L-1
(4) = **24 calls**. Times 9 fixtures times 2 repeats = **432 batch calls per
lane**, before the train, infer and model columns. byte-lm's 1995 s over ~500
calls is about 4 s per Metal round trip on a 34,944-parameter model.

So for this family "shrink the fixture" is the wrong lever; there is no size
to remove. The lever is the call count, and the call count IS the batch
invariance claim.

**The CPU column proves this directly.** The same lane, one fixture at two
repeats, CPU host route against its per-fixture share of the Metal record:

| lane | CPU, per fixture | Metal, per fixture | ratio |
|---|---:|---:|---:|
| byte-lm | 5.2 s | ~222 s | ~43x |
| byte-lm-resident | 5.0 s | ~193 s | ~39x |
| samba | 12.8 s | ~106 s | ~8x |
| samba-untied-dropout-accum | 33.8 s | ~173 s | ~5x |
| hdbscan | 17.4 s | ~52 s | ~3x |
| hdbscan-leaf | 19.0 s | ~49 s | ~3x |
| transformer | 0.72 s | ~58 s | ~80x |
| mamba2 | 75.2 s | ~66 s | **~0.9x (CPU slower)** |

A lane whose arithmetic finishes in 5 seconds does not contain 222 seconds of
arithmetic. The gap is device round trips, and it is widest exactly where the
input is smallest. Note the ordering inverts: byte-lm is the most expensive
lane on Metal and among the CHEAPEST on a CPU.

**The clinching pair is transformer and mamba2.** Both are fed the same shape,
a `(2, 16, 32)` slab of 1024 floats, by the same helper. Their CPU-to-Metal
ratios are 80x and 0.9x. A ratio that swings by ~90x between two lanes holding
the SAME fixture cannot be a property of the fixture. (mamba2 being slower on
a CPU than on Metal says its host binding is the slow part, which is a
separate matter and not a fixture question; its input is 1024 floats and there
is nothing to take away.)

That is why byte-lm, byte-lm-resident, mamba1/2/3, mamba2-dtlimit,
transformer, transformer-window and samba are **left at their current size on
this branch**. There is no size in them to remove, and cutting their probe
call count would cut the batch-invariance claim itself (section 1d).

The two lanes with real arithmetic to remove are hdbscan and hdbscan-leaf
(6000 rows, an n^2 brute-force k-NN and mutual reachability graph) and
samba-untied-dropout-accum (96 rows of 17 over three steps at accumulation 4,
the most expensive lane of the set on a CPU). Those are the shrink candidates.

### 1d. The one knob that is free, and why it is not obviously safe

`--batch-alone N` changes how many rows are checked alone but **not** the
recorded hash: `_eval_batch_rows` folds only the whole-batch bytes into the
digest (`digest.update(f"rows:{label}:{n}")` plus `whole`), and `n` is the
held-out row count, not `alone`. The CLI says the same. So lowering it cuts
~16 of 24 calls per cell with no record invalidation.

**CORRECTION (measured, 2026-09-16). It is NOT the biggest cost lever, and an
earlier revision of this file said it was.** Counting calls is not measuring
them. Run on Metal, one fixture, two repeats:

| lane | full | `--no-batch` | the batch part |
|---|---:|---:|---:|
| byte-lm | 231.8 s | 204.8 s | **27 s, about 12%** |

So the batch probe is roughly an eighth of byte-lm, not the bulk of it, and
removing it entirely would not make the lane cheap. The ~16 of 24 calls it
accounts for are evidently much cheaper per call than the train, infer and
model work around them. The conclusion in the next paragraph is unchanged and
now rests on a measurement rather than a call count: **this knob is not worth
touching**, because it buys about 12% at the cost of a check whose negative
control cannot judge the change.

**But the shipped negative control cannot judge that change.**
`MOJOLEARN_IDENTITY_BATCH_SABOTAGE` perturbs the FIRST element of the
whole-batch answer (`_sabotage_rows` flips the low bit of the first element of
the first floating output), which row 0 alone already catches. It therefore
fails identically at `alone=16` and `alone=4`, and proves nothing about the
rows that stopped being checked. Lowering `BATCH_ALONE` on the strength of
that control would be exactly the "verification that cannot fail" trap. A
discriminating probe (perturb only row k>4) is owed before this knob moves.
**Not changed on this branch.**

### 1e. What a CPU number can and cannot say about the record

Every check on this branch ran the CPU host route, because the 0.8.6 Apple
column held the Metal lock throughout and no box was rented. The route is
sound for the question "does this cell still hold, and can it still fail":
the CPU column is the fourth column the gate diffs at `--require-columns 4`.

It is NOT a predictor of the Metal saving, and this lane does not present it
as one. The two costs are different shapes. Measured here, one fixture at two
repeats on the CPU column: hdbscan 17.4 s, hdbscan-leaf 19.0 s, samba 12.8 s.
Nine fixtures puts hdbscan near 157 s of CPU against the 471 s it took on
Metal, and the neural lanes invert the other way, because on Metal their cost
is ~24 device round trips per cell (section 1c) and on a CPU there is no round
trip to pay for.

**Superseded during the lane, for the better.** The 0.8.6 Apple record was
stopped partway and the Metal lock released, so the Metal numbers below are
MEASURED, not projected. The route is this worktree's harness driven by the
0.8.6 wheel's interpreter, which carries the Metal bindings (44 `.so`,
`vendor()` reads `metal`), taken through `mac_slot.sh metal`, one job at a
time. The harness labels such a run `vendor=arm64` (`platform.machine()`,
which is what it uses on any install that is not CPU-only); that is the
label, not the device.

**The route cross-checks against the record**, which also validates the
per-fixture arithmetic used everywhere above (record seconds / 9 fixtures):

| lane | derived from the record log | measured now on Metal | CPU host route |
|---|---:|---:|---:|
| hdbscan | 52.3 s | **45.7 s** | 17.4 s |
| hdbscan-leaf | 48.8 s | **47.8 s** | 19.0 s |
| samba-untied-dropout-accum | 173.4 s | **168.0 s** | 33.8 s |
| byte-lm | 221.7 s | **231.8 s** | 5.2 s |

Four independent measurements of the same quantity agreeing inside ~12% (three
of them inside 5%), with the CPU route a factor of 2.5 to 45 away from all of
them, is the check that the Metal column really is Metal and that dividing a
record group by 9 is a sound per-fixture unit. The column projections in
section 1b-i rest on that unit, including for the lanes that cannot be
re-measured.

The byte-lm row also confirms the framing this lane was handed: 231.8 s per
fixture over nine fixtures is **34.8 minutes** to prove that lane's numbers
equal, against the 33 minutes the record shows. The number is real. What the
CPU column proves is that **5.2 s of it is arithmetic** and the rest is the
harness talking to the device; see 1c and 1d for why that is a probe-count
problem rather than a fixture problem.

### 1f. LEFT BIG: samba-untied-dropout-accum, and the paths only the large input reaches

This was the most promising shrink candidate on paper: 33.8 s per fixture on
the CPU column, the most expensive of the set, and a visibly "workload
shaped" fixture of 96 rows of 17 over three steps. It is **left at its
current size**, because both of its dimensions turned out to be load-bearing
and each one is a path nothing else reaches.

**Rows cannot come down, measured.** The lane's docstring stakes its claim on
"four accumulation microbatches ... a split clause 9.2 admits at A = 4", and
the admitted pairs come from `training.accumulation_is_aligned`. Asked
directly:

| rows | tokens | A=4 | A=2 | A=1 |
|---:|---:|---|---|---|
| 32 | 512 | **True** | True | True |
| 16 | 256 | **False** | True | True |
| 8 | 128 | **False** | False | True |

At 16 rows A=4 is refused outright, so halving the rows does not shrink the
lane, it deletes the claim the lane exists to make. 32 rows per step is the
smallest size at which the accumulation split is admitted at all.

**Steps cannot come down, from the source.** The lane runs three steps under
`WarmupCosineLR(1e-3, warmup_steps=2, total_steps=8, min_lr=1e-5)`.
`python/mojolearn/_training_impl.py:1642` defines that as linear warmup over
`warmup_steps`, then a cosine in `p = (t - warmup) / (total - warmup)`
evaluated as exact rational arithmetic through `_cos_pi_interval` and
`_decide_f32`, "never `math.cos`". With `warmup_steps=2`, steps 1 and 2 are
the linear arm and **step 3 is the first step that evaluates the cosine at
all**. Cutting to two steps would leave the cosine branch, and the exact
rational path under it, completely unexercised.

So: 3 steps x 32 rows is the floor, and this is the truthful "this one
genuinely needs its size" result. The two paths only the full fixture
reaches are **the A=4 aligned accumulation split** and **the first cosine
step of the warmup schedule**.

## 1g. THE SHRINK, AND THE ONE GATE IT HAD TO PASS

Thirteen lanes shrunk and merged to main (f4e589395). The rule was mechanical
and per lane: cut the fixture, run the family's sabotage host build **at the
new size**, and require the cell to read DIVERGENT. An arm that went inert
meant stepping the size back up until it fired again. **No lane ships at a
size whose arm did not fire.**

The arms were confirmed live at the ORIGINAL sizes first (all eight candidate
families, DIVERGENT x8), so a firing arm at the new size is not an artifact of
the change. Runs on the CPU host route, one core, `nice -n 19`,
`MOJOLEARN_NUMERIC_MODE=identical` set explicitly so a run cannot silently
refuse before the first fit and produce nothing. Production set and sabotage
twin built from this tree for all thirteen families.

| lane | production | CPU after | sabotage arm |
|---|---|---:|---|
| `holtwinters` | STABLE | **0.64 s** | DIVERGENT |
| `spectral` | STABLE | **3.01 s** | DIVERGENT |
| `hdbscan-leaf` | STABLE | **3.47 s** | DIVERGENT |
| `byte-lm-resident` | STABLE | **3.68 s** | DIVERGENT |
| `byte-lm` | STABLE | **4.22 s** | DIVERGENT |
| `gbdt-lossguide-newtoncosine` | STABLE | **6.44 s** | DIVERGENT |
| `gbdt-parametric-losses` | STABLE | **7.16 s** | DIVERGENT |
| `hdbscan` | STABLE | **8.65 s** | DIVERGENT |
| `samba` | STABLE | **10.75 s** | DIVERGENT |
| `gbdt-pair-logit` | STABLE | **11.61 s** | DIVERGENT |
| `samba-untied-dropout-accum` | STABLE | **17.77 s** | DIVERGENT |
| `gbdt-nan-modes` | STABLE | **18.61 s** | DIVERGENT |
| `mamba2-dtlimit` | STABLE | **79.15 s** | DIVERGENT |

**Twelve of thirteen are under 30 s. One is not, and it is worth naming.**
`mamba2-dtlimit` at 79.15 s barely moved: the untouched `mamba2` lane costs
75.19 s on the same route, so cutting its sequence length from 16 to 8 bought
almost nothing. Its input was already 1024 floats; what is slow is the mamba
host binding itself, not the fixture. That is a lane where the honest answer
is that shrinking the fixture is the wrong tool, and the cost belongs to
whoever owns that binding.

### What this does NOT claim

These are CPU-host-route seconds for ONE fixture at two repeats. They are the
right numbers for "can this still fail and how cheap is the check", and they
are NOT the Apple column's seconds: the record runs nine fixtures on Metal,
where the cost profile is different by up to 45x per lane in either direction
(section 1c). The Metal before/after is measured separately and reported with
the column estimate.

## 1h. THE METAL NUMBERS, WHICH ARE MUCH LESS IMPRESSIVE THAN THE CPU ONES

The CPU table in 1g is the wrong number to quote at anyone asking what the
Apple column costs, and the difference is not small. Measured on Metal, one
fixture at two repeats, same route before and after:

| lane | Metal before | Metal after | change |
|---|---:|---:|---|
| hdbscan | 45.7 s | **32.4 s** | 1.4x |
| hdbscan-leaf | 47.8 s | **20.7 s** | 2.3x |
| samba-untied-dropout-accum | 168.0 s | **135.5 s** | 1.2x |
| byte-lm | 231.8 s | **242.0 s** | **none** |

Against the record's per-fixture share (record seconds / 9), for the lanes
with no same-route before:

| lane | record share | Metal after | change |
|---|---:|---:|---|
| gbdt-parametric-losses | 189.0 s | **81.8 s** | 2.3x |
| gbdt-nan-modes | 107.7 s | **76.0 s** | 1.4x |
| gbdt-lossguide-newtoncosine | 93.1 s | **64.7 s** | 1.4x |
| gbdt-pair-logit | 88.2 s | **58.4 s** | 1.5x |
| mamba2-dtlimit | 92.8 s | **90.5 s** | none |
| byte-lm-resident | 192.8 s | **170.7 s** | 1.1x |
| samba | 106.1 s | **124.3 s** | **worse** |

**byte-lm cutting three AdamW steps to one changed nothing on Metal** (231.8 s
to 242.0 s, inside noise), and samba got slower. That is the section 1c result
holding up under the intervention: those lanes' Metal cost is per-operation
device overhead on a tiny model, so removing arithmetic removes nothing. The
lanes that did move, GBDT and hdbscan, are exactly the ones that were fitting
a genuinely large input.

**Honest headline: the shrink takes roughly 20% off the Metal column, not a
transformation.** The transformation is on the CPU column, where the same
thirteen lanes now finish in 0.64 s to 18.6 s (one exception, 1g), which is
where routine verification belongs anyway.

### The caveat on every Metal number above

These runs pair THIS tree's harness with the 0.8.6 wheel's interpreter, which
is what supplies the Metal bindings. Where a lane uses an API added after the
freeze, the pairing breaks: `spectral` read REFUSED with
`SpectralClustering.__init__() got an unexpected keyword argument
'prediction_data'` (DEVIATION 2860, merged after 0.8.6). **That is the
measurement route, not the shrink**: spectral is STABLE on the CPU column with
this tree's package and its sabotage arm fires. holtwinters' 1.58 s is
similarly a same-route figure with no comparable before, since it and spectral
were recorded as a single two-lane group (1817 s for the pair).

## 1i. THE DURABLE GUARD: a moved fixture cannot be compared to a stale reference

Regenerating `verify_reference/table.json` fixes today. It does not stop the
next person shrinking a fixture and forgetting, so the check matters more than
the regeneration.

`build_table` now records `lane_revisions`, the harness revisions the table was
generated against, and `stale_reference_lanes(table, harness)` returns every
lane that still has cells in the table whose recorded revision is not the
harness's. A table generated before the key existed carries no revision, which
is not evidence of being current, so it reads stale. `_verify_all` drops those
lanes from the comparison, names them, and refuses outright if that leaves
nothing to check.

**Watched it fail first, which is the only reason to trust it:**

| case | result |
|---|---|
| today's shipped table | **13 STALE**, exactly the shrunk thirteen |
| a table carrying the current revisions | `[]` |
| a table with no cells for those lanes | `[]` (nothing to compare, not stale) |
| one revision deliberately corrupted | `['hdbscan']` |
| a selection mixing stale and clean lanes | stale dropped, clean kept, named in the log |
| a selection where every lane is stale | refuses outright |

### The table itself is NOT regenerated here, deliberately

A regeneration today would not be my lane's change. The ref-level diff against
a regenerated table shows 444 references lost (all thirteen shrunk lanes, none
outside), 332 gained (eleven lanes other lanes merged since the table was last
built), and **72 reference values changed across ten lanes this branch never
touched** (metrics, spectral-precomputed, radius, holtwinters-multiplicative,
the gmm-sample pair and others), because newer committed records now win over
older ones. Carrying ten other lanes' reference changes inside this commit
would be wrong, and the release regenerates the table as a normal step.

So the thirteen references are **stale by design until the next release
record**, and the guard above is what makes that safe: nothing compares
against them in the meantime.

<!-- TASK1-RESULTS -->

## 2. `par-*` leaves the release record's scope

### Where the scope actually lives

There is **no in-repo list of the release record's lanes**. The record legs
run the harness with every lane it defines, minus a `--skip` string computed
per leg from the lanes already recorded:

    ~/mojolearn-evidence/release-0.8.6/scripts/record_body.template.sh:34
      timeout -k 30 @IDSECS@ .../identity_break.py --vendor @LABEL@ \
        --json "$OUT/identity_break.@LABEL@.json" --skip '@SKIP@'

    ~/mojolearn-evidence/release-0.8.6/scripts/make_record_body.sh
      SKIP=$(python3 -c "... {k.split('/')[0] for k in j['cells']} ...")

`--skip` is resume bookkeeping, not scope. So `tools/identity_break.py`'s
`LANES` dict has been the record's scope by default. The scope is now stated
explicitly in that file, with the reasons beside it:

    RECORD_EXCLUDED_PREFIXES = ("par-",)   # the declaration
    record_excluded_lanes()                # 39 lanes, derived from LANES
    record_lanes()                         # the 160 a record runs

and enforced in `run()`: a FULL-COLUMN run (no `--lanes`) drops them and
prints `# OUT OF RECORD SCOPE (39 lanes): ...`. `--lanes` is never filtered,
so the two-device `par` legs and the CPU identity gate, which both pass
`--lanes` explicitly, are unaffected.

### What is excluded and why

All 39 `par-*` lanes. The reasons, in the file and here:

1. **Cost out of all proportion to what they prove.** They were the whole
   remaining AMD gap for 0.8.6 (28 of the 30 owed lanes) and the AMD legs
   never reached them inside a 60-minute lease. See section 3.
2. **Several cannot be covered honestly at all.**
   `docs/lanes/LANE_STATUS_lane-cpu-training-par-wave3.md` already established
   this: `par-byte-lm-model-pool` and `par-byte-lm-offload` compare host
   arithmetic with itself under a pooled label, and 21 more are cooperative
   families whose split lives inside the GPU binding with no host restatement.
3. **In the record they are not even the two-device claim.** `_par_devices()`
   defaults to `MOJOLEARN_PAR_DEVICES=0`, so every `par-*` cell in a release
   record is a ONE-device run. The drivers' actual claim, that two devices
   hash equal to one, is made by the dedicated two-device legs
   (`bench/results/identity_break/2026-09-15_par-lanes-new/`, `.../multi_gpu/`),
   not by the release record. The record was paying GPU-hours for the weaker
   half of the claim.

### The consequence, written down rather than discovered later

All 11 CPU-covered `par-*` lanes (`par-scaler`, `par-arima`, `par-holtwinters`,
`par-queries-knn`, `par-queries-radius`, `par-queries-kde`,
`par-reference-knn`, `par-reference-knn-reg`, `par-forest`, `par-forest-et`,
`par-mlp`) are in `host_surface.record_covered_lanes()`, which the CPU
identity gate diffs against `TRAINING_GPU_COLUMNS` at `--require-columns 4`.
Today that points at the 2026-09-14 166-lane record, which carries all of
them, so **this change does not move the gate today**. The day
`TRAINING_GPU_COLUMNS` is repointed at a record taken under the new scope,
those 11 lanes lose their GPU columns and the gate fails unless they are
dropped from the covered set or admitted as OWED. That is a decision for the
lane that repoints the columns, and it is named here so it is not a surprise.

## 3. The AMD position for 0.8.6

**AMD stopped at 162 of 192 lanes.** NVIDIA finished 192 of 192. The AMD
column is `~/mojolearn-evidence/release-0.8.6/records/amd-mi300x-gfx942.json`,
1458 cells: train 1457 STABLE and **1 MOVED**, infer 1296 STABLE / 162 N/A,
model 1080 STABLE / 378 N/A, batch 1332 STABLE / 126 N/A.

### The 30 owed lanes

28 `par-*` plus `iforest` and `iforest-tuned`:

    par-arima par-mlp par-samba par-byte-lm par-queries-knn par-queries-radius
    par-queries-kde par-reference-knn par-reference-knn-reg
    par-graph-agglomerative par-graph-spectral par-graph-umap par-ordered-rmse
    par-feature-freq par-boosting-pointwise par-holtwinters
    par-byte-lm-model-pool par-byte-lm-offload par-samba-clip iforest
    iforest-tuned par-iforest par-forest-pool par-gmm par-resample par-hdbscan
    par-cholesky par-kernel-ridge par-nystroem par-rbf-sampler

**22 of the 30 were already recorded on AMD** in the committed 166-lane record
(`bench/results/identity_break/2026-09-14_166-lanes/amd-mi325x-gfx942.json`,
which carries 31 `par-*` lanes including `par-arima` and `iforest`). Only
**8 have never been recorded on any AMD column**: `par-forest-pool`,
`par-gmm`, `par-resample`, `par-hdbscan`, `par-cholesky`, `par-kernel-ridge`,
`par-nystroem`, `par-rbf-sampler`. So this is a lease-budget gap, not an
impossibility, and it must not be written up as one.

**The 30 owed lanes are a contiguous tail, positions 163 to 192 of 192.** The
last lane AMD ever recorded is `par-scaler` at 162; the first it never
recorded is `par-arima` at 163. The column did not trail off across a
scattered set of expensive lanes, it stopped dead at one lane boundary and
never moved again across two further legs.

That reframes `iforest` (182) and `iforest-tuned` (183): they are not lanes
that were "cut off last", they are two cheap 16-estimator isolation-forest
fits sitting BEHIND `par-arima` in the queue that were never reached. Once
`par-*` leaves the record scope (section 2) they stop being blocked, and
AMD's remaining gap for the next record is those two lanes alone.

### The legs

**Five** AMD legs, all Hot Aisle, all `minutes=60`, 299 cents/hour. Legs 1 and
2 were 8core, legs 3 to 5 13core. Leg evidence for 1, 3, 4, 5 is under
`~/mojolearn-evidence/release-0.8.6/records/`; all five are also under
`bench/results/identity_break/2026-09-16_release-0.8.6/` in the
`fix/release-post-record-allowlist` worktree (read only from there, it is
another lane's branch).

| leg | started (UTC) | result |
|---|---|---|
| 1 | 2026-09-15T23:13:47Z | 62 lanes, `identity_break_exit=124` (2400 s bound) |
| 2 | 2026-09-16T00:00:14Z | **failed, 0 lanes.** R2 staging succeeded (4 keys, 50 s) but `remote/` is empty, `remote_console.log` is empty and teardown read `exit=1`. Why it failed belongs to the release lane, not here |
| 3 | 2026-09-16T01:06:33Z | `exit=124`, 693,237 B of partials; `rf-score-weighted/wide` read MOVED here |
| 4 | 2026-09-16T01:53:15Z | log ends `Killed` just after `par-scaler` (lane 162) |
| 5 | 2026-09-16T02:51:06Z | **0 lanes** - see below |

So of five 60-minute leases, one recorded 62 lanes, two recorded the rest up
to lane 162, and **two produced nothing at all**. The column has been stuck at
the `par-scaler` / `par-arima` boundary since leg 4.

### PROVIDER LIMIT: par-arima is a LEASE-BUDGET WALL on one provider, not an impossibility

This is a finding about the provider and the lane, **not a gap in our work**.

**Say what it is not, first.** This is NOT a claim that `par-arima` cannot be
recorded, or that it is slow on AMD in general. It **was** recorded on an AMD
MI325X in the committed 166-lane record
(`bench/results/identity_break/2026-09-14_166-lanes/amd-mi325x-gfx942.json`),
along with 30 other `par-*` lanes. The lane is recordable and has been
recorded. Nor is it claimed that it would fail at 90 minutes; no such lease
was ever bought, so that is untested.

What IS measured is narrower and it is about one provider's lease budget: on
Hot Aisle, whose runner caps a lease at 60 minutes in enforced code, a 2400 s
identity process on a healthy MI300X box, asked for this lane FIRST, emitted
nothing at all. Within that cap the lane could not be got on to the 0.8.6
column, and a longer single lease is not purchasable there. Money does not
solve it on that provider; a different provider, a longer bound, or a cheaper
driver would.

AMD leg 5 was pointed at exactly the 30 owed lanes, `par-arima` first
(`extra_body.sh`, a `--skip` of the 162 already recorded). The box was
healthy and said so:

    02:54:04Z wheel_fetch_exit=0      02:54:12Z pip_exit=0
    02:54:12Z import_exit=0 0.8.6 hip 02:54:14Z identity_check_exit=0
    02:54:22Z verify_quick_exit=0     02:54:22Z host_surface ... bindings 15

The identity process then ran its full bound and died on the timeout:

    02:56:22Z partial_uploaded bytes=      (every 2 minutes, always empty)
    ...
    03:34:23Z partial_uploaded bytes=
    03:34:26Z identity_break_exit=124

`timeout -k 30 2400` - **2400 seconds, 30 lanes asked for, zero lanes
produced, a 0-byte `identity_break.log`.** The harness writes its JSON after
every lane, and all twenty partial uploads were empty, so no lane completed.

A longer single lease is not purchasable. The runner refuses it in enforced
code:

    tools/hotaisle_leg.sh:319-321
      if [ "$MINUTES" -gt 60 ]; then
        echo "--minutes $MINUTES REFUSED: 60 is the maximum lease (a second leg, never an extension)"

So within the provider's hard cap, `par-arima` as currently written and driven
cannot be recorded on Hot Aisle at any price. **What is NOT claimed:** that
par-arima would fail at 90 minutes, or that it is slow on AMD in general - it
recorded fine on MI325X in the 166-lane record. What is measured is that a
2400 s process on a healthy MI300X box, asked for it first, emitted nothing.

### Open, owed to a box

- `rf-score-weighted/wide` reads **MOVED** on AMD MI300X, hashes
  `49be8ea935a47640` then `50ce4a9f62cddf8e`. Printed from every column
  first, as the rule requires: Apple M4 (group34) and NVIDIA H100 both read
  **STABLE at `49be8ea935a47640`**, the same hash as AMD's first repeat. So
  AMD is the column standing alone and it disagrees with ITSELF run to run;
  this is box nondeterminism, not a cross-vendor divergence. Unresolved, needs
  an AMD box.
- The 8 `par-*` lanes never recorded on any AMD column stay owed, and under
  section 2 they are owed to a two-device `par` leg rather than to a release
  record.

## Rules this lane ran under

One core (`nice -n 19`, thread knobs 1, one process at a time), own worktree,
never the shared checkout. No box rented, GPU or CPU. No Metal job: the 0.8.6
Apple column held the Metal lock the whole time, and every check here ran the
CPU host route, which is bitwise equal to Metal. Host bindings were built into
a scratch directory, never into the shared checkout.

## Resume, for a session holding none of this context

    cd /Users/andrewhendel/CascadeProjects/mojolearn      # SHARED: never build or commit here
    git worktree add -b lane/identity-fixtures-light <scratch>/wt-light main
    cd <scratch>/wt-light

Build the CPU host set (about 40 to 90 s each, one core):

    for f in core estimators training mamba transformer hdbscan; do
      MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_HOST_OUTDIR=<scratch>/hostbuild \
        nice -n 19 sh bindings/build_${f}_host.sh
    done
    MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_BYTE_LM_HOST_OUTDIR=<scratch>/hostbuild \
      nice -n 19 sh bindings/build_byte_lm_host.sh

The sabotage twin of the same set (the negative control):

    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" \
      MOJOLEARN_HOST_OUTDIR=<scratch>/hostbuild-sab ... (same loop)

Run a lane on the CPU column (NEVER take the Metal lock while a record runs):

    export PYTHONPATH=$PWD/python MOJOLEARN_NUMERIC_MODE=identical
    MOJOLEARN_HOST_DIR=<scratch>/hostbuild \
      nice -n 19 python3 tools/identity_break.py --lanes <lane> --fixtures base \
        --repeats 2 --json prod.json
    # the harness header MUST read host.column=cpu; if it does not, stop.

The control, which must fire before any fixture is trusted:

    MOJOLEARN_HOST_DIR=<scratch>/hostbuild-sab MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
      nice -n 19 python3 tools/identity_break.py --lanes <lane> --fixtures base \
        --repeats 2 --json sab.json
    python3 tools/identity_break.py --diff prod.json sab.json   # must read DIVERGENT

Merge gates, all three, before any push to main:

    python3 tools/docs_facts.py --check
    python3 packaging/wheel_ci.py pins .
    python3 packaging/wheel_ci.py inventory python/mojolearn

## Pods

None rented on this branch.
