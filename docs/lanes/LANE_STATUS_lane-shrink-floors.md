# LANE STATUS: shrink-floors (2026-09-16)

Two jobs, and the second one matters more.

1. **Reverse the two shrinks that made a cell unable to fail**, `byte-lm` and
   `samba-untied-dropout-accum`, each proved from the FAILING side first.
2. **Make a shrink floor something a change has to walk past**, because the
   second reversal is not an oversight. A floor for that lane was written
   down, a later lane did not honour it, and a third document then recorded
   the opposite. Nobody lied. The floor had no mechanism.

Branch `lane/shrink-floors`, cut from `main` at `16b29b9cf`. No box rented.
Every Metal job took the shared lock through `mac_slot.sh metal`, one at a
time, one core, `nice -n 19`, `MOJOLEARN_NUMERIC_MODE=identical`. The 0.8.6
macOS wheel supplies the bindings in a private venv under this session's
scratchpad, so no other agent could replace a `.so` under a queued job; the
sha256 of every mojolearn binding the process had MAPPED was printed from
inside each process and is identical in both runs (for example
`_mojolearn_byte_lm.so b7a8c87228a6693f`, `_mojolearn_training.so
e98663609bad1b23`, `_mojolearn_mamba.so 207408a0ba597c13`).

## 1. THE HEADLINE

**A floor in prose is not a floor.** Floors now live in the code, on the lane,
in the same expression as the reason, and `tools/fixture_floors.py` refuses a
change that walks past one. It has been watched to refuse five different
violations, including the exact one that happened on 2026-09-16.

**Both blind cells are reversed**, `byte-lm` to two steps and
`samba-untied-dropout-accum` to three, each measured BLIND at the shipped size
and DETECTED after, against the real lane function rather than a sketch of it.

**The other twelve shrinks were right**, and that is measured, not assumed.

## 2. THE TWO REVERSALS, from the failing side

One probe, `probe_reversals.py`, run twice: once against a FROZEN copy of the
harness at `16b29b9cf` (the shipped, unfixed side) and once against this
branch. Nothing else differs between the two runs.

### 2a. `byte-lm`, one step to two

`_byte_lm_params` sets every `norm1_w` and every `norm2_w` to a vector of
ONES, so at the input of the only hashed step the two are bitwise EQUAL and a
read-side exchange of them is the identity function.

UNFIXED SIDE (`for k in range(1)`):

```
=== byte-lm, worktree source has 1 step(s) ===
  block0.norm1_w == block0.norm2_w bitwise at step-1 INPUT ? True
  block0.norm1_w == block0.norm2_w bitwise AFTER step 1    ? False   max|diff| 2.000e-03
  block1.norm1_w == block1.norm2_w bitwise at step-1 INPUT ? True
  block1.norm1_w == block1.norm2_w bitwise AFTER step 1    ? False   max|diff| 2.000e-03
  CONTROL state_dict round trip, no swap: 63bd514c0e71982a -> 63bd514c0e71982a  INERT (good)
  CONTROL reconstruction vs the REAL lane: 63bd514c0e71982a vs 63bd514c0e71982a  SAME CELL
  BLIND    norm1_w/norm2_w EXCHANGED at step 1, 1-step cell: 63bd514c0e71982a -> 63bd514c0e71982a
  DETECTED norm1_w/norm2_w EXCHANGED at step 2, 2-step cell: 4e8dfbf5daf0e950 -> e8e6e62e0144f33e
  PRODUCTION byte-lm: fit1 63bd514c0e71982a  fit2 63bd514c0e71982a  -> STABLE   3.45s + 3.26s
```

Read the four weight lines together: the pair is equal where the kernel reads
it at step 1 and apart by `2.000e-03` after it, so **the lane had exactly one
step and therefore exactly zero opportunities to see the exchange**.

**A REAL SABOTAGE BUILD ALREADY SHOWED THIS, and nobody acted on it.**
`docs/lanes/LANE_STATUS_lane-metal-launch-overhead.md` section 4.1 records a
build of `training/byte_lm.mojo` whose block copy actually EXCHANGED `norm1_w`
and `norm2_w`. Against it, `byte-lm/base` read **IDENTICAL x2** while
`byte-lm-resident` diverged, and the paragraph beside that result says in so
many words that "the shrunken one-step cell is blind to this whole defect
class". That is a compiled sabotage, not a data perturbation, so it is
stronger evidence than anything in this lane, and it sat on main. It is now
cited and marked fixed there. The floor is what turns that observation into
something the next change has to walk past.

**Two controls make the arm worth reading**, and both are new here. The
exchange is applied through `state_dict()` / `load_state_dict()` on ONE
trainer, so the AdamW moments and the step count carry exactly as they do in
the lane; the same round trip with NO swap is INERT, so the arm is not
measuring the round trip. And the probe's reconstruction hashes the SAME cell
as the real `LANES["byte-lm"]` function, so the verdicts are about the shipped
cell rather than a lookalike. (The audit that found this used a fresh trainer
per step, which restarts the optimizer state; that is why its two-step hashes,
`c16f90ad63967074` and `d22def89a7bb626f`, differ from the ones above. The
one-step hash, where the two constructions coincide, agrees exactly.)

### 2b. `samba-untied-dropout-accum`, one step to three

`_Schedule._progress` returns the LINEAR warmup value whenever
`t <= warmup_steps`, and this lane runs `WarmupCosineLR(1e-3, warmup_steps=2,
total_steps=8, min_lr=1e-5)`. So the cosine decay, and the exact rational
`_cos_pi_interval` / `_decide_f32` path under it, are first evaluated at step
3. This arm is applied to the REAL lane function, by replacing the schedule
class it constructs.

UNFIXED SIDE (`for k in range(1)`):

```
=== samba-untied-dropout-accum, worktree source has 1 step(s) ===
  WarmupCosineLR.lr_at(1) = 0.000500000024   (t <= warmup_steps: the LINEAR arm, cosine never evaluated)
  WarmupCosineLR.lr_at(2) = 0.00100000005    (t <= warmup_steps: the LINEAR arm, cosine never evaluated)
  WarmupCosineLR.lr_at(3) = 0.000933682604   (the COSINE arm)
  baseline cell f5297e8abd4a4c51
  BLIND    WarmupCosineLR -> WarmupLinearLR (same peak/warmup/total/min): f5297e8abd4a4c51 -> f5297e8abd4a4c51
  BLIND    WarmupCosineLR -> ConstantLR(warmup 2):                       f5297e8abd4a4c51 -> f5297e8abd4a4c51
  PRODUCTION samba-untied-dropout-accum: fit1 f5297e8abd4a4c51  fit2 f5297e8abd4a4c51  -> STABLE   9.10s + 8.38s
```

The cell could not tell a cosine schedule from a linear one, or from a
constant one.

### 2c. FIXED SIDE: both arms fire, and one prediction was checked

Same probe, same process shape, this branch:

```
=== byte-lm, worktree source has 2 step(s) ===
  CONTROL state_dict round trip, no swap: 4e8dfbf5daf0e950 -> 4e8dfbf5daf0e950  INERT (good)
  CONTROL reconstruction vs the REAL lane: 4e8dfbf5daf0e950 vs 4e8dfbf5daf0e950  SAME CELL
  BLIND    norm1_w/norm2_w EXCHANGED at step 1, 1-step cell: 63bd514c0e71982a -> 63bd514c0e71982a
  DETECTED norm1_w/norm2_w EXCHANGED at step 2, 2-step cell: 4e8dfbf5daf0e950 -> e8e6e62e0144f33e
  PRODUCTION byte-lm: fit1 4e8dfbf5daf0e950  fit2 4e8dfbf5daf0e950  -> STABLE   4.60s + 4.35s

=== samba-untied-dropout-accum, worktree source has 3 step(s) ===
  baseline cell 4610c84719bc1a25
  DETECTED WarmupCosineLR -> WarmupLinearLR (same peak/warmup/total/min): 4610c84719bc1a25 -> ca7ab47450d8081c
  DETECTED WarmupCosineLR -> ConstantLR(warmup 2):                       4610c84719bc1a25 -> 5f64a675ec9f8924
  PRODUCTION samba-untied-dropout-accum: fit1 4610c84719bc1a25  fit2 4610c84719bc1a25  -> STABLE   20.85s + 17.30s
```

**One prediction, made before the edit and checked after.** On the UNFIXED
side the probe's two-step arm hashed `4e8dfbf5daf0e950`. That arm was a
reconstruction; the REAL lane did not run two steps yet. After the reversal,
`LANES["byte-lm"]` hashes `4e8dfbf5daf0e950` exactly. The reconstruction the
verdicts rest on was the lane.

**Production is STABLE on both**, two fits in one process agreeing bit for
bit, which is the harness's own MOVED test.

**The new cells** are `byte-lm` `4e8dfbf5daf0e950` and
`samba-untied-dropout-accum` `4610c84719bc1a25` on the `base` fixture, Apple
Metal, 0.8.6 bindings. Both replace the cells every committed column carries.

### 2c. SEQUENCING: now, not deferred

Both reversals bump `LANE_REVISIONS` (`byte-lm` to `steps-2-1`,
`samba-untied-dropout-accum` to `steps-3-1`) and therefore DROP those lanes'
references: every committed column hashed the one-step input, so those cells
read OWED to the next record rather than DIVERGENT against different bytes.

Doing it today costs nothing extra. `release/0.8.7` already requires all four
record columns to be retaken, so the reference these reversals drop was going
to be regenerated anyway. Deferring would have cost a whole re-record. This is
the same reasoning applied to the UMAP fixes, and it is written here so it can
be reversed if the release plan changes.

### 2d. THE REFACTOR MOVED NOTHING ELSE, checked rather than argued

Thirteen lanes had a fixture size rewritten from a literal into a floored
local (`X[:1500]` became `rows = 1500; X[:rows]`). Eleven of those are meant
to be a RENAME. Reading the diff and declaring it obvious is a check that
cannot fail, so it was measured two ways, both CPU only and both derived from
the two sources rather than typed out.

**Every size site in the file, both sources, resolved the way the gate
resolves it** (`eq_sites.py`): the before source through literals, this branch
through its floored locals, across all 158 lanes and 146 sites.

```
  byte-lm                          steps      before [1]      after [2]      MOVED  ok
  samba-untied-dropout-accum       steps      before [1]      after [3]      MOVED  ok
  samba-untied-dropout-accum       batch      before [32]     after [96]     MOVED  ok
  ... 143 more sites, all `same`
ok: only the two reversals moved a number
```

(`samba-untied-dropout-accum batch` reads 96 after because the site is
`_ids(X, steps * batch, 17)`, three steps of 32 rows; the floored local is
still 32, which is what the floor holds.)

**And the arrays those sites build are byte-identical** (`eq.py`), on three
fixtures rather than one, `base`, `ties` and `odd`: the gbdt row slices, both
hdbscan slices, spectral, holtwinters, both mamba2 slabs, samba's windows,
byte-lm's one-step ids and the tokenizer's 4096 bytes all hash the same on
both sources, while the two reversals' ids are DIFFERENT and of the expected
new shape. Neither check needs a binding, so neither is a GPU cost.

## 3. THE FLOOR MECHANISM

### 3a. What went wrong, exactly

`docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md:306` is titled
"LEFT BIG: samba-untied-dropout-accum" and records TWO floors for that lane,
32 rows and 3 steps, with the reason for each. The next lane cut it to one
step. `docs/lanes/FIXTURE_SHRINK_SCOPE.md` table A carried forward only the
ROWS half ("32 rows KEPT because `accumulation_is_aligned`"), so the steps read
like the free dimension. `docs/lanes/LANE_STATUS_lane-neural-shape-shrink.md`
then stated as FACT that the lane "keeps ... the third step because it is the
first that evaluates the cosine arm". It did not.

Three documents, no bad faith, and a floor that nothing held anyone to.

### 3b. The shape chosen

A floor is declared ON the lane, directly above the body it constrains, with
the number and the reason in one expression:

    @lane("samba-untied-dropout-accum")
    @floor(steps=(3, "step 3 is the FIRST step that evaluates the cosine arm at
                      all ... measured blind at one step and detected at three
                      (2026-09-16, docs/lanes/LANE_STATUS_shrink-blindness-audit.md
                      section 3; the same floor was written in prose in
                      ...section 1f BEFORE the cut and did not stop it, which is
                      why it is code now)"),
           batch=(32, "32 rows per step is T = 512 tokens, the smallest size at
                       which `training.accumulation_is_aligned` admits the A = 4
                       split this lane exists to claim ..."))
    def _(ml, X, yc, yr, Xh=None):
        steps, batch = 3, 32          # FLOORED, see @floor above
        ids = _ids(X, steps * batch, 17)
        losses = [... for k in range(steps)]

`tools/fixture_floors.py` enforces it from the SOURCE with `ast` alone, no
numpy and no bindings, so it runs in `light-checks` (the only workflow that
starts by itself) and again as two tests in the CPU gate
(`python/mojolearn/tests/test_host_surface.py`).

### 3c. The four rules, and which ones are derived

| | rule | derived, or declared |
|---|---|---|
| 1 | A revision key that NAMES a size must carry a floor on that dimension, and the key's number must be the number the lane runs at | **DERIVED** from `LANE_REVISIONS`, the place a fixture change has to be recorded |
| 2 | The value is read from the LANE BODY, from a local named for the dimension; below the floor is a refusal that PRINTS THE REASON | derived from the body |
| 3 | The site must READ that local, so `steps = 3` next to `range(1)` is refused by name | derived from the body |
| 4 | The reason must be at least 60 characters and cite a date, a `docs/` path or a `lane/` name | declared, checked |

**Why rule 1 is derived and not a list.** This repository has been bitten by
hand-kept lists that rot into agreeing with everything: a `stale reference`
check that asked `lane in LANE_REVISIONS` was true forever and silently barred
lanes from the public set permanently. So the scope here is not a list of
lanes that need floors. It is `LANE_REVISIONS`, where a fixture change already
HAS to be recorded (without an entry, every committed column reads DIVERGENT,
which is loud), and specifically the keys that already NAME a size in their
first token: `rows-1500-1`, `obs-128-1`, `steps-3-1`, `seqlen-8-1`. Shrink a
fixture, record it the way the harness already requires, and the lane is in
scope for a floor the same moment.

**The key must agree with the body**, which is a tie the repository did not
have. `LANE_REVISIONS["hdbscan"] = "rows-4000-1"` and `rows = 4000` in the
lane are now checked against each other, so a key that says one size while the
body runs another is refused by name. That is the shape of the failure this
lane exists to stop, in the record rather than in a document.

**The one declaration, and how it fails closed.** `LANE_REVISIONS` is broader
than "was shrunk": it also carries arithmetic changes. Those keys name no size
and must say what they DID change in `NON_SIZE_REVISIONS`, and an entry there
is REFUSED if its key does name a size. Four lanes are in it: `tokenizer` (a
vocabulary swap), `byte-lm-resident` (a model shape), and `umap` and
`par-graph-umap` (a row-separable `transform`).

**This rule was rewritten because the gate caught its first version.** The
original scope was "every dimension a site rule can find in a shrunk lane",
and when `lane/umap-batch-determinism` merged into this branch the check
immediately refused `umap` and `par-graph-umap` for having no `rows` floor.
They had not been shrunk at all; their revision records an ARITHMETIC change.
A rule that demands a floor with no measurement behind it produces boilerplate
reasons, which is the rot this whole mechanism is against. Keying off what the
revision key already says is both narrower and stronger, because it added the
key-versus-body check above.

**What it deliberately does not do.** It does not judge whether a floor is the
RIGHT number; only a measurement does that, which is why rule 4 forces every
reason to cite one. And a dimension whose site expression is not a fixture
size this file can read (`_ids(X, steps * shape.batch, ...)`, where the batch
comes from the model config) requires no floor, which is a real hole: a future
lane could dodge by making a site unreadable. That is recorded here rather
than papered over.

### 3d. THE REFUSALS, WATCHED

A floor mechanism that has not been seen to refuse anything is the prose floor
it replaces. `tools/fixture_floors.py --self-test` mutates the REAL harness
source SEVEN ways and requires each to be refused BY NAME, and adds two more
rules that cannot be reached by a one-line mutation on a source written to
break exactly them. It runs in the gate BEFORE the check itself, so a check
that has stopped being able to refuse fails loudly rather than passing
quietly, and an anchor that stops matching is reported as REFUSE-TEST BROKEN
rather than passing as a clean mutation.

```
# unmutated tools/identity_break.py: 0 violation(s)

# MUTATION: a floored lane is cut below its floor
    REFUSED samba-untied-dropout-accum: REFUSED, steps = 1 is BELOW the floor of 3.
          WHY THE FLOOR IS THERE: step 3 is the FIRST step that evaluates the cosine arm at all...
          If the floor is wrong, change it HERE with the measurement that says so; do not cut past it.

# MUTATION: the floor decorator is deleted
    REFUSED byte-lm: SHRUNK (LANE_REVISIONS['byte-lm'] = 'steps-2-1') and has a `range(...)` loop or
          comprehension whose body calls `train_step`, but declares no floor for it...

# MUTATION: the site stops reading the floored local, leaving the floor looking satisfied
    REFUSED mamba2-dtlimit: the length argument of `_seq(X, b, l, ...)` does not read the floored
          local `seqlen`; it is `4`. A literal at the site means the fixture can be cut without
          touching `seqlen = 8` or the reason above it.

# MUTATION: a reason nothing can be traced back to
    REFUSED mamba2-dtlimit: floor(seqlen=...): the reason is not traceable. It must be at least 60
          characters and cite a date, a docs/ path or a lane/ name...

# SYNTHETIC: a @floor() that is not on a lane function
    REFUSED line 6: a @floor() that is not on an @lane() function. Nothing enforces it, and it
          reads as though something does.

# SYNTHETIC: a reason copied word for word from another floor
    REFUSED b: the reason for rows is copied word for word from a's rows floor. A floor's reason
          is about THIS lane's fixture; if the measurement really is shared, say which lane it
          was taken on.

# MUTATION: the revision key says one size and the body runs another
    REFUSED hdbscan: LANE_REVISIONS['hdbscan'] = 'rows-3000-1' says rows 3000, but the lane runs
          at rows = 4000. A revision key that disagrees with the body is how a record ends up
          describing bytes nobody made.

# MUTATION: a revision key that names no size at all, undeclared
    REFUSED spectral: LANE_REVISIONS['spectral'] = 'made-it-smaller-1' names no size this file
          knows (batch, obs, rows, seqlen, steps). Either name one, so the floor and the record
          agree, or say in NON_SIZE_REVISIONS what changed instead.

# MUTATION: the non-size list used on a revision that DOES name a size
    REFUSED NON_SIZE_REVISIONS['holtwinters']: REFUSED, its revision 'obs-128-1' DOES name a size
          (observations 128). Declare the floor instead of the exemption.

ok: the floor check refuses every violation above
```

The FIRST mutation is the 2026-09-16 failure itself, replayed: cut
`samba-untied-dropout-accum` to one step and the gate now hands back the
paragraph that says why three.

## 4. THE TWELVE THAT ARE FINE

This is as much the result as the two failures are, and it is measured rather
than assumed. Every row and observation cut cost NO detection, and where a
resolution ladder was run the SMALLER fixture was as sharp or SHARPER than the
one it replaced:

| lane | cut | detection after the cut |
|---|---|---|
| `spectral` | 2000 to 512 rows | resolves **1e-4 at 512** against 1e-3 at 2000: SHARPER |
| `holtwinters` | 512 to 128 observations | resolves **1e-7 at 128** against 1e-5 at 512: SHARPER; all 128 observations live one at a time |
| `hdbscan` | 6000 to 4000 rows | 1e-7 at 4000 AND at 6000: unchanged |
| `hdbscan-leaf` | 6000 to 2000 rows | 1e-7 at 2000 AND at 6000: unchanged |
| `gbdt-lossguide-newtoncosine` | 20000 to 1500 rows | still splits on all 16 columns, 13 of 20 trees still reach `max_leaves=32` |
| `gbdt-pair-logit` | 20000 to 1500 rows | 178 query groups, 994 explicit pairs, grades 0..4 all present |
| `gbdt-parametric-losses` | 20000 to 1500 rows | ten distinct arms of eleven, union of at least 13 of 16 columns |
| `gbdt-nan-modes` | 20000 to 1500 rows | the arm is inert at 1500, 6000 AND 20000, so the row count is not what is wrong |
| `byte-lm-resident` | 2 blocks d32 to 1 block d16 ff32 | 11 of 11 parameter tensors live; the SHAPE cut cost nothing |
| `samba` | 6 windows 3 steps to 2 windows 1 step | no bitwise-equal same-shape pair, layer swap and AdamW betas both detected |
| `mamba2-dtlimit` | `(2,16,32)` to `(2,8,32)` | 3 of 3 clamps, 8 of 8 positions, 8 of 9 weights |
| `tokenizer` | GPT-2 table to a 512-rank synthetic vocabulary | 8 of 8 byte flips and a different document all move the cell |

**The shrink programme was mostly right.** All of those numbers now live in
the `@floor(...)` on the lane that carries them, so the measurement travels
with the constraint instead of with a document.

Those measurements are the audit's
(`docs/lanes/LANE_STATUS_shrink-blindness-audit.md`); this lane did not re-run
them, and re-running them is not what the floors rest on. What this lane adds
is that each one is now attached to the number it justifies.

## 5. WHAT THIS DOES NOT CLAIM

- Only `byte-lm` and `samba-untied-dropout-accum` were re-measured here, on
  the `base` fixture, before and after. The other twelve lanes' numbers are
  the audit's and are cited, not reproduced.
- No sabotage binary was built and no library source was edited. Every arm is
  a data, parameter or constructor perturbation through the shipped 0.8.6
  bindings, which is what makes it cheap and build-independent, and also means
  none of it is a statement about a specific sabotage define.
- The floor checker reads SOURCE. It cannot tell whether a floor is the right
  number, and it cannot floor a dimension whose site is not a literal it can
  resolve (section 3c). Both limits are stated in the file's own docstring.
- Three dead arms that the shrink did NOT cause (`gbdt-nan-modes`,
  `mamba3`/`transformer` RMSNorm swaps, `mamba2-dtlimit`'s `dt_bias`) belong
  to `lane/dead-arms` and were not touched here.

## 6. COST

Measured on this Mac through the Metal slot, one core, per fixture, mean of
two fits:

| lane | before the reversal | after | added per fixture | added to a nine-fixture two-repeat Apple column |
|---|---:|---:|---:|---:|
| `byte-lm` | 3.36 s | 4.48 s | **+1.12 s** | about **20 s** |
| `samba-untied-dropout-accum` | 8.74 s | 19.08 s | **+10.34 s** | about **186 s** |

So the two reversals together add about **3.4 minutes** to an Apple column,
against the seven hours a full pass measured at. The audit's estimate for
`byte-lm` was about 34 s; measured here it is about 20 s, and it did not
estimate `samba-untied-dropout-accum`, which is the larger of the two by an
order of magnitude.

These are wall times on a shared M4 under the Metal lock, mean of the two
fits the harness does per cell, with the second fit warm in both lanes
(byte-lm 4.60 then 4.35; samba-untied 20.85 then 17.30). They are a cost
estimate, not a benchmark.

## 7. WHAT THIS BRANCH LEAVES OWED

- Two lanes' references are dropped by design (section 2c). `release/0.8.7`
  retakes all four record columns; those two cells read OWED until it does.
- The `docs/lanes/LANE_STATUS_lane-neural-shape-shrink.md` and
  `docs/lanes/FIXTURE_SHRINK_SCOPE.md` statements that contradicted the code
  are corrected in place on this branch, each with a dated note saying what
  was wrong, rather than quietly rewritten.
- `python/mojolearn/tests/test_host_surface.py` could not be run in this
  worktree, which has no built bindings, because the module imports
  `mojolearn`. Both new test BODIES were executed standalone against this
  branch's `tools/identity_break.py` and both pass; the gate will run them.
- Nothing floors the `byte-lm` shape, the `tokenizer` vocabulary or `umap`'s
  transform, because none of them is a size (sections 3c and
  `NON_SIZE_REVISIONS`). `tools/fixture_floors.py --list` prints the
  unreadable size sites at the end so the hole is on the record:
  `byte-lm` and `byte-lm-resident` both carry `_ids(X, steps * shape.batch,
  ...)`, whose batch comes from `ByteLanguageModelConfig` rather than from a
  fixture size.

## Rules this lane ran under

Own worktree at `~/mojolearn-wt/shrink-floors`, never the shared checkout. Own
venv, so no binding on a shared path could be replaced under a queued job, and
the loaded bindings' sha256 were printed from inside every process. One core,
`nice -n 19`, every thread and job knob at 1, one process at a time. Every
Metal job took the lock through `mac_slot.sh metal` and released it; the queue
was honored and both jobs waited behind other agents' runs. No box rented. No
`git stash`, no `git add -A`, no history rewritten.

## Resume

    SP=<scratchpad>
    python3 -m venv $SP/venv086
    $SP/venv086/bin/pip install ~/mojolearn-evidence/release-0.8.6/macos-wheel/*.whl numpy
    MAC_SLOTS=4 bash ~/mojolearn-evidence/tools/mac_slot.sh metal \
      env MOJOLEARN_NUMERIC_MODE=identical WT=~/mojolearn-wt/shrink-floors \
      nice -n 19 $SP/venv086/bin/python \
      ~/mojolearn-evidence/shrink-floors-2026-09-16/probe_reversals.py both

The probe and both run logs are in
`~/mojolearn-evidence/shrink-floors-2026-09-16/`: `probe_reversals.py`,
`out_before.txt` (the frozen harness at `16b29b9cf`) and `out_after.txt`.

## Pods

None rented on this branch.
