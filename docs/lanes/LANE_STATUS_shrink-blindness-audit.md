# LANE STATUS: shrink-blindness audit (2026-09-16)

Fourteen identity lanes were shrunk to cut the Apple column's cost, and every
shrink was validated by checking the cell still MATCHED its reference. That
check cannot separate "agrees because it is correct" from "agrees because it
can no longer see". This lane asks the other question, once per lane: **can
this cell still be made to FAIL?**

Branch `lane/shrink-blindness-audit`, cut from `origin/main` at `83be19f71`.
No box rented. Every job took the shared Metal lock through `mac_slot.sh
metal`, one at a time, one core, `nice -n 19`,
`MOJOLEARN_NUMERIC_MODE=identical` set explicitly. Nothing in the shared
checkout was edited and no library source was edited at all.

## THE HEADLINE

**Two of the fourteen are blind, both for the same reason: they were cut to
ONE training step.**

| | lane | what it can no longer see |
|---|---|---|
| 1 | `byte-lm` | a block copy that EXCHANGES `norm1_w` and `norm2_w` on the way in |
| 2 | `samba-untied-dropout-accum` | the entire cosine arm of `WarmupCosineLR`, and the exact rational `_cos_pi_interval` / `_decide_f32` path under it |

`byte-lm-resident` is blind to the same read-side swap by the same mechanism,
but its `grads` part still catches the write side of that defect, so it is
listed separately below rather than counted as a third blind cell.

**The other twelve are fine.** The row and observation cuts
(`gbdt-*` 20000 to 1500, `hdbscan` 6000 to 4000, `hdbscan-leaf` 6000 to 2000,
`spectral` 2000 to 512, `holtwinters` 512 to 128, `mamba2-dtlimit` L=16 to
L=8) cost NO measurable detection. Where a resolution ladder was run the
smaller fixture was as sharp or SHARPER than the one it replaced.

**Three dead arms were found that the shrink did NOT cause**, listed in
section 5 because they are the same kind of defect and they are all live
today: `gbdt-nan-modes` cannot see its own `nan_mode` at ANY size, and the
`mamba3` and `transformer` lanes cannot see a swap of two of their RMSNorm
weights at any size.

## 1. THE INSTRUMENT, and why not a one-ULP nudge

A one-ULP input perturbation tests SENSITIVITY, not COVERAGE, and it would
have passed `byte-lm`. The defect that started this is a PERMUTATION the
shrunken cell cannot see, not a magnitude it cannot resolve.

So every probe here is STRUCTURAL and applied to DATA OR PARAMETERS, one
tensor, column, observation or knob at a time, with the cell's own train-column
hash (`identity_break._train_hash`) as the read-out. No kernel source was
edited and no sabotage binary was built, so no result here can be confounded
by a build. Two questions per lane:

- **(a) PERMUTATION.** Can the cell distinguish two tensors it reads? Measured
  by exchanging them and by asking whether they are bitwise equal at every
  point in the hashed computation. Two tensors that are equal wherever the
  kernel reads them are a swap NOTHING downstream can see.
- **(b) DEAD COVERAGE.** Does the cell read the tensor, column, observation or
  branch at all? Measured by perturbing it structurally (`+0.25` on a weight,
  `x1.25` on a column, a different schedule class, a different `nan_mode`) and
  asking whether the hash moves.

**The probe was seen to FIRE before it was trusted.** It was run first on
`byte-lm`, where the answer is known, and it reproduced the known blindness
exactly (section 2). Across the audit it returned BLIND on 8 of 96 probes and
DETECTED on the rest, and every BLIND verdict below is paired with a CONTROL
in which the same probe DETECTS once the blinding condition is removed.

**Route.** The 0.8.6 macOS wheel supplies the bindings, this worktree supplies
the harness, in a private venv under this session's scratchpad so no other
agent can overwrite a `.so` under a queued job. The sha256 of every mojolearn
binding the process had MAPPED was printed from inside each process, next to
its results (`vmmap` of its own pid); the digests were identical in all seven
jobs, for example `_mojolearn_byte_lm.so b7a8c87228a6693f`,
`_mojolearn_gbdt.so 78690951dd2af1ba`, `_mojolearn_mamba.so 207408a0ba597c13`,
`_mojolearn_training.so e98663609bad1b23`.

**One lane could not be reached on its shipped construction.**
`SpectralClustering(prediction_data=True)` is post-0.8.6 and this binding
refuses the keyword, so the spectral cell was measured WITHOUT it. The lane's
own docstring says `prediction_data=True` "moves no train byte", and the train
column is the only column this audit reads, so the measurement stands for the
train column and for nothing else.

## 2. `byte-lm` and `byte-lm-resident`: BLIND, and the mechanism is exact

`_byte_lm_params` sets every `*norm1_w` and every `*norm2_w` to a vector of
ONES. The shrink cut the lane from three AdamW steps to one
(`tools/identity_break.py:1545-1546`, `:2244-2245`). So the two RMSNorm
weights of a block are **bitwise equal at the input of the only step the cell
hashes**, and exchanging them there is the identity function.

Measured, `--fixtures base`, one core:

```
=== LANE byte-lm (steps-1-1) ===
  1-step cell (the shipped shrink) 63bd514c0e71982a
    at step 1 input : block0.norm1_w == block0.norm2_w bitwise? True
    after step 1    : block0.norm1_w == block0.norm2_w bitwise? False  max|diff| 2.000e-03
    at step 1 input : block1.norm1_w == block1.norm2_w bitwise? True
    after step 1    : block1.norm1_w == block1.norm2_w bitwise? False  max|diff| 2.000e-03
  BLIND    norm1_w/norm2_w SWAPPED at step 1, 1-step cell -> 63bd514c0e71982a
  2-step cell c16f90ad63967074
  DETECTED norm1_w/norm2_w SWAPPED at step 2, 2-step cell -> d22def89a7bb626f
```

and the same shape of result for `byte-lm-resident` at its one block of
d_model 16 (`927bd17804146a19` unchanged under the swap; the 2-step arm moves
`85cabf57208faebf` to `db9a8b203c8930cd`).

Read the middle two lines together. The weights separate by `2.000e-03` after
ONE step, so at step 2 the exchange is no longer the identity and the cell
sees it. **The lane has exactly one step and therefore exactly zero
opportunities to see it.** Every parameter tensor was also perturbed one at a
time and all 20 (byte-lm) and 11 (resident) moved the cell, so this is a
permutation blindness, not dead coverage.

**Which half of the defect class is lost.** A block copy has a read side
(unpack) and a write side (pack). Measured at ONE step:

| | `byte-lm` | `byte-lm-resident` |
|---|---|---|
| read-side swap (unpack) | **BLIND** | **BLIND** |
| write-side swap of the step output (pack) | DETECTED | DETECTED |
| `export_gradients()` entries exchanged (the `grads` part) | n/a | DETECTED |

`grad(block0.norm1_w) == grad(block0.norm2_w) bitwise? False`, which is why the
resident lane's per-tensor `grads` part catches the write side at one step
while `byte-lm`, which hashes only `loss`, `params` and `logits`, has no
per-tensor part at all. So the class the one-step cells have lost is
specifically **a permutation applied where the kernel READS the two norm
weights**.

**Smallest reversal that restores detection, measured:** `range(1)` to
`range(2)` with `_ids(X, 2 * shape.batch, ...)`, in both lanes. Two steps is
enough; three is not needed. Cost is one extra AdamW step per fixture per
repeat, about 1.9 s on Metal for `byte-lm` and 0.85 s for the resident shape,
so about 34 s and 15 s added to a nine-fixture two-repeat Apple column.

**A zero-GPU-cost alternative, if a hash move is acceptable.** Give `norm2_w`
a different constant from `norm1_w` in `_byte_lm_params` (they are the only
tensors in that helper set to a constant). That breaks the symmetry at step
zero and costs no extra step, but it moves every byte LM cell and re-records
four lanes on three vendors, so it is the more expensive change in practice
even though it is free at run time.

## 3. `samba-untied-dropout-accum`: BLIND to the branch its own doc says it exists to reach

The lane runs `WarmupCosineLR(1e-3, warmup_steps=2, total_steps=8,
min_lr=1e-5)`. `_Schedule._progress` (`python/mojolearn/_training_impl.py`)
returns the LINEAR warmup value whenever `t <= warmup_steps`, so the cosine
`_decay` and the exact rational `_cos_pi_interval` / `_decide_f32` path under
it are first evaluated at **step 3**. The shrink cut the lane to ONE step
(`tools/identity_break.py:2301-2303`).

```
=== LANE samba-untied-dropout-accum (steps-1-1) ===
  baseline cell f5297e8abd4a4c51
  BLIND    WarmupCosineLR -> WarmupLinearLR (same peak/warmup/total/min)
  BLIND    WarmupCosineLR -> ConstantLR(warmup 2)
    WarmupCosineLR.lr_at(1) = 0.000500000024
    WarmupCosineLR.lr_at(2) = 0.00100000005
    WarmupCosineLR.lr_at(3) = 0.000933682604
  DETECTED WarmupCosineLR -> WarmupLinearLR at THREE steps (the pre-shrink size)
```

The cell cannot tell a cosine schedule from a linear one, or from a constant
one. The control on the last line is the same probe at the pre-shrink size,
and it fires.

**This is the shrink that contradicts its own recorded reasoning.**
`docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md` section 1f is titled
"LEFT BIG: samba-untied-dropout-accum" and says, of three steps, "step 3 is
the first step that evaluates the cosine at all. Cutting to two steps would
leave the cosine branch, and the exact rational path under it, completely
unexercised. So: 3 steps x 32 rows is the floor". The lane was nevertheless
cut to one step. `docs/lanes/FIXTURE_SHRINK_SCOPE.md` table A carried forward
only the ROWS half of that reasoning ("32 rows KEPT because
accumulation_is_aligned"), and
`docs/lanes/LANE_STATUS_lane-neural-shape-shrink.md` then states as fact that
the lane "keeps ... the third step because it is the first that evaluates the
cosine arm of the warmup schedule". It does not keep it.

**Smallest reversal:** `range(1)` to `range(3)` with `_ids(X, 96, 17)`, which
is exactly the size section 1f already declared the floor. Two steps is NOT
enough here; the schedule needs `t > warmup_steps`.

## 4. The twelve that are fine, and what each was asked

Everything in this section DETECTED, or is a designed invariance, or was
measured to be as sharp after the shrink as before.

| lane | shrink | probe | verdict |
|---|---|---|---|
| `tokenizer` | GPT-2 table to a 512-rank synthetic vocabulary | 8 byte flips across the 4096-byte document; a different document | DETECTED 8/8 and 1/1 |
| `gbdt-parametric-losses` | 20000 to 1500 rows | all 11 loss parts pairwise; feature coverage from the model text | DETECTED (10 distinct of 11; see 5c) |
| `gbdt-nan-modes` | 20000 to 1500 rows | `nan_mode` Min vs Max at 1500, 6000 and 20000 rows | arm INERT AT EVERY SIZE, see 5a (not a shrink loss) |
| `gbdt-lossguide-newtoncosine` | 20000 to 1500 rows | leaf counts per tree; feature coverage | DETECTED; 13 of 20 trees still reach `max_leaves=32` |
| `gbdt-pair-logit` | 20000 to 1500 rows | `pairs_weight` scaled by 3; query and pair census | DETECTED; 178 query groups, 994 explicit pairs, grades 0..4 all present |
| `hdbscan` | 6000 to 4000 rows | each of the 4 fitted columns x1.25; resolution ladder 1e-1..1e-7 | DETECTED 4/4; resolution 1e-7 at 4000 AND at 6000 |
| `hdbscan-leaf` | 6000 to 2000 rows | same | DETECTED 4/4; resolution 1e-7 at 2000 AND at 6000 |
| `spectral` | 2000 to 512 rows | each of the 4 fitted columns x1.25; resolution ladder | DETECTED 4/4; resolution **1e-4 at 512 against 1e-3 at 2000**, so the smaller fixture is SHARPER |
| `holtwinters` | 512 to 128 observations | every one of the 128 observations; `seasonal_periods` 11/13/24; additive vs multiplicative vs none; resolution ladder | DETECTED 128/128 and 5/5; resolution **1e-7 at 128 against 1e-5 at 512** |
| `byte-lm-resident` | 2 blocks d32 to 1 block d16 ff32 | 11 parameter tensors one at a time; the `grads` part | DETECTED 11/11 and the write-side swap; the SHAPE cut cost nothing. Its STEP cut is section 2 |
| `samba` | 6 windows 3 steps to 2 windows 1 step | 20 parameter tensors scanned for bitwise-equal same-shape pairs; layer order swapped; AdamW `betas` | no equal pairs; layer swap DETECTED; betas DETECTED |
| `mamba2-dtlimit` | `(2,16,32)` to `(2,8,32)` | 9 weight tensors; 8 sequence positions; three `dt_limit` values | DETECTED 8/9 weights, 8/8 positions, 3/3 clamps. The one BLIND weight is blind at L=16 too, see 5d |

**One probe that mattered and came back clean.** All four `steps-1-1` lanes
were suspected of losing the AdamW moment machinery, on the argument that at
step 1 the first and second moments start at zero so `betas` cancels out of
the bias-corrected update. **Measured, it does not.** Changing `beta1` 0.9 to
0.5 and `beta2` 0.999 to 0.900 moves every one of the four cells at ONE step,
at two steps and at three. The one-step cells still see the optimizer.

**One invariance worth naming, not a defect.** Zeroing fixture columns 4..15
does not move the `hdbscan` or `hdbscan-leaf` cell, because both lanes fit
`X[:n, :4]` by construction. That is the slice, not a blindness.

## 5. Dead arms the shrink did NOT cause

**5a, 5b and 5d ARE FIXED**, on `lane/dead-arms`, 2026-09-16. Each mechanism
below was reproduced independently before anything was changed, each fix was
watched to fire with the differing values printed, and the reference cost is
written down. `docs/lanes/LANE_STATUS_dead-arms.md` has the measurements. Two
corrections to what is below: 5b reaches `transformer-window` as well, which
carries the same pair through the same helper, and 5d is worse than one-sided,
because `dt_limit=(0.1, 0.1)`, a clamp returning a CONSTANT, read IDENTICAL to
production at both lengths. 5c and 5f are not fixed and stand as written.


These are not shrink regressions. They were measured in the course of the
audit, they are the same defect class, and each one is a cell that cannot fail
in a way its own docstring says it should.

### 5a. `gbdt-nan-modes` cannot see `nan_mode`, at any size

The lane exists to reach `nan_mode` Min and Max, "which is why no lane above
could reach them". Both arms hash the SAME bytes:

```
   1500 rows: min=a21e31ec9cc7597d max=a21e31ec9cc7597d -> IDENTICAL (arm INERT)
   6000 rows: min=58abad67c1742b5c max=58abad67c1742b5c -> IDENTICAL (arm INERT)
  20000 rows: min=7f15e34e477a4eae max=7f15e34e477a4eae -> IDENTICAL (arm INERT)
```

**The mechanism, measured.** `_with_nan` writes NaN into columns 5, 6 and 7.
`labels_for` builds `y_clf` from columns 3 and 4 ONLY, and the fitted trees
split on columns 3 and 4 only. The NaN therefore never reaches a split and
cannot change a prediction. Moving the NaN onto the columns the fit uses
turns the arm on immediately:

```
  NaN in the lane's own columns 5,6,7 : min=a21e31ec9cc7597d max=a21e31ec9cc7597d  IDENTICAL
  NaN in columns 3,4                  : min=da54f1f980861f61 max=d8e2a3e5305a859b  DISTINCT
```

`nan_mode` DOES reach the quantizer (the model text reads `feature 5 ... nan
as_false` against `nan as_true`), so the bug is the fixture, not the library.
The train column is inert; the saved model bytes differ, so a `model` column
would still separate the two arms.

**Fix, one line:** `_with_nan` should write its NaN into columns 3 and 4, the
columns `labels_for` reads, instead of 5, 6 and 7. The labels are derived from
the CLEAN fixture, so the arm becomes live without moving any label.

### 5b. `mamba3` and `transformer` cannot see a swap of two RMSNorm weights, at any size

`_block_weights(..., ones=(...))` sets several norm weights to ones. Where two
of them have the SAME SHAPE, the block reads two bitwise-identical tensors and
exchanging them is the identity. Neither lane trains, so nothing ever
separates them:

```
  mamba3      : same-shape bitwise-equal ones pairs [('B_norm.weight','C_norm.weight')]   BLIND
  transformer : same-shape bitwise-equal ones pairs
                [('input_layernorm.weight','post_attention_layernorm.weight')]            BLIND
  (control, both) the SAME swap once the two are no longer equal              DETECTED
```

`mamba2-dtlimit` is clear: its two `ones` are shaped `(32,)` and `(64,)`, so
no swap between them is even legal.

**Fix:** give one member of each pair a different constant in the lane's
`ones=` handling, or drop it from `ones` and let it come from the hashed
stream. Either moves those two lanes' cells once.

### 5c. `gbdt-parametric-losses` has ten distinct arms, not eleven

`MAE` and `Quantile` hash the same bytes at 1500 rows AND at 20000 rows.
Quantile at its default alpha of one half IS MAE, so this is expected
arithmetic rather than a defect, but the `Quantile` part of that cell adds
nothing over the `MAE` part and should either carry an explicit
`loss_alpha` away from 0.5 or say in the docstring that it is a duplicate.

### 5d. `mamba2-dtlimit` reads `dt_bias` one-sidedly, at both lengths

`dt_bias + 0.25` does not move the cell. Neither does `+0.01`, `+0.1`, `+1`,
`+4` or `+16`; `-4` and `-16` do. The clamp the lane exists to exercise,
`dt_limit=(0.01, 0.1)`, saturates at its UPPER bound on this fixture, so every
positive shift of `dt_bias` is clamped away. **This is not the shrink**: the
same probe is BLIND at the pre-shrink length L=16, and it DETECTS at L=8 once
the clamp is opened to the default `(0, inf)`. The lane's headline claim is
intact (all three `dt_limit` changes are DETECTED); what is dead is the
upstream `dt_bias` path under the lane's own clamp.

### 5e. Feature coverage of the four shrunk GBDT lanes at 1500 rows

Parsed from the model text, so it is what the fitted ensemble ACTUALLY splits
on, not an importance score.

| lane / arm | columns split on | of 16 |
|---|---|---:|
| `gbdt-lossguide-newtoncosine` | 0..15 | **16** |
| `gbdt-pair-logit`, generated pairs | 1,2,3,4,6,7,11,12,14,15 | 10 |
| `gbdt-parametric-losses` LogLinQuantile | 0,1,4,5,6,8,9,11,12,13,14,15 | 12 |
| `gbdt-parametric-losses` Poisson | 1,3,4,6,12 | 5 |
| `gbdt-parametric-losses` MAPE | 1,3 | 2 |
| `gbdt-parametric-losses` CrossEntropy | 3,4 | 2 |
| `gbdt-parametric-losses` MAE and Quantile | 0 | 1 |
| `gbdt-nan-modes` (Min) | 3,4 | 2 |

The cell of `gbdt-parametric-losses` is eleven parts, so its coverage is the
UNION, at least 13 of 16 over the six arms sampled. `gbdt-lossguide-
newtoncosine` reaches every column at 1500 rows despite `feature_fraction=0.5`,
so its row cut cost it nothing here. The narrow arms are narrow because of what
their target depends on, not because of the row count: `CrossEntropy` and
`gbdt-nan-modes` fit `y_clf`, which `labels_for` derives from columns 3 and 4
alone. That is also the mechanism in 5a. These counts were taken at 1500 rows
only; they are reported as the state of the lanes today, not as a before and
after.

### 5f. The synthetic tokenizer vocabulary barely reaches the merge loop

Not a size shrink, but the same question. On the lane's 4096 fixture bytes the
513-rank synthetic vocabulary emits 4092 tokens: **4088 single bytes and 4
two-byte tokens**, a compression of 1.001 bytes per token. The byte map and
the round trip are covered; the BPE merge loop is exercised four times. The highest id emitted is 397, so ids above
255 do occur and the merge table is reached; the `<|endoftext|>` id is 512 and
does NOT appear, so `allow_endoftext=True` is passed but its special-token
branch has nothing to match on this fixture. Byte liveness is 8 of 8, so the
cell is not blind, but anyone treating this lane as coverage of the merge
machinery should know how thin that coverage is.

## 6. What this does NOT claim

- One fixture, `base`, at one repeat per probe. A full nine-fixture sweep was
  not run and is not needed for a coverage question; it costs 37 minutes of
  the one scarce GPU per pass.
- No sabotage binary was built. Every result is a data or parameter
  perturbation through the shipped 0.8.6 bindings, which is what makes them
  cheap and build-independent; it also means none of them is a statement about
  a specific sabotage define.
- The `spectral` numbers are for the train column of a fit without
  `prediction_data=True`, because this binding refuses that keyword (section
  1).
- The GBDT feature-coverage counts are parsed from the model text
  (`split <tree> <level> <feature> <bin>` and Lossguide's
  `node <tree> <idx> <feature> <bin> <l> <r>`), which is archive inspection,
  not a second evaluator.
- Nothing here was merged or pushed at the time of writing, and no reversal
  proposed in sections 2, 3 or 5 had been applied. Sections 5a, 5b and 5d were
  applied afterwards on `lane/dead-arms` (see the note at the head of section
  5); sections 2 and 3 still stand unapplied.

## Rules this lane ran under

Own worktree under this session's scratchpad, never the shared checkout. Own
venv, so no binding on a shared path could be replaced under a queued job; the
loaded bindings' sha256 were printed from inside every process. One core,
`nice -n 19`, `MOJOLEARN_COMPILE_JOBS=1`, one process at a time. Every job took
the Metal lock through `mac_slot.sh metal` and released it; the FIFO queue was
honored and one job waited behind another agent's `rf-*` run. No box rented.
No `git stash`, no `git add -A`, no history rewritten, nothing pushed.

## Resume

    SP=<scratchpad>
    git worktree add -b lane/shrink-blindness-audit $SP/wt origin/main
    python3 -m venv $SP/venv086
    $SP/venv086/bin/pip install ~/mojolearn-evidence/release-0.8.6/macos-wheel/*.whl numpy
    WT=$SP/wt bash ~/mojolearn-evidence/tools/mac_slot.sh metal \
      env MOJOLEARN_NUMERIC_MODE=identical $SP/venv086/bin/python <probe>.py

The probes and every run log are in
`~/mojolearn-evidence/shrink-blindness-audit-2026-09-16/`: `jobA.py` (the
per-lane batteries), `jobB.py` (GBDT), `jobC.py` (AdamW moments), `jobD.py`
(the symmetric-initialization census), `jobF.py`/`jobG.py` (the nan-mode
collapse chased to a size and to its mechanism) and `jobH.py` (GBDT feature
coverage).

## Pods

None rented on this branch.
