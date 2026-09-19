# The two-device pair for `par-ordered` and `par-border-types`

`docs/VERIFICATION_MATRIX.md` listed both under "no two-device column", two of
the three `par-*` lanes that had none. Their claim is the one
`identity_break._par_devices` states for every driver: **a two-device column
must hash equal, cell for cell, to the one-device column of the same commit.**
It is not stateable on one device -- `lane_applicability.degenerate` holds them
on any one-device column, because with one shard the equality is not false, it
is not expressible -- so the pair, off ONE build on ONE box at ONE commit, is
the only thing that discharges them.

One lease. Pod `1uhca4yluoyhy9`, **2x NVIDIA GeForce RTX 4090** (sm_89, driver
580.159.04), RunPod, `MOJOLEARN_GEMM_LEG_GPU_COUNT=2`, commit
`3d203f75fe887c09881cf1d154c354fa5cf49b37`, 20:51:31Z to 21:09:56Z of body
time inside a 60-minute lease. `nvidia-smi -L` reported `visible_gpus=2` and
the body refuses the two-device phase by name if it is not 2 -- a second copy
of the first column would make "they agree" a tautology.

| file | `package.par_devices` | `admit` | `admit(par_axis=True)` | cells |
|---|---|---|---|---|
| `...par-two-device-gaps.json` | `0` | ADMISSIBLE | ADMISSIBLE | 18 STABLE, 0 moved, 0 refused |
| `...par-two-device-gaps.two-device.json` | `0,1` | `par_devices 0,1` | ADMISSIBLE | 18 STABLE, 0 moved, 0 refused |

18 = 2 lanes x 9 fixtures, `--repeats 2`, default fixture set at the default
size, no sabotage switch set anywhere in the run. The second file is
INADMISSIBLE TO THE DEFAULT RULE ON PURPOSE and that is correct: a two-device
answer must never be read as the one-device reference. `par_axis=True` is the
rule that admits it, `verification_matrix.par_two_device` is the only reader
that uses it, and it credits only lanes whose name starts `par-`.

## What the pair says

`identity_break.py --diff`, run ON THE BOX while it was still rented so that a
disagreeing lane could be re-run SOLO before anyone called it real (the body
does that automatically and found nothing to do):

    summary:                 IDENTICAL=18
    summary (infer/model):   IDENTICAL=36
    summary (batch):         IDENTICAL=18

`par_diff_exit=0`. No DIVERGENT, MOVED or RELOAD-MOVED cell on either axis.

## The lease before this one, and why it did not finish

Pod `uurynr41em0zb0`, the same box shape and the same body, 20:01Z to 20:44Z.
Its one-device column is complete and admissible; its two-device column was
**cut off by the phase timeout at 16 of 18 cells** (`column_two_exit=124`),
leaving `complete: false`, which `admit` refuses by name as an "incomplete
identity_break checkpoint" -- correctly, and that is why that pair is not the
one committed here. The 16 cells it did record are IDENTICAL to their
one-device partners; the two it never reached are `par-border-types/negative`
and `par-border-types/odd`.

It ran out of budget because **it compiled all 56 binding families from
source, 1738 s of a 2600 s body budget**, and a par-`*` lane list derives no
declared family from `host_surface.FAMILIES` (these lanes have no host route
at all), so the column phase refuses everything until the body's own backstop
reads `bindings/build_gbdt.sh` out of the refusal text and builds it. That is
the body working as designed; it was simply too slow to fit twice.

## THE BINDING CACHE, MEASURED: THE SAME BUILD PHASE, 1738 s AND THEN 160 s

These two leases are the first back-to-back pair the R2 binding cache has ever
served, and the difference between them is the whole of it:

| | `uurynr41em0zb0` | `1uhca4yluoyhy9` |
|---|---|---|
| staged map | `entries=55` | `entries=110` |
| `bincache_outcomes` | `56 miss+built-uploaded` | **`55 hit`**, 1 `bypass-destination-exists+built-uploaded` |
| build phase | **1738 s**, every family compiled | **160 s**, 29.3 MB downloaded |
| `elapsed_after_all_families` | 1725 s | **59 s** |

Same arch (`sm_89`), same image, same `repo_path`, same OS, same
`MOJOLEARN_COMPILE_JOBS=16`. The one non-hit is `build_gbdt`, rebuilt by the
backstop after the cache had already placed it: `bypass-destination-exists` is
the cache refusing to overwrite a file that is already on disk, which is the
right answer and not a miss.

Both legs ran the SAME lane list, but that is no longer what makes them share
a key. Until commit `900038b24` it was: `MOJOLEARN_GAP_LANES`,
`MOJOLEARN_GAP_SLUG`, `MOJOLEARN_GAP_COMMIT_DIR` and `MOJOLEARN_GAP_BUDGET`
were all inside `build_env`, so the cache partitioned itself by the errand the
leg was about to run and the only leg that could ever hit an entry was one
running the same lanes into the same directory. The 110 objects now under
`bincache/v1/sm_89/` are 55 written under the old key shape, which nothing
will ever read again, and 55 under the fixed one, which is what was hit here.

Evidence: `~/mojolearn-evidence/par-two-device-sep19/nvidia{,-run2}/`, with
both gate files, both `par_diff.log`s, and `remote/bincache/provenance.tsv`
naming every family and its outcome.

## Why this pair is in a subdirectory

Another lane's leg was recording into this same directory in the same hour,
under the SAME slug and therefore the SAME two file names -- `MOJOLEARN_GAP_SLUG`
decides both, and two legs that pick the same slug collide by construction.
Its pair (`par-forecast-arima`, commit `900038b24`) landed at
`../README.md` and `../nvidia-...par-two-device-gaps{,.two-device}.json` in
commit `a266030f1`, four minutes before this one, and the first version of
this commit overwrote all three. They are restored at their original paths,
byte for byte, and this pair moved down one level instead. Nothing here
changes what that leg recorded or what its README says about it.

`admit` was re-asked at THIS path after the move, because it reads the path:
`ordered-and-border-types/` carries none of the excluded tokens and both
answers are unchanged.

## The three `par-*` lanes still without a two-device column, read out of the matrix

`par-ivf`, `par-forecast-arima` and `par-forecast-holtwinters`, and they are
short of one for two different reasons, which matters to whoever buys the next
lease:

* `par-forecast-arima` HAS a pair, in this directory, at `../`. Both halves
  carry `complete: false` -- the leg that made them was cut off -- and `admit`
  refuses an incomplete checkpoint by name whether or not `par_axis` is passed,
  so `par_two_device` cannot read either one and the lane counts as having
  none. One cell of real agreement is recorded there and its README says so;
  it needs the run finished, not redone from nothing.
* `par-forecast-holtwinters` and `par-ivf` have NO column on any box at all.
  (A commit message of mine, `a2b6fcc0a`, said holtwinters was the
  `complete: false` case. It is not; only `par-forecast-arima` is.)
