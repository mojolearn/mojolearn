# Eight `par-*` lanes whose negative control was recorded as a crash

`lane/broken-par-sabotage-arms`, 2026-09-20. Apple M4, CPU host route only,
one core, `nice -n 19`, base fixture, `--repeats 2`. No GPU, no rental, no
Metal job. Every column here was produced by this branch's
`tools/identity_break.py` against host bindings built from this tree.

## The question, and what it was NOT

Four lanes -- `par-ivf`, `par-queries-nn`, `par-rbf-sampler`,
`par-forecast-arima` -- ran clean on a CPU column and REFUSED under a
sabotage host build. Three explanations were live: the arm perturbs something
the binding validates before arithmetic runs; the sabotage build simply did
not carry a binding those lanes reach; or the lanes have no CPU-reachable arm
at all, the way 36 other `par-*` lanes refuse for a structural reason.

**None of the three.** Measured here:

* NOT a missing binding. Both columns carry **32 host families**. The
  sabotage column is the clean 32-family set with exactly five bindings
  replaced by `-D MOJOLEARN_HOST_SABOTAGE=1` builds, every other file the
  same one the clean column loaded. Its own metadata says so:
  `host.families[*].sabotage` reads true for `_mojolearn_arima_host`,
  `_mojolearn_core_host`, `_mojolearn_ivf_host`, `_mojolearn_ivf_search_host`
  and `_mojolearn_kernel_methods_host`, and false for the other 27. The
  column witnesses its own binary.
* NOT structurally CPU-unreachable. All four ran **STABLE with every part**
  on the clean CPU column.
* NOT a misplaced arm. The arm fires, changes arithmetic and never touches a
  validation. What refuses is the LANE'S OWN IN-CELL ORACLE.

## What actually happened

Each of the eight lanes held its sharded driver byte for byte to the plain
call through `_same_bytes`, which RAISES. Under a sabotage build the two
sides are legitimately different computations -- a per-call perturbation
lands once per shard on the driver route and once in total on the plain one
-- so the equality fires, the lane raises before it can hash anything, and
the cell reads REFUSED. `negative_control_moves` does not count a REFUSED
cell, so a control that was working perfectly credited nothing.

This is `par-scaler`'s defect, named and fixed on 2026-09-17
(lane/sabotage-sweep). The pass was not extended to these eight. The remedy
is the same: `_mismatch_bytes` / `_oracle_mismatch` to get the message
without the raise, build the parts, then `raise NumericalMismatch(msg,
parts)`, which `_run_reference` catches and hashes.

## The one define behind all eight: `core`

`-D MOJOLEARN_HOST_SABOTAGE=1` on the `core` family also arms
`bindings/hotpath_helpers.mojo::HOTPATH_SABOTAGE`, which perturbs the GENERIC
array helpers -- `cast_elements`, `gather_*`, `reduce_stat`,
`equal_elements`, the encoders -- that every driver's shard staging runs
through. So the define reaches lanes in families it does not name. Isolation
runs, one family at a time, `par-ivf`:

| families built with the define | `par-ivf/base` |
|---|---|
| none (clean) | `82ac27da71d34ea9` STABLE |
| `ivf_search` alone | `82ac27da71d34ea9` STABLE -- arm inert on this lane |
| `ivf` alone | `1a0e8a596286d0c3` STABLE, cell MOVED |
| `ivf` + `ivf_search` | `1a0e8a596286d0c3` STABLE, cell MOVED |
| `ivf` + `core` | `889fbcdf70592e89` DIVERGENT, oracle fires |

`ivf` alone reproduces the hashes this lane's docstring already recorded
(`82ac27da71d34ea9 -> 1a0e8a596286d0c3`, infer `fd6eed35b1338d17 ->
c70054c87bc14b51`), which is why the defect was not seen when the lane was
written: it was measured under a narrower build than the gate's. The same
holds for `par-rbf-sampler` (`kernel_methods` alone moves the cell;
`kernel_methods` + `core` fires the oracle) and `par-forecast-arima`
(`arima` alone moves it; `arima` + `core` fires the oracle).

## Before and after, five-family sabotage column against the clean column

Part lists compared as LISTS, one entry per repeat, never as strings.
`compare_par_arms.py` self-checks first: it perturbs a value and requires the
comparison to see it, and refuses a column whose parts are strings.

| lane | cell clean | cell sabotaged | DIVERGENT parts |
|---|---|---|---|
| `par-ivf` | `82ac27da71d34ea9` | `889fbcdf70592e89` | 7 of 9 (`cand`, `thin_cand` unmoved) |
| `par-queries-nn` | `6869ae45e01fcb7a` | `37cd37aefcd550f0` | 2 of 2 |
| `par-rbf-sampler` | `7ddfa0d92256da95` | `5858b9f3543b1afb` | 1 of 3 (`weights`, `offset` unmoved) |
| `par-forecast-arima` | `a44b54d65768bf5e` | `6a9df62099271197` | 2 of 2 |

12 DIVERGENT parts in total. Both repeats agree on every value, so
`stable_digest` accepts them; at `--repeats 1` it would refuse every one and
the move would be silently discarded, which is why every column here is
`--repeats 2`.

All EIGHT fixed lanes, under the `core` define alone
(`cpu-apple-m4.par-sweep.clean.json` against
`cpu-apple-m4.par-sweep.sabotage-core-only.json`). One define, one build,
every lane DIVERGENT:

| lane | cell clean | cell sabotaged | DIVERGENT parts |
|---|---|---|---|
| `par-ivf` | `82ac27da71d34ea9` | `d49808dbbabbe21d` | 7 of 9 |
| `par-queries-nn` | `6869ae45e01fcb7a` | `37cd37aefcd550f0` | 2 of 2 |
| `par-rbf-sampler` | `7ddfa0d92256da95` | `4ac55fc9d9971178` | 1 of 3 |
| `par-forecast-arima` | `a44b54d65768bf5e` | `f0e99cb969994ece` | 2 of 2 |
| `par-arima` | `377790cb117abb61` | `39df44a7d09f3200` | 4 of 4 |
| `par-holtwinters` | `aa71e2e60daefca2` | `f9d07f713823e563` | 10 of 10 |
| `par-queries-knn` | `4b0dd36744b84c37` | `e2990428299692e1` | 4 of 4 |
| `par-queries-radius` | `b4fa4ebf9623f5d0` | `6b1b106b658b7bd1` | 3 of 4 (`radius` unmoved) |

33 DIVERGENT parts. `par-ivf`, `par-rbf-sampler` and `par-forecast-arima`
land on different sabotaged hashes here than in the five-family table above,
because here only `core` carries the define and there five families did; both
are the same lane doing the same thing under two different builds.

## `admit()` and the credit

`python/mojolearn/_verify_reference.py::admit`, and
`tools/verification_matrix.py::negative_control_moves`:

| column | `admit()` |
|---|---|
| `cpu-apple-m4.clean.json` | ADMITTED (None) |
| `cpu-apple-m4.sabotage-five-families.before-fix.json` | a host binding reads back sabotage |
| `cpu-apple-m4.sabotage-five-families.json` | a host binding reads back sabotage |

| lane | before the fix | after the fix |
|---|---|---|
| `par-ivf` | REFUSED, credited: NONE | DIVERGENT, credited: `train` |
| `par-queries-nn` | REFUSED, credited: NONE | DIVERGENT, credited: `train` |
| `par-rbf-sampler` | REFUSED, credited: NONE | DIVERGENT, credited: `train` |
| `par-forecast-arima` | REFUSED, credited: NONE | DIVERGENT, credited: `train` |

And over the `core`-only sweep pair, all eight lanes
(`cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json`, taken when the
first four were already fixed and the second four were not, against
`...sabotage-core-only.json`):

| column | `admit()` |
|---|---|
| `cpu-apple-m4.par-sweep.clean.json` | ADMITTED (None) |
| `cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | a host binding reads back sabotage |
| `cpu-apple-m4.par-sweep.sabotage-core-only.json` | a host binding reads back sabotage |

| lane | before | after |
|---|---|---|
| `par-ivf` | DIVERGENT, credited: `train` | DIVERGENT, credited: `train` |
| `par-queries-nn` | DIVERGENT, credited: `train` | DIVERGENT, credited: `train` |
| `par-rbf-sampler` | DIVERGENT, credited: `train` | DIVERGENT, credited: `train` |
| `par-forecast-arima` | DIVERGENT, credited: `train` | DIVERGENT, credited: `train` |
| `par-arima` | REFUSED, credited: NONE | DIVERGENT, credited: `train` |
| `par-holtwinters` | REFUSED, credited: NONE | DIVERGENT, credited: `train` |
| `par-queries-knn` | REFUSED, credited: NONE | DIVERGENT, credited: `train` |
| `par-queries-radius` | REFUSED, credited: NONE | DIVERGENT, credited: `train` |

The sabotage columns are correctly refused for reference admission and
correctly credited as negative controls. `infer`, `model` and `batch` read
REFUSED in a cell whose oracle fired, because `_run_reference` skips the
remaining probes for that repeat; that is `par-scaler`'s behavior too, and
the train column is what carries the catch.

## The sweep: all 59 `par-*` lanes, and four more of the same shape

Clean and `core`-only-sabotage columns over every `par-*` lane.

| CPU column, clean | lanes |
|---|---|
| ran STABLE with full parts | 23 |
| REFUSED, no host binding restates the cooperative driver | 36 |

The 36 are structural: their message names the reason in the driver's own
words ("its shards are device row tiles"), or a missing byte-LM GPU binding,
or a CUDA/HIP-only guard. A CPU box cannot supply a control for them and
this lane does not try.

| under the `core` arm alone, of the 23 | before the fix | after |
|---|---|---|
| CAUGHT, a part moved and is credited | 13 | **17** |
| INERT, every part at its clean hash | 6 | 6 |
| BROKEN, ran clean and REFUSED sabotaged | 4 | **0** |

No `par-*` lane that runs on a CPU column now refuses under this arm. The
four that moved from BROKEN to CAUGHT were found by running, not by reading:

| lane | refusal text under the `core` arm |
|---|---|
| `par-arima` | `fit_arima ar and plain ar differ: 15 bytes of 16` |
| `par-holtwinters` | `fit_exponential_smoothing level_ and plain level_ differ: 6054 bytes of 8000` |
| `par-queries-knn` | `ParallelQueries kneighbors distances and plain distances differ: 1523 bytes of 2048` |
| `par-queries-radius` | `ParallelQueries radius_neighbors counts and plain counts differ: 50 bytes of 512` |

All four carry the same remedy on this branch, and all four are now
DIVERGENT with `train` credited.

The six INERT lanes are `par-forest-reg`, `par-mlp`, `par-samba`,
`par-samba-clip`, `par-scaler` and `par-scaler-minmax`. INERT here means only
that the `core` define does not reach them; `par-scaler` and
`par-scaler-minmax` already carry the fixed shape, and their own families'
arms move them.

## The wider audit

`audit_bare_oracles.py` reads `tools/identity_break.py` as an AST and reports
every lane whose in-cell oracle is a bare `_same_bytes` in the lane BODY.

| `tools/identity_break.py` | lanes with a bare body oracle | lambda-only |
|---|---|---|
| at the branch point | **57** | 5 |
| this branch | **49** | 7 |

Most of the 57 are `par-*`. Eight are fixed here -- the ones a CPU column can
show firing today. The other 49 have the identical shape and have not been
seen to fire, either because no CPU route reaches them (36 of the `par-*`
ones) or because no arm built so far separates their two sides. That is not a
guarantee: any future family define that lands asymmetrically turns one of
them into the same silent non-catch, and the audit script is committed here
so the list can be re-read rather than re-discovered.

A call to `_same_bytes` inside the INFER lambda is a weaker case -- it costs
the infer, model and batch columns rather than the cell, and the train hashes
survive. `par-arima` and `par-forecast-arima` moved into that weaker group
here: their body oracle is fixed and the lambda's is left, because it
compares two doors of the SAME route rather than a sharded route against a
plain one.

## The clean column did not move

Every one of the 59 `par-*` cells, before and after the eight-lane edit, on
the clean host set: 0 verdict changes, 0 moved hash lists, 0 changed part
names. No recorded reference is invalidated by this branch
(`clean_regression.py`, `cpu-apple-m4.par-sweep.clean.before-fix.json`
against `cpu-apple-m4.par-sweep.clean.json`).

## Files

Columns are named so `admit()` refuses the sabotage ones by basename, which
is the intended behavior: `_EXCLUDED_BASENAME_TOKENS` matches `sabotage`.

* `cpu-apple-m4.clean.json`, `...clean.before-fix.json` -- the four lanes,
  five-family host dir, production bindings.
* `cpu-apple-m4.sabotage-five-families.json`,
  `...before-fix.json` -- the same under the five sabotage bindings. The
  `before-fix` column is the RECORD OF THE DEFECT: four REFUSED cells.
* `cpu-apple-m4.sabotage-<families>.<lane>.json` -- the isolation runs, one
  or two families with the define.
* `cpu-apple-m4.par-sweep.*.json` -- all 59 `par-*` lanes, clean and
  `core`-only, before and after the fix.
* `compare_par_arms.py`, `sweep_report.py`, `verdicts.py`,
  `audit_bare_oracles.py` -- the readers, each with a self-check that must
  fail on a perturbed input before it prints an answer.

## What this does not say

The `batch` part of `par-queries-nn` has a separate, live defect: it fails on
TWO devices and only on two. Nothing here touches the device axis or the
batch part; at one device the partition is one worker per shard, as every
`par-*` lane's is. The change is to how a disagreement these cells ALREADY
detected is reported, not to what they compute.

These columns are one box, one vendor, one fixture. They say the arms are
seen to move on a CPU column. They say nothing about the 36 structural lanes,
which are owed a two-device run.
