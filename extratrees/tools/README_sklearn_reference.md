# `sklearn_reference.txt` — the quality band, and what it is not

## Read this paragraph before you use the file

**Treating `sklearn_reference.txt` as a correctness gate is a mistake, and it is
a mistake this lane can name precisely.** scikit-learn draws every random number
in `node_split_random` from ONE sequential 32-bit xorshift stream (`our_rand_r`,
`sklearn/utils/_random.pxd:20-34`) advancing a single state word, so draw *k*
depends on draws *0..k-1*; and the ORDER of those draws is the order of the
Fisher-Yates feature walk (`sklearn/tree/_splitter.pyx:592`), whose path depends
on which features were found constant, which depends on the data
(`_splitter.pyx:611-621`). Our builder draws counter-based keyed values off
`(seed, tree_id, node_id, feature_id)` precisely so that it is
order-independent, because a builder that evaluates features in parallel cannot
reproduce a sequential stream without serializing the thing being parallelized.
Two further deviations widen the gap before a single number is compared: our
feature sampler is cuML's rather than sklearn's Fisher-Yates (DEVIATION 131),
and constant features are re-discovered per node rather than inherited down the
tree (DEVIATION 132). So the trees are **not** the same trees, cannot be made
the same trees, and no quantity in this file is an expected value for anything
we compute. It is a range that a correct implementation should land inside; a
number outside it is a **finding to report**, never a target to tune toward,
and nothing in this lane may be adjusted to move our number into a band. That
adjustment would be fitting our learner to another learner's noise. The exact
oracles for this lane are elsewhere: a host-side Mojo transcription of
`node_split_random` using OUR keyed draws
(`extratrees/original/host_splitter.mojo`) and the analytic fixtures whose
answer is hand-computable for every admissible threshold
(`extratrees/original/fixtures.mojo`).

Per `STANDING_ORDERS.md` rule 4 this file is also never computed on a real
dataset. Every fixture is constructed and adversarial.

## What is pinned

| | |
|---|---|
| scikit-learn | **1.9.0** (tag `77def0e`), from the `bench` pixi environment |
| numpy | 2.5.2 |
| python | 3.14.7 |
| fixture seed | 20260821, the same seed `fixtures_check.mojo` uses |
| sklearn seeds | 21, `random_state = 0 .. 20` |
| estimator parameters | scikit-learn's own defaults, nothing set but `random_state` |

The parameter dict is written into the file in full, both for classification and
for regression, so a version bump that MOVES a default shows up as a diff rather
than as silence. `sklearn_reference.py` also asserts the five parameters the
band's meaning depends on (`bootstrap=False`, `max_features='sqrt'` for
classification and `1.0` for regression, `n_estimators=100`,
`min_samples_split=2`, `min_samples_leaf=1`) and refuses to run if a future
sklearn changes one.

Nothing is tuned. The estimators are constructed with **no** keyword arguments
except the seed, which makes "these are sklearn's defaults" a mechanical fact
rather than a claim in a comment.

## The train / holdout split

    row r is TRAIN iff (r % 4) < 2, HOLDOUT otherwise.

A pure function of the row index: no shuffle, no RNG, identical in Python and in
any Mojo reader. 50/50.

It is a period-4 rule and not one of the two obvious alternatives, for reasons
that are properties of these fixtures:

* **Index parity fails on `all_constant`.** Its labels are `r % 2`, so an
  even/odd split puts every class-0 row in train and every class-1 row in
  holdout. The reported accuracy would be 0.0 as an artifact of the split rule
  rather than a fact about any learner.
* **A fixed prefix fails on `separable_gap`, `tie_pair` and
  `regression_step`.** All three are block structured — rows 0..127 are one
  class or one target level, rows 128..255 the other — so a prefix split trains
  on one class and tests on the other.

A period-4 rule takes exactly half of every block of four and half of every
parity class, so it breaks both structures at once.

## Regenerating

Two commands, in this order. The first is Mojo and dumps the fixtures; the
second is Python, checks them cell for cell, and only then trains.

```
pixi run mojo run -I . extratrees/original/fixture_parity_check.mojo > /tmp/etdump.txt
pixi run -e bench python extratrees/tools/sklearn_reference.py --dump /tmp/etdump.txt
```

Parity only, training nothing and writing nothing:

```
pixi run -e bench python extratrees/tools/sklearn_reference.py --dump /tmp/etdump.txt --parity-only
```

The dump itself is **not committed**. It is about 1.5 MB of regenerable text and
committing it would put two copies of the same fixture bytes under version
control, one of which could silently go stale. What is committed instead is its
`sha256`, on the `parity dump_sha256` line, so a regeneration against a dump
produced by a changed `fixtures.mojo` moves a line in the diff.

## Why the parity step exists and why it comes first

A quality band measured on data that merely RESEMBLES ours is a band on somebody
else's dataset, and would be a confound rather than a reference. So
`sklearn_reference.py` reimplements `splitmix64`, `cell_hash`, `unit_float`,
`unit_float20_open`, `signed_unit` and every shaped-column branch in Python, and
compares them **cell for cell** against a dump the Mojo program itself produced.
If one cell disagrees it prints the disagreement and exits without training.

The comparison is on **float bit patterns**, never on decimal text. This
repository has a standing finding that `String(Float32)` does not round-trip —
0.46% of float32 values come back one ULP wrong, and `String(Float32(1.4e-45))`
is the string `"0.0"` — so a decimal comparison would accept exactly the
one-ulp disagreement that a float32-versus-float64 slip in this port produces.
The dump carries `<decimal>/<hexbits>` on every value and the reader ignores the
decimal half. That is not theoretical: sabotage A below flips one bit of one
cell, and the two decimals are **character-identical** while the hex differs.

Python integers are arbitrary precision and Mojo's `UInt64` wraps, so every
addition, multiplication and shift in the ported hash is masked to 64 bits
explicitly.

## The sabotages, because a check never seen to fail is not evidence

Each one was applied to a scratch copy, run, and observed. One per mechanism.

| | sabotage | result |
|---|---|---|
| A | flip ONE bit of ONE X cell in the dump (`bf6b1706` → `bf6b1707`) | **red**, `hashed_cls X[0,0]`. Both decimals printed `-0.91832006`, so a decimal comparison would have passed it |
| B | drop the `& MASK64` from `splitmix64`'s second multiply | **red**, every hashed cell |
| C | build the near-constant column with `next_up` where `fixtures.mojo` uses `next_down` | **red**, `shaped_all X[·,2]` only — the one column that shape reaches |
| D | widen `unit_float`, `unit_float20_open` and `signed_unit` to float64 | **green, and that is the correct answer** — see below |
| E | draw the label plane from `SALT_Y` instead of `SALT_LABEL` | **red**, `y[·]` and `label[·]` together |

**D is a no-op sabotage and its greenness is a measurement, not a gap.**
`fixtures.mojo` claims in its docstring that every float32 operation it performs
is exact by construction: values live on the 2^-24 grid, the upper band of each
gap fixture lives on the coarser 2^-20 grid because `9.0f + 2^-24` would round
back to `9.0f`, and `2*u - 1` is Sterbenz-exact. If that claim holds, the width
of the intermediates cannot be observable, and D says it is not observable
across all 48,384 feature cells. D is therefore the check that the exactness
claim is true rather than aspirational, and it also means this port is immune to
Mojo's multiply-then-add contraction into an FMA in these expressions, because
the multiply is exact in every one of them.

## What the file says today, in one table

`accuracy` is holdout accuracy, `mse` is holdout mean squared error, both as
min/median/max over the 21 sklearn seeds. `depth` and `leaves` are the MEAN over
the 100 trees of one forest, then min/median/max over the same 21 seeds.

| fixture | task | metric band (min / median / max) | mean depth (min/med/max) | mean leaves (min/med/max) |
|---|---|---|---|---|
| `hashed_cls` | 3-class, pure noise | 0.2852 / 0.3125 / 0.3516 | 15.35 / 15.54 / 15.82 | 152.4 / 153.7 / 154.7 |
| `hashed_reg` | regression, pure noise | 0.3661 / 0.3730 / 0.3837 | 16.32 / 16.58 / 16.74 | 256 / 256 / 256 |
| `shaped_all` | 2-class, 10 shapes | 0.4609 / 0.4922 / 0.5156 | 15.35 / 15.60 / 15.91 | 184.9 / 186.0 / 188.2 |
| `shaped_constant_heavy` | 2-class, 45/48 constant | 0.4297 / 0.4609 / 0.4805 | 15.91 / 16.31 / 16.53 | 194.4 / 195.9 / 197.1 |
| `separable_gap` | 2-class, separable | 1.0 / 1.0 / 1.0 | 2.83 / 3.41 / 4.07 | 6.15 / 7.94 / 9.73 |
| `regression_step` | regression, step | 1.953e-5 / 2.539e-4 / 1.016e-3 | 2.79 / 3.17 / 3.38 | 3.79 / 4.20 / 4.40 |
| `tie_pair` | 2-class, duplicated feature | 1.0 / 1.0 / 1.0 | 1.91 / 2.08 / 2.58 | 3.67 / 4.11 / 5.27 |
| `all_constant` | 2-class, no split exists | 0.5 / 0.5 / 0.5 | 0 / 0 / 0 | 1 / 1 / 1 |

The four hashed and shaped rows have labels drawn from their own salt, so they
are a function of nothing in X and their bands are chance-level by construction.
What those rows check is that a learner is not somehow beating chance on noise,
and — far more usefully — the SHAPE of the trees it grows.

## The two degenerate rows, and the prediction they make checkable

`DEVIATIONS.md` 132 and 151 together predict that **our trees are shallower than
sklearn's on data with many constant columns**, and this file is where that
prediction stops being prose:

* **151.** sklearn's loop guard (`_splitter.pyx:573-577`) keeps drawing past
  `max_features`, up to all `n_features`, for as long as every feature drawn so
  far was constant. If any non-constant feature exists anywhere, sklearn finds
  one and the node splits. We evaluate the sampled `colids` exactly once; if all
  of them are constant, the node becomes a leaf.
* **132.** sklearn threads `n_constant_features` down the tree, so a feature
  found constant at an ancestor is excluded from every descendant's draw.
  Nothing is inherited on our side.

`shaped_constant_heavy` is built so that this is the common case rather than a
corner: 45 of its 48 columns fail the float32 `1e-7` constant test, and
`max_features='sqrt'` draws 6 of 48, so the probability that a node samples six
constants is `C(45,6)/C(48,6) ≈ 0.664`. **sklearn's measured answer is that it
barely notices**: mean depth 16.31 and mean leaves 195.9, against 15.60 and
186.0 on `shaped_all`, which has ten columns and only three constant ones. It
grows to full depth anyway, exactly as fact (b) of deviation 151 guarantees.

So the concrete comparison, when our builder can run these fixtures, is:
sklearn 16.31 mean depth and 195.9 mean leaves on `shaped_constant_heavy`.
Anything materially below that is deviation 151 being paid, and it should be
**reported and priced**, not tuned away.

`all_constant` is the floor of the same axis: every column is exactly constant,
so no split exists at any node and sklearn's own trees are depth 0 with exactly
one leaf, invariant across all 21 seeds. Its labels are `r % 2`, so the node is
impure and "no split" cannot be right for the wrong reason. Any implementation
reporting a depth above 0 here is broken, and that single row is the one place
in this file where an exact number IS demanded of us — not because sklearn said
so, but because no split exists.

## Format

Line-oriented, one `key value` per line, floats as `<decimal>/<hexbits>` where
the hexbits are the big-endian IEEE-754 **float64** bit pattern. A reader that
needs the value must parse the hex; the decimal is for a human reading a diff.
Fixture blocks are delimited by `fixture <name>` … `end_fixture`.
