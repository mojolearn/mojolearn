# The CatBoost comparison inverts with scale, and covtype is the small end

Taken 2026-08-20 on the M4, quiet box, AC power, at `028da27`.
`bench/interleaved/catboost_interleaved.mojo`, our GPU against CatBoost's CPU,
depth 6, 20 trees, arms alternating per rep, three reps, both arms on
CatBoost's own quantization grid with the pool quantized outside the timed
region. Ratios are of medians. **Every shape run is reported here**; the
ladder was fixed before the first number arrived.

| shape | borders | CatBoost CPU | ours GPU | ratio | loss vs CatBoost |
|---|---|---|---|---|---|
| covtype 581,012 x 53 | 254 | 9.75 | 19.28 | 1.98x slower | match to 8 digits |
| covtype 581,012 x 53 | 128 | 9.71 | 18.16 | 1.87x slower | match to 8 digits |
| synth 581,012 x 53 | 254 | 16.37 | 24.18 | 1.48x slower | **DIVERGES, ours better** |
| synth 581,012 x 53 | 128 | 15.78 | 20.17 | 1.28x slower | match to 8 digits |
| synth 800,000 x 100 | 254 | 29.77 | 35.46 | 1.19x slower | match to 8 digits |
| synth 800,000 x 100 | 128 | 30.02 | 28.87 | **1.04x FASTER** | match to 8 digits |
| synth 2,000,000 x 100 | 254 | 80.72 | 60.22 | **1.34x FASTER** | **DIVERGES, ours worse** |
| synth 2,000,000 x 100 | 128 | 80.31 | 53.75 | **1.49x FASTER** | match to 8 digits |
| synth 4,000,000 x 100 | 254 | 188.79 | 103.10 | **1.83x FASTER** | match to 8 digits |
| synth 4,000,000 x 100 | 128 | 188.72 | 96.85 | **1.95x FASTER** | match to 8 digits |

ms per tree.

## The mechanism, which is visible in the raw times rather than the ratios

From 800,000 to 4,000,000 rows at 100 features, five times the data:

    CatBoost CPU     29.77 -> 188.79 ms/tree     6.34x
    ours GPU         35.46 -> 103.10 ms/tree     2.91x

CatBoost's per-tree cost grows slightly faster than the data. Ours grows at
roughly half that rate, because a large part of what a tree costs us does not
depend on how many rows there are. **That fixed cost is a defect at 581k rows
and an advantage at 4M rows, and it is the same fixed cost in both places.**
The crossover sits near 800,000 rows at 128 borders and between 800,000 and
2,000,000 at 254.

The corollary is unwelcome and should not be lost: work that removes per-tree
control-plane overhead moves the crossover down and helps every small shape,
and it does approximately nothing for the 4M column. Those are different
lanes and a win in one is not a win in the other.

## covtype's loss is mostly shape, not real data

The control is the fourth and fifth rows: synthetic data at covtype's exact
581,012 x 53. It loses too, 1.28x and 1.48x, against covtype's 1.87x and
1.98x. So the shape accounts for most of the deficit and the data for the
rest. We are not bad at covtype, we are bad at narrow tables with under a
million rows, and covtype is one.

## An open defect: 254 borders does not always reproduce CatBoost's model

At 128 borders our final MSE matches CatBoost's to eight significant figures
at **all five** shapes. At 254 borders it matches at three of five and departs
at two, once in our favor and once against:

    synth   581,012 x 53   ours 0.14419073   CatBoost 0.14598164   ours better
    synth 2,000,000 x 100  ours 0.14395397   CatBoost 0.14285144   ours worse

Both directions rules out a simple precision floor and points at a split
decision going a different way, but nothing here identifies it. Until it is
identified, **the 254-border speed numbers at those two shapes are not
like-for-like** and should not be quoted, including the 1.34x at 2M, which is
a win on a model CatBoost did not train. The 4M and 800k rows at 254 borders
are unaffected and stand.

## What this does not say

The large shapes are synthetic, drawn from `rng.normal` with a simple additive
target, because `tools/interleaved_prep.py` offers exactly one real dataset.
A real table at 4M rows could behave differently, and the one real dataset
here is at the end of the ladder where we lose. **The claim supported is that
we win on wide tables of millions of rows in this generator, not that we win
on large data generally.** Getting a second real dataset into the prep tool is
the cheapest way to raise that ceiling.

Nothing above was run more than three reps and no shape beyond 4,000,000 rows
was attempted; that is where this box runs out of memory, not where the curve
was seen to stop.
