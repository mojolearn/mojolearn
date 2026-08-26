# HIGGS at 1,000,000 rows on the M4: the large-row regime flips the forest

MacBook Pro M4, FAST on both sides, commit 5b728e2 (worktree clean of Mojo
changes; only leg-script commits past e07ddd5). Same protocol as
`2026-08-26-APPLE-trees.md`: our arm on the M4 GPU, the vendors' CPU arms
(the only arms they have on Apple silicon), three timed rounds after one
untimed warm-up, alternating inside ONE process, `nice 19`, one lane per
process. HIGGS 1,000,000 x 28 front prefix, scored against the same fixed
500,000-row tail as every other rung of the ladder. Logs and table in
`mac-2026-08-26`-style layout under `mac-higgs-2026-08-26/`.

## The three lanes at 1M rows

| lane | ours (GPU) | their CPU arm | | our logloss | theirs | our AUC | theirs |
|---|---|---|---|---|---|---|---|
| rf | **26,909.8 ms** | sklearn-rf-cpu 84,216.7 | **3.13x FASTER** | 0.538850 | 0.538873 | 0.809906 | 0.809778 |
| et | **20,944.4 ms** | sklearn-et-cpu 28,761.3 | **1.37x FASTER** | 0.621923 | **0.621178** | 0.762823 | **0.764018** |
| gbdt-symmetric | 3,762.4 ms | catboost-cpu 3,806.1 | parity (1.01x) | **0.542247** | 0.543225 | **0.800537** | 0.799673 |

## What the rung changes

**The forest verdict INVERTS with rows.** At covtype (522,911 rows,
`2026-08-26-APPLE-trees.md`) our rf was 2.16x SLOWER than sklearn's CPU
forest. At 1M HIGGS rows it is 3.13x FASTER with accuracy identical to
5e-5. The covtype loss was a small-rows artifact of our fixed overheads;
sklearn's CPU forest scales far worse with rows than our GPU arm does. The
1M+ regime is where the GPU-access thesis pays on the forest lanes, on the
very box where the vendors have no GPU arm at all.

et wins 1.37x but is the one lane still behind on accuracy (AUC -0.0012
here, accuracy -0.0067 at covtype) — the extratrees tie-break/sampler
defects are the open item, tracked in extratrees/DEVIATIONS.md.

gbdt-symmetric at parity on time with better logloss: CatBoost's CPU arm
is genuinely strong at this shape (28 features), and our win column on the
Mac at shipped year/covtype sizes already exists; the 1M higgs rung adds
"never worse, fits better".

## Caveats

* Medians of three; the rf opponent's spread (69.8-85.4 s) shows ambient
  load, but the alternation protocol means both arms saw it.
* Our hashes were stable across rounds per lane (FAST does not promise
  this; it happened).
* No 5M rung on the Mac by policy — that load goes to rented boxes.
