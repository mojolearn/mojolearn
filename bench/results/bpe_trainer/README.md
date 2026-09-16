# A bitwise-deterministic byte-level BPE vocabulary trainer: the evidence

Measured 2026-09-16 on lane `lane/bpe-vocab-trainer`, M4 arm64, one core,
`nice -n 19`, one process at a time. Nothing was rented.

## What is being claimed, and what is not

**Not** "bitwise identical across GPU vendors". Vocabulary training is
host-only in every library — Hugging Face, SentencePiece and tiktoken all
train on a CPU, because counting, sorting and merging is not a matmul
workload — so there is no GPU path here, no vendor column, and nothing owed to
a GPU record. Inventing one would be inventing a column for something that
has no device code.

The claim is:

> **The same corpus and config produce the same vocabulary bytes on any
> machine and architecture.**

and it rests on four properties, each of which is visible in
`tokenizer/train/bpe_train.mojo` and asserted below:

1. **A total order on the tie-break.** Highest count, then the smallest
   `(left_id, right_id)`. The pair is held as `left_id * V + right_id`, so
   comparing keys ascending *is* comparing the pair lexicographically, and
   distinct pairs have distinct keys. There is no pair the rule cannot
   separate.
2. **No reduction order to get wrong.** Counting is single-threaded, so there
   is no thread axis and no per-shard merge order.
3. **No iteration order reaches the result.** Selection is a total order over
   `(count, key)`, so the same set of pairs yields the same winner however it
   is walked.
4. **No floats anywhere in selection.** Counts, ids and the comparison are
   integers end to end.

## Three implementations, held to each other

| what | where |
| --- | --- |
| the trainer | `tokenizer/train/bpe_train.mojo` |
| an independent second implementation of the same stated algorithm | `python/mojolearn/_bpe_trainer.py` |
| the gate that requires them to write **identical bytes** | `tokenizer/checks/trainer_check.mojo` |

`pixi run check-bpe-trainer`:

```
synthetic:       3306 corpus bytes, 497 tokens (241 merges), 207 ties broken, 433 groups
ties:             216 corpus bytes, 292 tokens  (36 merges),  35 ties broken,  18 groups
synthetic_small: 1176 corpus bytes, 320 tokens  (64 merges),  42 ties broken, 158 groups
reach: 284 tie-broken selections across the fixtures
trainer_check: PASS
```

Per fixture the gate asserts, and all passed: the **`ranks.tsv` bytes**, the
**`tokenizer.json` bytes**, a **round trip** of the corpus through the shipped
encoder (`impl/bpe.mojo`, not a second copy), and a **repeat** of training in
the same process.

## The sabotage, seen to fail

`pixi run check-bpe-trainer-sabotage` compiles
`-D MOJOLEARN_BPE_TRAINER_SABOTAGE=1`, which reverses **only** the tie-break:

```
trainer_check: SABOTAGE SEEN TO FAIL (6 failures), which is what this build is for
```

All six comparisons (three fixtures × two formats) diverged, and the diff is
the tie-break's own signature — on the `ties` fixture the sabotaged build takes
the largest key where the reference takes the smallest:

```
ours      "ÿ":255, "yz":256,      "wx":257,      "uv":258,
reference "ÿ":255, "Ġa":256, "Ġb":257, "Ġc":258,
```

The Python side carries the same arm as `MOJOLEARN_BPE_TRAINER_SABOTAGE=1`,
and it moves every fixture:

| fixture | clean `ranks.tsv` | sabotaged |
| --- | --- | --- |
| synthetic | `8ad2aabaebe4f7f8` | `648ee0ad88acf262` |
| ties | `e2322ff43bdfb998` | `41ac40b650c553d7` |
| synthetic_small | `35ab5fc278e7e63a` | `13d146bc3070e1db` |

**Why `n_ties_broken` is carried out of the trainer and asserted non-zero.**
This sabotage reverses the tie-break and *nothing else*, so on a corpus that
never produces a tie it would be **inert** and the gate would pass while
testing nothing. The gate therefore fails if no fixture broke a tie, and
`ties.corpus` is engineered to tie (18 two-letter words at equal frequency, so
their pairs reach the top count together and only the stated total order can
separate them).

## Determinism, per axis

`python3 tools/bpe_trainer_determinism.py` — **15/15 rows as required**:

| axis | comparison | verdict |
| --- | --- | --- |
| repeats | 5 runs, identical settings | IDENTICAL (217 ties) |
| vocabulary size | 2 runs at 300 / 512 / 1000 | IDENTICAL |
| corpus order | forward vs reversed vs rotated | IDENTICAL |
| corpus order | 2 runs each within reversed, rotated | IDENTICAL |
| min_frequency | 2 runs at 2 / 3 / 5 | IDENTICAL |
| ties | 2 runs on the engineered tie corpus | IDENTICAL (35 ties) |
| ties | the tie-break is REACHED | 35 > 0 |
| **control** | vocab 512 vs 513 — MUST DIFFER | **DIFFERS** |
| **control** | tie-break reversed — MUST DIFFER | **DIFFERS** |

**Corpus order cannot reach the result, by construction rather than by luck.**
The corpus is a list of documents, each pre-tokenized alone, and the merge
loop sums integer counts over groups and rewrites each group independently. So
neither the document order nor the grouping is observable in the output. The
axis measures that rather than assuming it.

**Threads.** There is no thread axis: counting is single-threaded. That is a
choice, recorded as a property and not measured as an axis — a parallel count
would have to merge per-shard counts in shard index order.

**Architecture.** Selection is over integers end to end, so no
floating-point behaviour can vary the result. Measured here on **arm64 only**.
The x86_64 leg has **not** been run and is owed; it needs a Linux CPU host, and
nothing was rented for this lane. Stated as unmeasured rather than inferred.

## Interop: the `tokenizer.json` round trip, measured

`python3 tools/bpe_trainer_interop.py build/bpe_trainer --self-test --python HFENV/bin/python`

Hugging Face `tokenizers` **0.23.2**, installed in a throwaway venv outside the
repo. It is **not** a mojolearn dependency and nothing about it ships.

25 held-out samples: 23 adversarial (trailing newline, trailing spaces,
interior whitespace runs, upper and lower contractions, combining marks, CJK,
emoji, NBSP, U+001C, control bytes, CRLF) plus generated text from a different
seed than any training corpus.

| fixture | pre-tokenizer shape | result |
| --- | --- | --- |
| synthetic | `Split` + `ByteLevel` | **25/25 identical to ours** |
| synthetic | `ByteLevel(use_regex=true)` | 25/25 identical to ours |
| ties | both shapes | 25/25 identical to ours |
| synthetic_small | both shapes | 25/25 identical to ours |

### The control, and why the first version of it was worthless

A comparison that reports IDENTICAL because it compared nothing is worth
nothing, so three perturbations of a real trained vocabulary must each make it
**fail**:

| perturbation | samples that differ |
| --- | --- |
| drop the most frequent merge | 2/25 |
| keep only the first 5 merges | 14/25 |
| swap the ids of `a` and `b` | 12/25 |

The **first version of this control swapped two adjacent merges and changed
nothing** — merges 2 and 3 never compete on these samples, and BPE applies
merges by rank — so it printed "the comparison can fail" while proving the
opposite. It was caught because the control is run and read, not assumed. Each
arm is now checked separately so one strong arm cannot cover for a dead one.

### Why we emit `Split` and not plain `ByteLevel`

Our pre-tokenizer is the GPT-2 pattern with **possessive** quantifiers, and
`\s++$` is implemented as **end of haystack**. Hugging Face's
`ByteLevel(use_regex=true)` runs *its* GPT-2 regex, where `$` is an ordinary
anchor — a reading that could disagree on text ending in a newline.

Measured directly against our reference pre-tokenizer on 24 adversarial
samples, **both shapes reproduced our splits 24/24**, trailing-newline cases
included. So the two agree on everything tried. We still emit the `Split`
form, which carries our literal pattern in the file, because agreeing today is
not the same as continuing to agree across a future `tokenizers` release. The
file **states** the pattern rather than hoping for it. The `ByteLevel` column
is kept in the harness as a standing cross-check.

### One thing the emitted file cannot express

`<|endoftext|>` is an added **special** token, which Hugging Face always
splits out. That corresponds to our `allow_endoftext=True` reading. The
`allow_endoftext=False` reading — the thirteen characters as ordinary text —
has no `tokenizer.json` spelling. Said plainly rather than papered over.

## The identity lane

`bpe-trainer` in `tools/identity_break.py`. Hashes both emitted formats plus
`n_tokens`, `n_merges` and `n_ties_broken`, over a corpus derived from the
fixture's own bytes. `MOJOLEARN_BPE_TRAINER_SABOTAGE=1` is its sabotage arm.

The lane records **one hash per cell on every device class**, because there is
no device code to vary: what it measures is that the same binary and the same
corpus produce the same vocabulary bytes on every box.

Measured on the lane body, clean against `MOJOLEARN_BPE_TRAINER_SABOTAGE=1`:

| part | clean | sabotaged | |
| --- | --- | --- | --- |
| `ranks` | `7e3ddd048d436846` | `f842f02ddef858a1` | **MOVED** |
| `tokenizer_json` | `7ccc662b9848f806` | `fcda36701031d1de` | **MOVED** |
| `n_ties_broken` | `d13880042d3c804e` | `8df80df3360534bf` | **MOVED** |
| `n_tokens` | `177c88281b081a6f` | `177c88281b081a6f` | same |
| `n_merges` | `957a46faf78b9dbc` | `957a46faf78b9dbc` | same |
| cell `train_hash` | `ae5ab7f0d202f77d` | `da26be2a2d515cee` | **MOVED** |

**`n_tokens` and `n_merges` do not move, and the lane says so.** The
vocabulary still fills to `vocab_size`; it fills with *different tokens*. A
cell that hashed only the sizes would call this sabotage inert, which is
exactly why both artifact byte-hashes are in the cell and why the counters
alone are not trusted to carry it.

**A note on how this was exercised.** A full `tools/identity_break.py` run
needs built identical binaries, which this checkout does not have
(`_backend.select()` refuses), so the lane *body* was called directly out of
`LANES` against the real `BpeVocabularyTrainer` door with `_backend` stubbed.
The lane's first run inside the CPU identity gate, where the bindings exist,
is still owed.

## The ecosystem baseline, and a qualification to it

Our claim is worth stating against a known baseline rather than an assumed
one. Measured on this lane, `tokenizers` 0.23.2, arm64, one core,
`RAYON_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`, fixed corpus,
`vocab_size=300`:

| configuration | processes | distinct vocabularies |
| --- | --- | --- |
| `BpeTrainer`, no prefix or suffix | 8 | **1** |
| `BpeTrainer` with `continuing_subword_prefix="##"` | 6 | **6** |
| `WordPieceTrainer` | 8 | **8** (sizes 165 *and* 167) |

**This qualifies the earlier finding that Hugging Face BPE is bitwise
reproducible.** It is — *in the no-prefix, no-suffix configuration that was
measured*. Turn on `continuing_subword_prefix` and it is not, and the
vocabularies differ in content and even in size, not merely in id assignment.

The mechanism is not float arithmetic. `bpe/trainer.rs::tokenize_words`
iterates an `AHashMap` and inserts prefixed tokens in **hash order**; those
ids are exactly what `Merge::cmp` uses to break count ties. ahash's default
features include `runtime-rng`, and its state is documented as unique per
instance, which is why the variation appears *within* a single process and not
only across processes. Without a prefix the branch inserts nothing, because
the alphabet is already built by a path that sorts explicitly "for
determinism".

Two consequences worth keeping straight:

- Hugging Face **WordPiece is not a separate algorithm** in `tokenizers`. Its
  trainer holds a `BpeTrainerBuilder`, trains BPE, and converts. So "HF
  WordPiece is nondeterministic" and "HF BPE with a prefix is
  nondeterministic" are the same finding.
- Our trainer's tie-break orders by `(left_id, right_id)` where the ids come
  from a **sorted, fully determined** construction — the 256 single bytes in
  byte order, then merges in the order they were made. There is no hash-ordered
  id assignment anywhere for a tie-break to inherit.

## What this evidence does not cover

- **arm64 only.** The x86_64 leg is owed.
- **Small corpora.** The fixtures are hundreds to thousands of bytes, which is
  what a *bitwise* check needs. Nothing here says anything about training at
  the scale a production vocabulary is built at.
- **Synthetic corpora.** Generated from an integer recurrence so nothing is
  committed. Tie density differs by corpus kind, and a code or prose corpus
  was not measured.
- **One `tokenizers` release.** 0.23.2. The determinism lane's finding that
  three releases agreed is evidence, not a promise.
