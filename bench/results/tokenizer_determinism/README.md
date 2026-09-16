# Are tokenizer vocabulary trainers bitwise reproducible?

Measured 2026-09-16 on lane `lane/tokenizer-trainer-determinism`. The question
decides whether mojolearn builds its own vocabulary trainer or pins an
existing one. It was measured, not reasoned about.

Subjects, both pinned to the version actually tested:

| trainer | version |
|---|---|
| Hugging Face `tokenizers` | 0.23.2 |
| Google `sentencepiece` | 0.2.2 |

Neither is a mojolearn dependency. Both were installed into a throwaway venv
outside the repo. No vocabulary is vendored, no corpus is committed, and the
wheel still carries no vocabularies.

## Corpus

`corpus/enwik8/input.txt` from R2 (bucket `mojolearn-data`, sha256
`2b49720e…c024a8`, 100,000,000 bytes), already staged, openly licensed, not
re-downloaded from anywhere new. The first 16,777,436 bytes are cut at line
boundaries into four shards of about 4 MB, and a 256 KB held-out slice taken
from *after* the training bytes is the sample text, so the tokenization layer
is not merely re-reading training text.

`tools/tokdet_corpus.py` does the cutting. Boundaries are chosen by scanning
forward from exact byte offsets, so the same source file yields the same
shards on any machine. Nothing samples or shuffles.

## The comparison, and why it is allowed to be believed

Three layers, because "the files differ" and "the vocabularies differ" are
different findings:

- **bytes** — sha256 of every artifact. Normalizes nothing.
- **struct** — the token→id map, and the merge/piece list compared as a
  **sequence** and as a **set**. Sequence-versus-set is what separates "the
  same merges in a different order" from "different merges".
- **tokenize** — 400 held-out lines through both, comparing **id sequences
  and piece sequences**. Ids differ but pieces match ⇒ a renumbering, the
  same tokenization spelled differently. Pieces differ ⇒ the vocabularies
  genuinely disagree.

`tools/tokdet_compare.py --self-test` makes the comparison fail before anyone
trusts an "identical" from it. Both directions are controlled:

- **POSITIVE** — an exact copy must compare identical, so a comparison that
  always screams is caught too.
- **NEGATIVE** — one-at-a-time perturbations of a *real* trained vocabulary,
  each asserting the specific layer it must trip.

Measured result, both trainers, both directions:

| control | layers that fired | required | passed |
|---|---|---|---|
| POSITIVE exact copy | (none) | none | yes |
| HF swap two vocab ids | bytes, struct | bytes, struct | yes |
| HF reorder two merges | bytes, struct | bytes, struct | yes |
| HF drop one merge | bytes, struct, tokenize | bytes, struct, tokenize | yes |
| SP bump one score | bytes, struct | bytes, struct | yes |
| SP swap two pieces | bytes, struct | bytes, struct | yes |
| SP drop one piece | bytes, struct, tokenize | bytes, struct, tokenize | yes |

The two reordering perturbations fired bytes and struct while leaving
tokenization untouched. That is the "different spelling, same result" case,
and the comparison distinguishes it rather than collapsing it.

**How small a difference can it see?** Down to one float32 ulp. A single score
in a trained SentencePiece model was moved by one ulp
(`-10.3556652 → -10.3556662`, delta 9.54e-07) and both the byte and struct
layers caught it. That control is only meaningful because the probe itself was
verified first: an earlier attempt moved the score by one *float64* ulp, which
rounds straight back to the same float32 on storage, so the file never changed
and the control reported "identical" while proving nothing. The probe now
re-reads the stored value and asserts it actually moved before the verdict is
allowed to count.

Beyond the file-level controls, the matrix trains **end-to-end control arms
that must differ** (a vocabulary size off by one), so the whole
train-then-compare pipeline is exercised, not only a file the harness edited
itself. Those arms came back DIFFERS for both trainers.

## Two harness bugs found by these controls

Recorded because both would have produced a confident wrong answer.

**1. The harness manufactured a SentencePiece difference.** SentencePiece
bakes `trainer_spec` into the `.model` file, `model_prefix` and input paths
included. Giving each run its own output directory changed the `.model` bytes
while the vocabulary was identical, and SentencePiece looked nondeterministic
on *every* axis. A proto-level diff of two "nondeterministic" runs found
exactly one differing field: `model_prefix`. Fixed at the source (each run
trains with cwd set to its own directory under the constant relative prefix
`sp`), not by normalizing the field away in the comparator, which would have
blinded the byte layer to real changes.

**2. A check that could not fail.** An early seed test compared two artifacts
that had both failed to be produced; `shasum < missing` returned the same
empty result on both sides and the test printed "IDENTICAL". The comparison
helper now refuses to give a verdict when an artifact is missing or empty,
and that refusal is itself exercised as a control.

## Findings: Hugging Face `tokenizers` 0.23.2, BPE

One core, `RAYON_NUM_THREADS=1`, fresh process per run, M4 arm64.

| axis | comparison | verdict |
|---|---|---|
| repeated runs | 5 runs, identical settings | IDENTICAL |
| vocabulary size | 2 runs at vocab 1,000 | IDENTICAL |
| vocabulary size | 2 runs at vocab 32,000 | IDENTICAL |
| corpus order | forward vs reversed vs rotated | IDENTICAL |
| corpus order | 2 runs each within reversed, rotated | IDENTICAL |
| control (vocab+1) | must differ | DIFFERS |

All three artifacts (`tokenizer.json`, `vocab.json`, `merges.txt`) are
byte-identical across every one of these runs. Hugging Face embeds no paths
or timestamps, so byte equality is meaningful straight out of the box.

**And the stability is structural, not luck.** A serializer that happened to
iterate a hash map in a stable order would give byte-identical files today and
stop doing so on a library upgrade, so it matters which one this is. Measured
on a trained artifact: the vocabulary is written in strict **id order**, ids
contiguous `0..n-1`, in both `tokenizer.json` and `vocab.json`, and that order
is demonstrably **not** alphabetical (the keys are not sorted), which is what
would otherwise confound the check. `merges.txt` preserves training order,
most frequent merge first (`Ġ t`, `h e`, `Ġ a`). So the serializer imposes a
total order on output rather than inheriting one from an iteration order, and
the byte equality rests on that rather than on a coincidence.

Notably, **corpus order does not reach the result at all** for HF: forward,
reversed and rotated shard lists produce byte-identical artifacts.

## Findings: `sentencepiece` 0.2.2, BPE

One core, `--num_threads 1`, fresh process per run, M4 arm64, after the
harness fix.

| axis | comparison | verdict |
|---|---|---|
| repeated runs | 5 runs, identical settings | IDENTICAL |
| vocabulary size | 2 runs at vocab 1,000 | IDENTICAL |
| vocabulary size | 2 runs at vocab 32,000 | IDENTICAL |
| corpus order | forward vs reversed vs rotated | bytes DIFFER, vocabulary IDENTICAL |
| corpus order | 2 runs each within reversed, rotated | IDENTICAL |
| control (vocab+1) | must differ | DIFFERS |

The corpus-order row is the one that needs reading carefully, and it is the
"files differ but every input tokenizes identically" case:

- `sp.model` bytes differ, because `trainer_spec.input` records the file list
  in the order it was given.
- `sp.vocab` is **byte-identical**.
- piece sequence, scores, types and the token→id map are **identical**.
- 400 held-out lines tokenize to the **identical 11,629 tokens**.

So corpus order changes the recorded metadata, not the vocabulary.

`random_seed` is **not** a settable field in sentencepiece 0.2.2 — it is
rejected as an unknown `TrainerSpec` field — so there is no seed to pin.

### The one place SentencePiece is genuinely not reproducible: sampled mode

Everything above trains on the whole corpus. SentencePiece also has a sampling
path, `input_sentence_size` with `shuffle_input_sentence`, which is the
commonly recommended setting for large corpora. **It does not reproduce.**

Two runs at identical settings (50,000 sentences sampled from 192,026,
shuffled, one thread, same corpus, same order, fresh processes):

| layer | result |
|---|---|
| `sp.model` bytes | DIFFER |
| `sp.vocab` bytes | DIFFER |
| pieces unique to run A / run B | 946 / 946 |
| token→id entries differing | 6,949 |
| held-out lines tokenizing to different **pieces** | 135 of 400 |

This is not a renumbering and not a metadata difference. The vocabularies
genuinely disagree about how text splits:

```
line 5   A: ▁* ▁''[[ Air bor ne ▁E arly ▁War ning ]] ▁Air craft '';
         B: ▁* ▁''[[ Air bor ne ▁Early ▁War ning ]] ▁Air craft '';
line 6   A: ▁* ▁'' Tran sport ▁Air craft '';
         B: ▁* ▁'' T ransport ▁Air craft '';
```

The fail-first arm confirms the sampling actually did something: a sampled run
also differs from the full-corpus run, so "DIFFER" here is not an artifact of a
comparison that differs against everything.

Because 0.2.2 exposes no `random_seed`, **this cannot be pinned from the
API.** The only way to keep SentencePiece reproducible is to never use the
sampling path: train on the whole corpus, with `input_sentence_size=0`.

## Findings: the unigram trainers

One core, one thread, 16 MB, fresh process per run, M4 arm64. Unigram scores
with floating-point EM, so this is where float accumulation would be expected
to show. It does show — but not in the trainer the reasoning would predict.

| trainer | two runs, identical settings | verdict |
|---|---|---|
| `sentencepiece` 0.2.2 unigram | `sp.model`, `sp.vocab` | IDENTICAL |
| HF `tokenizers` 0.23.2 unigram | `tokenizer.json`, `unigram.json` | **DIFFER** |

It is the opposite way round from BPE: Hugging Face is the reproducible one at
BPE and the unreproducible one at unigram.

**What differs in HF unigram is the scores, not the vocabulary.** The token
set is identical — zero tokens appear in one run and not the other — and
1,953 of 8,000 entries (24.4%) disagree on their log-probability:

| statistic | value |
|---|---|
| median absolute delta | 1.78e-15 |
| maximum absolute delta | 0.0054 |
| deltas above 1e-3 | 39 |
| deltas above 1e-2 | 0 |

That is float-accumulation noise, not a different answer.

**Does it reach the tokenization?** Almost never, but not never. On 54,417
held-out lines (5.5 MB, disjoint from the training slice):

- total token count identical: 1,840,337 both ways
- **2 lines** tokenize to different **pieces**
- 281 lines agree on pieces but differ in **ids**, because equal scores get
  ordered differently

The two genuine differences are an exact tie, a run of dashes regrouping:

```
A: - - ---- ---- ---- ---- ---- -    ---- ---- ---- ---- ----
B: - - ---- ---- ---- ---- ---- ---- -    ---- ---- ---- ----
```

This is the finding the brief asked to be separated, and it lands between the
two clean cases. It is not "the files differ but everything tokenizes
identically" — 2 lines in 54,417 really do tokenize differently, and 281 more
get different ids. For a bitwise-reproducibility claim, "almost always the
same" is a failure, not a pass.

The comparison was not blind when it reported 2: the same comparison on a
genuinely different vocabulary (this unigram model against the BPE model)
flags 47,441 of the same 54,417 lines.

## Thread count, and the unigram trainers

Run on one rented RunPod CPU pod (16 vCPU, `runpod/base:1.3.1-ubuntu2204`,
$0.48/hr), because one core per agent on the shared Mac means a trainer
spinning up every core cannot be measured there.

**First, the knob is real.** A thread axis is worthless if the setting never
reaches the trainer, so both knobs were checked before the runs were believed:

- The Hugging Face native module `tokenizers.abi3.so` contains
  `RAYON_NUM_THREADS`, `RAYON_RS_NUM_CPUS`, `TOKENIZERS_PARALLELISM` and
  `rayon-core`, so the trainer does parallelize through rayon and reads that
  variable. Each run is a fresh process, because rayon builds its global pool
  once, on first use, and reads the variable at that moment.
- SentencePiece echoes the setting back in its own `trainer_spec`
  (`num_threads: 4`), so the value is accepted rather than silently dropped.

PENDING — filled in from the pod leg.

## Reproducing

See `docs/lanes/LANE_STATUS_lane-tokenizer-trainer-determinism.md`.
