# Are tokenizer vocabulary trainers bitwise reproducible?

Measured 2026-09-16 on lane `lane/tokenizer-trainer-determinism`. The question
decides whether mojolearn builds its own vocabulary trainer or pins an
existing one. It was measured, not reasoned about.

> **The answer is BUILD, and this file first said PIN.** An earlier revision
> recommended pinning Hugging Face `tokenizers` BPE on the strength of the
> tables below. That recommendation is **withdrawn**; see **Resolution** at the
> end. The measurements stand, but every one of them was taken in a single
> trainer configuration, and the axis this lane did not move is the one that
> decides the question. mojolearn's own deterministic BPE trainer is on main
> (`lane/bpe-vocab-trainer`, `bench/results/bpe_trainer/README.md`).

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

### What bounds that result is the CONFIGURATION, not the version

Every row above was measured in one trainer configuration, byte-level BPE with
no continuation prefix. This lane wrote down "do not generalize a trainer from
one model type" after the unigram result, and then generalized across
configurations *of* one model type anyway. Moving that axis gives a different
answer.

`tools/tokdet_prefix_config.py` trains three arms in **eight fresh processes**
each, one core, `RAYON_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`, on a
corpus the script generates itself from shared word stems so that count ties
are common. Fresh processes are the point, because the cause is a per-process
hash seed that a single process cannot see.

| arm | distinct vocabularies in 8 processes |
|---|---|
| `BpeTrainer`, no prefix (**the control**) | **1** |
| `BpeTrainer`, `continuing_subword_prefix="##"` | **8** |
| `WordPieceTrainer` | **8** |

The no-prefix arm is the fail-first control and the reason the other two can be
read at all. It came back 1 of 8, so neither the corpus nor the harness is the
cause of what the other two arms show.

**The vocabularies genuinely disagree**, and this is not a renumbering. Between
two `##` runs, 6 to 8 tokens are in one vocabulary and not the other, 15 shared
tokens carry different ids, and the merge list differs as a **set** and not only
as a sequence. The differing tokens reach the tokenization, so two runs of the
same trainer on the same corpus at the same settings split the same word
differently.

```
'ken'    run A  k ##en          run B  k ##e ##n
'buil'   run A  buil            run B  bu ##i ##l
'stan'   run A  stan            run B  st ##a ##n
```

Those counts move between invocations of the check itself, which is the
signature of the cause rather than noise in the measurement.

**The mechanism is a hash iteration order used as a tie-break.** `tokenizers`
hands out ids to the prefixed continuation tokens while iterating a hash map,
and those ids are what settles a count tie between two candidate merges. A
version pin does not reach it, a thread pin does not reach it, and it is
invisible from the API. `lane/bpe-vocab-trainer` found this independently by
reading `bpe/trainer.rs`; the numbers here are a separate replication, on a
different corpus, run after that lane merged.

So the honest form of the HF BPE row is **"reproducible in the byte-level,
no-prefix configuration"**. That happens to be the configuration mojolearn's own
GPT-2-format tokenizer consumes, so the safe region is the region we would have
used. The problem is not that the pin would fail today. The problem is that the
boundary of the safe region is undocumented, unpromised, and silent, and a user
who steps over it gets a different vocabulary with no error.

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

## Does a pinned version actually hold?

"Pin the version" is only a real answer if it is the *version* doing the work.
If a vocabulary changed on every upgrade, pinning would be less a condition
than a cage, so both trainers were retrained on the same corpus under older
releases. Every version trains at equal directory depth, because SentencePiece
embeds its relative input paths and unequal depths would manufacture a
difference again.

**Hugging Face BPE is byte-identical across three releases.**

| version | `tokenizer.json` | `merges.txt` |
|---|---|---|
| 0.23.2 (reference) | `4724bad6ac72632c` | `68bc8f9f5493ca43` |
| 0.22.1 | `4724bad6ac72632c` SAME | `68bc8f9f5493ca43` SAME |
| 0.20.3 | `4724bad6ac72632c` SAME | `68bc8f9f5493ca43` SAME |

Same vocabulary, same merge sequence, same tokenization. The artifact does not
move across the range tested.

**SentencePiece keeps the vocabulary but not the artifact.** Between 0.2.0 and
0.2.2 the `sp.model` bytes differ while `sp.vocab` is byte-identical. The
differing field is not the vocabulary and not `trainer_spec`, which is
identical; it is `normalizer_spec.precompiled_charsmap`, the NFKC
normalization table baked into the model, which grew from 237,561 to 240,007
bytes.

That one needed checking rather than assuming, because a changed normalizer
can change how text normalizes and therefore how it tokenizes. It does not
here: across the 54,417-line held-out sample both versions produce identical
pieces and identical ids, 1,660,030 tokens each, while the fail-first arm on
the same text flags 41,010 lines.

sentencepiece 0.1.99 has no CPython 3.12 wheel. That is recorded as **no
verdict**, not as a pass.

Three releases is evidence, not a guarantee about future ones. But it means a
pinned Hugging Face BPE vocabulary is reproducible *because* of the algorithm
and serializer, not because of a frozen build.

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

Box: 16 vCPU x86_64, Linux 5.15 glibc 2.35, CPython 3.14.7, `tokenizers`
0.23.2, `sentencepiece` 0.2.2. The box cut its corpus from the same R2 enwik8
and produced **byte-identical shards** to the Mac's, which is what makes the
cross-architecture comparison below mean anything.

**Hugging Face BPE is identical at every thread count.**

| comparison | verdict |
|---|---|
| 1 vs 2 vs 4 vs 8 vs 16 threads | IDENTICAL |
| repeats within t=2, t=4, t=8, t=16 | IDENTICAL |
| corpus order, on x86 | IDENTICAL |
| vocabulary 1,000 and 32,000, on x86 | IDENTICAL |
| control (vocab+1) | DIFFERS |

**SentencePiece BPE's vocabulary is thread-invariant.** This run predates the
`model_prefix` fix, so its `sp.model` byte layer is contaminated and every
SentencePiece byte row reads DIFFERS for that reason alone; those rows are not
evidence of anything. The struct and tokenize layers are unaffected, and at
every thread count from 1 to 16, and across repeats and orders, they report:
identical piece set, identical piece→id map, identical sequence, zero score
differences, zero tokenization differences.

**The thread knob demonstrably engaged**, so the axis is not vacuous. Wall
times at vocabulary 8,000: Hugging Face 7.8 s at one thread against 4.0 s at
four; SentencePiece unigram 87.7 s at one thread against 42.4 s at sixteen.

### Cross-architecture

The same configuration, trained on Apple M4 arm64 and on x86_64 Linux from
byte-identical shards, produces byte-identical Hugging Face BPE artifacts:

| artifact | arm64 | x86_64 |
|---|---|---|
| `tokenizer.json` | `4724bad6ac72632c` | `4724bad6ac72632c` |
| `vocab.json` | `52f3ce45d4a75f86` | `52f3ce45d4a75f86` |
| `merges.txt` | `68bc8f9f5493ca43` | `68bc8f9f5493ca43` |

So an HF BPE vocabulary is reproducible across architectures, not merely
across runs on one machine.

### One gap, and it is mine

The unigram matrix on the pod produced **no verdicts**. Every unigram run
completed — the timings are in `unigram_console.txt` — but the harness then
crashed in its own self-test, because the Hugging Face perturbations assumed a
`merges` list and a Unigram model has scores instead. `tokdet_matrix.py` then
died on the missing file rather than recording a failed self-test, taking all
the finished comparisons with it.

Both bugs are fixed: Unigram models get their own controls, and a self-test
that produces no JSON is now recorded as a failure that marks the matrix
unvalidated instead of aborting it.

The unigram **thread** axis is therefore unmeasured. It was not re-rented,
because Hugging Face unigram is already disqualified at one thread and
SentencePiece unigram's thread behavior cannot change the recommendation. The
one-thread unigram results above stand on their own. The resume command is in
the lane status file.

### Cost

Pod `5dd31t9s9o3j2k`, 16 vCPU at $0.48/hr, billed 1,905 s from create to
verified delete: **$0.2540**. Deleted and verified gone (HTTP 204, then 404,
then absent from the pod listing). $0.48/hr is the 16 vCPU price;
`docs/RUNPOD_CPU_LEG.md` now records that the rate scales with `--vcpu` and
that its $0.24 figure is the 8 vCPU one.

## Summary

| trainer and model | bitwise reproducible | where it breaks |
|---|---|---|
| HF `tokenizers` BPE, no prefix | **yes** | nowhere measured on this axis |
| HF `tokenizers` BPE, `##` prefix | **no** | 8 distinct vocabularies in 8 processes |
| HF `tokenizers` WordPiece | **no** | 8 distinct vocabularies in 8 processes |
| HF `tokenizers` unigram | **no** | scores wobble run to run |
| `sentencepiece` BPE, whole corpus | **yes** | `.model` carries version and path metadata |
| `sentencepiece` BPE, sampled | **no** | the sampling draw, and it cannot be pinned |
| `sentencepiece` unigram | **yes** (1 thread) | thread axis unmeasured |

Hugging Face BPE held on every axis this lane originally moved. Repeated runs,
vocabulary 1,000 and 8,000 and 32,000, forward and reversed and rotated corpus
order, 1 through 16 threads, arm64 against x86_64, and three library versions.
It does **not** hold across the axis this lane did not originally move, the
trainer's own configuration.

## RETIRED: what pinning would have required

**This section and the recommendation that followed it are superseded.** They
are kept because a retracted argument that leaves no trace is worse than a wrong
one, and because the conditions below are still the right conditions for anyone
who uses Hugging Face `tokenizers` directly.

Pinning Hugging Face `tokenizers` BPE would have required stating that the model
is BPE and never unigram, that the corpus bytes and vocabulary size are fixed,
and that the version is pinned as hygiene. Thread count, corpus file order and
architecture all turned out not to matter. Two warnings attach and still stand.
Never use SentencePiece's `input_sentence_size` sampling, which does not
reproduce and has no `random_seed` to pin in 0.2.2, and never extend a
reproducibility claim to a Hugging Face unigram vocabulary.

A third condition was missing from that list, and it is the one that retires it.
The trainer must be in the byte-level, no-prefix configuration, because
`continuing_subword_prefix` and `WordPieceTrainer` both leave the reproducible
region silently. A condition nobody can check from the API is not a condition a
library should ship behind.

## Resolution: build it, and `lane/bpe-vocab-trainer` was right

**This lane recommended pinning Hugging Face BPE. That recommendation is
withdrawn.** `lane/bpe-vocab-trainer` merged a deterministic byte-level BPE
trainer to main on 2026-09-16 (`dd41ba82e`), and it is the right call. Three
reasons, in the order they matter.

**1. The pin's guarantee is a coincidence of one configuration, and the boundary
is invisible.** Measured above. Hugging Face does not document or promise
reproducibility, and the region where it holds is bounded by a hash iteration
order that no version pin, thread pin or API flag reaches. Depending on an
undocumented property means depending on something nobody has agreed to keep.

**2. Under a pin we would inherit a defect we are not allowed to write.** Using
a hash map's iteration order to settle a tie is exactly the class of thing this
project's rules forbid, and the same rules forbid reproducing a reference
library's bug. Pinned, we could neither fix it nor gate against it. Owned, the
tie-break is a stated total order in our source, highest count then smallest
`(left_id, right_id)`, which is a sentence a reader can check against the code.

**3. The two options differ in what a USER can verify, and that is the real
difference.** Under a pin, a user verifies reproducibility by running a closed
trainer twice and diffing the bytes. That is an existence check over whichever
configurations they happened to try, it says nothing about the one they did not
try, and it cannot distinguish "reproducible by construction" from "reproducible
so far". Under the owned trainer, a user runs `pixi run check-bpe-trainer` and
gets a Mojo implementation held **file byte for file byte** against an
independent Python implementation of the *stated* algorithm, plus a sabotage arm
watched failing, plus `n_ties_broken` carried out of the trainer so the gate
fails if no fixture exercised the rule the sabotage reverses. The property is
falsifiable in our harness rather than observed in someone else's binary.

There is a fourth reason that is about distribution rather than determinism.
mojolearn ships a byte-level BPE tokenizer and, since 2026-09-15, ships no
vocabulary at all. Without a trainer, the only route from a user's corpus to a
vocabulary our own tokenizer can load runs through a third-party library. That
is not third-party data in the wheel, and it was never the concern the
no-third-party-data rule was written for, but it does mean the only usable path
through our own component would depend on something we do not control and cannot
gate.

### What the pin argument got right, and what carries forward

The measurements are not withdrawn, only the conclusion drawn from them.

- **Hugging Face BPE, no prefix, is reproducible across repeats, vocabulary
  size, corpus order, 1 through 16 threads, arm64 against x86_64, and three
  library versions.** That is the interop baseline the trainer's
  `tokenizer.json` round trip is measured against, and it is why that round trip
  is a meaningful check rather than a comparison against a moving target.
- **Hugging Face unigram is not reproducible**, 1,953 of 8,000 scores differ and
  2 of 54,417 held-out lines retokenize. `lane/bpe-vocab-trainer`'s phase 3
  already cites these numbers. A deterministic unigram trainer is the one nobody
  else can offer, and it is where the differentiator actually lives.
- **SentencePiece unigram is reproducible at one thread** and its BPE
  vocabulary is thread-invariant, so SentencePiece remains the cheaper answer
  for unigram until we write one.
- **Never use SentencePiece's sampling path.** It does not reproduce and 0.2.2
  exposes no `random_seed`.
- **The x86_64 leg the trainer still owes can reuse this harness.**
  `tools/tokdet_corpus.py` already cut byte-identical corpus shards on Apple M4
  arm64 and on x86_64 Linux from the same R2 object, and
  `tools/runpod_cpu_leg.sh --lane tokdet` is the rented-CPU path that carried it.
  Nothing new has to be built to close that gap.

### What this lane cost, and why it was still worth it

One rented CPU pod at $0.2540 and a day of local one-core runs, to reach a
conclusion that a separate lane reached by writing the trainer. The measurements
survive as the ecosystem baseline and as the bound on the interop claim, and the
negative result is on the record so that "why not just pin Hugging Face" has a
measured answer the next time it is asked.

## What this evidence does not cover

Stated so the recommendation is not read wider than it was measured.

- One corpus, 16 MB of enwik8. English prose and wiki markup. A code corpus
  was not tried, and tie density differs by corpus kind.
- Nothing at the scale a real vocabulary is trained at. 16 MB, not 100 GB.
- The unigram thread axis, which was lost to a harness bug and deliberately
  not re-rented.
- Future library versions. Three releases agreeing is evidence, not a promise.
- The configuration arms ran on a small generated corpus at vocabulary 300, not
  on enwik8 at 8,000. That is the scale at which count ties are dense enough to
  reach the tie-break, and it is why the no-prefix control is trained on the
  same corpus rather than assumed. Whether the same mechanism bites a
  32,000-token `##` vocabulary on 16 MB of prose is untested, and it does not
  need to be, because a trainer that is nondeterministic anywhere cannot be
  pinned as deterministic.

## Reproducing

See `docs/lanes/LANE_STATUS_lane-tokenizer-trainer-determinism.md`.
