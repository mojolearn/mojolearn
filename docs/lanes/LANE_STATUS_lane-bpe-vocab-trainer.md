# LANE STATUS: `lane/bpe-vocab-trainer`

A bitwise-deterministic **byte-level BPE vocabulary trainer**, in Mojo, beside
the tokenizer. Phase 1 of three (BPE, WordPiece, unigram). 2026-09-16.

## Resume

```bash
cd /Users/andrewhendel/mojolearn-wt/bpe-trainer          # the lane worktree
git rev-parse --abbrev-ref HEAD                          # lane/bpe-vocab-trainer

pixi run check-bpe-trainer            # THE GATE: Mojo vs Python, byte for byte
pixi run check-bpe-trainer-sabotage   # the negative control; MUST fail

python3 tools/bpe_trainer_determinism.py          # every axis, 15/15
python3 tools/bpe_trainer_interop.py build/bpe_trainer --self-test \
    --python <HFENV>/bin/python                   # the tokenizer.json round trip

# train a vocabulary from the command line
mojo run -I . tokenizer/train/train_main.mojo OUT 32000 2 corpus1.txt corpus2.txt
```

The Hugging Face environment is a **throwaway venv outside the repo**
(`python3 -m venv hfenv && hfenv/bin/pip install tokenizers==0.23.2`).
`tokenizers` is not a mojolearn dependency and nothing about it ships.

Gates before merge, all green: `docs_facts --check`,
`packaging/wheel_ci.py pins .`, `packaging/wheel_ci.py inventory python/mojolearn`.

## What shipped

| file | what it is |
| --- | --- |
| `tokenizer/train/bpe_train.mojo` | the trainer |
| `tokenizer/train/emit.mojo` | the two output formats |
| `tokenizer/train/train_main.mojo` | the command line |
| `tokenizer/checks/trainer_check.mojo` | THE GATE, plus the sabotage arm |
| `python/mojolearn/_bpe_trainer.py` | an independent second implementation |
| `python/mojolearn/tokenizer.py` | `BpeVocabularyTrainer`, `TrainedBpeVocabulary` |
| `tools/bpe_trainer_determinism.py` | the axis matrix |
| `tools/bpe_trainer_interop.py` | the Hugging Face round trip, with its control |
| `tools/identity_break.py` | the `bpe-trainer` lane |
| `bench/results/bpe_trainer/README.md` | **the evidence** |

## The claim, stated correctly

**Not** "bitwise identical across GPU vendors". Vocabulary training is
host-only in every library — Hugging Face, SentencePiece and tiktoken all
train on a CPU, because counting, sorting and merging is not a matmul
workload. There is no GPU path in `tokenizer/`, so there is no vendor column
and nothing is owed to a GPU record. The claim is:

> The same corpus and config produce the same vocabulary bytes on any machine
> and architecture.

**Owed:** the x86_64 leg. Measured on arm64 only; nothing was rented.

## Evidence, in one line each

- **Mojo vs an independent Python implementation, byte for byte**, both
  formats, three fixtures: PASS, with round trip and in-process repeat.
- **Sabotage seen to fail**: 6/6 comparisons diverge, with the tie-break's own
  signature in the diff.
- **Determinism 15/15 rows**: repeats, three vocabulary sizes, corpus order
  (forward/reversed/rotated), `min_frequency`, the tie corpus; both
  must-differ controls fire.
- **Interop 25/25 on both `tokenizer.json` shapes**, with three perturbation
  controls each seen to fail first.

Full numbers: `bench/results/bpe_trainer/README.md`.

## Why BPE was the cheap case

BPE selects over **integer counts**. Nothing in the selection is a float,
there is no score to normalize and no iterative refinement, so the *only*
thing that can vary between two correct implementations is **which pair wins a
tie**. A total order settles that permanently: highest count, then the
smallest `(left_id, right_id)`. That is the whole determinism story, and it
costs a comparison.

So deterministic BPE does not buy a property the ecosystem lacks — Hugging
Face BPE already had it, in the configuration that was measured. What it buys
is that the property is **ours by construction and verified in our own
harness**, rather than inherited and hoped for. The next two phases are where
the differentiator actually lives.

**One real catch, and it is the reason `n_ties_broken` is carried out of the
trainer and asserted non-zero.** The sabotage reverses the tie-break and
nothing else, so on a corpus that never ties it is **inert** and a passing gate
would mean nothing. `ties.corpus` is engineered to tie, and the gate fails if
no fixture broke one.

## Phase 2: WordPiece

**Headline: the selection can be exact integer arithmetic — but that answer is
less useful than it sounds, because no shipping implementation uses the
criterion everyone quotes.**

- The commonly stated criterion `count(ab) / (count(a)·count(b))` never needs
  division. Cross-multiply: `a·e·f > d·b·c`. Counts are strictly positive, so
  the argmax is preserved *exactly*. It is also strictly better than the float
  form, which can round two distinct rationals to the same double and
  manufacture ties.
  **Overflow is real and computed:** each side is a product of three counts,
  so `u64` overflows above a max symbol count of ~2.64M — i.e. on any real
  corpus. Needs 128-bit products (safe to ~7×10¹²), or a float pre-filter with
  an exact 128-bit fallback.
- **But Schuster & Nakajima 2012 does not state that formula.** The paper's
  step 3 is "choose the new word unit that increases the likelihood on the
  training data the most", with the LM rebuilt each iteration. That is a
  likelihood delta, not a ratio, and comparing two of them exactly means
  comparing sums of `n·log n` terms — exact in principle, absurd in practice.
  The Google trainer was never open-sourced.
- **Hugging Face WordPiece is BPE.** Its trainer holds a `BpeTrainerBuilder`,
  trains BPE, and converts with `WordPiece::from_bpe`. No score, no division,
  no float in the file.
- **TensorFlow Text is a third algorithm**: a count threshold plus a binary
  search on that threshold. Integer counts throughout; its only float affects
  the search's stopping rule, never which token wins.

**Measured, and it qualifies our own earlier BPE finding.** `tokenizers`
0.23.2, one core, `RAYON_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`:
`WordPieceTrainer` gave **8 distinct vocabularies in 8 processes** (sizes 165
*and* 167), and `BpeTrainer` **with `continuing_subword_prefix="##"` gave 6
distinct in 6 runs**, while plain `BpeTrainer` gave 1 in 8. The cause is not
arithmetic: `bpe/trainer.rs::tokenize_words` iterates an `AHashMap` and assigns
prefixed-token ids in **hash order**, and those ids are what `Merge::cmp` uses
to break count ties. So **"HF BPE is bitwise reproducible" holds only for the
no-prefix, no-suffix configuration.**

**Cost.** Small, if we pick a criterion and say which. The merge loop, the
corpus handling, both output formats and the lane shape are already built; the
new work is the scoring comparison (128-bit cross-multiplication), the `##`
continuation prefix in the emitters, and a WordPiece `model` block. The
`tokenizer.json` shape is known: five fields, `type` optional,
`unk_token` / `continuing_subword_prefix` / `max_input_chars_per_word` /
`vocab` all **required** by the deserializer, vocab emitted by ascending id.
Encoding is greedy longest-match-first from the left, and a word that fails at
any position becomes a single unk rather than a partial tokenization.

**The decision phase 2 must make first:** which criterion we implement.
Matching HF means implementing BPE and calling it WordPiece. Matching TF Text
means the threshold search. Implementing the ratio means matching neither.
That choice should be made deliberately, and stated in the docs, because it
determines what our vocabulary is compatible with.

## Phase 3: unigram — the one worth the most

Unigram selects on **likelihood**: an EM loop over float log-probabilities
with a Viterbi lattice and a pruning step. Hugging Face's is measurably
nondeterministic (1,953 of 8,000 scores differ; 2 of 54,417 held-out lines
retokenize), so **a deterministic unigram trainer is a claim nobody else can
make.**

**Where the floats actually are:**

1. **The E-step expected-count accumulation.** HF chunks sentences by
   `len / current_num_threads()` and combines with a rayon `reduce`, whose
   grouping is explicitly unspecified and whose split tree depends on
   work-stealing — so **fixing the thread count is not sufficient**.
   SentencePiece uses a fixed stride partition and a fixed index-order merge,
   so it is pinned *given* `--num_threads`, but its output is then a function
   of that flag (default 16), in `float32`.
2. **Per-sentence forward-backward**, log-domain, sequential. Carries a hard
   `vmax > vmin + 50` truncation — a discontinuity that any reimplementation
   must reproduce bit for bit.
3. **The M-step**, a digamma-based update, sequential in id order; inherits
   its determinism entirely from the E-step.
4. **The pruning loss and its ranking.** The cut is a hard boundary, so one
   transposition puts one piece in and another out, which changes the model for
   the next EM round and therefore every later count. SentencePiece has an
   explicit total-order tie-break; **HF's sort is float-only with none**.

**A non-float cause too**, which matters because it is cheap to fix and easy to
miss: HF's `finalize` iterates an `AHashSet` and hands out `min_score_penalty`
increments in **hash order**. That is a per-process source of differing scores
with nothing to do with reductions, and it plausibly explains the large tail of
our measured deltas (max 5.4e-3) while reduction noise explains the median
(1.78e-15). **Those two causes have not been separated.**

**Do this cheap measurement before building anything.** Rerun HF unigram with
`TOKENIZERS_PARALLELISM=false`, which takes the serial fold and removes every
rayon-derived cause; and rerun SentencePiece at `--num_threads=1` and `=16`,
twice each. **If single-threaded HF unigram turns out bitwise stable, phase 3
collapses from a ~4,000-line trainer to a pinned flag and a documented
constraint** — the same shape the BPE decision took. This is hours, not days.

**If we do build it**, roughly 3,500–5,500 lines of Mojo: seed vocabulary
(suffix array + LCP, maximal repeats) 1,500–2,500; lattice, Viterbi,
forward-backward, n-best(2) 600–900; **bit-exact transcendentals**
(`exp`, `log`, `log1p`, `digamma`) 400–800; pinned reduction, plus an optional
fixed-point posterior accumulator 150–500; M-step, corrected loss and
serialization 400–600; verification harness 300–500.

- **Genuinely hard:** the transcendentals, and choosing the fixed-point scale.
  A fixed-point posterior accumulator is the single highest-value change
  available, because it makes the E-step sum **exactly order-independent**
  rather than merely order-pinned — the summand is `integer count × posterior
  in [0,1]`, which is bounded and non-negative.
- **Merely tedious:** everything else.
- **The primary risk is that there is no oracle.** Determinism *changes the
  result*, so we could not validate by "produces the same vocabulary as
  SentencePiece" — we would be deliberately not producing it. HF also carries
  an open loss defect (`alternatives.len()` where SentencePiece uses
  `alternatives[i].size()`), which our rules forbid reproducing, and SP's own
  answer varies with `--num_threads`. An unnoticed math error would surface
  only as a worse tokenizer, and noticing costs a model training run.
- **Secondary risk: cost per cell.** Unigram training is orders of magnitude
  more expensive than BPE, so the axis matrix is CPU-hours, not seconds.

## What phase 1 leaves reusable

Most of the scaffolding, which is why phases 2 and 3 are cheaper than they
look:

- **Corpus handling.** Documents pre-tokenized alone, deterministic generated
  corpora, nothing committed. Reusable unchanged.
- **The output formats.** `emit.mojo`'s hand-rolled JSON writer, the stated
  escaping rule and the byte-to-unicode spelling are model-agnostic; a
  WordPiece or Unigram `model` block is a new branch in one function.
- **The interop harness.** `bpe_trainer_interop.py` is parameterized by the
  emitted file and the sample set; a new model type needs a different `model`
  block and nothing else.
- **The lane shape.** Two independent implementations held byte-for-byte, an
  env/define sabotage arm, and a `n_ties_broken`-style reach counter proving
  the sabotaged rule is actually exercised. That pattern transfers directly —
  and for unigram the reach counter matters *more*, since a pruning tie-break
  that never fires would make its sabotage inert in exactly the same way.
- **What does not transfer:** the "no floats anywhere" property. Phase 3 will
  need the bit-exact transcendental work instead, which is the libm-free path
  this repo has walked once already.
