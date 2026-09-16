# LANE STATUS: `lane/tokenizer-trainer-determinism`

**CLOSED 2026-09-16, as a NEGATIVE RESULT.** This lane recommended pinning
Hugging Face `tokenizers` BPE instead of writing our own vocabulary trainer.
**That recommendation is withdrawn.** `lane/bpe-vocab-trainer` merged a
deterministic byte-level BPE trainer to main the same morning (`dd41ba82e`),
and it is right. Nothing on this branch contradicts it any more, and nothing
here should be read as an argument against it.

**Question.** Are the two vocabulary trainers the ecosystem actually uses,
Hugging Face `tokenizers` and SentencePiece, bitwise reproducible? The answer
decides whether mojolearn builds its own vocabulary trainer or pins one of
theirs. Decision support, not a build. No vocabulary trainer was written on
this branch.

**Results:** `bench/results/tokenizer_determinism/README.md`.

## Why the recommendation was wrong

Every axis this lane moved was moved inside **one trainer configuration**,
byte-level BPE with no continuation prefix. `tools/tokdet_prefix_config.py`
moves the configuration axis, in eight fresh processes per arm at one core:

| arm | distinct vocabularies in 8 processes |
|---|---|
| `BpeTrainer`, no prefix (the fail-first control) | **1** |
| `BpeTrainer`, `continuing_subword_prefix="##"` | **8** |
| `WordPieceTrainer` | **8** |

The disagreement is genuine and not a renumbering. Between two `##` runs, 6 to
8 tokens are in one vocabulary and not the other, the merge list differs as a
**set**, and the same word encodes to different pieces (`buil` becomes `buil`
in one run and `bu ##i ##l` in the other). The cause is a hash iteration order
used to hand out continuation-token ids, and those ids settle count ties; no
version pin or thread pin reaches it. `lane/bpe-vocab-trainer` found the same
thing independently by reading `bpe/trainer.rs`. This is a separate
replication, on a different corpus, run after that lane merged.

This lane had already written down "do not generalize a trainer from one model
type" and then generalized across configurations *of* one model type. That is
the lesson worth carrying, and it is the same shape as the trap already in the
list below.

The deeper reason, which does not depend on the numbers: a pin and an owned
implementation differ in **what a user can verify**. A pinned trainer is
verified by running a closed binary twice and diffing, which is an existence
check over the configurations that user happened to try. The owned trainer is
verified by `pixi run check-bpe-trainer`, which holds Mojo against an
independent Python implementation of a *stated* algorithm file byte for file
byte, with a sabotage arm watched failing and `n_ties_broken` asserted non-zero
so a passing gate cannot be vacuous. A hash-order tie-break is also exactly the
kind of defect this project forbids writing and forbids reproducing; pinned, we
could neither fix it nor gate against it.

## What carries forward to `lane/bpe-vocab-trainer`

- The ecosystem baseline the trainer's `tokenizer.json` round trip is measured
  against. HF BPE with no prefix is byte-identical across repeats, vocabulary
  size, corpus order, 1 through 16 threads, arm64 against x86_64 and three
  library versions, so that round trip compares against a fixed target.
- The unigram numbers phase 3 already cites (HF unigram, 1,953 of 8,000 scores
  differ, 2 of 54,417 held-out lines retokenize; SentencePiece unigram
  identical at one thread).
- **The x86_64 leg the trainer owes needs nothing new built.**
  `tools/tokdet_corpus.py` already cut byte-identical shards on Apple M4 arm64
  and on x86_64 Linux from the same R2 object, and
  `tools/runpod_cpu_leg.sh --lane tokdet` is the rented-CPU path that carried
  it. Ask before renting.
- The standing warnings. Never use SentencePiece's `input_sentence_size`
  sampling, which does not reproduce and has no `random_seed` in 0.2.2. Never
  extend a reproducibility claim to a Hugging Face unigram vocabulary.

## State

- Harness committed: `tools/tokdet_corpus.py`, `tools/tokdet_train_hf.py`,
  `tools/tokdet_train_sp.py`, `tools/tokdet_compare.py`,
  `tools/tokdet_matrix.py`, `tools/tokdet_pod_cmd.sh`,
  `tools/tokdet_prefix_config.py` (the configuration axis, self-contained,
  carries its own fail-first control and exits non-zero when the control does
  not hold).
- Falsification gate passes for both trainers, in both directions (an exact
  copy compares identical; six one-at-a-time perturbations each trip the layer
  they must). Never trust a verdict from this harness without it.
- Local axes (repeats, vocabulary size, corpus order) measured at one core,
  for BPE and unigram, both trainers.
- Thread-count axis measured on one rented CPU pod, which also repeats every
  other axis on x86 so each axis has one coherent column.
- Gates run clean on this branch: `docs_facts --check` (13 facts),
  `packaging/wheel_ci.py pins .` (56 build scripts),
  `packaging/wheel_ci.py inventory python/mojolearn` (85 modules).

## Constraints this lane ran under

- **One core**, `nice -n 19`, own worktree, for everything local. The
  thread-count axis is the only reason a pod was involved.
- **CPU only.** No Metal, no GPU.
- Trainers live in a **throwaway venv outside the repo**. They are not
  mojolearn dependencies, no vocabulary is vendored, no corpus is committed.

## Resume

Worktree: `/Users/andrewhendel/CascadeProjects/mojolearn-wt/tokenizer-determinism`

Evidence, venv and run artifacts, none of it committed:
`/Users/andrewhendel/mojolearn-evidence/tokenizer-trainer-determinism/`. The
lane's original scratchpad was under `/private/tmp/claude-501/...` and the OS
reaped it in the 2026-09-16 crash, which is why the configuration replication
was rerun into `~/mojolearn-evidence/` instead.

```sh
E=/Users/andrewhendel/mojolearn-evidence/tokenizer-trainer-determinism
python3 -m venv $E/hfenv && $E/hfenv/bin/pip install 'tokenizers==0.23.2'
MAC_SLOTS=4 bash ~/mojolearn-evidence/tools/mac_slot.sh run nice -n 19 \
    env RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    $E/hfenv/bin/python tools/tokdet_prefix_config.py $E/run
```

Rebuild the throwaway environment:

```sh
cd /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/tokdet
/opt/homebrew/bin/python3.12 -m venv venv
./venv/bin/pip install 'tokenizers==0.23.2' 'sentencepiece==0.2.2' protobuf
```

Stage the corpus from R2 and cut it (never re-download from upstream):

```sh
cd /Users/andrewhendel/CascadeProjects/mojolearn
sh tools/dataset_store.sh pull corpus/enwik8/input.txt <scratch>/enwik8.txt
<scratch>/venv/bin/python tools/tokdet_corpus.py --source <scratch>/enwik8.txt \
    --out <scratch>/corpus16 --bytes 16777216 --shards 4
```

**Run the falsification gate first. A verdict from this harness means nothing
until this passes, and it must be watched failing on the perturbed side.**

```sh
W=/Users/andrewhendel/CascadeProjects/mojolearn-wt/tokenizer-determinism
$V $W/tools/tokdet_compare.py --kind hf --sample <scratch>/corpus16/sample.txt \
    --self-test --dir <a trained hf run dir> --tmp <scratch>/_st_hf
$V $W/tools/tokdet_compare.py --kind sp --sample <scratch>/corpus16/sample.txt \
    --self-test --dir <a trained sp run dir> --tmp <scratch>/_st_sp
```

Local axes, one core (about 4 minutes for BPE, both trainers):

```sh
nice -n 19 $V $W/tools/tokdet_matrix.py --corpus-dir <scratch>/corpus16 \
    --out <scratch>/local_bpe --python $V --threads 1 --reps 5 --axis-reps 2 \
    --base-vocab 8000 --vocabs 1000,32000 --orders rev,rot --kinds hf,sp --models bpe
```

Thread-count axis on a pod. `tools/tokdet_pod_cmd.sh` is a **template**: the
presigned corpus URL and the pinned sha256 are substituted on the Mac and the
rendered copy stays in the scratchpad, never committed. Substitute with a
literal string replace, **not `sed`** — a presigned URL contains `&`, which
`sed` expands to the whole match and silently corrupts the URL.

```sh
cd $W
URL=$(sh tools/dataset_store.sh presign corpus/enwik8/input.txt 14400)
SHA=$(awk -F'\t' '$1=="corpus/enwik8/input.txt"{print $3}' bench/results/dataset_store/manifest.tsv)
# render with python str.replace, chmod 600, then verify 0 placeholders remain
bash tools/runpod_cpu_leg.sh --lane tokdet --vcpu 16 --lease 100 --build "" \
    --cmd-file <scratch>/pod_cmd_real.sh --rent
```

## Pod

- Pod `5dd31t9s9o3j2k`, name `mojolearn-cpu-tokdet-20260916-101422`, 16 vCPU,
  `runpod/base:1.3.1-ubuntu2204`, **$0.48/hr** (not the $0.24 in
  `docs/RUNPOD_CPU_LEG.md`, which is stale for 16 vCPU).
- Created 06:14:26, bill starts there. Dead-man armed before the create;
  on-pod watchdog armed for 100 minutes.
- **Deleted and verified gone**: HTTP 204, then GET returned 404, then absent
  from the pod listing. Dead-man cancelled only after that.
- **Actual spend $0.2540**, billed 1,905 s from create to verified delete. The
  run phase itself was 1,763 s.
- Results fetched to
  `/Users/andrewhendel/mojolearn-evidence/tokenizer-trainer-determinism/2026-09-16_101422-tokdet/remote/leg_out`,
  outside the repo. Only summaries came back; the trained vocabularies stayed
  on the box. `user_cmd.sh` in that directory is the FETCHED copy, in which the
  presigned R2 URL is already redacted; the rendered copy that carried the real
  URL was never written into the repo.
- If a pod ever survives: `bash tools/runpod_cpu_leg.sh list`, then
  `reap <POD_ID>`.

## Open gap: the unigram thread axis

Every unigram run on the pod completed, and then the harness crashed in its own
self-test before computing a single comparison, so that axis has **no
verdicts**. The cause was mine: the Hugging Face perturbations assumed a
`merges` list, which a Unigram model does not have, and `tokdet_matrix.py` died
on the missing JSON instead of recording a failed self-test. Both are fixed —
Unigram models now have their own controls, and a self-test that produces no
JSON is recorded as a failure that marks the matrix unvalidated rather than
aborting it.

It was **not** re-rented, deliberately: Hugging Face unigram is already
disqualified at one thread, and SentencePiece unigram's thread behavior cannot
change the build-or-pin recommendation. The one-thread unigram results stand.

To close it (about 12 minutes on a pod, roughly $0.10):

```sh
cd $W   # render pod_cmd_real.sh as above, with the unigram matrix only
bash tools/runpod_cpu_leg.sh --lane tokdetuni --vcpu 16 --lease 45 --build "" \
    --cmd-file <scratch>/pod_cmd_real.sh --rent
```

## Traps this lane already hit

- **`sed` corrupts a presigned URL.** `&` in the replacement means "the whole
  match". Use a literal replace and then count remaining placeholders — and
  print the count, not `$?`.
- **A comparison of two missing files reports IDENTICAL.** `shasum < missing`
  gives the same empty result on both sides. The helper now refuses a verdict
  on a missing or empty artifact.
- **SentencePiece bakes `trainer_spec` into `.model`**, `model_prefix` and
  input paths included, so a harness that gives each run its own directory
  manufactures a byte difference and SentencePiece looks nondeterministic on
  every axis. Fixed at the source, not normalized away in the comparator.
- **`random_seed` is not settable** in sentencepiece 0.2.2; it is rejected as
  an unknown `TrainerSpec` field.
- A shuffled-sample control is **not** a differing arm when
  `input_sentence_size` exceeds the corpus sentence count. Check the line count
  before treating it as evidence.
- **A float64 ulp is a no-op on a float32 field.** Perturbing a SentencePiece
  `score` by one float64 ulp rounds straight back to the same float32 on
  storage, so the file never changes and the control reports "identical" while
  proving nothing. Re-read the stored value and assert it actually moved before
  letting the verdict count. Done correctly, both layers see a 1-ulp float32
  move (delta 9.54e-07).
- **Do not generalize a trainer from one model type.** Hugging Face is the
  reproducible one at BPE and the *unreproducible* one at unigram;
  SentencePiece is the reverse. Measure each model type.
- **Do not generalize a trainer from one CONFIGURATION either, which is how
  this lane reached a wrong recommendation.** HF BPE is reproducible with no
  continuation prefix and NOT reproducible with one, and the difference is
  invisible from the API. Having written the trap above, this lane then walked
  into its sibling. An axis you did not move is not an axis that does not
  exist.
- **A per-process cause is invisible inside one process.** The hash seed is
  redrawn per process, so five repeats inside one interpreter would have
  reported IDENTICAL for every arm. Every run of
  `tools/tokdet_prefix_config.py` is a fresh subprocess for that reason.

## What was found

| trainer / model | bitwise reproducible? |
|---|---|
| HF `tokenizers` 0.23.2 BPE, no prefix | yes, on every axis measured |
| HF `tokenizers` 0.23.2 BPE, `##` prefix | **no**, 8 distinct in 8 processes |
| HF `tokenizers` 0.23.2 WordPiece | **no**, 8 distinct in 8 processes |
| HF `tokenizers` 0.23.2 unigram | **no** — scores wobble, 2 of 54,417 held-out lines retokenize |
| `sentencepiece` 0.2.2 BPE, full corpus | yes (corpus order moves `.model` bytes, not the vocabulary) |
| `sentencepiece` 0.2.2 BPE, sampled | **no**, and unpinnable — no `random_seed` in 0.2.2 |
| `sentencepiece` 0.2.2 unigram | yes |
