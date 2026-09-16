# LANE STATUS: `lane/tokenizer-trainer-determinism`

**Question.** Are the two vocabulary trainers the ecosystem actually uses —
Hugging Face `tokenizers` and SentencePiece — bitwise reproducible? The answer
decides whether mojolearn builds its own vocabulary trainer or pins one of
theirs. Decision support, not a build. **No vocabulary trainer was written and
none should be written on this branch.**

**Results:** `bench/results/tokenizer_determinism/README.md`.

## State

- Harness committed: `tools/tokdet_corpus.py`, `tools/tokdet_train_hf.py`,
  `tools/tokdet_train_sp.py`, `tools/tokdet_compare.py`,
  `tools/tokdet_matrix.py`, `tools/tokdet_pod_cmd.sh`.
- Falsification gate passes for both trainers, in both directions (an exact
  copy compares identical; six one-at-a-time perturbations each trip the layer
  they must). Never trust a verdict from this harness without it.
- Local axes (repeats, vocabulary size, corpus order) measured at one core.
- Thread-count axis and the unigram trainers measured on one rented CPU pod.
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
Scratchpad (corpus, venv, run artifacts, none of it committed):
`/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/tokdet`

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
- **Must be verified deleted.** `bash tools/runpod_cpu_leg.sh list` should show
  no `mojolearn-cpu-tokdet-*`; `reap <POD_ID>` if one survives.

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
