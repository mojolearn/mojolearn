# LANE STATUS: bpe-builder-native (lane/bpe-builder-native)

**STATE 2026-09-18: WIP, step 0 (predictions registered, nothing built yet).**

Worktree `~/mojolearn-wt/bpe-builder-native`, branch `lane/bpe-builder-native`, cut from
origin/main 0e4715f7c. Evidence goes under `~/mojolearn-evidence/bpe-builder-native-sep18/`,
never in the repo. Context: `docs/lanes/LANE_STATUS_tokenized-corpus.md`.

## The brief (short)

1. Replace the dense V x V pair-count table in `tokenizer/train/bpe_train.mojo` with a
   structure whose memory scales with the pairs that occur; state the bound; peak RSS before
   and after on a small input and at the full 50,256-rank job.
2. Expose the Mojo trainer through the tokenizer host binding; `BpeVocabularyTrainer` /
   `lm_corpus.prepare()` default to it; the Python trainer stays reference and fallback.
3. Determinism: Mojo and Python ranks files byte-identical on several small corpora (tie-heavy,
   non-ASCII / Unicode edge); full rerun reproduces ranks sha 3d547b17...; pre-change and
   post-change Mojo agree.
4. identity_break `bpe-trainer`, `bpe-vocabulary`, `tokenized-corpus` IDENTICAL before vs after
   at --repeats 2 on 9 fixtures; a sabotage build reads DIVERGENT; verification_matrix --check.
5. Wall time and peak memory for the full job. Full job on a RunPod CPU pod only.

## Predictions (registered BEFORE any run)

- P1 old vs new Mojo trainer: byte-identical ranks and tokenizer.json on every small corpus and
  at full size (ranks sha 3d547b17821cf465...). The selection code and the key encoding
  `left * V + right` are unchanged; only where a count is stored changes.
- P2 Mojo (binding) vs Python trainer: byte-identical on every small corpus, including the
  tie-heavy and Unicode-edge ones. A failure here would most likely be a PRE-TOKENIZER
  disagreement (Mojo `impl/pretokenize.mojo` vs `_tokenizer_synthetic.pretokenize`), not the
  merge loop.
- P3 peak RSS at 50,256 ranks: old ~20.2 GB (the V x V Int table, 50,256^2 x 8 B) plus the
  corpus; new well under 1 GB (estimate 100-300 MB: 20 MB of corpus, the pre-token groups as
  Int lists, and a hash table sized by the distinct pairs of one pass).
- P4 small input (vocab 512): old and new both near the process baseline (the old table is
  only 2 MB there). At vocab 8,192 the old table is 537 MB and the new one is not.
- P5 wall time: new within 0.5x-2x of old on the same box (hash probe vs dense index; the
  dense table's cache misses over 20 GB may make the old one SLOWER, not faster).
- P6 identity lanes: IDENTICAL x2 on 9/9 fixtures before (Python trainer) vs after (Mojo
  trainer through the binding); the MOJOLEARN_BPE_TRAINER_SABOTAGE arm (env, and the
  `-D` build define in the binding) reads DIVERGENT 9/9.

## Log

(empty)

## Candidates (not opened)

(none yet)
