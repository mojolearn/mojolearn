# LANE STATUS: bpe-builder-native (lane/bpe-builder-native)

**STATE 2026-09-18: WIP, step 1 done locally (sparse pair counts, small + medium proofs); binding next; full-size pod run owed.**

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

### Step 1: the dense V x V table replaced (commit after f680a4203)

`tokenizer/train/bpe_train.mojo`: `PairCounts`, an open-addressing table keyed by the same
integer `left * V + right`, cleared by walking only occupied slots. Two other changes, both the
same output by construction: the per-merge rewrite of each group is IN PLACE (write index never
passes read index) instead of a fresh List per group per merge; `train_bpe` takes a runtime
`break_ties_high` (the sabotage define's runtime spelling, for the binding's env arm).

MEMORY BOUND (in the struct's docstring): with N0 = adjacent positions in the initial DISTINCT
pre-token groups (<= unique pre-token bytes), every pass has P <= N0 distinct pairs; the table
is a power of two <= 4 * max(P, 256) slots x 2 Ints plus `touched`: peak <= 72 * N0 bytes,
independent of V. Old: 8 * V^2 bytes (20.2 GB at V = 50,256) whatever the corpus.

Old binary = main's source built before the edit (`bin/train_main_old`), new =
`bin/train_main_new`, sabotage = new with -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1. M4, one core,
nice 19, `/usr/bin/time -l`. Evidence `~/mojolearn-evidence/bpe-builder-native-sep18/small/`
(`make_corpora.py`, `py_ref.py`, `run_mojo.sh`, `compare_*.txt`).

| corpus (docs, bytes, V, minf) | Python ranks sha16 | old Mojo | new Mojo | sabotage | old RSS | new RSS |
|---|---|---|---|---|---|---|
| synthetic (1, 3,306, 512, 2) | 8ad2aabaebe4f7f8 | same | same | 648ee0ad88acf262 | 16.9 MB | 12.9 MB |
| ties (1, 216, 300, 2) | e2322ff43bdfb998 | same | same | 41ac40b650c553d7 | 14.3 MB | 12.7 MB |
| synthetic_small (1, 1,176, 320, 3) | 35ab5fc278e7e63a | same | same | 13d146bc3070e1db | 14.5 MB | 12.8 MB |
| ties_dense, 132 equal-count 2-letter words (1, 396, 600, 1) | 4aaa98235f7acc2e | same | same | eaa00876e625fad4 | 19.7 MB | 12.8 MB |
| runs_overlap, `aaaa`/`abab` runs (3, 235, 300, 1) | 00a662c31504614b | same | same | 5d1ea5b82d92c72e | 14.3 MB | 12.7 MB |
| unicode_edge: combining marks, ZWJ emoji, flags, CJK, RTL, Devanagari, NBSP/ideographic/em space, CRLF, NUL, DEL, U+2028/9, U+0085, VT/FF, BOM, astral (2, 4,524, 700, 2) | 33e9740a75d3b695 | same | same | df2a8327881f3dd7 | 20.8 MB | 12.9 MB |
| many_docs (40, 6,337, 900, 2) | a0a58af115a829b6 | same | same | 86680f263413f989 | 27.7 MB | 13.1 MB |
| min_freq_high (1, 4,656, 2000, 7) | 41820a50dad30ddb | same | same | afdf6e970d30fbdf | 78.3 MB | 12.9 MB |
| enwik8 first 200 KB (1, 200,000, 1500, 2) | 76b249d1cab1d8eb | same | same | c47b8339bda177c7 | 63.1 MB, 4.20 s | 14.7 MB, 0.85 s |
| enwik8 first 1 MB, V 8192 (no Python arm) | - | 95901de8634441ab | same | - | **586.1 MB, 82.53 s** | **21.2 MB, 14.28 s** |

tokenizer.json agrees three ways on every row too (`compare_trainmain.txt`). Counters (tokens,
merges, groups, ties) equal Python's on every row. `invalid_utf8` (lone continuation bytes,
surrogate encodings, 0xFF, all 256 bytes): train_main REFUSES it old and new alike (rc 1) --
the CLI reads each file as a Mojo String; it goes through the binding in step 2 instead.
Prediction P1 held (small); P4 held; P5 was too pessimistic: the new one is 5.0x / 5.8x FASTER
(the dense table's misses and the per-group allocations were the cost).

## Candidates (not opened)

(none yet)
