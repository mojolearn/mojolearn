# LANE STATUS: bpe-builder-native (lane/bpe-builder-native)

**STATE 2026-09-18: WIP, steps 1-2 done locally (sparse pair counts; binding door, Mojo == Python on ten corpora); identity lanes DONE (IDENTICAL before vs after, both sabotage arms DIVERGENT); full-size pod run owed.**

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

### Step 2: the trainer through the tokenizer host binding

`bindings/_mojolearn_tokenizer_host.mojo`: `bpe_train(text_addr, offsets_addr, [n_docs, n_bytes,
vocab_size, min_frequency, break_ties_high])` -> `_BpeTrainedHandle`; `bpe_trained_sizes`,
`bpe_trained_copy`. `tokenizer_host_sabotage()` also reads True for
-D MOJOLEARN_BPE_TRAINER_SABOTAGE=1. `python/mojolearn/tokenizer.py`:
`BpeVocabularyTrainer(..., backend="auto" | "mojo" | "python")`; auto = Mojo when the binding
exports `bpe_train`, else the Python reference; `stats["backend"]` records which ran;
`MOJOLEARN_BPE_TRAINER_SABOTAGE=1` reaches both backends. `lm_corpus` records the backend in
the vocabulary recipe. `host_surface.py`: the three exports, `tokenizer/train/bpe_train.mojo`
added to the family's host modules.

Through the PUBLIC door (`small/door_compare.py`, binding `host/after`), each corpus trained by
backend "mojo" and "python", written files compared with each other and with `py_ref.py`'s:
all ten SAME (`door_compare_clean.txt`), including `invalid_utf8` (ranks 629a768ceeb357e1),
which the CLI cannot read. Negative controls, both SEEN TO FAIL on all ten:
`MOJOLEARN_BPE_TRAINER_SABOTAGE=1` (both backends move together, and each reproduces the
CLI sabotage build's sha, e.g. synthetic 648ee0ad88acf262; `door_compare_envsabotage.txt`);
the binding built with -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1 (`host/after_trainer_sabotage`):
refused by `load_host_module` without MOJOLEARN_HOST_ALLOW_SABOTAGE=1, and with it the Mojo
backend DIFFERS from Python on every corpus (`door_compare_buildsabotage.txt`).

Also green: `pixi run check-bpe-trainer` PASS (284 tie-broken selections);
`check-bpe-trainer-sabotage` SABOTAGE SEEN TO FAIL (6 failures); `test_tokenizer_manifest`
(9 checks), `test_tokenizer_surface` (22 of 23, GPT-2 files skip), pytest
`test_host_surface.py` + `test_models_loader.py` 206 passed / 4 skipped;
`tools/verification_matrix.py --check` OK (236 lanes, 237 public API entries).

### Step 4: identity lanes (bench/results/identity_break/2026-09-18_bpe-builder-native/)

M4, one core, nice 19, `--repeats 2`, nine fixtures, lanes tokenizer, bpe-trainer,
bpe-vocabulary, tokenized-corpus. Before = main 0e4715f7c (`git archive` into
`~/mojolearn-evidence/bpe-builder-native-sep18/before_src`, its binding built from that
source); after = b77a26e93 with its binding (`host/after_b77a26e93`).

- before vs after: `summary: IDENTICAL=36`. base rows printed from `diff.after.txt`:
  `bpe-trainer/base IDENTICAL x2 6ed8b49585df3d85 | 6ed8b49585df3d85`,
  `bpe-vocabulary/base IDENTICAL x2 875b2b4bc2a4c302`, `tokenized-corpus/base IDENTICAL x2
  2580bb7a34e5ae01`, `tokenizer/base IDENTICAL x2 08b10bbc6f4b565b`.
- before vs binding built with -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1: `DIVERGENT=27,
  IDENTICAL=9`; the 27 are every bpe-trainer / bpe-vocabulary / tokenized-corpus cell, e.g.
  `bpe-trainer/base DIVERGENT parts differ: ranks,tokenizer_json,n_tokens,n_merges,n_ties_broken
  | f5172d25e6499662`; the 9 IDENTICAL are the tokenizer lane (not reached, correct). Since
  the Python code is the same in the after and sabotage arms, this is what shows the after arm
  ran the MOJO trainer.
- before vs env MOJOLEARN_BPE_TRAINER_SABOTAGE=1 (clean binding): `DIVERGENT=27, IDENTICAL=9`,
  same pattern.
- `tools/verification_matrix.py --check` read STALE after the record landed (d8be5d5f0 said OK;
  that line was written before the check's output was read -- the `&&` chain ran on `tail`'s exit
  code). `--write` moved bpe-vocabulary's and tokenized-corpus's sabotage evidence to this
  record's `m4.trainer.sabotage-build.json`; `--check` then OK (236 lanes, 237 entries).

### Step 3b: a standing gate for backend agreement

`test_tokenizer_surface.py::test_trainer_mojo_backend_writes_the_python_reference_bytes`: the
three reference fixtures plus an invalid-UTF-8 / all-256-bytes / empty-document corpus, Mojo
backend vs Python backend, ranks + tokenizer.json + merges + ties equal, ties reached, and
`auto` picks mojo. Clean binding: GREEN 23 of 24 (GPT-2 files skip). Binding built with
-D MOJOLEARN_BPE_TRAINER_SABOTAGE=1: `FAIL test_trainer_mojo_backend_writes_the_python_reference_bytes:
AssertionError: ranks differ (vocab_size 512)`, RED 1 of 24.

### Step 5: the full-size job (pod RENTED 2026-09-18 ~15:50 local, running)

`tools/runpod_cpu_leg.sh` gained `--stage 'KEYS'` (tools/stage_from_r2.sh, strict; the CPU
runner had no way to stage a corpus). Body `~/mojolearn-evidence/bpe-builder-native-sep18/pod/
body.sh` (template beside it; main's bpe_train.mojo embedded as base64, sha fd4751832a5f...):
heads of the staged corpora checked against the M4 recipe (enwik8 5985c81c..., pile_github
0bee7f53...) before anything runs; A new CLI alone; B `BpeVocabularyTrainer(backend="mojo")`
alone; C old CLI bounded at 6,000 s (VmHWM sampled every 15 s so the peak survives a bound)
beside `lm_corpus.prepare()` on all of enwik8 (the built-in default). Pod: 8 vCPU, memory
flavors (>= 32 GB), lease 175 min. PREDICTIONS for it: A and B ranks sha 3d547b17...; A peak
< 1 GB; old peak ~20.2 GB (VmHWM ~ 19.7 GiB); A wall well under the M4's 4,274 s.

LEG 1 (pod c2fj95nckg26vq, 8 vCPU cpu memory flavor, 64 GB cgroup limit, AMD EPYC 7713P,
$0.44/hr; R2 staged 2 keys in 13 s, strict; heads matched the M4 recipe):
- A (new CLI) and C's old CLI DID NOT RUN: the body called bare `mojo build`, which finds no
  `std` outside `pixi run` (build_new.log / build_old.log). Fixed in body2 (`pixi run
  --manifest-path ... mojo build`, rehearsed on the live pod: built).
- **B, the PUBLIC DOOR `BpeVocabularyTrainer(vocab_size=50256, min_frequency=2,
  backend="mojo")` on the two 10 MB heads: ranks sha256
  3d547b17821cf46502f275a441dd6ded9682a4ddcacde993a1ff836f39c4122d and tokenizer.json
  7ae8b893219619cf73e64b262367cc6e6d2f9f1aa9c1a32b8fb7e3cb84d972e9 -- BOTH EQUAL to the
  M4 dense-table run's files.** 50,256 tokens, 50,000 merges, 47,825 ties, 220,165 groups
  (all equal to the M4 log). Wall 2,532.9 s, one core, EPYC 7713P. Peak RSS of the WHOLE
  Python process (interpreter + numpy + both documents + trainer) 156,880 KiB = 153 MiB
  (ru_maxrss; VmHWM sampler agrees). The dense-table trainer allocated 20.2 GB on the same job.
- C's `lm_corpus.prepare()` on all of enwik8 is running on leg 1.
Legs B (new CLI alone) and C (old CLI alone, bounded 9,600 s) go to two more pods so the old
arm's 20 GB and the new arm's timing do not share a box.

## Candidates (not opened)

(none yet)
