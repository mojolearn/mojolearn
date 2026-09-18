# LANE STATUS: tokenized-corpus (lane/tokenized-corpus)

**STATE 2026-09-18 ~11:40 local: DONE and MERGED to main (18c15b346; rename alone landed
first as 5bde47f20). No pods live: ph5zsazy7u4kxn, vn4vonca6du36q, d8klbzo3aga6d6,
nqbmrf373yti99 all read HTTP 404 from the RunPod API.**

Open questions for Andrew (not acted on):
1. The built-in default trains with `BpeVocabularyTrainer`, which is the PURE PYTHON trainer; at
   50,256 ranks on 10 MB that is impractical (the Mojo trainer took 71 min for the same job).
   Candidate follow-up: expose `tokenizer/train/bpe_train.mojo` through the tokenizer binding
   (held file-byte-equal to the Python one already by check-bpe-trainer) and drop its dense
   V x V count table (20.2 GB at 50,256). Not opened.
2. Should mojolearn ever ship a trained vocabulary? It would be derived from third-party text.
   Today: none ships; ours is in R2 only.
3. Default vocabulary size 50,256 ranks (GPT-3 Small's 50,257 ids) and 10 MB training sample:
   picked to match the target shape, not measured for quality.
Other candidates (not opened): a Mojo cut path for the Llama 3 / Qwen 2 patterns; a sharded
parallel `prepare` (documents are independent); parquet (FineWeb) input to `prepare`.

Written for a session with no context. Worktree `~/mojolearn-wt/tokenized-corpus`,
branch `lane/tokenized-corpus`. Original brief (binding):
`~/mojolearn-evidence/relaunch-sep18/BRIEF_tokenized-corpus_original.txt`. Two items were
ADDED on relaunch (2026-09-18): A, rename `GPT2Tokenizer`; B, tokenization as a built-in
part of training with a flag for the user's own vocabulary. Evidence lives under
`~/mojolearn-evidence/tokenized-corpus-sep18/`, never in the repo.

## Item A: the rename, DONE

`GPT2Tokenizer` -> `BpeTokenizer` (Python class and Mojo struct), `load_gpt2_tokenizer_from`
-> `load_bpe_tokenizer_from`, the binding's `gpt2_*` entries -> `bpe_*`, `_Gpt2Handle` ->
`_BpeHandle`.

- `mojolearn.GPT2Tokenizer` and `mojolearn.tokenizer.GPT2Tokenizer` stay importable: they
  are in `__all__` and shipped in 0.8.x. The alias is the SAME class object, reached through
  a module `__getattr__` that raises a DeprecationWarning. It is declared as
  `_DEPRECATED_ALIASES = {"GPT2Tokenizer": "BpeTokenizer"}` in `tokenizer.py`, which
  `tools/verification_matrix.py` now reads as an alias (without that it counted the old name
  as a public algorithm with ZERO lanes, 19 -> 20 laneless; seen and fixed).
- The binding ABI WAS renamed. The compatibility cost I could name: a binding built before
  the rename (an older `MOJOLEARN_HOST_DIR`) exports `gpt2_*` only. `tokenizer.py` resolves
  each entry as `bpe_<name>` else `gpt2_<name>` (`_Entries`), so an old binding keeps
  loading. Checked: the pre-rename build driven by the renamed Python encodes, batches and
  decodes. The bincache key is a sha256 over the binding sources, so a renamed source never
  hits an old cached binary. The wheel ships the Python and the binding together.
- Kept the GPT-2 name only where it names a GPT-2 thing: `GPT2_PAT_STR` /
  `_GPT2_TIKTOKEN` / `_GPT2_HF` (the pre-tokenization pattern), `GPT2_ENDOFTEXT` (the
  format's special token spelling), `from_files(encoder_json, vocab_bpe)` and the
  `MOJOLEARN_GPT2_ENCODER_JSON` / `_VOCAB_BPE` test variables (the real GPT-2 file format),
  and `models/tokenizer.py`'s `"gpt2"` pattern key.
- `tokenizer/NOT_IMPLEMENTED.tsv`: first column header `their_thing` -> `capability`; rows
  unchanged.
- `tokenizer/README.md` now states the TWO CUT PATHS: the Mojo binding cuts the GPT-2 pattern
  only; `python/mojolearn/models/tokenizer.py` cuts GPT-2, Llama 3 and Qwen 2 in Python. Not
  closed in this lane.

### The rename moved no bit (identity_break, M4, one core, nice 19, `--repeats 2`)

Pre-rename binding built from main's source into `rename/host_before/`, post-rename into
`rename/host_after/`, sabotage (`-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1`) into
`rename/host_after_sabotage/`. JSONs and `--diff` outputs in
`~/mojolearn-evidence/tokenized-corpus-sep18/rename/`.

| arm vs before | tokenizer train | tokenizer infer | tokenizer batch | bpe-trainer train |
|---|---|---|---|---|
| after (renamed) | IDENTICAL x2, 9/9 fixtures (`base` 08b10bbc6f4b565b) | IDENTICAL 9/9 (`base` 1939b74da0333639) | IDENTICAL 9/9 (`base` 6e235476cc31241f) | IDENTICAL 9/9 (`base` 6ed8b49585df3d85) |
| host sabotage build | DIVERGENT 9/9 (`base` -> 74458a3ecb971805: ids, decoded, roundtrip differ; n_vocab agrees) | DIVERGENT 9/9 | DIVERGENT 9/9 | IDENTICAL (not reached, correct) |
| `MOJOLEARN_BPE_TRAINER_SABOTAGE=1` | IDENTICAL (not reached, correct) | IDENTICAL | IDENTICAL | DIVERGENT 9/9 (`base` -> f5172d25e6499662, all five parts) |

Also green after the rename: `pixi run check-tokenizer` (22/22 ids and round trips),
`pixi run check-bpe-trainer`, `test_tokenizer_surface.py` (22 of 23, GPT-2 files test
skipped: no GPT-2 files present), `test_tokenizer_manifest.py` (with a new alias test),
`test_host_surface.py` + `test_models_loader.py` (pytest, 215 passed / 4 skipped), and
`tools/verification_matrix.py --check`. The sabotage build fails the surface test 11 of 23.

## Item B and the original brief: DONE

- THE ENTRY POINT. There was none. `LanguageModelTrainer.train_step(ids)` takes one
  materialized batch and fetches nothing; every corpus-reading loop was a probe in `tools/`
  (`lm_step_memory_probe.py`, `lm_recycle_probe.py`, `lm_shards_probe.py`,
  `lm_shakedown_resume.py`), all over `CorpusBatches` (raw bytes). Added:
  `python/mojolearn/lm_corpus.py` (public, `mojolearn.lm_corpus`: `prepare`,
  `TokenizedCorpus`, `TokenBatches`, `require_vocabulary`, `tokenizer_for`) and
  `tools/lm_train.py` (corpus -> prepare -> trainer -> steps; default trains our
  vocabulary, `--vocab` takes the user's, `--bytes` uses `CorpusBatches` unchanged).
- `LanguageModelTrainer` now refuses a `data_schedule` whose `vocabulary.n_vocab` is not its
  `vocab_size` (the one checked schedule field) and exposes `.data_schedule`.
- `BpeTokenizer.identity` / `TrainedBpeVocabulary.identity`: sha256 of the CANONICAL rank
  file + n_vocab; the same table from a rank file, encoder.json+vocab.bpe or a trained object
  has one identity.
- Smoke-tested locally (200 KB corpus, 512 ranks): prepare 3.8 s, rerun reuses the cache,
  the user-vocabulary arm lands on the same id sha, a synthetic tokenizer is refused by name.
  Evidence `~/mojolearn-evidence/tokenized-corpus-sep18/smoke/`.
- LANES, RUN AND RECORDED (`bench/results/identity_break/2026-09-18_tokenized-corpus/`, M4,
  one core, `--repeats 2`, 9 fixtures, commit 081e14fa9): `bpe-vocabulary`
  (TrainedBpeVocabulary: write_ranks / write_tokenizer_json bytes, render equality, identity,
  tokenizer() and written-file round trip) and `tokenized-corpus` (lm_corpus.prepare, cache
  reuse, user-vocabulary arm, TokenBatches steps, data_schedule, require_vocabulary refusal).
  clean vs replay IDENTICAL=18; clean vs `MOJOLEARN_BPE_TRAINER_SABOTAGE=1` DIVERGENT=18; clean
  vs tokenizer host sabotage build DIVERGENT=18. `base` clean hashes: bpe-vocabulary
  875b2b4bc2a4c302, tokenized-corpus 2580bb7a34e5ae01. Every flag part read [1,...,1]
  (printed directly, not inferred from a hash).
- `tools/verification_matrix.py`: `tokenizer.TrainedBpeVocabulary` removed from
  NOT_ALGORITHMS (68a9241ac had filed it as a result container); it now reads 1 lane,
  seen(build). `--write` then `--check` green (230 lanes, 236 public API entries).
  `lm_corpus.*` each read 1 lane, seen(build). No GPU column (host-only, like bpe-trainer).

## The built-in design (Item B), as built

DEFAULT = TOKENIZED. `lm_corpus.prepare(corpus)` with no `vocab` trains our vocabulary
(`BpeVocabularyTrainer`, 50,256 ranks by default, on the first 10 MB of the corpus's TRAIN
range only, never validation/test), tokenizes the whole corpus once, caches. `vocab=` (rank
file, `(encoder.json, vocab.bpe)`, `BpeTokenizer`, `TrainedBpeVocabulary`) uses the user's
token map and trains nothing. Byte mode: `tools/lm_train.py --bytes` -> `CorpusBatches`,
whose file (`tools/lm_step_memory_probe.py`) this lane does not touch at all.

WHY BYTES STAY THE DEFAULT IN THE PROBES. `lm_step_memory_probe.py`, `lm_recycle_probe.py`,
`lm_shards_probe.py`, `lm_shakedown_resume.py` keep `--corpus` = bytes. They are timing and
resume probes whose recorded numbers are byte runs; switching their default would make every
future number incomparable with the recorded ones and would make each pod leg train a
vocabulary first. The tokenized default lives in the new entry point instead.

CACHE AND MANIFEST SCHEMA (`python/mojolearn/lm_corpus.py` docstring is authoritative):

    <cache>/vocab/<corpus sha16>-v<vocab_size>-f<min_frequency>-s<sample bytes>/{ranks.tsv, vocabulary.json}
    <cache>/vocab/user-<vocabulary sha16>/{ranks.tsv, vocabulary.json}
    <cache>/tokens/<corpus sha16>-<vocabulary sha16>-d<document bytes>/{tokens.i32, manifest.json}

- `vocabulary.json`: schema `mojolearn.bpe-vocabulary.v1`, `identity` {schema, sha256 (of the
  canonical `rank<TAB>lowercase-hex` text), n_ranks, n_vocab, endoftext_id}, `recipe`
  (trainer, format, tie_break, vocab_size, min_frequency, corpus_sha256, sample, counts,
  train_seconds) or {source} for a user map.
- `manifest.json`: schema `mojolearn.byte-lm.tokens.v1` (sibling of
  `mojolearn.byte-lm.corpus.v1`): source {path, sha256, bytes, schema, manifest_sha256,
  source_url}, vocabulary (identity), encoder, endoftext rule, document_rule, document_bytes,
  n_documents, dtype int32 little-endian, sha256 + bytes + tokens of `tokens.i32`,
  bytes_per_token, max_id, ids_above_255, train/validation/test ranges IN TOKENS (exact:
  documents never span a range boundary), schedule `train-range-modulo.v1`, timings.
- Reuse: corpus sha, vocabulary sha and document_bytes pick the directory; the id array is
  re-hashed against the manifest on every reuse and a mismatch is REFUSED by name, never
  silently rebuilt. Entries are built in a temp sibling and renamed into place.
- The model carries the vocabulary: `TokenBatches.data_schedule()` has
  `vocabulary: {schema, sha256, n_vocab}`; the trainer refuses n_vocab != vocab_size and keeps
  the schedule in every checkpoint; `lm_corpus.tokenizer_for(model, vocab)` /
  `require_vocabulary` refuse a tokenizer of another table by sha256.
- `TokenBatches` fixes a defect `CorpusBatches` keeps for bit-compatibility: its modulus is
  the TRAIN range in tokens, so a long run reads no validation or test id.

NO VOCABULARY SHIPS. Trained tables live in the user's cache (and ours in
~/mojolearn-evidence / R2). Question for Andrew, not acted on: whether mojolearn should ever
ship a built-in trained vocabulary (it would be derived from third-party text).

## The vocabulary job (FINISHED 10:38, outcome below)

pid 95203, `train_main`, started 09:26 local 2026-09-18, one core, nice 19: 50,256 ranks
(n_vocab 50,257 with `<|endoftext|>`) on the first 10 MB of enwik8 + the first 10 MB of
pile_github. Command: `~/mojolearn-evidence/tokenized-corpus-sep18/vocab/user_cmd.sh`, log
`train.log` there, output prefix `mojolearn-bpe-50257-v1` in that directory. Its binary is in
a dead session's /private/tmp scratchpad; if the process dies, rebuild `train_main` into
`~/mojolearn-evidence` and rerun.

FINDING (2026-09-18 ~09:50): `tokenizer/train/bpe_train.mojo` allocates a DENSE
`vocab_size x vocab_size` Int table (`counts`, line ~253): at 50,256 that is 20.2 GB, on a
16 GB Mac. `top` showed train_main at 19 GB, 18 GB of it compressed (zero pages compress),
swap 443 MB used, system memory free 58%. Not killed (owed). If it dies or thrashes, rerun on a
RunPod CPU pod with >= 32 GB RAM. The dense table is a follow-up CANDIDATE (a hash map of
touched keys is the same output by construction), not done in this lane.

FINDING: `BpeVocabularyTrainer.train` is the PURE PYTHON trainer (`_bpe_trainer.py`), which
recounts every pair of every pre-token group on every merge. 256 merges on 100 KB took about
3.8 s including tokenizing. A 50,256-rank vocabulary on 10 MB through the built-in default is
not practical in Python; measured rate and the question for Andrew go in the report.

## GPU leg (launched 2026-09-18 ~10:05 local)

`tools/lm_vocab_witness_body.sh` on RunPod NVIDIA through `tools/gemm_remote_leg.sh`
(`--payload gemm --rent --allow-concurrent`, R2 staging strict, enwik8 + pile_github). It
builds, trains an 8,192-rank vocabulary on the pod with train_main (the 50,256 one is the
local owed job), tokenizes all of enwik8 once (throughput), runs the gradient witness (tokens
vs bytes, same shape/seed/vocab_size) and the byte-path-vs-main comparison
(`tools/lm_byte_path_main_copy.py` rebuilds main's `_byte_lm_impl.py` byte for byte, sha
d6948abc...; a seed-2 arm must differ). Output:
`~/mojolearn-evidence/tokenized-corpus-sep18/pod/<stamp>/`. The pod is reaped by the leg;
verify 404 after.

GPU leg 1 (2026-09-18_135642, RTX 4090 pod ph5zsazy7u4kxn, 96-vCPU x86, 404-verified): builds
of base (85 s) and byte_lm (206 s) OK; the tokenizer host refused `MOJOLEARN_TARGET_COLUMN=nvidia`
(it is a CPU-column binding) and train_main then lacked the generated Unicode table, so no
vocabulary and nothing after ran (extra_exit=4). Fixed in the body (CPU column for the tokenizer
build, explicit gen_unicode_table.sh). Relaunched.

VOCAB JOB OUTCOME (pid 95203, finished 10:38 local): `real 4274.27 s` (71.2 min), user 4083 s,
M4 one core nice 19. 50,256 tokens (50,000 merges) from 220,165 pre-token groups, 47,825 ties
broken, despite the 20 GB dense count table (it compressed). Files in
`~/mojolearn-evidence/tokenized-corpus-sep18/vocab/`:
`mojolearn-bpe-50257-v1.ranks.tsv` sha256 3d547b17821cf46502f275a441dd6ded9682a4ddcacde993a1ff836f39c4122d
(967,306 bytes, 50,256 lines), `.tokenizer.json` sha256
7ae8b893219619cf73e64b262367cc6e6d2f9f1aa9c1a32b8fb7e3cb84d972e9. Not in the repo (derived from
third-party text).

GPU leg 2 (2026-09-18_140910, pod vn4vonca6du36q): the box went network-unreachable a few
minutes into the payload ("neutral hosts DO answer ... the BOX is the silent end"); nothing came
home. Infra failure, not a result. Lease self-kill at 60 min; 404 to be verified.

## THROUGHPUT (measured, M4 one core nice 19, 2026-09-18 11:00)

`tools/lm_train.py --corpus training/corpus/enwik8/input.txt --vocab mojolearn-bpe-50257-v1.ranks.tsv
--prepare-only` (`encode_batch`, 16 documents of <= 1 MiB per call, 96 documents):
100,000,000 bytes -> 26,834,102 ids (3.727 bytes/id) in 22.93 s of encode, 24.05 s tokenize,
25.9 s end to end: 4.36 MB/s = 1.170 M ids/s on ONE core. 25B tokens / 1.170 M/s = 21,370 s =
5.9 core-hours (~93 GB of text at this ratio); documents are encoded alone, so it splits across
cores with no change in ids (96 vCPUs: ~4 minutes). NOT slow enough to matter; not optimized.
Cache reuse: 1.3 s (re-hash of the 107 MB id array). A second tokenization into a fresh cache gave
the same sha256 2ad0c690bc4438d5ddab5cb481c4b247ba61b7dd3d841a3d7c0a7d0f94b2c253.
ids above 255: 21,609,992 of 26,834,102 (80.5%); max id 50,253. Token ranges: train
[0, 24,141,115), validation [24,141,115, 25,479,817), test [25,479,817, 26,834,102).

R2 (pinned in bench/results/dataset_store/manifest.tsv, pulled back and re-hashed OK):
  vocab/mojolearn-bpe-50257-v1/ranks.tsv        967,306  3d547b17...4122d
  vocab/mojolearn-bpe-50257-v1/tokenizer.json 1,977,368  7ae8b893...972e9
  corpus/enwik8/tokens/mojolearn-bpe-50257-v1/tokens.i32   107,336,408  2ad0c690...2c253
  corpus/enwik8/tokens/mojolearn-bpe-50257-v1/manifest.json      1,773  51898a61...74d9a

Trainer cost for comparison: the Mojo trainer took 71.2 min for 50,000 merges on 20 MB; the
Python `BpeVocabularyTrainer` (what `prepare()` with no `vocab` calls) recounts every pair per
merge in pure Python and is far slower (not measured at this size). The built-in default is
correct but not practical at 50,256 ranks until the Mojo trainer is reachable from Python
(candidate follow-up, not opened).

GPU leg 3 (2026-09-18_150327, pod d8klbzo3aga6d6, 404-verified): the vocabulary staged from R2
(sha verified on the box), but the tokenizer host build refused again, now on
MOJOLEARN_GPU_ARCHS ("a CPU build takes no MOJOLEARN_GPU_ARCHS"), so prepare found no binding
(extra_exit=5). Fixed with `env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu`, and this
time the exact command was run locally under the pod's env first and built. All three pods so far
(ph5zsazy7u4kxn, vn4vonca6du36q, d8klbzo3aga6d6) read HTTP 404 from the API.

## GPU leg 4: DONE (2026-09-18_151432, RTX 4090 pod nqbmrf373yti99, 404-verified)

Record: `bench/results/lm_vocab_witness/2026-09-18_rtx4090/` (README has the table). With the
50,257-id vocabulary from R2, same shape/seed/initial weights:
- tokens arm: embedding rows >= 256 nonzero 57 (step 0) and 62 (step 1), exactly the distinct
  ids >= 256 in each batch; max |g| 2.8e-3 / 6.8e-3.
- bytes arm (CorpusBatches): embedding rows >= 256 nonzero 0 and 0, max |g| exactly 0.0.
- lm_head: all 50,001 rows >= 256 nonzero in BOTH arms (softmax), 5e-7 under bytes vs 3e-3 under
  tokens. The brief's "99.5% of those rows would finish training at init" holds for the
  EMBEDDING only; the unembedding rows do train under bytes (pushed down, never up).
- byte path vs main: EQUAL sha256 for loss, gradients, parameters, m, v, flags at both steps
  (branch python vs main's exact `_byte_lm_impl.py`); seed-2 arm DIFFERS (the check can fail).
- enwik8 tokenized on the x86 box: SAME sha256 2ad0c690... as the M4; 2.35 M ids/s one core
  (EPYC 7B13), so 25B tokens = 2.95 core-hours there (5.9 on the M4).
