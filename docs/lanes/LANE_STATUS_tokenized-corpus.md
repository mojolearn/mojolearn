# LANE STATUS: tokenized-corpus (lane/tokenized-corpus)

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

## Item B and the original brief: IN PROGRESS (WIP, pushed)

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

## The vocabulary job (OWED, do not kill)

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
