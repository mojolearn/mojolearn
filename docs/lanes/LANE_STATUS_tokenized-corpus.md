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

## Item B and the original brief: IN PROGRESS

See the sections below as they land.

## The vocabulary job (OWED, do not kill)

pid 95203, `train_main`, started 09:26 local 2026-09-18, one core, nice 19: 50,256 ranks
(n_vocab 50,257 with `<|endoftext|>`) on the first 10 MB of enwik8 + the first 10 MB of
pile_github. Command: `~/mojolearn-evidence/tokenized-corpus-sep18/vocab/user_cmd.sh`, log
`train.log` there, output prefix `mojolearn-bpe-50257-v1` in that directory. Its binary is in
a dead session's /private/tmp scratchpad; if the process dies, rebuild `train_main` into
`~/mojolearn-evidence` and rerun.
