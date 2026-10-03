# Lane n-pyneural (neural family)

Read `docs/plans/cpu-gpu-cleanup/COMMON_BRIEF.md` and `docs/plans/HOST_ROUTE_REMOVAL.md` first, then `CLAUDE.md`.

Worklist: `docs/plans/cpu-gpu-cleanup/rows/n-pyneural.tsv` (38 rows). Columns: rule, class, owner, state, path, occ, text.
`text` is the offending line; `occ` is its occurrence index among identical lines in that file.

Files you own (prefixes): `python/mojolearn/neural_inference.py`, `python/mojolearn/lm_corpus.py`, `python/mojolearn/language_model.py`, `python/mojolearn/_tokenizer_synthetic.py`, `python/mojolearn/_causal_lm_fixtures.py`, `python/mojolearn/_bpe_trainer.py`, `x_cnn/`, `tokenizer/`, `bindings/_mojolearn_x_cnn.mojo`, `python/mojolearn/_expansion_cnn.py`, `python/mojolearn/_byte_lm_impl.py`, `python/mojolearn/lowbit.py`, `python/mojolearn/tokenizer.py`, `python/mojolearn/models`, `python/mojolearn/_cnn`, `python/mojolearn/byte_lm`, `bindings/_mojolearn_tokenizer`, `bindings/_mojolearn_byte_lm`

For every row: make the GPU path do that work on the device, with a fixed fold order and no serial chain, and delete the host route.
Or, where the row is CPU-only-install code that a GPU install imports by mistake, move the import behind the CPU-only path.
Clear as many rows as you can, working from the biggest problems down.
