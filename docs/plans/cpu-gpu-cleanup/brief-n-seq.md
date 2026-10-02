# Lane n-seq (neural family)

Read `docs/plans/cpu-gpu-cleanup/COMMON_BRIEF.md` and `docs/plans/HOST_ROUTE_REMOVAL.md` first, then `CLAUDE.md`.

Worklist: `docs/plans/cpu-gpu-cleanup/rows/n-seq.tsv` (32 rows). Columns: rule, class, owner, state, path, occ, text.
`text` is the offending line; `occ` is its occurrence index among identical lines in that file.

Files you own (prefixes): `python/mojolearn/_x_sequence_`, `arima/`, `sequence/`, `holtwinters/`, `tsa/`, `bindings/holtwinters_host_predict.mojo`, `bindings/arima_exog_layout.mojo`, `bindings/_mojolearn_arima`, `bindings/_mojolearn_sequence`, `bindings/_mojolearn_holtwinters`, `bindings/_mojolearn_tsa`

For every row: make the GPU path do that work on the device, with a fixed fold order and no serial chain, and delete the host route.
Or, where the row is CPU-only-install code that a GPU install imports by mistake, move the import behind the CPU-only path.
Clear as many rows as you can, working from the biggest problems down.
