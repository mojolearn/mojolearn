# Lane c-metrics-prep (classical family)

Read `docs/plans/cpu-gpu-cleanup/COMMON_BRIEF.md` and `docs/plans/HOST_ROUTE_REMOVAL.md` first, then `CLAUDE.md`.

Worklist: `docs/plans/cpu-gpu-cleanup/rows/c-metrics-prep.tsv` (59 rows). Columns: rule, class, owner, state, path, occ, text.
`text` is the offending line; `occ` is its occurrence index among identical lines in that file.

Files you own (prefixes): `bindings/_mojolearn_preprocessing.mojo`, `bindings/_mojolearn_metrics.mojo`, `python/mojolearn/parallel_preprocessing.py`, `x_metrics/`, `metrics/`, `x_prep/`, `preprocessing/`, `resample/`, `naive_bayes/`, `kde/`, `bindings/_mojolearn_x_metrics.mojo`, `bindings/_mojolearn_x_prep.mojo`, `bindings/_mojolearn_resample.mojo`, `python/mojolearn/preprocessing.py`, `python/mojolearn/_expansion_metrics.py`, `python/mojolearn/_expansion_prep.py`, `python/mojolearn/_metrics_impl.py`, `python/mojolearn/model_selection.py`

For every row: make the GPU path do that work on the device, with a fixed fold order and no serial chain, and delete the host route.
Or, where the row is CPU-only-install code that a GPU install imports by mistake, move the import behind the CPU-only path.
Clear as many rows as you can, working from the biggest problems down.
