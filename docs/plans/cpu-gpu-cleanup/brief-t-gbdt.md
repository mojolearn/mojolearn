# Lane t-gbdt (trees family)

Read `docs/plans/cpu-gpu-cleanup/COMMON_BRIEF.md` and `docs/plans/HOST_ROUTE_REMOVAL.md` first, then `CLAUDE.md`.

Worklist: `docs/plans/cpu-gpu-cleanup/rows/t-gbdt.tsv` (46 rows). Columns: rule, class, owner, state, path, occ, text.
`text` is the offending line; `occ` is its occurrence index among identical lines in that file.

Files you own (prefixes): `gbdt/`, `xtrees/`, `bindings/_mojolearn_trees.mojo`, `bindings/_mojolearn_estimators.mojo`, `python/mojolearn/_expansion_trees.py`, `python/mojolearn/_gbdt_adapters.py`, `bindings/_mojolearn_gbdt`, `bindings/_mojolearn_xtrees`

For every row: make the GPU path do that work on the device, with a fixed fold order and no serial chain, and delete the host route.
Or, where the row is CPU-only-install code that a GPU install imports by mistake, move the import behind the CPU-only path.
Clear as many rows as you can, working from the biggest problems down.
