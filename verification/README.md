# verification/: the one home of verification data

Everything a verification reads or records lives here, except the two homes below that stay where they are
for a reason. `bench/results/` holds boards and measurements only.

| What | Where | Who reads or writes it |
|---|---|---|
| What `python -m mojolearn verify` ships | `python/mojolearn/verify_reference/` (unchanged) | the wheel packers copy it; verify opens it from the installed package |
| The recorded identity columns (the verify table's inputs, `INDEX.tsv` beside them) | `verification/records/identity/` | read: `_verify_reference.repository_record_paths` (`verify --all --emit-reference`, maintainer path), `tools/identity_columns.py`, `verification_matrix.py`, `part_set_census.py`, the release identity tools; written: `tools/admit_identity_columns.sh`, `tools/retain_run_summary.py`, `tools/gap_column_leg.sh`; the wheel packers copy `host_surface.TRAINING_GPU_COLUMNS` and `GBDT_CTR_MODELS_DIR` from here |
| Recorded classical host columns and saved models | `verification/records/classical_host/` | `tools/classical_host_gate.py` (`host_surface.CLASSICAL_GPU_COLUMNS`), `cpu-identity-gate.yml`, `test_host_surface.py`; two files are rows of `verification/records/CELLS_INDEX.tsv` |
| Recorded forest / GBDT host fixtures | `verification/records/forest_host/` | `tools/forest_host_gate.py`, `forest-host-gate.yml`, `cpu-identity-gate.yml` (`host_surface.FOREST_RECORDED_ROOT`) |
| Column-shaped records outside the identity columns (CELLS_INDEX rows and their run directories) | `verification/records/cells/{verification_faults,small_training,verify_external}/` | `tools/verification_matrix.py`, `tools/part_set_census.py` (through `verification/records/CELLS_INDEX.tsv`), `test_verify_small.py` |
| The dated release records up to 0.8.65 that documents still cite | `verification/records/release_verification/` | `tools/release_evidence_set.py provenance_history` (MOVED_DIR; older ones through git at LEGACY_DIR) |
| Native test fixtures | `checks/native_tests/fixtures/` (stays) | the native tests beside them own them |
| The three Mamba reference cases a live reader opens, with their generator | `checks/native_tests/fixtures/mamba/` (`base_b2_l4_d8`, `mamba2/m2_base_b2_l4_d32`, `mamba3/m3_base_b2_l4_d32`, `gen_corpus.py`, README, the two family manifests) | the Mamba Mojo checks' byte gates, `tools/mamba_corpus_check.py`, the installed-surface release gates, the Mamba package tests; every other case regenerates with `python checks/native_tests/fixtures/mamba/gen_corpus.py --family <mamba1/mamba2/mamba3> --out <dir>` |

Rules: a record is never edited or moved by hand except with `git mv` of a whole group, together with every reader,
in one commit; the indexes move with their data and `python3 tools/record_index.py check` stays `listed_ok`, 0
unlisted. A new `*.log` is not added here (`.gitignore`, `tools/hooks/check_tree_additions.sh`); a tracked log moved
in with its record stays.

## Where it was (records and older docs cite the old paths)

A path that is part of a record's content (a JSON that stores its own path, a table citation, a manifest row) is data
and keeps the old spelling. Read it through this map:

| Old path | New path | Moved |
|---|---|---|
| `bench/results/identity_break/` | `verification/records/identity/` (the shipped table cites records by `dir/file` under this root, so it is unchanged) | 2026-11-25, lane verify-home |
| `mamba/corpus/` | `checks/native_tests/fixtures/mamba/` (three of its 46 cases; the rest deleted, in git history and regenerable) | 2026-11-25, lane verify-home |
| `bench/results/forest_host/` | `verification/records/forest_host/` | 2026-11-25, lane verify-home |
| `bench/results/classical_host/` | `verification/records/classical_host/` | 2026-11-25, lane verify-home |
| `bench/results/verification_faults/` | `verification/records/cells/verification_faults/` | 2026-11-25, lane verify-home |
| `bench/results/small_training/` | `verification/records/cells/small_training/` | 2026-11-25, lane verify-home |
| `bench/results/verify_external/` | `verification/records/cells/verify_external/` | 2026-11-25, lane verify-home |
| `bench/results/release_verification/` | `verification/records/release_verification/` | 2026-11-25, lane verify-home |
| `bench/results/CELLS_INDEX.tsv` | `verification/records/CELLS_INDEX.tsv` (every row is under verification/records) | 2026-11-25, lane verify-home |

Record sidecars that cite an old path (for example the run scripts `identity_break/2026-09-15_arima-exog/cpu_cmd.sh`
and `2026-09-15_ties-sabotage/leg_*.sh`, which name `bench/results/classical_host/...`) are the record of how that run
was made and are not edited. So are ledgers citing them (`tools/identity_lanes/coverage_sabotage_proven.tsv` cites
`bench/results/verification_faults/...`; no tool opens those citations) and the dated release waivers in `docs/release/`. Readers that meet an old spelling in a record map it: the admission
READMEs under `bench/results/reference_admissions/` cite `bench/results/identity_break/retained/...`, which
`tools/release_requirements_retained.py` and `tools/release_identity_reference.py` read at the new root, and a git
revision older than the move is read at the old root (`LEGACY_IDENTITY_BREAK`, `identity_at_freeze.RECORD_OUTPUTS`).
The Mamba schema name `mojolearn.mamba.corpus.hash.v1`, the `MOJOLEARN_MAMBA_CORPUS_*` variables and `gen_corpus.py`'s own
text are names, not paths, and are unchanged. Experiment records under `experiments/` (catalogs, registrations whose
sha256 is pinned) cite `mamba/corpus/gen_corpus.py` and are not edited. Comments in Mojo and product modules that cite a record path are citations, not readers, and keep the old spelling.
