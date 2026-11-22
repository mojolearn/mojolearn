# verification/: the one home of verification data

Everything a verification reads or records lives here, except the two homes below that stay where they are
for a reason. `bench/results/` holds boards and measurements only.

| What | Where | Who reads or writes it |
|---|---|---|
| What `python -m mojolearn verify` ships | `python/mojolearn/verify_reference/` (unchanged) | the wheel packers copy it; verify opens it from the installed package |
| Recorded classical host columns and saved models | `verification/records/classical_host/` | `tools/classical_host_gate.py` (`host_surface.CLASSICAL_GPU_COLUMNS`), `cpu-identity-gate.yml`, `test_host_surface.py`; two files are rows of `verification/records/CELLS_INDEX.tsv` |
| Recorded forest / GBDT host fixtures | `verification/records/forest_host/` | `tools/forest_host_gate.py`, `forest-host-gate.yml`, `cpu-identity-gate.yml` (`host_surface.FOREST_RECORDED_ROOT`) |
| Column-shaped records outside the identity columns (CELLS_INDEX rows and their run directories) | `verification/records/cells/{verification_faults,small_training,verify_external}/` | `tools/verification_matrix.py`, `tools/part_set_census.py` (through `verification/records/CELLS_INDEX.tsv`), `test_verify_small.py` |
| The dated release records up to 0.8.65 that documents still cite | `verification/records/release_verification/` | `tools/release_evidence_set.py provenance_history` (MOVED_DIR; older ones through git at LEGACY_DIR) |
| Native test fixtures | `checks/native_tests/fixtures/` (stays) | the native tests beside them own them |

Rules: a record is never edited or moved by hand except with `git mv` of a whole group, together with every reader,
in one commit; the indexes move with their data and `python3 tools/record_index.py check` stays `listed_ok`, 0
unlisted. A new `*.log` is not added here (`.gitignore`, `tools/hooks/check_tree_additions.sh`); a tracked log moved
in with its record stays.

## Where it was (records and older docs cite the old paths)

A path that is part of a record's content (a JSON that stores its own path, a table citation, a manifest row) is data
and keeps the old spelling. Read it through this map:

| Old path | New path | Moved |
|---|---|---|
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
`bench/results/verification_faults/...`; no tool opens those citations) and the dated release waivers in `docs/release/`.
