# verification/: the one home of verification data

Everything a verification reads or records lives here, except the two homes below that stay where they are
for a reason. `bench/results/` holds boards and measurements only.

| What | Where | Who reads or writes it |
|---|---|---|
| What `python -m mojolearn verify` ships | `python/mojolearn/verify_reference/` (unchanged) | the wheel packers copy it; verify opens it from the installed package |
| Recorded forest / GBDT host fixtures | `verification/records/forest_host/` | `tools/forest_host_gate.py`, `forest-host-gate.yml`, `cpu-identity-gate.yml` (`host_surface.FOREST_RECORDED_ROOT`) |
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
