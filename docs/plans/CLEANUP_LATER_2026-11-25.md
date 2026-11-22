# Cleanup left for later (2026-11-25)

The tree went from about 34,700 tracked files to about 12,900 on 2026-11-25 (run output, dead scripts, dead Mojo
files and functions, packed fixtures, old plans). What remains is real work, not deletion. Four items, each a
project of its own. None blocks the next release except the last, which happens at the release.

## 1. Python tests to Mojo (737 files under `python/mojolearn/tests/`)

- About 120,000 lines. The inventory classed 350 files as NATIVE or MIXED (their numeric checks belong in Mojo);
  the rest test the Python boundary and stay.
- None of the Mojo suites that inventory names exist yet, so today these Python tests ARE the coverage: no
  Python test is deleted before its Mojo suite is registered and green.
- Rule: AGENTS.md "Move native correctness tests to Mojo". The inventory and per-file table are in git history
  (plan TEST_MIGRATION_2026-11-14).
- Proof per ported file: the new suite in `checks/native_tests/registry.mojo`, one full native run green on the
  CPU box, then the Python file (or its native functions) deleted in the same change.

## 2. Bindings (about 1,070 files under `bindings/`)

- Mechanical merging is used up: 35 small modules were inlined into their one importer on 2026-11-25
  (`~/mojolearn-evidence/briefs-20261121/dead-code-work/inline_single_importer.py`). What stopped the rest:
  285 small files have several importers, 53 are named in build-group or audit lists, 28 would mix
  Python-holding code into Python-free code (AGENTS.md "Python objects only at the public boundary").
- Going further is a redesign per family: one UNPACK/PACK file and one RUN file per family in place of one set
  per public call, with the build-group lists and the audit tables updated in the same change.
- Proof per family: parse of every group, the full native run, the tool tests. One family per round.
- A trap already met: a module docstring moved into the middle of another file is an expression at file scope and
  does not compile; turn it into comments.

## 3. Release leg scripts (`tools/*_leg.sh`, `tools/gemm_remote_leg.sh` at about 6,000 lines)

- The per-vendor copies differ by hundreds of lines. One parameterized leg would replace several.
- It can only be proven by a release rehearsal, so do it right after a release, never before one.
- Same for the Linux legs that pick the newest results directory by walking a folder, and for
  `tools/retain_run_summary.py`, which writes measurement summaries into the checkout.

## 4. Identity records (`verification/records/identity/`, 1,292 files)

- These are the inputs of the verify table and the verify modules read them directly
  (`python/mojolearn/_verify_reference.py`, `_verify_all.py`, `_identity.py`), so they cannot be packed or
  trimmed without a built wheel to prove `python -m mojolearn verify` afterwards.
- At the next release: record every lane on every device once, build the table from the new records only, then
  delete every older record and its index rows in the same change (Andrew, 2026-11-25).
- `verification/records/cells/` (343) and `release_verification/` (77) get the same treatment then.

## Smaller leftovers

- The member boundary door imports every family's door, so build partitions and per-hunk lane selection are lost
  (3 tool tests red: `test_lane_select.py::test_a_hunk_inside_one_mojo_def_selects_only_the_lanes_reaching_it`,
  `test_runtime_partition_closures.py` x2). Design done (Fable, 2026-11-25:
  `~/mojolearn-evidence/briefs-20261121/member-door-partitions-RESULT.md`): 39 families get a published boundary
  slot in the existing entry table, resolved once per caller image. Decide first: xdecomp's Isomap/MDS/LLE kinds
  (second entry from runtime_manifold, or move to the MANIFOLD family); add a native check that admits one kind
  per family (a missing slot must not fall back silently); read how a pending Python error crosses images.
  Build-time precision only: no product behaviour or measured row depends on it.
- 6 tests of `test_model_portability_evidence` read 19 logs an earlier cleanup deleted from a reviewed record.
- The tools check on the CPU box uses about 7 of 32 cores: its test files are not split across the box.
- Fixture archives (`tools/fixture_packs.py`): after regenerating a case, run `pack`; runners call `unpack`.

## 5. The Mojo test harness (Andrew, 2026-11-22: "add to the plan"; first step of section 1)

Do not wait for Modular's test tooling. One small harness in the repo, a few hundred lines:

- a case table (name, function, parameters), so one suite covers a family by data, not by file;
- one registry that launches every suite, which removes the 247 wrapper files under `checks/native_tests`
  (each only calls its suite's `run_checks()`);
- one process per check, across all cores (the native runner already does this);
- one summary format the box tools read.

When: right after the proving run is green on the integration build, in the same change as the wrapper cut. Proof:
the same check names as before, full native run green on the CPU box and the L40S. Then the 164 numeric Python
tests move into tables (section 1), each Python test deleted only after its Mojo suite is green.
