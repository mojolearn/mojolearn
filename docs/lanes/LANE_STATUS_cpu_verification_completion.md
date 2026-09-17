# CPU verification completion — active work

Started 2026-09-17 from main `32acd33d8`, following the user's explicit request
to keep going to every exposed algorithm and ship verification in the wheel.
Worktree: `~/mojolearn-wt/cpu-verification-completion`.
Evidence: `~/mojolearn-evidence/cpu-verification-completion`.
This is an active checkpoint, not a release qualification statement.

Baseline: 128 available CPU lanes, 51 withheld, 50 parallel exclusions. The
51 are 27 missing references, five stale references, five held for separate
records, two unwatched, and 12 awaiting additional vendor evidence. Historical
native controls cover every mapped lane/part for 52/246 appendix entries on
at least one fixture. That is not all-fixture/current-artifact coverage.

Changes in this branch:

- `tools/verify_cpu_batch.py` runs bounded per-lane clean/native sabotage arms,
  preserves incomplete records, checks input/source context, and requires all
  nine training controls. `--properties` additionally records extended checks.
- Scoped strict table generation: `verify --all --lanes ... --reference-table
  BASE --emit-reference CANDIDATE`. It refuses missing/conflicting parts and
  preserves unrelated cells and the legacy table policy. Coverage exposes
  per-lane admission policy in the installed package.
- Four fix-record lanes have strict candidate references and public selection
  changes pending installed-wheel/all-fixture-control gates. Legacy `identity`
  stays restricted to its fixed record so it cannot compare pre-fix answers.
  `embedding`, `embedding-sort`, and `ivf-euclidean` have all-nine historical
  training controls. `ivf` lacks an admitted ties training control and must get
  a fresh run before this batch is integrated. Native code already has the
  ties value perturbation; do not invent an additional one without a rerun.
- The first fresh metrics H/C/V controls moved eight fixtures but not negative.
  `33c6d80da` adds a sabotage-only zero-entropy branch fault; rerun is owed.
  No production numerical operation changed.

Current validation: 276 focused tests pass. The eleven classical lanes pass
CPU base-fixture preflight. Explicit replay of the five separate-record lanes
had 144 IDENTICAL, 63 N/A, 18 OWED (all missing kmeans-sqrt inference/model/batch
references), zero divergences/refusals. Installed fix-record replay stopped at
its all-nine native-control assertion for ivf; do not count that as passing.

In flight: RunPod CPU pod `0krigxfoki5p54`, two vCPUs, 60-minute on-pod lease
and verified dead-man, at $0.06/hour. It runs source `f0d53b4ed248` with six
production families and five sabotage families, then eleven lanes in
`verify_cpu_batch.py`, 180-second budget per arm. Runner session logs at
`cloud-classical-run.log`, fetched results under `cloud-classical/remote/leg_out`.
At last observation all eleven builds passed and metric H/C/V was 8/9 controls;
GBDT weighted-score controls were running. Let the runner finish, retain its
failed cells, and verify teardown. Never kill it merely to start another pod.

Only one cloud job (two vCPUs) and one local single-thread worker may run.
Local work uses mac_slot.py and all numerical library thread limits of one.
The eight-lane neural extended-property preflight has a 300-second total local
budget and preserves its checkpoint if incomplete. No Metal run. No PyPI upload.

Next: finish classical records, rerun metrics plus IVF (include inference
bindings needed by saved-model replay), admit only complete observed results,
repeat installed-wheel replay, then continue fresh neural/low-bit records and
additional properties. Keep complete hardware/parallel/final-wheel gates open.
