# Lane status: lane/identity-record-next (the 178-lane identity record)

Updated 2026-09-15 14:20 UTC by the record2 agent. STOPPED on Andrew's order.

## The order

GPU records are required only for PyPI releases, and then only one Apple, one NVIDIA and
one AMD column. No full records otherwise, no two-device columns, no extra vendors, no
new boxes. Running legs could finish. `TRAINING_GPU_COLUMNS` and the gate are NOT
switched.

## What this branch carries

- `bench/results/identity_break/2026-09-15_178-lanes/`: a PARTIAL record at df617c699,
  README first. Complete: the Apple M4, NVIDIA H100 and AMD MI325X one-device columns,
  1602 cells each, `diff.three-columns.txt` IDENTICAL=1602 (infer/model IDENTICAL=1998,
  N/A=1206; batch IDENTICAL=1278, N/A=324); against the 166-lane record only the nine
  kmeans-sqrt cells moved (the 9fde8f5f7 fix). Also the two-H100 par column (all 39 par
  lanes, IDENTICAL x3 per vendor) and a two-MI300X par column missing par-samba-clip,
  par-gmm, par-hdbscan, par-kernel-ridge and par-rbf-sampler.
- `unmerged-gate-switch.patch` in that directory: the gate switch and the removal of
  `TRAINING_FIX_*`, run locally only, not applied.
- `docs/lanes/BRIEF_rf_reg_gamma_ig_moved_under_contention_2026-09-15.md`: one MOVED cell
  on the MI325X with five processes sharing the GPU; OPEN.
- No code change against main.

## Running

Nothing. Every box this lane rented is verified deleted: RunPod pods 2o8sm4yf2we58b,
dx1hal547mt7u2, d6guwv4ll9lyii, zzeusxtb0d7qmp and wnjvi7fhvdi0w1; DigitalOcean droplets
600659534 and 600667291 (each leg log reads HTTP 404 after the delete).

## Evidence

`~/mojolearn-evidence/record2/legs/` (every raw leg directory of this lane) and
`~/mojolearn-evidence/record2/scratch/` (bodies, generator `make_body.py`, merge parts,
Apple chunk logs). The 166-lane record's untracked raw legs that lived in the reused
worktree are in `~/mojolearn-evidence/record166-legs-harness/`. The worktrees were removed.

## If a release wants these columns

1. Take only the three one-device columns (already here and complete).
2. `git apply bench/results/identity_break/2026-09-15_178-lanes/unmerged-gate-switch.patch`
   on a branch from main, resolve against whatever main changed, then
   `python3 tools/docs_facts.py --check`, `python3 packaging/wheel_ci.py pins .`,
   `python3 packaging/wheel_ci.py inventory python/mojolearn`, and
   `cd python && python -m pytest mojolearn/tests/test_host_surface.py mojolearn/tests/test_cpu_training_misc.py`
   (needs built bindings), then the seven-runner gate. If main moved past df617c699 in a
   way that changes GPU bits, the columns must be retaken instead.
