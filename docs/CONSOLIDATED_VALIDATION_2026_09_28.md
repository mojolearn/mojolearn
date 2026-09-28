# Consolidated validation status, 2026-09-28

The Apple branches were merged into main before the coordinated native
comparison. Main also contains `apple-merged-owed` at `79559ad0f` and the
final `neighbors-apple2` commit `75ddcae2e`.

## Completed Apple and AMD evidence

All **445 single-device configurations agree** between Metal on M4 and HIP
on AMD, with **1,671 matching numerical parts**, no missing records, and no
remaining differences in the final base-fixture comparison. Each selected
GPU record also passed comparison with its local CPU record. The comparison
binds the expected lane and batch revisions and preserves the source commit
and original record hash for every selected column.

- [Final comparison](../bench/results/consolidated_check/2026-09-28_final-base/crossvendor.json)
- [Exact 445-lane plan](../bench/results/consolidated_check/2026-09-28_final-base/plan.json)
- [Evidence scope](../bench/results/consolidated_check/2026-09-28_final-base/README.md)

The inventory contains 504 configurations. The 59 physical parallel-device
configurations are explicitly outside this single-device comparison. A base
fixture comparison is not a claim about every input size or configuration.
Declared inapplicable properties remain visible; they are not numerical passes.

The original base sweep ran once at `308878e80679`. Subsequent runs covered
only affected lanes. GLM, TreeSHAP, the repaired probes and regression metrics
received all-nine-fixture evidence. For the remaining new batch probes, the
eight additional fixtures were collected separately and combined with the
existing base records; the base fits were not repeated.

## Repairs found by the coordinated checks

- GLM fixture generation now uses package-owned exponential arithmetic;
  platform NumPy exponential implementations had produced different targets.
- TreeSHAP skips unreachable zero-cover paths before undefined division.
- Metal uses the established eigensolver by default. The new eigensolver
  triggered an actual Metal compiler crash; its experimental opt-in remains
  unqualified. The SVD improvement and CUDA/HIP defaults are retained.
- Preprocessing handles absent optional CPU exports without treating them as
  missing mandatory implementations.
- All 57 formerly undeclared batch properties now have executable probes or
  explicit structural reasons. Bootstrap probes compare replicate distributions;
  NMF probes its row-independent inverse instead of a globally converged solve;
  LayerNorm retains the odd fixture's full feature width.
- Regression metrics explicitly return a canonical NaN for an undefined
  zero-weight infinite score, before platform-dependent invalid arithmetic.

## Reference admission

The 230 missing-reference lanes and two ordinary stale GMM lanes have been
resolved: **232 scoped admissions**, with complete nine-fixture witnesses and
independent device-class corroboration. Existing unrelated cells and record
prefixes were preserved in every admission. No reference was changed merely
to hide a mismatch.

The sole remaining stale lane is `par-gmm`, whose physical multi-device
qualification is still owed. The admission records and unchanged raw JSONs
are committed under `bench/results/identity_break/2026-09-28_admitted-*`.
A persistent local archive, including failed diagnostic records and SHA-256
manifests, is at `~/mojolearn-evidence/consolidated-2026-09-28/`.

## CUDA and release work still pending

The original NVIDIA host was retired at 19:19 UTC before this task's queued
numerical sweep ran. A replacement shared A40 host subsequently became
available. The [pinned CUDA continuation](../tools/consolidated_check/CUDA_RESUME_20260928.md)
covers all 445 lanes exactly once at their applicable source snapshots;
CUDA qualification remains pending until its saved records pass comparison.

No PyPI release has been made by this task. The optional `mojolearn[verify]`
dependency and bounded verifier workers are on main. Expanded runtime APIs
still have the separate [strict NumPy policy blocker](NUMPY_RELEASE_BLOCKERS_2026-09-28.md).
Base import without NumPy does not discharge that API-level audit. Candidate
wheel installation and the normal release gates remain required.
