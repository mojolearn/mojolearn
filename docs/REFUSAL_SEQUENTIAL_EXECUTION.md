# Sequential refusal closure and evidence admission

Source-only plan, September 6, 2026. Nothing in this document was executed.
Only root runs validation, builds or measurements, one job at a time, with
at most three CPU cores (the existing remote guards use two). Subagents must
never run them. Preserve the frozen 0.6.0 release snapshot and staged wheels.

## What the 9 / 10 / 8 audit actually counts

The sibling `mlsys/results/refusal-audit-2026-09-06.json` classifies 27
historical round-11 refusals: nine intentional, ten implemented elsewhere
in current source, eight needing ports or capacity changes. Two additional
named CPU ExtraTrees configurations are host/no-card evidence. This is a
source inventory, not a new 27-case numerical certificate. The matrix is
round 11, commit `144aa5b`; the audit describes later library source at
`2e53699e`. Keep those source scopes distinct.

Read-only review of `mlsys/paper/gen_numbers.py` around its refusal section
found these admission weaknesses; no sibling paper file was edited:

- The assertions check total list lengths against refusal counts and host
  list length against host/no-card counts. A duplicate replacing a missing
  cell, a wrong cell name, or reassignment to the wrong family/category can
  preserve both sums and pass.
- Neither exact refusal-set membership nor disjointness of categories and
  host sets is checked against the retained E2/E2U cell records. The script
  does not require each listed refusal to actually have a refused verdict
  on both vendors with the expected named error.
- `matrix_round` and `matrix_commit` are not bound to the selected `rnd`.
  That round is the first entry marked `reported_in_this_paper`; uniqueness
  of that marker is not checked. A later round with the same totals could
  silently inherit this old classification.
- The audit's `status`, date, source revision and source-document contents
  are not validated or hash-bound. Counts alone cannot substantiate the
  “implemented” label or its later-source provenance. The earlier generator
  section does assert NVIDIA/AMD aggregate-count equality; that does not
  check the identity of the cells in either set.
- These are Python `assert` statements, so optimized Python can remove
  them. A future fail-closed audit should use explicit validation errors,
  require the exact schema/category keys, and reject malformed/duplicate
  JSON members before generating paper macros.

Before changing paper classifications, require exact family-specific sets
of refused cell IDs from both retained vendor matrices; categories must be
pairwise disjoint and their union must equal those sets. Bind the unique
reported round/commit and the source-audit revision explicitly. Validate the
two host IDs and null-card scope independently. Preserve the old record and
publish a separate new round with hashes rather than overwriting history.

## Current additive source changes, separate from frozen 0.6.0

Distance-weighted k-NN voting, Manhattan/Minkowski p=1, and brute-force cosine
already existed. The new k-NN edit extends the pinned selector through k=1024:
256-pair staging for k<=256, 1024-pair staging only for the extension. It is
authored and unqualified; k>1024 and kd_tree remain refused. See
[extended k-NN status](../neighbors/EXTENDED_KNN_SOURCE_STATUS.md).

PCA whitening has additive native transform/inverse exports and wrapper state
validation authored in the current checkout. It uses the existing FP32
sample-count/singular-value schedule and skip-zero singular-value rule;
degenerate columns do not promise unit variance. The current source gate is
`python/mojolearn/tests/test_pca_whitening_surface.py`. Presence of these
exports in source does not put them in the frozen 0.6.0 binaries. Do not
silently patch staged artifacts or relabel the historical `pca_whiten` row.

## Small cases first, then exact historical cells

These are proposed admission fixtures, not observed results. Freeze input
bytes and parameters before execution; use separate fresh result directories
and retain every failed attempt. Each row is an independent serial root job
or bounded subcase of one job, never a parallel GPU launch.

| Order | Small initial case | Required behavior before matrix rerun |
|---|---|---|
| 1 | k-NN distance vote: index `[0,0,2]`, query `[0]`, k=3; labels `[7,3,7]`, targets `[[2,-8],[6,4],[100,100]]` | Zero-distance row mask excludes the distant point; tied classifier chooses 3, probabilities .5/.5; regression gives `[4,-2]`. Then index `[1,2,4]` changes the weighted winner relative to uniform. Existing authored gate covers these. |
| 2 | k-NN pinned capacity: two queries and 1056 one-dimensional dyadic points with duplicates | Independent sorted distance/index reference and repeated bytes at k=256,257,300,513,1024; k=1025 refuses. The authored gate includes all these boundaries (root added 300 after the audit); it remains unexecuted. The exact `knn_k300` matrix cell must also be exercised before closing that historical configuration. |
| 3 | Brute-force Manhattan: four two-dimensional integer points, origin query, tied L1 distances; cosine: nonzero axes and their positive multiples | Manhattan total ordering, explicit cosine zero-vector refusal, scale/tie behavior under the pinned profile; no kd_tree or ball-cover cosine admission. Use independent reference and raw distance/index arrays. |
| 4 | PCA whitening: fixed nondegenerate 8x3 FP32 matrix, two components; then a rank-deficient repeated column | Transform/inverse agreement with the declared schedule, sample-count convention, skip-zero behavior and input/state validation. Compare components, singular values, explained variance, scores and reconstruction, not just variance≈1. |
| 5 | OLS: 8x1 planted slope/intercept; full-row-rank 4x6 underdetermined matrix; eight observations with nonuniform weights including one zero | Separate scalar fit, minimum-norm solution and weighted centering checks against an independent FP64 reference; zero-total weight refuses. Preserve each of the three historical cell IDs. |
| 6 | Weighted DBSCAN: five 2D points, a close pair with weights changing its core status, one isolated point | Weight changes core threshold as specified; labels/core flags and traversal stages retained. Nonconverged chain/maxiter cases must still refuse. |
| 7 | KDE cosine: four nonzero 2D points and two heldout points | Verify metric and density normalization semantics, support restrictions and finite outputs against an independent reference; accepting a name is insufficient. |
| 8 | ExtraTrees classifier best-first: 16x2 fixed FP32 data, two classes, one tree, max_leaf_nodes=3 | Actual best-first frontier selection and leaf budget; full model/heldout predictions plus growth-stage card. Then rerun historical max_leaf_nodes=64 cell. |

After each small gate passes, rerun its **unchanged historical E2/E2U spec**
from `tools/e2_matrix_fit.py` / `tools/e2u_matrix_fit.py` and the retained
`spec` object. Small replacements do not close the old configuration's cell.
In particular the historical OLS wide fixture is 64x100, and the PCA/TSVD
wide capacity cells are distinct consumers of the shared Gram path.

Then address remaining genuine ports separately: Scott bandwidth host rule
with an explicit sample/dimension/FP32 policy; shared wide Gram capacity for
both PCA and TSVD; Manhattan DBSCAN; OWL-QN logistic L1; full SVD; randomized
SVD with fixed RNG/oversampling/power-iteration state. Each needs its own
bounded fixture and effective control before a full matrix rerun. k=300 is
now source-authored capacity work, not a remotely validated closed port.

Keep all nine intentional refusals named in the audit. Do not alias invalid
options to supported algorithms, return unconverged DBSCAN labels, synthesize
a Quantile Newton Hessian, or convert requested CPU ExtraTrees rows to GPU
rows simply to reduce a refusal count.

## Required raw evidence and vendor order

The original NVIDIA matrices are
`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/e2_cells.json` and
`e2u/e2u_cells.json`; AMD siblings are under
`bench/results/e1/2026-08-23_172650-mojolearn-e2-amd/`. Retain their referenced
per-stage `.card` files, input hashes, specs, model serialization/prediction
hashes, and actual refusal strings. A `.cell.json` or summary count without
its referenced card cannot establish full-stage identity.

For each new round retain the exact source inventory, compiler flags and
runtime, installed binary and wrapper hashes, native vendor/mode readback,
fixture bytes, command, guard log and hash-bound zero-exit receipt. Capture
raw numerical outputs alongside E2/E2U stage cards wherever practical;
stage hashes are evidence of recorded intermediates, not an independent
mathematical oracle. Refusal cases need an actual reached named refusal;
host cases need their own model/output evidence and remain host/no-card.

Execute NVIDIA first, then AMD on the exact same frozen source/fixtures,
comparing the two retained runs. Apple is the third **separate** evidence
column, after those succeed and only when root has current authorization
for Apple execution; this plan does not override a no-Apple-testing
instruction. Historical Apple cards may describe their original source
round but cannot qualify new k-NN/PCA binaries. Until a matching new Apple
leg exists, report the new scope as NVIDIA/AMD only. No new cloud resources
or local model jobs are authorized by this document.

After all applicable jobs and comparisons pass, root can produce a new
round report and separately update the source-audit categories. Neither
9+10+8=27 nor a successfully generated paper macro is numerical closure.
