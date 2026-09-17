# Verification evidence audit: remaining gaps

The 246-entry appendix is an inventory, not a claim that all entries have
sabotage, CPU/GPU and multi-GPU qualification. This work follows the user's
request to close the remaining gaps. Comprehensive 0.8.7 qualification remains
open; no release was published and no cloud resources were rented here.

## Closed in this change

- The wheel's coverage inventory now carries historical evidence per lane and
  per appendix entry. Every source record has a path, full commit and SHA-256.
  Build controls and harness controls are separate. All entries explicitly
  remain outside release qualification by this historical inventory.
- The sabotage audit no longer treats an exception, N/A, missing output,
  single repeat, or an unstable clean arm as a working negative control.
  The wheel snapshot additionally requires matching full commits, input and
  held-out fingerprints, device selection, lane revisions, protocol and vendor.
  This tightens accounting; it does not turn previously missing experiments
  into completed experiments.
- One-versus-multiple GPU evidence requires distinct requested device indices,
  the same commit, input/protocol and binding digests, and repeated stable
  hashes on both sides. Differences are retained. Recorded device requests
  are not independently attested physical use.
- Public report comparisons now require matching input and protocol provenance.
  Identical output hashes from incompatible or undocumented experiments return
  INCOMPARABLE, exit 4. New reports record the requested parallel devices too.
- Both wheel builders include all 18 saved CTR model fixtures, drawn from the
  single existing source directory and checked against declared digests.
  CPU replay resolves them without an environment variable or source checkout.
  Corrupt or missing models refuse. The two CTR inference lanes passed all
  nine fixtures twice from an installed CPU development wheel and were removed
  from the `unwatched` set. They now join default public CPU verification.

## What the current inventory actually establishes

229 registered lanes, 246 appendix entries, 222 enumerated API entries.
CPU selection has 122 available, 57 withheld, 50 parallel exclusions and no
undeclared CPU route among the remaining registered lanes. An available lane
can still have missing reference parts; execution remains the deciding check.

44 appendix entries have a matching historical build negative control for
every mapped lane/part on at least one recorded fixture under the stricter
pairing rules. This is not all-fixture sabotage coverage or a current-build
result. Other historical evidence may exist outside the scanned records.

39 of 50 parallel lanes have retained comparable one-versus-two-GPU training
records: NVIDIA has 39 lanes across all nine fixtures; AMD has 9 across all
nine under the strict same-binding pairing rule. These historical counts are
not qualification of the 0.8.7 candidate. The following 11 have no qualifying
pair in the snapshot on either vendor:

- par-forest-reg, par-forest-et-clf
- par-boosting-clf, par-boosting-reg
- par-gram-ols, par-gram-pca, par-gram-tsvd
- par-cd-elasticnet, par-svm-svr
- par-scaler-minmax, par-queries-nn

`tools/verification_evidence.py --write` regenerates the wheel snapshot from
repository records. It records the generator and harness digests and whether
the source tree was dirty. The inventory detects a changed harness instead of
silently presenting the snapshot as generated from that harness. Historical
observations never enter the numerical reference table through this tool.

## Validation

266 focused tests passed, including malformed/missing control arms, changed
input/protocol comparisons, host coverage and reference admission. The wheel
was built through setuptools with prebuilt CPU candidate bindings, without GPU
bindings or native compilation. It is a development wheel, not a fresh native
release build. Tests ran outside the checkout with no host/model/harness path
overrides:

- Coverage: all 246 appendix entries and historical evidence loaded.
- CTR: 18 lane/fixture cells, two executions each; 54 IDENTICAL parts,
  36 explicitly N/A, zero divergences, refusals or missing references.
- Two real OLS executions: public comparison agrees.
- Changed held-out fingerprint, unchanged result hashes: INCOMPARABLE, exit 4.
- Wheel ZIP contains all 18 model fixtures; the loader checks their digests.

Evidence and the development-wheel receipt are retained in
`bench/results/verification-evidence-probe/2026-09-17/`. The probe path excludes
these diagnostics from numerical reference admission. CPU work used single-thread
numerical libraries and at most two simultaneous single-core tasks. No GPU job,
native build, cloud rental or PyPI publication occurred in this change.

## Remaining work, in execution order

1. Freeze the source intended for release, including concurrent native changes.
   Build the final Linux and macOS wheels with at most three CPU workers total.
   The old 0.8.7 freeze and development wheels are not that artifact.
2. Execute clean and real sabotage builds for uncovered algorithms and variants.
   Retain clean/sabotaged pairs at the same source, fixture and protocol; prove
   that each arm changes a relevant result. A generic comparator perturbation
   is not a substitute for a native negative control.
3. Qualify installed final-wheel CPU and GPU routes, including applicable batch,
   gradient, ragged, step/full and sampler/replay properties. Keep the bounded
   Apple release policy; do not launch the historical seven-hour Apple matrix.
4. Record all 50 existing parallel lanes on one and two GPUs for NVIDIA and AMD,
   with the same installed wheel and binding digests. Cover the 11 unrecorded
   lanes first, and verify actual device use alongside requested indices.
5. Review unimplemented parallel surfaces separately from qualification of
   exposed drivers. The earlier multi-GPU audit names cross-validation folds,
   GPC, IVF, matmul, TSA prediction, embedding/metrics decomposition candidates.
   Do not claim those implementations exist or infer universal applicability
   from the number 246.
6. Admit the resulting references, replay them from the exact final wheels,
   run release admission, then publish. Missing references, stale fixtures and
   failed or absent negative controls stay open until observed.

Paid hardware work is awaiting the user's cloud spending cap, requested during
this session. The local Mac cannot supply NVIDIA/AMD multi-GPU evidence.
