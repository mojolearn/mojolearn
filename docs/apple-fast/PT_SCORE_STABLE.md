# WIP: centered PowerTransformer numerical repair

Checkpoint requested by manager/user handoff. **Uncompiled, unvalidated,
opt-in only; do not merge or time yet.** Branch `lane/apple-fast-pt-precision`,
worktree `~/mojolearn-wt/pt-precision`. Baseline merged main `12fdd6697`.
New define `MOJOLEARN_PT_SCORE_STABLE`, binding x_prep; implies score search
and COLBATCH, disables incompatible speculative/fused-transform paths.
Old `MOJOLEARN_PT_SCORE` remains available to reproduce held bc112b172.

Actual prior failure: stress columns0–5 improved; nearconstant6/7 failed.
Reference lambda52.6079/-63.3852 lies outside[-8,8], and reference column7
standardized output collapsed to a constant (normalityNaN). Saved artifacts
and diagnosis: `~/mojolearn-evidence/apple-fast/pt-score-stress/`.

## Implemented WIP

- Homogeneous-sign columns use affine-equivalent coordinates
  `v=(g_lambda(x)-g_lambda(anchor))/t(anchor)^a`, evaluated with
  `log1p(sign*(x-anchor)/t(anchor))` and small-argument expm1 series. No
  original large offset/scale is materialized. Mixed-sign columns retain
  prior score representation.
- Same coordinates are used in parallel GPU score, standardized output and
  inverse. Anchor/sign are GPU-derived and retained in fitted state. Centered
  score Jacobian uses signed log-ratio, preserving the original MLE.
- Standardized homogeneous columns extend the bracket using observed
  log-span (radius max8,min1e6,8/span). This admits nearconstant optima beyond
  +/-8 while bounding new narrow-column exponents. Raw unstandardized output
  retains original[-8,8] bounds. Boundary/overflow behavior still needs review.
- New helper module `x_prep/pt_center.mojo`; no CPU fit fallback/native f64.
- Oracle correction preserves sklearn fitted lambdas and likelihood metrics;
  it standardizes mathematically equivalent centered float64 transforms.
  A 160-digit Decimal check on17actual stress rows independently compares
  against the original power formula. Legacy reference diagnostics retained.
  No existing lambda/NLL/output/normality thresholds changed. Added inverse
  roundtrip gate with1e-5 tolerance.
- `tools/pt_score_quality.py repair-reference MAIN.npz OUT.npz` corrects a
  saved full oracle without model fits, preserving GPU arrays/lambda/NLL.
- `tools/pt_score_stable_verified.py` adapts the manager's existing manifest,
  loaded-binary hash, artifact hash and quality-before-timing gate; takes
  exact full SOURCE SHA, confirms stable export false/true in A/B and restores
  installed binding after quality. Fresh tags only.

## Review pass 2026-10-04 (wave 2, still uncompiled)

Reviewed against the math; fixes are in the commit that adds this section.

- Offsets: `pt_init` q = [METHOD, ST, d, LAMBDA, STATE, LEVAL, ANCHOR, KIND,
  STANDARDIZE]; `pt_apply`/`pt_inverse` q = [..., OUT, ANCHOR, KIND]. ANCHOR =
  col_stats mean (row 1), bracket span from min/max (rows 3/4). STATE words
  8/9 carry anchor/kind; the golden fold that also uses them never runs under
  the define (COLBATCH skips pt_map, pt_fold goes to the score kernels).
  New guard: pt_init skips the centered block when ANCHOR is `_NONE`; the
  Python layer never sets bit 32 on the host binding.
- Score: centered NLL = -(l-1) sum J + n/2 log Var(v) + n a log t(anchor);
  d/dl of the last term is n*kind*log t(anchor), exactly what the centered
  Jacobian (signed log-ratio) subtracts, so the score's zero is the
  original MLE. Checked for both kinds (a = l, a = 2 - l).
- Cross-sign queries (inference only): the old form subtracted two
  materialized psi values and divided by t(anchor)^a (cancellation and
  inf/inf at |l| >> 8). Now `cross_scaled` forms psi(x)/t^a as
  (exp(bL - aL0) - exp(-aL0))/b and psi(anchor)/t^a as -kind*expm1(-aL0)/a
  (kind*L0 at a = 0). Inverse uses the same pieces. Zero is same-side for a
  nonpositive column (psi(0) = 0 on both branches).
- Raw output (standardize=False): bracket stays [-8, 8], fitted state keeps
  no anchor, transform/inverse use the original power: raw output equals g(x).
- Overflowed score (|a lg| past exp range at a far bracket edge, possible
  for wide columns at radius 8): bisection now cuts toward the pivot
  (0 centered positive, 2 centered negative, 1 mixed) instead of stalling
  at the midpoint forever. Old `MOJOLEARN_PT_SCORE` keeps its behavior.
- Near-constant stress cols 6/7 (x = 10 +- 0.001): span ~ 4.5e-4, radius
  ~ 1.8e4, so |a lg| <= 8 everywhere in the bracket; 50 bisections give
  width 3e-11. In centered coords lg ~ 1e-4 with f32 relative error, the
  series path keeps y and dy relative-accurate, so the f32 score resolves
  lambda far below the gate (lambda gate needs only beating main's 0.89
  relative error; objective must not be worse than main's by 1e-7).

### Reference correction and its justification

sklearn's lambdas are kept (52.6079, -63.3852). Evidence they are the MLE,
not optimizer noise: (1) the scipy llf at those lambdas beats both GPU arms
by 1.4e-5 and 3.5e-5 per observation (diagnosis.jsonl), consistent with
skew-driven optima of a 1e-3-wide normal column (sample skew sd 0.0077
needs |l - 1| ~ 30); (2) new committed check `check_objective_oracle`
computes the f64 NLL in centered coordinates (log Var(g) = log Var(v) +
2a log t(anchor), no saturated term) and asserts scipy's llf equals it
within 1e-9 per observation at reference AND arm lambdas, and that the
reference lambda is a local minimum (steps 1e-2 max(1,|l|)). Only the
reference *transform* is replaced: sklearn's own transform materializes
(11^-63 - 1)/-63 = 1/63 exactly in f64, so its column 7 is constant (std
0, normality NaN); the stable centered transform is the same standardized
function (positive affine map), checked against 160-digit Decimal on 17
rows. Gates, thresholds and fixtures are unchanged; the new checks can only
fail a run.

Known limits: mixed-sign near-constant columns (e.g. 0 +- 1e-3) keep the
[-8, 8] raw-coordinate search (no fixture; any lambda there gives nearly the
same output). A model fitted in FAST with |lambda| > 8 then transformed by
an IDENTICAL binding uses the original power (overflow possible).

Compile spec (manager, M2): both x_prep arms of this branch head,
A = "" and B = "-D MOJOLEARN_PT_SCORE_STABLE":
`compile_arms_m2.sh lane/apple-fast-pt-precision x_prep MOJOLEARN_PT_SCORE_STABLE`.

## Owed after handoff

1. Review centered equations/signs, wider-bracket behavior, Python arena
   offsets/fitted state, same-sign/cross-sign inference and inverse, and
   raw unstandardized behavior. Compile both x_prep arms on manager M2:
   `compile_arms_m2.sh lane/apple-fast-pt-precision x_prep MOJOLEARN_PT_SCORE_STABLE`.
2. Provision dependencies and verified arms on the exact SOURCE checkout;
   queue M3 quality only:
   `python tools/pt_score_stable_verified.py quality SOURCE pt-centered-quality`.
   This runs full fixtures, corrected reference and Decimal oracle checks.
3. Inspect all per-column gates, especially stress6/7 and inverse output.
   No threshold relaxation, dropping stress fixtures or default promotion.
4. Only after matching PASS, one scored M3 pair per dataset via
   `python tools/pt_score_stable_verified.py timing SOURCE pt-centered-quality pt-centered-taxi taxi`
   and analogous istella. Root owns queue/cloud and isolation windows.

Only Python AST syntax parsing ran locally (3files passed). No numerical
validation, Mojo build, GPU test, cloud action or timing was run. This is a
recoverable implementation checkpoint, not an accuracy/speed claim.
