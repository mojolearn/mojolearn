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
