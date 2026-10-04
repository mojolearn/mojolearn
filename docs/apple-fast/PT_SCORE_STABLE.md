# WIP: centered PowerTransformer numerical repair

**PROMOTED: FAST+Apple DEFAULT, rollback `MOJOLEARN_PT_SCORE_STABLE_OFF`** (w2-pt-centered2-quality PASS; taxi 293.4 -> 190.8 ms, istella 2206.6 -> 1530.5 ms). Original checkpoint note follows.

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

### Reference correction and its justification (v2, `centered-mle-v2`)

**v1 premise refuted.** v1 kept sklearn's lambdas (52.6079, -63.3852). M3
job w2-pt-centered-quality (f88ed2cf6) failed in arm A's reference dump:
`check_objective_oracle` found that scipy's llf and the centered f64 NLL
agree at sklearn's column 7 lambda, but a step of 1e-2 * 63.4 lowers that
NLL. So sklearn's column 7 lambda is not even a local optimum of its own
objective. Cause: sklearn's YJ objective forms ((x+1)^l - 1)/l naively; at
l ~ -63 every row rounds to 1/63 in f64 (the same collapse as its column 7
output, std 0), so its optimizer works on rounding noise there.

**v2 reference = the likelihood optimum, computed stably.** For every
homogeneous-sign column (all Box-Cox columns; YJ columns with min >= 0 or
max <= 0), `centered_mle` minimizes scipy's NLL definition evaluated in
centered f64 coordinates (log Var(g) = log Var(v) + 2a log t(anchor), no
saturated term). The range is the device's standardized bracket
mid +- max(8, 8/span), widened to contain sklearn's lambda, and widened
again x4 while the minimum sits on an edge. A 401-point scan then bounded
Brent (xatol 1e-12 relative) finds it. Mixed-sign columns keep sklearn's
lambda: they are well conditioned and passed before. sklearn's lambdas stay in
the artifact as `<fixture>_reference_sklearn_lambda`, and every moved column
is printed (`PT-ORACLE-LAMBDA`), but they are not the target.

Why this is the right target: PowerTransformer's contract is the maximum
likelihood lambda. sklearn is only one f64 implementation of it, and here it
is shown wrong by its own llf. No arm output enters the computation, and arm A
(main) is scored against the same target, so the comparison stays fair. On
well-conditioned columns the optimum equals sklearn's to optimizer tolerance
(the printed max_relative_shift shows this). Checks that guard the target,
which can only fail a run:
1. scipy llf equals the centered NLL within 1e-9 per observation at the
   reference and stress arm lambdas.
2. The reference lambda is a local minimum (steps 1e-2 max(1,|l|)), now on
   every fixture.
3. The stable transform at the reference lambda matches a 160-digit
   Decimal evaluation of the original formula (stress, 17 rows).

The reference output, objective and normality all derive from the v2
lambda. Gate thresholds (lambda 1e-5, output RMS 1e-5, NLL 1e-7, normality
1e-4, roundtrip 1e-5, all relative to main's error) and fixtures are
unchanged. `compare` refuses an artifact that is not `centered-mle-v2`.

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
