# E2 RESULTS — the sub-feature matrix, three GPU vendors, bit for bit

**Claim demonstrated 2026-08-23 (round 2, commit `53d56ef`)**: every
decision-tree sub-feature the Python surface exposes — 64 gradient-boosted
configurations across all 13 CatBoost losses, 4 bootstrap types, 4 score
functions, the leaf-estimator overrides, both searchers, random strength,
the categorical paths (CTRs, one-hot, permutation counts), NaN modes, bin
widths 1/15/64/128/254, depths 2/6/10, weights, the overfitting detector,
border subsampling and a 200k-row regime; 13 Extra Trees configurations
including the host arm; 18 Random Forest configurations across
gini/entropy/poisson/gamma/inverse_gaussian — plus the four Mojo-only
training paths (depthwise growth, lossguide growth, MultiClassOneVsAll,
the feature-parallel searcher) — trains **bit-identically on Apple M4
(Metal), NVIDIA H100 (PTX) and AMD MI325X (HIP)** from one source at one
commit under `NUMERIC_IDENTICAL`, with per-stage certificates:

| column | cells | IDENTICAL (full card) | IDENTICAL (host arm, prediction hash) | REFUSED= (same message) | divergent |
|---|---|---|---|---|---|
| NVIDIA H100 vs Apple M4 | 99 | 93 | 2 | 4 | **0** |
| AMD MI325X vs Apple M4 | 99 | 93 | 2 | 4 | **0** |

The four Mojo-only cards are byte-identical on all three vendors
(`gbdt_depthwise` 94 records, `gbdt_lossguide` 205, `gbdt_multiclass_ova`
112, `gbdt_feature_parallel` 4). Both portable-arithmetic certificate lines
print the same number on all three: `check-portable-translog`
8705486125800438413, `check-portable-sqrtcos` 12295913102197186379.

**Train-here-infer-there**: every model fitted on the Mac (95 saved
`.model.npz`, all five families) was loaded on the H100 and on the MI325X
and predicted there: 95/95 prediction hashes equal on both, probabilities
included (`cross_infer_mac_models_on_box.json` in each box directory).

This extends E1 (one configuration per family, `E1_RESULTS.md`) to the
whole sub-feature matrix, and it took one round of fixes — which is the
method working: round 1 named every divergence by stage, the ledger named
the mechanism, one deviation closed the class.

## The two rounds

| | round 1, `c7a3493` | round 2, `53d56ef` |
|---|---|---|
| NVIDIA | IDENTICAL 29 (+2 host) · REFUSED= 3 · **OUTPUT-ONLY 42 · DIVERGENT 15** | IDENTICAL 93 (+2) · REFUSED= 4 · **0** |
| AMD | IDENTICAL 80 (+2 host) · REFUSED= 3 · **OUTPUT-ONLY 2 · DIVERGENT 4** | IDENTICAL 93 (+2) · REFUSED= 4 · **0** |
| cross-infer (raw) | 88/88 on both boxes | 95/95 on both boxes |
| cross-infer (proba) | 3 mismatches on both (host sigmoid) | 0 |

`OUTPUT-ONLY@stage` means the prediction hash matched but a card stage did
not — an output identity without a certificate, the E1 NVIDIA RMSE finding
— and the matrix counts it as a divergence.

**Round 1's divergences were ONE finding plus its relatives**, and every
ET and RF cell was already identical on both vendors:

1. **`std.math.sqrt` is not correctly rounded on NVIDIA.** `check-ieee-arith`
   on the H100: 180,714 of 2^20 hashed patterns one ulp off, 176,577 of
   them NORMAL (so not the denormal policy); division exact; fma exact;
   Cosine score shape 141,895 wrong. Mojo lowers `sqrt` to an approximate
   PTX sqrt; Metal and HIP are correctly rounded. Every `score /
   sqrt(denum_sqr)` therefore carried NVIDIA's last bit: 42 cells
   OUTPUT-ONLY at `winners.scores`, and where the wiggle flipped an argmax
   on tied scores (the Quantile class, ±α gradients) the tree itself
   diverged. **This is E1's unexplained NVIDIA `tree001.winners.scores`
   divergence, named.** IDENTITY_PATHS row 10's sentence "IEEE-correct on
   normals everywhere measured" was true of two vendors.
2. The last unrouted device transcendentals on the tree paths: the
   Bayesian bootstrap's `-log(u)` (first histogram divergent on NVIDIA),
   Box-Muller's `sqrt`/`log`/`cos` in `random_gen` (the random_strength
   and Poisson-bootstrap cells, both vendors), Lq's `__powf` (first
   histogram divergent on both vendors).
3. The probability links are HOST arithmetic: the wrapper's Logloss /
   CrossEntropy sigmoid was numpy's double `exp` (the host libm's last bit
   differs macOS↔Linux) and the Mojo links used Mojo's own host `exp`
   (~1e-13 off libm, measured). Raw margins identical, `proba` not.

**DEVIATION 258 closed all three** (`checks/numerics.mojo`): `portable_sqrtf`
— correctly rounded BY CONSTRUCTION (exponent-halving seed, three Heron
steps through the exact divide, candidate selection on the fma residual
with tiny inputs scaled up), measured 0 of 2^20 mismatches against the
float64 reference, so Metal/HIP bits do not move and NVIDIA's join them;
`portable_cosf` (Cephes, ≤ 2 ulp on Box-Muller's range); `portable_powf`;
`portable_exp64` for the links (host-only, Cephes double through fma);
`identical_sqrt/cos/pow/exp64` two-arm wrappers (FAST = stdlib verbatim;
every FAST cell hash measured unchanged). Gate `check-portable-sqrtcos`
runs in bootstrap phase 1 beside `check-portable-translog`.

Also landed on the way (all measured by this matrix): DEVIATION 407 (RF
criteria reachable from Python), 457/458 (ET: mode-gated host dispatch; the
regressor's `max_features` was silently overwritten by the binding — a
reach failure the matrix found because 1.0/0.5/0.1/3/"sqrt" hashed the
same), 257 (CatBoost's `EnsureNewtonIsAvailable` ported: Quantile/MAE/MAPE/
LogLinQuantile refuse Newton; the Huber-δ=1 / Quantile-Newton / MAE-Newton
hash coincidence was the sha256 of 20,000 zeros — CatBoost CPU reproduces
the blow-up), and the first smoke's lesson that `cat_features`,
`permutation_count` and `one_hot_features` read as INERT on a target whose
signal lived elsewhere (the cat cells now fit a target that depends on the
codes; all six knobs are distinct models).

## What REFUSED= means here

Four cells refuse BY NAME with the same message on all three vendors and
count as passes: `et_clf_bootstrap`, `et_clf_entropy`, `et_clf_maxleaf`
(options the ET port does not carry, each with a cited reason) and
`gbdt_quantile_newton` (CatBoost's own refusal). "This configuration does
not claim identity" is a certified answer.

## Method

- `tools/e2_matrix_fit.py` — 99 cells, one subprocess and one identity
  card each, prediction + proba + loss-curve + model-file hashes; inputs
  are a pure function of the seed (integer-exact targets, hashed NaNs and
  category codes) and their hashes are recorded and compared.
- `tools/e2_mojo_cards.sh` — the Python-unreachable paths, one fit per
  card, run twice as a run-to-run control.
- `tools/e2_matrix_diff.py` — N directories → the verdict table
  (IDENTICAL / OUTPUT-ONLY@stage / DIVERGENT@stage / REFUSED= / …);
  `identity_trace_diff.py` underneath for the stage walk.
- `tools/e2_remote_leg.sh` — one GPU droplet leg: create → bundle/clone the
  exact commit → bootstrap (phases 0-4) → cross-infer the Mac models →
  fetch → DESTROY, with an EXIT trap and a detached one-hour dead-man.
  Round 2 cost: 11 minutes on the H100, 8 on the MI325X.
- The Mac reference runs in a clean worktree at the same commit.

## Artifacts

`bench/results/e1/` (cards, e2_cells.json, e1_fits.json, e2_mojo_cards.json,
cross_infer_*.json, bootstrap.log; model files excluded, their sha256 in
e2_cells.json):

| run | Apple M4 | NVIDIA H100 | AMD MI325X |
|---|---|---|---|
| round 1 `c7a3493` | `2026-08-23_062356-MacBook-Air-1-terrabyte` | `2026-08-23_103613-mojolearn-e2-nv` | `2026-08-23_104900-mojolearn-e2-amd` |
| round 2 `53d56ef` | `2026-08-23_065934-MacBook-Air-1-terrabyte` | `2026-08-23_110204-mojolearn-e2-nv` | `2026-08-23_111450-mojolearn-e2-amd` |

Reproduce the tables:

    python3 tools/e2_matrix_diff.py bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte:APPLE \
        bench/results/e1/2026-08-23_110204-mojolearn-e2-nv:NVIDIA \
        bench/results/e1/2026-08-23_111450-mojolearn-e2-amd:AMD

## Open

- `check-ieee-arith` still exits non-zero on NVIDIA (it characterizes the
  vendor; the approximate sqrt is a fact about the column, and the gate
  says so). Under IDENTICAL no tree path consumes the device sqrt any more.
- The `_mojolearn_estimators` binding (PCA/Jacobi, the linear-algebra lane)
  failed to BUILD on AMD in both rounds (`jacobi_eigh_device.mojo:95`, a
  `block_size=32` instantiation on the 64-wide wavefront); the tree
  bindings built and ran. Reported to that lane.
- Not in the matrix: `nan_mode='Forbidden'` (raises), `MultiClassOneVsAll`
  from Python (Mojo card only), `bagging_temperature != 1` (the `pow` path
  is routed and gated but no cell exercises it), ordered boosting / fold
  tasks (no training path reaches them, per the Mojo-cards audit).
