# Tree lane integration into main — 2026-09-10

Merge parents: main `9bf5115aa98a0d07b93c0177c4ddee3c5aef8556` and tree lane
`cadb551b` (including the previously completed GPU pipeline/tree slices).
The isolated integration checkout preserves the user's other worktrees.

Only a metrics docstring conflicted: keep main's corrected explicit FAST
selection wording together with the lane's historical qualification boundary.
Both branches' exports and pixi tasks were retained, including
ByteLanguageModelConfig and the new GBDT adapters/scalers. Merged TOML/Python
syntax checks passed. Source whitespace checks exclude preserved raw compiler
logs, whose diagnostic snippets contain upstream trailing whitespace.

Local platform: Apple M4, Mojo 1.0.0 ed45d567, Python 3.13.15, NumPy 2.5.2,
sklearn 1.8.0. All three GBDT extensions rebuilt from the integrated final
lazy-capacity source. Broad build gates were skipped; focused checks below
were run under the shared lock. Other native artifacts were staged from the
previously checked lane; this is not a new full-wheel qualification.

- `python_tests.log`: 114 passed (feature fraction, child Hessian, minimum gain,
  ET GPU-only refusal and protocol/ABI coverage).
- `feature_fraction.smoke.log`: 147 small fit/check cells pass across modes,
  growth policies, RMSE/Logloss, fractions, repeats, adapters and weighted eval.
- All 36 pre-feature default model/prediction fingerprints remain unchanged:
  18 public capture comparisons and 18 in `default_after.json`.
- All 18 sampled mixed-layout models/predictions match the initial allocation
  implementation (`sampled_after.json` versus `sampled_before_reuse.json`).
- `ab_harness_smoke.*`: old and new native extensions load together and produce
  matching complete models/predictions in FAST/IDENTICAL. One round on 256 rows
  only verifies the harness; stability is null and timings establish no gain.
- `pipeline_cv.smoke.log`: six tiny GPU StandardScaler → GBDT fold fits pass
  across all modes with training-fold ownership.
- `et_refusal.smoke.log`: integrated public surface still rejects CPU/invalid
  binding selectors before data-pointer access in all modes. Native wrapper
  and unchanged GPU forest checks are retained in the preceding ET evidence.

The public sampling smoke emitted eight CoreAnalytics context diagnostics;
assertions and exit succeeded. Origin/resource impact and long-run stability
remain unresolved. Native projection/arena reuse checks were separately run
on the lane before integration. No dedicated GPU speed result is asserted.

Main's AMD RDNA auto-detection differs from the earlier lane base. CDNA/M4
results do not qualify RDNA. Dedicated AMD/NVIDIA testing awaits an idle endpoint
or a rental spending limit; inventory found only the protected training pod and
no DigitalOcean droplets. No remote training environment was modified.

Next measurement: `docs/lanes/TREE_GPU_MEASUREMENT_NEXT.md`. The reference
must preserve integrated hardware detection and other common dependencies,
changing only the three allocation-reuse files to the initial implementation.
