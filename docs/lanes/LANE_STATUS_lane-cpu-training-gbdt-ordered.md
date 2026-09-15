# Lane status: lane/cpu-training-gbdt-ordered (2026-09-15)

CPU (host) training for gbdt-ordered-rmse, gbdt-feature-freq,
gbdt-pointwise-l2-bayesian-eval and gbdt-categorical-ctr, bitwise IDENTICAL to
the 166-lane record's GPU columns.

## Done (all on this branch, pushed)

- `gbdt/host/gbdt_oracle_feature_freq.mojo`: gbdt_fit_two_level_feature_freq on
  the host (tensor column, two synchronized levels over the symmetric oracle's
  histograms). Export `gbdt_fit_two_level_feature_freq`; sample_weight refuses.
- `gbdt/host/gbdt_oracle_ordered.mojo`: train_ordered_rmse / fit_ordered_rmse
  (fold arm of the pointwise searcher: 8-bit fixed-point and half-byte float
  histograms, dynamic cosine scorer, ordered Newton leaves). Export
  `gbdt_fit_ordered_rmse`; sample_weight refuses. Its structure search also has
  a `plain_l2` single-task arm.
- `gbdt/host/gbdt_oracle_pointwise.mojo`: gbdt_fit with use_pointwise_searcher
  (L2, Bayesian bootstrap, boost from average, weights, eval set, Iter detector,
  best model). Every other value on that arm refuses by name.
- `gbdt/host/gbdt_oracle_onehot.mojo` + `gbdt_host_fit(one_hot_in=...)`:
  categorical and one-hot columns under SymmetricTree with Logloss. The lane's
  categorical column has 2 categories, so no CTR is built on any fixture (the
  Apple model texts carry no ctr record); a column above one_hot_max_size (2)
  refuses by name as a CTR column.
- host_surface.py declares all four lanes; workflow triggers on the oracles;
  tests in `python/mojolearn/tests/test_cpu_training_gbdt_ordered.py`.
- tools/identity_break.py: the gbdt-categorical-ctr docstring corrected (it
  claimed a CTR feature with two permutations).

## Evidence

- Local (M4, one core): all four lanes IDENTICAL x4 on 36 train, 72 infer and
  model, 36 batch cells; the MOJOLEARN_HOST_SABOTAGE build DIVERGENT on all.
  After merging main (gbdt-losses), all 16 gbdt lanes IDENTICAL x4 (144 train,
  270 infer/model, 144 batch). JSONs, diffs, Apple model texts and gate logs:
  `~/mojolearn-evidence/cpu-training-gbdt-ordered/`.
- Gate 34936445589 at 302c90a92: green on all seven runners, 252 IDENTICAL x4
  cells per lane (7 x 36), sabotage caught.
- Gate 34959448460 at merge dc4db18ec: cancelled by the 60-minute job timeout
  (exit 143 in the sabotage step; main's gate 34956243867 at 2cf5562b1 died the
  same way). Runners that finished read every cell IDENTICAL x4.
- Commit 3b3ab79ad raises the gate job timeout to 120 minutes. Gate
  34972329290 at 3b3ab79ad was queued when this file was written.

## Running

No rented box. Only CI: CPU identity gate 34972329290 (head 3b3ab79ad).

## Next commands

    cd <worktree of lane/cpu-training-gbdt-ordered>
    gh run view 34972329290 --json status,conclusion,jobs
    # if green on all seven runners:
    git fetch origin && git merge origin/main     # resolve lists/spans only
    python3 tools/docs_facts.py --write && python3 tools/docs_facts.py --check
    python3 packaging/wheel_ci.py pins . && python3 packaging/wheel_ci.py inventory python/mojolearn
    (cd python && python -m pytest -q mojolearn/tests/test_host_surface.py)
    git push origin HEAD:lane/cpu-training-gbdt-ordered && git push origin HEAD:main
    # if the merge resolved code (bindings/_mojolearn_gbdt_host.mojo), gate it first.

Local rebuild and check, one core:

    MOJOLEARN_HOST_OUTDIR=<dir> MOJOLEARN_BUILD_JOBS=1 sh bindings/build_gbdt_host.sh
    MOJOLEARN_HOST_OUTDIR=<dir> MOJOLEARN_BUILD_JOBS=1 sh bindings/build_core_host.sh
    MOJOLEARN_HOST_DIR=<dir> PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical \
      python3 tools/identity_break.py --lanes gbdt-ordered-rmse,gbdt-feature-freq,gbdt-pointwise-l2-bayesian-eval,gbdt-categorical-ctr --json cpu.json
    python3 tools/identity_break.py --diff $(python3 python/mojolearn/host_surface.py --training-gpu-columns) cpu.json --require-columns 4 --lanes <same>
