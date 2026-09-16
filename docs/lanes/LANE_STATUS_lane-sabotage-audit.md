# lane/sabotage-audit status

Goal. Answer, for every one of the 199 `tools/identity_break.py` lanes,
whether its sabotage has been SEEN to make the lane divergent, close the cheap
gaps on the CPU host route, and write down precisely which cells no sabotage
can move and why.

## Done

- `docs/lanes/SABOTAGE_AUDIT_2026-09-16.md`, the audit and the per-lane table.
  126 lanes seen to move, 43 with an arm never seen to move, 30 with no
  sabotage that reaches them at all. The denominator is `len(LANES)` with the
  harness loaded as a module: 176 lanes carry a literal `@lane` decorator and
  23 more are registered in loops. Twenty-one of those 23 were already (a);
  `gp-optimize` and `gp-optimize-restarts` are in no host family and are (c).
- `bench/results/identity_break/2026-09-16_sabotage-audit/`, sixteen lanes
  closed on this Mac's CPU route, one core, no box rented, no Metal job.
  Round 1 (estimators, core): `pca`, `pca-whiten`, `tsvd`, `ols`, `ridge`,
  `logistic`, `logistic-multiclass`, `kde`, `knn`, `knn-clf`, `knn-reg`, 79
  parts moved and 9 not. Round 2 (linalg, resample): `gemm-pinned`,
  `gemm-transposed`, `bootstrap`, `permutation-test`, `monte-carlo`, 14 parts
  moved and 4 not.
- Static check that every one of the 32 host families' sabotage defines
  reaches at least one `comptime if` arm in its declared `host_modules`.

## Open, with reasons

- THE GEMM LEAF ARM WAS INERT ON `ties`, and is FIXED on branch
  `lane/sabotage-evidence` at cc5234c00, which replaces the order walk with
  `gemm_oracle_sabotage_value_flip` and keeps the old arm behind
  `MOJOLEARN_GEMM_ORACLE_SABOTAGE_LEGACY_ORDER` so the defect stays watchable.
  NOT YET ON MAIN (`origin/main` 8cc2002b0 still carries the order arm), so
  the finding stands against main until that branch merges.
- The CTR read-back is FIXED on the same branch; a CTR-define build now
  records `sabotage: true`. Also not yet on main.
- `par-forest` and `par-forest-et` are OWED, not cheap. Both are in
  `covered_lanes()` and a host arm reaches their families, but no committed
  CPU column carries either lane's cell, production or sabotage, so nothing in
  the tree says whether the CPU route runs them or refuses them.
- The input-copy `model` cells. `kde`, `knn`, `knn-clf`, `knn-reg` measured
  here, `radius` and `radius-manhattan` already failing the owed check 0 of
  18. The saved file holds the fitted index and scalars, no arithmetic, so no
  host arm reaches it. Proposed fix is `model_na="n/a:input-index"` on those
  lanes, the mechanism the Embedding lane already uses at
  `tools/identity_break.py` lines 3116 and 3147. NOT APPLIED here, because it
  changes what the gate demands and wants Andrew's decision.
- `kmeans-cosine` hashes a refusal sentence and wants the same kind of named
  exemption.
- `forest_host_sabotage()` does not report the CTR arm, so a
  `MOJOLEARN_GBDT_CTR_HOST_SABOTAGE` column records `sabotage: false` and
  cannot witness its own binary.
- Holt-Winters sabotage is inert on `denormal`, `denormal_ftz` and `wide`
  (6 of 36 cells), and `tsvd` model is inert on `ties`.
- The 28 par-* lanes need a two-device box for any negative control.

## Resume

All commands from a worktree of this branch. One core, no Metal.

    W=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/wt-sabaudit
    cd "$W"

Build a family production and sabotage (about 60 to 90 seconds each on one
core), then run the two columns and compare. Substitute the family and lanes.

    export PATH=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin:$PATH
    export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=1
    env MOJOLEARN_HOST_OUTDIR="$W/host-prod" nice -n 19 sh bindings/build_host_family.sh gbdt
    env MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" \
        MOJOLEARN_HOST_OUTDIR="$W/host-sab" nice -n 19 sh bindings/build_host_family.sh gbdt

    PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/python
    export PYTHONPATH="$W/python" OMP_NUM_THREADS=1
    env MOJOLEARN_HOST_DIR="$W/host-prod" nice -n 19 "$PY" tools/identity_break.py \
        --lanes gbdt-symmetric,gbdt-rmse --fixtures base,ties --repeats 2 \
        --vendor cpu-apple-m4 --json "$W/cpu-prod.json"
    env MOJOLEARN_HOST_DIR="$W/host-sab" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 nice -n 19 "$PY" \
        tools/identity_break.py --lanes gbdt-symmetric,gbdt-rmse --fixtures base,ties \
        --repeats 1 --vendor cpu-apple-m4 --json "$W/cpu-sab.json"

The sabotage column must record `host.families[*].sabotage = true`; if it does
not, the build define did not reach the binding and the column proves nothing.

Gates before pushing to main.

    python3 tools/docs_facts.py --check
    python3 packaging/wheel_ci.py pins .
    python3 packaging/wheel_ci.py inventory python/mojolearn

## Not done on purpose

No lane, algorithm or arm was added or changed. This branch is verification
quality only. No box was rented and the Metal lock was never taken, because
the 0.8.6 Apple record held it throughout.
