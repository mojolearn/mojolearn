# lane/sabotage-evidence

Branch `lane/sabotage-evidence`, from main at bfb8f725a, merged with
`origin/main` at 8cc2002b0. Worktree
`/Users/andrewhendel/CascadeProjects/mojolearn-wt/sabotage-evidence`.

CLOSED 2026-09-16. It set out to commit sabotage columns for 51 lanes on
rented CPU. That campaign was CANCELLED mid-run, deliberately, and the pod was
deleted. What survives is three code fixes and one finding, which were the
cheap part and the valuable part. No evidence from the leg is committed.

## The finding

**The CPU gate manufactures exactly the evidence we are missing, on every run,
and throws it away.**

`.github/workflows/cpu-identity-gate.yml` builds the whole host set a second
time with the manifest's per-family sabotage defines (step "Build the sabotage
host set (MOJOLEARN_HOST_SABOTAGE)", line 630), runs every covered lane under
it into `$GATE_OUT/cpu-sab-<slot>.json` (line 682), and requires the
four-column diff to exit non-zero or the job fails. That output then goes to
`actions/upload-artifact` as `cpu-identity-gate-<slot>` (line 743) and nowhere
else. No step commits it, and no column from that workflow exists anywhere
under `bench/results/`: the only three `cpu-sab*` files in the tree come from
manual lane runs.

So the negative control for those lanes IS watched failing, on every gate run,
and the proof expires with the artifact. That is a fact about our RECORDS, not
about our arms.

## What is committed

1. `1334beeed` **The GEMM leaf sabotage arm could not fail on `ties`.**
   `gemm/host/gemm_oracle.mojo` walked each accumulation leaf DESCENDING.
   Reversing a sum whose values add exactly cannot change it, and `ties` is
   integer valued (`rng.integers(0, 6)`), so `gemm-pinned` and
   `gemm-transposed` read UNMOVED there under a build that was supposed to be
   wrong. That site is the ONLY arm `linalg`, `mamba` and `transformer` reach
   and one of two for `training` and `neural`, so on `ties` those five
   families had no working negative control at all. Replaced with the value
   flip `lane/ties-sabotage` already applied to the neighbor and IVF families.
   The old arm is kept behind `MOJOLEARN_GEMM_ORACLE_SABOTAGE_LEGACY_ORDER`,
   set by no build script and no gate, so the defect can be WATCHED failing
   rather than believed.
2. `1334beeed` **A CTR sabotage column could not witness its own arm.**
   `forest_host_sabotage()` returned `FOREST_HOST_SABOTAGE` alone, so a
   binding built with `MOJOLEARN_GBDT_CTR_HOST_SABOTAGE` recorded
   `sabotage: false` and `_backend`'s `MOJOLEARN_HOST_ALLOW_SABOTAGE` guard
   never fired for it. It now reports either arm, which is what
   `_forest_host.py` already ORs when deciding to refuse a load.
3. `323c8980d` **`admit()` read the directory name and silently discarded
   clean columns.** It matched every excluded token against the WHOLE path, so
   a clean column sitting beside the negative control it belongs with was
   refused for its neighbor's name, with no error to read. Fixed NARROWLY:
   `sabotage` now matches the file name alone, while `partial`, `probe`,
   `unfixed` and `post-merge-smoke` keep whole-path matching because each
   marks a whole directory of such runs, and
   `2026-09-14_kmeans-sqrt-fix/unfixed/` holds columns taken with the bug
   still present.

Neither Mojo fix touches a production path: both live under `comptime if` arms
a production build does not compile, or report a flag that is false in one.

### The blast radius, measured before the rule changed

| over the 495 committed columns | |
|---|---|
| admitted before | 220 |
| admitted after | 222 |
| **newly refused** | **0** |
| newly admitted | 2, both provably clean |
| sabotage-signal columns admitted | 0 |

Recovered: `2026-09-15_metrics-sabotage-coverage/cpu-prod.json` and
`2026-09-15_ties-sabotage/x86-runpod/cpu-x86.json`. The obvious "just match
the basename" fix would have admitted eight `unfixed/` and `probe/` columns,
which is why this was measured first.

`python/mojolearn/tests/test_verify_reference_admit.py` was WATCHED FAILING
before the fix existed: 2 failed and 6 passed against the unfixed rule, the
two failures being the trap and the six passes being the blast-radius guard.
Against the fix, 8 passed.

Type-check of the Mojo, one core: five builds, all exit 0, and the three
linalg binaries carry three DISTINCT sha256 values (production, value arm,
legacy order arm), which is what proves the legacy define selects a different
binary rather than silently doing nothing.

## The cancelled campaign, and the pod

One RunPod CPU pod, `h1z3i1ron6s0t0`
(`mojolearn-cpu-sabevid-20260916-095653`), 16 vCPU, created 09:56:53Z, DELETED
10:21:11Z, delete VERIFIED (HTTP 204, then GET 404, and absent from the
listing). Elapsed 24 m 18 s at $0.48/hr. **Spend $0.194.** Nothing it produced
is committed and its working directory was discarded.

It was cancelled before it finished, on purpose. The never-cancel-an-owed-run
rule exists so a cancel does not hand the fetch, build and decode cost to the
NEXT attempt. There is no next attempt, so finishing would have been the
waste. Part of the output was also already invalid: four of the GBDT lanes in
its list (`gbdt-parametric-losses`, `gbdt-nan-modes`,
`gbdt-lossguide-newtoncosine`, `gbdt-pair-logit`) are being shrunk, and the
harness refuses to diff across differing fixture sizes, so those columns could
never have been compared to anything later.

That last point is worth keeping. An earlier scope check against
`docs/lanes/FIXTURE_SHRINK_SCOPE.md` found bucket A to be just `hdbscan` and
`hdbscan-leaf`, neither of them in this lane's 51. The scope later grew to ten
lanes. **A fixture-scope check is only true at the moment it is read**, and
evidence recorded against a lane whose fixture is in flight is stale by
construction.

## The honest end state

51 single-device lanes have a sabotage arm that EXISTS and has NOT been
observed to fire in committed evidence. The CI gate proves them on every run
and discards the proof. Nobody is chasing them.

That is a fact about our records, not a debt, and it should not be read as a
to-do list. The CPU path exists so we CAN verify everything on occasion, not
so that complete committed evidence is held for every lane at all times. If
these lanes ever matter, they ride a broad CPU sweep somebody runs
deliberately, not a campaign.

The lanes, for reference: arima-exog, arima-exog-seasonal, bootstrap,
byte-lm-host-infer, byte-lm-host-infer-threaded, byte-lm-host-train,
cross-entropy-arms, cross-val, et-clf, et-clf-entropy-bestfirst, et-reg,
et-reg-bootstrap-parallel, gbdt-adapter-clf, gbdt-adapter-reg,
gbdt-adapter-score-weighted, gbdt-depthwise, gbdt-exact-mae, gbdt-lossguide,
gbdt-lossguide-newtoncosine, gbdt-multiclass, gbdt-nan-modes, gbdt-onevsall,
gbdt-pair-logit, gbdt-parametric-losses, gbdt-query-rmse, gbdt-rmse,
gbdt-symmetric, gbdt-yeti-rank, gemm-pinned, gemm-transposed, kde, knn,
knn-clf, knn-reg, logistic, logistic-multiclass, monte-carlo, optim-adam-clip,
optim-sgd, pca, pca-whiten, permutation-test, rf-clf, rf-clf-balanced-parallel,
rf-clf-entropy-log2-noboot, rf-reg, rf-reg-gamma-ig, rf-reg-poisson,
rf-score-weighted, training-primitives, tsvd.

## Counting, for whoever quotes these numbers

All figures here are in REGISTRY terms, **199 lanes**. The 176 that
`docs/lanes/SABOTAGE_AUDIT_2026-09-16.md` asked its question of is the count
of `@lane(` occurrences and describes nothing: 23 lanes are loop-registered
and invisible to a decorator count. 166, 136, 47 and 192 appear in various
records and are each legitimate as "the lanes that run covered", but none is a
total.

The baseline was RE-TAKEN on the merged tree rather than quoted from
`docs/VERIFICATION_MATRIX.md`, which was generated at bfb8f725a before main
moved:

| verdict | published (bfb8f725a) | merged tree (8cc2002b0) |
|---|---|---|
| seen(build) | 110 | **112** |
| seen(harness) | 7 | 7 |
| declared | 57 | 57 |
| none | 25 | **23** |

`par-samba` and `par-samba-clip` moved from `none` to `seen(build)` through
`lane/cpu-verifier-par-samba`, on an Apple CPU column and with no two-device
box. That is somebody else's evidence, and quoting the published 110 would
have credited it to this lane. Worth noting for whoever owns the `none`
bucket: it is 23, not 25, and it is evidently less closed than it looks.

The 23 loop-registered lanes were CHECKED, since "nobody looked" is not the
same as "uncovered". All 23 are already `seen(build)`: five kde kernel and
metric variants, six knn and three radius metric variants, three gp kernels,
`gp-sample-y` and its normalized twin, `gp-optimize` and its restarts twin,
and two `gmm-sample` lanes. Nothing is owed for them, and the set of decorator
names not in the registry is EMPTY. The instinct behind the concern was sound:
four of those lanes (`knn-cosine`, `knn-rbc`, `radius`, `radius-manhattan`)
are exactly where the old order-only arm was inert on `ties`, which
`lane/ties-sabotage` found and fixed.

## Owed, and NOT done here

- **25 lanes still claim DIVERGENT on a CI run nobody can open.** In
  `python/mojolearn/host_surface.py`, the gbdt, rf and trees family comments
  cite gate runs 34884487749, 34900811380 and others. No committed column in
  this tree carries them. They were to be rewritten against this lane's
  evidence; that evidence does not exist, so THE COMMENTS ARE UNCHANGED. They
  should eventually be rewritten to say the arm is declared and has not been
  observed to fire, rather than citing a log that cannot be followed.
- **`docs/lanes/SABOTAGE_AUDIT_2026-09-16.md` needs two corrections** and this
  branch cannot make them: it lives on `lane/sabotage-audit`, which is
  unmerged and whose file is not on main.
  1. Its denominator is 176 and should be 199, with a note about the 23
     loop-registered lanes.
  2. It says `par-forest` and `par-forest-et` are closable without a box AND
     that `par-*` lanes refuse on a CPU column. Both cannot be true. MEASURED:
     both ARE in `host_surface.covered_lanes()`, and NO committed CPU column
     carries either. Settling it needs a CPU column that actually reaches them
     or a two-device box.
- The `model` cells of the k-NN, radius and KDE lanes hold only the caller's
  own input, so no arm that exists can move them (audit finding 2). The named
  fix is `model_na="n/a:input-index"` at the lane, not an exemption in the
  owed checker.

## Resume, if a broad sweep is ever run deliberately

The leg command is in this branch's history and needs only a lane list. Check
`docs/lanes/FIXTURE_SHRINK_SCOPE.md` AT THAT MOMENT, not from any earlier
reading, and hold every lane in its "will change" and "undecided" buckets.

```sh
bash tools/runpod_cpu_leg.sh list          # before and after, always
bash tools/runpod_cpu_leg.sh reap POD_ID   # deletes and verifies
```

## Gates

`docs_facts --check`, `packaging/wheel_ci.py pins .` and
`packaging/wheel_ci.py inventory python/mojolearn` all pass on the merged
tree (13 facts and 12 marked spans, 56 build scripts, 85 modules), as does
`python/mojolearn/tests/test_verify_reference_admit.py` (8 passed).
