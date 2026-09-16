# lane/sabotage-evidence

Branch `lane/sabotage-evidence`, from main at bfb8f725a. Worktree
`/Users/andrewhendel/CascadeProjects/mojolearn-wt/sabotage-evidence`.

THE GAP. `docs/VERIFICATION_MATRIX.md` (branch `lane/verification-matrix`,
be88fa4e0) counts 199 lanes and only 110 whose negative control has been SEEN
to fail in committed evidence: 110 `seen(build)`, 7 `seen(harness)`, 57
`declared`, 25 `none`. The 89 uncovered lanes are a gap in EVIDENCE, not in
arms. Every uncovered single-device lane already has a sabotage define.

## What this lane does

1. Commits the CPU gate's sabotage column BESIDE its clean one, which the gate
   already builds on every run and then throws away (verified, see below), for
   the 51 single-device lanes whose control had never been watched to fail.
2. Fixes two defects the sabotage audit found
   (`docs/lanes/SABOTAGE_AUDIT_2026-09-16.md`, branch `lane/sabotage-audit`):
   the inert GEMM order arm, and the forest binding's sabotage read-back.

## Verified before building on it

The census's central claim holds. `.github/workflows/cpu-identity-gate.yml`
builds the whole host set again with the manifest's per-family defines (step
"Build the sabotage host set (MOJOLEARN_HOST_SABOTAGE)", line 630), runs every
covered lane under it into `$GATE_OUT/cpu-sab-<slot>.json` (line 682), and
requires the four-column diff to exit non-zero. That output goes ONLY to
`actions/upload-artifact` as `cpu-identity-gate-<slot>` (line 743). No step
commits it, and no column from that workflow exists under `bench/results/`:
the only three `cpu-sab*` files in the tree come from manual lane runs
(`2026-09-15_metrics-sabotage-coverage`, `2026-09-15_cpu-verifier-gaps-7`).
The gate therefore produces exactly the evidence the matrix is missing, on
every run, and discards it when the artifact expires.

## The baseline this lane's delta is measured against

NOT the published 110. `docs/VERIFICATION_MATRIX.md` was generated at
bfb8f725a, and main moved while this lane worked. `origin/main` at 8cc2002b0
is merged into this branch, and the baseline was RE-TAKEN on that merged tree
before any evidence from this lane landed:

| verdict | published (bfb8f725a) | baseline on the merged tree |
|---|---|---|
| seen(build) | 110 | **112** |
| seen(harness) | 7 | 7 |
| declared | 57 | 57 |
| none | 25 | **23** |

The two lanes that moved are `par-samba` and `par-samba-clip`, from `none` to
`seen(build)`, closed by `lane/cpu-verifier-par-samba` through
`bench/results/identity_break/2026-09-16_cpu-par-samba/cpu-apple-m4.sabotage.json`.
That is somebody else's evidence. Diffing against the published 110 would have
credited this lane with two lanes it did not touch, which is why the baseline
is re-taken rather than quoted.

Worth noting for whoever owns the `none` bucket: it is 23, not 25, and those
two were closed WITHOUT a two-device box, on an Apple CPU column. The brief
here says not to attempt the `par-*` lanes, and that still stands for this
lane, but the bucket is evidently not as closed as it looks.

All numbers in this file are in REGISTRY terms (199 lanes), never the
grep-visible 176.

## The 51 lanes

`declared` or `seen(harness)`, single-device, and CPU-covered per
`python/mojolearn/host_surface.py --covered-lanes`. All 51 are in
`--record-covered-lanes`, so each is diffed against the three GPU columns of
`2026-09-14_166-lanes` rather than resting on the sabotage arm alone.

arima-exog, arima-exog-seasonal, bootstrap, byte-lm-host-infer,
byte-lm-host-infer-threaded, byte-lm-host-train, cross-entropy-arms, cross-val,
et-clf, et-clf-entropy-bestfirst, et-reg, et-reg-bootstrap-parallel,
gbdt-adapter-clf, gbdt-adapter-reg, gbdt-adapter-score-weighted, gbdt-depthwise,
gbdt-exact-mae, gbdt-lossguide, gbdt-lossguide-newtoncosine, gbdt-multiclass,
gbdt-nan-modes, gbdt-onevsall, gbdt-pair-logit, gbdt-parametric-losses,
gbdt-query-rmse, gbdt-rmse, gbdt-symmetric, gbdt-yeti-rank, gemm-pinned,
gemm-transposed, kde, knn, knn-clf, knn-reg, logistic, logistic-multiclass,
monte-carlo, optim-adam-clip, optim-sgd, pca, pca-whiten, permutation-test,
rf-clf, rf-clf-balanced-parallel, rf-clf-entropy-log2-noboot, rf-reg,
rf-reg-gamma-ig, rf-reg-poisson, rf-score-weighted, training-primitives, tsvd

The 25 `none` lanes are all `par-*` multi-GPU drivers and are out of scope by
the brief. The 9 `declared` and 4 `seen(harness)` `par-*` lanes are equally out
of scope: a par lane needs two devices, which no CPU pod has.

## HELD pending the fixture shrink

Another agent is shrinking oversized identity fixtures and publishes its scope
to `docs/lanes/FIXTURE_SHRINK_SCOPE.md`. A sabotage proven live at one fixture
size can be inert at another, which is exactly finding 1 below. Any lane in
that file's "will change" or "still undecided" buckets is held out of the
committed column rather than recorded against a fixture that is about to stop
existing.

RESOLVED, nothing held. `docs/lanes/FIXTURE_SHRINK_SCOPE.md` landed on
`origin/lane/identity-fixtures-light`. Its bucket A ("will change") is at most
`hdbscan` and `hdbscan-leaf`, bucket C ("undecided") is EMPTY, and the file
states that at that commit no lane body has been edited and every fixture is
still at its 0.8.6 size. Neither bucket A lane is among the 51, so no lane is
held and everything recorded here is against a current fixture.

The one thing to carry forward: bucket A is still conditional on a reach
sweep. If `hdbscan` or `hdbscan-leaf` does shrink, its sabotage must be
re-proven AT THE NEW SIZE, not inherited. Neither is in this lane's scope.

## The audit asked its question of 176 lanes; the registry holds 199

`grep -c '@lane('` returns exactly 176, and the harness registry holds 199.
The 23 difference are LOOP-REGISTERED lanes that no grep can see, so the
audit's opening line, "asked of all 176 lanes", left them outside the question
as asked. 176 is purely an artifact of how the file was counted and describes
nothing. (166, 136, 47 and 192, which appear in various records, are each
legitimate as "the lanes that run covered", but none of them is a total
either.)

`tools/verification_matrix.py` needs no fix here. It loads the harness as a
module rather than grepping it and already reports 199, which is why every
number in this lane status is in registry terms.

CHECKED, since "nobody looked" is not the same as "uncovered". All 23 are
already `seen(build)`:

| group | lanes |
|---|---|
| kde kernel and metric variants (5) | `kde-cosine-minkowski`, `kde-epanechnikov-l1`, `kde-exponential-chebyshev`, `kde-linear-cosine`, `kde-tophat-sqeuclidean` |
| knn metric variants (6) | `knn-chebyshev`, `knn-cosine`, `knn-manhattan`, `knn-minkowski-p3`, `knn-rbc`, `knn-sqeuclidean` |
| radius metric variants (3) | `radius-chebyshev`, `radius-manhattan`, `radius-minkowski-p3` |
| gp kernels (3) | `gp-matern12`, `gp-matern32`, `gp-matern52-ard` |
| gp sampling (2) | `gp-sample-y`, `gp-sample-y-normalize` |
| gp optimizer (2) | `gp-optimize`, `gp-optimize-restarts` |
| gmm sampling (2) | `gmm-random-init-sample`, `gmm-sample` |

None is in this lane's 51, for the good reason that this lane targets
`declared` and `seen(harness)` lanes and these were already `seen(build)`.
Nothing is owed for them, and the set `decorator names - registry names` is
EMPTY, so there is no stale or renamed lane hiding either.

The concern that raised them is still the right instinct, and it has already
been paid once. These metric and kernel variants are exactly where an inert
arm hides, and four of them (`knn-cosine`, `knn-rbc`, `radius`,
`radius-manhattan`) are the lanes whose `ties` cells the OLD order-only arm
could not move. That was found and fixed by `lane/ties-sabotage`, with the
value flip this lane now applies to the GEMM leaf as well.

OWED, and not fixable from this branch: the audit document itself still says
176 and should say 199, with a note that 23 lanes are loop-registered and
invisible to a decorator count. `docs/lanes/SABOTAGE_AUDIT_2026-09-16.md`
lives on `lane/sabotage-audit`, which is not merged and whose file is not on
main, so this branch cannot correct it. The same applies to the
`par-forest` / `par-forest-et` contradiction above. Both are handed back to
that lane's owner.

## The two code fixes

1. `gemm/host/gemm_oracle.mojo`. The arm walked each leaf DESCENDING. An order
   perturbation cannot move a sum whose values add exactly, and `ties` is
   integer valued (`rng.integers(0, 6)`), so `gemm-pinned` and
   `gemm-transposed` read UNMOVED there under a build meant to be wrong. That
   site is the ONLY arm `linalg`, `mamba` and `transformer` reach and one of
   two for `training` and `neural`. Replaced with the value flip
   `lane/ties-sabotage` already used for the neighbor and IVF families
   (`gemm_oracle_sabotage_value_flip`). The OLD arm is kept behind
   `-D MOJOLEARN_GEMM_ORACLE_SABOTAGE_LEGACY_ORDER=1`, set by no build script
   and no gate, so the defect can be WATCHED failing rather than believed.
2. `bindings/_mojolearn_forest_host.mojo`. `forest_host_sabotage()` returned
   `FOREST_HOST_SABOTAGE` only, so a binding built with the CTR define
   recorded `sabotage: false` and a column could not witness its own arm. Now
   returns `FOREST_HOST_SABOTAGE or GBDT_CTR_HOST_SABOTAGE`, which is what
   `python/mojolearn/_forest_host.py` already ORs when it decides to refuse.

Neither touches a production code path: both live under `comptime if` arms a
production build does not compile, or report a flag that is false in one.

## Resume commands

Type-check the two edits with one core, without installing an env in the
worktree (it has no `.pixi`) and without building into the shared checkout:

```sh
bash /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/\
e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/mojocheck.sh
```

The leg, as a DRY RUN first (creates nothing, bills nothing). Drop `--rent` to
re-check, add it to create the pod:

```sh
cd /Users/andrewhendel/CascadeProjects/mojolearn
FAM=$(python3 python/mojolearn/host_surface.py --families | tr ' ' ',')
bash tools/runpod_cpu_leg.sh --lane sabevid \
  --worktree /Users/andrewhendel/CascadeProjects/mojolearn-wt/sabotage-evidence \
  --cmd-file <scratchpad>/leg_cmd.sh \
  --build "$FAM" --sabotage-build "$FAM" \
  --sabotage-defines "-D MOJOLEARN_HOST_SABOTAGE=1 -D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1 -D MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1 -D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1" \
  --include bench/results/identity_break/2026-09-14_166-lanes \
  --vcpu 16 --lease 150 --jobs 8 --rent
```

Pod hygiene: `bash tools/runpod_cpu_leg.sh list` before and after, and
`bash tools/runpod_cpu_leg.sh reap POD_ID` if a pod outlives its leg. One pod
at a time, verified deleted, no GPU rental of any kind.

Re-run the census after the evidence lands (the tool belongs to
`lane/verification-matrix`; run it, do not edit it):

```sh
python3 tools/verification_matrix.py --json > after.json
```

## Gates

`docs_facts --check`, `packaging/wheel_ci.py pins .` and
`packaging/wheel_ci.py inventory python/mojolearn` all pass on this worktree
with both code edits applied (13 facts and 12 marked spans, 56 build scripts,
85 modules).

## The 25 claims believed on a CI log

The sabotage audit found 25 lanes whose DIVERGENT claim lives only in a
`python/mojolearn/host_surface.py` comment citing a CI gate run (34884487749,
34900811380 and others). No committed column in this tree carries them, so as
written the claim cites a log nobody can open. Andrew's instruction is prove it
or delete it.

Derived from the audit's own `lane_table.md` rather than by eye: the 25 are
exactly the category (b) lanes in the gbdt (14), rf (8) and trees (3)
families.

- 23 of the 25 are in this lane's 51 and are settled by the committed column
  here, each comment rewritten to cite the record path and what moved, or to
  say plainly that the arm does not move the lane.
- 2 are NOT settleable here: `par-forest` and `par-forest-et`. MEASURED, not
  assumed: both ARE in `host_surface.covered_lanes()`, and NO committed CPU
  column anywhere in the tree carries a cell for either. The audit lists them
  among ten `par-*` lanes it calls closable without a box, while also stating
  that a `par-*` lane REFUSES on a CPU column because it asks for devices.
  Both cannot be true, and nothing in this tree settles which is. Their
  comments are therefore rewritten to the honest form, that the arm is
  declared and has never been observed to fire, with the CPU reachability
  recorded as the open question. Settling it needs either a CPU column that
  actually reaches them or a two-device box. Owed, and named here so it is
  picked up rather than forgotten.

The comment sites to rewrite are `host_surface.py` lines 288 (rf-clf, rf-reg),
437 to 449 (the four GBDT lanes, including the two "as above" back references)
and the batch comments around 460. Rewrite them rather than appending a
correction: a correction that quotes the sentence it supersedes leaves the old
claim greppable forever.

## A wart worth knowing: "sabotage" in a PATH refuses a CLEAN column

`python/mojolearn/_verify_reference.admit` rejects a column when any of
`_EXCLUDED_NAME_TOKENS` ("sabotage", "partial", "probe", "unfixed",
"post-merge-smoke") appears anywhere in the lowercased PATH, not just the file
name. So a clean column committed under a directory named `..._sabotage-...`
is refused admission for its neighbor's sin.

FIXED, not worked around. `sabotage` now matches the FILE NAME alone, while
`partial`, `probe`, `unfixed` and `post-merge-smoke` keep matching the whole
path, because each of those marks a whole DIRECTORY of such runs and
`2026-09-14_kmeans-sqrt-fix/unfixed/` holds columns taken with the bug still
present. The obvious "just match the basename" fix would have admitted eight
of those `unfixed/` and `probe/` columns, which is exactly why the blast
radius was measured BEFORE the rule changed.

Blast radius over the 495 committed columns, old rule against new:

| | columns |
|---|---|
| admitted before | 220 |
| admitted after | 222 |
| NEWLY REFUSED | 0 |
| newly admitted | 2, both provably clean |
| sabotage-signal columns admitted | 0 |

The two recovered are
`2026-09-15_metrics-sabotage-coverage/cpu-prod.json` and
`2026-09-15_ties-sabotage/x86-runpod/cpu-x86.json`.

`python/mojolearn/tests/test_verify_reference_admit.py` was WATCHED FAILING
first, which is the whole point. Against the unfixed rule it reads 2 failed
and 6 passed, the two failures being the trap and the six passes being the
blast-radius guard. Against the fix it reads 8 passed.

Because the rule is fixed, this lane's evidence no longer needs a euphemism
for a directory name. It is still committed under
`bench/results/identity_break/2026-09-16_negative-controls/`, which reads
better than naming a directory after the thing it is a control for, and the
sabotage column is refused by its `host.families[*].sabotage` read-back rather
than by its name, which is the stronger of the two paths and the one the
forest read-back fix repairs.

This also un-breaks the audit's own
`bench/results/identity_break/2026-09-16_sabotage-audit/` directory: its clean
`cpu-prod.json` becomes admissible where it sits, with no rename needed on
`lane/sabotage-audit`.

## State

The leg is composed, dry-run clean, and RUNNING. Pod `h1z3i1ron6s0t0`
(`mojolearn-cpu-sabevid-20260916-095653`), 16 vCPU, billed at $0.48/hr rather
than the $0.24 in the docs BECAUSE of `--vcpu 16`; an 8 vCPU pod is the $0.24
rate. It ships commit 1334beeed with `dirty_tracked=0`, so the two code fixes
are in the build the evidence is taken against.

Both code fixes are committed (1334beeed) and type-check on one core: five
builds, all exit 0, and the three linalg binaries carry three distinct sha256
values (production, value arm, legacy order arm), which is what proves the
legacy define actually selects a different binary rather than silently doing
nothing.

Still owed when the leg lands: commit the column pair, rewrite the 23 claims,
re-run the census and record the before/after, and report any lane whose
sabotage did NOT move it.
