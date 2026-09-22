# Verification workstream restart plan

Handoff: 2026-09-17. Baseline main: `ab9a3de26`. This plan is committed after
that baseline. During handoff, origin/main advanced to `2fa81d74f` with the
concurrent low-bit/model/block-options integration. Read
[that handoff](lanes/HANDOFF_2026-09-17_lowbit_inference.md) and reconcile its
new surfaces, engineering rules and outstanding device jobs before resuming.
The counts and test results below describe the verification baseline, not
qualification of those newer native changes. Do not terminate its jobs.

All implementation and evidence from this verification workstream
are merged into main; this document does not certify a release.

## Follow-up: CPU TSA admission, 2026-09-17

The next two queued lanes, `select-d` and additive `holtwinters`, are now
promoted with strict historical references and installed development-wheel
replay. CPU availability is 128, with 51 withheld and 50 parallel exclusions.
The rest of this document's counts describe the original baseline. See
[the batch status](lanes/LANE_STATUS_tsa_reference_promotion.md) for evidence,
packaging fixes, limitations and the remaining queue. Full reference regeneration
and final artifact qualification remain owed.

## Follow-up: CPU completion, 2026-09-17

CPU availability is now **162**, with **17 withheld** and **50 parallel
exclusions**. The installed wheel passed all nine fixtures twice for twelve
new low-bit lanes (396 IDENTICAL, 144 N/A) and 29 repaired lanes (954 IDENTICAL,
351 N/A). No DIVERGENT, OWED or REFUSED results. All 32 host bindings were
freshly built; exact artifact and source receipts are retained in
`bench/results/cpu-verification-completion-probe/2026-09-17/installed-162/`.

The initial complete 150-lane replay found 29 missing-reference failures;
all original failures remain retained. The repaired 249 numerical references
match independently measured wheel bytes, without changing any existing
numerical reference. This is scoped installed CPU verification, not final
release qualification or a PyPI publication.

[The active checkpoint](lanes/LANE_STATUS_cpu_verification_completion.md)
tracks the completed 99-lane native-control audit, resource limits, failures,
remaining five ordinary neural lanes and twelve vendor-held lanes. All 176
numerical non-parallel lanes now have all-nine historical native training
controls, exposed in the installed wheel. The refreshed wheel passes 180
additional numerical checks on the repaired-control lanes; release qualification
remains incomplete. All rentals from this lane are verified deleted. Read it
before resuming or renting another pod.

## Objective and current position

Finish honest, user-runnable verification for every exposed algorithm and
variant in the 0.8.7 candidate, including applicable CPU/GPU routes, native
negative controls, batch/sequence properties and one-versus-multiple GPU checks.
Expose scope, results, input/protocol provenance and reference hashes through
the installed package. The interface is `python -m mojolearn verify`, not a
hosted endpoint. No PyPI publication occurred in this workstream.

Current counts (recompute on restart):

| Inventory | Status |
|---|---|
| Inventory algorithms/variants | 246 mapped entries, not 246 distinct classes |
| Registered harness lanes | 229, including 50 parallel lanes |
| Enumerated source API entries | 222; a separate accounting from the appendix |
| CPU coverage inventory | 126 available, 53 withheld, 50 parallel exclusions |
| Strict historical native controls | 52/246 entries have controls for every mapped lane/part on at least one fixture |
| Historical multi-GPU pairs | 39/50 lanes on NVIDIA; 9/50 on AMD |
| Bundled numerical reference table | Legacy admission policy; regeneration and final-wheel replay owed |
| Final release qualification | Incomplete |

The 52-entry count is neither all-fixture coverage nor current-wheel
certification. Available lanes can still owe individual reference properties.
The bundled table has 708 historical column references supported by one sample
(177 each for train, infer, model and batch). These are column references,
not algorithms. New generation requires two matching samples, input witnesses,
held-out witnesses and matching property protocols. Optional property hashes
remain incomplete. Do not hide missing evidence by declaring properties N/A.

## Operating constraints and authorization

- User authorized fixes in worktrees, commits, pushes and merges to main, plus
  RunPod and Cloudflare R2. Do not ask again for this existing scope.
- At most **three CPU cores total for this work**: ordinarily two cloud vCPUs
  and one local worker. Set OMP, OpenBLAS, MKL, NumExpr and local Accelerate
  thread limits to one; use at most two compiler workers. Do not run two
  two-vCPU jobs concurrently. GPU rentals must respect the CPU limit too.
- Read repository engineering rules. Keep Mac Metal work bounded and serialized;
  do not relaunch the historical multi-hour Apple matrix. Use `tools/mac_slot.py`
  for local work and respect other jobs.
- Use R2-backed runners and `tools/dataset_store.sh` for needed large datasets.
  Identity fixtures are generated deterministically; these tests need no large
  external dataset. Every rental needs a watchdog, retained evidence and
  verified deletion, including failed runs. Never terminate other users' jobs.
- Main contains unrelated untracked results and other worktrees/branches.
  Do not stage or merge them indiscriminately. This handoff covers this
  verification workstream; inspect provenance before any separate integration.

## Restart procedure

1. Fetch origin, check status and source changes since the baseline, and create
   an isolated worktree from current main. Reconcile concurrent native changes
   before deciding which artifact a result qualifies.
2. Read this plan and the three lane status files linked below. Read
   `host_surface.PUBLIC_PENDING_LANES` and `_verification_catalog.py`; the latter
   records the census provenance.
3. Capture installed `verify --coverage --json`, record wheel SHA-256 and binding
   digests, and compare with the source inventory. Run the matrix check. Do not
   assume an old installed 0.8.7 wheel contains current main's commands.
4. Pick a bounded group of gaps with shared bindings. State the required clean,
   sabotage, reference and installed-wheel checks before renting hardware.

## Work queue, in order

1. **Close the next CPU/reference gaps.** KPSS, select-d and both Holt-Winters
   variants now have 36 all-fixture native controls. `select-d` and additive
   `holtwinters` still lack bundled references/default promotion. Evaluate their
   retained clean columns through the strict generator, inspect conflicts and
   changed parts, then replay the resulting candidate table from installed
   wheels. Do not hand-edit hashes or silently replace unrelated references.
   Remaining `unwatched` candidates include `samba` and `gbdt-yeti-rank`;
   re-read the manifest before acting. Add CPU routes only where appropriate;
   existing implementation, manifest declaration and observed verification are
   three different claims.
2. **Complete per-lane controls and applicable properties.** Work through all
   246 mapped entries plus additional exposed lanes. For each applicable part,
   require stable clean output and a genuine changed-result native control at
   matching source/input/protocol. Refusal, N/A, one repeat or unstable output
   is not a successful control. Cover train, inference, save/reload, batch and
   applicable gradient, batch-size, ragged, step/full and sampler/replay checks.
   Record justified inapplicability explicitly. New APIs need manifest,
   harness/catalog mapping, packaged dependencies and user-facing verification.
3. **Close the eleven parallel lanes with no strict historical pair:**
   `par-forest-reg`, `par-forest-et-clf`, `par-boosting-clf`, `par-boosting-reg`,
   `par-gram-ols`, `par-gram-pca`, `par-gram-tsvd`, `par-cd-elasticnet`,
   `par-svm-svr`, `par-scaler-minmax`, `par-queries-nn`.
   Run one and two GPUs using the same wheel and binding digests on NVIDIA and
   AMD; check actual device use as well as requested indices. Then qualify all
   50 existing drivers on the release artifacts. Separately assess unimplemented
   parallel surfaces; do not infer that all 246 entries require multi-GPU paths.
4. **Freeze and qualify final artifacts.** Build final Linux/macOS wheels from
   the intended source, record digests, install outside the checkout without
   host/harness/model overrides, and execute supported CPU/GPU routes and
   properties. Reused Mac development bindings and source-harness cloud runs
   do not qualify these artifacts. Follow bounded Apple release policy.
5. **Regenerate, replay, admit, publish.** Review strict table regeneration for
   missing parts, conflicts and superseded columns; keep legacy evidence labeled.
   Replay the exact final packaged references. Complete repository release
   admission and publication prerequisites. Do not label the release comprehensive
   or publish a wheel as qualified while required evidence is absent.

## Commands and known runner dependencies

```sh
python -m mojolearn verify --coverage --json
python -m mojolearn verify --lanes spectral,metrics-fowlkes-mallows,arima-exog,arima-exog-seasonal --repeats 2
python3 tools/verification_matrix.py --check
python3 tools/verification_evidence.py --write
python3 tools/verification_matrix.py --write
```

The generators update historical inventories, not numerical reference hashes.
Use the documented `verify --all --emit-reference PATH` procedure in VERIFY.md
for a staging table before reviewing changes to the bundled table.

`tools/verify_tsa_negative_controls.sh` requires production `core,tsa,forecast`,
sabotage `tsa,forecast`, and both `MOJOLEARN_HOST_SABOTAGE=1` and
`MOJOLEARN_KPSS_DECISION_SABOTAGE=1` defines. The core transpose must be present
in both arms. `tools/verify_cpu_promotion.sh` requires production
`core,estimators,metrics,arima` and sabotage `metrics,arima` with
`MOJOLEARN_HOST_SABOTAGE=1`. The public verifier always runs an OLS comparator
self-test: omitting estimators makes the whole command fail even when all
requested lane hashes match. Keep this check enabled.

Use `tools/runpod_cpu_leg.sh` with `--vcpu 2 --jobs 2 --disk 20 --lease 35`,
`--envs default,test`, a clean committed source tree, a unique output directory,
and the appropriate script via `--cmd`. Check its current help before rental.
Two-vCPU disk=40 was rejected by the provider. Fetched files are beneath
`OUT/remote/leg_out/`. The latest slow leg spent 557 seconds on source staging,
22 seconds building and 37 seconds executing: distinguish transport from test
cost. A future R2 source-staging improvement is optional, not a verification gate.

## Evidence and completion criteria

Read:

- [Evidence audit](lanes/LANE_STATUS_verification_evidence_audit.md)
- [Reference admission and TSA controls](lanes/LANE_STATUS_reference_admission_integrity.md)
- [Latest CPU promotions](lanes/LANE_STATUS_cpu_public_promotion.md)
- [Public verification](VERIFY.md) and [generated matrix](VERIFICATION_MATRIX.md)

Committed evidence lives under `bench/results/identity_break/2026-09-17_tsa-negative-controls/`,
`bench/results/identity_break/2026-09-17_cpu-public-promotion/` and the associated
`reference-admission-probe/`, `cpu-public-promotion-probe/` and
`verification-evidence-probe/` directories. `.json.gz` reports are lossless.
Failed dependency gates and verified teardown receipts are retained alongside
successful reruns; do not erase failed attempts or count them as passing gates.

Local working artifacts (not release artifacts) are under
`/Users/andrewhendel/mojolearn-evidence/{verification-evidence-audit,reference-admission-integrity,cpu-public-promotion}/`.
The latest `cpu-public-promotion/installed-env` and `dist/` contain the tested
Mac development wheel. No rented pod from these batches remains running.
Latest validation: 261 focused tests, matrix consistency, 36/36 native controls,
117 IDENTICAL public parts plus 63 N/A on Mac and Linux, comparator self-test,
and installed-wheel default-selection/coverage assertions. Latest cloud cost
$0.0179; prior TSA batch $0.0063. CPU limits were respected.

For each resumed batch, retain source/artifact hashes, full results and failures;
check default public selection and installed packaging; update inventories and
honest gap counts; run relevant tests; commit, merge and push main; verify remote
synchronization and cloud teardown. Finish with remaining gaps, not just passes.
