# CPU verification, useful multi-GPU work, and 0.8.7 progress

Updated 2026-09-18 after release job 105616007971 completed.
This report separates implemented routes, measured evidence, and publication.

## What we are working on

1. Finish and publish the frozen 0.8.7 wheel after its exact artifact passes
   release gates. Keep new native features on main for a later qualified wheel.
2. Make the CPU verifier cover supported algorithms broadly, with bitwise
   comparisons, applicable batch properties, and meaningful fault controls.
   Public CPU training and parallel CPU products are not goals.
3. Build useful multi-GPU capabilities and prove actual physical GPU execution,
   numerical behavior, and the claimed capacity or throughput benefit.

## Completed and merged

- Six polynomial/sigmoid/Laplacian KernelRidge/Nystroem CPU reference routes and
  saved-model inference: 31 native runtime tests, 216 applicable CPU/Apple
  property matches, 54 detected training fault injections. Independent
  NVIDIA/AMD and installed-wheel admission is still owed for these routes.
- Reference-sharded neighbor numerical mismatches now retain their evidence;
  the byte-LM fault loader gets its required explicit opt-in. Targeted native
  Apple controls passed; full release recertification remains outstanding.
- Vocabulary and corpus preparation joined default CPU verification with
  recorded clean/replay evidence and two fault controls across nine fixtures.
  Their references are explicitly CPU-only. Admission passed 224 relevant tests.
- Saved-model fault checks require a changed CPU output; an optional GPU
  column disagreement alone cannot create a false pass. Regression reproduced
  before the fix; 41 binding-free orchestration tests passed on main/release.
- GPU visibility masks reject duplicated tokens and invalid indices before
  worker creation. Physical identity checks remain in the CV feature branch.
- Six unused merged worktrees were removed (about 17 GiB of files); branches,
  active work, release artifacts, and uncommitted native evidence were retained.

## Coverage snapshot

| Route classification | Count |
| --- | ---: |
| Default public CPU verification | 163 |
| Ordinary CPU routes pending reference qualification/admission | 23 |
| Parallel routes | 50 |
| Total harness routes | 236 |

Of the parallel routes, 18 have logical CPU sharding and 32 require GPUs.
No nonparallel route is outside the public/pending inventory. These are routes,
not unique algorithms or a percentage of completed proof. Batch invariance and
other properties are checked where declared; N/A is not numerical proof.

Pending groups: five neural, twelve classical, and six new kernel variants.
See [the completion plan](CPU_IDENTITY_AND_MULTI_GPU_PLAN.md) for names and
admission requirements. Whole loaded-CausalLM bitwise proof remains open.

## Release status and newly diagnosed blocker

0.8.7 is not published. Release branch: release/087-final; native freeze:
d9185528e1ba78e61e091b37489b52daebde342d. The old-source run is
[35350125464](https://github.com/mojolearn/mojolearn/actions/runs/35350125464).

- Wheel build/verification succeeded.
- ARM64 CPU certification failed on stale UMAP expectations and fault checks.
- Hosted Apple certification was cancelled at the 90-minute job limit while
  executing the fault sweep. GitHub's annotation explicitly confirms timeout;
  this is not evidence of a numerical crash. Its UMAP gate also failed.
- x86 CPU certification has started and is building host bindings.
- Proof-tool fixes are selectively backported. Their changed source snapshot
  needs fresh matching GPU qualification; old artifact receipts do not qualify
  the repaired tools automatically.

The job timeout is now 180 minutes because two full-column steps alone each
have 50-minute budgets, with builds and additional gates around them. The matrix
remains serial and each job retains its two-worker limit. This changes no
numerical verdict or lane scope and does not restart the active old run.
Retained diagnosis: bench/results/cpu_certification/2026-09-18-apple-timeout.

Prepared in release commit a380012dd: each installed Linux GPU qualification
leg now captures repeated UMAP hashes, all nine GPU saved models, and CPU replay
against that fresh column. The archived source commit is explicitly recorded.
Two shell orchestration tests cover successful command ordering and five
failure-stage subcases; a failed supplement cannot leave the surface's success
marker in place. No native run or reference supersession is claimed yet.

## Multi-GPU status

Cross-validation is implemented and committed on lane/multigpu-cv, off main.
Its software tests passed 91 with 10 optional sklearn skips. The runner retains
per-fold models/predictions, one/two/reversed device comparisons, and UUID/PCI
placement. No real two-GPU run has occurred; actual execution traces remain owed.

The other planned scopes remain open: generic loaded-LM inference, distributed
IVF index storage/search, GPC, ARIMA/Holt-Winters prediction, and general GEMM.
Existing parallel routes do not establish completion of these new scopes.

## Next steps, in execution order

Latest user steering: independent architecture jobs should run in parallel.
The matrix now allows three jobs, each with its existing two-shard cap (up to
six numerical workers). No new Apple run was dispatched. Parallel scheduling
can reduce matrix wall time from the sum of job times to the longest job, at
best about 3x for three equally long jobs; it does not reduce individual fit
time or total numerical work. Earlier serial-worker notes above are historical.

Measured hosted Apple costs: 7.4 minutes production build, 46.1 minutes clean
full-column sweep, 5.1 minutes fault-build compilation, then an unfinished fault
sweep before the job timeout. Default fixture dimensions are 20,000 rows by
16 columns (many lanes slice smaller inputs), with nine variants and repeats.
This is maintainer coverage, not an appropriate default lightweight install
test. Next design work: a separate small training-identity profile with pinned
per-lane edge cases and explicitly generated CPU/GPU references; preserve the
broad profile for release/regression work. Do not reduce existing fixture sizes
in place or compare small inputs against large-input reference hashes. Measure
the slowest lanes before choosing sizes, preserving the boundary/bug cases.

Update after user review: hold another Apple certification run while checkpointing
and CPU implementation work proceed. The workflow now uploads intermediate
saved-model, clean-column and fault-column reports before the final upload.
These preserve finished stages if a later stage fails; they are not automatic
cross-run restore, nor do they preserve every cell if a running stage is killed.
run-column --resume now preserves local shard checkpoints and delegates reuse
validation to the harness (source/binary bytes, environment, machine provenance
and protocol must match). It deletes and recomputes the merged verdict, retains
logs, and starts missing shards fresh. Existing failed cells stay failed.
Changing checkpoint locations or environment can intentionally refuse reuse.
Automatic artifact restore and safe stage-selection remain follow-up work.

The installed CLI now exposes --inference (the bundled saved-model suite,
alias of --models-only) and --training (fit-based routes and their properties,
excluding the separate bundled-model suite). Existing flags retain behavior.
These clearer names are on main for the next wheel; frozen-release proof fixes
remain separate. Consolidate remaining CPU implementation work with targeted
tests, then perform broad final qualification rather than repeating the whole
architecture matrix for each small edit.

1. Retain and inspect the remaining x86 old-source results. Resolve failures
   individually; do not reuse a timeout or refusal as proof of agreement.
2. Finish the UMAP evidence repair: fresh NVIDIA/AMD captures after the
   row-separable transform change, CPU replay, and explicit supersession of
   all nine historical saved-model expectations. Preserve historical records.
3. Requalify the final repaired release tools against the exact wheel; rerun
   CPU architecture gates with the corrected budget. Publish only once required
   gates pass, then verify PyPI downloads and hashes.
4. Run CV on real NVIDIA and AMD pairs with execution traces and negative
   controls; merge the feature only when those physical gates pass.
5. Qualify the 23 pending CPU routes and implement the remaining useful GPU
   scopes according to the completion plan, checkpointing tested increments.

No new rentals during the active release CPU matrix. Local native/compiler work
uses the shared slot and one worker. Memory pressure returned to normal in the
latest check, but swap remains substantial and another owner's native job holds
the local slot. Keep active release/CV worktrees; do not delete their evidence.

Handoffs: [CPU proof](lanes/LANE_STATUS_cpu-proof-followup.md),
[host-only admission](lanes/LANE_STATUS_host-only-verifier.md), and the
release branch's docs/lanes/HANDOFF_release087_proof_repairs.md. CV details live
on its feature branch in docs/lanes/LANE_STATUS_multigpu-cv.md.
