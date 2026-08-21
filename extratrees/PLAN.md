# extratrees/ — histogram-free ExtraTrees, the parallel lane

Written 2026-08-21, alongside `ensemble/PLAN.md` (the RF lane). The two are
designed to run as PARALLEL LANES and this file records why that is safe:
they share NO compute-path machinery, and the substrate they would share is
deliberately duplicated instead (rule 12: parallelize across directories,
serialize on files).

## What this is, and why it gets its own directory

The histogram-free formulation of Extremely Randomized Trees (Geurts, Ernst
& Wehenkel 2006; sklearn's `RandomSplitter` in `sklearn/tree/_splitter.pyx`
is the reference implementation): per node, draw ONE random real-valued
threshold per candidate feature inside that feature's range over the node's
rows, score the induced partition, keep the best feature. No global
quantiles, no bin index, no histogram — per node the state is a few
accumulators per candidate feature instead of bins x stats, so the
3.26M-cell footprint the GBDT port measured as ~90% of per-tree fixed cost
does not exist here.

**Nobody has this on a GPU** (re-verified 2026-08-21: the only ET-on-GPU
implementations in existence are LightGBM's histogram-space `USE_RAND` CUDA
kernel — random BIN, a different algorithm — and cuML's unlanded design in
FEA #8133). So this directory mirrors NO incumbent GPU source; it is a
port-from-paper, which the house rule allows only with the deviation block
saying so explicitly. That is also why it is a top-level sibling directory
(like `cluster/`, `dbscan/`, `glm/`) and not part of `ensemble/`, which
mirrors cuML file-for-file.

This is the library's one genuinely novel candidate and the paper hook.

## The gate, shared with the RF lane

The gather-bandwidth probe (`ensemble/PLAN.md` Step 0) gates this lane
identically: the min/max pass and the accumulate pass are still scattered
gathers. The probe runs ONCE, before either lane starts kernels. This lane
wins on FOOTPRINT, not arithmetic intensity — if the probe says gathers
lose, both lanes stop together.

## Design sketch (to be firmed against the reference before kernels)

Per node batch (breadth-first frontier, same shape as any level builder):

1. **Range pass**: per (node, sampled feature), min/max of the feature over
   the node's rows. One gather, block reduce.
2. **Draw**: threshold ~ Uniform(min, max) per (node, feature). Constant
   features (min == max) are excluded, per sklearn.
3. **Score pass**: per (node, feature), accumulate left/right stats against
   the drawn threshold — class counts for Gini, (count, sum, sum-of-squares)
   for MSE. A handful of registers per candidate, no shared-memory
   histogram.
4. **Select**: best score per node, tie-break on a total order (metric,
   feature id, threshold) — the cuML `split.cuh` discipline, cited not
   invented.
5. **Partition**: stable partition of the node's rows by the winning
   predicate; children join the frontier.

Defaults are sklearn's: `bootstrap=False` (the whole learn set, every tree —
so this lane does not even need a bootstrap sampler), `max_features='sqrt'`
classification / `1.0` regression.

## Determinism and the oracle problem, stated up front

Exact parity with sklearn is NOT achievable on a GPU: sklearn draws
thresholds from one sequential 32-bit xorshift stream (`our_rand_r`,
`sklearn/utils/_random.pxd:20-34` -- NOT MT19937, which only seeds the
per-tree state) in traversal order, and that order depends on the
Fisher-Yates feature walk and on which features were found constant, which
is the wrong shape for a parallel builder. The recorded deviation: threshold
draws are keyed counter-based on `(seed, tree_id, node_id, feature_id)` —
order-independent, bit-reproducible across Metal/CUDA/HIP by construction
(the same property the RF lane gets from integer bins).

Oracles, in the house pattern:

1. **Exact**: a host-side Mojo transcription of the splitter using the SAME
   keyed draws — per-node comparison of ranges, thresholds, scores and the
   chosen split, cell for cell on 4096-row hashed fixtures.
2. **Analytic**: fixtures where the correct split is hand-computable for any
   threshold in a known interval; adversarial constant features; sabotage
   per mechanism (rule 8).
3. **Quality band**: sklearn `ExtraTreesClassifier/Regressor` at fixed seed,
   holdout accuracy/MSE bands — never bitwise, and never a gate on a real
   dataset (rule 4).

## Lane ownership (binding while both lanes run)

- This lane owns `extratrees/` and NOTHING else. It may not touch
  `ensemble/`, `core/`, `gbdt/`, or any shared harness file.
- Substrate the RF lane also needs (flat tree node struct, predict
  traversal, row partition) is DUPLICATED here in minimal form, not shared.
  Reconciliation into `core/` is the orchestrator's merge-time decision,
  not either lane's.
- Deviation numbers **130–159** are reserved for this lane, assigned up
  front (rule 3; the ledger stood at 90 when reserved). The RF lane holds
  100–129.
- Own checks only; the full suite runs once at merge (rule 12). Benchmark
  arms: interleaved, our GPU vs sklearn CPU `n_jobs=-1` on the same
  machine, sklearn defaults, one process (rules 5 and 7).

## What this lane is NOT

- Not the sklearn-API-parity ET. That product — cuML's #8133 design, a
  random-splitter flag on the ported cuML builder — is a ~150-line serial
  follow-on inside `ensemble/` AFTER RF lands, because it touches the RF
  lane's files by definition. It cannot be parallelized and should not be.
- Not a LightGBM port. Their `USE_RAND` kernel is a cited precedent for
  randomized split gating, not a source being mirrored.


---

# Session 2 log — decisions taken, and what is open

Appended 2026-08-21 by the ExtraTrees lane. Everything above this line is the
brief; everything below is what happened to it on contact with the upstreams.

## Decision: MIRROR cuML for everything except the split rule

Andrew, mid-session: *"you should be mirroring what cuml does."* Taken
literally and applied structurally. The sketch above described a builder in the
abstract ("breadth-first node frontier, same shape as any level builder"); that
abstraction is now replaced by cuML's actual files, pinned at `00094f7`:

| ours | cuML |
|---|---|
| `ported/decisiontree/batched_levelalgo/split.mojo` | `split.cuh` |
| `ported/decisiontree/batched_levelalgo/dataset.mojo` | `dataset.h` |
| `ported/decisiontree/batched_levelalgo/objectives.mojo` | `objectives.cuh` |
| `ported/decisiontree/batched_levelalgo/builder.mojo` | `builder.cuh` (`NodeQueue` + `Builder`) |
| `ported/decisiontree/batched_levelalgo/kernels/builder_kernels.mojo` | `kernels/builder_kernels.cuh` |
| `ported/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo` | `kernels/builder_kernels_impl.cuh` |
| `ported/decisiontree/flatnode.mojo` | `cpp/include/cuml/tree/flatnode.h` |
| `mojo_only/pcg_rng.mojo` | RAFT `random/detail/rng_device.cuh` + cuML's fnv1a32 chain |

What that buys, concretely: the node work queue, the frontier batching, the
ragged-node `WorkloadInfo` scheme, the split record, the tie-break, the
validity test, the in-place swap partition, the leaf-value pass and the RNG
keying all stop being things this lane designs. **The invented surface shrinks
to exactly one kernel**: cuML's `computeSplitKernel`
(`builder_kernels_impl.cuh:216-340`) builds a histogram over quantile bins,
scans it to a CDF and calls `objective.Gain` over all bins. Ours replaces that
middle with range -> keyed draw -> single-candidate score. It lives in the
mirrored file under a `DEVIATION BLOCK`, per `PORTING_RULES.md` rule 4 ("a
replacement for a step of their file lives IN THAT FILE").

Reading their partition also killed a deviation outright — see `DEVIATIONS.md`
134, which is now a NOT-a-deviation entry.

## Decision: the gather-probe gate is superseded, and here is the reasoning

The brief says: write no kernel until `bench/results/GATHER_PROBE_*.md` exists,
and stop if it reports the GPU below ~2x CPU gather bandwidth. **That file does
not exist** (checked; the RF lane owns the probe and this lane must not run
it). Andrew then said, mid-session: *"DO NOT DO NOT check for time in
subagents or in your flow... I will optimize time later. But you should be
mirroring what cuml does."*

The probe is a pure perf triage — its only output is a go/no-go on bandwidth —
so a standing instruction to defer perf and get the basics right is the same
instruction that empties the gate of content. Recorded rather than assumed:
**this lane proceeds on correctness work, takes no timing numbers at all, and
quotes no perf claim.** If the probe lands and reports gathers losing, that is
a finding about the shape of the eventual kernels, not about the host oracle,
the objectives, the RNG or the checks — none of which would be wasted.

## RULED by Andrew, 2026-08-21

1. **DEVIATION 135 — regression accumulation: FIXED POINT.** Closed. Labels are
   quantized once on the host by a power-of-two scale derived from the whole
   dataset's magnitude sum; every device accumulation is integer, so the answer
   is order-independent and identical across vendors. Implemented in
   `mojo_only/fixed_point.mojo`. The scale had to be derived rather than copied
   from root's file, because this lane's exact split comparison cross-multiplies
   into `Int128` and a scale sized only for the accumulator overflows it — see
   the ledger entry for the arithmetic and the resolution schedule.
2. **The deviation ledger stays in `extratrees/DEVIATIONS.md`.** Not merged into
   the root `PORTING.md`. It remains written in `PORTING.md`'s format, so that
   a future merge is an append rather than an edit, but no merge is planned.

## BLOCKING, and it needs Andrew

**cuML's feature sampler never draws column 0 at `k = 1`, and we ported it
faithfully, and it makes the learner wrong.**

Measured (`_probe`, and the failure it causes is in `quality_band_check`):

    n=2 k=1  ->  col0 drawn   0 of 64 nodes, col1 64
    n=3 k=1  ->  col0 drawn   0 of 64
    n=4 k=1  ->  col0 drawn   0 of 64
    n=8 k=1  ->  col0 drawn   0 of 64

Not under-drawn. NEVER drawn. Root cause is `builder_kernels.cuh:231-232`,
which passes `mask[0]` — a FLAG, 0 or 1 — as `SubtractLeft`'s
`tile_predecessor_item`, so CUB compares the block MINIMUM against the previous
iteration's flag and marks it a duplicate when they are equal. Column 0 is the
minimum whenever it is drawn.

**The consequence is not statistical, it is fatal on small `n`.** On the
`separable_gap` fixture — 2 columns, column 0 perfectly separates the classes,
column 1 is noise — `max_features='sqrt'` gives `k = 1`, so our learner can
never see the separating column. It splits on noise forever: accuracy 0.523
against sklearn's exact 1.0, mean depth 10.8 against their 3.4, mean leaves
70.5 against their 7.9. `tree_check` did not catch it because it runs at
`max_features = 1.0`, where every column is a candidate.

**Why this is not mine to decide.** Rule 1 says theirs is right and ours is
wrong until their file says otherwise, and their file says this. But
`PORTING_RULES.md` rule 4 is equally explicit that a workaround which would
change the algorithm is not a workaround, it is a FORK, and it needs Andrew.
The two options, priced:

1. **Keep it.** The port stays faithful and the defect is documented and
   asserted (`feature_sampler_check` pins the starvation, so a later
   "improvement" turns red). Price: our ExtraTrees is materially worse than
   sklearn's whenever `max_features` resolves to a small `k` and a low-index
   column matters — which for `sqrt` is every dataset under ~10 columns, and
   for column 0 specifically, every dataset.
2. **Fix it as a numbered deviation.** Pass a sentinel that cannot collide
   with a column id (their `IdxT` is signed, so `-1` works) as the tile
   predecessor. One line, and it makes the sampler uniform. Price: our sampler
   no longer reproduces cuML's, so any future per-cell comparison against a
   cuML run would disagree by construction — though no such comparison is
   possible on this hardware anyway, since their GPU arm does not run here.

**Recommendation: option 2**, as a numbered deviation with this measurement
attached, because a faithful port of a defect that makes the learner unable to
see a feature is a port of a bug rather than of an algorithm. Not taken
unilaterally.

Until it is ruled, `quality_band_check` FAILS on `separable_gap` and
`tools/check.sh` is therefore red on exactly that one row. That is deliberate:
the alternative is a green suite that hides a known-wrong learner.

## Still open

1. **`n_estimators` / forest-level defaults.** Not touched. This lane is
   building the LEARNER; sklearn's `ExtraTreesClassifier` defaults
   (`n_estimators=100`, `bootstrap=False`, `max_features='sqrt'` for
   classification and `1.0` for regression) are recorded as the target, and the
   forest wrapper is downstream of the tree working.
2. **The gather probe.** `bench/results/GATHER_PROBE_*.md` still does not exist.
   The reasoning that supersedes it as a gate is above; it remains true that
   nobody has measured gather bandwidth on this box for this access pattern.

## Where this stands against the plan above

The design sketch at the top of this file listed five steps, a set of defaults
and three oracles. Measured against it:

| plan item | host | device |
|---|---|---|
| 1. range pass (min/max per node, feature) | DONE | in flight |
| 2. threshold draw, keyed counter-based | DONE | — |
| 3. score pass (a few accumulators, no histogram) | DONE | — |
| 4. select, total-order tie-break | DONE | — |
| 5. stable partition, children join the frontier | DONE (cuML's swap partition, ported) | — |
| breadth-first frontier | DONE | — |
| `bootstrap=False` | DONE (`True` refused by name) | — |
| `max_features` sqrt/1.0 | DONE as a ratio; the `sqrt` name is the caller's to resolve | — |
| oracle 1: exact host transcription | DONE | — |
| oracle 2: analytic + adversarial fixtures | DONE | — |
| oracle 3: sklearn quality band | in flight | — |

**The learner works end to end and so does the forest.** A separable fixture
comes out exactly right at every seed, a regression step function is fitted
exactly, an all-constant fixture yields a single leaf, and a 12-tree forest has
no two trees alike and votes the average of its trees per cell. Fifteen checks,
run by `tools/check.sh`, every one with a sabotage per mechanism that was seen
to turn it red.

**What is NOT done, stated as the gap it is:**

1. **No kernel has been enqueued.** This is the whole thesis — GPU access on a
   machine where the incumbents' GPU arms refuse to run — and none of it
   exists yet. One kernel (the range pass) is in flight. `PORTING_RULES.md` is
   explicit that a kernel is not ported until it has been enqueued, and
   compiling is not evidence.
2. **No number has been measured, deliberately.** Perf is deferred by the repo
   owner. Nothing in this directory quotes a duration and nothing should until
   that is lifted.
3. **The sklearn quality band has not been run**, so "our trees are shallower
   than sklearn's on constant-heavy data" (deviations 132 and 151) is a
   prediction, not a measurement.
4. **The deviation range 130-159 is FULL.** The device work is using 160+.
5. **No Python binding**, and deviation 154 records a debt against it:
   `min_weight_fraction_leaf` and `monotonic_cst` have no field to refuse and
   therefore no error.

## Rules this lane earned, beyond the ones it inherited

- **A sabotage that stays green is a finding about the CHECK.** It happened
  six times here and every time the fixture or the claim was the defect, not
  the code: the boundary rows that did not exist, the `min_samples_leaf` branch
  that is unreachable at its default, the exact-vs-float comparator with no
  adversarial pair, the denominators that were all equal, the feature sampler
  that never ran because `max_features` defaults to 1.0, and the `max(1, ...)`
  floor that no `max_features` reached.
- **A claim in a docstring is a claim, and gets sabotaged like code.** Three
  were false and are deleted: that `break` differs from `continue` in
  `NodeQueue.push`, that partitioning before `Push` is load-bearing, and that
  the validity guard around the partition is observable. All three are
  measured equivalences now, kept as the upstream has them.
