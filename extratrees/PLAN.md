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

Defaults are sklearn's: `bootstrap=False` (the whole learn set, every tree),
`max_features='sqrt'` classification / `1.0` regression. `bootstrap=True`
is honoured too since DEVIATION 460 (cuML's sampler through the RF lane's
Philox port), and `criterion='entropy'` since DEVIATION 459.

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

## RULED and FIXED — Andrew, 2026-08-21: "don't port bugs, fix them"

The blocking item recorded here — cuML's sampler never drawing column 0 at
`k = 1`, which made the learner unable to see a separating feature — is ruled
and fixed. Deviations 164 and 165 carry the before/after measurements.

**The standing rule this changes, stated so it does not swing too far:** the
upstream is the authority on DESIGN, not on defects. A behaviour of theirs is
ported when it is a choice, even an odd one, and fixed when it is demonstrably
a mistake that changes an answer. The bar for calling something a mistake is a
MEASUREMENT, not an opinion, and a fix ships with a check that asserts the
fixed behaviour so it cannot silently rot.

What the fix bought, measured on the same harness that found the defect:

| row | before | after | sklearn |
|---|---|---|---|
| `separable_gap` accuracy | 0.523 | **1.0** | 1.0 |
| `separable_gap` mean depth | 10.78 | **2.99** | [2.83, 4.07] |
| `separable_gap` mean leaves | 70.5 | **6.52** | [6.15, 9.73] |
| `tie_pair` mean depth | 5.60 | **2.59** | [1.91, 2.58] |
| `shaped_all` accuracy | 0.457 | **0.484** | [0.461, 0.516] |

`quality_band_check` passes, and `tools/check.sh` is green again.

**One row remains outside sklearn's band and it is not a defect:**
`shaped_constant_heavy`, where deviation 151 says a node ALL of whose sampled
columns are constant becomes a leaf here, while sklearn's loop guard keeps
drawing past `max_features` in exactly that regime (its second disjunct
extends the loop only while every feature drawn so far was constant) and so
can still find a split. That regime is the whole difference: the covtype
audit (2026-08-26, deviation 132's retraction) proved sklearn spends its
per-node draw budget on already-known constants exactly like us, so its
effective `max_features` over non-constant features is NOT larger than ours
— the superseded claim that used to stand here. We grow depth 0.74 against
their 16.3 — and score 0.543 against their band of [0.430, 0.480], i.e.
BETTER, because the fixture's labels are noise and not overfitting it is an
advantage there. On a fixture with real signal in a constant-heavy frame the
sign would flip. Priced, not fixed: closing it means re-drawing when every
sampled column is constant, which neither upstream does for a mixed draw.

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
and three oracles. Every one of them now exists on BOTH the host and the GPU.

| plan item | host | device |
|---|---|---|
| 1. range pass (min/max per node, feature) | DONE | DONE |
| 2. threshold draw, keyed counter-based | DONE | DONE |
| 3. score pass (a few accumulators, no histogram) | DONE | DONE |
| 4. select, total-order tie-break | DONE | DONE |
| 5. partition, children join the frontier | DONE | DONE |
| leaf values | DONE | DONE |
| breadth-first frontier | DONE | host, as cuML's is — and MERGED ACROSS TREES (deviation 211) |
| `bootstrap=False` | DONE | DONE |
| `bootstrap=True`, `max_samples` | DONE (DEVIATION 460) | DONE |
| `criterion='entropy'` / `'log_loss'` | DONE (DEVIATION 459) | DONE |
| `max_features` sqrt/1.0 | DONE | DONE |
| oracle 1: exact host transcription | DONE | it IS the device's oracle |
| oracle 2: analytic + adversarial fixtures | DONE | DONE |
| oracle 3: sklearn quality band | DONE | via the host, which is identical |

**THE HEADLINE: a tree grown on the GPU is the SAME TREE as one grown on the
host.** 9 configurations, 747 nodes, 0 differing in `(colid, quesval,
left_child_id, instance_count)`, 0 differing row predictions, 0 of 2241 leaf
values differing — and since deviation 183 closed, that holds with cuML's
zero-gain gate ON as well, 689 nodes against 689.

That identity is not luck. It is what three earlier decisions bought:
deviation 135's fixed point made the accumulators integers, deviation 142's
amendment put the same explicit `fma` on both sides of the draw, and deviation
144's exact rational made the comparison exact rather than approximate. Each
was argued at the time as a determinism decision; this is the thing they were
determinism decisions FOR.

**Twenty-nine checks, run by `tools/check.sh`, twelve of them enqueuing
kernels**, every one with a sabotage per mechanism that was seen to turn it
red. (This sentence said twenty-two while twenty-eight ran; both counts
recomputed from `check.sh` and `grep -l DeviceContext` on 2026-08-22,
rule 17.)

## What is NOT done, stated as the gap it is

1. **The perf deferral is LIFTED (Andrew, 2026-08-22: "please continue
   improving performance for extra trees... I think we should parallelize to
   use gpu").** The first act under it is DEVIATION 211 — the frontier batch
   spans trees, cuML's stream-pool overlap expressed as a wider grid, since
   Metal has no streams. Gated bit-identical by `device_batched_check`
   (twenty-nine checks now); measured in `bench/results/` per the alternating
   in-window discipline, not here. The lane's checks still quote no duration.
2. **The forest and the estimator DO use the device path** —
   `fit_classification_device` and `fit_extra_trees_classifier_device`, with
   the dataset uploaded once for the forest (deviation 184) and every refusal
   firing on both arms. The device forest is bit-identical to the host forest:
   0 of 1552 nodes and 0 of 4656 leaf values.
3. **Regression RUNS on the device, estimator included** --
   `train_regression_device`, the same `Builder::train` loop with the two
   kernels instantiated for `CLASSIFICATION = False` and cuML's
   `MSEObjectiveFunction` in place of their Gini one, which is what their
   `template <typename ObjectiveT> struct Builder` is for. Structure is
   identical to the host tree (0 of 691 nodes) and leaf values differ by at
   most half a quantization step -- deviation 135's ruling, measured, not a
   shortfall. `fit_extra_trees_regressor_device` exposes it through the
   sklearn surface since deviation 188 CLOSED (both regressor arms share
   `regressor_plan`, so refusals cannot drift between them).
4. **Deviation 151** — we stop when every sampled column is constant and
   sklearn keeps drawing. Priced, measured on `shaped_constant_heavy`, not
   fixed.
5. **The Python binding exists**: `bindings/_mojolearn_trees.mojo` (its own
   extension, so it is not a merge point with the estimators or gbdt
   bindings), `python/mojolearn/extratrees.py` (`ExtraTreesClassifier` /
   `ExtraTreesRegressor`, sklearn's surface honoured or refused by name,
   `device="gpu"|"cpu"`), built and GATED by `bindings/build_trees.sh` --
   the gate launches both device fits and asserts the gpu/cpu classifier
   arms bit-identical through Python. NOT yet registered in the package
   `__init__.py` or the wheel script: both are another lane's working
   files, so top-level `mojolearn.ExtraTreesClassifier` and wheel inclusion
   are handoff items, named here rather than quietly absent.

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
